"""
RANS 2-equation solver for turbulent flows — DOLFINx 0.10.0+

Model-agnostic solver: each turbulence model (Wilcox 2006, SST, k-ε LB)
is a self-contained class that provides UFL coefficients to a fixed
equation template. Zero model-specific branches in this file.

Features:
- Pseudo-transient continuation to steady state
- Adaptive time stepping with CFL ceiling and residual-gated growth
- IPCS fractional-step momentum solver (AB2/CN for channel, Picard for BFS)
- Generic 2-equation turbulence transport template

References:
    Nek5000 RANS tutorial: https://nek5000.github.io/NekDoc/tutorials/rans.html
    Moser, Kim, Mansour (1999), Phys. Fluids 11(4):943-945
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)
from dolfinx.io import VTXWriter

from ufl import (
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
    lhs,
    nabla_grad,
    rhs,
)

try:
    from dolfinx_mpc import MultiPointConstraint
    from dolfinx_mpc import apply_lifting as mpc_apply_lifting
    from dolfinx_mpc import assemble_matrix as mpc_assemble_matrix
    from dolfinx_mpc import assemble_vector as mpc_assemble_vector

    HAVE_MPC = True
except ImportError:
    HAVE_MPC = False

from dolfinx_rans.config import (
    BFSGeom,
    BoundaryInfo,
    ChannelGeom,
    NondimParams,
    SolveParams,
    TurbParams,
)
from dolfinx_rans.geometry import (
    infer_first_offwall_spacing,
    initial_k_bfs,
    initial_k_channel,
    initial_omega_bfs,
    initial_omega_channel,
    initial_velocity_bfs,
    initial_velocity_channel,
    mark_bfs_boundaries,
    mark_channel_boundaries,
)
from dolfinx_rans.models import create_model
from dolfinx_rans.utils import (
    HistoryWriterCSV,
    StepTablePrinter,
    compute_bulk_velocity,
    diagnostics_scalar,
    diagnostics_vector,
    eval_wall_shear_stress,
    fmt_pair_sci,
    prepare_wall_shear_stress,
)


# =============================================================================
# Live plotting during iteration
# =============================================================================


def _plot_convergence_live(history_file: Path, save_path: Path):
    """Plot convergence history from CSV (called during iteration).

    Single figure: residuals on left y-axis (log), dt/CFL on right y-axis (linear).
    """
    import csv
    import matplotlib.pyplot as plt

    if not history_file.exists():
        return

    data: dict[str, list[float]] = {"iter": [], "res_u": [], "res_k": [], "res_w": [], "dt": [], "cfl_max": []}
    with open(history_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["iter"].append(int(row["iter"]))
            data["res_u"].append(float(row.get("res_u", row.get("residual", 1.0))))
            data["res_k"].append(float(row.get("res_k", 1.0)))
            data["res_w"].append(float(row.get("res_w", 1.0)))
            data["dt"].append(float(row["dt"]))
            cfl = row.get("cfl_max", "")
            data["cfl_max"].append(float(cfl) if cfl else 0.0)

    iters = data["iter"]
    if len(iters) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Left y-axis: residuals (log scale)
    ax.semilogy(iters, data["res_u"], "b-", linewidth=1.2, label="Momentum (u)")
    ax.semilogy(iters, data["res_k"], "r-", linewidth=1.2, label="TKE (k)")
    ax.semilogy(iters, data["res_w"], "g-", linewidth=1.2, label=r"Omega ($\omega$)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log)")
    ax.set_title("Convergence History")
    ax.grid(True, alpha=0.3, which="both")

    # Right y-axis: dt and CFL_max
    ax2 = ax.twinx()
    ax2.plot(iters, data["dt"], "k--", linewidth=0.8, alpha=0.6, label="dt")
    if any(v > 0 for v in data["cfl_max"]):
        ax2.plot(iters, data["cfl_max"], "k:", linewidth=0.8, alpha=0.6, label=r"CFL$_{\max}$")
    ax2.set_ylabel(r"dt / CFL$_{\max}$")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_fields_live(u, p, k, omega, nu_t, domain, geom, Re_tau, step, save_path: Path):
    """
    Plot field profiles + 2D contours during iteration. MPI-safe.

    Phase 1 (all ranks): extract line profiles and gather 2D fields.
    Phase 2 (rank 0): matplotlib figure with 3x3 layout.

    Layout:
        Row 0: 2D contours -- u+, k+, nu_t/nu
        Row 1: 1D profiles -- u+ (semilog + law of wall), k+, omega+ (semilogy)
        Row 2: 1D profiles -- v+, p+, nu_t/nu
    """
    import matplotlib.pyplot as plt
    from dolfinx_rans.plotting import extract_fields_on_line, gather_scalar_field

    comm = domain.comm
    nu = 1.0 / Re_tau
    x_mid = geom.Lx / 2
    y_vals = np.linspace(0.001, geom.Ly - 0.001, 80)

    # Phase 1: all ranks participate in extraction
    u_profile, v_profile, p_profile, k_profile, w_profile, nut_profile = \
        extract_fields_on_line(
            [u.sub(0), u.sub(1), p, k, omega, nu_t],
            y_vals, x_mid, domain, comm=comm,
        )

    # Gather 2D fields (u.sub(0).collapse() is cheap -- DOF mapping, not assembly)
    ux_x, ux_y, ux_vals = gather_scalar_field(u.sub(0).collapse(), comm)
    k_x, k_y, k_vals = gather_scalar_field(k, comm)
    nut_x, nut_y, nut_vals = gather_scalar_field(nu_t, comm)

    # Phase 2: rank 0 only
    if comm.rank != 0:
        return

    y_plus = y_vals * Re_tau
    ar = geom.Lx / geom.Ly  # channel aspect ratio (~6.28)
    contour_h = 16 / (3 * ar)  # width per panel / aspect ratio
    fig, axes = plt.subplots(3, 3, figsize=(16, contour_h + 8),
                             gridspec_kw={"height_ratios": [contour_h, 4, 4]})

    try:
        # --- Row 0: 2D contour plots ---
        from dolfinx_rans.plotting import _tricontour
        _tricontour(axes[0, 0], ux_x, ux_y, ux_vals, f"u+ (max={np.max(ux_vals):.1f})", geom)
        _tricontour(axes[0, 1], k_x, k_y, k_vals, f"k+ (max={np.max(k_vals):.3f})", geom)
        _tricontour(axes[0, 2], nut_x, nut_y, nut_vals / nu, f"nu_t/nu (max={np.max(nut_vals)/nu:.0f})", geom)

        # --- Row 1: u+ (semilog), k+, omega+ ---
        ax = axes[1, 0]
        ax.semilogx(y_plus, u_profile, "b-", linewidth=1.5)
        y_visc = np.linspace(1, 11, 30)
        y_log = np.linspace(11, 300, 30)
        ax.semilogx(y_visc, y_visc, "k--", linewidth=0.8, alpha=0.4, label="u+=y+")
        ax.semilogx(y_log, 2.5 * np.log(y_log) + 5.5, "k:", linewidth=0.8, alpha=0.4, label="log law")
        ax.set_xlabel("y+")
        ax.set_ylabel("u+")
        ax.set_title(f"u+ (max={np.max(u_profile):.1f})")
        ax.set_xlim(1, Re_tau)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(y_plus, k_profile, "r-", linewidth=1.5)
        ax.set_xlabel("y+")
        ax.set_ylabel("k+")
        ax.set_title(f"k+ (max={np.max(k_profile):.3f})")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        w_safe = np.maximum(w_profile, 1e-10)
        ax.semilogy(y_plus, w_safe, "g-", linewidth=1.5)
        ax.set_xlabel("y+")
        ax.set_ylabel("omega+")
        ax.set_title(f"omega+ (max={np.max(w_profile):.1e})")
        ax.grid(True, alpha=0.3)

        # --- Row 2: v+, p+, nu_t/nu ---
        ax = axes[2, 0]
        ax.plot(y_plus, v_profile, "c-", linewidth=1.5)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("y+")
        ax.set_ylabel("v+")
        ax.set_title(f"v+ (max |v|={np.max(np.abs(v_profile)):.2e})")
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        ax.plot(y_plus, p_profile, "k-", linewidth=1.5)
        ax.set_xlabel("y+")
        ax.set_ylabel("p+")
        ax.set_title(f"p+ (range={np.max(p_profile)-np.min(p_profile):.2e})")
        ax.grid(True, alpha=0.3)

        ax = axes[2, 2]
        ax.plot(y_plus, nut_profile / nu, "m-", linewidth=1.5)
        ax.set_xlabel("y+")
        ax.set_ylabel("nu_t/nu")
        ax.set_title(f"nu_t/nu (max={np.max(nut_profile)/nu:.0f})")
        ax.grid(True, alpha=0.3)

    except Exception as e:
        for ax in axes.flat:
            ax.text(0.5, 0.5, f"Extract failed:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

    fig.suptitle(f"Iteration {step}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    # Save numbered snapshot
    path = Path("fields0000000.png") if save_path is None else save_path
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    if f"{step}" not in path.name:
        stem = path.stem
        parent = path.parent
        path = parent / f"{stem}{step:07d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _infer_y_first_general(domain, wall_facets, fdim):
    """Infer first off-wall spacing from wall facet geometry (any domain shape)."""
    from dolfinx.mesh import compute_midpoints

    comm = domain.comm
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    # Get midpoints of wall facets
    wall_mids = compute_midpoints(domain, fdim, wall_facets)

    # Get all mesh node coordinates
    x = domain.geometry.x

    # For each node, compute minimum distance to any wall facet midpoint.
    # This is approximate but sufficient for the omega_wall BC.
    if wall_mids.shape[0] == 0 or x.shape[0] == 0:
        local_min = np.inf
    else:
        # Vectorized: distance from each mesh node to nearest wall midpoint
        dists = np.min(np.linalg.norm(
            x[:, None, :2] - wall_mids[None, :, :2], axis=2
        ), axis=1)
        positive = dists[dists > 1e-12]
        local_min = float(np.min(positive)) if positive.size > 0 else np.inf

    y_first = float(comm.allreduce(local_min, op=MPI.MIN))
    if not np.isfinite(y_first):
        y_first = 1e-3  # fallback
        if comm.rank == 0:
            print(f"WARNING: Could not infer y_first from mesh, using fallback {y_first}")
    elif comm.rank == 0:
        print(f"Inferred y_first = {y_first:.6e} from wall facets")
    return y_first


# =============================================================================
# Main solver
# =============================================================================


def solve_rans_kw(
    domain,
    geom,
    turb: TurbParams,
    solve: SolveParams,
    results_dir: Path,
    nondim: NondimParams = None,
    boundaries: BoundaryInfo = None,
):
    """
    Solve RANS 2-equation flow with pseudo-transient continuation.

    Model-agnostic: the turbulence model is selected from turb.model and
    provides coefficients to a generic transport template.

    Supports both channel flow (ChannelGeom) and backward-facing step (BFSGeom).

    Args:
        domain: DOLFINx mesh
        geom: Geometry (ChannelGeom or BFSGeom)
        turb: Turbulence model parameters
        solve: Solver parameters
        results_dir: Output directory
        nondim: Nondimensional scaling (required for channel, optional for BFS)
        boundaries: Pre-computed boundary info (required for BFS)

    Returns:
        Tuple of (u, p, k, omega, nu_t, V, Q, S, domain, step, t)
    """
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    comm = domain.comm

    is_bfs = isinstance(geom, BFSGeom)

    if not is_bfs and nondim is None:
        raise ValueError("nondim is required for channel geometry")

    if is_bfs:
        h = geom.step_height
        ER = geom.expansion_ratio
        H_inlet = h / (ER - 1.0)
        H_outlet = H_inlet + h
        Ly = H_outlet
        H = H_inlet  # reference height for turbulence ICs
        Re_tau = nondim.Re_tau if nondim is not None else 100.0
        nu = 1.0 / Re_tau
        rho = 1.0
        use_body_force = False
        u_bulk_init = 1.0
    else:
        Ly = geom.Ly
        H = Ly if geom.use_symmetry else Ly / 2.0
        Re_tau = nondim.Re_tau
        nu = 1.0 / Re_tau
        rho = 1.0
        use_body_force = nondim.use_body_force
        u_bulk_init = 15.0  # Conservative initial

    if comm.rank == 0:
        if is_bfs:
            print(f"BFS MODE: Re_tau = {Re_tau}", flush=True)
        else:
            print(f"NONDIMENSIONAL MODE: Re_tau = {Re_tau}", flush=True)
        print(f"  nu* = 1/Re_tau = {nu:.6f}", flush=True)
        print(f"  Body force: f_x = {1.0 if use_body_force else 0.0}", flush=True)

    # ── Instantiate turbulence model ──────────────────────────────
    model = create_model(turb.model)

    dt = solve.dt
    u_noslip = np.array([0.0, 0.0], dtype=PETSc.ScalarType)

    # Function spaces
    V = functionspace(domain, ("Lagrange", 2, (gdim,)))
    Q = functionspace(domain, ("Lagrange", 1))
    S = functionspace(domain, ("Lagrange", 1))

    # Boundary facets — geometry-dispatched
    if is_bfs:
        if boundaries is None:
            boundaries = mark_bfs_boundaries(domain, geom)
        wall_facets_tb = boundaries.wall_facets
        outlet_facets = boundaries.outlet_facets
        inlet_facets = boundaries.inlet_facets

        # Use user-specified y_first; fall back to mesh inference
        if geom.y_first > 0:
            y_first = geom.y_first
        else:
            y_first = _infer_y_first_general(domain, wall_facets_tb, fdim)
    else:
        bottom_facets, top_facets, left_facets, right_facets = mark_channel_boundaries(domain, geom.Lx, Ly)
        outlet_facets = right_facets

        y_first_cfg = geom.y_first
        y_first = infer_first_offwall_spacing(domain, Ly, geom.use_symmetry)
        if y_first_cfg > 0:
            rel_err = abs(y_first - y_first_cfg) / y_first_cfg
            if rel_err > geom.y_first_tol_rel:
                raise ValueError(
                    "Mesh/BC inconsistency at solve stage: "
                    f"requested y_first={y_first_cfg:.6e}, measured from mesh={y_first:.6e}, "
                    f"relative error={100.0 * rel_err:.1f}% exceeds tolerance "
                    f"{100.0 * geom.y_first_tol_rel:.1f}%."
                )

        if geom.use_symmetry:
            wall_facets_tb = bottom_facets
        else:
            wall_facets_tb = np.concatenate([bottom_facets, top_facets])

    if comm.rank == 0:
        print(f"Velocity DOFs: {V.dofmap.index_map.size_global}", flush=True)
        print(f"Pressure DOFs: {Q.dofmap.index_map.size_global}", flush=True)
        print(f"Scalar DOFs (k, {model.scalar_name}): {S.dofmap.index_map.size_global}", flush=True)
        print(f"Turbulence model: {model.display_name}")
        print("Setting up BCs...", flush=True)

    # Trial/test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    k_trial = TrialFunction(S)
    phi_k = TestFunction(S)
    w_trial = TrialFunction(S)
    phi_w = TestFunction(S)

    # Solution functions
    u_ = Function(V, name="velocity")
    u_s = Function(V)
    u_n = Function(V)
    u_n1 = Function(V)
    p_ = Function(Q, name="pressure")
    phi = Function(Q)

    k_ = Function(S, name="k")
    k_n = Function(S)
    k_prev = Function(S)

    omega_ = Function(S, name="omega")
    omega_n = Function(S)
    omega_prev = Function(S)

    nu_t_ = Function(S, name="nu_t")
    nu_t_old = Function(S)

    S_mag_ = Function(S, name="S_magnitude")

    # Constants
    dt_c = Constant(domain, PETSc.ScalarType(dt))
    nu_c = Constant(domain, PETSc.ScalarType(nu))
    rho_c = Constant(domain, PETSc.ScalarType(rho))

    if use_body_force:
        f = Constant(domain, PETSc.ScalarType((1.0, 0.0)))
        if comm.rank == 0:
            print("Using body force f_x = 1.0 (pressure-gradient equivalent)")
    else:
        f = Constant(domain, PETSc.ScalarType((0.0, 0.0)))

    # ── Model setup ───────────────────────────────────────────────
    model.setup(
        domain, S, k_prev, omega_prev, k_n, omega_n,
        nu_t_, u_n, nu_c, nu, turb, wall_facets_tb, is_bfs, geom,
        y_first=y_first,
    )

    # Wall distance (model decides if needed and which method)
    if model.needs_wall_distance():
        y_wall = model.compute_wall_distance(S, wall_facets_tb, is_bfs, geom)
    else:
        y_wall = None

    # Field specs from model
    k_spec = model.get_k_field_spec()
    scalar_spec = model.get_scalar_field_spec()

    # ── Initial conditions ────────────────────────────────────────
    if is_bfs:
        u_n.interpolate(lambda x: initial_velocity_bfs(x, u_bulk_init, Ly, H))
        k_n.interpolate(lambda x: initial_k_bfs(x, u_bulk_init))
    else:
        u_n.interpolate(lambda x: initial_velocity_channel(x, u_bulk_init, Ly, geom.use_symmetry))
        k_n.interpolate(lambda x: initial_k_channel(x, u_bulk_init))

    # All models start from omega IC, then convert to model scalar
    omega_init_func = Function(S)
    if is_bfs:
        omega_init_func.interpolate(lambda x: initial_omega_bfs(x, u_bulk_init, H, nu=nu, H_outlet=Ly))
    else:
        omega_init_func.interpolate(lambda x: initial_omega_channel(x, u_bulk_init, H, nu))

    omega_n.x.array[:] = model.convert_omega_to_scalar_ic(
        omega_init_func.x.array, k_n.x.array
    )

    u_n1.x.array[:] = u_n.x.array
    u_.x.array[:] = u_n.x.array
    k_.x.array[:] = k_n.x.array
    omega_.x.array[:] = omega_n.x.array
    k_prev.x.array[:] = k_n.x.array
    omega_prev.x.array[:] = omega_n.x.array

    nu_t_.x.array[:] = model.initial_nu_t(k_n.x.array, omega_n.x.array)
    nu_t_.x.array[:] = np.clip(nu_t_.x.array, 0, turb.nu_t_max_factor * nu)
    nu_t_old.x.array[:] = nu_t_.x.array

    wall_dofs_V_tb = locate_dofs_topological(V, fdim, wall_facets_tb)
    wall_dofs_S_tb = locate_dofs_topological(S, fdim, wall_facets_tb)

    bc_walls_u = dirichletbc(u_noslip, wall_dofs_V_tb, V)

    # Build velocity BCs list — geometry-dispatched
    bcu = [bc_walls_u]

    if is_bfs:
        # BFS: add inlet velocity Dirichlet BC (parabolic profile)
        u_inlet_func = Function(V)
        u_inlet_func.interpolate(lambda x: initial_velocity_bfs(x, u_bulk_init, Ly, H))
        inlet_dofs_V = locate_dofs_topological(V, fdim, inlet_facets)
        bc_inlet_u = dirichletbc(u_inlet_func, inlet_dofs_V)
        bcu.append(bc_inlet_u)

        # Inlet k and turbulence scalar Dirichlet BCs
        inlet_dofs_S = locate_dofs_topological(S, fdim, inlet_facets)
        k_inlet_val = max(1.5 * (0.05 * u_bulk_init) ** 2, 1e-8)
        omega_inlet_val = np.sqrt(k_inlet_val) / max(0.07 * H, 1e-10)
        bc_k_inlet = dirichletbc(PETSc.ScalarType(k_inlet_val), inlet_dofs_S, S)
        scalar_inlet_val = model.compute_inlet_scalar(k_inlet_val, omega_inlet_val)
        bc_w_inlet = dirichletbc(PETSc.ScalarType(scalar_inlet_val), inlet_dofs_S, S)
    elif hasattr(geom, "use_symmetry") and geom.use_symmetry:
        # Channel symmetry BC at top: v=0 (y-component only)
        V_y_sub = V.sub(1)
        V_y_collapsed, _ = V_y_sub.collapse()
        zero_vy_solve = Function(V_y_collapsed)
        zero_vy_solve.x.array[:] = 0.0
        top_dofs_Vy_solve = locate_dofs_topological((V_y_sub, V_y_collapsed), fdim, top_facets)
        bc_sym_v = dirichletbc(zero_vy_solve, top_dofs_Vy_solve, V_y_sub)
        bcu.append(bc_sym_v)

    if use_body_force:
        # Periodic channel: no pressure Dirichlet BC (gauge set by nullspace)
        bcp = []
    else:
        outlet_dofs_Q = locate_dofs_topological(Q, fdim, outlet_facets)
        bc_pressure = dirichletbc(PETSc.ScalarType(0.0), outlet_dofs_Q, Q)
        bcp = [bc_pressure]

    bc_k_wall = dirichletbc(PETSc.ScalarType(k_spec.wall_value), wall_dofs_S_tb, S)
    bck = [bc_k_wall]
    if is_bfs:
        bck.append(bc_k_inlet)

    bcw = []
    if scalar_spec.has_wall_dirichlet:
        bc_w_wall = dirichletbc(PETSc.ScalarType(scalar_spec.wall_value), wall_dofs_S_tb, S)
        bcw = [bc_w_wall]
    if is_bfs:
        bcw.append(bc_w_inlet)

    if comm.rank == 0:
        if not scalar_spec.has_wall_dirichlet:
            print(
                f"Wall BC for {model.scalar_name}: no wall Dirichlet (model-specific damping)",
                flush=True,
            )
        else:
            y_first_cfg = geom.y_first if hasattr(geom, "y_first") else y_first
            print(
                f"Wall {model.scalar_name} BC: value = {scalar_spec.wall_value:.2e} "
                f"(y_first mesh = {y_first:.6f}, requested = {y_first_cfg:.6f})",
                flush=True,
            )

    # ── Periodic BCs via dolfinx_mpc (channel only) ──────────────
    if use_body_force:
        if not HAVE_MPC:
            raise RuntimeError(
                "dolfinx_mpc is required for periodic channel flow. "
                "Install with: conda install -c conda-forge dolfinx_mpc"
            )
        Lx = geom.Lx
        _pbc_tol = 1e-6

        def periodic_indicator(x):
            """Identify slave DOFs on the right boundary."""
            return np.isclose(x[0], Lx, atol=_pbc_tol)

        def periodic_relation(x):
            """Map right boundary coords to left boundary coords."""
            out = x.copy()
            out[0] -= Lx
            return out

        mpc_V = MultiPointConstraint(V)
        mpc_V.create_periodic_constraint_geometrical(
            V, periodic_indicator, periodic_relation, bcu, tol=_pbc_tol,
        )
        mpc_V.finalize()

        mpc_Q = MultiPointConstraint(Q)
        mpc_Q.create_periodic_constraint_geometrical(
            Q, periodic_indicator, periodic_relation, bcp, tol=_pbc_tol,
        )
        mpc_Q.finalize()

        mpc_S = MultiPointConstraint(S)
        mpc_S.create_periodic_constraint_geometrical(
            S, periodic_indicator, periodic_relation, bck + bcw, tol=_pbc_tol,
        )
        mpc_S.finalize()

        if comm.rank == 0:
            print(f"Periodic MPC: V={mpc_V.num_local_slaves} slaves, "
                  f"Q={mpc_Q.num_local_slaves}, S={mpc_S.num_local_slaves}")

        # Backsubstitute ICs so slave DOFs satisfy the periodic constraint
        for fn in [u_n, u_n1, u_]:
            mpc_V.backsubstitution(fn)
        for fn in [k_n, k_, k_prev, omega_n, omega_, omega_prev]:
            mpc_S.backsubstitution(fn)
    else:
        mpc_V = None
        mpc_Q = None
        mpc_S = None

    if comm.rank == 0:
        print("Setting up weak forms...", flush=True)

    comm.barrier()  # Sync before form setup

    # ── Generic turbulence form template ──────────────────────────
    c = model.get_form_coefficients()

    # k-equation: dk/dt + u·∇k = ∇·((ν + σ_k·ν_t)∇k) + P_k - R_k·k
    F_k = (
        (k_trial - k_n) / dt_c * phi_k * dx
        + dot(u_n, grad(k_trial)) * phi_k * dx
        + (nu_c + c.sigma_k * c.nu_t_diff_k) * inner(grad(k_trial), grad(phi_k)) * dx
        + c.reaction_k * k_trial * phi_k * dx
        - c.production_k * phi_k * dx
    )

    a_k = form(lhs(F_k))
    L_k = form(rhs(F_k))

    if comm.rank == 0:
        print("  k-equation forms ready", flush=True)

    # Scalar equation: dφ/dt + u·∇φ = ∇·((ν + σ_φ·ν_t)∇φ) + P_φ - R_φ·φ + CD
    F_w = (
        (w_trial - omega_n) / dt_c * phi_w * dx
        + dot(u_n, grad(w_trial)) * phi_w * dx
        + (nu_c + c.sigma_phi * c.nu_t_diff_phi) * inner(grad(w_trial), grad(phi_w)) * dx
        + c.reaction_phi * w_trial * phi_w * dx
        - c.production_phi * phi_w * dx
        - c.cross_diffusion * phi_w * dx
    )

    a_w = form(lhs(F_w))
    L_w = form(rhs(F_w))

    if comm.rank == 0:
        print(f"  {model.scalar_name}-equation forms ready", flush=True)

    # IPCS momentum
    nu_eff = nu_c + nu_t_
    mu_eff = rho_c * nu_eff

    if is_bfs:
        F1_mom = rho_c / dt_c * dot(u - u_n, v) * dx
        F1_mom += inner(dot(u_n, nabla_grad(u)), v) * dx
        F1_mom += mu_eff * inner(grad(u), grad(v)) * dx
        F1_mom -= dot(p_, div(v)) * dx
        F1_mom -= dot(f, v) * dx
    else:
        F1_mom = rho_c / dt_c * dot(u - u_n, v) * dx
        F1_mom += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
        F1_mom += 0.5 * mu_eff * inner(grad(u + u_n), grad(v)) * dx
        F1_mom -= dot(p_, div(v)) * dx
        F1_mom -= dot(f, v) * dx

    a1 = form(lhs(F1_mom))
    L1 = form(rhs(F1_mom))

    if comm.rank == 0:
        print("  Momentum forms ready", flush=True)

    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho_c / dt_c * div(u_s) * q * dx)

    a3 = form(rho_c * dot(u, v) * dx)
    L3 = form(rho_c * dot(u_s, v) * dx - dt_c * dot(nabla_grad(phi), v) * dx)

    # ── Create matrices and vectors ──────────────────────────────
    # MPC case: dolfinx_mpc creates matrices with extended sparsity
    # pattern for the master-slave coupling; regular case uses dolfinx.
    if mpc_V is not None:
        A_k = mpc_assemble_matrix(a_k, mpc_S, bcs=bck)
        b_k = mpc_assemble_vector(L_k, mpc_S)
        A_w = mpc_assemble_matrix(a_w, mpc_S, bcs=bcw)
        b_w = mpc_assemble_vector(L_w, mpc_S)
        A1 = mpc_assemble_matrix(a1, mpc_V, bcs=bcu)
        b1 = mpc_assemble_vector(L1, mpc_V)
        A2 = mpc_assemble_matrix(a2, mpc_Q, bcs=bcp)
        b2 = mpc_assemble_vector(L2, mpc_Q)
        A3 = mpc_assemble_matrix(a3, mpc_V, bcs=[])
        b3 = mpc_assemble_vector(L3, mpc_V)
    else:
        A_k = create_matrix(a_k)
        b_k = create_vector(S)
        A_w = create_matrix(a_w)
        b_w = create_vector(S)
        A1 = create_matrix(a1)
        b1 = create_vector(V)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(Q)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(V)

    # Pressure nullspace for periodic channel (no Dirichlet pin)
    if use_body_force:
        pressure_ns = PETSc.NullSpace().create(constant=True, comm=comm)
        A2.setNullSpace(pressure_ns)
    else:
        pressure_ns = None

    if comm.rank == 0:
        print("  Pressure forms ready", flush=True)

    # Solvers (MPI-compatible preconditioners)
    solver_k = PETSc.KSP().create(comm)
    solver_k.setOperators(A_k)
    solver_k.setType(PETSc.KSP.Type.BCGS)
    pc_k = solver_k.getPC()
    pc_k.setType(PETSc.PC.Type.HYPRE)
    pc_k.setHYPREType("boomeramg")
    solver_k.setTolerances(rtol=1e-8)

    solver_w = PETSc.KSP().create(comm)
    solver_w.setOperators(A_w)
    solver_w.setType(PETSc.KSP.Type.BCGS)
    pc_w = solver_w.getPC()
    pc_w.setType(PETSc.PC.Type.HYPRE)
    pc_w.setHYPREType("boomeramg")
    solver_w.setTolerances(rtol=1e-8)

    solver1 = PETSc.KSP().create(comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    solver1.getPC().setType(PETSc.PC.Type.JACOBI)
    solver1.setTolerances(rtol=1e-8)

    solver2 = PETSc.KSP().create(comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.CG)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")
    solver2.setTolerances(rtol=1e-8)

    solver3 = PETSc.KSP().create(comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    solver3.getPC().setType(PETSc.PC.Type.JACOBI)
    solver3.setTolerances(rtol=1e-8)

    if comm.rank == 0:
        print("  Linear solvers ready", flush=True)

    def _reassemble_matrix(A, a, bcs, mpc=None):
        if mpc is not None:
            mpc_assemble_matrix(a, mpc, bcs=bcs, A=A)
        else:
            A.zeroEntries()
            assemble_matrix(A, a, bcs=bcs)
            A.assemble()

    def _reassemble_vector(b, L, a, bcs, mpc=None):
        if mpc is not None:
            mpc_assemble_vector(L, mpc, b=b)
            mpc_apply_lifting(b, [a], [bcs], mpc)
        else:
            with b.localForm() as loc:
                loc.set(0.0)
            assemble_vector(b, L)
            apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

    def _reassemble_vector_no_bc(b, L, mpc=None):
        if mpc is not None:
            mpc_assemble_vector(L, mpc, b=b)
        else:
            with b.localForm() as loc:
                loc.set(0.0)
            assemble_vector(b, L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Precompute wall shear stress forms (JIT compilation happens once here)
    wss_ctx = prepare_wall_shear_stress(u_, domain, nu)

    # VTX output
    vtx_vel = None
    vtx_turb = None
    if solve.snapshot_interval > 0:
        vtx_vel = VTXWriter(comm, results_dir / "snps" / "velocity.bp", [u_], engine="BP4")
        vtx_vel.write(0.0)
        vtx_turb = VTXWriter(comm, results_dir / "snps" / "turbulence.bp", [p_, k_, omega_, nu_t_], engine="BP4")
        vtx_turb.write(0.0)

    if comm.rank == 0:
        print(f"\nSolving RANS {model.display_name}", flush=True)
        print(f"dt={dt}, t_final={solve.t_final}, picard_max={solve.picard_max}", flush=True)
        if solve.snapshot_interval > 0:
            print(
                f"Saving snapshots every {solve.snapshot_interval} steps to {results_dir / 'snps'}",
                flush=True,
            )
        print(flush=True)

    t = 0.0
    step = 0
    residual_prev = 1.0
    ema_ratio = 1.0
    ema_alpha = 0.1
    prev_u_bulk = None
    prev_tau_wall = None

    table = None
    hist = None
    if comm.rank == 0 and solve.log_interval > 0:
        scalar_range_label = f"{model.scalar_name}[min,max]"
        table = StepTablePrinter([
            ("iter", 6),
            ("dt", 9),
            ("res", 9),
            ("U_bulk", 9),
            ("tau_wall", 7),
            ("u_max", 9),
            ("k[min,max]", 20),
            (scalar_range_label, 20),
            ("nu_t/nu", 12),
            ("CFL", 9),
        ])
        hist = HistoryWriterCSV(
            results_dir / "history.csv",
            ["iter", "dt", "residual", "res_u", "res_k", "res_w", "U_bulk", "tau_wall", "u_max", "k_min", "k_max", "omega_min", "omega_max", "nu_t_nu_max", "cfl_max"],
            enabled=True,
        )

    # Pre-compute loop-invariant values
    import dolfinx.cpp.mesh as _cmesh
    tdim = domain.topology.dim
    num_local_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_local_cells, dtype=np.int32)
    h_cells = _cmesh.h(domain._cpp_object, tdim, cell_indices)
    h_min = float(comm.allreduce(np.min(h_cells), op=MPI.MIN))

    cfl_target = float(solve.cfl_target)
    if cfl_target <= 0.0:
        raise ValueError("solve.cfl_target must be positive.")
    u_max_init = float(comm.allreduce(np.max(np.abs(u_.x.array)), op=MPI.MAX))
    current_dt = min(solve.dt, cfl_target * h_min / max(u_max_init, 1e-12), solve.dt_max)
    if comm.rank == 0:
        print(f"  h_min={h_min:.4e}, u_max_init={u_max_init:.3f}")
        print(f"  CFL_target={cfl_target} → dt_init={current_dt:.4e} (dt_max={solve.dt_max})")

    ur_kw = solve.under_relax_k_omega
    alpha_u = 0.7  # velocity under-relaxation factor

    u_n_old = np.empty_like(u_n.x.array)
    u_n_picard = np.empty_like(u_n.x.array)

    def field_residual(new_arr, old_arr):
        diff = new_arr - old_arr
        diff2 = float(comm.allreduce(np.dot(diff, diff), op=MPI.SUM))
        norm2 = float(comm.allreduce(np.dot(new_arr, new_arr), op=MPI.SUM))
        return float(np.sqrt(diff2) / max(np.sqrt(norm2), 1e-10))

    # Main iteration loop
    if comm.rank == 0:
        print("Entering iteration loop...", flush=True)

    while step < solve.max_iter:
        dt_c.value = current_dt
        t += current_dt
        step += 1

        # Save old values for residual computation
        u_n_old[:] = u_n.x.array
        u_n_picard[:] = u_n.x.array
        k_prev.x.array[:] = k_n.x.array
        omega_prev.x.array[:] = omega_n.x.array
        nu_t_old.x.array[:] = nu_t_.x.array

        for picard_iter in range(solve.picard_max):
            # STEP 1: MOMENTUM (IPCS)
            _reassemble_matrix(A1, a1, bcu, mpc=mpc_V)
            _reassemble_vector(b1, L1, a1, bcu, mpc=mpc_V)
            solver1.solve(b1, u_s.x.petsc_vec)
            u_s.x.scatter_forward()
            if mpc_V is not None:
                mpc_V.backsubstitution(u_s)

            # STEP 2: PRESSURE CORRECTION
            _reassemble_vector(b2, L2, a2, bcp, mpc=mpc_Q)
            if pressure_ns is not None:
                pressure_ns.remove(b2)
            solver2.solve(b2, phi.x.petsc_vec)
            phi.x.scatter_forward()
            if mpc_Q is not None:
                mpc_Q.backsubstitution(phi)

            p_.x.petsc_vec.axpy(1.0, phi.x.petsc_vec)
            p_.x.scatter_forward()
            if mpc_Q is not None:
                mpc_Q.backsubstitution(p_)

            # STEP 3: VELOCITY CORRECTION
            _reassemble_vector_no_bc(b3, L3, mpc=mpc_V)
            solver3.solve(b3, u_.x.petsc_vec)
            u_.x.scatter_forward()
            if mpc_V is not None:
                mpc_V.backsubstitution(u_)

            # STEP 4: UPDATE u_n WITH UNDER-RELAXATION (Picard)
            u_n.x.array[:] = alpha_u * u_.x.array + (1.0 - alpha_u) * u_n_picard
            u_n.x.scatter_forward()
            u_n_picard[:] = u_n.x.array

            # STEP 4b: Update model auxiliary fields (SST: F1/F2 blending; others: no-op)
            model.update_auxiliary_fields(
                np.maximum(k_.x.array, k_spec.clip_min),
                np.maximum(omega_.x.array, scalar_spec.clip_min),
            )

            # STEP 5: k-equation
            _reassemble_matrix(A_k, a_k, bck, mpc=mpc_S)
            _reassemble_vector(b_k, L_k, a_k, bck, mpc=mpc_S)
            solver_k.solve(b_k, k_.x.petsc_vec)
            k_.x.scatter_forward()
            if mpc_S is not None:
                mpc_S.backsubstitution(k_)

            k_.x.array[:] = np.clip(k_.x.array, k_spec.clip_min, k_spec.clip_max)
            k_.x.array[:] = ur_kw * k_.x.array + (1.0 - ur_kw) * k_prev.x.array
            k_.x.scatter_forward()

            # STEP 6: scalar equation (ω or ε)
            _reassemble_matrix(A_w, a_w, bcw, mpc=mpc_S)
            _reassemble_vector(b_w, L_w, a_w, bcw, mpc=mpc_S)
            solver_w.solve(b_w, omega_.x.petsc_vec)
            omega_.x.scatter_forward()
            if mpc_S is not None:
                mpc_S.backsubstitution(omega_)

            omega_.x.array[:] = np.clip(omega_.x.array, scalar_spec.clip_min, scalar_spec.clip_max)
            omega_.x.array[:] = ur_kw * omega_.x.array + (1.0 - ur_kw) * omega_prev.x.array
            omega_.x.scatter_forward()

            # STEP 7: Update nu_t (model-dependent)
            S_mag_.interpolate(model.S_mag_expr)
            S_mag_arr = S_mag_.x.array

            nu_t_raw = model.compute_nu_t(k_.x.array, omega_.x.array, S_mag_arr)
            nu_t_raw = np.clip(nu_t_raw, 0, turb.nu_t_max_factor * nu)
            ur_nu_t = solve.under_relax_nu_t
            nu_t_.x.array[:] = ur_nu_t * nu_t_raw + (1.0 - ur_nu_t) * nu_t_old.x.array
            nu_t_.x.scatter_forward()

            # STEP 8: Picard convergence check
            dk = k_.x.array - k_prev.x.array
            dk_norm = float(np.sqrt(comm.allreduce(np.dot(dk, dk), op=MPI.SUM)))
            k_norm = float(np.sqrt(comm.allreduce(np.dot(k_.x.array, k_.x.array), op=MPI.SUM)))
            res_k_val = dk_norm / max(k_norm, 1e-10)

            if res_k_val < solve.picard_tol:
                break

            k_prev.x.array[:] = k_.x.array
            omega_prev.x.array[:] = omega_.x.array
            nu_t_old.x.array[:] = nu_t_.x.array

        # Compute residuals
        res_u = field_residual(u_.x.array, u_n_old)
        res_k = field_residual(k_.x.array, k_n.x.array)
        res_w = field_residual(omega_.x.array, omega_n.x.array)
        residual = max(res_u, res_k, res_w)

        # Update for next step
        u_n1.x.array[:] = u_n_old
        u_n.x.array[:] = u_.x.array
        u_n.x.scatter_forward()
        k_n.x.array[:] = k_.x.array
        omega_n.x.array[:] = omega_.x.array

        # ── POST-STEP: Logging, CSV, VTX snapshots, PNG plots ─────
        do_log = step % solve.log_interval == 0
        do_snapshot = solve.snapshot_interval > 0 and step % solve.snapshot_interval == 0
        collect_physical = solve.enable_physical_convergence or do_log or do_snapshot

        if collect_physical:
            if not is_bfs:
                U_bulk = compute_bulk_velocity(u_, geom.Lx, Ly)
            else:
                U_bulk = 0.0
            tau_wall = eval_wall_shear_stress(wss_ctx)
        else:
            U_bulk = 0.0
            tau_wall = 0.0

        physical_converged = True
        if solve.enable_physical_convergence:
            if prev_u_bulk is None or prev_tau_wall is None:
                physical_converged = False
            elif step >= solve.physical_convergence_start_iter:
                du_bulk = abs(U_bulk - prev_u_bulk) / max(abs(prev_u_bulk), 1e-12)
                dtau_wall = abs(tau_wall - prev_tau_wall) / max(abs(prev_tau_wall), 1e-12)
                physical_converged = (
                    (solve.physical_u_bulk_rel_tol <= 0.0 or du_bulk <= solve.physical_u_bulk_rel_tol)
                    and (solve.physical_tau_wall_rel_tol <= 0.0 or dtau_wall <= solve.physical_tau_wall_rel_tol)
                )
            prev_u_bulk = U_bulk
            prev_tau_wall = tau_wall

        if do_log or do_snapshot:
            ud = diagnostics_vector(u_)
            kd = diagnostics_scalar(k_)
            wd = diagnostics_scalar(omega_)
            nu_t_ratio = float(np.max(nu_t_.x.array)) / nu
            cfl_max = float(ud["umax"]) * current_dt / h_min

        if comm.rank == 0 and table is not None and do_log:
            table.row([
                f"{step:6d}",
                f"{current_dt:9.2e}",
                f"{residual:9.1e}",
                f"{U_bulk:9.3f}",
                f"{tau_wall:7.4f}",
                f"{float(ud['umax']):9.3f}",
                fmt_pair_sci(float(kd["min"]), float(kd["max"]), prec=1),
                fmt_pair_sci(float(wd["min"]), float(wd["max"]), prec=1),
                f"{nu_t_ratio:12.1f}",
                f"{cfl_max:9.2e}",
            ])
            hist.write({
                "iter": step,
                "dt": current_dt,
                "residual": residual,
                "res_u": res_u,
                "res_k": res_k,
                "res_w": res_w,
                "U_bulk": U_bulk,
                "tau_wall": tau_wall,
                "u_max": float(ud["umax"]),
                "k_min": float(kd["min"]),
                "k_max": float(kd["max"]),
                "omega_min": float(wd["min"]),
                "omega_max": float(wd["max"]),
                "nu_t_nu_max": nu_t_ratio,
                "cfl_max": cfl_max,
            })

        # VTX snapshots
        if vtx_vel is not None and do_snapshot:
            vtx_vel.write(t)
            vtx_turb.write(t)

        # Convergence plot at log_interval
        if comm.rank == 0 and do_log:
            _plot_convergence_live(results_dir / "history.csv", results_dir / "convergence.png")

        # Field PNG plots at snapshot_interval
        if do_snapshot:
            if is_bfs:
                from dolfinx_rans.plotting import plot_bfs_fields
                save_path = results_dir / f"bfs_fields{step:07d}.png"
                plot_bfs_fields(u_, p_, k_, omega_, nu_t_, domain, geom, nu, save_path=save_path)
            else:
                _plot_fields_live(
                    u_, p_, k_, omega_, nu_t_, domain, geom, Re_tau, step,
                    results_dir / f"fields{step:07d}.png",
                )

        # Convergence check
        if residual < solve.steady_tol and (not solve.enable_physical_convergence or physical_converged):
            dt_convergence_ok = True
            if solve.min_iter > 0 and step < solve.min_iter:
                dt_convergence_ok = False
            if solve.min_dt_ratio > 0.0 and current_dt < solve.min_dt_ratio * solve.dt:
                dt_convergence_ok = False

            if dt_convergence_ok:
                if comm.rank == 0:
                    print(f"\n*** CONVERGED at iteration {step} (residual = {residual:.2e}) ***")
                break

        # Adaptive dt
        u_max_now = float(comm.allreduce(np.max(np.abs(u_.x.array)), op=MPI.MAX))
        dt_cfl = cfl_target * h_min / max(u_max_now, 1e-12)
        residual_ratio = residual / max(residual_prev, 1e-15)
        ema_ratio = ema_alpha * residual_ratio + (1.0 - ema_alpha) * ema_ratio

        dt_candidate = current_dt
        if ema_ratio < solve.dt_growth_threshold:
            dt_candidate *= solve.dt_growth
        elif ema_ratio > 1.05:
            dt_candidate *= 0.5

        current_dt = min(dt_candidate, dt_cfl, solve.dt_max)
        if solve.min_dt_ratio > 0.0:
            dt_min_abs = solve.min_dt_ratio * solve.dt
            current_dt = max(current_dt, min(dt_min_abs, dt_cfl, solve.dt_max))
        residual_prev = residual

    # Final snapshot
    if vtx_vel is not None:
        vtx_vel.write(t)
        vtx_vel.close()
        vtx_turb.write(t)
        vtx_turb.close()

    # Cleanup
    for slv in [solver_k, solver_w, solver1, solver2, solver3]:
        slv.destroy()
    for mat in [A_k, A_w, A1, A2, A3]:
        mat.destroy()
    for vec in [b_k, b_w, b1, b2, b3]:
        vec.destroy()

    if hist is not None:
        hist.close()

    return u_, p_, k_, omega_, nu_t_, V, Q, S, domain, step, t
