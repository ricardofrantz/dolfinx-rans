"""
RANS k-omega solver for turbulent channel flow - DOLFINx 0.10.0+

Standard k-omega model with:
- Pseudo-transient continuation to steady state
- Adaptive time stepping with hysteresis
- Wall-refined mesh with geometric or tanh stretching
- Optional Durbin realizability limiter
- High-Re benchmark workflow (Nek5000-style periodic channel)
- Optional external cross-code profile comparison

GOVERNING EQUATIONS (NONDIMENSIONAL)
====================================
Scaling: delta = half-channel height, u_tau = friction velocity
    nu* = 1/Re_tau (nondimensional viscosity)

Momentum:
    du/dt + (u.grad)u = -grad(p) + div[(nu* + nu_t*)grad(u)] + f_x
    where f_x = 1 (body force to maintain u_tau = 1)

k-equation:
    dk/dt + u.grad(k) = P_k - beta*k*omega + div[(nu* + sigma_k*nu_t*)grad(k)]

omega-equation:
    domega/dt + u.grad(omega) = gamma*(omega/k)*P_k - beta*omega^2 + div[(nu* + sigma_w*nu_t*)grad(omega)]

WALL BOUNDARY CONDITIONS
========================
k: Dirichlet k = 0
omega: Dirichlet omega = 6*nu*/(beta_1*y^2) with y = first cell height

REFERENCE
=========
Nek5000 RANS tutorial: https://nek5000.github.io/NekDoc/tutorials/rans.html
Legacy DNS reference: Moser, Kim, Mansour (1999), Phys. Fluids 11(4):943-945
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh
from dolfinx.fem import (
    Constant,
    Expression,
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

try:
    from dolfinx_mpc import MultiPointConstraint
    from dolfinx_mpc import apply_lifting as mpc_apply_lifting
    from dolfinx_mpc import assemble_matrix as mpc_assemble_matrix
    from dolfinx_mpc import assemble_vector as mpc_assemble_vector

    HAVE_MPC = True
except ImportError:
    HAVE_MPC = False

import ufl
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
    sqrt,
    sym,
)

from dolfinx_rans.config import (
    BETA_0,
    BETA_STAR,
    GAMMA,
    SIGMA_D0,
    SIGMA_K,
    SIGMA_W,
    SQRT_BETA_STAR,
    SST_A1,
    SST_BETA1,
    SST_BETA2,
    SST_GAMMA1,
    SST_GAMMA2,
    SST_SIGMA_K1,
    SST_SIGMA_K2,
    SST_SIGMA_W1,
    SST_SIGMA_W2,
    ChannelGeom,
    NondimParams,
    SolveParams,
    TurbParams,
)
from dolfinx_rans.geometry import (
    compute_wall_distance_channel,
    create_channel_mesh,
    infer_first_offwall_spacing,
    initial_k_channel,
    initial_omega_channel,
    initial_velocity_channel,
    mark_channel_boundaries,
)
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
    """Plot convergence history from CSV (called during iteration)."""
    import csv
    import matplotlib.pyplot as plt

    if not history_file.exists():
        return

    iters, residuals, res_u, res_k, res_w = [], [], [], [], []
    with open(history_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iter"]))
            residuals.append(float(row["residual"]))
            res_u.append(float(row.get("res_u", row["residual"])))
            res_k.append(float(row.get("res_k", row["residual"])))
            res_w.append(float(row.get("res_w", row["residual"])))

    if len(iters) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(iters, residuals, "k-", linewidth=2, label="max(u,k,omega)")
    ax.semilogy(iters, res_u, "b--", linewidth=1, alpha=0.7, label="u")
    ax.semilogy(iters, res_k, "r--", linewidth=1, alpha=0.7, label="k")
    ax.semilogy(iters, res_w, "g--", linewidth=1, alpha=0.7, label="omega")
    ax.axhline(1e-6, color="gray", linestyle=":", alpha=0.5, label="tol=1e-6")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (relative L2 norm)")
    ax.set_title("Convergence History")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
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

    # Save numbered snapshot AND latest (for quick viewing)
    numbered = save_path.parent / f"fields_{step:05d}.png"
    fig.savefig(numbered, dpi=150, bbox_inches="tight")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main solver
# =============================================================================


def solve_rans_kw(
    domain,
    geom: ChannelGeom,
    turb: TurbParams,
    solve: SolveParams,
    results_dir: Path,
    nondim: NondimParams,
):
    """
    Solve RANS k-omega channel flow with pseudo-transient continuation.

    Args:
        domain: DOLFINx mesh
        geom: Channel geometry
        turb: Turbulence model parameters
        solve: Solver parameters
        results_dir: Output directory
        nondim: Nondimensional scaling parameters

    Returns:
        Tuple of (u, p, k, omega, nu_t, V, Q, S, domain, step, t)
    """
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    comm = domain.comm

    Ly = geom.Ly
    H = Ly if geom.use_symmetry else Ly / 2.0  # H = delta (half-channel height)

    # Nondimensional mode
    Re_tau = nondim.Re_tau
    nu = 1.0 / Re_tau
    rho = 1.0
    use_body_force = nondim.use_body_force
    u_bulk_init = 15.0  # Conservative initial

    if comm.rank == 0:
        print(f"NONDIMENSIONAL MODE: Re_tau = {Re_tau}", flush=True)
        print(f"  nu* = 1/Re_tau = {nu:.6f}", flush=True)
        print(f"  Body force: f_x = {1.0 if use_body_force else 0.0}", flush=True)

    dt = solve.dt
    k_min = turb.k_min
    k_max_limit = turb.k_max
    omega_min = turb.omega_min
    C_lim = turb.C_lim

    # Function spaces
    V0 = functionspace(domain, ("Lagrange", 2, (gdim,)))
    Q0 = functionspace(domain, ("Lagrange", 1))
    S0 = functionspace(domain, ("Lagrange", 1))

    # Boundary facets
    bottom_facets, top_facets, left_facets, right_facets = mark_channel_boundaries(domain, geom.Lx, Ly)

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
    omega_wall_val = 6.0 * nu / (BETA_0 * y_first**2)

    if geom.use_symmetry:
        # Half-channel: wall at bottom, symmetry at top
        # Bottom: no-slip (u=0, v=0), k=0, omega=omega_wall
        # Top: symmetry (v=0, du/dy=0 natural, dk/dy=0 natural, domega/dy=0 natural)
        wall_facets_tb = bottom_facets  # Only bottom wall for symmetry case
        wall_dofs_V0 = locate_dofs_topological(V0, fdim, bottom_facets)
        wall_dofs_S0 = locate_dofs_topological(S0, fdim, bottom_facets)

        u_noslip = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
        bc_walls_u0 = dirichletbc(u_noslip, wall_dofs_V0, V0)
        bc_k_wall0 = dirichletbc(PETSc.ScalarType(0.0), wall_dofs_S0, S0)

        # Symmetry BC at top: v=0 (y-component only)
        # DOLFINx 0.10.0 requires Function on collapsed subspace for subspace BC
        V0_y_sub = V0.sub(1)
        V0_y_collapsed, _ = V0_y_sub.collapse()
        zero_vy = Function(V0_y_collapsed)
        zero_vy.x.array[:] = 0.0
        top_dofs_Vy = locate_dofs_topological((V0_y_sub, V0_y_collapsed), fdim, top_facets)
        bc_sym_v0 = dirichletbc(zero_vy, top_dofs_Vy, V0_y_sub)

        bcs_u0 = [bc_walls_u0, bc_sym_v0]
    else:
        # Full channel: walls at both top and bottom
        wall_facets_tb = np.concatenate([bottom_facets, top_facets])
        wall_dofs_V0 = locate_dofs_topological(V0, fdim, wall_facets_tb)
        wall_dofs_S0 = locate_dofs_topological(S0, fdim, wall_facets_tb)

        u_noslip = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
        bc_walls_u0 = dirichletbc(u_noslip, wall_dofs_V0, V0)
        bc_k_wall0 = dirichletbc(PETSc.ScalarType(0.0), wall_dofs_S0, S0)

        bcs_u0 = [bc_walls_u0]

    use_periodic = use_body_force
    mpc_V = None
    mpc_Q = None
    mpc_S = None

    if use_periodic:
        if not HAVE_MPC:
            raise RuntimeError(
                "Periodic BCs require dolfinx_mpc. Install: conda install -c conda-forge dolfinx_mpc"
            )
        tol = max(1e-10, 250 * np.finfo(domain.geometry.x.dtype).eps)
        map_eps = max(1e-12, tol)

        def periodic_boundary(x):
            return np.isclose(x[0], geom.Lx, atol=tol)

        def periodic_relation(x):
            y = x.copy()
            y[0] -= geom.Lx
            # Keep mapped points slightly inside the master side to avoid
            # point-location failures on quadrilateral boundary geometry.
            y[0] = np.where(np.isclose(y[0], 0.0, atol=tol), map_eps, y[0])
            return y

        use_topological_periodic = (
            geom.mesh_type == "quad"
            and (geom.stretching.lower() == "tanh" or geom.growth_rate > 1.0)
        )
        if use_topological_periodic:
            right_facets_sorted = np.unique(np.sort(right_facets)).astype(np.int32, copy=False)
            if right_facets_sorted.size == 0:
                raise RuntimeError("Could not locate right boundary facets for periodic constraint setup.")
            slave_tag = np.int32(1)
            right_tags = np.full(right_facets_sorted.shape, slave_tag, dtype=np.int32)
            right_mt = mesh.meshtags(domain, fdim, right_facets_sorted, right_tags)

        mpc_V = MultiPointConstraint(V0)
        if use_topological_periodic:
            mpc_V.create_periodic_constraint_topological(
                V0, right_mt, int(slave_tag), periodic_relation, bcs_u0, tol=tol
            )
        else:
            mpc_V.create_periodic_constraint_geometrical(
                V0, periodic_boundary, periodic_relation, bcs_u0
            )
        mpc_V.finalize()

        mpc_Q = MultiPointConstraint(Q0)
        if use_topological_periodic:
            mpc_Q.create_periodic_constraint_topological(
                Q0, right_mt, int(slave_tag), periodic_relation, [], tol=tol
            )
        else:
            mpc_Q.create_periodic_constraint_geometrical(Q0, periodic_boundary, periodic_relation, [])
        mpc_Q.finalize()

        mpc_S = MultiPointConstraint(S0)
        if use_topological_periodic:
            mpc_S.create_periodic_constraint_topological(
                S0, right_mt, int(slave_tag), periodic_relation, [bc_k_wall0], tol=tol
            )
        else:
            mpc_S.create_periodic_constraint_geometrical(
                S0, periodic_boundary, periodic_relation, [bc_k_wall0]
            )
        mpc_S.finalize()

        V = mpc_V.function_space
        Q = mpc_Q.function_space
        S = mpc_S.function_space
    else:
        V, Q, S = V0, Q0, S0

    if comm.rank == 0:
        print(f"Velocity DOFs: {V.dofmap.index_map.size_global}", flush=True)
        print(f"Pressure DOFs: {Q.dofmap.index_map.size_global}", flush=True)
        print(f"Scalar DOFs (k, omega): {S.dofmap.index_map.size_global}", flush=True)
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

    # S_magnitude needed for stress limiter (always computed)
    S_mag_ = Function(S, name="S_magnitude")

    # Determine turbulence model
    use_sst = turb.model.lower() == "sst"
    if comm.rank == 0:
        model_name = "k-omega SST (Menter 1994)" if use_sst else "k-omega (Wilcox 2006)"
        print(f"Turbulence model: {model_name}")

    # SST-specific: wall distance and blending functions
    if use_sst:
        y_wall = compute_wall_distance_channel(S, geom.use_symmetry)
        F1_ = Function(S, name="F1")  # Blending function (1 near wall, 0 in freestream)
        F2_ = Function(S, name="F2")  # SST limiter blending
        # Blended coefficients (will be updated each iteration)
        sigma_k_blend = Function(S, name="sigma_k")
        sigma_w_blend = Function(S, name="sigma_w")
        beta_blend = Function(S, name="beta")
        gamma_blend = Function(S, name="gamma")
        # Initialize blending to inner layer (F1=1)
        F1_.x.array[:] = 1.0
        F2_.x.array[:] = 1.0
        sigma_k_blend.x.array[:] = SST_SIGMA_K1
        sigma_w_blend.x.array[:] = SST_SIGMA_W1
        beta_blend.x.array[:] = SST_BETA1
        gamma_blend.x.array[:] = SST_GAMMA1
    else:
        y_wall = None
        F1_ = None
        F2_ = None
        sigma_k_blend = None
        sigma_w_blend = None
        beta_blend = None
        gamma_blend = None

    # Constants
    dt_c = Constant(domain, PETSc.ScalarType(dt))
    nu_c = Constant(domain, PETSc.ScalarType(nu))
    rho_c = Constant(domain, PETSc.ScalarType(rho))

    beta_star_c = Constant(domain, PETSc.ScalarType(turb.beta_star))

    # For Wilcox 2006, use fixed constants; for SST, these will be overridden
    if not use_sst:
        beta_c = Constant(domain, PETSc.ScalarType(BETA_0))
        gamma_c = Constant(domain, PETSc.ScalarType(GAMMA))
        sigma_k_c = Constant(domain, PETSc.ScalarType(SIGMA_K))
        sigma_w_c = Constant(domain, PETSc.ScalarType(SIGMA_W))
    else:
        # For SST, use Functions for blended coefficients (updated each iteration)
        # We'll use these in the weak form
        beta_c = beta_blend
        gamma_c = gamma_blend
        sigma_k_c = sigma_k_blend
        sigma_w_c = sigma_w_blend

    sigma_d_c = Constant(domain, PETSc.ScalarType(SIGMA_D0))  # Cross-diffusion

    if use_body_force:
        f = Constant(domain, PETSc.ScalarType((1.0, 0.0)))
        if comm.rank == 0:
            print("Using body force f_x = 1.0 (periodic channel)")
    else:
        f = Constant(domain, PETSc.ScalarType((0.0, 0.0)))

    # Initial conditions
    u_n.interpolate(lambda x: initial_velocity_channel(x, u_bulk_init, Ly, geom.use_symmetry))
    u_n1.x.array[:] = u_n.x.array
    u_.x.array[:] = u_n.x.array

    k_n.interpolate(lambda x: initial_k_channel(x, u_bulk_init))
    omega_n.interpolate(lambda x: initial_omega_channel(x, u_bulk_init, H, nu))

    k_.x.array[:] = k_n.x.array
    omega_.x.array[:] = omega_n.x.array
    k_prev.x.array[:] = k_n.x.array
    omega_prev.x.array[:] = omega_n.x.array

    nu_t_.x.array[:] = k_n.x.array / (omega_n.x.array + omega_min)
    nu_t_.x.array[:] = np.clip(nu_t_.x.array, 0, turb.nu_t_max_factor * nu)
    nu_t_old.x.array[:] = nu_t_.x.array

    # Periodic backsubstitution
    if use_periodic:
        mpc_V.backsubstitution(u_n)
        mpc_V.backsubstitution(u_n1)
        mpc_V.backsubstitution(u_)
        mpc_S.backsubstitution(k_n)
        mpc_S.backsubstitution(omega_n)
        mpc_S.backsubstitution(k_)
        mpc_S.backsubstitution(omega_)
        mpc_S.backsubstitution(nu_t_)
        mpc_S.backsubstitution(nu_t_old)

    wall_dofs_V_tb = locate_dofs_topological(V, fdim, wall_facets_tb)
    wall_dofs_S_tb = locate_dofs_topological(S, fdim, wall_facets_tb)

    bc_walls_u = dirichletbc(u_noslip, wall_dofs_V_tb, V)

    # Build velocity BCs list
    if use_periodic:
        # Use original-space BC objects with MPC assembly in periodic mode.
        # Avoid reduced-space subspace dof location, which is unstable in MPI here.
        bcu = bcs_u0
    elif geom.use_symmetry:
        # Add symmetry BC at top: v=0 (y-component only) for non-periodic solve loop.
        # DOLFINx 0.10.0 requires Function on collapsed subspace for subspace BC
        V_y_sub = V.sub(1)
        V_y_collapsed, _ = V_y_sub.collapse()
        zero_vy_solve = Function(V_y_collapsed)
        zero_vy_solve.x.array[:] = 0.0
        top_dofs_Vy_solve = locate_dofs_topological((V_y_sub, V_y_collapsed), fdim, top_facets)
        bc_sym_v = dirichletbc(zero_vy_solve, top_dofs_Vy_solve, V_y_sub)
        bcu = [bc_walls_u, bc_sym_v]
    else:
        bcu = [bc_walls_u]

    if use_periodic:
        bcp = []
    else:
        outlet_dofs_Q = locate_dofs_topological(Q, fdim, right_facets)
        bc_pressure = dirichletbc(PETSc.ScalarType(0.0), outlet_dofs_Q, Q)
        bcp = [bc_pressure]

    bc_k_wall = dirichletbc(PETSc.ScalarType(0.0), wall_dofs_S_tb, S)
    bck = [bc_k_wall]

    bc_w_wall = dirichletbc(PETSc.ScalarType(omega_wall_val), wall_dofs_S_tb, S)
    bcw = [bc_w_wall]

    if comm.rank == 0:
        print(
            f"Wall omega BC: omega_wall = {omega_wall_val:.2e} "
            f"(y_first mesh = {y_first:.6f}, requested = {y_first_cfg:.6f})",
            flush=True,
        )
        print("Setting up weak forms...", flush=True)

    comm.barrier()  # Sync before form setup

    # Turbulence forms
    omega_safe = omega_prev

    D_k = nu_c + sigma_k_c * nu_t_
    D_w = nu_c + sigma_w_c * nu_t_

    S_tensor = sym(grad(u_n))
    S_sq = 2.0 * inner(S_tensor, S_tensor)
    P_k = nu_t_ * S_sq

    # S_mag expression for stress limiter: |S| = sqrt(2*S_ij*S_ij)
    S_mag_ufl = sqrt(S_sq + 1e-16)
    S_mag_expr = Expression(S_mag_ufl, S.element.interpolation_points)

    # Gradient dot product for SST cross-diffusion term: grad(k) . grad(omega)
    grad_k_dot_grad_w_ufl = dot(grad(k_n), grad(omega_n))
    grad_k_dot_grad_w_expr = Expression(grad_k_dot_grad_w_ufl, S.element.interpolation_points)
    # Function to hold interpolated values
    grad_k_dot_grad_w_func = Function(S, name="grad_k_dot_grad_w") if use_sst else None

    # k-equation
    F_k = (
        (k_trial - k_n) / dt_c * phi_k * dx
        + dot(u_n, grad(k_trial)) * phi_k * dx
        + D_k * inner(grad(k_trial), grad(phi_k)) * dx
        + beta_star_c * omega_safe * k_trial * phi_k * dx
        - P_k * phi_k * dx
    )

    a_k = form(lhs(F_k))
    L_k = form(rhs(F_k))
    if use_periodic:
        A_k = mpc_assemble_matrix(a_k, mpc_S, bcs=bck)
        A_k.assemble()
        b_k = mpc_assemble_vector(L_k, mpc_S)
    else:
        A_k = create_matrix(a_k)
        b_k = create_vector(S)

    if comm.rank == 0:
        print("  k-equation forms ready", flush=True)

    # omega-equation with cross-diffusion term
    P_omega = gamma_c * S_sq

    # Cross-diffusion term: helps reduce freestream sensitivity
    # We use omega_n (from previous step) to avoid implicit coupling
    grad_k_dot_grad_w = dot(grad(k_n), grad(omega_n))
    # max(0, grad_k.grad_omega) via UFL conditional
    grad_kw_positive = ufl.conditional(ufl.gt(grad_k_dot_grad_w, 0.0), grad_k_dot_grad_w, 0.0)

    if use_sst:
        # SST: CD = (1-F1) * 2*sigma_w2/omega * max(0, grad_k.grad_omega)
        # F1_=1 near wall -> no cross-diffusion (k-omega behavior)
        # F1_=0 away from wall -> full cross-diffusion (k-epsilon transformed)
        sigma_w2_c = Constant(domain, PETSc.ScalarType(2.0 * SST_SIGMA_W2))
        cross_diff = (1.0 - F1_) * sigma_w2_c / omega_safe * grad_kw_positive
    else:
        # Wilcox 2006: CD = sigma_d/omega * max(0, grad_k.grad_omega)
        cross_diff = sigma_d_c / omega_safe * grad_kw_positive

    F_w = (
        (w_trial - omega_n) / dt_c * phi_w * dx
        + dot(u_n, grad(w_trial)) * phi_w * dx
        + D_w * inner(grad(w_trial), grad(phi_w)) * dx
        + beta_c * omega_safe * w_trial * phi_w * dx
        - P_omega * phi_w * dx
        - cross_diff * phi_w * dx  # Cross-diffusion is a source term on RHS
    )

    a_w = form(lhs(F_w))
    L_w = form(rhs(F_w))
    if use_periodic:
        A_w = mpc_assemble_matrix(a_w, mpc_S, bcs=bcw)
        A_w.assemble()
        b_w = mpc_assemble_vector(L_w, mpc_S)
    else:
        A_w = create_matrix(a_w)
        b_w = create_vector(S)

    if comm.rank == 0:
        print("  omega-equation forms ready", flush=True)

    # IPCS momentum
    nu_eff = nu_c + nu_t_
    mu_eff = rho_c * nu_eff

    F1_mom = rho_c / dt_c * dot(u - u_n, v) * dx
    F1_mom += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
    F1_mom += 0.5 * mu_eff * inner(grad(u + u_n), grad(v)) * dx
    F1_mom -= dot(p_, div(v)) * dx
    F1_mom -= dot(f, v) * dx

    a1 = form(lhs(F1_mom))
    L1 = form(rhs(F1_mom))
    if use_periodic:
        A1 = mpc_assemble_matrix(a1, mpc_V, bcs=bcu)
        A1.assemble()
        b1 = mpc_assemble_vector(L1, mpc_V)
    else:
        A1 = create_matrix(a1)
        b1 = create_vector(V)

    if comm.rank == 0:
        print("  Momentum forms ready", flush=True)

    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho_c / dt_c * div(u_s) * q * dx)
    if use_periodic:
        A2 = mpc_assemble_matrix(a2, mpc_Q, bcs=bcp)
        A2.assemble()
        b2 = mpc_assemble_vector(L2, mpc_Q)
    else:
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(Q)

    pressure_nullspace = None
    if use_periodic:
        pressure_nullspace = PETSc.NullSpace().create(constant=True, comm=comm)
        A2.setNullSpace(pressure_nullspace)

    if comm.rank == 0:
        print("  Pressure forms ready", flush=True)

    a3 = form(rho_c * dot(u, v) * dx)
    L3 = form(rho_c * dot(u_s, v) * dx - dt_c * dot(nabla_grad(phi), v) * dx)
    if use_periodic:
        A3 = mpc_assemble_matrix(a3, mpc_V, bcs=[])
        A3.assemble()
        b3 = mpc_assemble_vector(L3, mpc_V)
    else:
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(V)

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

    def assemble_matrix_maybe_mpc(A, a, bcs, mpc):
        A.zeroEntries()
        if mpc is None:
            assemble_matrix(A, a, bcs=bcs)
        else:
            mpc_assemble_matrix(a, mpc, bcs=bcs, A=A)
        A.assemble()

    def assemble_vector_maybe_mpc(b, L, a, bcs, mpc):
        with b.localForm() as loc:
            loc.set(0.0)
        if mpc is None:
            assemble_vector(b, L)
            apply_lifting(b, [a], [bcs])
        else:
            mpc_assemble_vector(L, mpc, b)
            mpc_apply_lifting(b, [a], [bcs], mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

    def assemble_vector_no_bc(b, L, mpc):
        with b.localForm() as loc:
            loc.set(0.0)
        if mpc is None:
            assemble_vector(b, L)
        else:
            mpc_assemble_vector(L, mpc, b)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    def backsubstitute(mpc, func):
        if mpc is not None:
            mpc.backsubstitution(func)

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
        print(f"\nSolving RANS k-omega channel flow", flush=True)
        print(f"dt={dt}, t_final={solve.t_final}, picard_max={solve.picard_max}", flush=True)
        if solve.snapshot_interval > 0:
            print(
                f"Saving snapshots every {solve.snapshot_interval} steps to {results_dir / 'snps'}",
                flush=True,
            )
        print(flush=True)

    t = 0.0
    step = 0
    current_dt = dt
    residual_prev = 1.0
    prev_u_bulk = None
    prev_tau_wall = None

    table = None
    hist = None
    if comm.rank == 0 and solve.log_interval > 0:
        table = StepTablePrinter([
            ("iter", 6),
            ("dt", 9),
            ("res", 9),
            ("U_bulk", 9),
            ("tau_wall", 7),  # Should be 1.0 for equilibrium!
            ("u_max", 9),
            ("k[min,max]", 20),
            ("w[min,max]", 20),
            ("nu_t/nu", 12),
        ])
        hist = HistoryWriterCSV(
            results_dir / "history.csv",
            ["iter", "dt", "residual", "res_u", "res_k", "res_w", "U_bulk", "tau_wall", "u_max", "k_min", "k_max", "omega_min", "omega_max", "nu_t_nu_max"],
            enabled=True,
        )

    # Pre-compute loop-invariant values
    omega_max_limit = 10.0 * omega_wall_val  # Scale wall-driven cap with current case
    ur_kw = solve.under_relax_k_omega
    alpha_u = 0.7  # velocity under-relaxation factor

    u_n_old = np.empty_like(u_n.x.array)

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

        if step == 1 and comm.rank == 0:
            print("  Starting iteration 1...", flush=True)

        # Save old values for residual computation and under-relaxation
        # (u_n will be updated inside Picard loop, so save it here)
        u_n_old[:] = u_n.x.array  # For residual: compare u_new vs u_old
        k_prev.x.array[:] = k_n.x.array
        omega_prev.x.array[:] = omega_n.x.array
        nu_t_old.x.array[:] = nu_t_.x.array

        for picard_iter in range(solve.picard_max):
            # =========================================================
            # STEP 1: MOMENTUM (IPCS) - uses current nu_t
            # =========================================================
            if step == 1 and picard_iter == 0 and comm.rank == 0:
                print("    Solving momentum...", flush=True)
            assemble_matrix_maybe_mpc(A1, a1, bcu, mpc_V)
            assemble_vector_maybe_mpc(b1, L1, a1, bcu, mpc_V)

            solver1.solve(b1, u_s.x.petsc_vec)
            u_s.x.scatter_forward()
            backsubstitute(mpc_V, u_s)

            # =========================================================
            # STEP 2: PRESSURE CORRECTION
            # =========================================================
            if step == 1 and picard_iter == 0 and comm.rank == 0:
                print("    Solving pressure...", flush=True)
            assemble_vector_maybe_mpc(b2, L2, a2, bcp, mpc_Q)
            if pressure_nullspace is not None:
                pressure_nullspace.remove(b2)

            solver2.solve(b2, phi.x.petsc_vec)
            phi.x.scatter_forward()
            backsubstitute(mpc_Q, phi)

            p_.x.petsc_vec.axpy(1.0, phi.x.petsc_vec)
            p_.x.scatter_forward()
            backsubstitute(mpc_Q, p_)

            # =========================================================
            # STEP 3: VELOCITY CORRECTION
            # =========================================================
            assemble_vector_no_bc(b3, L3, mpc_V)

            solver3.solve(b3, u_.x.petsc_vec)
            u_.x.scatter_forward()
            backsubstitute(mpc_V, u_)

            if step == 1 and picard_iter == 0 and comm.rank == 0:
                print("    Velocity correction done", flush=True)

            # =========================================================
            # STEP 4: UPDATE u_n WITH UNDER-RELAXATION
            # This ensures turbulence equations use fresh velocity,
            # but under-relaxation prevents instability from large changes
            # =========================================================
            u_n.x.array[:] = alpha_u * u_.x.array + (1.0 - alpha_u) * u_n_old
            u_n.x.scatter_forward()

            # =========================================================
            # STEP 4b: Update SST blending functions BEFORE k/omega solves
            # This ensures k and omega equations use correct blended coefficients
            # =========================================================
            if use_sst:
                # Get current k, omega values (from previous Picard iter or initial)
                k_arr = np.maximum(k_.x.array, k_min)
                omega_arr = np.maximum(omega_.x.array, omega_min)
                y_arr = y_wall.x.array
                y_safe = np.maximum(y_arr, 1e-10)

                # Compute grad(k).grad(omega) (uses k_n, omega_n from previous timestep)
                grad_k_dot_grad_w_func.interpolate(grad_k_dot_grad_w_expr)
                grad_kw_arr = grad_k_dot_grad_w_func.x.array
                CD_kw = np.maximum(2.0 * SST_SIGMA_W2 / omega_arr * grad_kw_arr, 1e-10)

                # F1: controls k-omega vs k-epsilon blending
                term1 = np.sqrt(k_arr) / (BETA_STAR * omega_arr * y_safe)
                term2 = 500.0 * nu / (y_safe**2 * omega_arr)
                term3 = 4.0 * SST_SIGMA_W2 * k_arr / (CD_kw * y_safe**2)
                arg1 = np.minimum(np.maximum(term1, term2), term3)
                F1_arr = np.tanh(arg1**4)

                # Update F1 (needed for omega equation cross-diffusion term)
                F1_.x.array[:] = F1_arr
                F1_.x.scatter_forward()

                # Blend coefficients: phi = F1*phi1 + (1-F1)*phi2
                sigma_k_blend.x.array[:] = F1_arr * SST_SIGMA_K1 + (1.0 - F1_arr) * SST_SIGMA_K2
                sigma_w_blend.x.array[:] = F1_arr * SST_SIGMA_W1 + (1.0 - F1_arr) * SST_SIGMA_W2
                beta_blend.x.array[:] = F1_arr * SST_BETA1 + (1.0 - F1_arr) * SST_BETA2
                gamma_blend.x.array[:] = F1_arr * SST_GAMMA1 + (1.0 - F1_arr) * SST_GAMMA2
                sigma_k_blend.x.scatter_forward()
                sigma_w_blend.x.scatter_forward()
                beta_blend.x.scatter_forward()
                gamma_blend.x.scatter_forward()

            # =========================================================
            # STEP 5: k-equation (now uses updated u_n for production)
            # =========================================================
            if step == 1 and picard_iter == 0 and comm.rank == 0:
                print("    Solving k-equation...", flush=True)
            assemble_matrix_maybe_mpc(A_k, a_k, bck, mpc_S)
            assemble_vector_maybe_mpc(b_k, L_k, a_k, bck, mpc_S)

            solver_k.solve(b_k, k_.x.petsc_vec)
            k_.x.scatter_forward()
            backsubstitute(mpc_S, k_)

            k_.x.array[:] = np.clip(k_.x.array, k_min, k_max_limit)
            k_.x.array[:] = ur_kw * k_.x.array + (1.0 - ur_kw) * k_prev.x.array
            k_.x.scatter_forward()
            backsubstitute(mpc_S, k_)

            # =========================================================
            # STEP 6: omega-equation (now uses updated u_n for production)
            # =========================================================
            if step == 1 and picard_iter == 0 and comm.rank == 0:
                print("    Solving omega-equation...", flush=True)
            assemble_matrix_maybe_mpc(A_w, a_w, bcw, mpc_S)
            assemble_vector_maybe_mpc(b_w, L_w, a_w, bcw, mpc_S)

            solver_w.solve(b_w, omega_.x.petsc_vec)
            omega_.x.scatter_forward()
            backsubstitute(mpc_S, omega_)

            omega_.x.array[:] = np.clip(omega_.x.array, omega_min, omega_max_limit)
            omega_.x.array[:] = ur_kw * omega_.x.array + (1.0 - ur_kw) * omega_prev.x.array
            omega_.x.scatter_forward()
            backsubstitute(mpc_S, omega_)

            # =========================================================
            # STEP 7: Update nu_t (model-dependent)
            # =========================================================
            # Compute strain rate magnitude |S| = sqrt(2*S_ij*S_ij)
            S_mag_.interpolate(S_mag_expr)
            S_mag_arr = S_mag_.x.array
            S_mag_safe = np.maximum(S_mag_arr, 1e-10)

            if use_sst:
                # =======================================================
                # SST Model: Compute F2 and apply SST nu_t limiter
                # (F1 and blended coefficients already updated in Step 4b)
                # =======================================================
                k_arr = np.maximum(k_.x.array, k_min)
                omega_arr = np.maximum(omega_.x.array, omega_min)
                y_arr = y_wall.x.array
                y_safe = np.maximum(y_arr, 1e-10)

                # F2 for SST nu_t limiter (computed with UPDATED k, omega)
                # arg2 = max(2*sqrt(k)/(beta*.omega.y), 500*nu/(y^2.omega))
                term2a = 2.0 * np.sqrt(k_arr) / (BETA_STAR * omega_arr * y_safe)
                term2b = 500.0 * nu / (y_safe**2 * omega_arr)
                arg2 = np.maximum(term2a, term2b)
                F2_arr = np.tanh(arg2**2)

                F2_.x.array[:] = F2_arr
                F2_.x.scatter_forward()

                # SST nu_t limiter: nu_t = a1*k / max(a1*omega, |S|*F2)
                denominator = np.maximum(SST_A1 * omega_arr, S_mag_safe * F2_arr)
                nu_t_raw = SST_A1 * k_arr / denominator
            else:
                # =======================================================
                # Wilcox 2006: nu_t = k/omega_tilde where omega_tilde = max(omega, C_lim*|S|/sqrt(beta*))
                # =======================================================
                omega_tilde = np.maximum(
                    omega_.x.array,
                    C_lim * S_mag_safe / SQRT_BETA_STAR,
                )
                omega_tilde = np.maximum(omega_tilde, omega_min)
                nu_t_raw = k_.x.array / omega_tilde

            # Apply limits and under-relaxation (both models)
            nu_t_raw = np.clip(nu_t_raw, 0, turb.nu_t_max_factor * nu)
            ur_nu_t = solve.under_relax_nu_t
            nu_t_.x.array[:] = ur_nu_t * nu_t_raw + (1.0 - ur_nu_t) * nu_t_old.x.array
            nu_t_.x.scatter_forward()
            backsubstitute(mpc_S, nu_t_)

            # =========================================================
            # STEP 8: Picard convergence check
            # =========================================================
            dk = k_.x.array - k_prev.x.array
            dk_norm = float(np.sqrt(comm.allreduce(np.dot(dk, dk), op=MPI.SUM)))
            k_norm = float(np.sqrt(comm.allreduce(np.dot(k_.x.array, k_.x.array), op=MPI.SUM)))
            res_k_val = dk_norm / max(k_norm, 1e-10)

            if res_k_val < solve.picard_tol:
                break

            k_prev.x.array[:] = k_.x.array
            omega_prev.x.array[:] = omega_.x.array
            nu_t_old.x.array[:] = nu_t_.x.array

        # Compute residuals (relative L2 norm of change from previous outer iteration)
        res_u = field_residual(u_.x.array, u_n_old)  # u^(n+1) vs u^n (saved at start)
        res_k = field_residual(k_.x.array, k_n.x.array)  # k^(n+1) vs k^n
        res_w = field_residual(omega_.x.array, omega_n.x.array)  # omega^(n+1) vs omega^n
        residual = max(res_u, res_k, res_w)  # Converge when ALL fields converge

        # Update for next step: Adams-Bashforth needs u^n and u^{n-1}
        # CRITICAL: u_n inside Picard loop was under-relaxed for stability.
        # For the NEXT timestep's AB extrapolation, we need the ACTUAL converged velocity.
        u_n1.x.array[:] = u_n_old  # u^{n-1} = velocity at start of this timestep
        u_n.x.array[:] = u_.x.array  # u^n = actual IPCS-corrected velocity (not under-relaxed!)
        u_n.x.scatter_forward()
        k_n.x.array[:] = k_.x.array
        omega_n.x.array[:] = omega_.x.array

        # =====================================================================
        # POST-STEP: Logging, CSV, VTX snapshots, and PNG plots
        # =====================================================================
        # log_interval:      print to screen + write CSV row (cheap)
        # snapshot_interval:  save VTX + generate PNG plots (expensive)
        #
        # MPI collectives (U_bulk, tau_wall) run during logging/snapshot by default.
        # =====================================================================

        do_log = step % solve.log_interval == 0
        do_snapshot = solve.snapshot_interval > 0 and step % solve.snapshot_interval == 0
        collect_physical = solve.enable_physical_convergence or do_log or do_snapshot

        # MPI collectives -- all ranks must participate (even if only rank 0 prints)
        if collect_physical:
            U_bulk = compute_bulk_velocity(u_, geom.Lx, Ly)
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
            ud = diagnostics_vector(u_)   # contains allreduce
            kd = diagnostics_scalar(k_)   # contains allreduce
            wd = diagnostics_scalar(omega_)  # contains allreduce
            nu_t_ratio = float(np.max(nu_t_.x.array)) / nu

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
            })

        # VTX snapshots (ADIOS2 for ParaView)
        if vtx_vel is not None and do_snapshot:
            vtx_vel.write(t)
            vtx_turb.write(t)

        # PNG plots -- decoupled from VTX
        if do_snapshot:
            # Flow fields: all ranks must participate (MPI-safe extraction)
            _plot_fields_live(u_, p_, k_, omega_, nu_t_, domain, geom, Re_tau, step, results_dir / "fields.png")
            if comm.rank == 0:
                # Residual history: reads CSV, no MPI needed
                _plot_convergence_live(results_dir / "history.csv", results_dir / "convergence.png")

            # Convergence check
        if residual < solve.steady_tol and (not solve.enable_physical_convergence or physical_converged):
            if comm.rank == 0:
                print(f"\n*** CONVERGED at iteration {step} (residual = {residual:.2e}) ***")
            break

        # Adaptive dt
        residual_ratio = residual / max(residual_prev, 1e-15)
        if residual_ratio < solve.dt_growth_threshold:
            current_dt = min(current_dt * solve.dt_growth, solve.dt_max)
        elif residual_ratio > 1.0 / solve.dt_growth_threshold:
            current_dt = max(current_dt / solve.dt_growth, solve.dt)
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
