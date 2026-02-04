"""
RANS k-ω solver for turbulent channel flow - DOLFINx 0.10.0+

Standard k-ω model with:
- Pseudo-transient continuation to steady state
- Adaptive time stepping with hysteresis
- Wall-refined mesh with geometric stretching
- Optional Durbin realizability limiter
- Validation against MKM DNS (Re_τ = 180, 590)

GOVERNING EQUATIONS (NONDIMENSIONAL)
====================================
Scaling: δ = half-channel height, u_τ = friction velocity
    ν* = 1/Re_τ (nondimensional viscosity)

Momentum:
    ∂u/∂t + (u·∇)u = -∇p + ∇·[(ν* + ν_t*)∇u] + f_x
    where f_x = 1 (body force to maintain u_τ = 1)

k-equation:
    ∂k/∂t + u·∇k = P_k - β*·k·ω + ∇·[(ν* + σ_k·ν_t*)∇k]

ω-equation:
    ∂ω/∂t + u·∇ω = γ·(ω/k)·P_k - β·ω² + ∇·[(ν* + σ_ω·ν_t*)∇ω]

WALL BOUNDARY CONDITIONS
========================
k: Dirichlet k = 0
ω: Dirichlet ω = 6ν*/(β₁·y²) with y = first cell height

REFERENCE
=========
DNS: Moser, Kim, Mansour (1999) "DNS of turbulent channel flow up to Re_τ=590"
     Physics of Fluids, 11(4):943-945. DOI: 10.1063/1.869966
"""

from dataclasses import dataclass
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
from dolfinx.mesh import CellType

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
    CellDiameter,
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

from dolfinx_rans.utils import (
    HistoryWriterCSV,
    StepTablePrinter,
    dc_from_dict,
    diagnostics_scalar,
    diagnostics_vector,
    fmt_pair_sci,
    load_json_config,
    prepare_case_dir,
    print_dc_json,
)

# k-ω model constants (Wilcox 1998)
BETA_STAR = 0.09
BETA = 0.075
SIGMA_K = 0.5
SIGMA_W = 0.5
GAMMA = BETA / BETA_STAR - SIGMA_W * 0.41**2 / np.sqrt(BETA_STAR)  # ≈ 0.556


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass(frozen=True)
class ChannelGeom:
    """Channel geometry parameters."""

    Lx: float  # Channel length (streamwise)
    Ly: float  # Channel height (2δ where δ = half-height)
    Nx: int  # Mesh cells in x
    Ny: int  # Mesh cells in y
    mesh_type: str  # "triangle" or "quad"
    y_first: float  # First cell height from wall (for y+ control)
    growth_rate: float  # Geometric stretching ratio (>1 for wall refinement)


@dataclass(frozen=True)
class NondimParams:
    """Nondimensional parameters for Re_τ-based scaling."""

    Re_tau: float  # Friction Reynolds number
    use_body_force: bool = True  # f_x = 1 to drive flow


@dataclass(frozen=True)
class TurbParams:
    """
    Turbulence model parameters (all required in config).

    beta_star: k-ω model constant (standard: 0.09)
    nu_t_max_factor: Max ν_t/ν ratio for stability
    omega_min: Floor on ω to prevent ν_t runaway (10 = best for U+)
    k_min: Floor on k for positivity (1e-10)
    k_max: Cap on k for safety (DNS k+_max ≈ 5)
    C_lim: Durbin realizability ν_t ≤ C_lim·k/(√6·|S|) (0 = disabled)
    """

    beta_star: float
    nu_t_max_factor: float
    omega_min: float
    k_min: float
    k_max: float
    C_lim: float


@dataclass(frozen=True)
class SolveParams:
    """
    Solver parameters (all required in config).

    dt: Initial pseudo-time step
    dt_max: Maximum dt for implicit stepping
    dt_growth: dt multiplier when converging well
    dt_growth_threshold: Only grow dt if residual_ratio < threshold (hysteresis)
    t_final: Max pseudo-time (safety limit, usually not reached)
    max_iter: Max iterations before giving up
    steady_tol: Convergence tolerance on velocity residual
    picard_max: Inner Picard iterations per time step
    picard_tol: Picard convergence tolerance
    under_relax_k_omega: Under-relaxation for k and ω (0.7 typical)
    under_relax_nu_t: Under-relaxation for ν_t (0.5 typical)
    log_interval: Print every N iterations
    snapshot_interval: Save VTX every N iterations (0 = disabled)
    out_dir: Output directory for results
    """

    dt: float
    dt_max: float
    dt_growth: float
    dt_growth_threshold: float
    t_final: float
    max_iter: int
    steady_tol: float
    picard_max: int
    picard_tol: float
    under_relax_k_omega: float
    under_relax_nu_t: float
    log_interval: int
    snapshot_interval: int
    out_dir: str

    @classmethod
    def from_dict(cls, d: dict):
        """Create from config dict. All fields required."""
        return cls(
            dt=d["dt"],
            dt_max=d["dt_max"],
            dt_growth=d["dt_growth"],
            dt_growth_threshold=d["dt_growth_threshold"],
            t_final=d["t_final"],
            max_iter=d["max_iter"],
            steady_tol=d["steady_tol"],
            picard_max=d["picard_max"],
            picard_tol=d["picard_tol"],
            under_relax_k_omega=d["under_relax_k_omega"],
            under_relax_nu_t=d["under_relax_nu_t"],
            log_interval=d["log_interval"],
            snapshot_interval=d["snapshot_interval"],
            out_dir=d["out_dir"],
        )


# =============================================================================
# Mesh utilities
# =============================================================================


def create_channel_mesh(geom: ChannelGeom, Re_tau: float = None):
    """
    Create channel mesh with optional wall refinement.

    If geom.growth_rate > 1.0, creates a wall-refined mesh using geometric
    stretching. The first cell height is set by geom.y_first.

    For low-Re wall treatment, need y+ < 2.5 at first cell center:
        y_first = y+_target / Re_tau  (in nondim units where delta=1)

    Args:
        geom: Channel geometry parameters
        Re_tau: Friction Reynolds number (for y+ reporting)
    """
    comm = MPI.COMM_WORLD

    if geom.growth_rate > 1.0 and geom.y_first > 0:
        # Wall-refined mesh with geometric stretching
        Ny = geom.Ny
        Ly = geom.Ly
        H = Ly / 2.0  # Half-height

        # Generate stretched distribution for lower half [0, H]
        y_lower = _generate_stretched_coords(geom.y_first, H, Ny // 2, geom.growth_rate)

        # Mirror for upper half [H, Ly]
        y_upper = Ly - y_lower[::-1]

        # Combine (remove duplicate at center)
        y_coords = np.concatenate([y_lower, y_upper[1:]])

        # Uniform x distribution
        x_coords = np.linspace(0, geom.Lx, geom.Nx + 1)

        # Report y+ at first cell
        if comm.rank == 0 and Re_tau is not None:
            y_plus_first = geom.y_first * Re_tau
            print(f"Wall-refined mesh: y_first = {geom.y_first:.6f}, y+ = {y_plus_first:.2f}")
            if y_plus_first > 2.5:
                print("  WARNING: y+ > 2.5, low-Re wall BC may be inaccurate")

        # Create mesh from coordinates
        domain = _create_mesh_from_coords(x_coords, y_coords, geom.mesh_type)
    else:
        # Uniform mesh
        if geom.mesh_type == "triangle":
            domain = mesh.create_rectangle(
                comm,
                [[0.0, 0.0], [geom.Lx, geom.Ly]],
                [geom.Nx, geom.Ny],
                cell_type=CellType.triangle,
            )
        else:
            domain = mesh.create_rectangle(
                comm,
                [[0.0, 0.0], [geom.Lx, geom.Ly]],
                [geom.Nx, geom.Ny],
                cell_type=CellType.quadrilateral,
            )

        if comm.rank == 0 and Re_tau is not None:
            dy = geom.Ly / geom.Ny
            y_plus_first = (dy / 2) * Re_tau
            print(f"Uniform mesh: dy = {dy:.6f}, y+ ≈ {y_plus_first:.1f} at first cell center")
            if y_plus_first > 2.5:
                print("  WARNING: y+ > 2.5, consider using growth_rate > 1 for wall refinement")

    return domain


def _generate_stretched_coords(y_first: float, H: float, N: int, growth: float) -> np.ndarray:
    """Generate geometrically stretched y-coordinates from wall to midplane."""
    if growth == 1.0:
        return np.linspace(0, H, N + 1)

    # Compute actual y_first to exactly fill H with N cells
    sum_factor = (growth**N - 1) / (growth - 1)
    y_first_actual = H / sum_factor

    # Generate coordinates
    y = np.zeros(N + 1)
    y[0] = 0.0
    dy = y_first_actual
    for i in range(1, N + 1):
        y[i] = y[i - 1] + dy
        dy *= growth

    y[-1] = H  # Ensure exact endpoint
    return y


def _create_mesh_from_coords(x_coords: np.ndarray, y_coords: np.ndarray, mesh_type: str):
    """Create mesh from x and y coordinate arrays."""
    import basix
    from dolfinx.mesh import CellType, create_mesh

    Nx = len(x_coords) - 1
    Ny = len(y_coords) - 1

    # Create vertices
    vertices = []
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            vertices.append([x, y])
    vertices = np.array(vertices, dtype=np.float64)

    # Create cells
    cells = []
    for j in range(Ny):
        for i in range(Nx):
            v0 = j * (Nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (Nx + 1)
            v3 = v2 + 1
            if mesh_type == "triangle":
                cells.append([v0, v1, v2])
                cells.append([v1, v3, v2])
            else:
                cells.append([v0, v1, v3, v2])
    cells = np.array(cells, dtype=np.int64)

    if mesh_type == "triangle":
        cell_type = CellType.triangle
    else:
        cell_type = CellType.quadrilateral

    domain = create_mesh(
        MPI.COMM_WORLD,
        cells,
        vertices,
        basix.ufl.element("Lagrange", cell_type.name, 1, shape=(2,)),
    )
    return domain


def mark_boundaries(domain, Lx: float, Ly: float):
    """Mark channel boundaries (bottom, top, left, right facets)."""
    fdim = domain.topology.dim - 1
    tol = 1e-10

    def bottom(x):
        return np.isclose(x[1], 0.0, atol=tol)

    def top(x):
        return np.isclose(x[1], Ly, atol=tol)

    def left(x):
        return np.isclose(x[0], 0.0, atol=tol)

    def right(x):
        return np.isclose(x[0], Lx, atol=tol)

    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)
    left_facets = mesh.locate_entities_boundary(domain, fdim, left)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right)
    return bottom_facets, top_facets, left_facets, right_facets


# =============================================================================
# Initial conditions
# =============================================================================


def initial_velocity_channel(x, u_bulk: float, Ly: float):
    """Parabolic initial velocity."""
    eta = 2.0 * x[1] / Ly - 1.0
    u_profile = 1.5 * u_bulk * (1.0 - eta**2)
    return np.vstack([u_profile, np.zeros(x.shape[1], dtype=PETSc.ScalarType)])


def initial_k_channel(x, u_bulk: float, intensity: float = 0.05):
    """Initial TKE from turbulence intensity."""
    k_val = 1.5 * (intensity * u_bulk) ** 2
    return np.full(x.shape[1], max(k_val, 1e-8), dtype=PETSc.ScalarType)


def initial_omega_channel(x, u_bulk: float, H: float, nu: float):
    """Initial ω profile blended from wall asymptotic to bulk value."""
    k_val = max(1.5 * (0.05 * u_bulk) ** 2, 1e-8)
    l_mix = 0.07 * H
    omega_bulk = np.sqrt(k_val) / l_mix

    y = x[1]
    Ly = 2 * H
    y_wall = np.minimum(y, Ly - y)
    y_wall = np.maximum(y_wall, 1e-10)

    omega_wall = 6.0 * nu / (BETA * y_wall**2)
    omega_wall = np.minimum(omega_wall, 1e8)

    blend = np.tanh(y_wall / (0.1 * H)) ** 2
    omega = (1 - blend) * omega_wall + blend * omega_bulk

    return np.maximum(omega, 1e-6).astype(PETSc.ScalarType)


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
    Solve RANS k-ω channel flow with pseudo-transient continuation.

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
    H = Ly / 2.0

    # Nondimensional mode
    Re_tau = nondim.Re_tau
    nu = 1.0 / Re_tau
    rho = 1.0
    use_body_force = nondim.use_body_force
    u_bulk_init = 15.0  # Conservative initial

    if comm.rank == 0:
        print(f"NONDIMENSIONAL MODE: Re_τ = {Re_tau}")
        print(f"  ν* = 1/Re_τ = {nu:.6f}")
        print(f"  Body force: f_x = {1.0 if use_body_force else 0.0}")

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
    bottom_facets, top_facets, left_facets, right_facets = mark_boundaries(domain, geom.Lx, Ly)
    wall_facets_tb = np.concatenate([bottom_facets, top_facets])

    wall_dofs_V0 = locate_dofs_topological(V0, fdim, wall_facets_tb)
    wall_dofs_S0 = locate_dofs_topological(S0, fdim, wall_facets_tb)

    u_noslip = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    bc_walls_u0 = dirichletbc(u_noslip, wall_dofs_V0, V0)

    y_first = geom.y_first
    omega_wall_val = 6.0 * nu / (BETA * y_first**2)
    bc_k_wall0 = dirichletbc(PETSc.ScalarType(0.0), wall_dofs_S0, S0)

    use_periodic = use_body_force
    mpc_V = None
    mpc_Q = None
    mpc_S = None

    if use_periodic:
        if not HAVE_MPC:
            raise RuntimeError(
                "Periodic BCs require dolfinx_mpc. Install: conda install -c conda-forge dolfinx_mpc"
            )
        tol = 250 * np.finfo(domain.geometry.x.dtype).resolution

        def periodic_boundary(x):
            return np.isclose(x[0], geom.Lx, atol=tol)

        def periodic_relation(x):
            y = x.copy()
            y[0] -= geom.Lx
            return y

        mpc_V = MultiPointConstraint(V0)
        mpc_V.create_periodic_constraint_geometrical(V0, periodic_boundary, periodic_relation, [bc_walls_u0])
        mpc_V.finalize()

        mpc_Q = MultiPointConstraint(Q0)
        mpc_Q.create_periodic_constraint_geometrical(Q0, periodic_boundary, periodic_relation, [])
        mpc_Q.finalize()

        mpc_S = MultiPointConstraint(S0)
        mpc_S.create_periodic_constraint_geometrical(S0, periodic_boundary, periodic_relation, [bc_k_wall0])
        mpc_S.finalize()

        V = mpc_V.function_space
        Q = mpc_Q.function_space
        S = mpc_S.function_space
    else:
        V, Q, S = V0, Q0, S0

    if comm.rank == 0:
        print(f"Velocity DOFs: {V.dofmap.index_map.size_global}")
        print(f"Pressure DOFs: {Q.dofmap.index_map.size_global}")
        print(f"Turbulence DOFs: {S.dofmap.index_map.size_global}")

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

    S_mag_ = Function(S, name="S_magnitude") if C_lim > 0 else None

    # Constants
    dt_c = Constant(domain, PETSc.ScalarType(dt))
    nu_c = Constant(domain, PETSc.ScalarType(nu))
    rho_c = Constant(domain, PETSc.ScalarType(rho))

    beta_star_c = Constant(domain, PETSc.ScalarType(turb.beta_star))
    beta_c = Constant(domain, PETSc.ScalarType(BETA))
    gamma_c = Constant(domain, PETSc.ScalarType(GAMMA))
    sigma_k_c = Constant(domain, PETSc.ScalarType(SIGMA_K))
    sigma_w_c = Constant(domain, PETSc.ScalarType(SIGMA_W))

    if use_body_force:
        f = Constant(domain, PETSc.ScalarType((1.0, 0.0)))
        if comm.rank == 0:
            print("Using body force f_x = 1.0 (periodic channel)")
    else:
        f = Constant(domain, PETSc.ScalarType((0.0, 0.0)))

    # Initial conditions
    u_n.interpolate(lambda x: initial_velocity_channel(x, u_bulk_init, Ly))
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
    if use_periodic:
        bcu = [bc_walls_u]
        bcp = []
    else:
        bcu = [bc_walls_u]
        outlet_dofs_Q = locate_dofs_topological(Q, fdim, right_facets)
        bc_pressure = dirichletbc(PETSc.ScalarType(0.0), outlet_dofs_Q, Q)
        bcp = [bc_pressure]

    bc_k_wall = dirichletbc(PETSc.ScalarType(0.0), wall_dofs_S_tb, S)
    bck = [bc_k_wall]

    bc_w_wall = dirichletbc(PETSc.ScalarType(omega_wall_val), wall_dofs_S_tb, S)
    bcw = [bc_w_wall]

    if comm.rank == 0:
        print(f"Wall ω BC: ω_wall = {omega_wall_val:.2e} (y_first = {y_first})")

    # Turbulence forms
    k_safe = k_prev
    omega_safe = omega_prev

    D_k = nu_c + sigma_k_c * nu_t_
    D_w = nu_c + sigma_w_c * nu_t_

    S_tensor = sym(grad(u_n))
    S_sq = 2.0 * inner(S_tensor, S_tensor)
    P_k = nu_t_ * S_sq

    S_mag_expr = None
    if C_lim > 0 and S_mag_ is not None:
        S_mag_ufl = sqrt(S_sq + 1e-16)
        S_mag_expr = Expression(S_mag_ufl, S.element.interpolation_points())

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

    # ω-equation
    P_omega = gamma_c * S_sq

    F_w = (
        (w_trial - omega_n) / dt_c * phi_w * dx
        + dot(u_n, grad(w_trial)) * phi_w * dx
        + D_w * inner(grad(w_trial), grad(phi_w)) * dx
        + beta_c * omega_safe * w_trial * phi_w * dx
        - P_omega * phi_w * dx
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

    # Solvers
    solver_k = PETSc.KSP().create(comm)
    solver_k.setOperators(A_k)
    solver_k.setType(PETSc.KSP.Type.BCGS)
    solver_k.getPC().setType(PETSc.PC.Type.ILU)
    solver_k.setTolerances(rtol=1e-8)

    solver_w = PETSc.KSP().create(comm)
    solver_w.setOperators(A_w)
    solver_w.setType(PETSc.KSP.Type.BCGS)
    solver_w.getPC().setType(PETSc.PC.Type.ILU)
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
    solver3.getPC().setType(PETSc.PC.Type.SOR)
    solver3.setTolerances(rtol=1e-8)

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

    # VTX output
    vtx_vel = None
    vtx_turb = None
    if solve.snapshot_interval > 0:
        vtx_vel = VTXWriter(comm, results_dir / "snps" / "velocity.bp", [u_], engine="BP4")
        vtx_vel.write(0.0)
        vtx_turb = VTXWriter(comm, results_dir / "snps" / "turbulence.bp", [p_, k_, omega_, nu_t_], engine="BP4")
        vtx_turb.write(0.0)

    if comm.rank == 0:
        print(f"\nSolving RANS k-ω channel flow")
        print(f"dt={dt}, t_final={solve.t_final}, picard_max={solve.picard_max}")
        if solve.snapshot_interval > 0:
            print(f"Saving snapshots every {solve.snapshot_interval} steps to snps/")
        print()

    t = 0.0
    step = 0
    current_dt = dt
    residual_prev = 1.0

    table = None
    hist = None
    if comm.rank == 0 and solve.log_interval > 0:
        table = StepTablePrinter([
            ("iter", 6),
            ("dt", 9),
            ("res", 9),
            ("|u|max", 9),
            ("k[min,max]", 20),
            ("ω[min,max]", 20),
            ("nu_t/nu", 12),
        ])
        hist = HistoryWriterCSV(
            results_dir / "history.csv",
            ["iter", "dt", "residual", "umax", "k_min", "k_max", "omega_min", "omega_max", "nu_t_max_ratio"],
            enabled=True,
        )

    # Main iteration loop
    while step < solve.max_iter:
        dt_c.value = current_dt
        t += current_dt
        step += 1

        k_prev.x.array[:] = k_n.x.array
        omega_prev.x.array[:] = omega_n.x.array
        nu_t_old.x.array[:] = nu_t_.x.array

        for picard_iter in range(solve.picard_max):
            # k-equation
            assemble_matrix_maybe_mpc(A_k, a_k, bck, mpc_S)
            assemble_vector_maybe_mpc(b_k, L_k, a_k, bck, mpc_S)

            solver_k.solve(b_k, k_.x.petsc_vec)
            k_.x.scatter_forward()
            backsubstitute(mpc_S, k_)

            k_.x.array[:] = np.clip(k_.x.array, k_min, k_max_limit)
            ur_kw = solve.under_relax_k_omega
            k_.x.array[:] = ur_kw * k_.x.array + (1.0 - ur_kw) * k_prev.x.array
            k_.x.array[:] = np.clip(k_.x.array, k_min, k_max_limit)
            k_.x.scatter_forward()
            backsubstitute(mpc_S, k_)

            # ω-equation
            assemble_matrix_maybe_mpc(A_w, a_w, bcw, mpc_S)
            assemble_vector_maybe_mpc(b_w, L_w, a_w, bcw, mpc_S)

            solver_w.solve(b_w, omega_.x.petsc_vec)
            omega_.x.scatter_forward()
            backsubstitute(mpc_S, omega_)

            omega_.x.array[:] = np.maximum(omega_.x.array, omega_min)
            omega_.x.array[:] = ur_kw * omega_.x.array + (1.0 - ur_kw) * omega_prev.x.array
            omega_.x.array[:] = np.maximum(omega_.x.array, omega_min)
            omega_.x.scatter_forward()
            backsubstitute(mpc_S, omega_)

            # Update nu_t with optional Durbin limiter
            nu_t_raw = k_.x.array / (omega_.x.array + omega_min)
            if C_lim > 0 and S_mag_expr is not None:
                S_mag_.interpolate(S_mag_expr)
                S_mag_arr = S_mag_.x.array
                SQRT6 = np.sqrt(6.0)
                S_mag_safe = np.maximum(S_mag_arr, 1e-10)
                nu_t_durbin = C_lim * k_.x.array / (SQRT6 * S_mag_safe)
                nu_t_raw = np.minimum(nu_t_raw, nu_t_durbin)
            nu_t_raw = np.clip(nu_t_raw, 0, turb.nu_t_max_factor * nu)
            ur_nu_t = solve.under_relax_nu_t
            nu_t_.x.array[:] = ur_nu_t * nu_t_raw + (1.0 - ur_nu_t) * nu_t_old.x.array
            nu_t_.x.scatter_forward()
            backsubstitute(mpc_S, nu_t_)

            # IPCS steps
            assemble_matrix_maybe_mpc(A1, a1, bcu, mpc_V)
            assemble_vector_maybe_mpc(b1, L1, a1, bcu, mpc_V)

            solver1.solve(b1, u_s.x.petsc_vec)
            u_s.x.scatter_forward()
            backsubstitute(mpc_V, u_s)

            assemble_vector_maybe_mpc(b2, L2, a2, bcp, mpc_Q)
            if pressure_nullspace is not None:
                pressure_nullspace.remove(b2)

            solver2.solve(b2, phi.x.petsc_vec)
            phi.x.scatter_forward()
            backsubstitute(mpc_Q, phi)

            p_.x.petsc_vec.axpy(1.0, phi.x.petsc_vec)
            p_.x.scatter_forward()
            backsubstitute(mpc_Q, p_)

            assemble_vector_no_bc(b3, L3, mpc_V)

            solver3.solve(b3, u_.x.petsc_vec)
            u_.x.scatter_forward()
            backsubstitute(mpc_V, u_)

            # Picard check
            dk = k_.x.array - k_prev.x.array
            dk_norm = float(np.sqrt(comm.allreduce(np.dot(dk, dk), op=MPI.SUM)))
            k_norm = float(np.sqrt(comm.allreduce(np.dot(k_.x.array, k_.x.array), op=MPI.SUM)))
            res_k_val = dk_norm / max(k_norm, 1e-10)

            if res_k_val < solve.picard_tol:
                break

            k_prev.x.array[:] = k_.x.array
            omega_prev.x.array[:] = omega_.x.array
            nu_t_old.x.array[:] = nu_t_.x.array

        # Update for next step
        u_n1.x.array[:] = u_n.x.array
        u_n.x.array[:] = u_.x.array
        k_n.x.array[:] = k_.x.array
        omega_n.x.array[:] = omega_.x.array

        # Residual
        du_local = u_.x.array - u_n1.x.array
        diff2 = float(comm.allreduce(np.dot(du_local, du_local), op=MPI.SUM))
        norm2 = float(comm.allreduce(np.dot(u_.x.array, u_.x.array), op=MPI.SUM))
        residual = float(np.sqrt(diff2) / max(np.sqrt(norm2), 1e-10))

        # Log
        if comm.rank == 0 and table is not None and step % solve.log_interval == 0:
            ud = diagnostics_vector(u_)
            kd = diagnostics_scalar(k_)
            wd = diagnostics_scalar(omega_)
            nu_t_ratio = float(np.max(nu_t_.x.array)) / nu

            table.row([
                f"{step:6d}",
                f"{current_dt:9.2e}",
                f"{residual:9.1e}",
                f"{float(ud['umax']):9.3f}",
                fmt_pair_sci(float(kd["min"]), float(kd["max"]), prec=1),
                fmt_pair_sci(float(wd["min"]), float(wd["max"]), prec=1),
                f"{nu_t_ratio:12.1f}",
            ])
            hist.write({
                "iter": step,
                "dt": current_dt,
                "residual": residual,
                "umax": float(ud["umax"]),
                "k_min": float(kd["min"]),
                "k_max": float(kd["max"]),
                "omega_min": float(wd["min"]),
                "omega_max": float(wd["max"]),
                "nu_t_max_ratio": nu_t_ratio,
            })

        # Save snapshot
        if vtx_vel is not None and step % solve.snapshot_interval == 0:
            vtx_vel.write(t)
            vtx_turb.write(t)

        # Convergence check
        if residual < solve.steady_tol:
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
