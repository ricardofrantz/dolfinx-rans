"""
Geometry, mesh generation, boundary marking, and initial conditions for dolfinx-rans.

Contains:
- Channel mesh creation with wall refinement (geometric/tanh stretching)
- Boundary marking for channel flow
- Initial condition profiles (velocity, k, omega)
- Wall distance computation (channel-specific and Eikonal PDE)
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh
from dolfinx.mesh import CellType

from dolfinx_rans.config import BETA_0, BFSGeom, BoundaryInfo, ChannelGeom


# =============================================================================
# Mesh utilities
# =============================================================================


def create_channel_mesh(geom: ChannelGeom, Re_tau: float = None):
    """
    Create channel mesh with optional wall refinement.

    If geom.use_symmetry=True (default), creates half-channel [0, delta] with:
      - Bottom (y=0): wall (no-slip)
      - Top (y=delta): symmetry (free-slip, du/dy=0)

    If geom.use_symmetry=False, creates full channel [0, 2*delta] with walls on both sides.

    Args:
        geom: Channel geometry parameters
        Re_tau: Friction Reynolds number (for y+ reporting)
    """
    comm = MPI.COMM_WORLD
    if geom.y_first_tol_rel < 0:
        raise ValueError(f"geom.y_first_tol_rel must be >= 0, got {geom.y_first_tol_rel}")
    stretching = geom.stretching.lower()
    if stretching not in {"geometric", "tanh"}:
        raise ValueError(
            f"Unknown geom.stretching='{geom.stretching}'. Expected 'geometric' or 'tanh'."
        )

    use_stretched = geom.y_first > 0 and (stretching == "tanh" or geom.growth_rate > 1.0)
    if use_stretched:
        # Wall-refined mesh
        Ny = geom.Ny
        Ly = geom.Ly

        if geom.use_symmetry:
            # Half-channel: only [0, Ly] with wall at bottom, symmetry at top
            y_coords = _generate_stretched_coords(
                y_first=geom.y_first,
                H=Ly,
                N=Ny,
                growth=geom.growth_rate,
                stretching=stretching,
            )
            if comm.rank == 0:
                print(f"Half-channel (symmetry at y={Ly:.2f})")
        else:
            # Full channel: [0, Ly] with walls at both ends
            H = Ly / 2.0  # Half-height
            y_lower = _generate_stretched_coords(
                y_first=geom.y_first,
                H=H,
                N=Ny // 2,
                growth=geom.growth_rate,
                stretching=stretching,
            )
            y_upper = Ly - y_lower[::-1]
            y_coords = np.concatenate([y_lower, y_upper[1:]])
            if comm.rank == 0:
                print(f"Full channel (walls at y=0 and y={Ly:.2f})")

        # Report first off-wall spacing actually used by the stretched mesh
        y_first_actual = float(y_coords[1] - y_coords[0])
        rel_err = abs(y_first_actual - geom.y_first) / max(abs(geom.y_first), 1e-16)
        if rel_err > geom.y_first_tol_rel:
            raise ValueError(
                "Inconsistent wall spacing settings: "
                f"requested y_first={geom.y_first:.6e}, implied by (Ly,Ny,growth)={y_first_actual:.6e}, "
                f"relative error={100.0 * rel_err:.1f}% exceeds tolerance "
                f"{100.0 * geom.y_first_tol_rel:.1f}%. "
                "Adjust Ny/growth_rate or set y_first to match the implied spacing."
            )

        # Report y+ at first off-wall point
        if comm.rank == 0 and Re_tau is not None:
            y_plus_first = y_first_actual * Re_tau
            print(
                "Wall-refined mesh: "
                f"mode = {stretching}, "
                f"y_first(requested) = {geom.y_first:.6f}, "
                f"y_first(actual) = {y_first_actual:.6f}, "
                f"y+_actual = {y_plus_first:.2f}"
            )
            if y_plus_first > 2.5:
                print("  WARNING: y+ > 2.5, low-Re wall BC may be inaccurate")

        # Create mesh with stretched coordinates
        domain = _create_stretched_mesh(geom.Lx, Ly, geom.Nx, Ny, y_coords, geom.mesh_type)
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
            print(f"Uniform mesh: dy = {dy:.6f}, y+ ~ {y_plus_first:.1f} at first cell center")
            if y_plus_first > 2.5:
                print("  WARNING: y+ > 2.5, consider using growth_rate > 1 for wall refinement")

    return domain


def _generate_stretched_coords(
    y_first: float,
    H: float,
    N: int,
    growth: float,
    stretching: str,
) -> np.ndarray:
    """Generate stretched y-coordinates with the requested stretching mode."""
    if stretching == "geometric":
        return _generate_stretched_coords_geometric(H, N, growth)
    if stretching == "tanh":
        return _generate_stretched_coords_tanh(y_first, H, N)
    raise ValueError(f"Unsupported stretching mode: {stretching}")


def _generate_stretched_coords_geometric(H: float, N: int, growth: float) -> np.ndarray:
    """
    Generate geometrically stretched y-coordinates from wall to midplane.

    With fixed (H, N, growth), the first spacing is determined by the
    geometric-series closure.
    """
    if growth == 1.0:
        return np.linspace(0, H, N + 1)

    # Compute first cell size to exactly fill H with N cells at given growth rate
    # Geometric series sum: H = dy1 * (growth^N - 1) / (growth - 1)
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


def _generate_stretched_coords_tanh(y_first: float, H: float, N: int) -> np.ndarray:
    """
    Generate wall-refined coordinates using:
        y(eta) = H * [1 - tanh(beta*(1-eta))/tanh(beta)], eta in [0,1]
    and solve beta so that the first spacing matches y_first.
    """
    if N <= 0:
        raise ValueError(f"N must be > 0 for tanh stretching, got {N}")
    if y_first <= 0 or y_first >= H:
        raise ValueError(f"tanh stretching requires 0 < y_first < H, got y_first={y_first}, H={H}")

    dy_uniform = H / N
    if y_first >= dy_uniform:
        return np.linspace(0, H, N + 1)

    eta = np.linspace(0.0, 1.0, N + 1)

    def dy1(beta: float) -> float:
        return H * (1.0 - np.tanh(beta * (1.0 - 1.0 / N)) / np.tanh(beta))

    beta_lo = 1e-12
    beta_hi = 1.0
    while dy1(beta_hi) > y_first:
        beta_hi *= 2.0
        if beta_hi > 1e6:
            raise ValueError(
                "Could not match requested y_first with tanh stretching. "
                f"Requested y_first={y_first:.6e}, H={H:.6e}, N={N}."
            )

    for _ in range(80):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        if dy1(beta_mid) > y_first:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid

    beta = 0.5 * (beta_lo + beta_hi)
    y = H * (1.0 - np.tanh(beta * (1.0 - eta)) / np.tanh(beta))
    y[0] = 0.0
    y[-1] = H
    return y


def _create_stretched_mesh(Lx: float, Ly: float, Nx: int, Ny: int, y_coords: np.ndarray, mesh_type: str):
    """Create mesh with stretched y-coordinates by deforming a uniform mesh."""
    # Create uniform mesh first
    if mesh_type == "triangle":
        cell_type = CellType.triangle
    else:
        cell_type = CellType.quadrilateral

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [Lx, Ly]],
        [Nx, Ny],
        cell_type=cell_type,
    )

    # Deform mesh: map uniform y to stretched y
    x = domain.geometry.x
    y_uniform = np.linspace(0, Ly, Ny + 1)

    for i in range(x.shape[0]):
        y_old = x[i, 1]
        # Find which interval y_old falls into and interpolate
        idx = np.searchsorted(y_uniform, y_old, side="right") - 1
        idx = max(0, min(idx, Ny - 1))
        # Linear interpolation within the cell
        dy_uniform = y_uniform[idx + 1] - y_uniform[idx]
        t = (y_old - y_uniform[idx]) / dy_uniform if abs(dy_uniform) > 1e-14 else 0.0
        x[i, 1] = y_coords[idx] + t * (y_coords[idx + 1] - y_coords[idx])

    return domain


# =============================================================================
# Boundary marking
# =============================================================================


def mark_channel_boundaries(domain, Lx: float, Ly: float):
    """
    Mark channel boundaries and return BoundaryInfo.

    Returns both the raw facet arrays (bottom, top, left, right) and a
    BoundaryInfo object with semantic names (wall, inlet, outlet, symmetry).
    """
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


def infer_first_offwall_spacing(domain, Ly: float, use_symmetry: bool, tol: float = 1e-12) -> float:
    """
    Infer first off-wall spacing directly from mesh coordinates.

    For half-channel this is min(y > 0).
    For full channel this is min(min(y, Ly-y) > 0).
    """
    y = domain.geometry.x[:, 1]
    if use_symmetry:
        wall_distance = y
    else:
        wall_distance = np.minimum(y, Ly - y)

    positive = wall_distance[wall_distance > tol]
    local_min = float(np.min(positive)) if positive.size > 0 else np.inf
    y_first = float(domain.comm.allreduce(local_min, op=MPI.MIN))
    if not np.isfinite(y_first):
        raise RuntimeError("Could not infer first off-wall spacing from mesh geometry.")
    return y_first


# =============================================================================
# Initial conditions
# =============================================================================


def initial_velocity_channel(x, u_bulk: float, Ly: float, use_symmetry: bool = True):
    """Parabolic initial velocity profile.

    For half-channel (use_symmetry=True): y in [0, delta], u = u_max * (1 - (1-y/delta)^2)
    For full channel: y in [0, 2*delta], u = u_max * (1 - (y/delta - 1)^2)
    """
    if use_symmetry:
        # Half-channel: y=0 is wall, y=Ly is centerline
        # u = 0 at y=0, u = u_max at y=Ly
        eta = x[1] / Ly  # eta in [0, 1]
        u_profile = 1.5 * u_bulk * (2*eta - eta**2)  # parabolic: 0 at wall, max at center
    else:
        # Full channel: y=0 and y=Ly are walls, centerline at y=Ly/2
        eta = 2.0 * x[1] / Ly - 1.0  # eta in [-1, 1]
        u_profile = 1.5 * u_bulk * (1.0 - eta**2)
    return np.vstack([u_profile, np.zeros(x.shape[1], dtype=PETSc.ScalarType)])


def initial_k_channel(x, u_bulk: float, intensity: float = 0.05):
    """Initial TKE from turbulence intensity."""
    k_val = 1.5 * (intensity * u_bulk) ** 2
    return np.full(x.shape[1], max(k_val, 1e-8), dtype=PETSc.ScalarType)


def initial_omega_channel(x, u_bulk: float, H: float, nu: float):
    """Initial omega profile blended from wall asymptotic to bulk value."""
    k_val = max(1.5 * (0.05 * u_bulk) ** 2, 1e-8)
    l_mix = 0.07 * H
    omega_bulk = np.sqrt(k_val) / l_mix

    y = x[1]
    Ly = 2 * H
    y_wall = np.minimum(y, Ly - y)
    y_wall = np.maximum(y_wall, 1e-10)

    omega_wall = 6.0 * nu / (BETA_0 * y_wall**2)
    omega_wall = np.minimum(omega_wall, 1e8)

    blend = np.tanh(y_wall / (0.1 * H)) ** 2
    omega = (1 - blend) * omega_wall + blend * omega_bulk

    return np.maximum(omega, 1e-6).astype(PETSc.ScalarType)


# =============================================================================
# Wall distance computation
# =============================================================================


def compute_wall_distance_channel(S, use_symmetry: bool = True):
    """
    Compute wall distance for channel flow geometry.

    For channel flow:
    - Half-channel (use_symmetry=True): wall at y=0, symmetry at y=Ly
      -> wall distance = y
    - Full channel (use_symmetry=False): walls at y=0 and y=Ly
      -> wall distance = min(y, Ly-y)

    Args:
        S: Scalar function space
        use_symmetry: True for half-channel, False for full channel

    Returns:
        y_wall: Function containing wall distance at each DOF
    """
    from dolfinx.fem import Function

    domain = S.mesh
    y_wall = Function(S, name="wall_distance")

    # Get DOF coordinates
    x_dofs = S.tabulate_dof_coordinates()
    y_coords = x_dofs[:, 1]

    if use_symmetry:
        # Half-channel: wall at y=0, wall distance = y
        y_wall.x.array[:] = y_coords
    else:
        # Full channel: walls at y=0 and y=Ly
        Ly = np.max(y_coords)
        y_wall.x.array[:] = np.minimum(y_coords, Ly - y_coords)

    # Ensure positive (minimum distance = small epsilon for numerical stability)
    y_wall.x.array[:] = np.maximum(y_wall.x.array, 1e-10)
    y_wall.x.scatter_forward()

    return y_wall


def compute_wall_distance_eikonal(S, wall_facets, relax: float = 0.01):
    """
    Solve Eikonal equation |grad(d)| = 1 with d = 0 on walls.

    Works for any geometry. Two-step approach:
      1. Linear Laplace warm-up: -nabla^2 d = 1,  d|_wall = 0
      2. Nonlinear Eikonal: sqrt(grad(d).grad(d)) * v + relax * grad(d).grad(v) = v
         solved with Newton iterations, initialized from step 1.

    The relaxation term (relax * inner(grad(d), grad(v)) * dx) stabilizes the
    nonlinear solve in regions where |grad(d)| → 0 (e.g., at symmetry planes
    or domain corners far from walls).

    Reference: joove123/k-epsilon; Tucker, P.G. (2003), Applied Mathematical
    Modelling, 27(3):189-198.

    Args:
        S: Scalar function space (Lagrange 1)
        wall_facets: Array of facet indices on wall boundaries
        relax: Diffusive regularization coefficient (default 0.01)

    Returns:
        d: Function containing wall distance at each DOF
    """
    import ufl
    from dolfinx.fem import (
        Constant,
        Function,
        dirichletbc,
        form,
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

    domain = S.mesh
    comm = domain.comm
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    # Wall Dirichlet BC: d = 0
    wall_dofs = locate_dofs_topological(S, fdim, wall_facets)
    bc_wall = dirichletbc(PETSc.ScalarType(0.0), wall_dofs, S)
    bcs = [bc_wall]

    # ─── Step 1: Linear Laplace warm-up ───
    # Solve -nabla^2 d = 1 with d = 0 on walls.
    # This gives a smooth, positive approximation to wall distance.
    d = Function(S, name="wall_distance")
    d_trial = ufl.TrialFunction(S)
    v = ufl.TestFunction(S)

    a_lap = form(ufl.inner(ufl.grad(d_trial), ufl.grad(v)) * ufl.dx)
    L_lap = form(Constant(domain, PETSc.ScalarType(1.0)) * v * ufl.dx)

    A_lap = create_matrix(a_lap)
    assemble_matrix(A_lap, a_lap, bcs=bcs)
    A_lap.assemble()

    b_lap = create_vector(S)
    with b_lap.localForm() as loc:
        loc.set(0.0)
    assemble_vector(b_lap, L_lap)
    apply_lifting(b_lap, [a_lap], [bcs])
    b_lap.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_lap, bcs)

    ksp_lap = PETSc.KSP().create(comm)
    ksp_lap.setOperators(A_lap)
    ksp_lap.setType(PETSc.KSP.Type.CG)
    pc = ksp_lap.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    ksp_lap.setTolerances(rtol=1e-10)
    ksp_lap.solve(b_lap, d.x.petsc_vec)
    d.x.scatter_forward()

    # Ensure non-negative (Laplace solution should be positive, but clip for safety)
    if not np.all(np.isfinite(d.x.array)):
        raise RuntimeError("Eikonal Laplace warm-up produced non-finite values")
    d.x.array[:] = np.maximum(d.x.array, 0.0)
    d.x.scatter_forward()

    ksp_lap.destroy()
    A_lap.destroy()
    b_lap.destroy()

    if comm.rank == 0:
        print(
            f"Eikonal warm-up (Laplace): d_max = {float(comm.allreduce(np.max(d.x.array), op=MPI.MAX)):.4f}",
            flush=True,
        )

    # ─── Step 2: Nonlinear Eikonal solve ───
    # F(d) = sqrt(grad(d).grad(d)) * v + relax * grad(d).grad(v) - v = 0
    # Uses Picard iteration: linearize |grad(d)| around current d.
    #
    # We avoid the full Newton since |grad(d)| has a singularity at grad(d)=0.
    # Instead: fixed-point iteration updating |grad(d)| from previous iterate.
    relax_c = Constant(domain, PETSc.ScalarType(relax))
    one_c = Constant(domain, PETSc.ScalarType(1.0))

    d_old = Function(S)
    d_old.x.array[:] = d.x.array

    d_new = ufl.TrialFunction(S)
    v_test = ufl.TestFunction(S)

    # |grad(d_old)| as linearization point
    grad_d_old_sq = ufl.dot(ufl.grad(d_old), ufl.grad(d_old))
    grad_d_old_mag = ufl.sqrt(grad_d_old_sq + 1e-16)

    # Linearized Eikonal: |grad(d_old)| * d_new/d_old_approx ≈ 1
    # Better form: grad(d_old)/|grad(d_old)| . grad(d_new) + relax * laplacian = 1
    # This is the standard Picard linearization of the Eikonal equation.
    n_hat = ufl.grad(d_old) / grad_d_old_mag  # unit normal direction
    a_eik = form(
        ufl.dot(n_hat, ufl.grad(d_new)) * v_test * ufl.dx
        + relax_c * ufl.inner(ufl.grad(d_new), ufl.grad(v_test)) * ufl.dx
    )
    L_eik = form(one_c * v_test * ufl.dx)

    A_eik = create_matrix(a_eik)
    b_eik = create_vector(S)

    ksp_eik = PETSc.KSP().create(comm)
    ksp_eik.setOperators(A_eik)
    ksp_eik.setType(PETSc.KSP.Type.BCGS)
    pc_eik = ksp_eik.getPC()
    pc_eik.setType(PETSc.PC.Type.HYPRE)
    pc_eik.setHYPREType("boomeramg")
    ksp_eik.setTolerances(rtol=1e-8)

    max_eik_iter = 30
    eik_tol = 1e-4
    for it in range(max_eik_iter):
        A_eik.zeroEntries()
        assemble_matrix(A_eik, a_eik, bcs=bcs)
        A_eik.assemble()

        with b_eik.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b_eik, L_eik)
        apply_lifting(b_eik, [a_eik], [bcs])
        b_eik.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_eik, bcs)

        ksp_eik.solve(b_eik, d.x.petsc_vec)
        d.x.scatter_forward()

        # Ensure non-negative
        d.x.array[:] = np.maximum(d.x.array, 0.0)
        d.x.scatter_forward()

        # Check convergence: relative change in d
        diff = d.x.array - d_old.x.array
        diff_norm = float(np.sqrt(comm.allreduce(np.dot(diff, diff), op=MPI.SUM)))
        d_norm = float(np.sqrt(comm.allreduce(np.dot(d.x.array, d.x.array), op=MPI.SUM)))
        rel_change = diff_norm / max(d_norm, 1e-10)

        d_old.x.array[:] = d.x.array

        if rel_change < eik_tol:
            if comm.rank == 0:
                print(f"Eikonal converged in {it + 1} iterations (rel_change = {rel_change:.2e})", flush=True)
            break
    else:
        if comm.rank == 0:
            print(
                f"Eikonal: max iterations ({max_eik_iter}) reached "
                f"(rel_change = {rel_change:.2e})",
                flush=True,
            )

    ksp_eik.destroy()
    A_eik.destroy()
    b_eik.destroy()

    # Final floor for numerical safety in turbulence models
    d.x.array[:] = np.maximum(d.x.array, 1e-10)
    d.x.scatter_forward()

    return d


# =============================================================================
# Backward-facing step geometry
# =============================================================================


def create_bfs_mesh(geom: BFSGeom):
    """
    Create backward-facing step mesh (L-shaped structured quad mesh).

    Domain layout (flow enters from left)::

        y=H_out ┌──────────────────────────────────────┐ top wall
                │  Upstream        Downstream          │
                │  channel         channel             │
                ├──────────┐                           │
         y=h    │  (wall)  │ step face                 │
                │  SOLID   │                           │
                │          │  recirculation zone       │
                └──────────┴───────────────────────────┘
              x=-L_up     x=0                       x=L_down
                           y=0 (bottom wall downstream)

    Ny_outlet must exceed Ny_inlet so that cells exist below the step.
    The downstream y-grid is split: Ny_step cells in [0, h] and Ny_inlet
    cells in [h, H_outlet]. The upstream block shares the upper portion.

    Args:
        geom: BFS geometry parameters

    Returns:
        domain: DOLFINx mesh
    """
    comm = MPI.COMM_WORLD
    h = geom.step_height
    ER = geom.expansion_ratio
    H_inlet = h / (ER - 1.0)
    H_outlet = H_inlet + h  # = ER * H_inlet

    L_up = geom.upstream_length * h
    L_down = geom.downstream_length * h

    Nx_up = geom.Nx_upstream
    Nx_down = geom.Nx_downstream
    Ny_in = geom.Ny_inlet
    Ny_out = geom.Ny_outlet
    Ny_step = Ny_out - Ny_in

    if Ny_step < 1:
        raise ValueError(
            f"Ny_outlet ({Ny_out}) must exceed Ny_inlet ({Ny_in}) "
            f"to mesh the step region [0, h]"
        )

    # 1D coordinate arrays
    x_up = np.linspace(-L_up, 0.0, Nx_up + 1)

    # Downstream: uniform up to x_stretch, then geometric growth beyond.
    # dx starts at the uniform value and grows by growth_rate per cell.
    # Nx_coarse is determined by how many cells fit before reaching L_down.
    x_stretch = 10.0 * h  # start coarsening after recirculation zone
    if x_stretch < L_down and geom.growth_rate > 1.0:
        dx_fine = L_down / Nx_down  # uniform cell size (baseline)
        Nx_fine = int(round(x_stretch / dx_fine))
        Nx_fine = max(Nx_fine, 1)
        x_fine = np.linspace(0.0, Nx_fine * dx_fine, Nx_fine + 1)
        # Coarse zone: grow from dx_fine, figure out how many cells fill the rest
        L_coarse = L_down - x_fine[-1]
        r = geom.growth_rate
        x_coarse = [x_fine[-1]]
        dx = dx_fine
        while x_coarse[-1] < L_down - 1e-12:
            dx *= r
            x_coarse.append(x_coarse[-1] + dx)
        x_coarse[-1] = L_down  # exact endpoint
        x_coarse = np.array(x_coarse)
        Nx_coarse = len(x_coarse) - 1
        x_down = np.concatenate([x_fine[:-1], x_coarse])
        if comm.rank == 0:
            print(f"  x-mesh: {Nx_fine} uniform (dx={dx_fine:.4f}) + "
                  f"{Nx_coarse} stretched (dx={dx_fine*r:.4f}→{dx*r:.4f})")
    else:
        x_down = np.linspace(0.0, L_down, Nx_down + 1)

    x_all = np.concatenate([x_up[:-1], x_down])  # unique x values

    y_low = np.linspace(0.0, h, Ny_step + 1)
    y_high = np.linspace(h, H_outlet, Ny_in + 1)
    y_all = np.concatenate([y_low[:-1], y_high])  # unique y values

    Nx = len(x_all) - 1  # Nx_up + Nx_down
    Ny = len(y_all) - 1  # Ny_out
    i_step = Nx_up        # index where x=0
    j_step = Ny_step      # index where y=h

    # Build node map: (i,j) -> global node index (-1 for solid step region)
    node_id = -np.ones((Nx + 1, Ny + 1), dtype=np.int64)
    coords = []
    n = 0
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            if i < i_step and j < j_step:
                continue  # inside solid step
            node_id[i, j] = n
            coords.append([x_all[i], y_all[j]])
            n += 1
    coords = np.array(coords, dtype=np.float64)

    # Quad cells (Basix vertex ordering: BL, BR, TL, TR)
    cells = []
    for j in range(Ny):
        for i in range(Nx):
            bl = node_id[i, j]
            br = node_id[i + 1, j]
            tl = node_id[i, j + 1]
            tr = node_id[i + 1, j + 1]
            if min(bl, br, tl, tr) < 0:
                continue
            cells.append([bl, br, tl, tr])
    cells = np.array(cells, dtype=np.int64)

    # Create DOLFINx mesh
    from basix.ufl import element as basix_element

    e = basix_element("Lagrange", "quadrilateral", 1, shape=(2,))
    domain = mesh.create_mesh(comm, cells, e, coords)

    if comm.rank == 0:
        ncells = domain.topology.index_map(domain.topology.dim).size_global
        print(f"BFS mesh: {ncells} quad cells")
        print(f"  H_inlet={H_inlet:.4f}, H_outlet={H_outlet:.4f}, h={h}")
        print(f"  Domain: x in [{-L_up:.1f}, {L_down:.1f}], y in [0, {H_outlet:.4f}]")

    return domain


def mark_bfs_boundaries(domain, geom: BFSGeom):
    """
    Mark BFS boundaries and return BoundaryInfo.

    Boundary tags:
        inlet:  x = -L_up (left face, above step)
        outlet: x = L_down (right face, full height)
        top:    y = H_outlet (top wall, full length)
        bottom_upstream: y = h, x <= 0 (step top surface)
        bottom_downstream: y = 0, x >= 0 (floor after step)
        step_face: x = 0, y in [0, h] (vertical face of step)
    """
    h = geom.step_height
    ER = geom.expansion_ratio
    H_inlet = h / (ER - 1.0)
    H_outlet = H_inlet + h
    L_up = geom.upstream_length * h
    L_down = geom.downstream_length * h

    fdim = domain.topology.dim - 1
    tol = 1e-10

    inlet_f = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], -L_up, atol=tol)
    )
    outlet_f = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], L_down, atol=tol)
    )
    top_f = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], H_outlet, atol=tol)
    )
    bottom_up_f = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: np.isclose(x[1], h, atol=tol) & (x[0] <= tol),
    )
    bottom_down_f = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], 0.0, atol=tol)
    )
    step_face_f = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: np.isclose(x[0], 0.0, atol=tol) & (x[1] <= h + tol),
    )

    wall_f = np.unique(np.concatenate([
        top_f, bottom_up_f, bottom_down_f, step_face_f,
    ]))

    return BoundaryInfo(
        wall_facets=wall_f,
        inlet_facets=inlet_f,
        outlet_facets=outlet_f,
        step_facets=step_face_f,
        top_facets=top_f,
        bottom_facets=np.unique(np.concatenate([bottom_up_f, bottom_down_f])),
    )


def initial_velocity_bfs(x, U_inlet, H_outlet, H_inlet):
    """Parabolic velocity profile over the inlet channel height.

    The inlet occupies y in [H_outlet - H_inlet, H_outlet]. The parabola
    peaks at the channel centre and is zero at both walls. Outside the
    inlet channel (below the step), velocity is zero.
    """
    y_bottom = H_outlet - H_inlet
    eta = (x[1] - y_bottom) / H_inlet
    eta = np.clip(eta, 0.0, 1.0)
    u_profile = 1.5 * U_inlet * 4.0 * eta * (1.0 - eta)
    # Zero velocity below the step (y < y_bottom)
    u_profile = np.where(x[1] < y_bottom, 0.0, u_profile)
    return np.vstack([
        u_profile.astype(PETSc.ScalarType),
        np.zeros(x.shape[1], dtype=PETSc.ScalarType),
    ])


def initial_k_bfs(x, U_inlet, intensity=0.05):
    """Uniform initial TKE for BFS from turbulence intensity."""
    k_val = max(1.5 * (intensity * U_inlet) ** 2, 1e-8)
    return np.full(x.shape[1], k_val, dtype=PETSc.ScalarType)


def initial_omega_bfs(x, U_inlet, H_inlet, nu=0.0, H_outlet=0.0):
    """Wall-blended initial omega for BFS.

    Blends from the wall-asymptotic value omega_wall = 6*nu/(beta_0*y_w^2)
    near walls to a mixing-length bulk estimate in the interior.  This
    avoids the sharp jump between a uniform IC and the Dirichlet omega_wall
    BC that previously caused early-iteration instability.

    When nu=0 (legacy call), falls back to uniform mixing-length estimate.
    """
    k_val = max(1.5 * (0.05 * U_inlet) ** 2, 1e-8)
    l_mix = 0.07 * H_inlet
    omega_bulk = np.sqrt(k_val) / max(l_mix, 1e-10)

    if nu <= 0.0 or H_outlet <= 0.0:
        return np.full(x.shape[1], omega_bulk, dtype=PETSc.ScalarType)

    # Wall distance: min distance to bottom (y=0) or top (y=H_outlet) wall
    y_wall = np.minimum(x[1], H_outlet - x[1])
    y_wall = np.maximum(y_wall, 1e-10)

    omega_wall = 6.0 * nu / (BETA_0 * y_wall**2)
    omega_wall = np.minimum(omega_wall, 1e8)  # Cap near-wall singularity

    # Smooth blend: tanh² transition over ~10% of inlet height
    blend = np.tanh(y_wall / (0.1 * H_inlet)) ** 2
    omega = (1.0 - blend) * omega_wall + blend * omega_bulk

    return np.maximum(omega, 1e-6).astype(PETSc.ScalarType)
