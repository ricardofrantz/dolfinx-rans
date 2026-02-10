"""
Plotting utilities for dolfinx-rans.

Visualizes mesh, initial conditions, and solution fields.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI


def plot_mesh(domain, geom, save_path: Path | None = None):
    """Plot the mesh with wall refinement highlighted.

    MPI-safe: all ranks participate in gathering cell data, only rank 0 plots.
    """
    comm = domain.comm

    # Phase 1: gather mesh cells to rank 0 (ALL ranks)
    coords = domain.geometry.x
    cells = domain.geometry.dofmap
    # Build polygon vertices per cell (local)
    local_polys = [coords[cell, :2].copy() for cell in cells]
    all_polys = comm.gather(local_polys, root=0)

    # Phase 2: plot (rank 0 only)
    if comm.rank != 0:
        return

    polys = [p for rank_polys in all_polys for p in rank_polys]
    ar = geom.Lx / geom.Ly  # aspect ratio
    fig_w = 14
    fig_h = fig_w / ar + 1.2  # match geometry + room for title/labels

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for pts in polys:
        ax.add_patch(plt.Polygon(pts, fill=False, edgecolor="k", linewidth=0.3))
    ax.set_xlim(0, geom.Lx)
    ax.set_ylim(0, geom.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{geom.Nx}×{geom.Ny} {geom.mesh_type}s  (y_first={geom.y_first:.4f}, growth={geom.growth_rate})")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved mesh plot: {save_path}")
    plt.close(fig)


def _extract_field_on_line(func, y_values, x_val, domain, comm=None):
    """Extract scalar field values along a vertical line at x=x_val.

    For evaluating multiple fields at the same points, use
    extract_fields_on_line() which builds the bb_tree only once.
    """
    results = extract_fields_on_line([func], y_values, x_val, domain, comm=comm)
    return results[0]


def extract_fields_on_line(funcs, y_values, x_val, domain, comm=None):
    """
    Extract multiple scalar fields along a vertical line at x=x_val.

    Builds the bounding-box tree and finds colliding cells ONCE,
    then evaluates all functions in a single pass.

    Args:
        funcs: list of DOLFINx Function or sub-function objects
        y_values: 1D array of y-coordinates to sample
        x_val: x-coordinate of the vertical line
        domain: DOLFINx mesh
        comm: MPI communicator. When provided, allreduce(SUM) combines
              results across ranks (each point owned by exactly one rank).
              Without comm, returns local values only (serial behavior).

    Returns:
        list of 1D arrays, one per function
    """
    from dolfinx import geometry

    points = np.zeros((len(y_values), 3))
    points[:, 0] = x_val
    points[:, 1] = y_values

    # Build bb_tree and find cells ONCE for all functions
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    # Evaluate at first colliding cell (owned or ghost — CG elements give the
    # same value regardless). Track which points this rank found so we can
    # divide by count after allreduce to cancel any double-counting at
    # partition boundaries.
    all_values = [np.zeros(len(y_values)) for _ in funcs]
    found = np.zeros(len(y_values))
    for i, point in enumerate(points):
        if len(cells.links(i)) > 0:
            cell = cells.links(i)[0]
            found[i] = 1.0
            for j, func in enumerate(funcs):
                # DOLFINx Function.eval — FE interpolation at point (not Python eval)
                result = func.eval(point, cell)  # noqa: S307
                all_values[j][i] = result[0]

    # MPI reduction with count-based averaging to handle double-counting
    if comm is not None:
        global_found = comm.allreduce(found, op=MPI.SUM)
        mask = global_found > 0
        for j in range(len(funcs)):
            all_values[j] = comm.allreduce(all_values[j], op=MPI.SUM)
            all_values[j][mask] /= global_found[mask]

    return all_values


def gather_scalar_field(func, comm):
    """
    Gather DOF coordinates and values from all ranks to rank 0.

    Excludes ghost DOFs to avoid duplicates. Returns data for 2D contour
    plots via tricontourf.

    Args:
        func: DOLFINx Function on a scalar function space
        comm: MPI communicator

    Returns:
        On rank 0: (x, y, vals) as 1D arrays
        On other ranks: (None, None, None)
    """
    S = func.function_space
    n_local = S.dofmap.index_map.size_local  # exclude ghosts
    dof_coords = S.tabulate_dof_coordinates()[:n_local]
    vals_local = func.x.array[:n_local].copy()

    all_coords = comm.gather(dof_coords, root=0)
    all_vals = comm.gather(vals_local, root=0)

    if comm.rank == 0:
        coords = np.concatenate(all_coords, axis=0)
        vals = np.concatenate(all_vals, axis=0)
        return coords[:, 0], coords[:, 1], vals
    return None, None, None


def write_channel_profile_csv(u, k, omega, nu_t, domain, geom, Re_tau, save_path: Path, n_points: int = 256):
    """
    Export centerline-normalized channel profiles for benchmark comparison.

    Output columns:
      y, y_plus, u_plus, k_plus, omega_plus, nu_t_over_nu
    """
    comm = domain.comm

    # Avoid sampling exactly on boundaries for robust point evaluation.
    eps = min(1e-6, max(1e-10, 0.1 * geom.y_first))
    if 2.0 * eps >= geom.Ly:
        eps = 1e-10

    y_vals = np.linspace(eps, geom.Ly - eps, n_points)
    x_mid = geom.Lx / 2.0

    u_profile, k_profile, omega_profile, nu_t_profile = extract_fields_on_line(
        [u.sub(0), k, omega, nu_t], y_vals, x_mid, domain, comm=comm,
    )

    if comm.rank != 0:
        return

    nu = 1.0 / Re_tau
    y_plus = y_vals * Re_tau
    nu_t_over_nu = nu_t_profile / nu

    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([y_vals, y_plus, u_profile, k_profile, omega_profile, nu_t_over_nu])
    np.savetxt(
        save_path,
        data,
        delimiter=",",
        header="y,y_plus,u_plus,k_plus,omega_plus,nu_t_over_nu",
        comments="",
    )
    print(f"  Saved profile CSV: {save_path}")


def plot_initial_conditions(u, p, k, omega, nu_t, domain, geom, Re_tau, save_path: Path | None = None):
    """Plot initial condition fields.

    MPI-safe: all ranks participate in extraction, only rank 0 plots.
    """
    comm = domain.comm

    # Phase 1: Extract data (ALL ranks)
    x_mid = geom.Lx / 2
    y_vals = np.linspace(0, geom.Ly, 100)

    u_profile, k_profile, omega_profile, nu_t_profile = extract_fields_on_line(
        [u.sub(0), k, omega, nu_t], y_vals, x_mid, domain, comm=comm,
    )

    # Phase 2: Plot (rank 0 only)
    if comm.rank != 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    y_plus = y_vals * Re_tau
    nu = 1.0 / Re_tau

    # u+ profile
    ax = axes[0, 0]
    ax.plot(y_plus, u_profile, "b-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("u⁺ (IC)")
    ax.set_title("Initial velocity profile")
    ax.grid(True, alpha=0.3)

    # k profile
    ax = axes[0, 1]
    ax.plot(y_plus, k_profile, "r-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("k⁺ (IC)")
    ax.set_title("Initial TKE profile")
    ax.grid(True, alpha=0.3)

    # omega profile (log scale)
    ax = axes[0, 2]
    ax.semilogy(y_plus, omega_profile, "g-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("ω⁺ (IC)")
    ax.set_title("Initial ω profile")
    ax.grid(True, alpha=0.3)

    # nu_t profile
    ax = axes[1, 0]
    ax.plot(y_plus, nu_t_profile, "m-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("ν_t⁺ (IC)")
    ax.set_title("Initial eddy viscosity")
    ax.grid(True, alpha=0.3)

    # nu_t / nu ratio
    ax = axes[1, 1]
    ax.plot(y_plus, nu_t_profile / nu, "m-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("ν_t/ν (IC)")
    ax.set_title("Eddy viscosity ratio")
    ax.grid(True, alpha=0.3)

    # Info text
    ax = axes[1, 2]
    ax.axis("off")
    info_text = (
        f"Initial Conditions\n"
        f"─────────────────\n"
        f"Re_τ = {Re_tau}\n"
        f"Domain: {geom.Lx:.2f} × {geom.Ly:.2f}\n"
        f"Mesh: {geom.Nx} × {geom.Ny}\n"
        f"y_first = {geom.y_first:.4f}\n"
        f"y⁺_first = {geom.y_first * Re_tau:.2f}\n"
        f"growth = {geom.growth_rate:.2f}"
    )
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved IC plot: {save_path}")
    plt.close(fig)


def plot_final_fields(u, p, k, omega, nu_t, domain, geom, Re_tau, save_path: Path | None = None):
    """Plot final solution fields as contours and profiles.

    MPI-safe: all ranks participate in extraction, only rank 0 plots.
    """
    comm = domain.comm

    # Phase 1: Extract data (ALL ranks must participate)
    x_mid = geom.Lx / 2
    y_vals = np.linspace(0.001, geom.Ly - 0.001, 100)

    u_profile, k_profile, omega_profile, nu_t_profile = extract_fields_on_line(
        [u.sub(0), k, omega, nu_t], y_vals, x_mid, domain, comm=comm,
    )

    # Gather 2D fields for contour plots
    ux_x, ux_y, ux_vals = gather_scalar_field(u.sub(0).collapse(), comm)
    k_x, k_y, k_vals = gather_scalar_field(k, comm)
    nut_x, nut_y, nut_vals = gather_scalar_field(nu_t, comm)

    # Phase 2: Plot (rank 0 only)
    if comm.rank != 0:
        return

    nu = 1.0 / Re_tau
    ar = geom.Lx / geom.Ly  # channel aspect ratio (~6.28)
    # Contour row height adapts to true geometry; profile rows are fixed
    contour_h = 16 / (3 * ar)  # width per panel / aspect ratio
    fig, axes = plt.subplots(3, 3, figsize=(16, contour_h + 8),
                             gridspec_kw={"height_ratios": [contour_h, 4, 4]})

    # --- Row 0: 2D contour plots ---
    _tricontour(axes[0, 0], ux_x, ux_y, ux_vals, "u⁺", geom)
    _tricontour(axes[0, 1], k_x, k_y, k_vals, "k⁺", geom)
    _tricontour(axes[0, 2], nut_x, nut_y, nut_vals / nu, "ν_t/ν", geom)

    # --- Row 1: 1D profiles ---
    delta = geom.Ly / 2
    y_lower = y_vals[y_vals <= delta]
    y_plus = y_lower * Re_tau

    # u+ (semilog with law of the wall)
    ax = axes[1, 0]
    u_lower = u_profile[: len(y_lower)]
    ax.semilogx(y_plus, u_lower, "b-", linewidth=2, label="RANS k-ω")
    y_visc = np.linspace(1, 11, 50)
    y_log = np.linspace(11, 300, 50)
    ax.semilogx(y_visc, y_visc, "k--", linewidth=1, alpha=0.5, label="u⁺=y⁺")
    ax.semilogx(y_log, 2.5 * np.log(y_log) + 5.5, "k:", linewidth=1, alpha=0.5, label="log law")
    ax.set_xlabel("y⁺")
    ax.set_ylabel("u⁺")
    ax.set_title("Mean velocity profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 600)

    # k+
    ax = axes[1, 1]
    k_lower = k_profile[: len(y_lower)]
    ax.plot(y_plus, k_lower, "r-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("k⁺")
    ax.set_title("Turbulent kinetic energy")
    ax.grid(True, alpha=0.3)

    # omega (log scale)
    ax = axes[1, 2]
    omega_lower = omega_profile[: len(y_lower)]
    ax.semilogy(y_plus, omega_lower, "g-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("ω⁺")
    ax.set_title("Specific dissipation")
    ax.grid(True, alpha=0.3)

    # --- Row 2: more profiles ---
    # nu_t/nu
    ax = axes[2, 0]
    nu_t_lower = nu_t_profile[: len(y_lower)]
    ax.plot(y_plus, nu_t_lower / nu, "m-", linewidth=2)
    ax.set_xlabel("y⁺")
    ax.set_ylabel("ν_t/ν")
    ax.set_title("Eddy viscosity ratio")
    ax.grid(True, alpha=0.3)

    # Linear u+ for near-wall check
    ax = axes[2, 1]
    ax.plot(y_plus, u_lower, "b-", linewidth=2, label="RANS k-ω")
    ax.plot(y_plus, y_plus, "k--", linewidth=1, alpha=0.5, label="u⁺=y⁺")
    ax.set_xlabel("y⁺")
    ax.set_ylabel("u⁺")
    ax.set_title("Near-wall velocity (linear)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[2, 2]
    ax.axis("off")
    info_text = (
        f"Final Solution\n"
        f"─────────────────\n"
        f"u⁺_max = {np.max(u_profile):.3f}\n"
        f"k⁺: [{np.min(k_profile):.2e}, {np.max(k_profile):.2e}]\n"
        f"ω⁺: [{np.min(omega_profile):.2e}, {np.max(omega_profile):.2e}]\n"
        f"ν_t/ν max = {np.max(nu_t_profile) / nu:.1f}\n"
    )
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved final fields plot: {save_path}")
    plt.close(fig)


def _tricontour(ax, x, y, vals, label, geom, n_levels=32):
    """Helper for tricontourf with safe level handling and correct aspect ratio."""
    vmin, vmax = np.min(vals), np.max(vals)
    if vmax - vmin < 1e-15:
        vmax = vmin + 1e-10  # Avoid zero-range levels (e.g., early iterations)
    levels = np.linspace(vmin, vmax, n_levels)
    tcf = ax.tricontourf(x, y, vals, levels=levels, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(label)
    ax.set_xlim(0, geom.Lx)
    ax.set_ylim(0, geom.Ly)
    plt.colorbar(tcf, ax=ax, shrink=0.8)


def plot_convergence(history_file: Path, save_path: Path | None = None):
    """Plot convergence history from CSV file.

    Shows residuals for each equation (momentum, k, omega) on log scale.
    This is the standard way to monitor CFD solver convergence.
    """
    if MPI.COMM_WORLD.rank != 0:
        return

    import csv

    # Read history
    data = {
        "iter": [],
        "dt": [],
        "res_u": [],
        "res_k": [],
        "res_w": [],
    }
    with open(history_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["iter"].append(int(row["iter"]))
            data["dt"].append(float(row["dt"]))
            # Individual equation residuals
            data["res_u"].append(float(row.get("res_u", row.get("residual", 1.0))))
            data["res_k"].append(float(row.get("res_k", 1.0)))
            data["res_w"].append(float(row.get("res_w", 1.0)))

    iters = data["iter"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residual history - each equation separately
    ax = axes[0]
    ax.semilogy(iters, data["res_u"], "b-", linewidth=1.2, label="Momentum (u)")
    ax.semilogy(iters, data["res_k"], "r-", linewidth=1.2, label="TKE (k)")
    ax.semilogy(iters, data["res_w"], "g-", linewidth=1.2, label="Omega (ω)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log)")
    ax.set_title("Equation Residuals")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=1e-10)  # Reasonable floor for log scale

    # Time step history
    ax = axes[1]
    ax.plot(iters, data["dt"], "k-", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("dt")
    ax.set_title("Adaptive Time Step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved convergence plot: {save_path}")
    plt.close(fig)
