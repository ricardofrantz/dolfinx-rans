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

    # Compute domain extents from actual mesh coordinates
    all_x = np.concatenate([p[:, 0] for p in polys])
    all_y = np.concatenate([p[:, 1] for p in polys])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    Lx = x_max - x_min
    Ly_plot = y_max - y_min

    ar = max(Lx / max(Ly_plot, 1e-10), 0.5)
    fig_w = 14
    fig_h = fig_w / ar + 1.2

    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 20)))
    for pts in polys:
        ax.add_patch(plt.Polygon(pts, fill=False, edgecolor="k", linewidth=0.3))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Geometry-specific title
    from dolfinx_rans.config import BFSGeom
    if isinstance(geom, BFSGeom):
        ncells = len(polys)
        ax.set_title(f"BFS mesh: {ncells} {geom.mesh_type}s  (h={geom.step_height}, ER={geom.expansion_ratio})")
    else:
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


def _mask_bfs_triangulation(x, y, geom):
    """Create a Triangulation with BFS step region masked out.

    Uses Delaunay triangulation but removes any triangle whose centroid
    falls inside the solid step region (x < 0 and y < step height).

    Returns:
        matplotlib.tri.Triangulation with mask applied, or None if geom
        is not a BFS geometry.
    """
    from matplotlib.tri import Triangulation
    from dolfinx_rans.config import BFSGeom

    if not isinstance(geom, BFSGeom):
        return None

    tri = Triangulation(x, y)
    triangles = tri.triangles

    # Centroid of each triangle
    cx = np.mean(x[triangles], axis=1)
    cy = np.mean(y[triangles], axis=1)

    # Mask triangles inside the solid step (upstream of step AND below step top)
    h = geom.step_height
    ER = geom.expansion_ratio
    y_step = h / (ER - 1.0)  # H_inlet = y-coordinate of step top
    mask = (cx < 0) & (cy < y_step)
    tri.set_mask(mask)
    return tri


def write_channel_profile_csv(u, k, omega, nu_t, domain, geom, Re_tau, save_path: Path, n_points: int = 256):
    """
    Export centerline-normalized channel profiles for RANS benchmark comparison.

    Output columns:
      y, y_over_delta, u, u_over_ubulk, k, omega, nu_t_over_nu
    """
    comm = domain.comm

    # For consistency with channel convention, use wall distance from
    # the nearest wall (0 in full channel).
    delta = geom.Ly if geom.use_symmetry else geom.Ly / 2.0

    # Avoid sampling exactly on boundaries for robust point evaluation.
    eps = min(1e-6, max(1e-10, 0.1 * geom.y_first))
    if 2.0 * eps >= delta:
        eps = 1e-10

    y_vals = np.linspace(eps, delta - eps, n_points)
    x_mid = geom.Lx / 2.0

    u_profile, k_profile, omega_profile, nu_t_profile = extract_fields_on_line(
        [u.sub(0), k, omega, nu_t], y_vals, x_mid, domain, comm=comm,
    )

    if comm.rank != 0:
        return

    y_over_delta = y_vals / max(delta, 1e-30)
    # Nek-style normalization is reported in a dedicated column:
    # - u: solver-native velocity
    # - u_over_ubulk: u / U_bulk
    valid = np.isfinite(y_over_delta) & np.isfinite(u_profile)
    if np.any(valid):
        u_bulk = float(np.trapezoid(u_profile[valid], y_over_delta[valid]) / max(y_over_delta[valid][-1] - y_over_delta[valid][0], 1e-30))
    else:
        u_bulk = 1.0
    u_profile_out = u_profile
    u_over_ubulk = u_profile / max(u_bulk, 1e-30)
    nu = 1.0 / Re_tau
    k_profile_out = k_profile
    omega_profile_out = omega_profile
    nu_t_over_nu = nu_t_profile / nu

    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([
        y_vals,
        y_over_delta,
        u_profile_out,
        u_over_ubulk,
        k_profile_out,
        omega_profile_out,
        nu_t_over_nu,
    ])
    np.savetxt(
        save_path,
        data,
        delimiter=",",
        header="y,y_over_delta,u,u_over_ubulk,k,omega,nu_t_over_nu",
        comments="",
    )
    print(f"  Saved profile CSV: {save_path}")


def _load_reference_profile_csv(path: Path | None):
    """Load reference profile CSV for dashed overlays on final profile plots."""
    if path is None:
        return None
    if not path.exists():
        return None

    import csv

    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    header = rows[0]
    y_over_delta = None
    if "y_over_delta" in header:
        y_over_delta = np.array([float(r["y_over_delta"]) for r in rows], dtype=float)
    elif "y_lower_0_to_1" in header:
        y_over_delta = np.array([float(r["y_lower_0_to_1"]) for r in rows], dtype=float)
    else:
        # Keep legacy wall-unit support only when Re_tau is directly available.
        if "y_plus" in header and "Re_tau" in header:
            re_tau = float(rows[0]["Re_tau"])
            if re_tau > 0:
                y_over_delta = np.array([float(r["y_plus"]) for r in rows], dtype=float) / re_tau
    if y_over_delta is None:
        return None
    if not np.all(np.isfinite(y_over_delta)):
        return None

    order = np.argsort(y_over_delta)
    out = {
        "label": f"Reference ({path.stem})",
        "y_over_delta": y_over_delta[order],
    }

    for key in ("u", "v", "pressure", "temp", "scalar_1", "scalar_2", "k", "omega", "u_over_ubulk"):
        if key in header:
            out[key] = np.array([float(r[key]) for r in rows], dtype=float)[order]

    # Backward compatibility:
    #   - u_plus / k_plus in old outputs are wall-units
    #   - u and k here are solver-native for RANS (already physical / non-wall scaled)
    if "u_over_ubulk" not in out:
        u_raw = None
        if "u" in header:
            u_raw = np.array([float(r["u"]) for r in rows], dtype=float)[order]
        elif "u_plus" in header:
            u_raw = np.array([float(r["u_plus"]) for r in rows], dtype=float)[order]
        if u_raw is not None:
            y_ref = np.asarray(out["y_over_delta"], dtype=float)
            u_bulk = float(np.trapezoid(u_raw, y_ref) / max(y_ref[-1] - y_ref[0], 1e-30))
            out["u_over_ubulk"] = u_raw / max(u_bulk, 1e-30)

    if "scalar_1" not in out:
        if "k_over_ubulk" in header:
            out["scalar_1"] = np.array([float(r["k_over_ubulk"]) for r in rows], dtype=float)[order]
        elif "k" in header:
            y_ref = np.asarray(out["y_over_delta"], dtype=float)
            k_ref = np.array([float(r["k"]) for r in rows], dtype=float)[order]
            k_bulk = float(np.trapezoid(k_ref, y_ref) / max(y_ref[-1] - y_ref[0], 1e-30))
            out["scalar_1"] = k_ref / max(k_bulk, 1e-30)
        elif "k_plus" in header:
            y_ref = np.asarray(out["y_over_delta"], dtype=float)
            k_plus = np.array([float(r["k_plus"]) for r in rows], dtype=float)[order]
            k_bulk = float(np.trapezoid(k_plus, y_ref) / max(y_ref[-1] - y_ref[0], 1e-30))
            out["scalar_1"] = k_plus / max(k_bulk, 1e-30)

    if "scalar_2" not in out:
        if "omega_over_ubulk" in header:
            out["scalar_2"] = np.array([float(r["omega_over_ubulk"]) for r in rows], dtype=float)[order]
        elif "omega_plus" in header:
            out["scalar_2"] = np.array([float(r["omega_plus"]) for r in rows], dtype=float)[order]
        elif "omega" in header:
            out["scalar_2"] = np.array([float(r["omega"]) for r in rows], dtype=float)[order]

    return out


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


def plot_final_fields(
    u,
    p,
    k,
    omega,
    nu_t,
    domain,
    geom,
    Re_tau,
    save_path: Path | None = None,
    reference_profile_csv: Path | None = None,
):
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

    reference = _load_reference_profile_csv(reference_profile_csv)

    nu = 1.0 / Re_tau
    ar = geom.Lx / geom.Ly  # channel aspect ratio (~6.28)
    # Contour row height adapts to true geometry; profile rows are fixed
    contour_h = 16 / (3 * ar)  # width per panel / aspect ratio
    fig, axes = plt.subplots(3, 3, figsize=(16, contour_h + 8),
                             gridspec_kw={"height_ratios": [contour_h, 4, 4]})

    # --- Row 0: 2D contour plots ---
    _tricontour(axes[0, 0], ux_x, ux_y, ux_vals, "u", geom)
    _tricontour(axes[0, 1], k_x, k_y, k_vals, "k", geom)
    _tricontour(axes[0, 2], nut_x, nut_y, nut_vals / nu, "ν_t/ν", geom)

    # --- Row 1: 1D profiles (pure y/delta, no y+) ---
    delta = geom.Ly if geom.use_symmetry else geom.Ly / 2.0
    y_lower = y_vals[y_vals <= delta]
    y_over_delta = y_lower / max(delta, 1e-30)

    # U/U_bulk
    ax = axes[1, 0]
    u_lower = u_profile[: len(y_lower)]
    u_bulk_local = float(
        np.trapezoid(u_lower, y_over_delta) / max(y_over_delta[-1] - y_over_delta[0], 1e-30)
    )
    ax.plot(y_over_delta, u_lower / max(u_bulk_local, 1e-30), "b-", linewidth=2, label="FEniCS (RANS k-ω)")
    if reference is not None:
        ref_y = np.asarray(reference["y_over_delta"], dtype=float)
        ref_u = np.asarray(reference.get("u_over_ubulk"), dtype=float)
        valid = np.isfinite(ref_y) & np.isfinite(ref_u)
        if np.any(valid):
            ax.plot(
                ref_y[valid],
                ref_u[valid],
                linestyle="--",
                color="tab:orange",
                linewidth=1.8,
                label=f"Nek ({reference['label']})",
            )
    ax.set_xlabel("y/delta")
    ax.set_ylabel("U/U_bulk")
    ax.set_title("Mean velocity profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)

    # k
    ax = axes[1, 1]
    k_lower = k_profile[: len(y_lower)]
    ax.plot(y_over_delta, k_lower, "r-", linewidth=2, label="FEniCS k")
    if reference is not None:
        ref_y = np.asarray(reference["y_over_delta"], dtype=float)
        ref_k = None
        ref_label = None
        if "scalar_1" in reference:
            ref_k = np.asarray(reference["scalar_1"], dtype=float)
            ref_label = "Nek scalar_1"
        if ref_k is not None:
            valid = np.isfinite(ref_y) & np.isfinite(ref_k)
            if np.any(valid):
                ax.plot(
                    ref_y[valid],
                    ref_k[valid],
                    linestyle="--",
                    color="tab:orange",
                    linewidth=1.8,
                    label=ref_label,
                )
                ax.legend(fontsize=8)
    ax.set_xlabel("y/delta")
    ax.set_ylabel("k")
    ax.set_title("Turbulent kinetic energy")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)

    # omega (log scale)
    ax = axes[1, 2]
    omega_lower = omega_profile[: len(y_lower)]
    ax.semilogy(y_over_delta, omega_lower, "g-", linewidth=2, label="FEniCS omega")
    if reference is not None:
        ref_y = np.asarray(reference["y_over_delta"], dtype=float)
        ref_w = None
        ref_label = None
        if "scalar_2" in reference:
            ref_w = np.asarray(reference["scalar_2"], dtype=float)
            ref_label = "Nek scalar_2"
        if ref_w is not None:
            valid = np.isfinite(ref_y) & np.isfinite(ref_w) & (ref_w > 0)
            if np.any(valid):
                ax.semilogy(
                    ref_y[valid],
                    ref_w[valid],
                    linestyle="--",
                    color="tab:orange",
                    linewidth=1.8,
                    label=ref_label,
                )
                ax.legend(fontsize=8)
    ax.set_xlabel("y/delta")
    ax.set_ylabel("omega")
    ax.set_title("Specific dissipation")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)

    # --- Row 2: more profiles ---
    # nu_t/nu
    ax = axes[2, 0]
    nu_t_lower = nu_t_profile[: len(y_lower)]
    ax.plot(y_over_delta, nu_t_lower / nu, "m-", linewidth=2)
    ax.set_xlabel("y/delta")
    ax.set_ylabel("ν_t/ν")
    ax.set_title("Eddy viscosity ratio")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 1.0)

    # Near-wall U/U_bulk (linear)
    ax = axes[2, 1]
    ax.plot(y_over_delta, u_lower / max(u_bulk_local, 1e-30), "b-", linewidth=2, label="FEniCS")
    if reference is not None:
        ref_y = np.asarray(reference["y_over_delta"], dtype=float)
        ref_u = np.asarray(reference.get("u_over_ubulk"), dtype=float)
        valid = np.isfinite(ref_y) & np.isfinite(ref_u)
        if np.any(valid):
            ax.plot(
                ref_y[valid],
                ref_u[valid],
                linestyle="--",
                color="tab:orange",
                linewidth=1.8,
                label="Nek",
            )
    ax.set_xlabel("y/delta")
    ax.set_ylabel("U/U_bulk")
    ax.set_title("Near-wall velocity")
    ax.set_xlim(0.0, 0.08)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[2, 2]
    ax.axis("off")
    info_text = (
        f"Final Solution\n"
        f"─────────────────\n"
        f"u_max = {np.max(u_profile):.3f}\n"
        f"k: [{np.min(k_profile):.2e}, {np.max(k_profile):.2e}]\n"
        f"omega: [{np.min(omega_profile):.2e}, {np.max(omega_profile):.2e}]\n"
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


def _tricontour(ax, x, y, vals, label, geom=None, n_levels=32):
    """Helper for tricontourf with safe level handling and correct aspect ratio.

    For BFS geometry, masks Delaunay triangles inside the solid step region.
    """
    vmin, vmax = np.min(vals), np.max(vals)
    if vmax - vmin < 1e-15:
        vmax = vmin + 1e-10  # Avoid zero-range levels (e.g., early iterations)
    levels = np.linspace(vmin, vmax, n_levels)
    # Build masked triangulation for BFS (removes step interior)
    tri = _mask_bfs_triangulation(x, y, geom)
    if tri is not None:
        tcf = ax.tricontourf(tri, vals, levels=levels, cmap="viridis")
    else:
        tcf = ax.tricontourf(x, y, vals, levels=levels, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(label)
    # Geometry-agnostic limits from data extents
    if geom is not None and hasattr(geom, "Lx"):
        ax.set_xlim(0, geom.Lx)
        ax.set_ylim(0, geom.Ly)
    else:
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
    plt.colorbar(tcf, ax=ax, shrink=0.8)


def plot_convergence(history_file: Path, save_path: Path | None = None):
    """Plot convergence history from CSV file.

    Single figure: residuals on left y-axis (log), dt/CFL_max on right y-axis (linear).
    """
    if MPI.COMM_WORLD.rank != 0:
        return

    import csv

    # Read history
    data: dict[str, list[float]] = {
        "iter": [], "dt": [], "res_u": [], "res_k": [], "res_w": [], "cfl_max": [],
    }
    with open(history_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["iter"].append(int(row["iter"]))
            data["dt"].append(float(row["dt"]))
            data["res_u"].append(float(row.get("res_u", row.get("residual", 1.0))))
            data["res_k"].append(float(row.get("res_k", 1.0)))
            data["res_w"].append(float(row.get("res_w", 1.0)))
            cfl = row.get("cfl_max", "")
            data["cfl_max"].append(float(cfl) if cfl else 0.0)

    iters = data["iter"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Left y-axis: residuals (log scale)
    ax.semilogy(iters, data["res_u"], "b-", linewidth=1.2, label="Momentum (u)")
    ax.semilogy(iters, data["res_k"], "r-", linewidth=1.2, label="TKE (k)")
    ax.semilogy(iters, data["res_w"], "g-", linewidth=1.2, label=r"Omega ($\omega$)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual (log)")
    ax.set_title("Equation Residuals")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=1e-10)

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

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved convergence plot: {save_path}")
    plt.close(fig)


# =============================================================================
# BFS-specific plotting
# =============================================================================


def plot_bfs_fields(u, p, k, omega, nu_t, domain, geom, nu, save_path=None):
    """
    Plot 2D contour fields for backward-facing step.

    Shows u, v, k, nu_t/nu as colormaps over the L-shaped domain.
    Uses actual mesh triangulation to avoid Delaunay artifacts across the step.
    All ranks participate in gathering data; only rank 0 plots.

    Args:
        u, p, k, omega, nu_t: Solution fields
        domain: DOLFINx mesh
        geom: BFSGeom
        nu: Kinematic viscosity
        save_path: Path to save PNG (optional)
    """
    comm = domain.comm

    # Phase 1: gather 2D fields (ALL ranks)
    ux_x, ux_y, ux_vals = gather_scalar_field(u.sub(0).collapse(), comm)
    uy_x, uy_y, uy_vals = gather_scalar_field(u.sub(1).collapse(), comm)
    k_x, k_y, k_vals = gather_scalar_field(k, comm)
    nut_x, nut_y, nut_vals = gather_scalar_field(nu_t, comm)

    # Phase 2: plot (rank 0 only)
    if comm.rank != 0:
        return

    h = geom.step_height
    fig, axes = plt.subplots(2, 2, figsize=(16, 6))

    _tricontour(axes[0, 0], ux_x, ux_y, ux_vals, "u (streamwise)", geom=geom)
    _tricontour(axes[0, 1], uy_x, uy_y, uy_vals, "v (wall-normal)", geom=geom)
    _tricontour(axes[1, 0], k_x, k_y, k_vals, "k (TKE)", geom=geom)
    nut_ratio = nut_vals / nu
    _tricontour(axes[1, 1], nut_x, nut_y, nut_ratio, f"nu_t/nu (max={np.max(nut_ratio):.0f})", geom=geom)

    fig.suptitle(f"BFS fields: h={h}, ER={geom.expansion_ratio}", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved BFS fields plot: {save_path}")
    plt.close(fig)


def plot_bfs_cf(u, domain, geom, nu, x_r=None, save_path=None):
    """
    Plot skin friction coefficient Cf(x/h) along the bottom wall.

    Negative Cf indicates reversed flow (recirculation zone).
    Marks reattachment point if provided.

    All ranks participate in Cf computation; only rank 0 plots.

    Args:
        u: Velocity vector function
        domain: DOLFINx mesh
        geom: BFSGeom
        nu: Kinematic viscosity
        x_r: Reattachment x-coordinate (optional, for annotation)
        save_path: Path to save PNG (optional)
    """
    from dolfinx_rans.utils import compute_cf_along_wall

    comm = domain.comm
    h = geom.step_height
    L_down = geom.downstream_length * h

    x_coords = np.linspace(0.5 * h, L_down * 0.98, 300)
    cf = compute_cf_along_wall(u, domain, nu, x_coords, y_wall=0.0)

    if comm.rank != 0:
        return

    x_over_h = x_coords / h

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_over_h, cf, "b-", linewidth=1.5)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    if x_r is not None:
        ax.axvline(x_r / h, color="r", linestyle=":", linewidth=1.2,
                   label=f"x_r/h = {x_r/h:.2f}")
        ax.legend(fontsize=10)
    ax.set_xlabel("x/h")
    ax.set_ylabel("Cf")
    ax.set_title(f"Skin friction (BFS, ER={geom.expansion_ratio})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved Cf plot: {save_path}")
    plt.close(fig)


def plot_bfs_profiles(u, k, nu_t, domain, geom, nu, x_stations_over_h=None, save_path=None):
    """
    Plot vertical profiles at multiple x/h stations downstream of the step.

    Shows u-velocity and TKE profiles at each station, revealing
    separated shear layer evolution and recovery.

    All ranks participate in extraction; only rank 0 plots.

    Args:
        u: Velocity vector function
        k: TKE function
        nu_t: Eddy viscosity function
        domain: DOLFINx mesh
        geom: BFSGeom
        nu: Kinematic viscosity
        x_stations_over_h: List of x/h values for profiles (default: [1,2,4,6,8,10])
        save_path: Path to save PNG (optional)
    """
    comm = domain.comm
    h = geom.step_height
    ER = geom.expansion_ratio
    H_inlet = h / (ER - 1.0)
    H_outlet = H_inlet + h

    if x_stations_over_h is None:
        L_down = geom.downstream_length
        # Reasonable default stations
        x_stations_over_h = [x for x in [1, 2, 4, 6, 8, 10, 15] if x < L_down * 0.9]

    n_stations = len(x_stations_over_h)
    n_points = 100
    y_vals = np.linspace(0.001, H_outlet - 0.001, n_points)

    # Extract profiles at each station (ALL ranks)
    u_profiles = []
    k_profiles = []
    for xh in x_stations_over_h:
        x_val = xh * h
        u_prof, k_prof = extract_fields_on_line(
            [u.sub(0), k], y_vals, x_val, domain, comm=comm,
        )
        u_profiles.append(u_prof)
        k_profiles.append(k_prof)

    # Plot (rank 0 only)
    if comm.rank != 0:
        return

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_stations))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Velocity profiles
    ax = axes[0]
    for i, xh in enumerate(x_stations_over_h):
        ax.plot(u_profiles[i], y_vals / h, color=colors[i], linewidth=1.3,
                label=f"x/h={xh}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("u")
    ax.set_ylabel("y/h")
    ax.set_title("Streamwise velocity profiles")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # TKE profiles
    ax = axes[1]
    for i, xh in enumerate(x_stations_over_h):
        ax.plot(k_profiles[i], y_vals / h, color=colors[i], linewidth=1.3,
                label=f"x/h={xh}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("y/h")
    ax.set_title("TKE profiles")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"BFS profiles: ER={geom.expansion_ratio}", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved BFS profiles plot: {save_path}")
    plt.close(fig)


def write_bfs_profile_csv(u, k, omega, nu_t, domain, geom, nu, save_path,
                           x_stations_over_h=None, n_points=200):
    """
    Export BFS vertical profiles at multiple x/h stations to CSV.

    Output columns:
        x_over_h, y, y_over_h, u, k, omega, nu_t_over_nu

    Args:
        u, k, omega, nu_t: Solution fields
        domain: DOLFINx mesh
        geom: BFSGeom
        nu: Kinematic viscosity
        save_path: Output CSV path
        x_stations_over_h: List of x/h values (default: [1,2,4,6,8,10])
        n_points: Points per profile
    """
    comm = domain.comm
    h = geom.step_height
    ER = geom.expansion_ratio
    H_inlet = h / (ER - 1.0)
    H_outlet = H_inlet + h

    if x_stations_over_h is None:
        L_down = geom.downstream_length
        x_stations_over_h = [x for x in [1, 2, 4, 6, 8, 10, 15] if x < L_down * 0.9]

    eps = 1e-6
    y_vals = np.linspace(eps, H_outlet - eps, n_points)

    all_rows = []
    for xh in x_stations_over_h:
        x_val = xh * h
        u_prof, k_prof, w_prof, nut_prof = extract_fields_on_line(
            [u.sub(0), k, omega, nu_t], y_vals, x_val, domain, comm=comm,
        )
        for j in range(n_points):
            all_rows.append([xh, y_vals[j], y_vals[j] / h,
                             u_prof[j], k_prof[j], w_prof[j], nut_prof[j] / nu])

    if comm.rank != 0:
        return

    data = np.array(all_rows)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        save_path,
        data,
        delimiter=",",
        header="x_over_h,y,y_over_h,u,k,omega,nu_t_over_nu",
        comments="",
    )
    print(f"  Saved BFS profiles CSV: {save_path}")
