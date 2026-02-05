"""
Plotting utilities for dolfinx-rans.

Visualizes mesh, initial conditions, and solution fields.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI


def plot_mesh(domain, geom, save_path: Path | None = None):
    """Plot the mesh with wall refinement highlighted."""
    if MPI.COMM_WORLD.rank != 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Get mesh coordinates and topology
    coords = domain.geometry.x
    cells = domain.geometry.dofmap

    # Left: Full mesh
    ax = axes[0]
    for cell in cells:
        pts = coords[cell]
        poly = plt.Polygon(pts[:, :2], fill=False, edgecolor="k", linewidth=0.3)
        ax.add_patch(poly)
    ax.set_xlim(0, geom.Lx)
    ax.set_ylim(0, geom.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Full mesh ({geom.Nx}×{geom.Ny} {geom.mesh_type}s)")

    # Right: Zoom near wall
    ax = axes[1]
    for cell in cells:
        pts = coords[cell]
        poly = plt.Polygon(pts[:, :2], fill=False, edgecolor="k", linewidth=0.5)
        ax.add_patch(poly)
    ax.set_xlim(0, geom.Lx / 4)
    ax.set_ylim(0, 0.15)  # Zoom to near-wall region
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Near-wall refinement (y_first={geom.y_first:.4f})")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved mesh plot: {save_path}")
    plt.close(fig)


def _extract_field_on_line(func, y_values, x_val, domain):
    """Extract scalar field values along a vertical line at x=x_val.

    For evaluating multiple fields at the same points, use
    extract_fields_on_line() which builds the bb_tree only once.
    """
    results = extract_fields_on_line([func], y_values, x_val, domain)
    return results[0]


def extract_fields_on_line(funcs, y_values, x_val, domain):
    """
    Extract multiple scalar fields along a vertical line at x=x_val.

    Builds the bounding-box tree and finds colliding cells ONCE,
    then evaluates all functions in a single pass.

    Args:
        funcs: list of DOLFINx Function or sub-function objects
        y_values: 1D array of y-coordinates to sample
        x_val: x-coordinate of the vertical line
        domain: DOLFINx mesh

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

    # Evaluate all functions at each point (one cell lookup per point)
    all_values = [np.zeros(len(y_values)) for _ in funcs]
    for i, point in enumerate(points):
        if len(cells.links(i)) > 0:
            cell = cells.links(i)[0]
            for j, func in enumerate(funcs):
                # DOLFINx Function.eval — evaluates FE interpolation at point
                result = func.eval(point, cell)  # noqa: S307
                all_values[j][i] = result[0]

    return all_values


def plot_initial_conditions(u, p, k, omega, nu_t, domain, geom, Re_tau, save_path: Path | None = None):
    """Plot initial condition fields."""
    if MPI.COMM_WORLD.rank != 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Sample points for profiles
    x_mid = geom.Lx / 2
    y_vals = np.linspace(0, geom.Ly, 100)

    # Extract profiles at channel center
    try:
        u_profile = _extract_field_on_line(u.sub(0), y_vals, x_mid, domain)
        k_profile = _extract_field_on_line(k, y_vals, x_mid, domain)
        omega_profile = _extract_field_on_line(omega, y_vals, x_mid, domain)
        nu_t_profile = _extract_field_on_line(nu_t, y_vals, x_mid, domain)

        # Nondimensional y+ coordinate
        y_plus = y_vals * Re_tau

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
        nu = 1.0 / Re_tau
        ax = axes[1, 1]
        ax.plot(y_plus, nu_t_profile / nu, "m-", linewidth=2)
        ax.set_xlabel("y⁺")
        ax.set_ylabel("ν_t/ν (IC)")
        ax.set_title("Eddy viscosity ratio")
        ax.grid(True, alpha=0.3)

    except Exception as e:
        print(f"  Warning: Could not extract IC profiles: {e}")
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Profile extraction failed", ha="center", va="center", transform=ax.transAxes)

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
    """Plot final solution fields as contours and profiles."""
    if MPI.COMM_WORLD.rank != 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Sample points for profiles
    x_mid = geom.Lx / 2
    y_vals = np.linspace(0.001, geom.Ly - 0.001, 100)  # Avoid exact boundaries

    try:
        u_profile = _extract_field_on_line(u.sub(0), y_vals, x_mid, domain)
        k_profile = _extract_field_on_line(k, y_vals, x_mid, domain)
        omega_profile = _extract_field_on_line(omega, y_vals, x_mid, domain)
        nu_t_profile = _extract_field_on_line(nu_t, y_vals, x_mid, domain)

        # Nondimensional y+ coordinate (use lower half only for symmetry)
        delta = geom.Ly / 2
        y_lower = y_vals[y_vals <= delta]
        y_plus = y_lower * Re_tau

        # u+ profile (lower half)
        ax = axes[0, 0]
        u_lower = u_profile[: len(y_lower)]
        ax.semilogx(y_plus, u_lower, "b-", linewidth=2, label="RANS k-ω")
        # Add law of the wall reference
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

        # k+ profile
        ax = axes[0, 1]
        k_lower = k_profile[: len(y_lower)]
        ax.plot(y_plus, k_lower, "r-", linewidth=2)
        ax.set_xlabel("y⁺")
        ax.set_ylabel("k⁺")
        ax.set_title("Turbulent kinetic energy")
        ax.grid(True, alpha=0.3)

        # omega profile (log scale)
        ax = axes[0, 2]
        omega_lower = omega_profile[: len(y_lower)]
        ax.semilogy(y_plus, omega_lower, "g-", linewidth=2)
        ax.set_xlabel("y⁺")
        ax.set_ylabel("ω⁺")
        ax.set_title("Specific dissipation")
        ax.grid(True, alpha=0.3)

        # nu_t / nu ratio
        nu = 1.0 / Re_tau
        ax = axes[1, 0]
        nu_t_lower = nu_t_profile[: len(y_lower)]
        ax.plot(y_plus, nu_t_lower / nu, "m-", linewidth=2)
        ax.set_xlabel("y⁺")
        ax.set_ylabel("ν_t/ν")
        ax.set_title("Eddy viscosity ratio")
        ax.grid(True, alpha=0.3)

        # Linear u+ profile for near-wall check
        ax = axes[1, 1]
        ax.plot(y_plus, u_lower, "b-", linewidth=2, label="RANS k-ω")
        ax.plot(y_plus, y_plus, "k--", linewidth=1, alpha=0.5, label="u⁺=y⁺")
        ax.set_xlabel("y⁺")
        ax.set_ylabel("u⁺")
        ax.set_title("Near-wall velocity (linear)")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 20)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    except Exception as e:
        print(f"  Warning: Could not extract final profiles: {e}")
        for ax in axes.flat[:5]:
            ax.text(0.5, 0.5, "Profile extraction failed", ha="center", va="center", transform=ax.transAxes)

    # Summary statistics
    ax = axes[1, 2]
    ax.axis("off")

    u_arr = u.x.array
    k_arr = k.x.array
    omega_arr = omega.x.array
    nu_t_arr = nu_t.x.array

    info_text = (
        f"Final Solution\n"
        f"─────────────────\n"
        f"u_max = {np.max(u_arr):.3f}\n"
        f"k: [{np.min(k_arr):.2e}, {np.max(k_arr):.2e}]\n"
        f"ω: [{np.min(omega_arr):.2e}, {np.max(omega_arr):.2e}]\n"
        f"ν_t/ν max = {np.max(nu_t_arr) * Re_tau:.1f}\n"
    )
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved final fields plot: {save_path}")
    plt.close(fig)


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
