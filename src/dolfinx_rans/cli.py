"""
Command-line interface for dolfinx-rans.

Usage:
    dolfinx-rans config.json
    python -m dolfinx_rans config.json
"""

import argparse
import sys
from pathlib import Path

from mpi4py import MPI

from dolfinx_rans.config import (
    BFSGeom,
    ChannelGeom,
    NondimParams,
    SolveParams,
    TurbParams,
)
from dolfinx_rans.geometry import (
    create_bfs_mesh,
    create_channel_mesh,
    mark_bfs_boundaries,
)
from dolfinx_rans.solver import solve_rans_kw
from dolfinx_rans.utils import (
    compute_bulk_velocity,
    compute_reattachment_length,
    dc_from_dict,
    diagnostics_scalar,
    diagnostics_vector,
    load_json_config,
    prepare_case_dir,
    print_dc_json,
)
from dolfinx_rans.plotting import (
    plot_mesh,
    plot_final_fields,
    plot_convergence,
    write_channel_profile_csv,
    plot_bfs_fields,
    plot_bfs_cf,
    plot_bfs_profiles,
    write_bfs_profile_csv,
)


def _resolve_reference_profile_csv(cfg: dict, cfg_path: Path) -> Path | None:
    """Resolve optional reference profile path for final dashed overlays."""
    bench = dict(cfg.get("benchmark", {}))
    ref_cfg = bench.get("reference_profile_csv")
    if isinstance(ref_cfg, str) and ref_cfg.strip():
        ref_path = Path(ref_cfg)
        if not ref_path.is_absolute():
            ref_path = (cfg_path.parent / ref_path).resolve()
        if ref_path.exists():
            return ref_path
        return None

    # Default convenience path for the re100k workflow.
    base = cfg_path.parent.parent / "nek_re100k"
    candidates = [
        base / "nek_profile_outer.csv",
        base / "nek_to_csv.csv",
        base / "nek_to_csv_symmetry.csv",
    ]
    for auto in candidates:
        if auto.exists():
            return auto
    return None


def _detect_geometry_type(cfg: dict) -> str:
    """Detect geometry type from config. Returns 'channel' or 'bfs'."""
    # New-style: explicit "geometry" section with "type" field
    geom_section = cfg.get("geometry", {})
    if isinstance(geom_section, dict) and "type" in geom_section:
        return geom_section["type"].lower()
    # Legacy: "geom" section implies channel
    if "geom" in cfg:
        return "channel"
    raise ValueError("Config must have either 'geometry' or 'geom' section")


def _run_channel(cfg, cfg_path, args, turb, solve_params):
    """Run channel flow case."""
    # Parse channel-specific config
    geom_raw = cfg.get("geometry", cfg.get("geom"))
    # Strip 'type' key if present (not a ChannelGeom field)
    geom_dict = {k: v for k, v in geom_raw.items() if k != "type"}
    geom = dc_from_dict(ChannelGeom, geom_dict, name="geom")
    nondim = dc_from_dict(NondimParams, cfg["nondim"], name="nondim")
    Re_tau = nondim.Re_tau

    if args.print_only:
        print_dc_json(geom)
        print_dc_json(solve_params)
        print_dc_json(nondim)
        return 0

    results_dir = Path(solve_params.out_dir)
    if MPI.COMM_WORLD.rank == 0:
        prepare_case_dir(results_dir, config_path=cfg_path, cfg=cfg, snps_subdir="snps")
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0:
        print("=" * 60)
        print("RANS k-ω CHANNEL FLOW - dolfinx-rans")
        print("=" * 60)
        print(f"Mode: NONDIMENSIONAL (Re_τ = {Re_tau})")
        print(f"Scaling: δ = 1, u_τ = 1, ν* = 1/Re_τ = {1.0/Re_tau:.6f}")
        stretch_mode = geom.stretching.lower()
        if geom.y_first > 0 and (stretch_mode == "tanh" or geom.growth_rate > 1.0):
            print(f"Mesh: {geom.Nx}×{geom.Ny} ({geom.mesh_type}, {stretch_mode})")
        else:
            print(f"Mesh: {geom.Nx}×{geom.Ny} ({geom.mesh_type})")
        print(f"Domain: {geom.Lx:.2f} × {geom.Ly:.2f}")
        print()

    domain = create_channel_mesh(geom, Re_tau=Re_tau)

    reference_profile_csv = _resolve_reference_profile_csv(cfg, cfg_path)
    if MPI.COMM_WORLD.rank == 0 and reference_profile_csv is not None:
        print(f"Reference overlay CSV: {reference_profile_csv}")

    plot_mesh(domain, geom, save_path=results_dir / "mesh.png")

    u, p, k, omega, nu_t, V, Q, S, domain, step, t = solve_rans_kw(
        domain, geom, turb, solve_params, results_dir, nondim=nondim
    )

    # Post-processing
    plot_final_fields(
        u, p, k, omega, nu_t, domain, geom, Re_tau,
        save_path=results_dir / "final_fields.png",
        reference_profile_csv=reference_profile_csv,
    )
    write_channel_profile_csv(
        u, k, omega, nu_t, domain, geom, Re_tau,
        save_path=results_dir / "profiles.csv",
    )

    history_file = results_dir / "history.csv"
    if history_file.exists():
        plot_convergence(history_file, save_path=results_dir / "convergence.png")

    U_bulk = compute_bulk_velocity(u, geom.Lx, geom.Ly)
    _print_summary(domain, u, p, k, omega, nu_t, U_bulk, results_dir)
    return 0


def _run_bfs(cfg, cfg_path, args, turb, solve_params):
    """Run backward-facing step case."""
    geom_dict = {k: v for k, v in cfg["geometry"].items() if k != "type"}
    geom = dc_from_dict(BFSGeom, geom_dict, name="geometry")
    nondim = dc_from_dict(NondimParams, cfg["nondim"], name="nondim")

    if args.print_only:
        print_dc_json(geom)
        print_dc_json(solve_params)
        print_dc_json(nondim)
        return 0

    results_dir = Path(solve_params.out_dir)
    if MPI.COMM_WORLD.rank == 0:
        prepare_case_dir(results_dir, config_path=cfg_path, cfg=cfg, snps_subdir="snps")
    MPI.COMM_WORLD.barrier()

    h = geom.step_height
    ER = geom.expansion_ratio
    H_inlet = h / (ER - 1.0)
    H_outlet = H_inlet + h

    if MPI.COMM_WORLD.rank == 0:
        print("=" * 60)
        print("RANS k-ω BACKWARD-FACING STEP - dolfinx-rans")
        print("=" * 60)
        print(f"Re_τ = {nondim.Re_tau}")
        print(f"Step height h = {h}, ER = {ER}")
        print(f"H_inlet = {H_inlet:.4f}, H_outlet = {H_outlet:.4f}")
        print(f"Mesh: {geom.Nx_upstream}+{geom.Nx_downstream} x {geom.Ny_outlet}")
        print()

    domain = create_bfs_mesh(geom)
    boundaries = mark_bfs_boundaries(domain, geom)

    plot_mesh(domain, geom, save_path=results_dir / "mesh.png")

    u, p, k, omega, nu_t, V, Q, S, domain, step, t = solve_rans_kw(
        domain, geom, turb, solve_params, results_dir,
        nondim=nondim, boundaries=boundaries,
    )

    # Post-processing
    nu = 1.0 / nondim.Re_tau

    # 2D contour fields
    plot_bfs_fields(u, p, k, omega, nu_t, domain, geom, nu,
                    save_path=results_dir / "bfs_fields.png")

    # Reattachment length (all ranks must participate)
    x_r = compute_reattachment_length(u, domain, nu, x_step=0.0, y_wall=0.0)
    if MPI.COMM_WORLD.rank == 0 and x_r is not None:
        print(f"  Reattachment length: x_r = {x_r:.3f}, x_r/h = {x_r/h:.2f}")

    # Skin friction along bottom wall
    plot_bfs_cf(u, domain, geom, nu, x_r=x_r, save_path=results_dir / "bfs_cf.png")

    # Vertical profiles at downstream stations
    plot_bfs_profiles(u, k, nu_t, domain, geom, nu,
                      save_path=results_dir / "bfs_profiles.png")

    # CSV export
    write_bfs_profile_csv(u, k, omega, nu_t, domain, geom, nu,
                          save_path=results_dir / "bfs_profiles.csv")

    history_file = results_dir / "history.csv"
    if history_file.exists():
        plot_convergence(history_file, save_path=results_dir / "convergence.png")

    _print_summary(domain, u, p, k, omega, nu_t, U_bulk=None, results_dir=results_dir)
    return 0


def _print_summary(domain, u, p, k, omega, nu_t, U_bulk, results_dir):
    """Print final solution diagnostics (all ranks must call)."""
    ud = diagnostics_vector(u)
    pd = diagnostics_scalar(p)
    kd = diagnostics_scalar(k)
    wd = diagnostics_scalar(omega)
    nutd = diagnostics_scalar(nu_t)

    if domain.comm.rank == 0:
        print("\n" + "─" * 50)
        print("FINAL SOLUTION SUMMARY")
        print("─" * 50)
        if U_bulk is not None:
            print(f"  U_bulk:    {U_bulk:.4f}")
        print(f"  u:         [{float(ud['c0_min']):.4f}, {float(ud['c0_max']):.4f}]")
        print(f"  v:         [{float(ud['c1_min']):.4f}, {float(ud['c1_max']):.4f}]")
        print(f"  p:         [{float(pd['min']):.4e}, {float(pd['max']):.4e}]")
        print(f"  k:         [{float(kd['min']):.4e}, {float(kd['max']):.4e}]")
        print(f"  ω:         [{float(wd['min']):.4e}, {float(wd['max']):.4e}]")
        print(f"  ν_t:       [{float(nutd['min']):.4e}, {float(nutd['max']):.4e}]")
        print("─" * 50)
        print(f"Results saved to {results_dir}/")
        print("=" * 60)


def main():
    """Run RANS k-ω solver from command line."""
    p = argparse.ArgumentParser(
        description="RANS k-ω solver for DOLFINx (channel + BFS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    dolfinx-rans channel_config.json
    dolfinx-rans bfs_config.json
    dolfinx-rans --print-only config.json

Environment:
    Requires DOLFINx 0.10.0+.
    Activate your FEniCSx environment before running.
        """,
    )
    p.add_argument("config", type=str, help="JSON config file")
    p.add_argument("--print-only", action="store_true", help="Print config and exit")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = load_json_config(cfg_path)

    # Common config sections
    turb = dc_from_dict(TurbParams, cfg["turb"], name="turb")
    solve_params = dc_from_dict(SolveParams, cfg["solve"], name="solve")

    geom_type = _detect_geometry_type(cfg)

    if geom_type == "channel":
        return _run_channel(cfg, cfg_path, args, turb, solve_params)
    elif geom_type == "bfs":
        return _run_bfs(cfg, cfg_path, args, turb, solve_params)
    else:
        print(f"ERROR: Unknown geometry type '{geom_type}'. Expected 'channel' or 'bfs'.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
