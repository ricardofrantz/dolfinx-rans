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

from dolfinx_rans.solver import (
    ChannelGeom,
    NondimParams,
    SolveParams,
    TurbParams,
    create_channel_mesh,
    solve_rans_kw,
)
from dolfinx_rans.utils import (
    compute_bulk_velocity,
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
)


def main():
    """Run RANS k-ω solver from command line."""
    p = argparse.ArgumentParser(
        description="RANS k-ω channel flow solver for DOLFINx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    dolfinx-rans channel_nek_re125k_like.json
    dolfinx-rans --print-only channel_nek_re125k_like.json

Environment:
    Requires DOLFINx 0.10.0+ and dolfinx_mpc for periodic BCs.
    Activate your FEniCSx environment before running.
        """,
    )
    p.add_argument("config", type=str, help="JSON config file")
    p.add_argument("--print-only", action="store_true", help="Print config and exit")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = load_json_config(cfg_path)

    # Parse config sections
    geom = dc_from_dict(ChannelGeom, cfg["geom"], name="geom")
    turb = dc_from_dict(TurbParams, cfg["turb"], name="turb")
    solve = dc_from_dict(SolveParams, cfg["solve"], name="solve")
    nondim = dc_from_dict(NondimParams, cfg["nondim"], name="nondim")

    Re_tau = nondim.Re_tau

    if args.print_only:
        print_dc_json(geom)
        print_dc_json(turb)
        print_dc_json(nondim)
        return 0

    results_dir = Path(solve.out_dir)
    if MPI.COMM_WORLD.rank == 0:
        prepare_case_dir(
            results_dir,
            config_path=cfg_path,
            cfg=cfg,
            snps_subdir="snps",
        )
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0:
        print("=" * 60)
        print("RANS k-ω CHANNEL FLOW - dolfinx-rans")
        print("=" * 60)
        print(f"Mode: NONDIMENSIONAL (Re_τ = {Re_tau})")
        print(f"Scaling: δ = 1, u_τ = 1, ν* = 1/Re_τ = {1.0/Re_tau:.6f}")
        print(f"Mesh: {geom.Nx}×{geom.Ny} ({geom.mesh_type})")
        print(f"Domain: {geom.Lx:.2f} × {geom.Ly:.2f}")
        print()

    domain = create_channel_mesh(geom, Re_tau=Re_tau)

    # Plot mesh
    plot_mesh(domain, geom, save_path=results_dir / "mesh.png")

    u, p, k, omega, nu_t, V, Q, S, domain, step, t = solve_rans_kw(
        domain, geom, turb, solve, results_dir, nondim=nondim
    )

    # Plot final results
    plot_final_fields(u, p, k, omega, nu_t, domain, geom, Re_tau, save_path=results_dir / "final_fields.png")
    write_channel_profile_csv(
        u, k, omega, nu_t, domain, geom, Re_tau,
        save_path=results_dir / "profiles.csv",
    )

    # Plot convergence history
    history_file = results_dir / "history.csv"
    if history_file.exists():
        plot_convergence(history_file, save_path=results_dir / "convergence.png")

    # Compute U_bulk (MPI collective - all ranks must call)
    U_bulk = compute_bulk_velocity(u, geom.Lx, geom.Ly)

    # MPI collectives — all ranks must participate
    ud = diagnostics_vector(u)
    pd = diagnostics_scalar(p)
    kd = diagnostics_scalar(k)
    wd = diagnostics_scalar(omega)
    nutd = diagnostics_scalar(nu_t)

    if domain.comm.rank == 0:
        print("\n" + "─" * 50)
        print("FINAL SOLUTION SUMMARY")
        print("─" * 50)
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
