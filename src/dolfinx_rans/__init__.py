"""
dolfinx-rans: RANS turbulence models for DOLFINx.

A FEniCSx-based implementation of Reynolds-Averaged Navier-Stokes
turbulence models for channel flow and backward-facing step.

Requirements:
    - DOLFINx 0.10.0+
    - numpy, matplotlib

Example:
    from dolfinx_rans import ChannelGeom, create_channel_mesh, solve_rans_kw
    geom = ChannelGeom(Lx=1.0, Ly=1.0, Nx=96, Ny=96, ...)
    domain = create_channel_mesh(geom, Re_tau=5200)
    u, p, k, omega, nu_t, ... = solve_rans_kw(domain, geom, turb, solve, results_dir, nondim)
"""

__version__ = "0.1.0"

from dolfinx_rans.config import (
    BFSGeom,
    ChannelGeom,
    NondimParams,
    SolveParams,
    TurbParams,
)
from dolfinx_rans.geometry import create_bfs_mesh, create_channel_mesh
from dolfinx_rans.rans_solver import solve_rans_kw

__all__ = [
    "__version__",
    "BFSGeom",
    "ChannelGeom",
    "NondimParams",
    "TurbParams",
    "SolveParams",
    "create_bfs_mesh",
    "create_channel_mesh",
    "solve_rans_kw",
]
