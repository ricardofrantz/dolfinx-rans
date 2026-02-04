"""
dolfinx-rans: RANS turbulence models for DOLFINx.

A FEniCSx-based implementation of Reynolds-Averaged Navier-Stokes
turbulence models for channel flow validation.

Requirements:
    - DOLFINx 0.10.0+
    - dolfinx_mpc (for periodic boundary conditions)
    - numpy, matplotlib

Example:
    from dolfinx_rans import solve_channel
    results = solve_channel("config.json")
"""

__version__ = "0.1.0"

from dolfinx_rans.solver import (
    ChannelGeom,
    NondimParams,
    TurbParams,
    SolveParams,
    create_channel_mesh,
    solve_rans_kw,
)

__all__ = [
    "__version__",
    "ChannelGeom",
    "NondimParams",
    "TurbParams",
    "SolveParams",
    "create_channel_mesh",
    "solve_rans_kw",
]
