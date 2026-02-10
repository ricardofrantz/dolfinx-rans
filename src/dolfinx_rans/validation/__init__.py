"""
Legacy DNS reference data for turbulent channel flow.

Data from Moser, Kim, Mansour (1999) "DNS of turbulent channel flow up to Re_τ=590"
Physics of Fluids, 11(4):943-945. DOI: 10.1063/1.869966

Available datasets:
    - mkm_re590: Re_τ = 590 channel flow
    - mkm_re180: Re_τ = 180 channel flow
"""

from dolfinx_rans.validation.mkm_re590 import (
    RE_TAU as RE_TAU_590,
    MEAN_VELOCITY as MEAN_VELOCITY_590,
    REYNOLDS_STRESSES as REYNOLDS_STRESSES_590,
    get_k_profile as get_k_profile_590,
    get_omega_profile as get_omega_profile_590,
)

from dolfinx_rans.validation.mkm_re180 import (
    RE_TAU as RE_TAU_180,
    MEAN_VELOCITY as MEAN_VELOCITY_180,
    REYNOLDS_STRESSES as REYNOLDS_STRESSES_180,
    get_k_profile as get_k_profile_180,
)

__all__ = [
    "RE_TAU_590",
    "MEAN_VELOCITY_590",
    "REYNOLDS_STRESSES_590",
    "get_k_profile_590",
    "get_omega_profile_590",
    "RE_TAU_180",
    "MEAN_VELOCITY_180",
    "REYNOLDS_STRESSES_180",
    "get_k_profile_180",
]
