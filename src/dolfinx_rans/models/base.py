"""
Abstract base class for RANS turbulence models.

Each model provides:
- UFL coefficients for the generic k/scalar transport equations
- NumPy-level field updates (nu_t, auxiliary blending functions)
- Wall BC metadata and initial condition conversions

The solver plugs these into a fixed equation template — zero model branches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FormCoefficients:
    """UFL expressions that parameterize the generic 2-equation template.

    The solver builds weak forms from these — models never construct forms.

    k-equation:
        dk/dt + u·∇k = ∇·((ν + sigma_k·ν_t_diff_k)∇k) + production_k - reaction_k·k

    scalar equation (ω or ε):
        dφ/dt + u·∇φ = ∇·((ν + sigma_phi·ν_t_diff_phi)∇φ) + production_phi
                        - reaction_phi·φ + cross_diffusion
    """

    sigma_k: Any  # k diffusion Prandtl number (Constant or Function)
    sigma_phi: Any  # scalar diffusion Prandtl number
    nu_t_diff_k: Any  # ν_t used in k diffusion (Function or UFL expr)
    nu_t_diff_phi: Any  # ν_t used in φ diffusion
    production_k: Any  # P_k source term (UFL expression)
    reaction_k: Any  # R_k coefficient (destruction = R_k · k)
    production_phi: Any  # P_φ source term
    reaction_phi: Any  # R_φ coefficient (destruction = R_φ · φ)
    cross_diffusion: Any  # CD term (UFL expression, 0 for k-ε)


@dataclass
class FieldSpec:
    """Clipping and wall-BC metadata for a scalar field."""

    clip_min: float
    clip_max: float
    has_wall_dirichlet: bool  # True → solver applies Dirichlet BC on walls
    wall_value: float = 0.0  # Dirichlet value (ignored if has_wall_dirichlet=False)


class RANSModel(ABC):
    """Abstract interface for a 2-equation RANS turbulence model."""

    # ── Metadata ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier, e.g. 'wilcox2006'."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label, e.g. 'k-ω (Wilcox 2006)'."""

    @property
    @abstractmethod
    def scalar_name(self) -> str:
        """Name of the second transported variable: 'ω' or 'ε'."""

    # ── Lifecycle ─────────────────────────────────────────────────

    @abstractmethod
    def setup(
        self,
        domain,
        S,
        k_prev,
        scalar_prev,
        k_n,
        scalar_n,
        nu_t_,
        u_n,
        nu_c,
        nu: float,
        turb_params,
        wall_facets: np.ndarray,
        is_bfs: bool,
        geom,
        y_first: float = 0.0,
    ) -> None:
        """One-time initialization after function spaces and fields exist.

        Models create any internal Functions (blending, damping) here and
        compute UFL sub-expressions needed by get_form_coefficients().

        Args:
            y_first: Actual first off-wall spacing (measured from mesh).
                     Used for omega wall BC: ω_wall = 6ν/(β₀·y₁²).
        """

    # ── Form coefficients ─────────────────────────────────────────

    @abstractmethod
    def get_form_coefficients(self) -> FormCoefficients:
        """Return UFL expressions for the generic transport template.

        Called once during form construction (not per iteration).
        """

    # ── Field specs ───────────────────────────────────────────────

    @abstractmethod
    def get_k_field_spec(self) -> FieldSpec:
        """Clipping bounds and wall BC info for k."""

    @abstractmethod
    def get_scalar_field_spec(self) -> FieldSpec:
        """Clipping bounds and wall BC info for the scalar (ω or ε)."""

    # ── Wall distance ─────────────────────────────────────────────

    @abstractmethod
    def needs_wall_distance(self) -> bool:
        """Whether this model requires wall-distance field y_wall."""

    @abstractmethod
    def compute_wall_distance(self, S, wall_facets, is_bfs, geom):
        """Compute and return wall-distance Function, or None."""

    # ── Per-Picard-iteration updates (NumPy level) ────────────────

    @abstractmethod
    def update_auxiliary_fields(
        self, k_arr: np.ndarray, scalar_arr: np.ndarray
    ) -> None:
        """Update blending functions, damping coefficients, etc.

        Called at the start of each Picard iteration, before k/φ solves.
        For models without auxiliary fields (Wilcox), this is a no-op.
        """

    @abstractmethod
    def compute_nu_t(
        self, k_arr: np.ndarray, scalar_arr: np.ndarray, S_mag_arr: np.ndarray
    ) -> np.ndarray:
        """Compute raw (unclipped, un-relaxed) eddy viscosity array."""

    # ── Initial conditions ────────────────────────────────────────

    @abstractmethod
    def convert_omega_to_scalar_ic(
        self, omega_arr: np.ndarray, k_arr: np.ndarray
    ) -> np.ndarray:
        """Convert ω initial condition to the model's scalar variable.

        For k-ω models this is identity; for k-ε this converts ω → ε.
        """

    @abstractmethod
    def initial_nu_t(
        self, k_arr: np.ndarray, scalar_arr: np.ndarray
    ) -> np.ndarray:
        """Compute initial ν_t from k and scalar ICs (before clipping)."""

    # ── Inlet BC ──────────────────────────────────────────────────

    @abstractmethod
    def compute_inlet_scalar(
        self, k_inlet: float, omega_inlet: float
    ) -> float:
        """Convert (k_inlet, ω_inlet) to the scalar inlet BC value."""
