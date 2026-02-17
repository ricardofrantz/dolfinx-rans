"""
k-ω SST turbulence model (Menter 1994).

Reference: Menter, F.R. "Two-equation eddy-viscosity turbulence models
           for engineering applications." AIAA Journal, 32(8), 1994.
"""

import numpy as np
from petsc4py import PETSc

from dolfinx.fem import Constant, Expression, Function

import ufl
from ufl import dot, grad, inner, sqrt, sym

from dolfinx_rans.config import BETA_0, KAPPA
from dolfinx_rans.geometry import compute_wall_distance_eikonal
from dolfinx_rans.models.base import FieldSpec, FormCoefficients, RANSModel


# Shared constants
BETA_STAR = 0.09
SQRT_BETA_STAR = np.sqrt(BETA_STAR)

# Inner layer (k-omega) constants — subscript 1
SIGMA_K1 = 0.85
SIGMA_W1 = 0.5
BETA1 = 0.075
GAMMA1 = BETA1 / BETA_STAR - SIGMA_W1 * KAPPA**2 / SQRT_BETA_STAR

# Outer layer (k-epsilon transformed) constants — subscript 2
SIGMA_K2 = 1.0
SIGMA_W2 = 0.856
BETA2 = 0.0828
GAMMA2 = BETA2 / BETA_STAR - SIGMA_W2 * KAPPA**2 / SQRT_BETA_STAR

# SST limiter constant
A1 = 0.31


class SSTModel(RANSModel):
    """k-ω SST (Menter 1994) with F1/F2 blending and ν_t limiter."""

    @property
    def model_name(self) -> str:
        return "sst"

    @property
    def display_name(self) -> str:
        return "k-ω SST (Menter 1994)"

    @property
    def scalar_name(self) -> str:
        return "ω"

    def setup(self, domain, S, k_prev, scalar_prev, k_n, scalar_n,
              nu_t_, u_n, nu_c, nu, turb_params, wall_facets, is_bfs, geom,
              y_first=0.0):
        self._domain = domain
        self._S = S
        self._nu = nu
        self._nu_c = nu_c
        self._nu_t_ = nu_t_
        self._k_prev = k_prev
        self._scalar_prev = scalar_prev
        self._k_n = k_n
        self._scalar_n = scalar_n
        self._u_n = u_n
        self._turb = turb_params
        self._omega_min = turb_params.omega_min
        self._k_min = turb_params.k_min
        self._wall_facets = wall_facets
        self._is_bfs = is_bfs
        self._geom = geom

        # Omega wall value (same formula as Wilcox)
        self._y_first = y_first
        self._omega_wall_val = 6.0 * nu / (BETA_0 * y_first**2)

        # Blending functions and blended coefficients (updated each Picard iter)
        self._F1 = Function(S, name="F1")
        self._F2 = Function(S, name="F2")
        self._sigma_k_blend = Function(S, name="sigma_k")
        self._sigma_w_blend = Function(S, name="sigma_w")
        self._beta_blend = Function(S, name="beta")
        self._gamma_blend = Function(S, name="gamma")

        # Initialize to inner-layer values (F1=1)
        self._F1.x.array[:] = 1.0
        self._F2.x.array[:] = 1.0
        self._sigma_k_blend.x.array[:] = SIGMA_K1
        self._sigma_w_blend.x.array[:] = SIGMA_W1
        self._beta_blend.x.array[:] = BETA1
        self._gamma_blend.x.array[:] = GAMMA1

        # grad(k)·grad(ω) for SST cross-diffusion and F1 computation
        grad_k_dot_grad_w_ufl = dot(grad(k_n), grad(scalar_n))
        self._grad_kw_expr = Expression(
            grad_k_dot_grad_w_ufl, S.element.interpolation_points
        )
        self._grad_kw_func = Function(S, name="grad_k_dot_grad_w")

        # Strain rate
        S_tensor = sym(grad(u_n))
        self._S_sq = 2.0 * inner(S_tensor, S_tensor)
        self._S_mag_expr = Expression(
            sqrt(self._S_sq + 1e-16), S.element.interpolation_points
        )

        # UFL sub-expressions for form coefficients
        beta_star_c = Constant(domain, PETSc.ScalarType(BETA_STAR))
        scalar_safe = ufl.max_value(scalar_prev, Constant(domain, PETSc.ScalarType(self._omega_min)))
        self._scalar_safe = scalar_safe

        # Production limiter (same as Wilcox)
        P_k_raw = nu_t_ * self._S_sq
        P_k_cap = 10.0 * beta_star_c * k_prev * scalar_safe
        self._prod_k = ufl.min_value(P_k_raw, P_k_cap)
        self._react_k = beta_star_c * scalar_safe

        # SST cross-diffusion: (1-F1) * 2*sigma_w2/omega * max(0, grad_k·grad_omega)
        grad_k_dot_grad_w = dot(grad(k_n), grad(scalar_prev))
        grad_kw_positive = ufl.conditional(
            ufl.gt(grad_k_dot_grad_w, 0.0), grad_k_dot_grad_w, 0.0
        )
        sigma_w2_c = Constant(domain, PETSc.ScalarType(2.0 * SIGMA_W2))
        self._cross_diff = (1.0 - self._F1) * sigma_w2_c / scalar_safe * grad_kw_positive

    def get_form_coefficients(self) -> FormCoefficients:
        return FormCoefficients(
            sigma_k=self._sigma_k_blend,
            sigma_phi=self._sigma_w_blend,
            nu_t_diff_k=self._nu_t_,
            nu_t_diff_phi=self._nu_t_,
            production_k=self._prod_k,
            reaction_k=self._react_k,
            production_phi=self._gamma_blend * self._S_sq,
            reaction_phi=self._beta_blend * self._scalar_safe,
            cross_diffusion=self._cross_diff,
        )

    def get_k_field_spec(self) -> FieldSpec:
        return FieldSpec(
            clip_min=self._turb.k_min,
            clip_max=self._turb.k_max,
            has_wall_dirichlet=True,
            wall_value=0.0,
        )

    def get_scalar_field_spec(self) -> FieldSpec:
        return FieldSpec(
            clip_min=self._omega_min,
            clip_max=10.0 * self._omega_wall_val,
            has_wall_dirichlet=True,
            wall_value=self._omega_wall_val,
        )

    def needs_wall_distance(self) -> bool:
        return True

    def compute_wall_distance(self, S, wall_facets, is_bfs, geom):
        # SST always uses Eikonal for wall distance (any geometry)
        y_wall = compute_wall_distance_eikonal(S, wall_facets)
        self._y_wall = y_wall
        return y_wall

    def update_auxiliary_fields(self, k_arr, scalar_arr):
        """Update F1, F2, and blended coefficients from current k, ω."""
        nu = self._nu
        k_safe = np.maximum(k_arr, self._k_min)
        omega_safe = np.maximum(scalar_arr, self._omega_min)
        y_arr = self._y_wall.x.array
        y_safe = np.maximum(y_arr, 1e-10)

        # Interpolate grad(k)·grad(ω) from current fields
        self._grad_kw_func.interpolate(self._grad_kw_expr)
        grad_kw_arr = self._grad_kw_func.x.array
        CD_kw = np.maximum(2.0 * SIGMA_W2 / omega_safe * grad_kw_arr, 1e-10)

        # F1: controls k-omega vs k-epsilon blending
        term1 = np.sqrt(k_safe) / (BETA_STAR * omega_safe * y_safe)
        term2 = 500.0 * nu / (y_safe**2 * omega_safe)
        term3 = 4.0 * SIGMA_W2 * k_safe / (CD_kw * y_safe**2)
        arg1 = np.minimum(np.maximum(term1, term2), term3)
        F1_arr = np.tanh(arg1**4)

        self._F1.x.array[:] = F1_arr
        self._F1.x.scatter_forward()

        # Blend coefficients: phi = F1*phi1 + (1-F1)*phi2
        self._sigma_k_blend.x.array[:] = F1_arr * SIGMA_K1 + (1.0 - F1_arr) * SIGMA_K2
        self._sigma_w_blend.x.array[:] = F1_arr * SIGMA_W1 + (1.0 - F1_arr) * SIGMA_W2
        self._beta_blend.x.array[:] = F1_arr * BETA1 + (1.0 - F1_arr) * BETA2
        self._gamma_blend.x.array[:] = F1_arr * GAMMA1 + (1.0 - F1_arr) * GAMMA2
        self._sigma_k_blend.x.scatter_forward()
        self._sigma_w_blend.x.scatter_forward()
        self._beta_blend.x.scatter_forward()
        self._gamma_blend.x.scatter_forward()

    def compute_nu_t(self, k_arr, scalar_arr, S_mag_arr):
        """SST ν_t limiter: ν_t = a₁·k / max(a₁·ω, |S|·F₂)."""
        k_safe = np.maximum(k_arr, self._k_min)
        omega_safe = np.maximum(scalar_arr, self._omega_min)
        S_mag_safe = np.maximum(S_mag_arr, 1e-10)

        # Compute F2 with updated k, omega
        nu = self._nu
        y_arr = self._y_wall.x.array
        y_safe = np.maximum(y_arr, 1e-10)

        term2a = 2.0 * np.sqrt(k_safe) / (BETA_STAR * omega_safe * y_safe)
        term2b = 500.0 * nu / (y_safe**2 * omega_safe)
        arg2 = np.maximum(term2a, term2b)
        F2_arr = np.tanh(arg2**2)

        self._F2.x.array[:] = F2_arr
        self._F2.x.scatter_forward()

        denominator = np.maximum(A1 * omega_safe, S_mag_safe * F2_arr)
        return A1 * k_safe / denominator

    def convert_omega_to_scalar_ic(self, omega_arr, k_arr):
        return omega_arr  # Identity for k-ω

    def initial_nu_t(self, k_arr, scalar_arr):
        return k_arr / (scalar_arr + self._omega_min)

    def compute_inlet_scalar(self, k_inlet, omega_inlet):
        return omega_inlet

    @property
    def S_mag_expr(self):
        return self._S_mag_expr
