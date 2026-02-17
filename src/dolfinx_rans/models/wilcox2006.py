"""
k-ω Wilcox 2006 turbulence model.

Reference: Wilcox, D.C. "Turbulence Modeling for CFD", 3rd ed.,
           DCW Industries, 2006.
"""

import numpy as np
from petsc4py import PETSc

from dolfinx.fem import Constant

import ufl
from ufl import dot, grad, inner, sqrt, sym

from dolfinx_rans.config import BETA_0, KAPPA
from dolfinx_rans.models.base import FieldSpec, FormCoefficients, RANSModel


# Wilcox 2006 constants
BETA_STAR = 0.09
SIGMA_K = 0.6
SIGMA_W = 0.5
GAMMA = BETA_0 / BETA_STAR - SIGMA_W * KAPPA**2 / np.sqrt(BETA_STAR)
SIGMA_D0 = 0.125  # Cross-diffusion coefficient (1/8)
C_LIM = 0.875  # Stress limiter constant (7/8)
SQRT_BETA_STAR = np.sqrt(BETA_STAR)


class Wilcox2006Model(RANSModel):
    """k-ω (Wilcox 2006) with stress limiter and cross-diffusion."""

    @property
    def model_name(self) -> str:
        return "wilcox2006"

    @property
    def display_name(self) -> str:
        return "k-ω (Wilcox 2006)"

    @property
    def scalar_name(self) -> str:
        return "ω"

    def setup(self, domain, S, k_prev, scalar_prev, k_n, scalar_n,
              nu_t_, u_n, nu_c, nu, turb_params, wall_facets, is_bfs, geom,
              y_first=0.0):
        self._domain = domain
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
        self._C_lim = turb_params.C_lim

        # Compute omega wall value for Dirichlet BC
        self._y_first = y_first
        self._omega_wall_val = 6.0 * nu / (BETA_0 * y_first**2)

        # Precompute UFL sub-expressions used in form coefficients
        self._beta_star_c = Constant(domain, PETSc.ScalarType(BETA_STAR))
        self._sigma_d_c = Constant(domain, PETSc.ScalarType(SIGMA_D0))

        S_tensor = sym(grad(u_n))
        self._S_sq = 2.0 * inner(S_tensor, S_tensor)

        # S_magnitude expression for interpolation
        from dolfinx.fem import Expression
        self._S_mag_expr = Expression(
            sqrt(self._S_sq + 1e-16), S.element.interpolation_points
        )

        # Production limiter: P_k <= 10*beta_star*k*omega
        scalar_safe = ufl.max_value(scalar_prev, Constant(domain, PETSc.ScalarType(self._omega_min)))
        P_k_raw = nu_t_ * self._S_sq
        P_k_cap = 10.0 * self._beta_star_c * k_prev * scalar_safe
        self._prod_k = ufl.min_value(P_k_raw, P_k_cap)
        self._react_k = self._beta_star_c * scalar_safe
        self._scalar_safe = scalar_safe

        # Cross-diffusion: sigma_d/omega * max(0, grad_k · grad_omega)
        grad_k_dot_grad_w = dot(grad(k_n), grad(scalar_prev))
        grad_kw_positive = ufl.conditional(
            ufl.gt(grad_k_dot_grad_w, 0.0), grad_k_dot_grad_w, 0.0
        )
        self._cross_diff = self._sigma_d_c / self._scalar_safe * grad_kw_positive

    def get_form_coefficients(self) -> FormCoefficients:
        d = self._domain
        return FormCoefficients(
            sigma_k=Constant(d, PETSc.ScalarType(SIGMA_K)),
            sigma_phi=Constant(d, PETSc.ScalarType(SIGMA_W)),
            nu_t_diff_k=self._nu_t_,
            nu_t_diff_phi=self._nu_t_,
            production_k=self._prod_k,
            reaction_k=self._react_k,
            production_phi=Constant(d, PETSc.ScalarType(GAMMA)) * self._S_sq,
            reaction_phi=Constant(d, PETSc.ScalarType(BETA_0)) * self._scalar_safe,
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
        return False

    def compute_wall_distance(self, S, wall_facets, is_bfs, geom):
        return None

    def update_auxiliary_fields(self, k_arr, scalar_arr):
        pass  # No blending functions

    def compute_nu_t(self, k_arr, scalar_arr, S_mag_arr):
        S_mag_safe = np.maximum(S_mag_arr, 1e-10)
        omega_tilde = np.maximum(
            scalar_arr,
            self._C_lim * S_mag_safe / SQRT_BETA_STAR,
        )
        omega_tilde = np.maximum(omega_tilde, self._omega_min)
        return k_arr / omega_tilde

    def convert_omega_to_scalar_ic(self, omega_arr, k_arr):
        return omega_arr  # Identity for k-ω

    def initial_nu_t(self, k_arr, scalar_arr):
        return k_arr / (scalar_arr + self._omega_min)

    def compute_inlet_scalar(self, k_inlet, omega_inlet):
        return omega_inlet

    @property
    def S_mag_expr(self):
        """Expression for |S| = sqrt(2·S_ij·S_ij), for solver interpolation."""
        return self._S_mag_expr
