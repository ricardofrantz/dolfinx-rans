"""
k-ε Lam-Bremhorst low-Reynolds-number turbulence model.

Reference: Lam, C.K.G. and Bremhorst, K. "A modified form of the
           k-ε model for predicting wall turbulence." ASME Journal
           of Fluids Engineering, 103(3), 1981.

Implementation follows joove123/k-epsilon (TurbulenceModel.py).
"""

import numpy as np
from petsc4py import PETSc

from dolfinx.fem import Constant, Expression

import ufl
from ufl import grad, inner, sqrt, sym

from dolfinx_rans.geometry import compute_wall_distance_channel, compute_wall_distance_eikonal
from dolfinx_rans.models.base import FieldSpec, FormCoefficients, RANSModel


# Lam-Bremhorst constants (joove123/k-epsilon/TurbulenceModel.py)
A1_LB = 0.0165
A2_LB = 20.5


class KepsilonModel(RANSModel):
    """k-ε (Lam-Bremhorst) with low-Re damping functions."""

    @property
    def model_name(self) -> str:
        return "kepsilon"

    @property
    def display_name(self) -> str:
        return "k-ε (Lam-Bremhorst)"

    @property
    def scalar_name(self) -> str:
        return "ε"

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
        self._wall_facets = wall_facets
        self._is_bfs = is_bfs
        self._geom = geom

        self._C_mu = turb_params.C_mu
        self._k_min = turb_params.k_min
        self._eps_min = turb_params.epsilon_min

        # Safe UFL expressions for k_prev and eps_prev
        k_prev_safe = ufl.max_value(k_prev, Constant(domain, PETSc.ScalarType(turb_params.k_min)))
        scalar_prev_safe = ufl.max_value(scalar_prev, Constant(domain, PETSc.ScalarType(turb_params.epsilon_min)))
        self._k_prev_safe = k_prev_safe
        self._scalar_prev_safe = scalar_prev_safe

        # Strain rate
        S_tensor = sym(grad(u_n))
        self._S_sq = 2.0 * inner(S_tensor, S_tensor)
        self._S_mag_expr = Expression(
            sqrt(self._S_sq + 1e-16), S.element.interpolation_points
        )

        # Damping functions (UFL expressions for use in forms)
        # y_wall will be set after compute_wall_distance() is called
        self._y_wall = None  # Placeholder

    def _build_ufl_expressions(self):
        """Build UFL damping-function expressions after y_wall is available."""
        domain = self._domain
        k_prev_safe = self._k_prev_safe
        scalar_prev_safe = self._scalar_prev_safe
        y_wall = self._y_wall
        nu_c = self._nu_c
        turb = self._turb
        k_min = self._k_min
        eps_min = self._eps_min

        keps_C_mu = Constant(domain, PETSc.ScalarType(turb.C_mu))
        keps_C1 = Constant(domain, PETSc.ScalarType(turb.C_epsilon_1))
        keps_C2 = Constant(domain, PETSc.ScalarType(turb.C_epsilon_2))
        keps_f_nu_min = Constant(domain, PETSc.ScalarType(turb.f_nu_min))

        y_wall_safe = ufl.max_value(y_wall, Constant(domain, PETSc.ScalarType(1e-10))) if y_wall is not None else Constant(domain, PETSc.ScalarType(1.0))

        # Re_t = k²/(ν·ε)
        Re_t = (k_prev_safe * k_prev_safe) / (nu_c * scalar_prev_safe + 1e-16)
        # Re_k = sqrt(k)·y/ν
        Re_k_expr = ufl.sqrt(k_prev_safe) * y_wall_safe / (nu_c + 1e-16)

        # f_ν = (1 - exp(-A1·Re_k))² · (1 + A2/Re_t)
        f_nu = (1.0 - ufl.exp(-A1_LB * Re_k_expr)) ** 2
        f_nu = f_nu * (1.0 + A2_LB / (Re_t + 1e-16))
        f_nu = ufl.min_value(ufl.max_value(f_nu, keps_f_nu_min), Constant(domain, PETSc.ScalarType(1.0)))

        # ν_t UFL expression (for diffusion in forms)
        nu_t_expr = keps_C_mu * f_nu * (k_prev_safe * k_prev_safe) / (scalar_prev_safe + 1e-16)
        self._nu_t_ufl_expr = nu_t_expr

        # Production: P_k = ν_t · S² (no limiter for k-ε)
        prod_k = nu_t_expr * self._S_sq

        # Reaction for k: ε/k
        gamma_expr = scalar_prev_safe / k_prev_safe
        react_k = gamma_expr

        # f₁ = 1 + (0.05/f_ν)³
        f_1 = 1.0 + (0.05 / (f_nu + 1e-16)) ** 3
        # f₂ = max(min(1 - exp(-Re_t²), 1), 0)
        f_2 = ufl.max_value(
            ufl.min_value(1.0 - ufl.exp(-Re_t**2), Constant(domain, PETSc.ScalarType(1.0))),
            Constant(domain, PETSc.ScalarType(0.0)),
        )

        # ε production: C₁·f₁·(ε/k)·ν_t·S²
        prod_e = keps_C1 * f_1 * prod_k * gamma_expr
        # ε reaction: C₂·f₂·(ε/k)
        react_e = keps_C2 * f_2 * gamma_expr

        self._prod_k = prod_k
        self._react_k = react_k
        self._prod_e = prod_e
        self._react_e = react_e

        # Store constants for form coefficients
        self._sigma_k_c = Constant(domain, PETSc.ScalarType(turb.sigma_k_kepsilon))
        self._sigma_eps_c = Constant(domain, PETSc.ScalarType(turb.sigma_epsilon))

    def get_form_coefficients(self) -> FormCoefficients:
        return FormCoefficients(
            sigma_k=self._sigma_k_c,
            sigma_phi=self._sigma_eps_c,
            nu_t_diff_k=self._nu_t_ufl_expr,
            nu_t_diff_phi=self._nu_t_ufl_expr,
            production_k=self._prod_k,
            reaction_k=self._react_k,
            production_phi=self._prod_e,
            reaction_phi=self._react_e,
            cross_diffusion=Constant(self._domain, PETSc.ScalarType(0.0)),
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
            clip_min=self._turb.epsilon_min,
            clip_max=self._turb.epsilon_max,
            has_wall_dirichlet=False,  # No wall Dirichlet for ε
        )

    def needs_wall_distance(self) -> bool:
        return True

    def compute_wall_distance(self, S, wall_facets, is_bfs, geom):
        if is_bfs:
            y_wall = compute_wall_distance_eikonal(S, wall_facets)
        else:
            y_wall = compute_wall_distance_channel(S, geom.use_symmetry)
        self._y_wall = y_wall
        # Now that y_wall is available, build UFL expressions
        self._build_ufl_expressions()
        return y_wall

    def update_auxiliary_fields(self, k_arr, scalar_arr):
        pass  # Damping functions are UFL expressions evaluated in the form

    def compute_nu_t(self, k_arr, scalar_arr, S_mag_arr):
        """Lam-Bremhorst ν_t = C_μ·f_ν·k²/ε."""
        tiny = 1e-16
        k_safe = np.maximum(k_arr, self._k_min)
        eps_safe = np.maximum(scalar_arr, self._eps_min)
        nu = self._nu

        y_arr = self._y_wall.x.array if self._y_wall is not None else np.ones_like(k_arr)
        y_safe = np.maximum(y_arr, 1e-10)

        Re_t_arr = (k_safe * k_safe) / (nu * eps_safe + tiny)
        Re_k_arr = np.sqrt(k_safe) * y_safe / (nu + tiny)
        f_nu_arr = (1.0 - np.exp(-A1_LB * Re_k_arr)) ** 2
        f_nu_arr = f_nu_arr * (1.0 + A2_LB / (Re_t_arr + tiny))
        f_nu_arr = np.clip(f_nu_arr, self._turb.f_nu_min, 1.0)

        return (self._C_mu * f_nu_arr * (k_safe * k_safe)) / (eps_safe + tiny)

    def convert_omega_to_scalar_ic(self, omega_arr, k_arr):
        """Convert ω → ε = C_μ · k · ω."""
        k_safe = np.maximum(k_arr, self._k_min)
        omega_safe = np.maximum(omega_arr, self._turb.omega_min)
        return np.maximum(self._C_mu * k_safe * omega_safe, self._eps_min)

    def initial_nu_t(self, k_arr, scalar_arr):
        """Initial ν_t = C_μ · k² / ε."""
        return self._C_mu * k_arr**2 / np.maximum(scalar_arr, self._eps_min)

    def compute_inlet_scalar(self, k_inlet, omega_inlet):
        """Convert (k, ω) inlet → ε = C_μ · k · ω."""
        return max(self._C_mu * k_inlet * omega_inlet, self._eps_min)

    @property
    def S_mag_expr(self):
        return self._S_mag_expr
