"""
Configuration dataclasses for dolfinx-rans.

Contains:
- Shared turbulence constants (BETA_0, KAPPA) used by geometry.py
- Geometry dataclasses (ChannelGeom, BFSGeom)
- Flow/turbulence/solver parameter dataclasses
- BoundaryInfo container for tagged facet arrays

Model-specific constants live in dolfinx_rans/models/*.py.
"""

from dataclasses import dataclass


# Shared turbulence constants used by geometry.py (omega wall IC, log law)
BETA_0 = 0.0708  # Base omega destruction coefficient (Wilcox 2006)
KAPPA = 0.41  # von Karman constant


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass(frozen=True)
class ChannelGeom:
    """Channel geometry parameters."""

    Lx: float  # Channel length (streamwise)
    Ly: float  # Channel height (delta if use_symmetry, else 2*delta)
    Nx: int  # Mesh cells in x
    Ny: int  # Mesh cells in y
    mesh_type: str  # "triangle" or "quad"
    y_first: float  # First cell height from wall (for y+ control)
    growth_rate: float  # Geometric stretching ratio (>1 for wall refinement)
    stretching: str = "geometric"  # "geometric" or "tanh"
    y_first_tol_rel: float = 0.05  # Hard-fail if implied y_first differs by more than this
    use_symmetry: bool = True  # Half-channel with symmetry BC at top (default: True)


@dataclass(frozen=True)
class BFSGeom:
    """Backward-facing step geometry parameters."""

    step_height: float  # h (reference length)
    expansion_ratio: float  # H_outlet / H_inlet (typically 2.0)
    upstream_length: float  # Length before step (multiples of h)
    downstream_length: float  # Length after step (multiples of h)
    y_first: float  # First cell height at walls
    Nx_upstream: int  # Mesh cells upstream
    Nx_downstream: int  # Mesh cells downstream
    Ny_inlet: int  # Mesh cells in inlet height
    Ny_outlet: int  # Mesh cells in outlet height
    mesh_type: str = "quad"
    stretching: str = "geometric"
    growth_rate: float = 1.15


@dataclass(frozen=True)
class NondimParams:
    """Nondimensional parameters for Re_tau-based scaling."""

    Re_tau: float  # Friction Reynolds number
    use_body_force: bool = True  # f_x = 1 to drive flow
    # BFS inlet overrides (0 = auto-compute from turbulence intensity)
    U_inlet: float = 0.0  # Inlet velocity magnitude (0 = use 1.0)
    k_inlet: float = 0.0  # Inlet TKE (0 = 1.5 * (0.05 * U_inlet)^2)
    epsilon_inlet: float = 0.0  # Inlet epsilon (0 = derived from k, omega)


@dataclass(frozen=True)
class FlowParams:
    """
    Flow parameters supporting multiple Reynolds number definitions.

    Re: Reynolds number (Re_tau for channel, Re_h for BFS)
    Re_type: "Re_tau" or "Re_h"
    U_inlet: Inlet bulk velocity (1.0 for nondimensional)
    use_body_force: Only for channel with periodic-like driving
    """

    Re: float  # Reynolds number
    Re_type: str = "Re_tau"  # "Re_tau" or "Re_h"
    U_inlet: float = 1.0  # Inlet bulk velocity
    use_body_force: bool = False  # Body force driving (channel only)


@dataclass(frozen=True)
class TurbParams:
    """
    Turbulence model parameters (all required in config).

    model: Turbulence model - "wilcox2006" or "sst" (default: wilcox2006)
    beta_star: k-omega model constant (standard: 0.09)
    nu_t_max_factor: Max nu_t/nu ratio for stability
    omega_min: Floor on omega to prevent nu_t runaway (10 = best for U+)
    k_min: Floor on k for positivity (1e-10)
    k_max: Cap on k for safety
    C_lim: Durbin realizability nu_t <= C_lim*k/(sqrt(6)*|S|) (0 = disabled, SST uses internal limiter)
    """

    beta_star: float
    nu_t_max_factor: float
    omega_min: float
    k_min: float
    k_max: float
    C_lim: float
    # Supported models: "wilcox2006", "sst", "kepsilon"
    model: str = "wilcox2006"
    C_mu: float = 0.09  # Kepsilon C_mu constant
    sigma_k_kepsilon: float = 1.0  # ε-model diffusion coefficient sigma_k
    sigma_epsilon: float = 1.3  # ε-model diffusion coefficient sigma_ε
    C_epsilon_1: float = 1.44  # Kepsilon C1 constant
    C_epsilon_2: float = 1.92  # Kepsilon C2 constant
    epsilon_min: float = 1e-16  # ε floor
    epsilon_max: float = 1e8  # ε upper guard
    f_nu_min: float = 0.01116225  # Lam-Bremhorst f_nu lower floor


@dataclass(frozen=True)
class SolveParams:
    """
    Solver parameters (all required in config).

    dt: Initial pseudo-time step
    dt_max: Maximum dt for implicit stepping
    dt_growth: dt multiplier when converging well
    dt_growth_threshold: Only grow dt if residual_ratio < threshold (hysteresis)
    t_final: Max pseudo-time (safety limit, usually not reached)
    max_iter: Max iterations before giving up
    steady_tol: Convergence tolerance on velocity residual
    cfl_target: Global target CFL (cfl = u_max*dt/h_min), defaults to 1.0 when omitted
    enable_physical_convergence: Also require physical metric changes to settle
    physical_u_bulk_rel_tol: Relative tolerance for successive U_bulk changes
    physical_tau_wall_rel_tol: Relative tolerance for successive tau_wall changes
    physical_convergence_start_iter: Start physical checks after this outer step
    picard_max: Inner Picard iterations per time step
    picard_tol: Picard convergence tolerance
    under_relax_k_omega: Under-relaxation for k and omega (0.7 typical)
    under_relax_nu_t: Under-relaxation for nu_t (0.5 typical)
    log_interval: Print every N iterations
    snapshot_interval: Save VTX every N iterations (0 = disabled)
    out_dir: Output directory for results
    min_iter: Minimum outer iterations before steady convergence check
    min_dt_ratio: Minimum dt ratio to initial dt required to accept convergence
    """

    dt: float
    dt_max: float
    dt_growth: float
    dt_growth_threshold: float
    t_final: float
    max_iter: int
    steady_tol: float
    picard_max: int
    picard_tol: float
    under_relax_k_omega: float
    under_relax_nu_t: float
    log_interval: int
    snapshot_interval: int
    out_dir: str
    cfl_target: float = 1.0
    enable_physical_convergence: bool = False
    physical_u_bulk_rel_tol: float = 1e-6
    physical_tau_wall_rel_tol: float = 1e-6
    physical_convergence_start_iter: int = 10
    min_iter: int = 1
    min_dt_ratio: float = 0.0


class BoundaryInfo:
    """
    Container for tagged boundary facets.

    Holds facet index arrays for each boundary type. Not all fields
    are populated for every geometry — e.g., channel has no step_facets,
    BFS has no symmetry_facets.
    """

    def __init__(
        self,
        wall_facets,
        inlet_facets=None,
        outlet_facets=None,
        symmetry_facets=None,
        step_facets=None,
        bottom_facets=None,
        top_facets=None,
        left_facets=None,
        right_facets=None,
    ):
        self.wall_facets = wall_facets
        self.inlet_facets = inlet_facets
        self.outlet_facets = outlet_facets
        self.symmetry_facets = symmetry_facets
        self.step_facets = step_facets
        self.bottom_facets = bottom_facets
        self.top_facets = top_facets
        self.left_facets = left_facets
        self.right_facets = right_facets
