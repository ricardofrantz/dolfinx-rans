# dolfinx-rans

RANS turbulence models for DOLFINx — a FEniCSx-based implementation of Reynolds-Averaged Navier-Stokes solvers.

## Features

- **Standard k-ω model** with Wilcox (1998) constants
- **Pseudo-transient continuation** to steady state
- **Adaptive time stepping** with hysteresis to prevent oscillation
- **Wall-refined mesh** with geometric stretching for proper y+ control
- **Optional Durbin realizability limiter** for stagnation regions
- **Periodic boundary conditions** via dolfinx_mpc
- **MKM DNS validation data** (Re_τ = 180, 590) included

## Requirements

- DOLFINx 0.10.0+
- dolfinx_mpc (for periodic BCs)
- numpy, matplotlib

```bash
# Install via conda (recommended)
conda install -c conda-forge fenics-dolfinx dolfinx_mpc

# Or pip (if fenics is available)
pip install dolfinx-rans
```

## Quick Start

```bash
# Run solver with config file
dolfinx-rans examples/channel_re590.json

# Or as Python module
python -m dolfinx_rans examples/channel_re590.json

# Quick smoke test (fewer iterations)
dolfinx-rans examples/channel_smoke.json
```

## Configuration

All parameters are **required** in the JSON config file. No hardcoded defaults.

### `geom` — Channel Geometry

| Parameter | Description |
|-----------|-------------|
| `Lx` | Channel length (streamwise) |
| `Ly` | Channel height (2δ where δ = half-height) |
| `Nx` | Mesh cells in x |
| `Ny` | Mesh cells in y |
| `mesh_type` | `"triangle"` or `"quad"` |
| `y_first` | First cell height. For low-Re BCs: y+ = y_first × Re_τ < 2.5 |
| `growth_rate` | Geometric stretching (>1 for wall refinement, 1 for uniform) |

**Wall Refinement:**
```
For Re_τ = 590, need y+ < 2.5:
    y_first < 2.5 / 590 = 0.00424

With y_first = 0.002 and growth_rate = 1.1:
    y+ ≈ 1.18 ✓
```

### `nondim` — Nondimensional Scaling

| Parameter | Description |
|-----------|-------------|
| `Re_tau` | Friction Reynolds number |
| `use_body_force` | `true` = body force f_x=1, periodic in x |

### `turb` — Turbulence Model

| Parameter | Description |
|-----------|-------------|
| `beta_star` | k-ω constant (standard: 0.09) |
| `nu_t_max_factor` | Max ν_t/ν ratio for stability |
| `omega_min` | ω floor to prevent ν_t runaway. **10 = best U+** |
| `k_min` | k floor for positivity (1e-10) |
| `k_max` | k cap for safety |
| `C_lim` | Durbin limiter ν_t ≤ C_lim·k/(√6·\|S\|). **0 = disabled** |

### `solve` — Solver Parameters

| Parameter | Description |
|-----------|-------------|
| `dt` | Initial pseudo-time step |
| `dt_max` | Maximum dt |
| `dt_growth` | dt multiplier when converging |
| `dt_growth_threshold` | Hysteresis threshold for dt growth |
| `t_final` | Max pseudo-time (safety limit) |
| `max_iter` | Max iterations |
| `steady_tol` | Convergence tolerance |
| `picard_max` | Inner Picard iterations per step |
| `picard_tol` | Picard convergence tolerance |
| `under_relax_k_omega` | Under-relaxation for k, ω (0.7 typical) |
| `under_relax_nu_t` | Under-relaxation for ν_t (0.5 typical) |
| `log_interval` | Print every N iterations |
| `snapshot_interval` | Save VTX every N iterations (0 = disabled) |
| `out_dir` | Output directory |

## Validation

Validated against MKM DNS (Moser, Kim, Mansour 1999) at Re_τ = 590:

| Metric | RANS k-ω | DNS | Notes |
|--------|----------|-----|-------|
| U_c+ | ~20-24 | 23.56 | ~±10% |
| k+_max | ~1-2 | 3.13 | Model limitation |

**Known limitation:** Standard k-ω under-predicts TKE in channel core. This is well-documented in turbulence modeling literature. SST k-ω would improve k prediction.

## Python API

```python
from dolfinx_rans import (
    ChannelGeom, NondimParams, TurbParams, SolveParams,
    create_channel_mesh, solve_rans_kw
)
from dolfinx_rans.validation import get_k_profile_590, MEAN_VELOCITY_590

# Create mesh
geom = ChannelGeom(Lx=6.28, Ly=2.0, Nx=48, Ny=64,
                   mesh_type="triangle", y_first=0.002, growth_rate=1.1)
domain = create_channel_mesh(geom, Re_tau=590)

# Run solver
u, p, k, omega, nu_t, V, Q, S, domain, step, t = solve_rans_kw(
    domain, geom, turb, solve, results_dir, nondim
)

# Compare to DNS
y_plus_dns, k_plus_dns = get_k_profile_590()
```

## Output

Results saved to `out_dir`:
- `flow_fields.png` — Contour plots
- `history.csv` — Convergence history
- `config_used.json` — Config snapshot
- `run_info.json` — Environment metadata
- `snps/*.bp` — VTX time series (ParaView)

## Governing Equations

Nondimensional RANS with δ = half-height, u_τ = friction velocity:

**Momentum:**
```
∂u/∂t + (u·∇)u = -∇p + ∇·[(ν* + ν_t)∇u] + f_x
where ν* = 1/Re_τ, f_x = 1
```

**k-equation:**
```
∂k/∂t + u·∇k = P_k - β*·k·ω + ∇·[(ν* + σ_k·ν_t)∇k]
```

**ω-equation:**
```
∂ω/∂t + u·∇ω = γ·S² - β·ω² + ∇·[(ν* + σ_ω·ν_t)∇ω]
```

**Wall BCs:**
- k = 0
- ω = 6ν*/(β·y_first²)

## References

1. Moser, Kim, Mansour (1999) "DNS of turbulent channel flow up to Re_τ=590" *Phys. Fluids* 11(4):943-945
2. Wilcox (1998) *Turbulence Modeling for CFD*, 2nd ed.
3. Durbin (1996) "On the k-ε stagnation point anomaly" *Int. J. Heat Fluid Flow* 17:89-90

## License

MIT License. See [LICENSE](LICENSE).
