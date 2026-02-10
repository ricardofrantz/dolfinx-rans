# dolfinx-rans

**RANS k-ω turbulence model solver for incompressible CFD using DOLFINx/FEniCSx**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOLFINx](https://img.shields.io/badge/DOLFINx-0.10.0+-green.svg)](https://fenicsproject.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg)](https://python.org/)

A finite element implementation of the standard k-ω two-equation turbulence model for solving Reynolds-Averaged Navier-Stokes (RANS) equations. Primary validation workflow is high-Re RANS benchmarking (Nek5000-style channel setup), with Re_τ=590 DNS retained as a legacy secondary check.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Validation](#validation)
- [Python API](#python-api)
- [Output](#output)
- [Background](#background)
  - [Why k-ω?](#why-k-ω)
  - [FEM Challenges](#challenges-of-rans-in-finite-elements)
  - [Governing Equations](#governing-equations)
- [References](#references)

---

## Quick Start

```bash
# Activate FEniCSx environment
conda activate fenicsx

# Primary high-Re benchmark case (Nek-like setup)
dolfinx-rans examples/channel_nek_re125k_like.json

# Legacy low-Re case (secondary check only)
dolfinx-rans examples/channel_re590.json
```

## Features

- **Standard k-ω model** with Wilcox (1998) constants
- **Pseudo-transient continuation** to steady state
- **Adaptive time stepping** with hysteresis to prevent oscillation
- **Wall-refined mesh** with geometric stretching for proper y⁺ control
- **Optional Durbin realizability limiter** for stagnation regions
- **Periodic boundary conditions** via dolfinx_mpc
- **Profile CSV export** for cross-code benchmark comparison (`profiles.csv`)
- **Config-driven regression gates** (bulk, wall shear, optional profile RMSE)

```
  Turbulent Channel Flow Setup (nondimensional)

                y
                ^
  y = 2δ        |   top wall: u = 0, k = 0, ω = ω_wall
                |  ┌──────────────────────────────────────────────┐
  x = 0         |  │  →  →  →  →  →  →  →  →  →  →  →  →  →      │   x = Lx
  (periodic)    |  │                                              │   (periodic)
      ↔         |  │  ------------- centerline: y = δ ---------   │      ↔
                |  │  →  →  →  →  →  →  →  →  →  →  →  →  →      │
  y = 0         |  └──────────────────────────────────────────────┘
                |   bottom wall: u = 0, k = 0, ω = ω_wall
                +------------------------------------------------------> x

              streamwise driving term: f_x = +1
```

## Installation

DOLFINx and dolfinx_mpc require conda-forge (not pip-installable). After setting up the FEniCSx environment, install dolfinx-rans with uv or pip.

### Step 1: Create FEniCSx Environment (conda)

```bash
# Create dedicated environment with DOLFINx 0.10.0+
conda create -n fenicsx python=3.11
conda activate fenicsx
conda install -c conda-forge fenics-dolfinx dolfinx_mpc petsc4py mpi4py
```

### Step 2: Install dolfinx-rans (uv)

```bash
# Activate your FEniCSx environment first
conda activate fenicsx

# Install dolfinx-rans (editable for development)
cd dolfinx-rans
uv pip install -e .

# Or install directly from GitHub
uv pip install git+https://github.com/ricardofrantz/dolfinx-rans.git
```

### Alternative: pip install

```bash
conda activate fenicsx
pip install -e .
# or
pip install git+https://github.com/ricardofrantz/dolfinx-rans.git
```

### Requirements Summary

| Package | Source | Notes |
|---------|--------|-------|
| DOLFINx 0.10.0+ | conda-forge | FEniCSx finite element library |
| dolfinx_mpc | conda-forge | Multi-point constraints (periodic BCs) |
| petsc4py | conda-forge | PETSc linear solvers |
| mpi4py | conda-forge | MPI parallelization |
| numpy | pip/conda | Numerical arrays |
| matplotlib | pip/conda | Plotting |

<details>
<summary><strong>Why dolfinx_mpc? (click to expand)</strong></summary>

For turbulent channel flow validation, we need **periodic boundary conditions** in the streamwise direction: `u(x=0, y) = u(x=Lx, y)`.

**The problem:** Standard FEM boundary conditions can only set values (Dirichlet) or flux (Neumann) at boundaries — neither can express "right boundary equals left boundary", which is a constraint between DOFs.

**How dolfinx_mpc solves it:**

```
Standard FEM:                  With MPC:
[K]{u} = {f}                   [K']{u'} = {f'}

DOFs: [u₀, u₁, ..., uₙ]        DOFs: [u₀, u₁, ..., uₘ]  where m < n

                               Right-boundary DOFs eliminated and
                               replaced by left-boundary DOFs
```

MPC modifies the stiffness matrix to enforce `u_right = u_left` as algebraic constraints, effectively reducing the system size.

**Why periodic + body force instead of inlet/outlet?**

| Approach | Pros/Cons |
|----------|-----------|
| **Inlet/outlet BCs** | ✗ Unknown inlet profile, entrance effects, long domain needed |
| **Periodic + body force** | ✓ Body force f_x = 1 drives fully developed channel, short domain sufficient, clean cross-code RANS comparison |

</details>

## Configuration

Core sections (`geom`, `nondim`, `turb`, `solve`) are required. `benchmark` is optional and enables regression/comparison gates.

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
| `y_first_tol_rel` | Relative tolerance for mesh/BC consistency check (hard fail if exceeded) |

**Wall Refinement:**
```
For Re_τ = 5200 (high-Re wall-resolved target), need y+ < 2.5:
    y_first < 2.5 / 5200 = 4.81e-4

With y_first ≈ 4.66e-4:
    y+ ≈ 2.43 ✓
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

### `benchmark` — Optional Regression/Comparison Gates

| Parameter | Description |
|-----------|-------------|
| `gate_u_bulk_bounds` | `[min, max]` gate for `U_bulk` from last history row |
| `gate_tau_wall_bounds` | `[min, max]` gate for wall shear `τ_wall` |
| `reference_profile_csv` | Optional Nek reference CSV (columns: `y_plus,u_plus[,k_plus]`) |
| `u_plus_rmse_max` | Optional max RMSE for `u_plus` vs reference profile |
| `k_plus_rmse_max` | Optional max RMSE for `k_plus` vs reference profile |

## Validation

Primary target is **high-Re RANS benchmarking**, aligned with Nek5000-style channel setups, not low-Re DNS matching.

Recommended workflow:
1. Run this solver with `examples/channel_nek_re125k_like.json`.
2. Run Nek5000 RANS channel tutorial case (or your house RANS baseline) at matching conditions.
3. Export/reference a profile CSV with `y_plus,u_plus[,k_plus]`.
4. Set `benchmark.reference_profile_csv` and RMSE thresholds in your config.
5. Re-run with `./run.sh` and use the built-in regression gate output.

Notes:
- This solver now writes `profiles.csv` for every run.
- Legacy Re_τ=590/DNS comparisons are still possible via `examples/channel_re590.json`, but are secondary checks.

## Python API

```python
from dolfinx_rans import (
    ChannelGeom, NondimParams, TurbParams, SolveParams,
    create_channel_mesh, solve_rans_kw
)

# Create mesh (high-Re Nek-like benchmark setup)
geom = ChannelGeom(
    Lx=1.0, Ly=1.0, Nx=96, Ny=96,
    mesh_type="quad", y_first=0.000466477, growth_rate=1.05
)
domain = create_channel_mesh(geom, Re_tau=5200)

# Run solver
u, p, k, omega, nu_t, V, Q, S, domain, step, t = solve_rans_kw(
    domain, geom, turb, solve, results_dir, nondim
)
```

## Output

Results saved to `out_dir`:
- `final_fields.png` — Contour plots and profiles
- `history.csv` — Convergence history
- `profiles.csv` — Wall-normal benchmark profile (`y_plus,u_plus,k_plus,...`)
- `config_used.json` — Config snapshot
- `run_info.json` — Environment metadata
- `snps/*.bp` — VTX time series (ParaView)

---

## Background

### Why k-ω?

RANS turbulence models solve time-averaged equations with a turbulent viscosity ν_t to model Reynolds stresses. The main two-equation models are:

| Model | Solves for | Wall behavior |
|-------|------------|---------------|
| k-ε | k (TKE), ε (dissipation) | Needs wall functions or damping |
| **k-ω** | k (TKE), ω (specific dissipation) | Natural wall BC: ω → 6ν/(βy²) |
| k-ω SST | Blends k-ω (wall) with k-ε (freestream) | Best of both worlds |

**We chose k-ω because:**

1. **Simple wall boundary conditions** — ω has a known asymptotic behavior at walls:
   ```
   ω_wall = 6ν / (β·y²)
   ```
   No wall functions or low-Re damping functions needed.

2. **Robust near-wall behavior** — k-ω is well-behaved in the viscous sublayer, unlike k-ε which has singularities.

3. **Educational clarity** — Standard k-ω is simpler than SST (no blending functions), making the code easier to understand and extend.

**Limitation:** Standard k-ω is sensitive to freestream ω values and under-predicts TKE in the channel core. SST addresses this but adds complexity.

**Further reading:**
- [CFD-Wiki: SST k-omega model](https://www.cfd-online.com/Wiki/SST_k-omega_model) — comprehensive overview of k-ω variants
- Menter (1994) "Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications" *AIAA J.* 32(8):1598-1605 — SST model derivation

### Challenges of RANS in Finite Elements

Most RANS solvers use **finite volume methods** (FVM), which naturally suit conservation laws. FEM offers advantages — higher-order accuracy, complex geometries, rigorous error analysis — but faces specific challenges with turbulence:

#### 1. Positivity — FEM doesn't respect physical bounds

**The problem:** k and ω must always be positive (they represent energy and frequency). In FVM, upwind schemes naturally preserve positivity. In FEM, the solution is a weighted sum of basis functions — and that sum can go negative near steep gradients.

**Why it crashes:** Negative k or ω → ν_t = k/ω blows up or flips sign → divergence.

```python
# Our solution: Hard clipping after each solve
k_.x.array[:] = np.clip(k_.x.array, k_min, k_max)
omega_.x.array[:] = np.maximum(omega_.x.array, omega_min)
```

#### 2. Convection-dominated transport — Galerkin oscillates

**The problem:** At high Re, k and ω transport is dominated by convection (u·∇k), not diffusion. Standard Galerkin FEM is centered (like central differencing) — it produces spurious wiggles near sharp gradients.

**Why FVM handles this better:** FVM naturally uses upwind schemes that add stabilizing diffusion in the flow direction.

**FEM workarounds:**
- **SUPG stabilization** — artificial streamline diffusion (complex to tune)
- **Fine mesh** — keep local Péclet number reasonable (our approach)
- **Pseudo-transient** — time-stepping adds implicit numerical diffusion

#### 3. Stiff nonlinear coupling — ν_t feedback loop

**The problem:** ν_t = k/ω couples all equations. Small changes in ω cause large swings in ν_t → changes diffusion everywhere → changes k, ω → runaway feedback.

```
ω ↑  →  ν_t ↓  →  less diffusion  →  steeper gradients  →  more production  →  ...
```

**Solution:** Under-relaxation breaks the feedback:

```python
k_new = 0.7·k_solved + 0.3·k_old      # blend new with old
nu_t_new = 0.5·nu_t_computed + 0.5·nu_t_old
```

#### 4. Wall resolution — y⁺ drives mesh cost

Low-Re wall treatment requires first cell in viscous sublayer (y⁺ < 2.5):

```
y⁺ = y·u_τ/ν = y·Re_τ/δ

For Re_τ = 590:  y_first < 2.5/590 ≈ 0.004
```

This forces thin wall cells → high aspect ratios → potential conditioning issues.

#### 5. Pressure-velocity coupling

IPCS splits Navier-Stokes into three sub-problems:
1. **Tentative velocity** — momentum without ∇p
2. **Pressure correction** — enforce ∇·u = 0
3. **Velocity correction** — project onto divergence-free space

Simpler than monolithic, but ν_t update timing within the loop affects stability.

**Further reading:**
- Carrier et al. (2021) "Finite element implementation of k−ω SST with automatic wall treatment" *Int. J. Numer. Methods Fluids* 93:3598-3627 — FEM-specific k-ω implementation
- Codina (1998) "Comparison of some finite element methods for solving the diffusion-convection-reaction equation" *Comput. Methods Appl. Mech. Eng.* 156:185-210 — stabilization for convection-dominated problems
- [FEniCS Book Ch. 21](https://fenicsproject.org/pub/book/book/fenics-book-2011-06-14.pdf) — Navier-Stokes in FEniCS

### Governing Equations

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

### Turbulence Modeling
1. Wilcox (1998) *Turbulence Modeling for CFD*, 2nd ed. — k-ω model foundation
2. Menter (1994) "Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications" *AIAA J.* 32(8):1598-1605 — SST model
3. Durbin (1996) "On the k-ε stagnation point anomaly" *Int. J. Heat Fluid Flow* 17:89-90

### Benchmark/Reference Cases
4. [Nek5000 RANS Tutorial (periodic channel)](https://nek5000.github.io/NekDoc/tutorials/rans.html) — high-Re RANS setup and model options
5. Moser, Kim, Mansour (1999) "DNS of turbulent channel flow up to Re_τ=590" *Phys. Fluids* 11(4):943-945 (legacy low-Re cross-check)
6. [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu/) — DNS data repository

### FEM Implementation
7. Carrier et al. (2021) "Finite element implementation of k−ω SST with automatic wall treatment" *Int. J. Numer. Methods Fluids* 93:3598-3627
8. Codina (1998) "Comparison of some finite element methods for solving the diffusion-convection-reaction equation" *Comput. Methods Appl. Mech. Eng.* 156:185-210
9. [FEniCS Book](https://fenicsproject.org/pub/book/book/fenics-book-2011-06-14.pdf) — Automated Solution of Differential Equations by the Finite Element Method

### Online Resources
- [CFD-Wiki: k-omega models](https://www.cfd-online.com/Wiki/SST_k-omega_model)
- [DOLFINx Documentation](https://docs.fenicsproject.org/dolfinx/)
- [dolfinx_mpc Documentation](https://jorgensd.github.io/dolfinx_mpc/)

## License

MIT License. See [LICENSE](LICENSE).
