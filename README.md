# dolfinx-rans

**A modern RANS k-ω turbulence solver for DOLFINx/FEniCSx 0.10+.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOLFINx](https://img.shields.io/badge/DOLFINx-0.10.0+-green.svg)](https://fenicsproject.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://python.org/)

A finite element RANS solver implementing the Wilcox (2006) standard k-ω and Menter (1994) k-ω SST turbulence models.
Use it as-is for turbulent channel flow, or as a validated foundation for extending to other geometries.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
  - [The Gap](#the-gap)
  - [Why k-ω](#why-k-ω)
  - [Why Channel Flow](#why-channel-flow)
  - [Audience](#audience)
- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
  - [Step 1: Create FEniCSx Environment](#step-1-create-fenicsx-environment-conda)
  - [Step 2: Install dolfinx-rans](#step-2-install-dolfinx-rans-uv)
  - [Requirements Summary](#requirements-summary)
  - [Why dolfinx_mpc?](#why-dolfinx_mpc)
- [Configuration Reference](#configuration-reference)
  - [geom — Channel Geometry](#geom--channel-geometry)
  - [nondim — Nondimensional Scaling](#nondim--nondimensional-scaling)
  - [turb — Turbulence Model](#turb--turbulence-model)
  - [solve — Solver Parameters](#solve--solver-parameters)
  - [benchmark — Regression Gates](#benchmark--regression-gates)
- [Governing Equations](#governing-equations)
  - [Nondimensionalization](#nondimensionalization)
  - [RANS Momentum](#rans-momentum)
  - [k-ω Transport Equations](#k-ω-transport-equations)
  - [Wilcox 2006 Model Constants](#wilcox-2006-model-constants)
  - [SST Extension](#sst-extension)
  - [Turbulent Viscosity](#turbulent-viscosity)
  - [Boundary Conditions](#boundary-conditions)
  - [Weak Forms and UFL Mapping](#weak-forms-and-ufl-mapping)
- [Numerical Method](#numerical-method)
  - [IPCS Pressure-Velocity Splitting](#ipcs-pressure-velocity-splitting)
  - [Pseudo-Transient Continuation](#pseudo-transient-continuation)
  - [Picard Iteration](#picard-iteration)
  - [PETSc Solver Configuration](#petsc-solver-configuration)
- [Stabilization: What Makes FEM RANS Hard](#stabilization-what-makes-fem-rans-hard)
  - [1. Positivity — FEM Doesn't Respect Physical Bounds](#1-positivity--fem-doesnt-respect-physical-bounds)
  - [2. Convection Dominance — Galerkin Oscillates](#2-convection-dominance--galerkin-oscillates)
  - [3. The ν_t Feedback Loop — Stiff Nonlinear Coupling](#3-the-ν_t-feedback-loop--stiff-nonlinear-coupling)
  - [4. Adaptive Time Stepping with Hysteresis](#4-adaptive-time-stepping-with-hysteresis)
  - [5. ω Floor and ν_t Limiting](#5-ω-floor-and-ν_t-limiting)
  - [6. Wall Resolution — y⁺ Drives Mesh Cost](#6-wall-resolution--y-drives-mesh-cost)
- [Architecture](#architecture)
  - [Module Map](#module-map)
  - [Function Spaces](#function-spaces)
  - [Data Flow](#data-flow)
- [Validation Against Nek5000](#validation-against-nek5000)
  - [Why Nek5000](#why-nek5000)
  - [Canonical Re = 100,000 Case](#canonical-re--100000-case)
  - [Comparison Workflow](#comparison-workflow)
  - [Regression Gates](#regression-gates)
- [Extending to Other Geometries](#extending-to-other-geometries)
- [Python API](#python-api)
- [Output Files](#output-files)
- [References](#references)
  - [Turbulence Modeling](#turbulence-modeling)
  - [FEM Implementation](#fem-implementation)
  - [Benchmark Cases](#benchmark-cases)
  - [Online Resources](#online-resources)
  - [Inspirations and Related Work](#inspirations-and-related-work)
- [License](#license)

---

## Why This Exists

### The Gap

DOLFINx is one of the most capable open-source finite element frameworks — automated variational forms, parallel assembly, flexible function spaces. Yet RANS turbulence solvers for it are scarce. Users who need turbulence modeling often start from scratch, re-discovering the same FEM-specific pitfalls (positivity violations, ν_t feedback instability, convection oscillations) that make RANS in finite elements harder than in finite volumes.

This project provides a working, validated k-ω RANS solver that exposes both the mathematics and the engineering decisions needed to make FEM RANS converge.

### Why k-ω

RANS turbulence models solve time-averaged equations with a turbulent viscosity $\nu_t$ to model Reynolds stresses. The main two-equation models are:

| Model | Solves for | Wall behavior |
|-------|------------|---------------|
| k-ε | k (TKE), ε (dissipation) | Needs wall functions or damping |
| **k-ω** | k (TKE), ω (specific dissipation) | Natural wall BC: $\omega \to 6\nu/(\beta_1 y^2)$ |
| k-ω SST | Blends k-ω (wall) with k-ε (freestream) | Best of both worlds |

**Decision:** We chose k-ω because it has the simplest wall boundary conditions of any two-equation model. The asymptotic wall value $\omega_\text{wall} = 6\nu/(\beta_1 y_1^2)$ is a direct Dirichlet BC — no wall functions, no damping functions, no distance-based switching. This is a significant advantage in FEM, where wall functions add implementation complexity and introduce errors that are hard to distinguish from numerical issues. The SST variant is also implemented for users who need better freestream behavior.

**Limitation:** Standard k-ω is sensitive to freestream $\omega$ values and may under-predict TKE in the channel core relative to DNS. The SST variant addresses freestream sensitivity but adds blending-function complexity.

### Why Channel Flow

Turbulent channel flow is the canonical RANS validation case because it removes all geometric ambiguity: one inlet condition (periodic), one driving mechanism (body force), and an exact momentum balance ($\tau_\text{wall} = 1$ in friction units). Any discrepancy between solver output and reference data is a model or numerics issue — not a meshing or BC problem.

**Decision:** We validate against Nek5000's RANS channel tutorial rather than DNS because the goal is cross-code RANS verification (do two codes with the same model produce the same answer?), not model-vs-physics validation (does RANS match DNS?). Nek5000 uses a well-tested finite-volume-style RANS implementation, so agreement confirms our FEM implementation is correct.

### Audience

- DOLFINx users who need RANS turbulence and want a working starting point
- Researchers exploring FEM-specific turbulence numerics
- Students learning how RANS models are implemented in variational form

---

## Quick Start

```bash
# 1. Activate FEniCSx environment
conda activate fenicsx

# 2. Install dolfinx-rans
cd dolfinx-rans && uv pip install -e .

# 3. Run canonical case (Re = 100,000 channel flow)
./run_channel.sh

# Results appear in channel/:
#   final_fields.png  — Contour plots and profiles
#   history.csv       — Convergence history
#   profiles.csv      — Wall-normal benchmark profile
```

The runner supports MPI parallelism:

```bash
./run_channel.sh 4                      # 4 MPI processes
./run_channel.sh 8 path/to/config.json  # 8 processes, custom config
```

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

---

## Features

**Physics**
- Standard k-ω model (Wilcox 2006) with cross-diffusion term
- k-ω SST model (Menter 1994) with F1/F2 blending functions
- Optional Durbin realizability limiter for stagnation regions
- Wilcox 2006 stress limiter ($\tilde{\omega} = \max(\omega, C_\text{lim} |S| / \sqrt{\beta^*})$)

**Numerics**
- Pseudo-transient continuation to steady state
- Adaptive time stepping with hysteresis to prevent oscillation
- Picard iteration with under-relaxation for k, ω, and ν_t
- IPCS fractional-step pressure-velocity coupling
- Wall-refined meshes with geometric or tanh stretching
- Half-channel (symmetry) or full-channel mode

**Infrastructure**
- Periodic boundary conditions via dolfinx_mpc
- Config-driven JSON workflow — one file controls everything
- Profile CSV export for cross-code benchmark comparison
- Config-driven regression gates (bulk velocity, wall shear, profile RMSE)
- VTX snapshots for ParaView time series
- Live convergence plots during iteration
- MPI-parallel execution

---

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

**Alternative: pip install**

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
| numpy ≥ 1.24 | pip/conda | Numerical arrays |
| matplotlib ≥ 3.7 | pip/conda | Plotting |

### Why dolfinx_mpc?

<details>
<summary><strong>Click to expand</strong></summary>

For turbulent channel flow, we need **periodic boundary conditions** in the streamwise direction: $u(x=0, y) = u(x=L_x, y)$.

**The problem:** Standard FEM boundary conditions can only set values (Dirichlet) or flux (Neumann) at boundaries — neither can express "right boundary equals left boundary", which is a constraint between DOFs.

**How dolfinx_mpc solves it:**

```
Standard FEM:                  With MPC:
[K]{u} = {f}                   [K']{u'} = {f'}

DOFs: [u₀, u₁, ..., uₙ]        DOFs: [u₀, u₁, ..., uₘ]  where m < n

                               Right-boundary DOFs eliminated and
                               replaced by left-boundary DOFs
```

MPC modifies the stiffness matrix to enforce $u_\text{right} = u_\text{left}$ as algebraic constraints, effectively reducing the system size.

**Why periodic + body force instead of inlet/outlet?**

| Approach | Pros/Cons |
|----------|-----------|
| **Inlet/outlet BCs** | Unknown inlet profile, entrance effects, long domain needed |
| **Periodic + body force** | Body force $f_x = 1$ drives fully developed channel, short domain sufficient, clean cross-code comparison |

</details>

---

## Configuration Reference

A single JSON file controls the entire simulation. Core sections (`geom`, `nondim`, `turb`, `solve`) are required. The `benchmark` section is optional and enables regression gates. Keys starting with `_` are ignored, allowing JSON comments.

**Full example** (the canonical Re = 100,000 case):

```json
{
  "_meta": {
    "purpose": "Canonical single-case run config",
    "reference": "Nek poiseuille_RANS uses Re=100000 via viscosity=-1e5 in .par"
  },
  "geom": {
    "Lx": 1.0,
    "Ly": 2.0,
    "Nx": 192,
    "Ny": 166,
    "mesh_type": "quad",
    "y_first": 0.001604628,
    "growth_rate": 1.0,
    "stretching": "tanh",
    "y_first_tol_rel": 0.2,
    "use_symmetry": false
  },
  "nondim": {
    "Re_tau": 1115.818661288065,
    "use_body_force": true
  },
  "turb": {
    "model": "wilcox2006",
    "beta_star": 0.09,
    "nu_t_max_factor": 2000.0,
    "omega_min": 1.0,
    "k_min": 1e-10,
    "k_max": 20.0,
    "C_lim": 0.0
  },
  "solve": {
    "dt": 0.0002,
    "dt_max": 0.01,
    "dt_growth": 1.05,
    "dt_growth_threshold": 0.8,
    "t_final": 10000.0,
    "max_iter": 1200,
    "steady_tol": 1e-3,
    "enable_physical_convergence": false,
    "picard_max": 6,
    "picard_tol": 1e-4,
    "under_relax_k_omega": 0.6,
    "under_relax_nu_t": 0.4,
    "log_interval": 10,
    "snapshot_interval": 50,
    "out_dir": "channel"
  },
  "benchmark": {
    "reference_profile_csv": ""
  }
}
```

### `geom` — Channel Geometry

| Parameter | Type | Description |
|-----------|------|-------------|
| `Lx` | float | Channel length (streamwise) |
| `Ly` | float | Channel height ($2\delta$ for full channel, $\delta$ for half-channel) |
| `Nx` | int | Mesh cells in x |
| `Ny` | int | Mesh cells in y |
| `mesh_type` | str | `"triangle"` or `"quad"` |
| `y_first` | float | First cell height from wall. For low-Re BCs: $y^+ = y_1 \cdot Re_\tau < 2.5$ |
| `growth_rate` | float | Geometric stretching ratio (>1 for wall refinement, 1 for uniform) |
| `stretching` | str | `"geometric"` or `"tanh"` (default: `"geometric"`) |
| `y_first_tol_rel` | float | Relative tolerance for mesh/BC consistency check (hard fail if exceeded) |
| `use_symmetry` | bool | `true` = half-channel with symmetry at top; `false` = full channel with walls on both sides |

**Decision:** We provide both geometric and tanh stretching. Geometric stretching (`y_{i+1} = r \cdot y_i`) is standard in CFD and gives explicit control via growth rate. Tanh stretching (`y(\eta) = H[1 - \tanh(\beta(1-\eta))/\tanh(\beta)]`) clusters more points near the wall for the same cell count. The solver computes `y_first` from the mesh geometry at runtime and hard-fails if it disagrees with the config — this catches silent mesh/BC mismatches that would produce wrong $\omega_\text{wall}$.

**Wall refinement example:**

```
For Re_τ ≈ 1116 (Re = 100,000), need y⁺ < 2.5:
    y_first < 2.5 / 1116 ≈ 2.24e-3

With y_first = 1.60e-3 and tanh stretching:
    y⁺ ≈ 1.79 ✓
```

### `nondim` — Nondimensional Scaling

| Parameter | Type | Description |
|-----------|------|-------------|
| `Re_tau` | float | Friction Reynolds number $Re_\tau = u_\tau \delta / \nu$ |
| `use_body_force` | bool | `true` = body force $f_x = 1$, periodic in x (default) |

**Decision:** Everything is nondimensionalized by $\delta$ (half-channel height) and $u_\tau$ (friction velocity), so $\nu^* = 1/Re_\tau$ and the expected equilibrium wall shear is $\tau_\text{wall} = 1$. This is the standard RANS nondimensionalization and matches the Nek5000 tutorial conventions.

### `turb` — Turbulence Model

| Parameter | Type | Description | Default rationale |
|-----------|------|-------------|-------------------|
| `model` | str | `"wilcox2006"` or `"sst"` | Wilcox 2006 is simpler and sufficient for channel flow |
| `beta_star` | float | k-ω constant $\beta^* = 0.09$ | Standard across all k-ω variants |
| `nu_t_max_factor` | float | Max $\nu_t / \nu$ ratio | Safety cap; see [ν_t limiting](#5-ω-floor-and-ν_t-limiting) |
| `omega_min` | float | ω floor to prevent $\nu_t$ runaway | See [ω floor](#5-ω-floor-and-ν_t-limiting) |
| `k_min` | float | k floor for positivity | `1e-10` prevents division by zero |
| `k_max` | float | k cap for safety | Prevents runaway during early iterations |
| `C_lim` | float | Durbin limiter: $\nu_t \le C_\text{lim} \cdot k / (\sqrt{6} |S|)$. `0` = disabled | Disabled by default; Wilcox 2006 has its own stress limiter |

### `solve` — Solver Parameters

| Parameter | Type | Description | Default rationale |
|-----------|------|-------------|-------------------|
| `dt` | float | Initial pseudo-time step | Small for stability during startup transient |
| `dt_max` | float | Maximum dt | Prevents excessively large steps |
| `dt_growth` | float | dt multiplier when converging | See [adaptive dt](#4-adaptive-time-stepping-with-hysteresis) |
| `dt_growth_threshold` | float | Hysteresis: only grow dt if $r_n / r_{n-1}$ < threshold | `0.8` means residual must be decreasing steadily |
| `t_final` | float | Max pseudo-time (safety limit) | Usually not reached — convergence check stops iteration |
| `max_iter` | int | Max outer iterations | Safety limit |
| `steady_tol` | float | Convergence tolerance on $\max(r_u, r_k, r_\omega)$ | `1e-3` is typical for RANS |
| `picard_max` | int | Inner Picard iterations per time step | See [Picard iteration](#picard-iteration) |
| `picard_tol` | float | Picard convergence tolerance | Inner loop exits early if k change is below this |
| `under_relax_k_omega` | float | Under-relaxation for k and ω | See [feedback loop](#3-the-ν_t-feedback-loop--stiff-nonlinear-coupling) |
| `under_relax_nu_t` | float | Under-relaxation for ν_t | `0.4` is aggressive but needed for stability |
| `log_interval` | int | Print diagnostics every N iterations | |
| `snapshot_interval` | int | Save VTX + PNG every N iterations (0 = disabled) | |
| `out_dir` | str | Output directory for results | |

**Decision:** Under-relaxation factors of 0.6 (k, ω) and 0.4 (ν_t) were determined empirically. Higher values cause oscillation in the ν_t feedback loop; lower values converge but take more iterations. The ν_t under-relaxation is more aggressive than k/ω because ν_t amplifies small changes through the momentum equation.

### `benchmark` — Regression Gates

| Parameter | Type | Description |
|-----------|------|-------------|
| `gate_u_bulk_bounds` | `[min, max]` | Acceptable range for final $U_\text{bulk}$ |
| `gate_tau_wall_bounds` | `[min, max]` | Acceptable range for final $\tau_\text{wall}$ |
| `reference_profile_csv` | str | Path to Nek reference CSV (columns: `y_over_delta,u_over_ubulk`) |
| `u_plus_rmse_max` | float | Max RMSE for velocity profile vs reference |
| `k_plus_rmse_max` | float | Max RMSE for TKE profile vs reference |

---

## Governing Equations

### Nondimensionalization

All equations are nondimensionalized by the half-channel height $\delta$ and friction velocity $u_\tau$:

$$
x^* = x/\delta, \quad u^* = u/u_\tau, \quad p^* = p/(\rho u_\tau^2), \quad t^* = t \cdot u_\tau/\delta, \quad \nu^* = \nu/(u_\tau \delta) = 1/Re_\tau
$$

Dropping the $*$ superscripts throughout, the nondimensional kinematic viscosity is:

$$
\nu = \frac{1}{Re_\tau}
$$

The body force $f_x = 1$ maintains a friction velocity of $u_\tau = 1$ in equilibrium, so the expected wall shear stress at convergence is $\tau_\text{wall} = \nu \cdot (\partial u / \partial y)|_\text{wall} = 1$.

### RANS Momentum

The Reynolds-averaged Navier-Stokes momentum equation with the Boussinesq eddy-viscosity hypothesis:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nabla \cdot [(\nu + \nu_t) \nabla \mathbf{u}] + \mathbf{f}
$$

where $\mathbf{f} = (1, 0)^T$ is the body force driving the channel flow, and $\nu_t$ is the turbulent eddy viscosity computed from the k-ω model.

### k-ω Transport Equations

**Turbulent kinetic energy (k):**

$$
\frac{\partial k}{\partial t} + \mathbf{u} \cdot \nabla k = P_k - \beta^* k \omega + \nabla \cdot [(\nu + \sigma_k \nu_t) \nabla k]
$$

**Specific dissipation rate (ω):**

$$
\frac{\partial \omega}{\partial t} + \mathbf{u} \cdot \nabla \omega = \gamma S^2 - \beta \omega^2 + \nabla \cdot [(\nu + \sigma_\omega \nu_t) \nabla \omega] + \frac{\sigma_d}{\omega} \max(\nabla k \cdot \nabla \omega, 0)
$$

where:
- $P_k = \nu_t S^2$ is the TKE production, with $S^2 = 2 S_{ij} S_{ij}$ and $S_{ij} = \frac{1}{2}(\partial u_i / \partial x_j + \partial u_j / \partial x_i)$
- The last term in the ω-equation is the Wilcox 2006 **cross-diffusion** term, which reduces freestream sensitivity

**Decision:** The ω-equation production term uses $\gamma S^2$ (not $\gamma (\omega/k) P_k$) following the Wilcox 2006 formulation. This avoids the singularity $\omega/k \to \infty$ near walls where $k \to 0$, and is the form used in the Nek5000 RANS tutorial.

### Wilcox 2006 Model Constants

| Constant | Symbol | Value | Notes |
|----------|--------|-------|-------|
| k destruction | $\beta^*$ | 0.09 | Standard across all k-ω variants |
| ω destruction | $\beta_0$ | 0.0708 | Updated from 0.075 (1998) |
| k diffusion | $\sigma_k$ | 0.6 | Updated from 0.5 (1998) |
| ω diffusion | $\sigma_\omega$ | 0.5 | |
| von Kármán | $\kappa$ | 0.41 | |
| Production ratio | $\gamma$ | $\beta_0/\beta^* - \sigma_\omega \kappa^2 / \sqrt{\beta^*}$ ≈ 0.507 | Chosen to yield correct log-layer slope |
| Cross-diffusion | $\sigma_{d0}$ | 0.125 (= 1/8) | New in Wilcox 2006 |
| Stress limiter | $C_\text{lim}$ | 0.875 (= 7/8) | New in Wilcox 2006 |

**Decision:** We use the Wilcox 2006 constants (not 1998). The 2006 revision added the cross-diffusion term $(\sigma_d / \omega) \max(\nabla k \cdot \nabla \omega, 0)$ and updated $\beta_0$, $\sigma_k$ to reduce freestream sensitivity — the primary weakness of the original k-ω model. The cross-diffusion term activates only where $\nabla k$ and $\nabla \omega$ are aligned (typically away from walls), mimicking the behavior of k-ε in the freestream while retaining k-ω's natural wall treatment.

### SST Extension

The k-ω SST (Shear Stress Transport) model blends Wilcox k-ω near walls with transformed k-ε in the freestream using a blending function $F_1$:

$$
\phi = F_1 \phi_1 + (1 - F_1) \phi_2
$$

where $\phi$ represents any blended coefficient ($\sigma_k$, $\sigma_\omega$, $\beta$, $\gamma$).

**Inner-layer (k-ω) constants:** $\sigma_{k1} = 0.85$, $\sigma_{\omega 1} = 0.5$, $\beta_1 = 0.075$

**Outer-layer (k-ε transformed) constants:** $\sigma_{k2} = 1.0$, $\sigma_{\omega 2} = 0.856$, $\beta_2 = 0.0828$

The cross-diffusion term for SST uses the blending:

$$
CD_\text{SST} = (1 - F_1) \frac{2 \sigma_{\omega 2}}{\omega} \max(\nabla k \cdot \nabla \omega, 0)
$$

### Turbulent Viscosity

**Wilcox 2006** with stress limiter:

$$
\nu_t = \frac{k}{\tilde{\omega}}, \quad \tilde{\omega} = \max\left(\omega, \; C_\text{lim} \frac{|S|}{\sqrt{\beta^*}}\right)
$$

The stress limiter prevents excess $\nu_t$ in regions of high strain rate (stagnation points).

**SST** with shear-stress transport limiter:

$$
\nu_t = \frac{a_1 k}{\max(a_1 \omega, \; |S| F_2)}, \quad a_1 = 0.31
$$

where $F_2$ is a second blending function that activates in boundary layers.

### Boundary Conditions

**Walls** ($y = 0$ and $y = 2\delta$ for full channel; $y = 0$ for half-channel):

| Variable | BC | Value |
|----------|-----|-------|
| $\mathbf{u}$ | Dirichlet (no-slip) | $\mathbf{u} = \mathbf{0}$ |
| $k$ | Dirichlet | $k = 0$ |
| $\omega$ | Dirichlet | $\omega_\text{wall} = 6\nu / (\beta_0 y_1^2)$ |

where $y_1$ is the first off-wall mesh spacing, inferred directly from the mesh geometry at runtime.

**Symmetry plane** ($y = \delta$ for half-channel):
- $v = 0$ (no wall-normal velocity)
- $\partial u / \partial y = 0$ (natural Neumann, not explicitly imposed)
- $\partial k / \partial y = 0$ (natural Neumann)
- $\partial \omega / \partial y = 0$ (natural Neumann)

**Streamwise** ($x = 0, x = L_x$):
- Periodic via dolfinx_mpc multi-point constraints

**Decision:** The $\omega_\text{wall}$ BC deserves explanation. It comes from the asymptotic solution of the ω-equation near a wall where $k \to 0$ and the viscous term dominates: $\omega \sim 6\nu/(\beta_0 y^2)$. This is applied at the first mesh point, not at $y = 0$ itself (where $\omega \to \infty$). The solver infers $y_1$ from the actual mesh coordinates and cross-checks against the config value to catch mismatches.

### Weak Forms and UFL Mapping

The equations are implemented in DOLFINx's UFL (Unified Form Language). Here is how the k-equation maps from math to code.

**Mathematical form** (semi-discrete in time):

$$
\frac{k^{n+1} - k^n}{\Delta t} + \mathbf{u}^n \cdot \nabla k^{n+1} + (\nu + \sigma_k \nu_t) \nabla k^{n+1} \cdot \nabla \phi_k + \beta^* \omega^n k^{n+1} = P_k
$$

**UFL implementation** (from `solver.py`):

```python
# Diffusion coefficient
D_k = nu_c + sigma_k_c * nu_t_

# Production: P_k = ν_t * 2 * S_ij * S_ij
S_tensor = sym(grad(u_n))
S_sq = 2.0 * inner(S_tensor, S_tensor)
P_k = nu_t_ * S_sq

# Weak form: find k_trial such that F_k(k_trial, phi_k) = 0
F_k = (
    (k_trial - k_n) / dt_c * phi_k * dx           # time derivative
    + dot(u_n, grad(k_trial)) * phi_k * dx         # convection
    + D_k * inner(grad(k_trial), grad(phi_k)) * dx # diffusion (integrated by parts)
    + beta_star_c * omega_safe * k_trial * phi_k * dx  # destruction
    - P_k * phi_k * dx                             # production (RHS)
)
```

The ω-equation follows the same pattern with an additional cross-diffusion source term:

```python
# Cross-diffusion: σ_d/ω * max(∇k·∇ω, 0) — explicit treatment
grad_k_dot_grad_w = dot(grad(k_n), grad(omega_n))
grad_kw_positive = ufl.conditional(
    ufl.gt(grad_k_dot_grad_w, 0.0), grad_k_dot_grad_w, 0.0
)
cross_diff = sigma_d_c / omega_safe * grad_kw_positive
```

**Decision:** The convection term $\mathbf{u} \cdot \nabla k$ uses the current velocity $\mathbf{u}^n$ (explicit in velocity) while $k$ is implicit (trial function). This semi-implicit treatment makes the system linear in $k$ at each step — we can extract `lhs()` and `rhs()` from UFL and solve a single linear system per equation per Picard iteration. Fully implicit treatment would require a nonlinear solver and is not worth the complexity for pseudo-transient RANS.

---

## Numerical Method

### IPCS Pressure-Velocity Splitting

The Incremental Pressure Correction Scheme (IPCS) splits the Navier-Stokes equations into three sequential linear solves per time step:

**Step 1 — Tentative velocity** (momentum without pressure gradient correction):

$$
\frac{\rho}{\Delta t}(\mathbf{u}^* - \mathbf{u}^n) + (\tilde{\mathbf{u}} \cdot \nabla) \frac{1}{2}(\mathbf{u}^* + \mathbf{u}^n) + \frac{1}{2} \mu_\text{eff} \nabla(\mathbf{u}^* + \mathbf{u}^n) = \nabla p^n + \mathbf{f}
$$

where $\tilde{\mathbf{u}} = \frac{3}{2}\mathbf{u}^n - \frac{1}{2}\mathbf{u}^{n-1}$ is an Adams-Bashforth extrapolation of the convecting velocity, and $\mu_\text{eff} = \rho(\nu + \nu_t)$.

**Step 2 — Pressure correction** (enforce divergence-free velocity):

$$
\nabla^2 \phi = -\frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*, \quad p^{n+1} = p^n + \phi
$$

**Step 3 — Velocity correction** (project onto divergence-free space):

$$
\rho \mathbf{u}^{n+1} = \rho \mathbf{u}^* - \Delta t \nabla \phi
$$

**Decision:** IPCS was chosen over monolithic (fully coupled) Navier-Stokes because it reuses the same matrix structure across iterations (only the RHS changes when $\nu_t$ is updated), keeps the linear systems smaller, and is well-tested in the FEniCSx ecosystem. The Adams-Bashforth convection extrapolation is second-order accurate and avoids the need to solve a nonlinear system for the convective term.

### Pseudo-Transient Continuation

We do not seek time accuracy — the $\partial/\partial t$ terms are artificial relaxation that drives the solution toward steady state. The "time step" $\Delta t$ controls the implicit damping: small $\Delta t$ adds strong damping (safe but slow), large $\Delta t$ reduces damping (fast but potentially unstable).

The outer loop structure is:

```
for step in 1..max_iter:
    for picard_iter in 1..picard_max:
        solve momentum (IPCS steps 1-3)
        under-relax velocity: u_n = α·u_new + (1-α)·u_old
        solve k-equation, clip, under-relax
        solve ω-equation, clip, under-relax
        update ν_t with under-relaxation
        check Picard convergence
    compute outer residual
    adapt dt based on residual trend
    check steady-state convergence
```

### Picard Iteration

Within each pseudo-time step, the nonlinear coupling between momentum and turbulence is resolved by Picard (fixed-point) iteration. Each Picard iteration:

1. Solves momentum with current $\nu_t$ (IPCS steps 1-3)
2. Under-relaxes velocity: $\mathbf{u}^n \leftarrow \alpha_u \mathbf{u}_\text{new} + (1 - \alpha_u) \mathbf{u}_\text{old}$
3. Updates SST blending functions (if using SST model)
4. Solves k with current velocity and $\omega$
5. Clips k to $[k_\text{min}, k_\text{max}]$, under-relaxes
6. Solves ω with current velocity and k
7. Clips ω to $[\omega_\text{min}, \omega_\text{max}]$, under-relaxes
8. Computes $\nu_t$ from updated k and ω, under-relaxes

The inner loop exits early if the relative change in k drops below `picard_tol`.

**Decision:** Under-relaxation is applied at three levels: velocity ($\alpha_u = 0.7$), k/ω (`under_relax_k_omega = 0.6`), and ν_t (`under_relax_nu_t = 0.4`). The ν_t relaxation is the most critical because ν_t couples all equations — a 10% change in ν_t causes ~10% change in effective viscosity everywhere, which changes production, which changes k and ω, which changes ν_t again. Without under-relaxation, this feedback loop diverges.

### PETSc Solver Configuration

Each linear subsystem uses a tuned PETSc solver:

| System | KSP | Preconditioner | Rationale |
|--------|-----|---------------|-----------|
| Momentum (step 1) | BiCGStab | Jacobi | Non-symmetric system; Jacobi is cheap and MPI-friendly |
| Pressure (step 2) | CG | Hypre BoomerAMG | SPD Laplacian; AMG gives mesh-independent convergence |
| Velocity correction (step 3) | CG | Jacobi | Mass matrix; cheap and well-conditioned |
| k-equation | BiCGStab | Hypre BoomerAMG | Convection-diffusion; AMG handles anisotropic meshes |
| ω-equation | BiCGStab | Hypre BoomerAMG | Same as k |

All solvers use `rtol = 1e-8`.

**Decision:** We use algebraic multigrid (BoomerAMG) for the pressure Poisson and turbulence transport equations because wall-refined meshes create highly anisotropic cells. Direct solvers or simple preconditioners (Jacobi, ILU) struggle with the resulting ill-conditioning. AMG automatically constructs a hierarchy that handles the anisotropy. Note: ILU is incompatible with MPI parallelism; BoomerAMG works in parallel out of the box.

---

## Stabilization: What Makes FEM RANS Hard

Most RANS solvers use finite volume methods (FVM), which naturally suit conservation laws. FEM offers advantages — higher-order accuracy, complex geometries, rigorous error analysis — but faces specific challenges with turbulence. This section documents what we tried, what failed, and what works.

### 1. Positivity — FEM Doesn't Respect Physical Bounds

**The problem:** $k$ and $\omega$ must always be positive (they represent energy and frequency). In FVM, upwind schemes naturally preserve positivity. In FEM, the solution is a weighted sum of basis functions — and that sum can go negative near steep gradients, especially during early iterations when the fields are far from converged.

**What happens without it:** Negative $k$ or $\omega$ → $\nu_t = k/\omega$ blows up or flips sign → immediate divergence.

**Solution:** Hard clipping after each linear solve, before under-relaxation:

```python
k_.x.array[:] = np.clip(k_.x.array, k_min, k_max)
omega_.x.array[:] = np.clip(omega_.x.array, omega_min, omega_max_limit)
```

where `omega_max_limit = 10 * omega_wall` scales the cap with the current case.

**Decision:** Clipping is mathematically inelegant (it violates the variational principle) but pragmatically essential. More sophisticated approaches exist — DG with slope limiters, variational inequalities — but they add enormous complexity. Clipping works reliably and the clipped values are small perturbations on the converged solution. The key insight is that `k_min = 1e-10` (not 0) prevents $k/\omega$ from producing `0/0 = NaN`, and the ω floor has a much larger impact on solution quality (see below).

### 2. Convection Dominance — Galerkin Oscillates

**The problem:** At high Re, the k and ω transport is dominated by convection ($\mathbf{u} \cdot \nabla k$), not diffusion. Standard Galerkin FEM is centered (like central differencing) — it produces spurious oscillations near steep gradients when the local mesh Péclet number is large.

**FVM handles this naturally** with upwind schemes that add stabilizing numerical diffusion in the flow direction.

**FEM workarounds:**
- **SUPG stabilization** — adds artificial streamline diffusion (complex to tune, especially with turbulence)
- **Fine mesh** — keep local Péclet number $Pe_h = |u| h / (2D)$ moderate
- **Pseudo-transient stepping** — the $k/\Delta t$ mass-matrix term acts as isotropic numerical diffusion

**Decision:** We rely on mesh refinement and pseudo-transient damping rather than SUPG. This is simpler to implement and debug, and the wall-refined mesh already provides the resolution where gradients are steepest. SUPG would help with coarser meshes but introduces its own tuning parameters and can interfere with turbulence production near walls.

### 3. The ν_t Feedback Loop — Stiff Nonlinear Coupling

**The problem:** $\nu_t = k/\omega$ couples all equations. Small changes in ω cause large swings in ν_t → changes diffusion in the momentum equation → changes velocity → changes production $P_k = \nu_t |S|^2$ → changes k and ω → changes ν_t again. This positive feedback loop is the primary source of instability in FEM RANS.

```
ω ↑  →  ν_t ↓  →  less diffusion  →  steeper gradients  →  more production  →  k ↑  →  ν_t ↑  → ...
```

**What failed:** No under-relaxation → oscillation and divergence within 5-10 iterations. Under-relaxing only k/ω but not ν_t → still oscillates because ν_t amplifies the changes. Using the same relaxation factor for all fields → either too aggressive (oscillates) or too conservative (takes thousands of iterations).

**What works:** Three-level under-relaxation with different factors:

```python
# Level 1: velocity (mild)
u_n = 0.7 * u_solved + 0.3 * u_old

# Level 2: turbulence quantities (moderate)
k_new  = 0.6 * k_solved  + 0.4 * k_old
w_new  = 0.6 * w_solved  + 0.4 * w_old

# Level 3: eddy viscosity (aggressive)
nu_t_new = 0.4 * nu_t_computed + 0.6 * nu_t_old
```

**Decision:** The ν_t under-relaxation factor (0.4) is the most critical. This means each iteration only moves ν_t 40% of the way toward its "correct" value, which slows convergence but prevents the feedback loop from overshooting. The factor was determined empirically: 0.5 oscillates on the Re = 100,000 case, 0.3 is safe but slow. 0.4 is the sweet spot for this class of problems.

### 4. Adaptive Time Stepping with Hysteresis

**The problem:** Fixed $\Delta t$ is either too small (wasting iterations during the converged phase) or too large (causing instability during the transient phase). Simple adaptive dt that grows whenever the residual decreases causes oscillation — grow dt → residual jumps → shrink dt → residual drops → grow dt → ...

**What works:** Hysteresis-based adaptation. The dt only grows when the residual ratio $r_n / r_{n-1}$ is below a threshold (0.8 by default), and only shrinks when the ratio exceeds $1/\text{threshold}$:

```python
residual_ratio = residual / residual_prev
if residual_ratio < dt_growth_threshold:        # residual clearly decreasing
    dt = min(dt * dt_growth, dt_max)
elif residual_ratio > 1.0 / dt_growth_threshold:  # residual clearly increasing
    dt = max(dt / dt_growth, dt_initial)
# Otherwise: hold dt steady (hysteresis band)
```

**Decision:** The `dt_growth_threshold = 0.8` means dt only grows when the residual dropped by at least 20%. The dead band between 0.8 and $1/0.8 = 1.25$ prevents oscillatory dt behavior. The growth factor `dt_growth = 1.05` is conservative (5% increase per step) to avoid overshooting the stability boundary.

### 5. ω Floor and ν_t Limiting

**The problem:** Where ω is small (channel core), $\nu_t = k/\omega$ can become extremely large — orders of magnitude larger than the physical turbulent viscosity. This "ν_t runaway" adds so much diffusion that the velocity profile flattens and the solution diverges from the correct shape.

**Solution:** Two complementary limits:

1. **ω floor** (`omega_min`): Prevents $\omega$ from dropping below a minimum value. This directly limits $\nu_t$ in the channel core.

2. **ν_t cap** (`nu_t_max_factor`): Hard cap at $\nu_t \le \text{factor} \cdot \nu$. Safety net for extreme cases.

3. **Wilcox 2006 stress limiter** (built-in): $\tilde{\omega} = \max(\omega, C_\text{lim} |S| / \sqrt{\beta^*})$. This limits ν_t in regions of high strain rate without affecting the log layer.

**Decision:** The `omega_min = 1.0` in the canonical config is a compromise. Higher values (10-100) produce a better $u^+$ profile shape but over-constrain the model in the channel core. Lower values (0.01) allow more physically correct ω but risk ν_t runaway during early iterations. The stress limiter ($C_\text{lim} = 7/8$) handles stagnation-point anomalies but is disabled by default (`C_lim = 0.0` in config) because it can interact with the ω floor in unexpected ways.

### 6. Wall Resolution — y⁺ Drives Mesh Cost

Low-Re wall treatment (direct integration to the wall, no wall functions) requires the first cell center in the viscous sublayer: $y^+ < 2.5$.

$$
y^+ = \frac{y_1 \cdot u_\tau}{\nu} = y_1 \cdot Re_\tau
$$

For the canonical case ($Re_\tau \approx 1116$): $y_1 < 2.5 / 1116 \approx 2.24 \times 10^{-3}$.

This forces thin wall cells → high aspect ratios (>100:1 on a 192 × 166 mesh) → potential conditioning issues in the linear solvers. AMG preconditioning handles this well, but simpler preconditioners (ILU, Jacobi) may struggle.

**Decision:** Tanh stretching concentrates more cells near the wall than geometric stretching for the same Ny, achieving lower $y^+$ without increasing total cell count. The mesh consistency check (`y_first_tol_rel`) catches cases where the config specifies one `y_first` but the mesh generator produces a different one — this mismatch would silently corrupt the $\omega_\text{wall}$ BC.

---

## Architecture

### Module Map

```
dolfinx-rans/
├── src/dolfinx_rans/
│   ├── solver.py      — Core solver: equations, BCs, IPCS loop, k-ω/SST models
│   ├── cli.py         — CLI entry point: config parsing, case setup, post-processing
│   ├── utils.py       — Config loading, diagnostics, wall shear stress, CSV logging
│   ├── plotting.py    — Mesh, field, convergence, and profile plots
│   └── validation/
│       └── nek_poiseuille_profile.py  — Nek5000 profile extraction and comparison
├── bfs/
│   ├── bfs_quick.json   — BFS quick debug case
│   └── ...
├── channel/
│   └── run_config.json   — Canonical case configuration
├── run_channel.sh         — One-command runner for channel case
├── run_bfs.sh             — One-command runner for BFS quick case
└── pyproject.toml         — Package metadata (hatchling build system)
```

### Function Spaces

| Variable | Space | Element | Degree | Rationale |
|----------|-------|---------|--------|-----------|
| Velocity $\mathbf{u}$ | V | Lagrange (vector) | 2 | Taylor-Hood P2-P1 for inf-sup stability |
| Pressure $p$ | Q | Lagrange (scalar) | 1 | Taylor-Hood P2-P1 |
| $k$, $\omega$, $\nu_t$ | S | Lagrange (scalar) | 1 | P1 matches pressure space; avoids interpolation issues |

**Decision:** P2-P1 (Taylor-Hood) velocity-pressure is the standard inf-sup stable pair. Using P1 for turbulence quantities (same as pressure) avoids the need to project between different-degree spaces when computing $P_k = \nu_t |S|^2$ from velocity gradients. P2 turbulence would give smoother profiles but doubles the turbulence DOF count with marginal accuracy benefit for RANS.

### Data Flow

```
                         ┌──────────────────────────────────┐
                         │         Picard Iteration          │
                         │                                    │
  u_n, p_  ─────────────►│  1. IPCS momentum  → u_new        │
                         │  2. Under-relax u   → u_n          │
                         │  3. [SST: update F1, blend coeffs] │
  k_n, omega_n, nu_t_ ──►│  4. Solve k         → k_new       │
                         │  5. Clip + under-relax k           │
                         │  6. Solve ω         → ω_new        │
                         │  7. Clip + under-relax ω           │
                         │  8. Compute ν_t     → ν_t_new      │
                         │  9. Under-relax ν_t                │
                         │ 10. Check Picard convergence       │
                         └──────────────────────────────────┘
                                        │
                                        ▼
                              Outer residual check
                              Adaptive dt update
                              Logging / snapshots
```

---

## Validation Against Nek5000

### Why Nek5000

Nek5000 is a well-established spectral-element CFD code with a validated RANS implementation. Comparing against Nek5000 is **cross-code verification**: do two independent implementations of the same turbulence model produce the same answer? This is more rigorous than comparing against DNS, which tests the model itself (and all RANS models disagree with DNS in known ways).

### Canonical Re = 100,000 Case

The canonical validation case matches the [Nek5000 RANS tutorial](https://nek5000.github.io/NekDoc/tutorials/rans.html):

| Parameter | Value |
|-----------|-------|
| Bulk Reynolds number $Re$ | 100,000 |
| Friction Reynolds number $Re_\tau$ | ~1,116 |
| Domain | $L_x \times L_y = 1 \times 2$ (full channel) |
| Mesh | 192 × 166 quads, tanh stretching |
| $y_1^+$ | ~1.79 |
| Turbulence model | Wilcox 2006 k-ω |
| Driving | Body force $f_x = 1$, periodic in x |

### Comparison Workflow

1. **Run dolfinx-rans:** `./run_channel.sh` → produces `channel/profiles.csv`
2. **Extract Nek5000 profile:**
   ```bash
   python -m dolfinx_rans.validation.nek_poiseuille_profile \
       --nek-case-dir path/to/poiseuille_RANS \
       --out-dir nek_re100k \
       --dolfinx-profiles channel/profiles.csv
   ```
3. **Configure regression gate:** Set `benchmark.reference_profile_csv` to the Nek reference CSV
4. **Re-run with gate:** `./run_channel.sh` — the post-run check compares profiles automatically

The comparison uses outer scaling ($U/U_\text{bulk}$ vs $y/\delta$) because it is independent of the wall-friction estimate and provides a shape comparison of the mean velocity profile.

### Regression Gates

The `run_channel.sh` script includes post-run regression checks driven by the `benchmark` config section:

- **Bulk velocity gate:** $U_\text{bulk}$ must fall within `gate_u_bulk_bounds`
- **Wall shear gate:** $\tau_\text{wall}$ must fall within `gate_tau_wall_bounds`
- **Profile RMSE gate:** RMSE of $U/U_\text{bulk}$ vs reference must be below `u_plus_rmse_max`
- **TKE RMSE gate:** RMSE of $k$ profile vs reference must be below `k_plus_rmse_max`

These gates catch regressions when modifying the solver — if a code change breaks the solution quality, the runner exits with an error.

---

## Extending to Other Geometries

The solver is structured so the turbulence model and numerical method are geometry-independent. To adapt to a new geometry:

**What to change:**
- **Mesh:** Replace `create_channel_mesh()` with your geometry (external mesh import via `dolfinx.io.gmshio` or similar)
- **Boundary conditions:** Update `mark_boundaries()` and BC definitions for your wall/inlet/outlet/symmetry surfaces
- **Wall distance:** Replace `compute_wall_distance_channel()` (SST only) with a general wall-distance computation
- **Initial conditions:** Provide appropriate ICs for your geometry
- **Body force / driving:** Replace $f_x = 1$ with your driving mechanism (pressure BC, inlet velocity, etc.)

**What to keep:**
- The IPCS splitting and Picard iteration loop
- Under-relaxation strategy and factors (start with the channel values)
- Clipping and ν_t limiting
- Turbulence model constants
- PETSc solver configuration

**Tuning caveat:** The under-relaxation factors and ω floor were tuned for channel flow. More complex geometries (recirculation zones, separation, stagnation) may require:
- Lower under-relaxation factors (0.3-0.5 for k/ω instead of 0.6)
- Higher ω floor initially, reduced as the solution develops
- The Durbin stress limiter (`C_lim > 0`) for stagnation regions
- SST model instead of standard k-ω for freestream-sensitive flows

---

## Python API

```python
from dolfinx_rans.solver import (
    ChannelGeom,
    NondimParams,
    TurbParams,
    SolveParams,
    create_channel_mesh,
    solve_rans_kw,
)

# Define geometry (full channel, tanh stretching)
geom = ChannelGeom(
    Lx=1.0, Ly=2.0, Nx=192, Ny=166,
    mesh_type="quad", y_first=0.001604628,
    growth_rate=1.0, stretching="tanh",
    y_first_tol_rel=0.2, use_symmetry=False,
)

# Nondimensional parameters
nondim = NondimParams(Re_tau=1115.82, use_body_force=True)

# Turbulence model (Wilcox 2006)
turb = TurbParams(
    model="wilcox2006", beta_star=0.09,
    nu_t_max_factor=2000.0, omega_min=1.0,
    k_min=1e-10, k_max=20.0, C_lim=0.0,
)

# Solver settings
solve = SolveParams(
    dt=0.0002, dt_max=0.01, dt_growth=1.05,
    dt_growth_threshold=0.8, t_final=10000.0,
    max_iter=1200, steady_tol=1e-3,
    picard_max=6, picard_tol=1e-4,
    under_relax_k_omega=0.6, under_relax_nu_t=0.4,
    log_interval=10, snapshot_interval=50,
    out_dir="results",
)

# Create mesh and run
domain = create_channel_mesh(geom, Re_tau=nondim.Re_tau)
u, p, k, omega, nu_t, V, Q, S, domain, step, t = solve_rans_kw(
    domain, geom, turb, solve, Path("results"), nondim,
)
```

The solver can also be run from the command line:

```bash
dolfinx-rans config.json           # Serial
mpirun -np 4 dolfinx-rans config.json  # Parallel
```

---

## Output Files

Results are saved to the directory specified by `solve.out_dir`:

| File | Description |
|------|-------------|
| `final_fields.png` | 2D contours (u, k, ν_t/ν) + 1D profiles (U/U_bulk, k, ω, ν_t/ν) |
| `history.csv` | Per-iteration convergence log: residual, dt, U_bulk, τ_wall, field ranges |
| `profiles.csv` | Wall-normal profile: `y, y_over_delta, u, u_over_ubulk, k, omega, nu_t_over_nu` |
| `convergence.png` | Residual and dt history plots |
| `mesh.png` | Mesh visualization with stretching |
| `config_used.json` | Exact config snapshot for reproducibility |
| `run_info.json` | Environment metadata (Python version, git SHA, timestamp) |
| `snps/*.bp` | VTX time series for ParaView (ADIOS2 format, if `snapshot_interval > 0`) |
| `fields.png` | Latest live field snapshot (overwritten each snapshot interval) |
| `fields_NNNNN.png` | Numbered field snapshots (preserved for animation) |

---

## References

### Turbulence Modeling

1. Wilcox, D.C. (2006). *Turbulence Modeling for CFD*, 3rd ed. DCW Industries. — Standard k-ω model with cross-diffusion and stress limiter updates.
2. Menter, F.R. (1994). "Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications." *AIAA Journal* 32(8):1598-1605. — SST model derivation and constants.
3. Durbin, P.A. (1996). "On the k-ε stagnation point anomaly." *Int. J. Heat and Fluid Flow* 17:89-90. — Realizability limiter for stagnation regions.
4. Wilcox, D.C. (1998). *Turbulence Modeling for CFD*, 2nd ed. DCW Industries. — Earlier k-ω constants (superseded by 2006 edition for this solver).

### FEM Implementation

5. Carrier, A., Errera, M.-P., Gilles, F., and Mouriaux, S. (2021). "Finite element implementation of k−ω SST with automatic wall treatment and target-function approach." *Int. J. Numer. Methods Fluids* 93:3598-3627. — FEM-specific k-ω implementation strategies.
6. Codina, R. (1998). "Comparison of some finite element methods for solving the diffusion-convection-reaction equation." *Comput. Methods Appl. Mech. Eng.* 156:185-210. — Stabilization for convection-dominated transport.
7. Donea, J. and Huerta, A. (2003). *Finite Elements Methods for Flow Problems*. Wiley. — IPCS and fractional-step methods.
8. Logg, A., Mardal, K.-A., and Wells, G.N. (2012). [*Automated Solution of Differential Equations by the Finite Element Method: The FEniCS Book*](https://fenicsproject.org/pub/book/book/fenics-book-2011-06-14.pdf). Springer. — FEniCS framework and Navier-Stokes implementation.

### Benchmark Cases

9. [Nek5000 RANS Tutorial (periodic channel)](https://nek5000.github.io/NekDoc/tutorials/rans.html) — High-Re RANS setup and cross-code reference.
10. Moser, R.D., Kim, J., and Mansour, N.N. (1999). "Direct numerical simulation of turbulent channel flow up to Re_τ = 590." *Phys. Fluids* 11(4):943-945. — Classic DNS reference (for model-vs-physics comparison, not used for solver verification).

### Online Resources

- [DOLFINx Documentation](https://docs.fenicsproject.org/dolfinx/)
- [dolfinx_mpc Documentation](https://jorgensd.github.io/dolfinx_mpc/)
- [CFD-Wiki: SST k-omega model](https://www.cfd-online.com/Wiki/SST_k-omega_model)
- [CFD-Wiki: Wilcox k-omega model](https://www.cfd-online.com/Wiki/Wilcox%27s_k-omega_model)

### Inspirations and Related Work

This solver did not emerge in a vacuum. The following projects, tutorials, and discussions shaped its design and helped debug implementation choices. Listed roughly in order of influence.

**FEniCS ecosystem — Navier-Stokes foundations:**

- [Dokken, J.S. — *The FEniCSx Tutorial: Navier-Stokes*](https://jsdokken.com/dolfinx-tutorial/chapter2/navierstokes.html) — The IPCS implementation in this solver follows Dokken's DOLFINx tutorial closely. The Adams-Bashforth convection extrapolation and `nabla_grad` convention come directly from here.
- [Dokken, J.S. — dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc) ([docs](https://jorgensd.github.io/dolfinx_mpc/)) — Periodic boundary conditions via multi-point constraints. The topological vs geometrical constraint paths and the `backsubstitution` pattern in this solver mirror the dolfinx_mpc demos.
- [Mortensen, M. and Valen-Sendstad, K. — *Oasis*](https://github.com/mikaem/Oasis) ([paper](https://arxiv.org/abs/1602.03643)) — High-performance FEniCS Navier-Stokes solver. Our IPCS splitting, P2-P1 function spaces, and solver architecture are influenced by Oasis's IPCS_ABCN scheme. Oasis demonstrated that FEniCS can match dedicated CFD codes in performance.
- [Langtangen, H.P. and Logg, A. — *FEniCS Tutorial*](https://github.com/hplgit/fenics-tutorial) (channels [ft07](https://github.com/hplgit/fenics-tutorial/blob/master/pub/python/vol1/ft07_navier_stokes_channel.py), [ft08](https://github.com/hplgit/fenics-tutorial/blob/master/pub/python/vol1/ft08_navier_stokes_cylinder.py)) — The classic IPCS channel flow and cylinder examples that introduced many users to FEniCS CFD. Our initial velocity and pressure stepping follow this lineage.
- [Kuchta, M. — ns-fenics](https://github.com/MiroK/ns-fenics) — Collection of FEniCS Navier-Stokes solvers comparing IPCS, mixed, and Yanenko schemes. Useful for understanding trade-offs between splitting approaches.

**RANS turbulence in FEniCS — prior art:**

- [Mortensen, M. and Langtangen, H.P. — *CBC.RANS*](https://launchpad.net/cbc.rans) ([paper](https://arxiv.org/abs/1102.2933)) — The original FEniCS RANS framework (2011). Demonstrated that k-ε and other RANS models could be implemented compactly in FEniCS's variational language. Our approach to decoupling turbulence equations from momentum (Picard iteration with under-relaxation) parallels CBC.RANS, though we target k-ω on modern DOLFINx rather than k-ε on legacy DOLFIN.
- [Langtangen, H.P. — *cbc.pdesys*](https://github.com/hplgit/fenics-tutorial/blob/master/.sandbox/cbcpdesys/pdesys-sphinx/_sources/pdesys.txt) — Langtangen's PDE system framework for FEniCS, which supported full RANS turbulence hierarchies: from Spalart-Allmaras through k-ε, k-ω SST, v2f, up to Reynolds stress models. Demonstrated how to decompose large coupled PDE systems into segregated subsystems with flexible linearization (Picard/Newton). The architectural pattern of "subsystem per equation + outer coupling loop" influenced our solver design.
- [Valen-Sendstad, K. and Mortensen, M. — *Implementing a k-ε Turbulence Model in FEniCS*](https://www.researchgate.net/publication/51994398_Implementing_a_k-epsilon_Turbulence_Model_in_the_FEniCS_Finite_Element_Programming_Environment) — Early paper on FEM RANS in FEniCS. Documents the same positivity and stability challenges we address here (clipping, under-relaxation, convection dominance).
- [joove123 — *k-epsilon*](https://github.com/joove123/k-epsilon) — Lam-Bremhorst k-ε implementation in FEniCS (master's thesis, Univ. Southern Denmark). Validates on channel flow and backward-facing step — the same canonical geometries. Shows how wall-damping functions can be handled in FEniCS's variational framework.
- [FEniCS 2024 Conference — *Implementation of k-ε in FEniCS*](https://scientificcomputing.github.io/fenics2024/implementation-of-the) (Marcibál, Univ. Southern Denmark) — Recent work confirming continued community interest in FEniCS RANS. Uses DOLFINx with dolfinx_mpc for periodic hills — similar stack to this solver.
- [FEniCS Discourse — *FEniCSx/DOLFINx implementation for turbulent*](https://fenicsproject.discourse.group/t/fenicsx-dolfinx-implementation-for-turbulent/14427) — Thread where users ask whether FEniCSx is suitable for RANS. Dokken points to the k-ε repo above as a starting point and notes it "should not be hard to convert to DOLFINx." This thread captures the community gap this solver aims to fill.
- [FEniCS Discourse — *Implementing k-ε: variational terms depending on a solution*](https://fenicsproject.discourse.group/t/implementing-k-epsilon-model-in-fenics-how-to-implement-terms-in-variational-formulation-that-depend-on-a-solution/13381) — Community discussion on the core challenge of RANS in FEM: how to handle terms like $P_k = \nu_t |S|^2$ where both $\nu_t$ and $|S|$ depend on the solution.
- [FEniCS Discourse — *BC application with dolfinx_mpc nonlinear solver*](https://fenicsproject.discourse.group/t/bc-application-when-using-non-linear-solver-dolfinx-mpc/18431) — RANS on a periodic hill with dolfinx_mpc; boundary condition handling challenges that parallel our periodic channel setup.

**Turbulence model specifications and verification:**

- [NASA Turbulence Modeling Resource — *Wilcox k-ω 2006*](https://turbmodels.larc.nasa.gov/wilcox.html) — Definitive reference for model constants, equations, and verification test cases. We cross-checked our constants against this page.
- [OpenFOAM — kOmega2006 source](https://cpp.openfoam.org/v12/classFoam_1_1RASModels_1_1kOmega2006.html) ([header](https://cpp.openfoam.org/v13/kOmega2006_8H_source.html)) — OpenFOAM's C++ implementation of the same Wilcox 2006 model. Useful for verifying constant values ($\beta_0 = 0.0708$, $\sigma_k = 0.6$, $\sigma_d = 1/8$, $C_\text{lim} = 7/8$) and cross-diffusion term structure.
- [CFD-Wiki — *Wilcox's modified k-ω model*](https://www.cfd-online.com/Wiki/Wilcox%27s_modified_k-omega_model) — Documents the 2006 updates (cross-diffusion, stress limiter) vs the 1998 baseline.
- [CFD Notes — *Enhancements to the k-ω model*](https://doc.cfd.direct/notes/cfd-general-principles/enhancements-to-the-k-omega-model) — Clear explanation of cross-diffusion and stress-limiter terms with derivations.

**Cross-code validation target:**

- [Nek5000 RANS Channel Tutorial](https://nek5000.github.io/NekDoc/tutorials/rans.html) — Our primary validation reference. Nek5000 uses a regularized k-ω in spectral elements; comparing our FEM k-ω output against Nek's RANS output confirms implementation correctness independent of spatial discretization.
- [Nek5000/NekExamples](https://github.com/Nek5000/NekExamples) — Source for the `poiseuille_RANS` case files.

---

## License

MIT License. See [LICENSE](LICENSE).
