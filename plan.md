# Plan: Model-Agnostic Turbulence Solver Refactoring

Date: 2026-02-17 (revised)

## Context

The solver currently supports 3 turbulence models (k-ω Wilcox 2006, k-ω SST, k-ε Lam-Bremhorst) but implements them via **37 conditional branches** scattered across `solver.py`. Adding or comparing models requires touching every branch. The reference repo (joove123/k-epsilon) uses a class hierarchy but only supports one model family.

**Goal**: Refactor into a model-agnostic solver where:
- Each model is a self-contained class
- The solver has zero model-specific branches
- Running any case (channel, BFS) with any model requires only a config change (`"model": "wilcox2006"` / `"sst"` / `"kepsilon"`)
- Adding a new model = adding one class file, zero solver changes

---

## 1. Equation Equivalence Audit (vs joove123/k-epsilon)

All 3 models fit the same generic 2-equation transport template:

### k-equation (universal)
```
dk/dt + u·∇k = ∇·((ν + σ_k·ν_t)∇k) + P_k - R_k·k
```
| Model | σ_k | P_k | R_k |
|-------|-----|-----|-----|
| Wilcox 2006 | 0.6 | min(ν_t·S², 10β\*kω) | β\*·ω |
| SST | F1-blended (0.85↔1.0) | min(ν_t·S², 10β\*kω) | β\*·ω |
| k-ε LB | 1.0 | ν_t·S² | ε/k |

### Scalar equation (ω or ε)
```
dφ/dt + u·∇φ = ∇·((ν + σ_φ·ν_t)∇φ) + P_φ - R_φ·φ + CD
```
| Model | σ_φ | P_φ | R_φ | CD |
|-------|-----|-----|-----|----|
| Wilcox 2006 | 0.5 | γ·S² | β₀·ω | σ_d/ω·max(0,∇k·∇ω) |
| SST | F1-blended (0.5↔0.856) | γ(F1)·S² | β(F1)·ω | (1-F1)·2σ_w2/ω·max(0,∇k·∇ω) |
| k-ε LB | 1.3 | C₁·f₁·(ε/k)·ν_t·S² | C₂·f₂·(ε/k) | 0 |

### ν_t closure
| Model | Formula |
|-------|---------|
| Wilcox 2006 | k / max(ω, C_lim·\|S\|/√β\*) |
| SST | a₁·k / max(a₁·ω, \|S\|·F₂) |
| k-ε LB | C_μ·f_μ·k²/ε |

### Wall BCs
| Model | k wall | scalar wall |
|-------|--------|-------------|
| k-ω family | k=0 (Dirichlet) | ω = 6ν/(β₀·y₁²) (Dirichlet) |
| k-ε LB | k=0 (Dirichlet) | no Dirichlet (damping handles near-wall) |

### Damping functions (k-ε Lam-Bremhorst)
```
f_μ = (1 - exp(-0.0165·Re_k))² · (1 + 20.5/Re_t),  clamped to [0.01116, 1.0]
f_1 = 1 + (0.05/f_μ)³
f_2 = 1 - exp(-Re_t²)
Re_t = k²/(ν·ε),  Re_k = √k·y/ν
```

**Verdict**: Our equations match joove exactly for k-ε LB. All 3 models share the same template — only coefficients differ.

### Solver structure comparison
| Aspect | Our solver | joove |
|--------|-----------|-------|
| Channel momentum | IPCS (AB2/CN, 2nd order) | IPCS (1st order) |
| BFS momentum | IPCS (Picard, 1st order) | Mixed coupled (MUMPS) |
| Turbulence coupling | Decoupled (NS then k/φ) | Same |
| Time stepping | Adaptive CFL + residual growth | CFL-only |
| Convergence | max(res_u, res_k, res_φ) | L² all fields |

Our solver is more general and robust.

---

## 2. Architecture: "Plugin Coefficients" Pattern

The model provides UFL expressions/Functions. The solver plugs them into a fixed generic template. The model does NOT build complete forms.

```
solver.py (owns template)          models/*.py (owns coefficients)
┌───────────────────────┐          ┌──────────────────────┐
│ IPCS momentum         │          │ sigma_k, sigma_phi   │
│ Picard iteration      │◄─────────│ production_k/phi     │
│ Adaptive dt           │ FormCoeffs│ reaction_k/phi      │
│ Generic k-eq template │          │ cross_diffusion      │
│ Generic φ-eq template │          │ compute_nu_t()       │
│ Diagnostics / I/O     │          │ update_auxiliary()   │
└───────────────────────┘          │ wall/inlet BCs       │
                                   └──────────────────────┘
```

### File structure
```
src/dolfinx_rans/models/
    __init__.py          # Factory: create_model("wilcox2006") → instance
    base.py              # ABC: RANSModel + FormCoefficients dataclass
    wilcox2006.py        # k-ω Wilcox 2006 (~150 lines)
    sst.py               # k-ω SST Menter 1994 (~200 lines)
    kepsilon.py          # k-ε Lam-Bremhorst (~180 lines)
```

### Base class interface (`base.py`)

```python
class RANSModel(ABC):
    # --- Metadata ---
    model_name: str          # "wilcox2006", "sst", "kepsilon"
    display_name: str        # "k-ω (Wilcox 2006)"
    scalar_name: str         # "ω" or "ε"

    # --- Lifecycle (called once) ---
    def setup(domain, S, k_prev, scalar_prev, k_n, scalar_n,
              nu_t_, u_n, nu_c, nu, turb_params,
              wall_facets, is_bfs, geom) → None

    # --- Form coefficients (UFL, called once for form construction) ---
    def get_form_coefficients() → FormCoefficients

    # --- Field specs (clipping, wall BC metadata) ---
    def get_scalar_field_specs() → (k_spec, phi_spec)

    # --- Wall distance ---
    def needs_wall_distance() → bool
    def compute_wall_distance(S, wall_facets, is_bfs, geom) → Function | None

    # --- Per-Picard-iteration (numpy) ---
    def update_auxiliary_fields(k_arr, scalar_arr) → None
    def compute_nu_t(k_arr, scalar_arr, S_mag_arr) → np.ndarray

    # --- Initial conditions ---
    def convert_omega_to_scalar_ic(omega_arr, k_arr) → np.ndarray
    def initial_nu_t(k_arr, scalar_arr) → np.ndarray
    def compute_inlet_scalar(k_inlet, omega_inlet) → float
```

### FormCoefficients dataclass
```python
@dataclass
class FormCoefficients:
    sigma_k: UFL              # k diffusion Prandtl number
    sigma_phi: UFL            # scalar diffusion Prandtl number
    nu_t_diffusion_k: UFL     # ν_t for k diffusion (Function or UFL expr)
    nu_t_diffusion_phi: UFL   # ν_t for φ diffusion
    production_k: UFL         # P_k source
    reaction_k: UFL           # R_k (destruction = R_k · k)
    production_phi: UFL       # P_φ source
    reaction_phi: UFL         # R_φ (destruction = R_φ · φ)
    cross_diffusion: UFL      # CD term (0 for k-ε)
```

### Generic form template (replaces 130 lines of branching in solver.py)
```python
c = model.get_form_coefficients()

F_k = (k_trial - k_n)/dt * phi_k * dx
    + dot(u_n, grad(k_trial)) * phi_k * dx
    + (nu_c + c.sigma_k * c.nu_t_diffusion_k) * inner(grad(k_trial), grad(phi_k)) * dx
    + c.reaction_k * k_trial * phi_k * dx
    - c.production_k * phi_k * dx

F_w = (w_trial - scalar_prev)/dt * phi_w * dx
    + dot(u_n, grad(w_trial)) * phi_w * dx
    + (nu_c + c.sigma_phi * c.nu_t_diffusion_phi) * inner(grad(w_trial), grad(phi_w)) * dx
    + c.reaction_phi * w_trial * phi_w * dx
    - c.production_phi * phi_w * dx
    - c.cross_diffusion * phi_w * dx
```

---

## 3. Branch-by-branch replacement map (solver.py)

| Lines | Current branch | Replaced by |
|-------|---------------|-------------|
| 421-426 | Model name validation | `model = create_model(turb.model)` |
| 434 | `use_kepsilon = ...` | Removed |
| 475 | `omega_wall_val = 0 if use_kepsilon...` | `phi_spec.wall_value` |
| 483-484 | scalar field label | `model.scalar_name` |
| 520 | `use_sst = ...` | Removed |
| 522-528 | Model label printing | `model.display_name` |
| 531-559 | Wall distance + SST F1/F2 setup | `model.compute_wall_distance()` + `model.setup()` |
| 570-591 | sigma_k, sigma_w, beta, gamma constants | `model.get_form_coefficients()` |
| 593-595 | scalar_label, min/max clip | `model.get_scalar_field_specs()` |
| 612-625 | Initial ω→ε conversion | `model.convert_omega_to_scalar_ic()` |
| 634-638 | Initial ν_t | `model.initial_nu_t()` |
| 662-666 | Inlet BC for ε vs ω | `model.compute_inlet_scalar()` |
| 687-691 | Wall BC for scalar | `phi_spec.needs_wall_dirichlet` |
| 694-706 | Wall BC logging | Model metadata |
| 713-840 | **Form construction** (k-eq + scalar-eq) | Generic template + `FormCoefficients` |
| 988 | CSV header label | `model.scalar_name` |
| 1107-1138 | SST blending update | `model.update_auxiliary_fields()` |
| 1178-1224 | ν_t computation (3-way) | `model.compute_nu_t()` |

---

## 4. Implementation Phases

### Phase 1: Foundation + Wilcox 2006 wiring
1. Create `models/base.py` (ABC + FormCoefficients)
2. Create `models/wilcox2006.py` (move constants from `config.py:21-32`, extract branch logic)
3. Create `models/__init__.py` (factory)
4. Wire into `solver.py` — replace Wilcox branches with model API
5. **Verify**: `channel/channel.jsonc` → history.csv identical pre/post

### Phase 2: SST model class
1. Create `models/sst.py` (constants from `config.py:40-53`, F1/F2 from `solver.py:1107-1138`)
2. Remove SST branches from `solver.py`
3. **Verify**: create and run an SST config

### Phase 3: k-ε Lam-Bremhorst
1. Create `models/kepsilon.py` (constants from `solver.py:105-110`, damping from `solver.py:718-748`)
2. Remove k-ε branches from `solver.py`
3. **Verify**: create and run a k-ε config

### Phase 4: Cleanup + cross-model configs
1. Move remaining model constants from `config.py` into model classes (keep shared `KAPPA`, `BETA_0`)
2. Remove `KEPSILON_*` from `solver.py`
3. Create 6 configs: {channel, bfs} × {wilcox2006, sst, kepsilon}
4. Clean imports

### Phase 5: Comparison runs
1. Run all 6 combinations
2. Overlay profiles (u+, k+, Cf) for model comparison
3. Document results

---

## 5. What stays unchanged

- `geometry.py` — mesh creation, boundary marking, ICs (all model-agnostic)
- `utils.py` — diagnostics, I/O (model-agnostic)
- `plotting.py` — visualization (model-agnostic)
- IPCS momentum scheme in `solver.py` — uses `nu_t_` Function (updated by model)
- Picard iteration loop structure — calls model methods at defined points
- Adaptive time stepping — model-agnostic
- `TurbParams` dataclass — no structural change; models read the fields they need

## 6. Existing code to reuse

- `geometry.py:compute_wall_distance_eikonal` — called by models that need wall distance
- `geometry.py:compute_wall_distance_channel` — called by k-ε for channel geometry
- `geometry.py:initial_omega_*` — all models use omega ICs; k-ε converts via model method
- `config.py:BETA_0, KAPPA` — shared turbulence constants (also used by geometry.py)

## 7. Verification protocol

1. **Regression**: history.csv at iterations 10, 50, 100 must be identical pre/post for existing configs
2. **Smoke**: reduced mesh (Nx=8, Ny=8), 10 iters, no NaN for all 3 models
3. **Interface**: grep `solver.py` for `use_kepsilon`, `use_sst` → zero matches after Phase 3
4. **Constant isolation**: grep `solver.py` for `BETA_STAR`, `SST_A1`, `KEPSILON_CMU` → zero after Phase 4
5. **Cross-model**: run all 6 case×model combinations, overlay profiles

## 8. Risks

| Risk | Mitigation |
|------|-----------|
| UFL expr vs Function in forms | Already mixed in current code — UFL handles both |
| SST Functions need scatter_forward | Model's `update_auxiliary_fields` owns scatter |
| S_sq needed by models | Model computes from `u_n` received in `setup()` |
| Cross-diffusion ∇k·∇ω | SST: internal Function; Wilcox: UFL dot product; k-ε: zero |
| Config backward compatibility | `TurbParams` unchanged; models read needed fields |
