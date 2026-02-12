# LOG

Last updated: February 10, 2026

## Scope Summary

1. Project was moved to high-Re workflow only.
2. Legacy `Re_tau=590` references/artifacts were removed.
3. Main high-Re case is `examples/channel_nek_re125k_like.json`.
4. Main blocker now is a 4-rank teardown crash in periodic/MPC runs.

## Environment Snapshot

- OS: Pop!_OS/Linux (from MPI runtime output)
- MPI: Open MPI `4.1.6`
- DOLFINx: `0.10.0`
- petsc4py: `3.24.4`
- mpi4py: `4.1.1`
- Active solver branch point: `fd19bc5` (`chore: remove re590 artifacts and switch to high-Re-only workflow`)

## Repro Configs Used

- `/tmp/channel_nek_4proc.json`
  - `use_body_force=true` (periodic path)
  - `max_iter=800`
  - `snapshot_interval=0`
  - `out_dir=re5200_nek_4proc`
- `/tmp/channel_nek_4proc_smoke.json`
  - `use_body_force=true` (periodic path)
  - `max_iter=2`
  - `snapshot_interval=0`
  - `out_dir=re5200_nek_4proc_smoke`
- `/tmp/channel_open_4proc_smoke.json`
  - `use_body_force=false` (non-periodic path)
  - `max_iter=2`
  - `snapshot_interval=0`
  - `out_dir=re5200_open_smoke_4proc`

## Test Matrix (What Was Actually Run)

1. `./run.sh 4 /tmp/channel_nek_4proc.json`
   - Result: solver advanced to iteration 800 with reasonable residual trend.
   - Final console error: `corrupted size vs. prev_size` + MPI abort (`exit 134`, signal 6).
   - Notes: `history.csv` updated during run; teardown crash occurs after compute phase.

2. `./run.sh 4 /tmp/channel_nek_4proc_smoke.json`
   - Result: 2 iterations run, then same teardown allocator failure.
   - Error signature repeated: `corrupted size vs. prev_size` (or `free(): invalid size` in earlier runs).

3. `./run.sh 4 /tmp/channel_open_4proc_smoke.json` (re-verified on current HEAD)
   - Result: process exits cleanly (no allocator crash).
   - Caveat: this short non-periodic setup is numerically unstable/poorly conditioned in 2 steps, but shutdown is clean.

4. Minimal import test under MPI
   - Command pattern: `mpirun -np 4 python -c "from mpi4py import MPI; import dolfinx_rans.plotting ..."`
   - Result: clean exit, no teardown crash.

5. Minimal MPC creation/finalize script under MPI
   - Script created `MultiPointConstraint` on multiple spaces and finalized constraints.
   - Result: clean exit, no teardown crash.

## Strong Inference So Far

- Crash is strongly associated with the periodic/body-force solve path (`use_body_force=true`) in full solver execution, not with generic MPI startup or plain plotting imports.
- The failure appears in shutdown/finalization phase after solve progress, but the root cause may still be memory corruption earlier in the periodic solve loop.

## Debug Experiments Already Tried (And Reverted)

The following solver edits were tested during diagnosis and then rolled back (to avoid carrying uncertain changes):

1. Removed explicit PETSc `destroy()` cleanup calls.
2. Disabled selected/all `mpc.backsubstitution()` calls.
3. Replaced built-in MPC backsubstitution with manual slave reconstruction.
4. Avoided in-place MPC vector assembly (`mpc_assemble_vector`) by using temporary vectors.
5. Avoided pressure nullspace removal call in periodic branch.
6. Added temporary keepalive references for MPC objects on returned Functions.

Outcome:
- None produced a robust fix for periodic 4-rank teardown crash.
- Some combinations avoided immediate crash in one scenario but either failed elsewhere or destabilized the run.
- All these code experiments were discarded; `src/dolfinx_rans/solver.py` is back to baseline.

## Reproduce Error (Primary)

```bash
./run.sh 4 /tmp/channel_nek_4proc.json
```

Expected:
1. Pre-flight checks pass.
2. Solver runs iterations.
3. MPI abort near/after end with allocator corruption message.

## Handoff Plan For Next Agents

1. Reproduce on clean baseline first.
   - `git status` should only show expected local docs/log changes.
   - Run the exact primary repro command above.

2. Build a bisect-style minimization script inside Python (no `run.sh`) that toggles one subsystem at a time:
   - periodic MPC setup only
   - matrix/vector assembly only
   - solve momentum only
   - add pressure correction
   - add k/omega solves
   - add postprocessing
   - objective: identify the smallest step that introduces heap corruption.

3. Run an MPI process-count sweep on periodic case:
   - `np=1,2,3,4,8` and record first failing `np`.
   - this can reveal race/partition-sensitive paths.

4. Check dependency ABI compatibility in environment.
   - verify that `dolfinx`, `dolfinx_mpc`, PETSc, and Open MPI packages come from consistent build stack.
   - mismatch here can produce exactly this class of shutdown corruption.

5. Add a temporary CI/debug script (not committed) that captures:
   - command
   - return code
   - final 50 log lines
   - run timestamp
   - then remove script before finalizing.

6. If root fix is not found quickly, implement a pragmatic short-term mode:
   - document parallel periodic case as known-broken in this environment,
   - keep serial periodic or parallel non-periodic for day-to-day dev checks.

## Practical Notes

- This repo currently has no unresolved source changes from debug patching; only `LOG.md` is local.
- Old result directories from earlier runs were cleaned before this update.

## Session Update (February 10, 2026, Follow-up)

### What Was Re-Checked

1. Environment consistency in active `fenicsx` env:
   - `mpirun`: `/home/rfrantz/miniforge3/envs/fenicsx/bin/mpirun` (MPICH/HYDRA 4.3.2)
   - `dolfinx`: `0.10.0`
   - `dolfinx_mpc`: `0.10.0`
   - `petsc4py`: `3.24.4`
   - `mpi4py`: `4.1.1`
2. Primary periodic smoke repro still fails:
   - `./run.sh 4 /tmp/channel_nek_4proc_smoke.json`
   - allocator abort remains reproducible (`corrupted size vs. prev_size` / `free(): invalid size`)
3. Non-periodic control remains clean:
   - `./run.sh 4 /tmp/channel_open_4proc_smoke.json`
   - exits with code 0

### Additional Minimization Runs

1. `max_iter=0` periodic (via `run.sh`) can exit cleanly in some runs.
2. `max_iter=1` periodic crashes.
3. `max_iter=1, picard_max=0` (no inner momentum/pressure/k/omega solves) still crashes.
4. `max_iter=1, picard_max=0, log_interval=1000` (no per-step diagnostics table/CSV row) still crashes.
5. Solver-only script (no CLI plotting/final summary) still crashes after solver return in periodic mode.
6. Repeated solver-only `max_iter=0` periodic runs still show teardown aborts in this session.

Conclusion from minimization:
- Full physics solves are not required to trigger the bug.
- Entering periodic solver path with MPC-backed objects is sufficient to produce nondeterministic heap corruption symptoms in teardown.

### Temporary Probe Patches (All Reverted)

1. Added keepalive references for MPC objects on returned fields/spaces.
2. Added explicit CLI-side teardown ordering and PETSc garbage cleanup.
3. Added solver debug markers around loop exit and cleanup (`destroy`, `hist.close`, return).
4. Added temporary env-gated skips for explicit PETSc destroy and selected outer-loop operations.

Outcome:
- Crash location moved between runs (during matrix destroy, just after solver return, or at interpreter shutdown), but no robust fix.
- All probe code was reverted; source files are back to baseline.

### Online Findings (Checked This Session)

1. FEniCS discourse thread on MPC lifetime:
   - `https://fenicsproject.discourse.group/t/mpc-lifetime-of-the-multipointconstraint/17250`
   - Key point: `MultiPointConstraint` must outlive objects tied to `mpc.function_space`; otherwise memory corruption/out-of-bounds is possible.
2. `dolfinx_mpc` release notes:
   - `https://github.com/jorgensd/dolfinx_mpc/releases`
   - `v0.10.1` notes include critical sparsity-pattern fixes (`#216/#215`) that are not present in installed `0.10.0`.
3. Historical parallel periodic guidance:
   - `https://github.com/jorgensd/dolfinx_mpc/issues/15`
   - use `mpc.function_space` for solution fields in parallel periodic settings (already satisfied in this solver).

### Updated Inference

- The periodic/MPC branch remains the only reliable trigger.
- Failure is consistent with earlier heap corruption detected late in teardown, not a deterministic single bad cleanup call.
- Given package state (`dolfinx_mpc 0.10.0`), an upstream MPC/library defect remains a strong candidate.
- Highest-value next verification is to test same reproducer against newer `dolfinx_mpc` build (>= `0.10.1`/main).

## Step 1 Outcome: Upgrade Test to `dolfinx_mpc 0.10.1`

### Isolated Upgrade Procedure

Performed in a cloned environment to avoid touching baseline:

1. `conda create --clone fenicsx -n fenicsx-mpc-main`
2. Removed old MPC packages from clone:
   - `conda remove dolfinx_mpc libdolfinx_mpc`
3. Built and installed `dolfinx_mpc` C++ library from source tag `v0.10.1`:
   - repo: `https://github.com/jorgensd/dolfinx_mpc.git`
   - tag: `v0.10.1` (`a444aa3006fdf492091443cc8c885c1eec006c2f`)
   - `cmake -S cpp -B build -G Ninja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX`
   - `cmake --build build --parallel && cmake --install build`
4. Installed Python package from same tag:
   - `python -m pip install --no-deps --no-build-isolation /tmp/dolfinx_mpc_v0101/python`
5. Verified in clone:
   - `dolfinx=0.10.0`, `dolfinx_mpc=0.10.1`, `petsc4py=3.24.4`, `mpi4py=4.1.1`, MPICH runtime.

### Repro Check on Upgraded Env

Command:

```bash
mpirun -np 4 python -m dolfinx_rans /tmp/channel_nek_4proc_smoke_mpc101.json
```

Result:
- Still fails with allocator corruption (`corrupted size vs. prev_size`) and MPI abort.
- Exit code: nonzero (`134` / terminated by MPI cleanup).

### MPI Process Sweep (A/B)

Same periodic smoke setup (`max_iter=2`, `snapshot_interval=0`) run in both envs:

1. Baseline `fenicsx` (`dolfinx_mpc 0.10.0`)
   - `np=1`: clean exit
   - `np=2`: clean exit
   - `np=3`: clean exit
   - `np=4`: crash (`free(): invalid size` / `corrupted size vs. prev_size`)
2. Upgraded `fenicsx-mpc-main` (`dolfinx_mpc 0.10.1`)
   - `np=1`: clean exit
   - `np=2`: clean exit
   - `np=3`: clean exit
   - `np=4`: crash (`corrupted size vs. prev_size`)

Conclusion:
- Upgrading from `dolfinx_mpc 0.10.0` to `0.10.1` does **not** resolve this repo’s periodic crash.
- First observed failing rank remains `np=4` in both environments.

## Stage Bisect Update (February 10, 2026, same session)

To isolate beyond process count, temporary standalone scripts were run under MPI (`np=4`) from `/tmp`:

1. `/tmp/mpc_stage_bisect.py` (coarse staged enablement)
2. `/tmp/mpc_stage3_bisect.py` (stage-3 micro steps)
3. `/tmp/mpc_symmetry_bc_bisect.py` (symmetry BC operation-level micro steps)

### Coarse Stages (`/tmp/mpc_stage_bisect.py`)

Baseline (`dolfinx_mpc 0.10.0`) and upgraded (`0.10.1`) both showed:

1. Stage 0 (mesh/spaces): clean
2. Stage 1 (MPC create/finalize): clean
3. Stage 2 (functions + initial backsubstitution): clean
4. Stage 3 (reduced-space BC setup): first failing stage (allocator abort)
5. Stage 4 (one-time assembly): also fails

### Stage-3 Micro-Bisect (`/tmp/mpc_stage3_bisect.py`, baseline env)

Levels are cumulative on top of Stage-2 baseline:

1. Level 0: no extra stage-3 ops → clean
2. Level 1: `locate_dofs_topological(V, wall_facets)` → clean
3. Level 2: `locate_dofs_topological(S, wall_facets)` → clean
4. Level 3: `dirichletbc(u_noslip, wall_dofs_V_tb, V)` → clean
5. Level 4: **add symmetry subspace dof location on reduced space** → crash appears
   - operation: `locate_dofs_topological((V.sub(1), V_sub_collapsed), fdim, top_facets)`
6. Levels 5/6: also crash-prone (as expected after level 4)

### Symmetry Operation-Level Bisect (`/tmp/mpc_symmetry_bc_bisect.py`)

On top of known-clean baseline, cumulative micro-steps:

1. Micro 0: none of symmetry-subspace ops → clean
2. Micro 1: `V_y_sub = V.sub(1)` → clean
3. Micro 2: `V_y_collapsed = V_y_sub.collapse()` → clean
4. Micro 3: create zero Function on collapsed subspace → clean
5. Micro 4: **tuple-based top dof location** → first repeatable crash trigger
6. Micro 5: dirichletbc creation may crash or pass depending on run (nondeterministic after trigger)

Key inference:
- The critical trigger is locating subspace dofs on the MPC-reduced space for symmetry BC, not matrix assembly or KSP solve itself.

### Additional Confirmation

1. Full-channel variant (`use_symmetry=false`, tolerance relaxed for mesh consistency) of stage-3 micro-bisect runs clean.
   - Supports symmetry-subspace path as trigger.
2. `dolfinx_mpc 0.10.1` still exhibits the same trigger point at micro-level 4.

## Candidate Fix Tested In-Repo (Current Working Tree)

### Change

In periodic mode, the velocity BC list for MPC assembly now reuses original-space BC objects (`bcs_u0`) instead of reconstructing symmetry BC on reduced space `V.sub(1)` during solve-loop setup.

Rationale:
- Avoids the identified crash trigger (`locate_dofs_topological((V.sub(1), V_sub_collapsed), ...)` on reduced space).
- Keeps top symmetry BC semantics via original BC set already used during periodic constraint setup.

### Validation on Current Working Tree

1. `./run.sh 4 /tmp/channel_nek_4proc_smoke.json`
   - repeated 3x
   - all 3 exits clean (`EXIT=0`)
   - no allocator corruption messages
2. Physics sanity for smoke run:
   - top symmetry still effectively enforced (`v` range approximately `[-0.0000, 0.0000]`)
3. Extended periodic check:
   - `/tmp/channel_nek_4proc_50_fix.json` (`max_iter=50`) completed compute and teardown cleanly
   - nonzero exit was due to configured regression gate (`tau_wall` threshold), **not** allocator crash
4. Non-periodic control:
   - `./run.sh 4 /tmp/channel_open_4proc_smoke.json` still exits cleanly

## Structured Rerun In Standard Results Layout (February 10, 2026, late session)

To remove ad-hoc investigation folders and keep artifacts in the normal case layout, the periodic 4-rank benchmark was rerun using:

1. Config file on disk (not `/tmp`): `re5200_nek/run_config.json`
   - based on `examples/channel_nek_re125k_like.json`
   - `solve.out_dir = "re5200_nek"`
   - `solve.snapshot_interval = 200`
   - `solve.max_iter = 800`
2. Command:
   - `./run.sh 4 re5200_nek/run_config.json | tee re5200_nek/run.log`

### Observed Outcome

1. Solver completed full 800 iterations.
2. No allocator-teardown signatures (`corrupted size`, `free(): invalid size`) in `re5200_nek/run.log`.
3. Finalization/plotting completed:
   - `Saved final fields plot: re5200_nek/final_fields.png`
   - `Results saved to re5200_nek/`
4. Regression gate still fails on wall shear:
   - `tau_wall=0.5207` outside configured `[0.90, 1.10]`
   - This is a physics/threshold issue, not the previous teardown crash.

### Files Now Present Under `re5200_nek/`

- `config_used.json`, `run_info.json`, `history.csv`
- `mesh.png`, `fields.png`, `fields_00200.png`, `fields_00400.png`, `fields_00600.png`, `fields_00800.png`
- `final_fields.png`, `convergence.png`, `profiles.csv`
- `snps/velocity.bp/*`, `snps/turbulence.bp/*`
- `run.log`

## Extended Convergence Test (Same Mesh) — `re5200_nek_long`

Objective: test whether low `tau_wall` is primarily an iteration-count issue on the existing mesh.

### Config / Command

1. Config: `re5200_nek_long/run_config.json`
   - based on `examples/channel_nek_re125k_like.json`
   - `geom`: unchanged (`Nx=96`, `Ny=96`, `growth_rate=1.05`, `y+_actual=2.43`)
   - `solve.max_iter=2500`
   - `solve.log_interval=50`
   - `solve.snapshot_interval=500`
2. Command:
   - `./run.sh 4 re5200_nek_long/run_config.json | tee re5200_nek_long/run.log`

### Outcome

1. Completed all 2500 iterations with clean teardown (no allocator crash signatures).
2. Final checkpoint:
   - `iter=2500`
   - `tau_wall=0.6533`
   - `U_bulk=15.6382`
   - `residual=3.2e-04`
3. Regression gate still fails (`tau_wall` below `[0.90, 1.10]`).

### Artifacts

- `re5200_nek_long/run.log`
- `re5200_nek_long/history.csv`
- `re5200_nek_long/profiles.csv`
- `re5200_nek_long/final_fields.png`
- `re5200_nek_long/convergence.png`
- `re5200_nek_long/fields_00500.png` ... `fields_02500.png`
- `re5200_nek_long/snps/*`

## Refined Near-Wall + More Elements Test — `re5200_nek_refined`

Objective: test user hypothesis that SEM-equivalent resolution may require more elements and tighter near-wall spacing in this finite-element setup.

### Config / Command

1. Config: `re5200_nek_refined/run_config.json`
   - `geom.Nx=192`
   - `geom.Ny=114`
   - `geom.growth_rate=1.05`
   - `geom.y_first=1.9277963481490493e-04` (implied by closure; `y+_actual=1.00`)
   - `solve.max_iter=1500`
   - `solve.log_interval=50`
   - `solve.snapshot_interval=300`
2. Command:
   - `./run.sh 4 re5200_nek_refined/run_config.json | tee re5200_nek_refined/run.log`

### Outcome

1. Completed all 1500 iterations with clean teardown (no allocator crash signatures).
2. Final checkpoint:
   - `iter=1500`
   - `tau_wall=0.4700`
   - `U_bulk=15.5171`
   - `residual=3.4e-04`
3. Regression gate still fails (`tau_wall` below `[0.90, 1.10]`).
4. Compared at equal-ish iteration windows, this refined setup trends to lower `tau_wall` than the coarse setup in the current transient window.

### Artifacts

- `re5200_nek_refined/run.log`
- `re5200_nek_refined/history.csv`
- `re5200_nek_refined/profiles.csv`
- `re5200_nek_refined/final_fields.png`
- `re5200_nek_refined/convergence.png`
- `re5200_nek_refined/fields_00300.png` ... `fields_01500.png`
- `re5200_nek_refined/snps/*`

## Nek Profile Export + Overlay Comparison (from `../nekStab/example/poiseuille_RANS`)

User-requested step: read Nek case data using the local `plot.py` workflow and export profile data for direct comparison.

### Source Case and Access Method

1. Source case path:
   - `../nekStab/example/poiseuille_RANS`
2. Script inspected:
   - `../nekStab/example/poiseuille_RANS/plot.py`
3. Data access method used (same helper stack as plot script):
   - `../nekStab/example/nekplot.py`
   - `nekplot.read_field(...)` on:
     - `../nekStab/example/poiseuille_RANS/BF_poiseuille_RANS0.f00001`

### Export/Comparison Outputs (all on disk)

Folder: `re5200_nek_compare/`

1. Nek exports:
   - `nek_profile_outer.csv`
   - `nek_profile_outer.dat`
   - `nek_profile_wall_units_assumed.csv`
2. DOLFINx exports used in overlay:
   - `re5200_nek_long_outer.csv`
   - `re5200_nek_refined_outer.csv`
3. Plots:
   - `comparison_outer.png`
   - `comparison_wall_units_assumed.png`
4. Metadata/assumptions:
   - `comparison_metadata.json`

### Key Metadata Captured

1. Nek baseflow time from field file: `196.5509840257`
2. Nek domain extents from field:
   - `x ∈ [0, 2π]`
   - `y ∈ [-1, 1]`
3. Nek outer-scale bulk velocity from extracted profile:
   - `U_bulk ≈ 0.99849`
4. Wall-units conversion for Nek marked as approximate:
   - derived from `poiseuille_RANS.par` viscosity convention (`viscosity=-1e5` interpreted as `Re=1e5`)

## Nek Reynolds Traceability (Precision Check Requested)

Direct file evidence used:

1. `../nekStab/example/poiseuille_RANS/poiseuille_RANS.par`
   - `[VELOCITY] viscosity = -1e5`
2. `../nekStab/example/poiseuille_RANS/poiseuille_RANS.log.4`
   - runtime echoes `viscosity = [-1e5]`
   - runtime parameter line reports `viscosity (1/Re) = 1.0e-5`
3. `../nekStab/example/poiseuille_RANS/README.md`
   - states `Re = 10^5` (bulk/half-height wording)

Conclusion:
- The Nek case is traceably `Re = 100000` by its own input/log convention.
- `Re_tau = 5200` is **not** from Nek files; it is from this repo’s config `examples/channel_nek_re125k_like.json`.

## Reusable Nek Extraction Script Added

New in-repo script:

- `src/dolfinx_rans/validation/nek_poiseuille_profile.py`

Purpose:
- Read Nek baseflow field using same helper stack as local Nek plot workflow.
- Export traceable profile files:
  - `nek_profile_outer.csv`
  - `nek_profile_outer.dat`
  - `nek_profile_wall_units.csv`
  - `nek_reference_profile.csv` (run.sh-compatible columns: `y_plus,u_plus`)
- Generate overlays:
  - `comparison_outer.png`
  - `comparison_wall_units.png`
- Write metadata including parsed `Re`, parsed `nu`, and estimated `Re_tau` from extracted wall gradient.

Run performed:

```bash
PYTHONPATH=src python src/dolfinx_rans/validation/nek_poiseuille_profile.py \
  --nek-case-dir ../nekStab/example/poiseuille_RANS \
  --out-dir nek_poiseuille_profile \
  --dolfinx-profiles re5200_nek_long/profiles.csv re5200_nek_refined/profiles.csv
```

Result highlights:
- `Re from par: 100000.0`
- `Estimated Re_tau from extracted baseflow: 1115.82`
- artifacts saved under `nek_poiseuille_profile/`

## Full-Channel (No Symmetry) Case With Nek-Linked Reference Gate

User-requested no-symmetry run and direct comparison against local Nek reference profile.

### Config

- `nek_re100k_fullchannel/run_config.json`
  - `geom.use_symmetry=false`
  - `geom.Ly=2.0` (full channel)
  - `nondim.Re_tau=1115.818661288065` (derived from extracted Nek baseflow metadata)
  - `benchmark.reference_profile_csv=../nek_poiseuille_profile/nek_reference_profile.csv`
  - `benchmark.u_plus_rmse_max=5.0`

### Important Runtime Note

1. First attempt with stretched full-channel mesh (`growth_rate=1.05`) failed in periodic MPC setup:
   - `RuntimeError: Newton method failed to converge for non-affine geometry`
2. Workaround used for successful run:
   - switched to uniform mesh (`growth_rate=1.0`, `y_first=0.0`)
   - kept periodic streamwise setup

### Successful Run (Controlled Window)

Command:

```bash
./run.sh 4 nek_re100k_fullchannel/run_config.json | tee nek_re100k_fullchannel/run.log
```

Outcome:

1. Completed cleanly to `max_iter=400` (no allocator crash).
2. Final values at iter 400:
   - `U_bulk=15.2533`
   - `tau_wall=0.0983`
3. Reference-profile gate result:
   - `u_plus RMSE=74.5987` (fails limit `5.0`)

Artifacts:
- `nek_re100k_fullchannel/run.log`
- `nek_re100k_fullchannel/history.csv`
- `nek_re100k_fullchannel/profiles.csv`
- `nek_re100k_fullchannel/final_fields.png`
- `nek_re100k_fullchannel/convergence.png`
- `nek_re100k_fullchannel/snps/*`

## 2026-02-12 - re100k Quad + Tanh Wall Refinement

User request:
- Keep a single canonical case (`re100k`), use quadrilateral mesh, and apply tanh near-wall refinement.

Code changes:
1. `src/dolfinx_rans/solver.py`
   - Added `geom.stretching` (`"geometric"` or `"tanh"`, default `"geometric"`).
   - Added tanh wall-stretch generator that solves for first-cell spacing from `y_first`.
   - Kept geometric generator as existing option.
   - Updated periodic MPC setup for stretched quad case to use topological constraints.
   - Added robust periodic relation mapping with small inward offset (`map_eps`) and practical tolerance.
2. `src/dolfinx_rans/cli.py`
   - Mesh header now prints stretching mode when wall refinement is active.
3. `run_re100k.sh`
   - Default generated `re100k/run_config.json` now uses:
     - `"mesh_type": "quad"`
     - `"stretching": "tanh"`
     - `"growth_rate": 1.0`

Validation run:
```bash
./run_re100k.sh
```

Observed startup diagnostics:
- `Mesh: 192×166 (quad, tanh)`
- `Wall-refined mesh: mode = tanh, y_first(requested) = 0.001605, y_first(actual) = 0.001605, y+_actual = 1.79`
- No `Newton method failed to converge for non-affine geometry` error.

Run completion:
- Reached `max_iter=50` and exited cleanly.
- Final table line:
  - `iter=50`, `res=1.2e-02`, `U_bulk=15.045`, `tau_wall=0.0807`

Artifacts written:
- `re100k/run_config.json`
- `re100k/results/mesh.png`
- `re100k/results/history.csv`
- `re100k/results/convergence.png`
- `re100k/results/final_fields.png`
- `re100k/results/profiles.csv`
- `re100k/results/fields.png`
- `re100k/results/fields_00050.png`
- `re100k/snps/velocity.bp/*`
- `re100k/snps/turbulence.bp/*`

## 2026-02-12 - Nek Extract Script in `nek_re100k/`

User request:
- Move/adapt extraction script into `nek_re100k/` and extract + plot from
  `nek_re100k/BF_poiseuille_RANS0.f00001`, adapting from Nek example workflow.

Implementation:
1. Added script:
   - `nek_re100k/nek_to_csv.py`
2. Source adaptation:
   - Uses helper path relative to script:
     `../../nekStab/example/poiseuille_RANS/` (for `nekplot.py`/`.par`)
   - Reads local field:
     `nek_re100k/BF_poiseuille_RANS0.f00001`

Execution:
```bash
python nek_re100k/nek_to_csv.py
```

Generated artifacts (all in `nek_re100k/`):
- `nek_to_csv.npz`
- `nek_to_csv.csv`
- `nek_to_csv.png`
- `nek_to_csv.json`

Key extracted metadata:
- `Re` from `.par`: `100000.0`
- estimated `Re_tau`: `1115.818661288065`
- `U_bulk`: `0.9984901828353354`

### Update: Export all requested quantities + print `U_bulk`

Script updated:
- `nek_re100k/nek_to_csv.py`

Now exported columns in `nek_re100k/nek_to_csv.csv`:
- `y_over_delta`
- `u`
- `v`
- `pressure`
- `scalar_1`
- `scalar_2`
- `u_over_ubulk`
- `y_plus`
- `u_plus`

Runtime print now includes:
- `U_bulk = ...`
- field keys found in Nek file

For `nek_re100k/BF_poiseuille_RANS0.f00001`, detected keys:
- `['p', 'vx', 'vy', 'vz']`

Therefore:
- `u`, `v`, `pressure` are populated from Nek fields.
- `scalar_1` and `scalar_2` are exported as `NaN` (not present in this field file), and this is recorded in:
  - `nek_re100k/nek_to_csv.json` -> `scalar_availability`.

### Update: direct scalar extraction + full-channel symmetry plot

After user check, script was updated to read Nek data directly via `pymech` (not filtered helper keys)
so `s01` and `s02` are captured explicitly from `elem.scal`.

Current run:
```bash
python nek_re100k/nek_to_csv.py
```

Runtime output now shows:
- `U_bulk = 9.9848895532955539e-01`
- `Field keys = ['p', 's01', 's02', 't', 'vx', 'vy']`

New/updated outputs in `nek_re100k/`:
- `nek_to_csv.csv` now includes: `u, v, pressure, temp, scalar_1, scalar_2`
- `nek_to_csv.png` multi-panel profile figure
- `nek_to_csv_symmetry.csv` with lower branch (`0->1`) and inverted upper branch (`2->1`)
- `nek_to_csv_symmetry.png` for symmetry check in full channel

Symmetry plotting requested by user was implemented as:
- lower profile: bottom wall to centerline (`y=0..1`)
- upper profile: top wall to centerline, inverted (`y=2..1`)

### Update: Dashed Nek overlays in `re100k/results`

User request:
- Overlay Nek reference values as dashed lines in the standard `re100k/results` plots
  to compare FEniCS vs Nek directly.

Implementation:
1. `src/dolfinx_rans/plotting.py`
   - `plot_final_fields(...)` now accepts optional `reference_profile_csv`.
   - Added CSV loader for reference profiles with `y_plus,u_plus` and optional
     `scalar_1,scalar_2,k_plus,omega_plus,nu_t_over_nu`.
   - Overlays dashed reference lines on:
     - `u+` semilog profile
     - `k+` profile (uses `k_plus` or fallback `scalar_1`)
     - `ω+` profile (uses `omega_plus` or fallback `scalar_2`)
     - near-wall linear `u+`
   - Dashed reference style uses orange dashed lines for visual separation.
2. `src/dolfinx_rans/cli.py`
   - Added reference-path resolver:
     - uses `benchmark.reference_profile_csv` when set
     - otherwise auto-detects `nek_re100k/nek_to_csv.csv` for re100k workflow
   - Prints resolved overlay path at runtime.

Validation run:
```bash
python nek_re100k/nek_to_csv.py
./run_re100k.sh
```

Observed runtime line:
- `Reference overlay CSV: /home/rfrantz/dolfinx-rans/nek_re100k/nek_to_csv.csv`

Result:
- Updated `re100k/results/final_fields.png` now contains dashed Nek overlays
  for direct comparison against the FEniCS profiles.

### Update: Pure Nek-style plotting axis + snapshots inside results

User request:
- Stop plotting FEniCS profiles in `y+`; use the same pure coordinate style as Nek plots.
- Keep snapshots under `results/` (not separate `re100k/snps` folder).

Implemented:
1. `src/dolfinx_rans/plotting.py`
   - Final profile panels switched to `y/delta` (pure coordinate).
   - Velocity panel now plots `U/U_bulk` vs `y/delta`.
   - `k` and `omega` plotted vs `y/delta`.
   - Near-wall panel uses `U/U_bulk` vs `y/delta` (no wall-unit axis).
   - Removed `u+`, `k+`, `ω+` labels from final figure annotations/titles.
2. `src/dolfinx_rans/solver.py`
   - Snapshot banner now prints explicit target:
     `Saving snapshots every ... to re100k/results/snps`
3. `run_re100k.sh`
   - Added post-run canonicalization to move any legacy `re100k/snps` into
     `re100k/results/snps`.

Validation runs completed:
- `./run_re100k.sh` (multiple reruns after plotting/path updates)
- final artifacts refreshed at `2026-02-12 15:29`:
  - `re100k/results/final_fields.png`
  - `re100k/results/snps/velocity.bp/*`
  - `re100k/results/snps/turbulence.bp/*`

Current folder layout:
- `re100k/results/snps/...`
- no top-level `re100k/snps` directory remains.
