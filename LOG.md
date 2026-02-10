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
