# Preserve sub-ULP conservative face-flux corrections

The native256 run `native_flow256_probe_v3` passed fourteen pressure linear
solves but its first projection stalled at divergence `4.0604510200448005e-6`.
The worst cut cell had volume `6.2169322674883179e-24`. Each intended correction
to its three open faces was less than half an ULP of the stored binary64 flux,
so repeating accurate pressure solves could not change those face values.
The unchanged projection acceptance bound is `1e-8`.

## Numerical state and consumers

`simple/ConservativeFlux.h` represents a conservative flux as a normalized pair
of binary64 high and low parts. Error-free sums and correctly rounded fused
multiply-add products retain contributions below the high part's ULP. Only
the native GPU mean-zero pressure path uses this representation; the existing
CPU double path remains available for comparison.

`ProjectionSolver.cpp` uses both parts for incremental pressure updates and
oriented shared-face balances. It rounds the completed balance once when
constructing the double linear RHS. The same face pair contributes with
opposite signs to its two incident cells. No cell is dropped and no local
residual is zeroed, redistributed or waived to pass a mass check.

BCG side fluxes and flux/area velocities consume both parts. The advective
face-velocity product, cell balance and existing Aphros redistribution operator
are evaluated with the pair before the resulting acceleration is rounded to
the existing double velocity field. The GPU operator, AMG preconditioner,
FGMRES, original-RHS checks and linear tolerances are unchanged.

Anderson candidates remain approximate double guesses. Rejected candidates
restore the full raw velocity, pressure, diffusion predictor and high/low flux
state. An accepted candidate is still subject to the original mass and equation
checks; only a fresh ordinary map output can be a completed physical step.

## Output and restart contract

Native twofold `flux.csv` has columns `id,flux,flux_low`. The physical flux is
the unevaluated sum of the last two binary64 values. Both use round-trip decimal
serialization. Intermediate `faces.csv` also retains `flux_low` and
`predicted_flux_low`; pressure-floor diagnostics retain both before/after low
parts. `projection_method.json` and step metrics declare
`conservative_flux_storage: twofold`. Reading only `flux` is suitable for a
rounded visual quantity, but is insufficient for tiny-cell conservation checks.

The separate `scripts/check_twisted_twofold_native_run.py` checks all completed
steps, original linear/fixed-point gates and independent 120-digit Decimal
accumulation of the exported face pairs. Its physical gates remain local
relative divergence `<1e-7`, sum of absolute cell imbalance over throughflow
`<1e-8`, and section spread `<1e-8`. No tiny-cell exclusion is used. The old
verification scripts and the running original Aphros reference remain unchanged.

`twisted_twofold_restart.py` creates `cirrus_projection_twofold_restart_v4`
checkpoints, and `run_twisted_twofold_solver.py` runs their continuations.
The lossless binary layout is little-endian binary64: seven fields per cell
`[u,v,w,p,u_diff,v_diff,w_diff]`, then all face high parts, then all low parts.
The C++ loader echoes these complete bytes. It rejects a legacy checkpoint
whose parent declares twofold storage, instead of silently dropping its low
parts. Old checkpoints whose parent actually used double flux remain supported.

## Verification completed before the native256 retry

The retained build is `D:/CirrusExperiments/cirrus-amg/builds/twofold_flux_v1`.
Its main executable SHA-256 is
`3434a08d5716c0f815919fd6e6235e77d34ecf48a94314f115350a8765af5a95`.
The native GPU operator-audit executable is byte-identical to the previous
validated build; no GPU linear implementation changed.

The MSVC arithmetic test compares 2,048 additions, 2,048 products and 2,047
divisions with 120-digit Decimal arithmetic. It also replays the actual three
failed native256 faces, verifies weighted divergence and redistribution, and
checks copies and the legacy double path. The reconstructed worst-cell
divergence is approximately `1.0143e-21`, below the original `1e-8` bound.
This is a local reproduction, not a complete physical256 acceptance.

The first MinGW build of this test failed the product error bound. Its default
math-library FMA behavior did not satisfy the required accuracy for some tested
operands. That failed test is retained; it is not presented as validated support
for this host arithmetic. The actual production MSVC toolchain passes the
strict arithmetic test and the independent exported-flow checks. Error-free
transforms require ordinary IEEE rounding without algebraic fast-math
reassociation and a correctly rounded FMA; rerun this test for another toolchain.

Complete CPU16/GPU16/GPU64 two-step regressions pass. CPU16 final and intermediate
CSV fields are byte-identical to the previous build. The largest original
field-comparison relative L2 difference is `2.34229514056648e-16` on GPU16 and
`1.0942528084053852e-14` on GPU64, against the unchanged `1e-6` regression gate.
The new independent pair-aware conservation checker passes both GPU meshes.
On GPU64 step 1, true relative divergence is `1.2413771466570664e-22`; a naive
high-only check gives `3.305861087650805e-11` and is recorded only as a diagnostic.

The native16 checkpoint preserves nonzero low parts byte-for-byte, resumes the
remaining physical step, and passes independent mass and field comparison.
The maximum continuation field difference is `1.8132842308567888e-16`.
A separately produced otherwise valid legacy checkpoint is explicitly rejected
by C++ before any time step because it omits the parent's low parts.

## Bounded native256 retry

`native_flow256_probe_v4` stopped at `2026-09-09T17:59:41.361118Z` when the
available-memory sample fell to `3,229,044,736` bytes, below the unchanged
3.5 GiB floor. Native peak working set was `9,285,300,224` bytes. This happened
during initialization, before conservative-flux state allocation and before
any linear solve or physical time step. It therefore neither validates nor
disproves the numerical correction. The two emitted embedded/interface checks
are byte-identical to the prior native256 initializer; the other setup files
were not reached.

The lease restored the original Aphros128 process (PID 42880, creation time
`1788913281.3414037`) at `18:00:11.824600Z`, about 260 seconds after suspension.
The separate restoration check verified its unchanged identity, inputs and
completed time prefix, and observed CPU time advance. No original reference
state was discarded or restarted. Increasing disk capacity does not remove
this host-RAM limit. The retry failed before the new high/low flux allocation,
so its memory stop is not evidence of the additional flux array's cost.

The summary is retained in
`D:/CirrusExperiments/cirrus-amg/checks/twofold_flux_summary_v1/result.json`.
The compressed repository evidence is under `results/twofold_conservative_flux`,
with immutable source snapshots, raw/compressed hashes and retained-input hashes.

## Remaining acceptance

The full native256 time step, steady convergence, independent Aphros agreement,
and the previously failed near-wall/spatial refinement gates remain unproven.
Further physical256 runs must retain the original geometry, PDE, tolerances,
3.5 GiB available-memory floor and 25-minute native runtime bound. A bounded
30-minute same-process pause of the original Aphros128 reference requires the
independent restoration watchdog and restoration verification. Large artifacts
remain on D. The next memory work should reduce native work-array storage for
inactive tiles while retaining topology, active-cell addressing, face actions
and exact native tile restoration; this optimization is not implemented here.
