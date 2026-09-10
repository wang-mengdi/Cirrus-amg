# Steady iteration of the original projection equations

The optional `steady_anderson_depth` accelerates convergence between fully
converged BCG/implicit-viscosity projection steps. Its default is zero. It keeps
the existing embedded octree, shared cut-face flux, wall closure and native GPU
FGMRES/AMG pressure and viscosity operators. It does not introduce a flow map or
replace the pressure solve with a global direct factorization.

This is **pseudo iteration toward the same discrete steady equations**. It is
not a physical transient trajectory. The momentum `time_step` is still a
discretization parameter: it enters Rhie--Chow interpolation even at a steady
fixed point, so it must match the ordinary/Aphros control. Timestep and spatial
convergence remain separate tests.

## Algorithm

The existing `anderson_depth` applies inside one physical projection step. The
new option applies outside that converged map. The first projection map has a
special pressure initialization; outer history starts with the second map.
History contains the existing dimensionless packed state: three velocity
components, pressure, the three implicit-diffusion guess components, and every
shared face flux. Including flux is necessary to preserve the affine mass
constraint when combining map outputs.

The existing Type-II Anderson implementation fits residual differences using
at most `depth` history directions. Its small Gram eigensolve has dimension
`depth` (5 in these tests); it is not a pressure matrix factorization. History
rank/regularization and coefficient safeguards are unchanged. Background:
[Walker and Ni, 2011](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf)
and [Pollock and Rebholz](https://arxiv.org/abs/1909.04638). These references
motivate the method; they do not establish convergence for this particular
cut-cell discretization.

A proposal is tested with factors 1, 1/2, 1/4 and 1/8. It must strictly decrease
the maximum of the original stationary momentum residual and the complete
stationary projection/diffusion fixed-point residual. It must also satisfy the
original continuity gate. The existing optional roundoff mass repair is
limited to a relative state change below `1e-10`. Failed optional linear trials
are recorded and rejected. If no proposal passes, the exact raw velocity,
pressure, diffusion guess and flux are restored, without a scale/unscale
rounding cycle.

Each subsequent iteration executes the original projection map to inner
convergence. The final result must be a **fresh raw map output**, never an
extrapolated proposal. Acceptance requires:

- velocity change below the original inner tolerance (`1e-11` here);
- complete inner fixed point and implicit viscosity below the original
  tolerance (`1e-8` here);
- original steady momentum, map velocity defect divided by `dt`, and complete
  stationary fixed point below `1e-8`;
- original continuity and cross-section flux gates, plus independent balance
  of actual shared face fluxes;
- every accepted native GPU pressure/diffusion solve retains true relative
  residual and compatibility correction at or below `1e-13`.

The initial version accepts only full native GPU pressure/viscosity with native
AMG, mean-zero pressure, a positive pseudo step, and at least two iterations.
Physical restart and operator/geometry probes are excluded. The history depth
must be a JSON integer in 0..10, checked before conversion to a C++ `int`.

## Output and reproduction

Set `steady_anderson_depth: 5` in a usual full-native `fluid_solver: "proj"`
case. `time_steps` then bounds the number of pseudo iterations, not physical
steps. Outputs are `iterate_0001/`, `steady_history.csv`,
`steady_acceleration.csv`, and `steady_summary.json`. Per-iterate case/metrics
explicitly say `fluid_solver: "proj_steady"` and have `steady_iteration` and
`pseudo_time`; no `physical_step`/`physical_time` is emitted. An exhausted
iteration budget without steady convergence returns a failure. Rejected
optional GPU trials, if any, have separate failure dumps and a decision log.

Final full fields are written even if convergence occurs between regular output
strides. Every converged pseudo iteration retains its final raw inner dump.
The final `solution.csv` and `flux.csv` must equal that raw dump exactly.

Run from `C:/Code/Cirrus-amg`, placing all new outputs on D:

```powershell
C:/Users/bear/miniconda3/python.exe scripts/run_twisted_solver.py `
  --config D:/CirrusExperiments/cirrus-amg/configs/steady_outer16_v1.json `
  --exe D:/CirrusExperiments/cirrus-amg/builds/steady_anderson_v1/simple_channel.exe `
  --measure-memory

C:/Users/bear/miniconda3/python.exe scripts/check_twisted_steady_iteration.py `
  --run D:/CirrusExperiments/cirrus-amg/runs/steady_outer16_v1 `
  --ordinary-final C:/Code/Cirrus-amg/output/twisted/ours_proj16_steady_v1/step_0128 `
  --output D:/CirrusExperiments/cirrus-amg/checks/your_new_steady16_check.json
```

The runner refuses to overwrite an existing run. The checker validates the
actual executable and compiled source snapshots, execution inputs, complete
pseudo history, convergence metrics, raw final state, GPU traces and
independent mass. The ordinary reference must be an actually completed steady
physical trajectory with matching geometry/physics/step size. Old transient
and Aphros checkers are not weakened or fed relabeled pseudo output.

The checker now requires an explicit choice: `--ordinary-final <step>` for
the comparison above, or `--state-only` for complete equation, topology and
mass validation without a field-equivalence claim. Both check actual native
cell/face ordering, positive measures, 2:1 face balance, pressure and viscosity
stencil counts, and wall traction/area consistency. State-only output explicitly
reports that same-grid equivalence, Aphros alignment and spatial convergence
have not been checked. It cannot establish accuracy by itself.

`scripts/analyze_twisted_refinement.py` can now consume a final pseudo steady
iterate using `--coarse-steady-iteration` or `--fine-steady-iteration`. It
revalidates that entire run with the state-only checker; the physical input
still requires its complete successful physical trajectory. Geometry, physical
parameters, wall closure, momentum step and all probe/sampling limits remain
the same. The output records the two iteration modes separately. For example,
after the 128 run has actually completed and passed its state checks:

```text
python scripts/analyze_twisted_refinement.py --coarse <64 final physical step> --fine <128 final iterate> --fine-steady-iteration --output <new D directory>
```

Actual 64-grid ordinary versus pseudo steady outputs exercised this path.
Their maximum flow/velocity/pressure/shear probe difference was
`8.813229372970224e-10`. This used the explicit `--same-grid-regression` option:
it is not a spatial convergence result. The existing wall sampling-sensitivity
gate still failed. Repeating the actual physical 32-to-64 comparison preserved
all previous numerical metrics and both probe CSV files exactly, including
the failed spatial acceptance. Scope and damaged-input regression checks
passed; results are retained under `D:/CirrusExperiments/cirrus-amg/checks/`
as `steady_state_modes_v1` and `steady_refinement_modes_v1`.

## Completed 16-grid experiment

The full original twisted circular tube has 2,744 fluid cells and 1,368 cut
cells at this coarse resolution. This particular control has no fluid
coarse/fine interfaces; the 64-grid experiment is needed to exercise wall
refinement.

`steady_outer_default_off16_v1` compared all eight physical steps with the old
`cycle16_steps8_v1` build. Both performed 66 inner iterations; all native linear,
full fixed-point and actual mass checks passed. The maximum relative L2 field
difference across velocity, pressure, cut-cell velocity, wall shear and shared
flux was `4.966958821343272e-16`.

`steady_outer16_v1` started from zero with `dt=0.005`, inner/outer depths 5,
maximum 64 pseudo iterations and output stride 4. It completed successfully
in **17 pseudo iterations**, with 14 accepted outer proposals and 137 total
inner iterations. The ordinary CPU/deferred control first satisfied its
momentum/temporal steady diagnostics at physical step 81 and continued to its
configured step 128. These are algorithm counts, **not a measured speedup**:
the control used a different linear backend/inner iteration strategy.

The actual 128-step ordinary final field is used for the numerical comparison:

| Quantity | Relative L2 difference |
| --- | ---: |
| Velocity | `3.881886000140811e-9` |
| Pressure, aligned volume mean | `3.2278084832275645e-9` |
| Cut-cell velocity | `4.136060636230495e-9` |
| Wall shear vector | `3.9687286556140535e-9` |
| Shared face flux | `3.840646948300535e-9` |

All meet the unchanged same-grid `1e-6` limit. Final steady momentum is
`8.39699118298727e-9`; the map velocity defect is
`8.386812037628483e-9`. All 1,027 accepted GPU calls meet `1e-13`; the maximum
is `9.944212656525213e-14`. The full checker passes, including independent
mass for all six retained field iterations and raw-dump equality.

The first rejection suite caught three integer narrowing cases (`2^32`,
`2^32+5`, and `-2^32`) in the experimental option parser. Those failures are
retained. The parser was fixed before promotion; the valid depth-5 numerical
experiment is unaffected. Build/source manifests distinguish that parser-only
revision from the field experiment.

The corrected parser build `steady_anderson_v3` was also run from zero
(`steady_outer16_v3`): it again converged in 17 iterations and passed the same
full numerical check. The maximum accepted GPU residual was
`9.944913569070662e-14`. Its 36-case rejection suite passed, including actual
CLI option rejection and corrupt synthetic output fixtures; an uncorrupted
rebased fixture was checked first to rule out unrelated path failures.

The actual `steady_outer_budget16_v3` run deliberately allowed only two
pseudo iterations. Both inner maps converged, but steady momentum was
`0.5523857109228338`; the solver returned **3**, wrote `steady_converged: false`,
and the completed-steady validator rejected it. Reaching the configured
iteration count therefore cannot silently certify a steady solution.

A final source review made both trial equation residuals explicitly finite
before taking their maximum (`std::max` alone can hide a NaN in its second
operand). The subsequent `steady_anderson_v4` build stopped its own compiler
tree when available RAM dropped below the existing 2.5 GiB guard. This was a
resource stop, not a numerical failure. A source-locked controller on D waited
for 4.3 GiB available for ten seconds, then built `steady_anderson_v5` and
ran/checked `steady_outer16_v5`. That controller subsequently completed with
unchanged sources at `2026-09-08T20:37:37.085293+00:00`: the final safeguard
revision **compiled and passed the complete 16-grid steady comparison**.
It again used 17 pseudo iterations, 137 inner iterations and 14 accepted
outer proposals. All 1,027 accepted GPU calls passed; their maximum true
relative residual was `9.944387884927714e-14`.

The verified executable is
`D:/CirrusExperiments/cirrus-amg/builds/steady_anderson_v5/simple_channel.exe`,
SHA-256 `2b7ad29a63f49b21dbc2d01dc52fea7729eb35aaa8e554a6c812d7052db3e5b5`.
The earlier checkpoint correctly recorded this validation as pending; the
completion evidence is an additional record, not a rewrite of that checkpoint.

An earlier build-wrapper validation failure is also retained as
`steady_anderson_v2/validation_failure.json`. The wrapper now identifies its
own build process by exact PID, creation time, executable and launched
arguments, while recording its working directories. It does not assume the
build tool's current directory is immutable. The v3 build completed all three
commands with unchanged sources and no resource stop.

## Completed adaptive 64-grid comparison

The adaptive 64-grid run `D:/CirrusExperiments/cirrus-amg/runs/steady_outer64_v1`
uses the same full twisted pipe, 136,632 fluid cells, 21,816 wall cut cells and
1,536 fluid coarse/fine faces. It started from zero using the preserved v1
executable with valid depth 5; it did not replace or restart the ordinary
128-grid trajectory. It returned exit code 0 at
`2026-09-08T20:34:39.612633+00:00`, with unchanged inputs.

The run reached steady convergence in **16 pseudo iterations**, using 206
inner iterations and 12 accepted outer proposals. Independent validation
against `ours_proj64_steady_original_rhs_v1/step_0128` passed:

| Quantity | Relative L2 difference |
| --- | ---: |
| Velocity | `8.736880180735178e-10` |
| Pressure, aligned volume mean | `7.708884087739371e-10` |
| Cut-cell velocity | `1.8260257648247471e-9` |
| Wall shear vector | `4.5879763060187264e-10` |
| Shared face flux | `8.785349717080301e-10` |

All are below the unchanged same-grid `1e-6` limit. Final steady momentum is
`7.290130712434424e-9`, map velocity defect is `1.5867072469047172e-9`, and
complete stationary fixed-point residual is `1.6394360786614824e-9`.
All 2,287 accepted GPU calls meet `1e-13`, with maximum
`9.961463138724023e-14`. Across the five retained full-field iterations, maximum
independently reconstructed relative divergence is `1.6755859895260623e-11`,
global absolute cell flux imbalance divided by throughflow is
`1.026385692893552e-14`, and section-flux spread is `4.721667729877198e-15`.
Every retained final velocity/pressure/flux table equals its fresh raw inner
map dump. The full proof is `D:/CirrusExperiments/cirrus-amg/checks/steady_outer64_v1.json`.

The v1 64-grid experiment and the v5 finite-guard 16-grid experiment are
identified separately by their actual compiled snapshots. A 64-grid rerun of
v5 is not claimed here.

The latest 64-grid fields can be opened in ParaView from
`D:/CirrusExperiments/cirrus-amg/runs/steady_outer64_v1_viz/solution.vtu`
and `walls.vtp` in the same directory. Color the volume by `Speed`, `Velocity`
or `Pressure`, and the wall by `WallShearMagnitude` or `WallShear`.
ParaView 5.13 actually read back all 136,632 volume cells and 21,816 wall
polygons; velocity, pressure and shear arrays equal the solver CSV bit for bit.
Closed-boundary and independent convex-hull geometry checks also passed.
This is one steady result, not a physical-time animation.

## Pending physical acceptance

The full zero-start 128-grid pseudo steady run is now on D in
`runs/steady_outer128_v1`, using the verified v5 executable, `dt=0.005`,
inner/outer history depth 5 and the original full native GPU operators.
It preserves the actual 790,216-cell adaptive geometry and 49,152 coarse/fine
faces. The original physical 128 control has a separately recorded temporary
pause; the original seeded Aphros64 reference has completed its full trajectory.
Launch and intermediate progress do not
establish steady convergence or a completed 64-to-128 comparison.

A redundant mixed linear-backend Aphros64 trial was intentionally stopped
after immutable capture and original-equation/face-mass validation of its
first two completed physical steps. Its real exit code is 15; it is not a
completed or steady reference. The first stop guard refused to act when the
run advanced beyond the captured first step; the second step was captured
before the actual stop. This freed resources while retaining the original
Aphros references. Capture and exact process-identity receipts remain on D
under `checks/hybrid64_retirement_preparation_v1` and `v2`.

The completed ordinary native64 field has now passed comparison with an
actually steady original Aphros64 full 128-step trajectory; see
[the independent 64-grid completion report](APHROS64_FULL_COMPLETION.md). That direct
comparison and the outer-to-ordinary same-grid equivalence above have separate
provenance. The separate zero-start Aphros64 control passed an additional
matched-time comparison at completed step 24 and was then intentionally
stopped, retaining its snapshot and actual exit code 15; see
[the cold-control completion record](APHROS_COLD_CONTROL_RETIREMENT.md).

The pseudo steady 64-grid output has also passed a direct comparison with
the complete original Aphros64 reference, through an entry point that retains
its actual iteration type. See [the direct steady/reference comparison](STEADY_APHROS_COMPARISON.md).
That checker independently validates the entire native steady run before
reading its final fields; it does not manufacture a physical time history.

Independent 128-grid steady alignment and near-wall spatial convergence remain
unproved. The recorded 32-to-64
spatial failures, including wall-shear error and interpolation sensitivity,
remain failures. There is no CFX result or physical-time trajectory claim.

The [native128 steady run](NATIVE128_STEADY_COMPLETION.md) has now completed
23 iterations and passed the full state checker, including actual native
topology, saved raw-map fields, mass and all 5,279 GPU linear solves.
The subsequent [64-to-128 spatial check](SPATIAL_CONVERGENCE_64_128.md) failed
the original flow, near-wall velocity, traction and sampling-sensitivity
gates. The original Aphros128 reference has actually started; it has not
completed or established independent fine-grid alignment yet.
