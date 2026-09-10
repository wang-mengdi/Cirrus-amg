# Stream BCG velocity components

`simple/ProjectionSolver.cpp::bcg` now predicts one velocity component at a
time. The shared side flux is still accumulated in the original face order.
Each component retains only its face gradient and six side gradients per
cell. At a coarse/fine interface, only the upwind Taylor row is evaluated;
the two full dense face-by-three Taylor fields are no longer constructed.
The three-component predicted face field remains available to the original
projection and conservative transport operations.

The scalar slope, source interpolation, transverse terms, zero-flux upwind
choice and boundary exclusions retain the original expressions and order.
The Taylor value uses an Eigen sparse row product. A first trial used a
single manual accumulator and was rejected by the bitwise comparison:
Eigen 5's original sparse product uses two accumulators. The retained row
product preserves that library reduction, instead of hardcoding its current
internal implementation. The rejected trial was never used for flow runs.

No GPU kernel, AMG parameter, pressure/viscosity operator, boundary condition,
Anderson setting, time step or physical acceptance gate changes.

## Verification

The independent operator harness extracts the preceding and current production
BCG function bodies, loads the retained native16 and wall-refined native64
meshes and operator coefficients, and evaluates both functions on identical
inputs. The twelve cases cover zero and nonconstant velocity, constant
velocity, mixed flux directions, all-forward and all-reverse open-face flux,
and zero flux with nonzero fields. Every face/component is compared bitwise;
stationary wall predictions must be exactly zero. All twelve pass, including
the 1,536 coarse/fine faces in the native64 fixture.

On native64, isolated old and new evaluator processes have measured cumulative
peak working sets of 192,684,032 and 152,621,056 bytes respectively. This
measurement includes fixture loading and operator setup; it is not a whole
CFD run or a native256 measurement.

For actual native256 counts (3,995,168 cells and 12,263,704 faces), the explicit
BCG temporary/return arrays change from `(12*nf+24*nc)*8 = 1,944,387,840` bytes
to `(4*nf+12*nc)*8 = 775,974,656` bytes, a logical reduction of 1,168,413,184
bytes. Sparse-product temporaries, allocator behavior, GPU workspaces and
Anderson history are outside this count. It is not an assertion that a
complete native256 steady run fits memory.

Fresh build `D:/CirrusExperiments/cirrus-amg/builds/streamed_bcg_v2` produces
`simple_channel.exe` SHA-256
`9867687c8ddb67d29d1cc52d1a4d69b951656d291089ab084d484f67e2b5942e`.
The native audit executable is byte-identical to the preceding validated
build. Operator-check evidence is in `checks/streamed_bcg_operator_v2` on D.

`checks/streamed_bcg_regression_v2/completion.json` also passes all complete
two-step trajectory regressions. CPU16 final fields and all retained
intermediate cell/face CSVs are bitwise identical. GPU16 and GPU64 pass the
existing fixed-point, actual mass, strict original-RHS linear and five-field
comparison gates. Maximum relative L2 differences are
`3.0952322306425673e-16` and `1.0943067020836994e-14`, respectively. Native64
whole-run peak working set changes from 692,699,136 to 655,011,840 bytes;
peak commit changes from 1,277,845,504 to 1,242,550,272 bytes. Small CPU16
whole-process peaks do not decrease; the change targets large BCG arrays.

The bounded native256 physical-step experiment is separate from this
arithmetic and trajectory proof. Neither local arithmetic
equivalence nor successful initialization resolves the outstanding spatial
or independent Aphros128 steady-alignment requirements.

## First native256 physical-step attempt

`runs/native_flow256_probe_v1` uses the fresh build, the original geometry,
depth-5 inner Anderson, dt=0.005 and all original physical/linear tolerances.
It requests one physical step, with a 25-minute experiment limit and the
existing 3.5 GiB available-memory floor. A separate coordinator and watchdog
bound the original CPU Aphros128 suspension to 30 minutes.

The new executable again completes initialization. All six geometry/operator/
material files are byte-identical to `native_operator256_probe_v17`, and all
method metadata is unchanged. At 16:15:35.197 UTC, available memory reaches
3,651,911,680 bytes and the guard stops only this experiment. Its exit code
is 15; all executed inputs and geometry remain unchanged. Peak working set
is 10,099,781,632 bytes and peak commit is 19,790,532,608 bytes. The first
physical-step folder was created, but no GPU linear solve or physical time
step completed. This is a resource failure, not an accepted flow result.
The successful BCG storage change is therefore not yet exercised in this
native256 attempt, which stops before the first pressure solve returns.

Whole-device GPU samples peak at 8,119 MiB. They include unrelated graphics
usage and cannot identify the specific allocation responsible for the stop.
The trace narrows the next investigation to first-solve workspaces. Read-only
inspection identifies a possible cell-only stride of 512 rather than the
736 values reserved for native node channels in the separate double GPU
fields. This has not been implemented or validated; the native float tile
layout and AMG topology must remain intact.

The original Aphros128 PID 42880 (creation time 1788913281.3414037) resumes
at 16:16:17.630 UTC, after 654.695 seconds. Post-resume CPU progress, all
immutable inputs and the same two completed physical steps through t=0.01
are verified. Probe and watchdog processes have exited. The independent
reference has not yet reached steady acceptance.

`results/streamed_bcg/receipt.json` records compressed reports, executable
source snapshots, the rejected arithmetic trial and the executed producer
scripts. Large geometry/flow files remain under D. The complete native256
flow, finer spatial checks and original-equation Aphros128 steady comparison
remain required work.
