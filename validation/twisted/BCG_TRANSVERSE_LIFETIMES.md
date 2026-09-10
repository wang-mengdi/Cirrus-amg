# Release BCG temporaries before their next consumer

The preceding native256 experiment completed its initial pressure projection
but was stopped by the unchanged memory guard before finishing a physical time
step. BCG prediction had two avoidable overlaps of dense temporary arrays.

`simple/ProjectionSolver.cpp` now accumulates the same six side fluxes per cell,
including both conservative-flux parts. It computes the three transverse speeds
with the existing paired sum/division and releases the high/low side sums before
allocating predicted face values and component gradients. A speed is independent
of the predicted component and of the face that consumes it. The original
upwind-face area check still runs when that speed is consumed; unused sides do
not introduce new errors or boundary conditions.

`predictedFlux()` also releases the three-component predicted face velocity
after converting it to conservative flux, before pressure projection starts.
Flux is allocated after BCG returns, so this change does not introduce a new
flux allocation during the BCG temporary peak. The output values and projection
algorithm are unchanged.

With both `SIMPLE_BCG_MEMORY=1` and `SIMPLE_CONSTRUCTION_MEMORY=1`, observations
record entry, side-flux workspace, completed transverse speeds, component
workspace, and released predicted face velocity. These observations neither
trim working sets nor change solver settings.

For the actual native256 counts (3,995,168 fluid cells and 12,263,704 faces), the
BCG dense-array overlap falls from 967,742,720 to 680,090,624 logical bytes.
The preceding side-flux phase needs 479,420,160 logical bytes. Releasing predicted
face velocity removes another 294,328,896-byte lifetime overlap with projection.
These are array-size calculations, not claims about total or peak process RAM.

## Verification

The retained MSVC test extracts the original and modified production BCG bodies
and uses the actual native16/64 geometry and face-gradient operators. All 24
cases are bitwise identical: six zero/constant/forward/reverse/mixed-velocity
and flux patterns, each with double or twofold flux. Twofold cases include
nonzero low-part inputs. Walls remain zero and all output components are finite.
The isolated adaptive64 evaluator's measured peak working set decreases from
161,763,328 to 151,932,928 bytes. This is an evaluator measurement, not whole-flow
or exclusive machine timing.

Build `D:/CirrusExperiments/cirrus-amg/builds/bcg_transverse_v1` changes only
`ProjectionSolver.cpp`. Its GPU operator-audit executable is byte-identical to
the preceding active-tile build. Complete flow regressions, independent exported
high/low conservation and a bounded native256 retry are checked separately.
None of these narrow checks replaces steady, Aphros or near-wall/grid acceptance.

Complete two-step CPU16/GPU16/GPU64 regressions pass. CPU16 final and intermediate
CSV fields remain byte-identical. The maximum five-field relative L2 difference
from the preceding build is `2.281051533139885e-16` on GPU16 and
`1.1026181452506004e-14` on GPU64, below the unchanged `1e-6` bound. Independent
120-digit accumulation of exported high/low flux passes both steps on both GPU
meshes. The largest GPU64 accepted linear residual is `9.961505751664848e-14`,
below `1e-13`. Its observed complete-run peak working set decreased from
675,758,080 to 665,575,424 bytes in these two runs; this is not an exclusive
performance comparison or an extrapolation to native256.

## Bounded native256 result

`native_flow256_probe_v6` completed initialization with all six geometry/material
setup files byte-identical to the previous native256 initializer. It passed
10 original linear solves (seven pressure and three implicit diffusion), with
maximum accepted relative residual `9.358427792470246e-14`. Initial pressure,
predicted-flux and velocity projections each finished in two passes, reporting
divergence `3.701974695067568e-17`, `9.872533587674103e-20` and
`5.29211753359825e-19`, respectively. These are solver-trace observations,
not an independent check of an exported, completed native256 flow field.

The first and second BCG evaluations completed; their component-workspace
observations retained at least `4,186,742,784` bytes of available memory.
After predicted face velocity was released, the corresponding observation was
`4,745,035,776` available bytes. The memory guard subsequently stopped the probe
at `2026-09-09T18:48:16.979695Z`, with `3,596,509,184` bytes available, below its
unchanged 3.5 GiB floor. Native peak working set was `11,092,185,088` bytes.
The source-order/trace review places the next pressure solve in the raw
fixed-point check; the exact allocation at the stop is not instrumented.
There is no completed physical time step or complete inner fixed-point check.

The original Aphros128 process resumed at `18:49:20.552385Z` with the same
identity, private state and inputs. Independent restoration verification observed
its CPU time advance and retained its original two-step time prefix to 0.01.

## Remaining work

The retained read-only review identifies 383,536,128 bytes in RHS, provisional,
residual and previous-velocity fields whose lifetimes can end earlier. The raw
diagnostic source/intermediate/diffused/predicted fields account for another
483,871,360 bytes that need not overlap subsequent Anderson trials after their
dump. These are future changes, not part of this commit.

Full coupled Anderson state has 321,839,040 bytes at native256. At depth five,
six retained `f`/`g` pairs require 3,862,068,480 bytes, plus two 321,839,040-byte
Gram-difference vectors and caller temporaries. This history had not yet been
created at the observed stop; it is a separate upcoming scaling limit. A history
store backed by D with bounded resident memory is a candidate for retaining the
full coupled state and depth. It requires numerical and complete-trajectory
verification before use. No state components, tolerances or wall probes were
removed in this work.

The summary and diagnostic are retained under
`D:/CirrusExperiments/cirrus-amg/checks/bcg_transverse_summary_v1` and
`D:/CirrusExperiments/cirrus-amg/checks/native_flow256_probe_v6/projection_progress.json`.
The compressed repository evidence is `results/bcg_transverse_lifetimes`, with
raw/compressed/source hashes. Full native256 flow, steady/Aphros agreement and
the previously failed near-wall/spatial refinement gates remain outstanding.
