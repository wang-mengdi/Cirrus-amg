# Lazy host storage for the native AMG hierarchy

The previous native256 physical probe stopped after the double GPU fields were
ready, before AMG setup completed. Its trace did not identify an individual
allocation. This change reduces host coefficient storage at that boundary and
adds observations inside AMG setup; it does not change the flow discretization.

## Implementation

`simple/NativeAmgPreconditioner.cu` keeps the existing topology map and all 512
cell masks per tile. Four double coefficient channels and the separate reaction
array are now allocated on first nonzero use. An absent array reads as positive
zero; setters also preserve negative zero. Additions still evaluate the same
old value plus the same contribution, in the original face/coarsening order.

Reaction terms remain separate from the diagonal channel, including at a pinned
pressure cell. Each leaf/ghost reaction array is released after its diagonal is
computed. After all levels are assembled, tiles are uploaded in the same map
order. `HADeviceGrid::setTileHost` copies synchronously, so the corresponding
host map node can then be erased. It never retained a second complete host tile
array; the saved storage is the coefficient map itself.

The original GPU tile layout, native periodic ghost completion, transfer maps,
float AMG cycles, double matrix-free operators, FGMRES and its original-RHS
acceptance checks are unchanged. With `SIMPLE_CONSTRUCTION_MEMORY=1`, the log
records topology, face coefficients, hierarchy and host-upload release, with
allocation counts and process memory. Payload bytes exclude map-node and heap
allocator overhead and are not a measurement of whole-process RSS.

## Verification

The retained build is `D:/CirrusExperiments/cirrus-amg/builds/lazy_amg_host_v1`.
Its source manifest freezes the compiled sources and executable hashes.

The CPU coefficient harness extracts both the previous and current production
host assemblers. It uses actual native16/64 mesh dumps and the audited complete
AMG topology. Eight cases per mesh cover different pressure pins, unpinned
pressure, mass/wall terms, variable face factors, zero coefficients and signed
zero. It compares every tile type, cell mask, double coefficient bit and float
upload bit, including masked cells and all coarsened/ghost levels.

Compact16/64 and full16/64 GPU audits pass independent all-face CPU operator
actions and manufactured solves. The full audits also verify original tile byte
restoration. On full64, pressure known-solution relative L2 is
`7.762528272728306e-12`; diffusion is `1.477443336889945e-12`.

Complete two-step CPU16 and GPU16/64 regression passes the existing convergence,
linear and mass gates. CPU final fields and intermediate CSVs are byte-identical.
The largest relative L2 difference across velocity, pressure, cut-cell velocity,
wall shear and shared flux is `2.1546663979995036e-16` on GPU16 and
`1.1848830593644135e-14` on GPU64.

For the 1,073-tile native64 AMG hierarchy, 544 coefficient arrays are needed.
Final host payload falls from 22,533,000 to 9,488,024 bytes. All reaction arrays
are released before upload. Whole-flow64 peak working set remains approximately
656 MB (655,339,520 before, 655,761,408 after); another stage determines that peak.
The isolated host coefficient evaluator's peak working set decreases from
110,313,472 to 103,018,496 bytes. This is a separate process, not a GPU flow peak.

## Native256 and remaining acceptance

The bounded physical256 probe uses the unchanged physical case and tolerances,
requires 11 GiB available RAM before launch, and retains a 3.5 GiB memory floor
and 25-minute native runtime limit. The existing original CPU Aphros128 process
may be suspended and trimmed for at most 30 minutes, with an independent
watchdog and explicit same-process resumption verification.

`native_flow256_probe_v3` completed both AMG hierarchies and initialization.
Each hierarchy has 31,314 tiles, 15,498 allocated coefficient arrays and a final
host payload of 270,703,536 bytes, compared with 657,594,000 bytes previously.
The payload is released after upload. Fourteen actual pressure solves pass
the original-RHS `1e-13` criterion. The first takes 960 iterations and returns
`9.0911045530949656e-14`; this is still only one linear solve, not a time step.

The experiment was deliberately stopped at 17:24:19 UTC on 2026-09-09 after
repeated out-of-bound projection stagnation. No memory or runtime limit was
relaxed. The separate stop producer identifies and terminates only this native
probe after checking that no other native GPU experiment is active. Therefore
the unchanged run wrapper records exit 15 and its pause coordinator reports an
error; the explicit diagnostic-stop record explains these terminal statuses.
The original CPU Aphros128 process resumed at 17:25:03 UTC after 1,126.80 seconds,
and its identity, inputs, completed-time prefix and renewed CPU progress pass
the independent restoration check.

The captured worst cell is 3,682,654, with volume `6.2169322674883179e-24`.
Its three open-face fluxes have a nonzero exact binary64 balance of
`2.5243548967072378e-29`, producing divergence `4.0604510200448005e-6`.
All three intended corrections are below half a binary64 ULP of their stored
flux. A 120-digit Decimal audit of the exact binary64 values confirms that all
three updates round back to the old flux, while retaining these intended
increments in exact arithmetic would reduce this cell's divergence to about
`1.0143e-21`. That conditional local calculation diagnoses the precision loss;
it does not validate a full higher-precision solver or global mass conservation.

The next numerical task is to preserve low-order face-flux corrections through
projection, consumers, checkpoints, dumps and independent conservation checks,
then repeat the original physical gates and Aphros comparison. Do not waive the
`1e-8` projection bound or accept the small linear residual as physical accuracy.

The full native256 physical, steady, reference and spatial acceptance remains
pending. Native128 steady/long-time agreement and the accepted Aphros64 results
do not remove the previously failed 64-to-128 near-wall/grid convergence gates.
Large builds, experiments and visualization outputs remain on D.
