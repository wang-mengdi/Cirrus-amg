# Store double work fields only for tiles containing fluid

The native256 physical retry `native_flow256_probe_v4` stopped for host memory
during initialization. Its last complete phase was `projection.quadratic_released`;
the next stage creates the native GPU operator. The independent double work
arrays still assigned 512 cell entries to every original tile, including
nonleaves, ghosts and completely inactive leaves.

`simple/NativeCompactGpu.cu` now gives a dense field serial to each original
leaf containing at least one fluid cell. Active tiles retain their original
relative order; physical cell order and local cell offsets are unchanged.
Original tile allocations, numerical channel layout, masks, neighbor pointers
and the metadata restoration lease remain the same. The serial only addresses
the separate double work arrays. Both AMG transfers and explicit face stencils
use the resulting physical-cell slot map.

Every inactive tile points to one extra zero coefficient block. An invalid
serial would be unsafe: the compact kernel can inspect a positive neighbor's
face coefficient before deciding whether that neighbor contributes. No fluid
cell owns a slot in this extra block, so its face coefficients and viscosity
skip flags stay zero. No inactive value is read through a nonzero coefficient.
The compact stencil and full face kernels themselves are unchanged.

`allocatedBytes()` already uses the actual field length and therefore measures
the reduced allocation. With construction diagnostics enabled, the solver
prints original, active and stored tile counts plus the old/new field lengths.
There is deliberately no change to the packed active-cell Krylov basis, the
AMG hierarchy, the finite-volume equations or any convergence threshold.

## Independent verification

The retained build is `D:/CirrusExperiments/cirrus-amg/builds/active_tile_fields_v1`.
Only `simple/NativeCompactGpu.cu` differs from the conservative-flux build.

Six operator cases pass on uniform16 and adaptive64: compact AMG, full AMG and
compact Jacobi PCG. They compare actions and manufactured solutions against
independent CPU assembly/face loops. The full and Jacobi cases also force a
failed one-iteration solve, then verify successful reuse. The metadata audit
compares every byte of every original native tile after initialization errors
and normal destruction; it checks numerical channels during operator/AMG use.

On adaptive64 the original 1,049 tiles reduce to 437 active tiles plus one zero
block. Each separate field has 224,256 entries instead of 537,088. In the full
operator audit, reported device allocation is 191,706,534 bytes versus the prior
230,184,870 bytes, a saving of 38,478,336 bytes. Uniform16 has all 16 tiles active;
its extra zero block adds 62,976 bytes in the same full audit. These are measured
operator allocations, not whole-machine or complete-flow memory estimates.

Complete two-step CPU16/GPU16/GPU64 regressions pass. CPU16 final and intermediate
CSV fields remain byte-identical. The largest five-field relative L2 difference
from the preceding twofold-flux build is `2.427651443210928e-16` on GPU16 and
`1.1133493291208476e-14` on GPU64, against the unchanged `1e-6` bound. Independent
120-digit accumulation of the exported face high/low parts passes every retained
step. The largest accepted GPU64 linear residual is `9.96149958187285e-14`, below
the unchanged `1e-13` threshold.

The guarded physical256 retry records 30,128 original tiles, 10,320 active tiles
and 10,321 stored blocks. Each separate work field therefore has 5,284,352
entries instead of 15,425,536, saving 81,129,472 bytes per double field. This
count is observed during the actual initializer; it does not by itself prove
physical-step completion or peak host/GPU memory savings.

Passing small operator/trajectory tests does not establish physical steady
flow, independent Aphros agreement or spatial/near-wall convergence. The bounded
native256 outcome and original-reference restoration are recorded separately.

## Native256 outcome and remaining memory work

`native_flow256_probe_v5` completed initialization. All six geometry/material
setup files are byte-identical to the previous native256 initializer. Its first
pressure projection finished in two passes: the second trace row reports
divergence `1.5702828171971214e-16`, below the original `1e-8` bound. Both pressure
linear solves passed the original-RHS test; their largest relative residual is
`9.86252485891788e-14`. This is a solver-trace observation, not an independent
conservation check of a completed exported native256 flow field.

At `2026-09-09T18:24:14.679056Z` the memory guard stopped only this native probe:
available memory was `3,682,537,472` bytes, below the unchanged 3.5 GiB floor.
Peak native working set was `10,539,495,424` bytes. No physical time step or
complete inner iteration finished. The original Aphros128 process resumed at
`18:24:47.146669Z` with its identity and inputs unchanged; a separate check
observed CPU time advance. Its completed time prefix remains two steps to 0.01.

The next candidate is the lifetime of BCG transverse-flux temporaries. Their
high/low side sums could be consumed into three transverse speeds per cell and
released before allocating face results and gradients. That should preserve
the existing arithmetic while reducing overlapping arrays. The exact allocation
at the stop was not instrumented, so this is a read-only proposal, not a proven
attribution or implemented optimization. The retained diagnostic is
`checks/native_flow256_probe_v5/projection_progress.json` on D.

The compressed evidence and its raw/compressed/retained-input hashes are under
`results/active_tile_double_fields`. Full native256 flow, steady/Aphros agreement
and the existing failed near-wall/spatial refinement gates remain outstanding.
