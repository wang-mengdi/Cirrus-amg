# Streamed host pressure diagnostics for native GPU projection

The native GPU backend no longer constructs the two global cell pressure
matrices during initialization. It already solves pressure through the native
face operator and AMG; those temporary host matrices were only needed to select
the compact pressure gauge and to evaluate material diagnostics before being
discarded. The CPU backend retains its explicit systems and original equations.

`simple/ProjectionDiagnostics.h` visits one column of `-D diag(area) G` at a
time, with the same face traversal, first contribution, subsequent addition
order and explicit cancellation zeros as the former Eigen CSC product.
It retains the diagonal and, when necessary, the action on the material probe.
Compact and deferred products remain separate until their probe images are
added and divided by density. The combined face gradient, material factors,
coarse/fine reconstruction, GPU kernels, AMG and all acceptance gates are
unchanged. Host divergence, gradient and viscosity operators still exist;
this change does not remove every host sparse operator.

## Validation

All new builds, large fields and detailed results are under
`D:/CirrusExperiments/cirrus-amg`. The preserved executable is
`builds/pressure_diagnostics_v1/simple_channel.exe`, SHA-256
`cfb1c23bd4d70f97eb8398f2411e80f4b164f5b1485186c377482b2c7521a7b6`.
The native audit executable is byte-identical to the preceding validated build.

- `validation/projection_assembly_test.cpp` compares the production helper
  against explicit Eigen products on retained native16/native64 meshes and two
  degenerate cases. All 41 stored-matrix comparisons match bitwise; eight
  diagnostic groups check the diagonal, gauge, empty/zero/constant/nonconstant
  probes, and all four combined compact/deferred images also match bitwise.
  Four invalid-dimension cases are rejected.
- Two CPU16 steps retain byte-identical final and intermediate fields.
- Two GPU16 and GPU64 steps pass the existing physical, fixed-point, mass and
  strict linear gates. Maximum relative L2 differences across the five checked
  fields are respectively `2.948657824585249e-16` and
  `1.0964590282354209e-14`, against `amg_periodic_v2`.
- Native128 completes full pressure and viscosity initialization. All six mesh
  and material/operator diagnostic files match the preceding build bitwise.
  Construction stages remain in the same order, with the GPU pressure phase
  now named `projection.pressure_diagnostics_ready`.
- CPU method metadata is unchanged. For GPU16/64/128 the only method-metadata
  difference is `host_pressure_matrices: released_after_setup -> not_assembled`.

Native128's total peak working set changes from 2,185,170,944 to 2,180,304,896
bytes, while peak commit changes from 3,829,469,184 to 3,840,536,576 bytes.
These nearly unchanged totals do not establish an overall memory or speed win:
later GPU configuration also contributes to the process peak.

## Native256 resource-limited attempt

`runs/native_operator256_probe_v15` is a **stopped initialization**, not a
completed operator or flow run. At 2026-09-09 14:40:53.931 UTC the existing
3.5 GiB available-memory guard stopped only native PID 68312 (creation time
1788964584.1782887). Available RAM was 3,754,930,176 bytes; observed process RSS
was 9,072,017,408 bytes. Cumulative OS peak RSS was 9,074,208,768 bytes and peak
commit was 10,910,851,072 bytes; the native process returned exit code 15.

The last completed construction marker was `projection.embedded_checked`.
The process had not completed `projection.face_matrices_ready`, so it had not
yet executed the new pressure diagnostic path. This attempt therefore provides
no native256 accuracy or completed-memory comparison for that path. The next
storage target is the temporary global deferred/Taylor row lists inside
`projectionAssembly::assemble`; they coexist with their output matrices.

The original CPU Aphros128 process was paused for 388.954 seconds under a
900-second bound with an independent watchdog. PID 42880 and creation time
1788913281.3414037 were preserved. The same process resumed at 14:41:42.685 UTC;
subsequent CPU progress, unchanged inputs and the complete physical-time prefix
were verified. That prefix now contains two completed steps, through `t=0.01`;
its latest temporal acceleration norm is `0.011021215606622723`, not a steady
acceptance. Both the probe and watchdog exited. The lease coordinator's
`passed: true` describes successful bounded control/restoration; its explicit
`native256_operators_completed: false` records the numerical work left undone.

Compact evidence and producer scripts are retained in
`validation/twisted/results/pressure_diagnostics/receipt.json`, with raw and
compressed hashes. Large geometry/field tables remain on D.

This is an implementation/storage validation. It does not establish native256
flow convergence or resolve the existing 64-to-128 spatial failures. The
independent original-equation Aphros128 steady comparison remains unfinished.
