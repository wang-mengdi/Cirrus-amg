# Direct pressure-face assembly

`simple/ProjectionAssembly.h` no longer retains global lists of deferred and
owner/neighbor Taylor rows while building their output matrices. It also avoids
the temporary row-major copy used to convert the correction into column-major
storage. This targets the native256 initialization peak observed before
`projection.face_matrices_ready` in the preceding attempt.

The first pass computes one interface's three rows, counts exact support and
discards the rows. The second recomputes that face and writes directly to the
final correction columns and Taylor rows. Faces remain ordered, so inner indices
arrive in the same order as the former conversion. `Row::add` retains the first
contribution, duplicate addition order and explicit cancellation zeros. The
temporary row data is bounded by a single face; column counts scale with cells.
The extra pass trades some setup arithmetic for lower temporary storage.

The only changed compiled source is `simple/ProjectionAssembly.h`. Pressure
diagnostics, diffusion, GPU kernels, AMG, boundary treatment, material weights
and physical acceptance criteria are unchanged. The CPU backend uses the same
discrete matrices as before.

## Verified comparisons

Artifacts use `D:/CirrusExperiments/cirrus-amg`. The new executable is
`builds/direct_projection_assembly_v1/simple_channel.exe`, SHA-256
`96bcefdcd3245615b2ba0651d56952bde528370b87256c447e2635e6b6982baf`.
The native audit executable remains byte-identical to the preceding build.

- The existing independent legacy-assembly fixture compares all stored indices
  and coefficient bits on actual native16/native64 meshes and two degenerate
  cases: 41 matrix comparisons pass, including the combined full gradient.
  Diagonal/gauge/probe actions, combined compact/deferred images and invalid
  dimension guards also pass.
- Two CPU16 steps retain bitwise-identical final fields and intermediate dumps.
- Two GPU16 and GPU64 steps pass all existing physical, fixed-point, mass and
  strict linear checks. Maximum relative L2 differences across the five field
  comparisons are `3.0935782388989197e-16` and `1.1183055780674719e-14`.
- Native128 completes full initialization with byte-identical mesh, material
  and operator-diagnostic files. Its 50 construction stages retain the same
  order. Method metadata is unchanged in CPU16 and GPU16/64/128.

At native128's `projection.face_matrices_ready` marker, cumulative peak RSS
decreases from 1,972,736,000 to 1,869,074,432 bytes (103,661,568 bytes lower).
By `projection.pressure_diagnostics_ready`, the respective peaks are
1,972,736,000 and 1,963,290,624 bytes: a later temporary also contributes.
Whole-initialization peaks remain similar, 2,180,304,896 versus 2,183,266,304
bytes. No overall speed or memory improvement is inferred from those totals.

## Native256 attempt and remaining storage work

`runs/native_operator256_probe_v16` was stopped by the unchanged 3.5 GiB memory
floor at 2026-09-09 15:06:31.601 UTC. Only native PID 62120 (creation time
1788966109.4159205) was terminated; no other native GPU experiment was running.
Available RAM was 3,741,118,464 bytes and sampled RSS 8,867,196,928 bytes.
Cumulative OS peak RSS was 8,873,414,656 bytes, peak commit 10,741,501,952
bytes, and native exit code 15. The last completed marker remained
`projection.embedded_checked`: full face assembly, diagnostics and GPU setup
did not complete. An earlier stop at a different working-set limit does not
establish a completed native256 memory comparison.

The bounded pause lasted 401.083 seconds. Original CPU Aphros128 PID 42880,
creation time 1788913281.3414037, resumed at 15:07:23.112 UTC and subsequent CPU
progress was verified. Inputs and the two-step physical-time prefix through
`t=0.01` are unchanged. The probe and independent watchdog both exited.

Inspection identifies a further construction-lifetime opportunity: zero
velocity, diffusion-guess, pressure and flux arrays occupy 321,839,040 bytes on
this mesh and are not used by the initialization diagnostics or GPU setup.
The interpolation constant probe adds 98,109,632 bytes beyond its last use in
the area/material loop. Deferring/releasing these allocations is not yet
implemented in this checkpoint. In contrast, `sideArea` is used by the material
acceleration diagnostic, so it cannot simply be deferred past that check.

Producer scripts, logs, manifests and compact evidence are archived under
`validation/twisted/results/direct_projection_assembly/receipt.json`. Large
geometry and flow fields remain on D.

These checks validate the storage transformation. They do not establish a
native256 physical solution or resolve the existing native64-to-native128
spatial failures. Original-equation Aphros128 steady alignment remains required.
