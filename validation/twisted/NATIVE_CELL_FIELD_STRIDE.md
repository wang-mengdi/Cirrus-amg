# Cell-only stride for native GPU double fields

`simple/NativeCompactGpu.cu` now uses 512 entries per original native tile in
its separate double cell fields. The former stride736 came from the native
float tile's node-channel capacity (9^3 nodes with padding). Cell coordinates
in [0,7]^3 map only to offsets [0,511]. The original `PoissonTile` float/node
channels retain their 736-entry stride and complete byte layout.

The four stride consumers change together: the total field allocation, host
active-cell slots and both owner/neighbor addresses in the matrix-free
regular-face kernel. Full cut/coarse-fine/wall stencils and the native AMG
transfer map receive those active-cell slots. Original neighbor pointers,
periodic ghosts, float AMG hierarchy, coefficients and reductions are unchanged.
Krylov basis arrays already use compact active-cell indices and do not change.

## Verification

`checks/native_cell_stride_operator_v1/result.json` verifies the following with
the fresh build under `D:/CirrusExperiments/cirrus-amg/builds/native_cell_stride_v1`:

- Native16 compact pressure/viscosity known-solution solves and periodic
  topology coverage.
- Native16 and native64 full pressure/viscosity GPU actions and manufactured
  solves against independent all-face CPU assembly, with spatially variable
  face coefficients.
- Complete original tile-byte preservation through failure before upload,
  failure after upload, operator/AMG application and normal destruction.

On full native64, 1,536 pressure interface faces and 66,984 custom viscosity
faces are covered. Pressure CPU/GPU relative residuals are
`9.32376814937703e-14` / `9.320294993179419e-14`, with known-solution relative
L2 error `7.762427619931432e-12`. Viscosity CPU/GPU residuals are
`7.360908421577565e-14` / `7.360338329728478e-14`, with known-solution error
`1.4774482321767082e-12`. All existing gates pass.

Reported full native64 GPU operator storage decreases from 259,086,918 to
230,184,870 bytes. The exact 28,902,048-byte reduction equals
`1049 tiles * (736-512) * (15 double fields * 8 + 3 skip-mask bytes)`.
This confirms removal of padding in the intended fields; it is not a claim
that all GPU storage decreases by the same percentage.

For native256's 30,128 tiles the corresponding post-first-solve allocation
reduction would be 830,086,656 bytes, plus a smaller host staging vector.
This is a count from the unchanged arrays, not an observed native256 flow
peak or proof that a full steady trajectory fits available memory.

Executable SHA-256 values:

- `simple_channel.exe`: `178a5c94b63214bf81f261ba8a7d2f4b25b07af01be4b8512d8714bcd17d0f40`
- `native_compact_gpu_audit.exe`: `4e65b321c678987917934bab4f5cebe42ba916fe0fc6950b3159a476675d0988`

`checks/native_cell_stride_regression_v1/completion.json` verifies complete
two-step CPU16/GPU16/GPU64 trajectories. CPU16 final fields and retained
intermediate CSVs are bitwise identical. GPU16/GPU64 pass the existing five
field, actual mass, fixed-point and strict linear gates; maximum field
relative L2 differences are `3.0904681189870546e-16` and
`1.202439299151589e-14`. The small native64 whole-process peak working set is
655,339,520 bytes versus 655,011,840 before the change; the proven storage
reduction is in the device arrays, not necessarily the measured host peak.

The first physical native256 step is a separate check. Full native256 steady
flow, spatial convergence and the original-equation Aphros128 steady
comparison remain required.

## Guarded native256 retry

`runs/native_flow256_probe_v2` retains the previous physical case and requests
the same first time step, with pressure-pass tracing added. It starts with
12,582,367,232 bytes available RAM and uses the unchanged 3.5 GiB stop floor.
At 16:38:41.640 UTC the floor is crossed (3,599,929,344 bytes available).
The guard stops only this experiment; exit code15 is recorded at 16:38:42.895.

The last completed marker is `projection.gpu_fields_ready`. Four retained
CPU operator/material diagnostic files are bitwise identical to the preceding
full native256 initializer, but the AMG setup and complete initializer have
not completed. There are no completed linear solves or physical time steps.
The method metadata and final mesh CSVs are unavailable at this stop point;
their absence must not be treated as a numerical difference or a passing
native256 validation.

Peak working set is 9,310,089,216 bytes, and peak commit is 11,532,861,440 bytes.
This is a partial-construction measurement and cannot be compared with the
previous completed initializer as if both covered the same work. The first
solve's predicted 830 MB savings has not yet been exercised at native256.

Original CPU Aphros128 PID42880, creation time1788913281.3414037, resumes at
16:39:17.635 UTC after 382.703 seconds. CPU progress, unchanged inputs and the
same two completed physical steps through t=0.01 are verified. The probe and
independent watchdog have exited. Original Aphros128 steady acceptance and
the outstanding spatial gates remain incomplete.

Evidence and executed producers are retained under `results/native_cell_stride`;
large data remains on D. Read-only follow-up identifies the AMG host tile map
as a possible contributor to this earlier construction peak. Its precise
allocation lifetime needs further measurement before the next optimization.
