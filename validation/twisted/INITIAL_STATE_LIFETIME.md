# Delay initial flow-state allocation

`simple/ProjectionSolver.cpp` now initializes the evolving zero velocity,
diffusion guess, pressure and face flux after operator construction and GPU
setup. These arrays are allocated before restart loading or any flow operation,
including in operator-only mode. The constructor still produces the same full
initial state; no arrays needed by the actual solver are omitted.

The interpolation constant probe is released after the material-weight loop,
and the compact pressure diagonal after selecting its gauge. `sideArea` stays
available because material acceleration diagnostics use it. No arithmetic,
boundary treatment, discretization, kernel, AMG setting or physical gate changes.

For the actual native256 mesh, the delayed flow arrays account for 321,839,040
bytes, the interpolation probe for 98,109,632 bytes, and the compact diagonal
for 31,961,344 bytes. These are allocation sizes with different lifetimes, not
an asserted reduction of the final process peak. Two construction markers,
`projection.material_factors_ready` and `projection.flow_fields_ready`, expose
the new allocation boundaries.

## Verification

All fresh build/run artifacts are under `D:/CirrusExperiments/cirrus-amg`.
The executable `builds/late_flow_fields_v1/simple_channel.exe` has SHA-256
`8344cc9861b32cc6aa93d45d8bfaaad44e845eb8e51cee8152a67abd5caa876a`.
Only `simple/ProjectionSolver.cpp` differs from the preceding validated build;
the native audit executable is byte-identical. The existing independently
verified coefficient assembler and pressure diagnostics are unchanged.

- Two CPU16 steps retain bitwise-identical final and intermediate fields.
- Two GPU16 and GPU64 steps pass all existing physical, fixed-point, mass and
  strict linear checks. Maximum relative L2 differences across the five field
  comparisons are `3.1241221398526487e-16` and `1.104766416801916e-14`.
- A native16 restart restores all 224,224 state bytes exactly, including the
  diffusion guess and every face flux. The checked complete prefix plus
  continuation agrees with the uninterrupted trajectory at both physical
  steps; maximum field relative L2 difference is `2.22543583933961e-16`.
  This specifically verifies that delayed allocation precedes checkpoint use.
- Native128 completes full initialization, retaining all flow arrays. All six
  geometry/material/operator diagnostic files are bitwise-identical to the
  preceding build. The original 50 stages retain their order around the two
  new allocation markers. Method metadata is unchanged.

The measured native128 whole-initialization peak working set changes from
2,183,266,304 to 2,121,527,296 bytes, 61,739,008 bytes lower. Peak commit changes
from 3,833,626,624 to 3,780,222,976 bytes. These measurements cover the complete
initializer; they do not measure physical-flow or steady-iteration memory.

## Native256 complete initialization

`checks/native_operator256_probe_v17/result.json` records the first successful
complete native256 initialization: 3,995,168 fluid cells, 12,263,704 faces,
280,576 coarse/fine pressure interface faces, 1,327,600 full viscosity faces,
and six native AMG levels. The process exits zero after `projection.ready`,
including allocation of every evolving flow-state array. All six setup outputs
are byte-identical to the retained geometry and operator diagnostic references.
The pressure and viscosity operators remain full, with matrix-free native GPU
FGMRES/AMG, Krylov dimension 20, CGS2 and twofold pressure iterate storage.

Peak working set is 9,592,401,920 bytes and peak commit is 16,778,743,808 bytes.
At `projection.ready`, working set is 8,795,275,264 bytes and available system
memory is 4,373,082,112 bytes. This is an initializer measurement: first-solve
Krylov workspace, convection temporaries and Anderson histories are not yet
covered. It does not establish that a complete native256 flow run fits memory.

The original Aphros128 reference process was suspended for a bounded memory
lease (900-second limit and independent resume watchdog), then resumed with
the same PID and creation time after 741.18 seconds. The post-resume check
confirms CPU progress, unchanged immutable inputs and the same two completed
physical steps through time 0.01. No baseline restart or numerical change was
made. This reference has not yet met its steady-state acceptance criteria.

Compact reports, frozen build sources and experiment producers are retained in
`results/late_flow_fields/`; large meshes, checkpoints and flow dumps remain on D.

These are construction and trajectory regressions. Native256 steady flow,
original-equation Aphros128 steady agreement and spatial convergence remain
separate required work.
