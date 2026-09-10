# SIMPLE implementation / validation work log

Branch: `feature/simple-octree-pipe`, based on `no-vtk` at `c8a0742`.
Task started 2026-09-06. Numerical validation is complete; this log preserves the intermediate observations and experiments. See SIMPLE_RESULTS.md for the final accepted results.

## Scope and acceptance

Keep native `HADeviceGrid<Tile>` octree topology and implement steady incompressible
Newtonian SIMPLE without flow maps. First implementation stores double fields and
sparse linear systems on CPU, while native CUDA grid creation/refinement and an
explicit field write-back remain active. There is no conversion to a uniform mesh.

Validate exact laminar channel solutions and independent, unmodified numerical
algorithms in Aphros. Baseline modifications are limited to diagnostic dumps and
an initial-condition module. Compare operators/states on identical grids; compare
physical errors and convergence on different grids.

Target for the resolved reference cases: relative velocity and conservative flow
errors below 0.5%, wall shear error below 1%, normalized momentum and mass residuals
below 1e-8, and decreasing errors with refinement. Uniform-grid intermediate
SIMPLE dumps should agree with Aphros to linear-solver accuracy. Coarse cases may
exceed the final accuracy target, but must be reported rather than hidden.

## Implemented

- `simple/SimpleMesh.*`, `OctreeMesh.cu`: native tile construction/refinement,
  stable leaf ordering, unique conservative subfaces at 2:1 interfaces, periodic
  coordinate images, geometry/volume closure checks, native float field snapshot.
- `SimpleSolver.*`: integral FV momentum, implicit diffusion/upwind convection,
  velocity under-relaxation, Rhie-Chow fluxes, symmetric pressure gauge fixing,
  pressure correction, physical pressure, nonorthogonal deferred correction,
  strict residual checks, cell/face/matrix dumps and conservative section metrics.
- No-slip wall diffusion uses the same quadratic one-sided derivative as Aphros.
  Cell pressure gradients at walls extrapolate interior values; wall flux remains
  impermeable. These two operations must not be conflated.
- `OperatorChecks.*`: independent mesh, affine reconstruction, random flux, and
  pressure flux/matrix identities. Scope explicitly excludes full solver testing.
- `scripts/compare_simple_aphros.py`: coordinate/coverage validation, periodic face
  normalization, pressure gauge normalization, delta-form RHS comparison.
- `validation/aphros/`: reproducible baseline scripts and diagnostics patch.

## Actual observations so far

- Native ny=8 uniform grid: 4096 cells, 12800 unique faces.
- Native ny=8 adaptive grid: 18432 cells, 512 coarse/fine subfaces.
- Geometry closure and independent linear-field/pressure identities pass near
  double roundoff. A weighted linear gradient at coarse/fine faces is not exact
  for quadratic fields; a general quadratic reconstruction is being evaluated.
- Uniform periodic planar channel, rho=1, nu=.01, gx=1, Lx=1, H=Lz=.125:
  SIMPLE with alphaU=.7/alphaP=.3 reaches normalized momentum <1e-8 after 318
  iterations. Relative velocity L2 error ~1.086e-8, normalized mass ~5e-14.
- The same test from a non-solenoidal velocity perturbation reaches the same
  solution. Its first two iterations match Aphros at atol=1e-11/rtol=1e-8 for
  matrix diagonal, delta RHS, predicted velocity, p', corrected velocity, and
  predicted/corrected face flux, with complete coordinate coverage.
- Exact Poiseuille cellcenter values still give a midpoint section quadrature
  error: ny=8 flow error .78125%. This is reported separately from pointwise error;
  Aphros shows the same .78125/.1953125/.048828% trend for ny=8/16/32.
- Eigen 5.0 BiCGSTAB source uses an absolute loop threshold despite a relative
  tolerance API. Unit-norm RHS normalization and original-matrix residual checks
  fixed false early stopping for small integral FV RHS values. The dependency was
  not edited.
- alphaU=1 trials are unstable (pressure-coupling perturbations grow), including
  uniform ny16; these are failures, not accepted cases. Keep under-relaxation.

## Second round observations

- Linear-interface adaptive ny8 converged in 787 steps: velocity L2 0.294856%,
  conservative Q 0.428449%, mean wall shear 0.044877%. The general quadratic
  interface reconstruction gives 0.290320%, 0.451400%, and 0.030664%, respectively.
  It improves some quantities but not all; refinement remains necessary.
- Uniform pressure-driven ny8 converged with velocity L2 1.11e-8 and physical
  pressure L2 1.71e-11. This is an analytic check, not an Aphros pressure-port check.
- Four-wall square duct ny8 has velocity L2 0.722869% and Q error 1.372963%,
  agreeing with the independent Aphros discretization. It is deliberately coarse.
- Optional Stokes-only Anderson acceleration is safeguarded by actual unrelaxed
  momentum and continuity checks and acts only on the next iteration input.
  Uniform ny8 took 7 rather than 318 steps; adaptive ny8 took 46 rather than 787.
  Adaptive Q-relative-error difference between accelerated/plain runs is 1.22e-8.
  Final acceptance still requires every raw SIMPLE-step convergence condition.
- Standalone Anderson tests cover a linear contraction, coupled SPD system,
  nonzero affine constraints, and failure guards. Both solves use true residuals.
- A fixed-threshold validation harness now records process return codes, binary,
  case and artifact hashes, complete baseline coverage, and independent operator
  checks. The first four cases in its full run passed; finer cases are running.

## Third round observations

- Added an exact-zero pressure RHS guard for later nonorthogonal passes; it
  preserves explicit-flux consistency checking instead of prematurely stopping.
- Added an adaptive channel with convection enabled to the fixed suite. Its
  pre-LDLT-build run converged in 806 steps: velocity L2 0.292648%, conservative
  Q error 0.435428%, mean wall shear error 0.031678%, continuity 1.90e-14.
  The final binary is being checked again rather than adopting an old binary's run.
- The 147456-cell Stokes CG run was stopped for performance investigation, not
  accepted as converged. On its actual pressure matrix, an independent Eigen test
  measured CG+IC 1.524 s per zero-guess RHS (334 iterations), versus LDLT analysis
  and factorization 97.767 s and cached solve 0.194 s. The LDLT process peaked near
  799 MiB. These are matrix timings, not an end-to-end speedup claim.
- Added pressure_solver=auto/cg/ldlt. Auto uses cached LDLT for Stokes with at
  least 100000 cells; smaller or convective cases use CG. Both solve the same
  compact matrix and retain the same true-residual checks and SIMPLE equations.
- Final build SHA256: 8fd60dbc25613fcdbb54f82be0a4026728990b8654dd8b330e84d544c7597970.
  All 11 cases are rerunning on this build. The harness now also checks decreasing
  refinement errors and the final raw SIMPLE velocity-change history value.

## Final validation

The final binary passed all 11 fixed cases, all required Aphros comparisons,
independent operator checks, native topology checks, and all three refinement
families. Final full audit returned full_pass=true and strict_current_binary_pass=true.
Adaptive ny16: velocity L2 0.074109%, conservative Q error 0.106835%, 115 steps.
Square duct ny16: velocity L2 0.241974%, Q error 0.247437%, 45 steps.
All normalized momentum residuals <1e-8; maximum continuity residual 5.11e-12.
Figures were generated from the actual CSV and visually checked. Source, build,
case and result provenance are recorded with the final report.

Actual CFX geometry/settings/results remain outside this standard-case validation.
The implementation is a CPU double reference on native CUDA octree topology;
curved cut cells, inlet development, turbulence and GPU matrix solving remain
future integration work, not completed features.

External baseline worktree/results:
`D:/Dropbox/Agent-simulation/simple-baseline`, Aphros revision
`b60ce3da52c19935fa24c778f62f02141eaf7f80`.
