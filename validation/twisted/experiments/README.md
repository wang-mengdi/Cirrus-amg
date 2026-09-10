# Rejected quadratic wall reconstruction experiment

`quadratic5.patch` applies to commit `b742956`. It is an archived experiment,
not a supported solver option. Apply it only to a separate checkout of that
commit, build `simple_channel`, and set `wall_reconstruction=quadratic5` in a
uniform embedded-tube case. Keep the standard geometry, rho=1, nu=0.01,
force=[1,0,0], Stokes equations, alpha_u=0.98, alpha_p=0.03, Anderson depth 5,
pressure_solver=amg, tolerance=1e-8 and linear_tolerance=1e-11.

The patch fits a three-dimensional quadratic to included cube-center values
in a 5x5x5 stencil. Coordinates are relative to the cut-wall point and scaled
by h; omitting the constant coefficient enforces zero velocity there. Pivoted
QR produces sparse normal-derivative weights. Rank and conditioning checks,
affine/quadratic polynomial checks, and global flux balance are included.
The compact implicit diffusion and Aphros deferred-source redistribution are
unchanged. Only Cirrus uses this experiment; Aphros retains its original wall
closure. The ordinary Aphros comparison intentionally rejects these results.

For the scalar F=1-((y-yc(x))^2+(z-zc(x))^2)/R^2, the n128 local gradient
relative L2 error falls from 1.886% to 0.1034%. This does **not** establish
flow accuracy. Completed Stokes solves give:

| ny | Q (m3/s) | Momentum residual |
|---|---:|---:|
| 16 | 7.7818428369e-5 | 8.30e-9 |
| 32 | 6.3869346288e-5 | 8.30e-9 |
| 64 | 5.5887882341e-5 | 8.89e-9 |

The 32-to-64 flow difference is 14.28%; velocity at the fixed nearest wall
probes differs by 40.61%, and wall traction differs by 8.42%. All physical
refinement gates fail. The candidate is rejected and removed from the normal
solver. Its outputs remain under `output/twisted/quad5_uniform{16,32,64}_v2`,
with the original executable `output/twisted/simple_channel_quad5_v1.exe`.
The first runs without `_v2` failed because that build omitted AMGCL; they
are preserved separately and are not numerical results.

A preceding 3x3x3 quadratic diagnostic was also rejected: the maximum normal
matrix condition number at n128 was 5.67e9 and its manufactured derivative
error was 88.25%. Widening the fit fixed that conditioning problem, but did
not fix the actual flow convergence. The interaction of the wall fit,
compact operator and redistributed deferred term would require a separate
consistency analysis before any renewed solver trial.

The evidence snapshot in `../results/wall_consistency_checkpoint` records
both improvements in the manufactured diagnostic and failed physical-flow
checks. No thresholds are relaxed and no failed experiment is promoted as
an Aphros-aligned solution.
