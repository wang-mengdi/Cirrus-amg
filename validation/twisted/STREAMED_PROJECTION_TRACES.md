# Componentwise face interpolation and compact pressure traces

The native256 v7 probe stopped after the first velocity-projection linear
solve, before its face-update trace completed. It had not reached Anderson
history or a complete physical step. This change addresses transient face
arrays in that path and retains the full diagnostic and physical gates.

`ProjectionSolver::velocityFlux` evaluates one component of the existing
interpolation operator at a time. Each open face still uses precisely its
normal component, area and original arithmetic. The full nf-by-3 face temporary
is replaced by one nf-vector. At 12,263,704 faces this removes 196,219,264 bytes
from that temporary's live storage.

For twofold pressure diagnostics, `ProjectionFluxTrace` previews the existing
`q + (-FluxPair::product(area, gradient))` operation without changing q. It
accumulates the resulting cell balance in the original face order, finds the
eventual worst cell using the original reduction, then retains old/new pairs
only for its incident open faces. It also records all original change counts,
velocity scales and intended-update statistics. The actual physical update
still uses `ConservativeFlux::subtractProduct`, and physical residuals are
computed independently from the resulting stored flux. The subsequent trace
verifies the preview against that actual residual, worst cell and retained
face values before writing the unchanged CSV columns. Double-only traces keep
their original full snapshot path. No conservation criterion is dropped.

The twofold trace no longer copies all high/low face values. Its preview uses
two cell-balance vectors and a rounded cell vector, then retains scalar
statistics and a few face records. The separate full face-delta array in the
trace is replaced by the same per-face scalar product. The pressure RHS is
released after its linear solve; the low correction is released after its
gradient/impulse use, retaining the exact maximum used by the unchanged
roundoff-cycle predicate. Impulse snapshots and full diagnostic fields remain.

The original expression `normalGradient * correction + normalGradient *
correctionLow` is deliberately retained. The initial v1 audit caught different
bits when it was rewritten as two separate sparse products followed by vector
addition: Eigen changes the reduction order. That failed audit is preserved;
the v2 tests and production build use the original expression.

## Verification

The production-MSVC audit extracts the old and new functions and uses the
actual native16 and adaptive64 operators. All 24 cases pass: zero, constant,
positive/negative and varying fields, sub-ULP updates and zero corrections,
each in double and twofold storage. Velocity-flux values, pressure-gradient
values, all pressure-update CSV columns and worst-cell face CSV columns are
bit-identical. Twofold preview arithmetic is also compared to every actual
updated face, and deliberate diagnostic inconsistency is rejected. At most
24 incident faces are retained in these test cases. The isolated test's peak
RSS is dominated by operator construction, so it is not a measurement of the
full-flow memory reduction.

Two completed physical steps on CPU16 remain byte-identical to the preceding
build, including intermediate dumps. GPU16/64 five-field maximum relative L2
differences are 2.20e-16 / 1.17e-14 against the unchanged 1e-6 comparison gate.
Both trajectories also pass independent 120-digit high/low flux balance.
The completed disk-history restart preserves exact state bytes and rejects
legacy tail loss. Dense/disk native16 steady runs both converge after 17 outer
iterations, with five-field differences below 3.34e-16 and independently
reconstructed relative divergence below 1.84e-21.

Build and run evidence is kept on D under
`D:/CirrusExperiments/cirrus-amg/builds/projection_trace_stream_v1` and
`runs/projection_trace_stream_*`. Terminal flow, restart, steady and native256
outcomes are recorded with the frozen source and runtime hashes. These internal
checks do not substitute for independent Aphros128 completion or actual
native256 steady, spatial and near-wall convergence.

## Guarded native256 v8 outcome

All six retained initialization tables/checks match the preceding native256
setup byte-for-byte. The initial projection completes with divergence
5.161667100794266e-17; the first predictor correction completes, but the second
predictor solve does not. Three successful GPU pressure calls have maximum
true residual 9.308745997449971e-14. All three completed face-update traces pass
the compact-preview versus actual-flux check. No implicit diffusion solve,
Anderson update, complete raw fixed-point map or physical step is adopted.

The memory guard stops only this probe at 2026-09-09 20:07:36.071 UTC, with
3,700,432,896 bytes available against the unchanged 3.5 GiB floor. Peak process
RSS is 10,252,103,680 bytes. At the stop, RSS is 10,002,325,504 bytes: the floor
is on whole-machine available RAM, and peak RSS alone does not identify the
stopping allocation or give a controlled comparison across probes. The
original CPU Aphros128 process is resumed at 20:08:03.391 UTC; its unchanged
identity, continued CPU progress and completed-time prefix are verified.

The next host-memory audit should inspect `NativeStorage::holder` in
`OctreeMesh.cu`. It downloads an `HAHostTileHolder<Tile>` for native leaf cells
and retains its full `mHostTiles` array for later field export. At 20,120 leaves
and the unchanged 44,880-byte Tile layout, its payload is 902,985,600 bytes.
That is source-level accounting, not a measured new optimization. Any compact
or external backing must preserve initial/unused channel bytes, all native
metadata, per-cell mapping, repeated exports and the binary native dump.
`dumpBinaryBlob()` also warrants an output-time allocation audit. Other
smaller candidates include the no-longer-used interpolation boundary-weight
vector and GPU face factors after all operator setup; cell volumes remain
necessary for Neumann RHS compatibility and must not simply be dropped.
