# Complete Anderson history on D and shorter projection temporaries

The native256 coupled state contains `7 * 3,995,168 + 12,263,704 = 40,229,880`
binary64 values. Depth 5 retains six complete input/output residual pairs, or
3,862,068,480 bytes. This is a future history-storage cost; the preceding v6
physical probe stopped during its raw fixed-point check, before history update.

`SIMPLE_ANDERSON_FILE_HISTORY=1` opts into checked binary64 scratch files under
`<output>/anderson_history/`. The default remains in-memory history. Inner maps
and steady outer maps own separate fresh directories. There is no reduction of
history depth or coupled state components. The pressure/velocity/viscosity/flux
normalization, least-squares safeguards, backtracking and physical acceptance
criteria are unchanged. These are approximate Anderson state vectors; the
existing full high/low face-flux fallback and mass repair are retained.

Each f/g file stores complete double vectors with per-block FNV-1a checksums
held by its owner. Gram products and candidate updates stream blocks of at
most 65,536 values per history entry. Block dot products use compensated sums;
their floating-point reduction order can differ from the dense implementation.
Checksums detect accidental corruption and incomplete reads; they are not
cryptographic authentication. Files are disposable history, not restart
checkpoints. A process killed by a resource guard can leave its scratch files.
Normal eviction, reset and destruction remove only owned files. An existing
directory is rejected, and unrelated files are never recursively removed.

`anderson_storage.csv` records retained/resident history values, peak numerical
scratch values and cumulative read/write bytes. Scratch counts exclude caller
input/output and the returned candidate, and include the temporary full
residual. They are allocation accounting, not process-RSS measurements.

Projection temporaries now die after their final use: diffusion RHS, provisional
velocity, previous velocity and diffusion residual before the raw fixed-point
check; source/intermediate/diffused/predicted fields after the unchanged raw-map
dump and before Anderson; packed input after history update. Within the residual
check, source pressure and the predicted/diffusion workspace also have shorter
lifetimes. GPU pressure matrices remain unassembled on the host. These lifetime
changes do not replace the native GPU linear solver or change its tolerance.

## Validation and remaining acceptance

Production build and large outputs are on D under
`D:/CirrusExperiments/cirrus-amg/builds/anderson_file_v1` and `runs/anderson_file_*`.
Frozen source snapshots and executable hashes identify the tested implementation.

- Production MSVC algebra tests: the old and new default dense JSON reports are
  identical. Twelve disk/dense comparisons use block sizes 1, 7, 64 and 65,536,
  state size 1,003 and scales 1e-120, 1 and 1e120. Candidate differences are below
  6.11e-15. Actual coupled linear-equation residual, nonzero affine constraints,
  coefficient/rank/NaN safeguards, reset/eviction, truncation and f/g bit damage
  are exercised.
- Actual native256 state size: all six full pairs occupy 3,862,068,480 bytes on
  D, with zero resident dense-history values. Peak process RSS was 973,619,200
  bytes, including the test's full input/output/candidate vectors. Seven updates
  agree with a repeated eight-component dense oracle within 5.44e-13. This is
  a storage/algebra test, not a physical256 flow result.
- Two completed physical steps on CPU16 are byte-identical to the preceding
  build, including intermediate dumps. GPU16/64 dense regressions have maximum
  five-field relative L2 differences of 2.33e-16 / 1.11e-14. Disk versus dense
  differences are 2.17e-16 / 1.10e-14, against the unchanged 1e-6 gate. Each GPU
  candidate also passes independent 120-digit high/low face-flux mass balance.
  GPU64 peak RSS is 624,058,368 bytes with dense history and 492,294,144 bytes
  with disk history; the preceding build used 665,575,424 bytes. These runs do
  not provide an exclusive timing benchmark.
- Disk-history restart16 restores all state bytes including both flux parts,
  passes the completed two-step comparison, and rejects a legacy checkpoint
  that would omit low parts. Original restart verifiers remain unchanged.
- Native16 steady tests exercise both inner and outer disk histories. Both
  backends converge after 17 outer iterations under the original equation and
  raw-map gates. The five final fields differ by at most 3.79e-16 relative L2;
  independent twofold relative divergence is below 1.85e-21 in both runs.

## Guarded native256 v7 outcome

The physical probe initialized all operators and produced byte-identical mesh,
embedded/interface checks and material tables against the retained native256
setup. Eight GPU solves (five pressure, three diffusion) completed, all below
1e-13; the maximum was 9.54575627365119e-14. Initial and predictor projections
finished, with final absolute divergence 6.573754967529628e-17 and
1.415513948187064e-19 respectively. The first velocity-projection solve finished,
but no corresponding face-update trace completed.

The guard stopped this probe at 2026-09-09 19:36:59.608 UTC when available RAM
fell to 3,677,605,888 bytes, below the unchanged 3.5 GiB floor. Peak process RSS
was 10,671,632,384 bytes. It did not reach the raw complete fixed-point check,
Anderson update or a completed physical step. No physical256 VTU is accepted.
Peak RSS from an earlier-stopped run is not a controlled measurement of the
total-flow memory saving. The original CPU Aphros128 process was resumed at
19:38:01.525 UTC, and its continued CPU progress, unchanged identity and retained
completed-time prefix were independently verified.

The next memory audit should inspect `velocityFlux`'s full three-component face
temporary and projection/trace workspaces. At this mesh size, a three-component
face field holds 294,328,896 bytes; a full high/low trace backup plus pressure
backup holds 228,180,608 bytes. These are source-level allocation counts, not
proof of which individual allocation triggered the observed stop. Any change
must retain full flux diagnostics and the existing physical acceptance gates.

Full outcomes are recorded in `results/anderson_file_history`. Independent Aphros128 completion and
native256 steady/spatial/near-wall convergence remain necessary for the main
goal; small regressions and successful linear solves do not establish them.
