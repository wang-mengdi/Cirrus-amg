# Checked native host Tile backing and streaming export

`SIMPLE_NATIVE_HOST_TILES_FILE=1` moves the leaf host holder's complete Tile
payload to `<output>/native_host_backing/tiles.bin` after embedded-cell mapping
is final and before projection operators are constructed. Use an output on D.
The default retains the original in-memory holder. This option changes storage;
pressure and implicit viscosity still use the configured native GPU operators.

For every original leaf, `offloadNativeHostTiles` verifies its position in the
holder, records a checksum over all `sizeof(Tile)` bytes, writes those bytes,
closes the file and reads every Tile back for verification. Only then does it
release the host Tile vector. It retains level/coordinate metadata and a stable
grouping of active fluid-cell IDs by leaf. Later remapping is rejected.

Export reads one complete Tile at a time and checks its checksum. It assigns
the same velocity and pressure values with the existing Tile indexing operator.
It copies either the first four physical channels or the full Tile to the same
original device address, matching the existing metadata-preservation flag.
Every mapped fluid cell is overwritten on every export; unmapped bytes retain
their original values. Export without a binary file still synchronizes fields.
The original grid, cut cells, coarse/fine connections and operator metadata
ownership remain in use. No global pressure or viscosity matrix is introduced.

`HADeviceGrid::dumpBinaryStream` uses the same header, compressed-level traversal
and complete raw Tile records as `dumpBinaryBlob`, with a Tile-sized staging
buffer. The vector API remains available. Production native exports now use
the stream API, checking open, write, flush and close errors.

The backing file is temporary storage for this live process, not a portable
checkpoint. Raw Tiles include metadata and addresses; FNV-1a checks detect
accidental corruption and are not a security guarantee. Existing directories
and storage manifests are rejected. Normal destruction removes only the owned
Tile file and the directory if empty, preserving foreign files. Forced process
termination can leave the file on D; terminal probe evidence indexes it.

At the 256 geometry, 20,120 leaves contain 902,985,600 bytes (about 0.841 GiB).
The retained grouping needs about 16 MB for 3,995,168 fluid IDs, plus small
offset/checksum tables. Streaming additionally avoids allocating the entire
native output blob (over 1.35 GB at 30,128 total tiles). These are allocation
sizes, not a claim about measured whole-process peak memory.

Validation is recorded under `results/native_host_storage` after completion.
The storage audit compares bytes on the same live GPU grid against both the
original host-holder export and an independently frozen `f285ab48` serializer.
It exercises metadata-preserving and full exports, repeated changing fields,
no-file synchronization, a failed output stream, backing corruption, duplicate
offload, mapping changes and ownership cleanup. Separate native operator audits
compare full pressure and viscosity with explicit CPU oracle operators.

The initial audit incorrectly required repeated GPU operator applications to
be bitwise identical. On adaptive64, coarse/fine floating atomic sums differ
even before any offload: the retained control measured scaled Linf differences
of 2.756e-16 before offload and 1.837e-16 after export. The corrected check records
both repetitions and uses the existing CPU-oracle operator tolerance (1e-12).
All raw Tile byte comparisons still require exact equality. The rejected audit
and both compiled source snapshots are retained; the production executable did
not change between these two audit builds.

Complete physical flow, exact twofold mass, restart and steady checks remain
separate from these byte tests. A bounded 256 probe does not establish steady
flow, agreement with Aphros, or spatial convergence.

The completed two-step regression retained CPU16 final and intermediate CSVs
byte for byte. GPU16 and GPU64 maximum relative L2 differences over velocity,
pressure, cut-cell velocity, wall shear and shared flux were 2.283e-16 and
1.124e-14. Both exact twofold conservation checks passed. GPU64 again used 8 and
27 inner iterations. Its observed peak working set was 458,014,720 bytes versus
492,568,576 for the retained preceding run; this is an observation across two
runs, not a controlled whole-machine memory benchmark. Both-part checkpoint
restore/continuation passed, and a legacy checkpoint omitting the tail remained
rejected.

Both native16 steady variants (dense and file-backed Anderson history) completed
17 outer iterations, passed the unchanged equation gates, and had maximum final
field relative L2 difference 4.255e-16. Their independent 120-digit twofold mass
checks had relative divergence below 1.981e-21.

## Bounded native256 flow probe

`native_flow256_probe_v9` ran the production executable from
`D:/CirrusExperiments/cirrus-amg/builds/native_host_storage_v2`, with both checked
host Tile backing and file-backed Anderson history enabled. The production
executable SHA-256 is
`7c361f13029c077115d855506e1b4468b2a329e57ff6a8ee6a013a673c325971`.

The offload reduced observed working set from 3,497,512,960 to 2,610,552,832 bytes
at its construction boundaries (886,960,128 bytes). The full probe's cumulative
peak working set was 9,944,985,600 bytes. The minimum sampled system availability
was 3,886,575,616 bytes, only about 123 MiB above the unchanged 3.5 GiB floor.
Ambient memory use varies; this does not certify enough headroom for a longer
run or concurrent large reference solve.

The native process hit its unchanged 1,500-second limit at
2026-09-09 21:15:16 UTC, with 4,688,351,232 bytes available. This was a time stop,
not a memory stop. It completed 26 pressure/viscosity linear solves, all below
the 1e-13 true relative residual gate. Two complete raw inner-map dumps and two
Anderson history updates were retained. Each map has 3,995,168 cell rows and
12,263,704 face rows; the history holds two full f/g pairs, 1,287,356,160 bytes on
D with zero resident history values. The second history update also read
1,287,356,160 bytes. The full native binary export was not reached at this size.

All seven retained raw/history/backing files total 8,331,461,976 bytes on D.
Their row counts (where applicable) and SHA-256 hashes are recorded in
`checks/native_flow256_probe_v9/retained_raw_artifacts.json`. These files are
diagnostic state, not a converged physical time step or a portable checkpoint.
No physical time step, steady256 solution, or new ParaView field was adopted.

The original CPU Aphros128 process (PID 42880, creation time 1788913281.3414037)
resumed at 21:16:11 UTC after a bounded 27-minute pause. Its private state,
immutable inputs and complete time prefix were preserved. The restoration
check observed increasing CPU time in the same process and no remaining native
probe or watchdog. Independent Aphros128 agreement and near-wall spatial
convergence remain outstanding.
