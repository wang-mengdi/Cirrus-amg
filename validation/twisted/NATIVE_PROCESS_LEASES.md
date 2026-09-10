# Preserve a native CUDA solve between bounded experiment slices

`scripts/run_twisted_persistent_solver.py` launches the ordinary verified solver
runner with its one native child initially suspended. The process identity record
includes the PID, creation time, original primary thread, executable/config hashes,
and launcher sources. The ordinary runner records completion only after the
native executable exits. Pausing does not create a completed physical step.

`scripts/native_process_lease.py` resumes that same process for a bounded slice.
The reference starts running, is parked and has its working set trimmed, then is
restored after the native process is parked and trimmed (or exits naturally).
Defaults retain the existing limits: native slice at most 1,500 s, reference pause
at most 1,800 s, 11 GiB starting headroom stable for 15 s, at most 180 s waiting for
headroom, 3.5 GiB available-RAM floor, and 15 GiB free on D. Memory and time checks
poll at 0.25 s; transition latency is included in the recorded reference pause.
Process records and experiment outputs are on D. Another live native GPU solver,
including a paused one or CUDA Aphros, prevents acquisition of a new slice.

The independent Python worker owns all normal process transitions and deadlines.
The coordinator never holds a transition lock while the worker is live. The
worker restores state if the coordinator exits. If the worker exits prematurely,
the coordinator can restore state after confirming worker exit. A deadline
fallback can terminate its own Python worker; it never terminates a solver.

Windows process wait-state labels are insufficient to determine whether the
application thread remains suspended: background threads can appear after a
suspend operation. `windows_process_control.py` uses documented
[SuspendThread](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-suspendthread)
and [ResumeThread](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-resumethread)
counts, keeps transitions idempotent between zero and one, and rejects externally
nested suspensions. Count inspection balances its temporary increment. This is
external experiment control, not synchronization between application threads.
`windows_atomic_json.py` retries Windows sharing denials for at most five seconds
when replacing status files. Concurrent status readers must also handle transient
sharing denials; an atomic replacement does not guarantee every open succeeds.

The native process retains its full state, including in-progress inner and outer
Anderson history and its CUDA context. No fields are reconstructed from partial
CSV dumps. This is not a durable checkpoint: process termination or reboot loses
that in-memory state, and paused processes still reserve memory and GPU resources.
The C++ operators, tolerances, and physical acceptance gates are unchanged.

## Checked evidence

All full outputs remain under `D:/CirrusExperiments/cirrus-amg`. Compact evidence,
failed attempts, executed scripts, source snapshots, and hashes are archived in
`results/native_process_leases/receipt.json`.

* `checks/native_process_leases_v3/result.json` passes normal bounded suspension,
  suspended coordinator, terminated coordinator, restored reference progress and
  unchanged 64 MiB CPU-sentinel state. It resumes the **same native process** left
  by the failed v2 test. `runs/native_process_leases64_v2` completes both physical
  steps with 8 and 27 inner iterations. Comparison against the uninterrupted
  `native_host_storage_gpu64_v1` passes all five field gates; maximum relative L2
  difference is **1.1338007282378248e-14**. Exact twofold mass checks also pass.
  The real original CPU Aphros128 process remained running throughout these tests.
* `checks/native_no_selected_dumps16_v2/result.json` checks the completed
  `runs/native_no_selected_dumps16_v1` against `native_host_storage_steady16_disk_v1`.
  Configurations differ only in output path and `dump_iterations=[]` versus
  `[0,1,2]`. All 17 completed raw maps and six scheduled full wall outputs agree,
  with maximum relative L2 **7.626153198733523e-16**. The unmodified steady validator
  and exact twofold mass check pass for both runs, including every raw map.
  Raw flux extraction preserves every row and both original text components.
  Final/converged map dumps and the normal field-output schedule remain mandatory.
* `checks/windows_atomic_json_v4/result.json` checks actual held-reader denial,
  bounded failure, successful replacement after release, and concurrent reads:
  147 complete records and four retried sharing denials, with no partial JSON.

The earlier failures are retained, not relabeled as passes. Lease v1 used an
unreliable process status label; its same-process recovery completed the native
flow checks but the CPU sentinel failed on a Windows sharing denial. Lease v2
exposed the suspended coordinator holding a transition lock. Atomic JSON v1 had
no overlapping reader and v2/v3 exposed reader-side sharing denials. The first
no-dump checker incorrectly expected full fields at every outer iteration; the
corrected checker uses every converged raw map and all scheduled full fields from
the already completed run, without rerunning or changing that solution.

These are process-control and output-schedule regressions. They do **not** establish
native256 convergence, native128/Aphros128 agreement, or spatial convergence of
near-wall velocity and wall shear. Those physical goal requirements remain open.

## First persistent native256 slice

`D:/CirrusExperiments/cirrus-amg/runs/native_flow256_persistent_v1` uses the same
native executable and twisted cut-cell geometry as the previous v9 probe. It
requests the already validated steady outer iteration, 64 maximum outer maps,
field output stride 16, and no optional selected inner dumps. The geometry,
operator metadata and both complete mesh tables match v9 byte for byte.

The first slice ran from 2026-09-09 22:12:39 UTC until its 1,500 s deadline. The
guardian parked native PID 71580 (creation time 1788991861.8417583) and restored
the original CPU Aphros128 PID 42880. The recorded reference pause was
1,526.772783 s, within the existing 1,800 s limit. Both reference threads resumed,
user CPU time advanced, and its previously completed time-history prefix remained
unchanged. The native executable was not terminated or restarted.

There are 38 completed GPU solves covering pressure and implicit viscosity, with
maximum accepted relative residual 9.965932364070488e-14. Available memory never
fell below 4,276,658,176 bytes in the recorded samples; maximum sampled native RSS
was 10,208,976,896 bytes. The first logged nonlinear update reports continuity
2.28068526555e-17, but no converged outer map is established. Some C++ history
streams remain buffered while the process is paused, so visible CSV rows are not
treated as a complete count of in-memory iterations.

The verified record is `checks/native_flow256_persistent_v1/slice_0001_progress.json`
under the D experiment root. `results/native_flow256_persistent_slice1/receipt.json`
archives completed controller records and immutable complete-prefix log snapshots.
It does not archive changing live logs as immutable numerical evidence, and does
not claim completed physical or spatial acceptance. Subsequent slices must resume
the recorded process; a cold restart would discard this progress.

### Measure warm resume separately from cold start

The first full resume attempt did not run: after 180 s, available memory stayed
below the 11 GiB cold-start criterion. A 10 GiB / 60 s preparation probe also
timed out before resuming. Both attempts restored the original reference and
kept the native state; neither is counted as GPU progress.

A subsequent explicitly bounded test used 9.5 GiB starting headroom, a 30 s
native slice, and a 240 s reference-pause ceiling. The 3.5 GiB operating floor,
15 s stable-headroom requirement, disk floor and all numerical gates remained
unchanged. That **same process** advanced from 38 to 39 completed GPU calls. The
resumed pressure call passed its original-RHS checks with residual
9.354665859288709e-14. The measured start availability was 10,671,812,608 bytes;
minimum availability was 8,037,875,712 bytes and maximum sampled native RSS was
3,259,224,064 bytes. The reference was restored after 71.761079 s.

This supports using `--minimum-start-gib 9.5` for subsequent slices of this
already initialized process. It does not change the default 11 GiB cold-start
criterion or prove a bound on later phases' memory use: the runtime floor and
existing 1,500/1,800 s limits continue to apply. No process state or numerical
configuration was changed to make the measurement pass. Later phases still
require observation and full physical acceptance.

`checks/native_flow256_persistent_v1/warm_resume_measurement_v2.json` on D is the
measurement; `results/native_flow256_warm_resume/receipt.json` retains both failed
preparations and the successful test. A `gpu_linear.csv` call spanning a pause
includes that pause in its wall-clock `seconds` value (call 39 reports about
706 s). Such a row is valid numerical evidence, not a standalone GPU speed
measurement; use slice activity records when accounting for execution time.

### Later slice reaches the memory floor

Slice 5 resumed the same PID 71580 with the measured 9.5 GiB warm-start
criterion. It completed another 18 linear solves (39 to 57 cumulatively), all
within the unchanged 1e-13 residual gate. Three complete disk-backed inner
Anderson-history records are visible. No converged outer map is established.

After 479.274916 active seconds, the guardian observed available RAM below the
3.5 GiB operating floor and parked the native process. Minimum sampled available
RAM was 3,750,600,704 bytes versus the 3,758,096,384-byte floor. The original CPU
Aphros128 reference resumed after 522.787005 seconds; its thread counts, process
identity, advancing user CPU time and existing time-history prefix were checked.
The native state remains in the same live process, not a durable checkpoint.

`checks/native_flow256_persistent_v1/slice_0005_progress.json` explicitly records
`observed_memory_floor_breach=true` and `full_slice_resource_acceptance=false`.
Its passing audit means valid completed linear records and verified restoration,
not completion of the planned 1,500-second slice or physical acceptance. The
immutable snapshots and executed producer are retained in
`results/native_flow256_memory_stop/receipt.json`. Bulky experiment data stay on D.

### Distinguish configured and observed warm headroom

Slice 6 did not resume the native process: the 9.5 GiB preparation condition
was not met within 180 s. Both process states were preserved, and the original
reference resumed after 181.575186 s. No additional GPU call is attributed to it.

Slice 7 was a 30-second probe configured with an 8 GiB starting threshold. The
same native process completed pressure call 58, whose original-RHS checks and
8.111347400771126e-14 relative residual passed. However, available memory had
actually recovered to 11,834,720,256 bytes at resume, above the prior 9.5 GiB
criterion. This run therefore does **not** validate operation starting at 8 GiB.
Minimum sampled availability was 6,169,190,400 bytes; the reference resumed after
56.437658 s. The 3.5 GiB operating floor and numerical configuration were unchanged.

The first v3 checker verified progress but described the configured threshold as
a measured setting. The corrected v4 checker uses the same completed run, keeps
the v3 result as superseded evidence, and explicitly records
`lower_start_headroom_accepted=false` and `subsequent_start_minimum_gib=9.5`.
Subsequent full slices retain 9.5 GiB; the 11 GiB cold-start default is unchanged.
`results/native_flow256_warm_headroom_observation/receipt.json` retains both
attempts, the corrected interpretation and immutable snapshots. Physical and
spatial acceptance remain incomplete.

### Slice 9 continues the same state after another preparation timeout

Slice 8 timed out before resuming, so it added no GPU work. Slice 9 ran the same
native process for 481.487025 seconds before the unchanged memory floor stopped
its slice. It added 19 completed linear solves, from 58 to 77, all within the
original tolerance. Four disk-history records are visible; there is still no
converged outer map. The original reference resumed after 509.255640 seconds.

The minimum recorded availability was 3,725,037,568 bytes, below the unchanged
3,758,096,384-byte floor. `slice_0009_progress.json` therefore records a verified
restoration and valid completed linear records, not full resource or physical
acceptance. Immutable evidence for slices 8 and 9 is retained in
`results/native_flow256_slice9/receipt.json`; solver state remains in the same
live process, not a durable restart file.

### Intra-slice working-set trim pilot

Slice 10 reached the same 3.5 GiB floor after about 162 seconds, ending at 81
completed GPU calls. Repeated full lease transitions preserve progress but incur
reference working-set restoration and another preparation wait.

The experimental D-side controller `configs/native_process_lease_trim_v1.py`
adds a worker-owned park/trim/resume inside an existing slice. It retains the
original coordinator fallback, never extends either deadline, and closes the
lease if a deadline or memory floor is reached before resuming. It checks that
private allocation and user CPU time remain unchanged while parked. The original
controller and both solver executables are unchanged.

Owned 64 MiB CPU sentinels pass normal resume, an injected exception after trim,
and expiration before resume; their data hashes survive and the reference is
restored in all cases. The first test incorrectly assumed a Python process had
one thread; the corrected sentinel records its actual primary thread ID. Both
attempts are retained. Neither test operates on the real flow processes.

A real 90-second pilot resumed the same native PID 71580, performed one trim after
15 seconds, and completed three additional pressure solves (81 to 84). The trim
lasted 0.640125 seconds. Its working set fell by 3,206,086,656 bytes while private
allocation was unchanged, but immediate available RAM rose by only 75,735,040
bytes. Working-set reduction is therefore not equated with reclaimed system RAM.
Minimum sampled availability was 4,827,246,592 bytes. The original Aphros128
reference resumed after 129.835456 seconds, within the pilot's 240-second bound.
All completed linear records pass the original residual and original-RHS checks.

This is a control/resource pilot, not a complete CUDA field regression or proof
of sustained memory benefit. Earlier full native64 pause/trim validation remains
separate. `results/native_intra_slice_trim/receipt.json` retains the executed
controller, CPU tests, closed slices and immutable complete-prefix snapshots.
The planned full trim measurement did not start: the original reference had
already exited under its own memory guard. The subsequent native-only probes
and final paused state are documented below.


### Reference exit and handoff to a larger machine

The original CPU Aphros128 process exited with code 15 at 2026-09-09
23:53:11 UTC. Its own preexisting guard observed available RAM below 2.5 GiB
for 10 seconds. Only two physical steps completed; no steady reference was
accepted. The process was not manually terminated by the lease controller.

Native-only slice 13 also reached the unchanged 3.5 GiB floor. Slice 14
completed its 150-second probe and parked the same native PID 71580 at
2026-09-10 00:07:01 UTC. There are 93 completed GPU linear records, all within
the existing residual gate, but no completed steady outer map. The final live
state is retained, not portable. No further local flow work is scheduled.

The user elected to wait for a larger machine. See
[REPRODUCE_ON_LARGE_MACHINE.md](REPRODUCE_ON_LARGE_MACHINE.md) for the current
acceptance status, frozen dependency bundle, fresh-run commands and remaining
checks. The goal remains incomplete and paused at the user's request.
