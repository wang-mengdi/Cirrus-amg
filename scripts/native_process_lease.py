"""Bound a live native solver time slice while preserving both process states.

The native process starts and ends suspended (or exits naturally). The reference
starts and ends running. An independent worker owns every normal transition and
deadline; suspending the coordinator cannot strand a transition lock. The
coordinator can restore state only after the worker exits. Solver processes are
never terminated and no numerical state is imported or rewritten.
"""
import argparse
from contextlib import contextmanager
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
import windows_process_control as control
import windows_atomic_json as atomic


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    atomic.write_json(path, value)


def identity(record, missing_ok=False):
    try:
        process = psutil.Process(record['pid'])
        if process.create_time() != record['creation_time']:
            if missing_ok:
                return None
            raise RuntimeError('Process ID was reused')
        if Path(process.exe()).resolve() != Path(record['executable']).resolve():
            raise RuntimeError('Process executable identity changed')
        if 'command' in record:
            actual = process.cmdline()
            if len(actual) != len(record['command']) or any(
                Path(a).resolve() != Path(b).resolve() for a, b in zip(actual, record['command'])):
                raise RuntimeError('Native process command changed')
        return process
    except psutil.NoSuchProcess:
        if missing_ok:
            return None
        raise


def observe(process):
    memory = process.memory_info()
    return {'utc': now(), 'pid': process.pid, 'creation_time': process.create_time(),
            'status': process.status(), 'cpu_seconds': sum(process.cpu_times()[:2]),
            'user_cpu_seconds': process.cpu_times().user,
            'rss_bytes': memory.rss, 'private_bytes': memory.private,
            'available_bytes': psutil.virtual_memory().available}


def trim(process):
    control.park(process)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    api.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    api.EmptyWorkingSet.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x0100 | 0x0400, False, process.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not api.EmptyWorkingSet(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


@contextmanager
def transition_lock(root):
    # Only the worker uses this while it is live. The coordinator may recover
    # after worker exit, when the kernel has released the worker's byte lock.
    with (root/'transition.lock').open('a+b') as stream:
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(.05)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def close_lease(root, settings, reason, actor):
    with transition_lock(root):
        if (root/'closed.json').exists():
            return json.loads((root/'closed.json').read_text())
        native = identity(settings['native'], missing_ok=True)
        before = None
        parked = None
        try:
            if native is not None:
                before = observe(native)
                control.park(native)
                trim(native)
                parked = observe(native)
                parked['primary_thread_suspended'] = control.primary_parked(native, settings['native']['primary_thread_id'])
                if not parked['primary_thread_suspended']:
                    raise RuntimeError('Native primary thread was not parked')
        except psutil.NoSuchProcess:
            native = None
        reference = identity(settings['reference'])
        reference_threads = control.resume(reference)
        restored = observe(reference)
        pause = root/'reference_paused.json'
        paused_seconds = time.time()-json.loads(pause.read_text())['pause_epoch'] if pause.exists() else 0.
        result = {'reason': reason, 'actor': actor, 'closed_utc': now(),
                  'native_before_park': before, 'native_parked': parked,
                  'native_exited': native is None, 'reference_restored': restored,
                  'reference_thread_counts_after_resume': reference_threads,
                  'reference_pause_seconds': paused_seconds,
                  'reference_pause_within_bound': paused_seconds <= settings['maximum_reference_pause_seconds'],
                  'solver_state_modified': False, 'native_terminated': False}
        write(root/'closed.json', result)
        return result


def coordinator_alive(settings):
    try:
        return psutil.Process(settings['coordinator_pid']).create_time() == settings['coordinator_creation_time']
    except psutil.NoSuchProcess:
        return False


def stop_reason(root, settings):
    if not coordinator_alive(settings):
        return 'coordinator_exited'
    if time.time() >= settings['reference_deadline_epoch']-15:
        return 'reference_deadline_guard'
    if psutil.disk_usage(root.drive+'/').free < settings['minimum_disk_free_bytes']:
        return 'disk_floor_guard'
    return None


def execute_slice(root, settings):
    native = identity(settings['native'])
    reference = identity(settings['reference'])
    reason = stop_reason(root, settings)
    if reason:
        return reason
    with transition_lock(root):
        before = observe(reference)
        pause_epoch = time.time()
        # Record before the first transition so fallback can account for a
        # worker failure midway through parking, without losing the deadline.
        pause_record = {'before': before, 'pause_epoch': pause_epoch}
        write(root/'reference_paused.json', pause_record)
        control.park(reference)
        paused = observe(reference)
        trim(reference)
        after_trim = observe(reference)
        pause_record.update(paused=paused, after_trim=after_trim)
        write(root/'reference_paused.json', pause_record)
        if after_trim['private_bytes'] != paused['private_bytes']:
            raise RuntimeError('Reference private allocation changed during trim')
    deadline = time.monotonic()+settings['headroom_timeout_seconds']
    stable = None
    while True:
        reason = stop_reason(root, settings)
        if reason:
            return reason
        available = psutil.virtual_memory().available
        current = time.monotonic()
        stable = (current if stable is None else stable) if available >= settings['minimum_start_available_bytes'] and not other_native(native.pid) else None
        write(root/'status.json', {'phase': 'waiting_for_headroom', 'utc': now(), 'available_bytes': available})
        if stable is not None and current-stable >= settings['headroom_stable_seconds']:
            break
        if current >= deadline:
            return 'headroom_timeout'
        time.sleep(.25)
    with transition_lock(root):
        start = time.time()
        active = {'started_utc': now(), 'start_epoch': start,
                  'deadline_epoch': min(start+settings['maximum_native_active_seconds'], settings['reference_deadline_epoch']-15),
                  'native_before_resume': observe(native)}
        write(root/'active.json', active)
        control.resume(native)
    with (root/'memory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['utc', 'available_bytes', 'native_rss_bytes', 'native_cpu_seconds'])
        writer.writeheader()
        while True:
            reason = stop_reason(root, settings)
            if reason:
                return reason
            native = identity(settings['native'], missing_ok=True)
            if native is None:
                return 'native_exited'
            try:
                sample = observe(native)
            except psutil.NoSuchProcess:
                return 'native_exited'
            row = {'utc': sample['utc'], 'available_bytes': sample['available_bytes'],
                   'native_rss_bytes': sample['rss_bytes'], 'native_cpu_seconds': sample['cpu_seconds']}
            writer.writerow(row)
            stream.flush()
            write(root/'status.json', {'phase': 'native_running', **row})
            if sample['available_bytes'] < settings['minimum_available_bytes']:
                return 'memory_floor_guard'
            if time.time() >= active['deadline_epoch']:
                return 'native_time_limit_guard'
            if identity(settings['reference']).cpu_times().user > paused['user_cpu_seconds']+.02 or other_native(native.pid):
                return 'external_process_state_change'
            time.sleep(.25)


def guard(root):
    settings = json.loads((root/'lease.json').read_text())
    write(root/'guardian_ready.json', {'pid': os.getpid(), 'creation_time': psutil.Process().create_time(), 'utc': now()})
    reason, error = 'worker_exception', None
    try:
        reason = execute_slice(root, settings)
    except BaseException as exc:
        error = repr(exc)
    finally:
        closed = close_lease(root, settings, reason, 'guardian')
        unchanged = all(sha(path) == digest for path, digest in settings['source_sha256'].items())
        result = {'scope': __doc__, 'passed': error is None and unchanged and closed['reference_pause_within_bound'],
                  'completed_utc': now(), 'error': error, 'closed': closed,
                  'inputs_unchanged': unchanged, 'source_sha256': settings['source_sha256'],
                  'physical_convergence_checked': False, 'goal_complete': False}
        write(root/'result.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}), flush=True)
    if not result['passed']:
        raise RuntimeError('Process lease failed; inspect result.json')


def other_native(native_pid):
    for process in psutil.process_iter(['name']):
        if process.pid == native_pid:
            continue
        try:
            if process.info['name'] in ('simple_channel.exe', 'native_compact_gpu_audit.exe') or (
                process.info['name'] == 'twisted_extended.exe' and
                process.environ().get('APHROS_TWISTED_CUDA_AMG') is not None):
                return True
        except psutil.NoSuchProcess:
            pass
    return False


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--guard':
        guard(Path(sys.argv[2]).resolve())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--active-seconds', type=float, default=1500)
    parser.add_argument('--pause-seconds', type=float, default=1800)
    parser.add_argument('--minimum-start-gib', type=float, default=11)
    parser.add_argument('--floor-gib', type=float, default=3.5)
    parser.add_argument('--stable-seconds', type=float, default=15)
    parser.add_argument('--headroom-timeout', type=float, default=180)
    parser.add_argument('--minimum-disk-free-gib', type=float, default=15)
    args = parser.parse_args()
    if not (0 < args.active_seconds <= 1500 and args.active_seconds+15 <= args.pause_seconds <= 1800):
        raise ValueError('Keep native slices at most 1500 seconds and reference pauses at most 1800 seconds')
    if not (args.minimum_start_gib >= args.floor_gib >= 3.5 and 0 <= args.stable_seconds <= args.headroom_timeout <= 180):
        raise ValueError('Invalid memory or headroom bounds')
    if args.minimum_disk_free_gib < 15:
        raise ValueError('Retain at least 15 GiB free for experiment output')
    root = args.output.resolve()
    if root.drive.lower() != 'd:':
        raise ValueError('Write process lease state on D')
    native_path, reference_path = args.native.resolve(strict=True), args.reference.resolve(strict=True)
    native_record = json.loads(native_path.read_text())
    reference_record = json.loads(reference_path.read_text())
    if native_record['pid'] == reference_record['pid']:
        raise ValueError('Native and reference must be different processes')
    native, reference = identity(native_record), identity(reference_record)
    if 'primary_thread_id' not in native_record:
        raise ValueError('Native process record must identify its original primary thread')
    if not control.primary_parked(native, native_record['primary_thread_id']) or not control.all_running(reference):
        raise RuntimeError('Require a parked native and a running reference before acquiring a lease')
    if other_native(native.pid):
        raise RuntimeError('Another native GPU experiment is live')
    sources = {str(Path(__file__).resolve()): sha(__file__), str(native_path): sha(native_path), str(reference_path): sha(reference_path)}
    sources[str(Path(control.__file__).resolve())] = sha(control.__file__)
    sources[str(Path(atomic.__file__).resolve())] = sha(atomic.__file__)
    sources[reference_record['executable']] = sha(reference_record['executable'])
    for path, digest in reference_record.get('source_sha256', {}).items():
        if sha(path) != digest:
            raise RuntimeError('Reference input changed')
        sources[path] = digest
    for key in ('executable', 'config'):
        value = sha(native_record[key])
        if value != native_record[key+'_sha256']:
            raise RuntimeError('Native run input changed')
        sources[native_record[key]] = value
    for path, digest in native_record.get('source_sha256', {}).items():
        if sha(path) != digest:
            raise RuntimeError('Native launcher source changed')
        sources[path] = digest
    root.mkdir(parents=True, exist_ok=False)
    (root/'transition.lock').write_bytes(b'0')
    owner = psutil.Process()
    settings = {'started_utc': now(), 'coordinator_pid': owner.pid, 'coordinator_creation_time': owner.create_time(),
                'native': native_record, 'reference': reference_record,
                'maximum_native_active_seconds': args.active_seconds,
                'maximum_reference_pause_seconds': args.pause_seconds,
                'headroom_timeout_seconds': args.headroom_timeout,
                'headroom_stable_seconds': args.stable_seconds,
                'reference_deadline_epoch': time.time()+args.pause_seconds,
                'minimum_start_available_bytes': int(args.minimum_start_gib*2**30),
                'minimum_available_bytes': int(args.floor_gib*2**30),
                'minimum_disk_free_bytes': int(args.minimum_disk_free_gib*2**30),
                'source_sha256': sources}
    write(root/'lease.json', settings)
    watcher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--guard', str(root)],
                               creationflags=subprocess.CREATE_NO_WINDOW)
    write(root/'guardian_identity.json', {'pid': watcher.pid, 'creation_time': psutil.Process(watcher.pid).create_time()})
    # This process never owns a transition lock while the worker is live.
    # A suspended/dead coordinator therefore cannot prevent timely restoration.
    forced = False
    while watcher.poll() is None:
        if time.time() >= settings['reference_deadline_epoch']-8:
            # Terminate only our Python control worker, never either solver.
            # Waiting for its exit makes fallback the sole transition owner.
            watcher.terminate()
            watcher.wait(timeout=3)
            forced = True
            break
        time.sleep(.2)
    if not (root/'closed.json').exists():
        closed = close_lease(root, settings, 'worker_deadline' if forced else 'worker_exited', 'coordinator_fallback')
        result = {'scope': __doc__, 'passed': False, 'completed_utc': now(),
                  'error': 'Worker did not finish restoration', 'closed': closed,
                  'worker_exit_code': watcher.returncode, 'source_sha256': sources,
                  'physical_convergence_checked': False, 'goal_complete': False}
        write(root/'result.json', result)
    if not (root/'result.json').exists():
        raise RuntimeError('Worker restored processes but did not finish its report')
    result = json.loads((root/'result.json').read_text())
    if forced or watcher.returncode != 0 or not result['passed']:
        raise RuntimeError('Process lease failed; inspect result.json')


if __name__ == '__main__':
    main()
