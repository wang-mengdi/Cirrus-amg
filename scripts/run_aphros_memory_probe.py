"""Replay a reference configuration with another executable and measure memory.

Preserve its physical configuration, original geometry, and runtime controls.
The optional resource stop applies only to the newly observed native child.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil
from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference, exe, case = args.reference.resolve(), args.executable.resolve(), args.output.resolve()
    runtime = json.loads((reference/'run_manifest.json').read_text(encoding='utf-8-sig'))
    env = runtime['environment']
    expected = {
        'OMP_WAIT_POLICY': 'PASSIVE', 'APHROS_TWISTED_DIRECT': '1', 'APHROS_SIMPLE_DUMP': 'simple',
        'APHROS_TWISTED_GEOMETRY': 'tube', 'APHROS_TWISTED_TIME_DUMP': 'tube',
        'APHROS_TWISTED_AMG': '1',
    }
    if any(env.get(k) != v for k, v in expected.items()):
        raise ValueError('Require the existing original-geometry AMG reference controls')
    flags = {
        'APHROS_TWISTED_FACTOR_TRACE': '-TraceFactorCache',
        'APHROS_TWISTED_CAPTURE_PRESSURE': '-DumpPressureSystem',
        'APHROS_TWISTED_CAPTURE_PRESSURE_FACES': '-DumpPressureFaces',
        'APHROS_TWISTED_VOLUME_COMPATIBILITY': '-VolumePressureCompatibility',
    }
    allowed = {*expected, *flags, 'OMP_NUM_THREADS', 'APHROS_TWISTED_FACTOR_CACHE',
               'APHROS_TWISTED_GEOMETRY_STATE_IN'}
    if any(v is not None and k not in allowed for k, v in env.items()):
        raise ValueError('Unrecognized active reference control')
    if any(env.get(k) not in (None, '1') for k in flags):
        raise ValueError('Unexpected reference boolean control')
    geometry = Path(runtime['geometry_state']).resolve()
    if str(geometry) != env['APHROS_TWISTED_GEOMETRY_STATE_IN']:
        raise ValueError('Different original geometry paths')
    if sha(geometry) != runtime['geometry_state_sha256']:
        raise ValueError('Original geometry changed')
    if sha(reference/'a.conf') != runtime['config_sha256']:
        raise ValueError('Reference configuration changed')
    build = json.loads((exe.parent/'build_manifest.json').read_text())
    if not build['passed'] or sha(exe) != build['executable_sha256']:
        raise ValueError('Executable is not a completed verified build')
    for name, value in build['source_sha256'].items():
        if sha(Path(name)) != value:
            raise ValueError('Build source changed: ' + name)
    repo = Path(__file__).resolve().parent.parent
    runner = repo/'scripts/run_twisted_baseline.ps1'
    pwsh = Path('C:/Users/bear/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe')
    case.mkdir(parents=True, exist_ok=False)
    for name in ('a.conf', 'case_manifest.json'):
        shutil.copyfile(reference/name, case/name)
    inputs = {str(p): sha(p) for p in (exe, geometry, runner, Path(__file__).resolve(),
              exe.parent/'build_manifest.json', reference/'run_manifest.json',
              reference/'a.conf', reference/'case_manifest.json', case/'a.conf', case/'case_manifest.json')}
    command = [str(pwsh), '-NoProfile', '-File', str(runner), '-CaseDirectory', str(case),
               '-Executable', str(exe), '-Threads', env['OMP_NUM_THREADS'], '-UseAmg',
               '-FactorCacheEntries', env['APHROS_TWISTED_FACTOR_CACHE'], '-GeometryState', str(geometry)]
    command += [flag for key, flag in flags.items() if env.get(key) == '1']
    (case/'probe_manifest.json').write_text(json.dumps({
        'scope': __doc__, 'reference': str(reference), 'command': command, 'source_sha256': inputs,
        'minimum_available_memory_bytes': 2*1024**3, 'required_low_memory_seconds': 10,
    }, indent=2)+'\n')
    identity = native = minimum_available = low_since = None
    peak_rss = peak_private = samples = 0
    stopped = False
    start = time.perf_counter()
    with (case/'wrapper.log').open('w') as log, (case/'memory.csv').open('x', newline='') as trace:
        writer = csv.writer(trace)
        writer.writerow(['elapsed_seconds', 'pid', 'working_set_bytes', 'private_bytes',
                         'peak_working_set_bytes', 'available_system_bytes'])
        process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        wrapper = psutil.Process(process.pid)
        print(json.dumps({'wrapper_pid': process.pid, 'case': str(case)}), flush=True)
        while process.poll() is None:
            if native is None:
                try:
                    matches = [p for p in wrapper.children(recursive=True)
                               if Path(p.exe()).resolve() == exe]
                except psutil.NoSuchProcess:
                    matches = []
                if len(matches) > 1:
                    raise ValueError('Multiple native solver children')
                if matches:
                    native = matches[0]
                    identity = {'pid': native.pid, 'creation_time': native.create_time(),
                                'creation_utc': datetime.fromtimestamp(native.create_time(), timezone.utc).isoformat(),
                                'executable': str(exe)}
                    (case/'process_identity.json').write_text(json.dumps(identity, indent=2)+'\n')
                    print(json.dumps({'native_process': identity}), flush=True)
            if native is not None:
                try:
                    current = psutil.Process(identity['pid'])
                    if current.create_time() != identity['creation_time'] or Path(current.exe()).resolve() != exe:
                        raise ValueError('Native process identity changed')
                    memory = current.memory_info()
                    available = psutil.virtual_memory().available
                    samples += 1
                    peak_rss = max(peak_rss, memory.peak_wset)
                    peak_private = max(peak_private, memory.private)
                    minimum_available = available if minimum_available is None else min(minimum_available, available)
                    writer.writerow([time.perf_counter()-start, native.pid, memory.rss,
                                     memory.private, memory.peak_wset, available])
                    trace.flush()
                    low_since = (low_since if low_since is not None else time.perf_counter()) if available < 2*1024**3 else None
                    if not stopped and low_since is not None and time.perf_counter()-low_since >= 10:
                        current.terminate()
                        stopped = True
                        print(json.dumps({'stopped_for_memory': True, 'identity': identity,
                                          'available_bytes': available}), flush=True)
                except psutil.NoSuchProcess:
                    pass
            time.sleep(.5)
        code = process.wait()
    unchanged = all(sha(Path(p)) == h for p, h in inputs.items())
    actual = json.loads((case/'run_manifest.json').read_text(encoding='utf-8-sig'))
    controls_equal = actual['environment'] == env
    summary = {'scope': __doc__+' Completion alone is not an accuracy check.',
               'native_process': identity, 'wrapper_exit_code': code,
               'stopped_for_sustained_low_memory': stopped, 'memory_samples': samples,
               'peak_working_set_bytes': peak_rss, 'maximum_observed_private_bytes': peak_private,
               'minimum_observed_system_available_bytes': minimum_available,
               'wall_seconds': time.perf_counter()-start, 'inputs_unchanged': unchanged,
               'identical_reference_runtime_controls': controls_equal, 'source_sha256': inputs}
    (case/'resource_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}), flush=True)
    if code or stopped or not unchanged or not controls_equal:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
