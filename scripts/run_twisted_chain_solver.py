"""Run a native tube case with optional verified recursively verified chain restart."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def windows_memory(process):
    """Read cumulative OS counters while the child process handle is alive."""
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(wintypes.HANDLE(int(process._handle)), ctypes.byref(counters), counters.cb):
        return None
    return {'peak_working_set_bytes': counters.PeakWorkingSetSize,
            'peak_pagefile_usage_bytes': counters.PeakPagefileUsage,
            'observed_private_bytes': counters.PrivateUsage}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dump-advection', action='store_true')
    parser.add_argument('--dump-operators', action='store_true', help='Read-only embedded sparse operators for consistency/stability diagnostics')
    parser.add_argument('--trace-projection', action='store_true', help='Read-only pressure projection residual and timing trace')
    parser.add_argument('--measure-memory', action='store_true', help='Record Windows process peak counters without changing solver settings')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Threads must be positive')
    if args.measure_memory and os.name != 'nt':
        parser.error('Process peak measurement is currently implemented for Windows')
    repo = Path(__file__).resolve().parents[1]
    config, exe = args.config.resolve(strict=True), args.exe.resolve(strict=True)
    case = json.loads(config.read_text(encoding='utf-8-sig'))
    geometry_hashes = {}
    if case.get('embedded_geometry'):
        geometry = Path(case['embedded_geometry'])
        if not geometry.is_absolute():
            geometry = repo/geometry
        geometry = geometry.resolve(strict=True)
        metadata = geometry.with_suffix('.meta.json')
        details = json.loads((metadata if metadata.exists() else geometry).read_text(encoding='utf-8-sig'))
        geometry_hashes[str(geometry)] = sha(geometry)
        if details['format'] == 'aphros_cut_geometry_v2':
            if details != json.loads(geometry.read_text(encoding='utf-8-sig')):
                raise ValueError('Packed geometry metadata sidecar differs from the actual input')
            for table in details['tables'].values():
                path = (geometry.parent/table['file']).resolve(strict=True)
                value = sha(path)
                if value != table['sha256']:
                    raise ValueError('Packed input differs from its geometry manifest')
                geometry_hashes[str(path)] = value
        # A legacy input without a sidecar may contain millions of JSON values.
        # Do not keep that temporary parse alive alongside the solver process.
        del details
    output = Path(case['output'])
    if not output.is_absolute():
        output = repo/output
    output = output.resolve()
    restart = None
    restart_inputs = {}
    if case.get('restart_checkpoint'):
        from twisted_chain_restart import load_checkpoint, copy_prefix
        checkpoint = Path(case['restart_checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint = repo/checkpoint
        restart, restart_inputs = load_checkpoint(checkpoint, case)
    # An existing result, failed run, or live run must never be overwritten.
    output.mkdir(parents=True, exist_ok=False)
    if restart is not None:
        copy_prefix(restart, output)
    settings = {'OMP_NUM_THREADS': str(args.threads), 'OMP_WAIT_POLICY': 'PASSIVE',
                'SIMPLE_ADV_DUMP': '1' if args.dump_advection else None,
                'SIMPLE_PROJECTION_TRACE': '1' if args.trace_projection else None,
                'SIMPLE_EMBEDDED_OPERATOR_DUMP': '1' if args.dump_operators else None}
    env = os.environ.copy()
    for key, value in settings.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    def now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {'executable': str(exe), 'executable_sha256': sha(exe),
                'config': str(config), 'config_sha256': sha(config),
                'environment': settings, 'started_utc': now(),
                'measure_memory': args.measure_memory,
                'geometry_input_sha256': geometry_hashes,
                'source_at_launch_sha256': {str(p.relative_to(repo)): sha(p)
                    for p in sorted((repo/'simple').glob('*')) if p.is_file()},
                'source_note': 'Launch-time sources; executable hash identifies the actual compiled program'}
    if restart is not None:
        manifest['restart_step'] = restart['physical_step']
        manifest['restart_input_sha256'] = restart_inputs
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Running {output.name}; log: {output / "run.log"}', flush=True)
    memory = {'samples': 0, 'scope': 'Windows cumulative process peak counters; private bytes are an observed maximum'} if args.measure_memory else None
    with (output/'run.log').open('w', encoding='utf-8') as log:
        with subprocess.Popen([str(exe), str(config)], cwd=repo, env=env,
                              stdout=log, stderr=subprocess.STDOUT) as result:
            while True:
                if args.measure_memory:
                    sample = windows_memory(result)
                    if sample is not None:
                        memory['samples'] += 1
                        for key, value in sample.items():
                            memory[key] = max(memory.get(key, 0), value)
                try:
                    result.wait(timeout=.2 if args.measure_memory else None)
                    # The process may peak between the final periodic sample and
                    # exit. Its Windows handle remains open until this context ends.
                    if args.measure_memory:
                        sample = windows_memory(result)
                        if sample is not None:
                            memory['samples'] += 1
                            for key, value in sample.items():
                                memory[key] = max(memory.get(key, 0), value)
                    break
                except subprocess.TimeoutExpired:
                    pass
    completion = {'exit_code': result.returncode, 'completed_utc': now(),
                  'executable_unchanged': sha(exe) == manifest['executable_sha256'],
                  'config_unchanged': sha(config) == manifest['config_sha256'],
                  'geometry_unchanged': all(sha(Path(path)) == value for path, value in geometry_hashes.items())}
    if restart is not None:
        completion['restart_unchanged'] = all(sha(Path(path)) == value for path, value in restart_inputs.items())
    if args.measure_memory:
        completion['resource_usage'] = memory
    (output/'run_completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    print(json.dumps(completion), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    if not all(completion[key] for key in ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
        raise RuntimeError('Executable, configuration, or geometry changed during this run')
    if restart is not None and not completion['restart_unchanged']:
        raise RuntimeError('Completed parent or restart checkpoint changed during this run')


if __name__ == '__main__':
    main()
