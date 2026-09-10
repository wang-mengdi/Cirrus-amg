"""Launch the ordinary verified solver runner with its native child initially suspended.

The same native process may be resumed by native_process_lease.py. The ordinary
runner writes completion only when the native process actually exits. No solver
state is imported, reconstructed, or changed by this wrapper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
import run_twisted_solver as ordinary


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process-record', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if os.name != 'nt':
        raise RuntimeError('This process lease runner requires Windows')
    record = args.process_record.resolve()
    if record.drive.lower() != 'd:' or record.exists():
        raise ValueError('Choose a fresh process record on D')
    record.parent.mkdir(parents=True, exist_ok=True)
    original_popen = subprocess.Popen
    sources = {str(Path(__file__).resolve()): sha(__file__),
               str(Path(ordinary.__file__).resolve()): sha(ordinary.__file__)}
    launched = False

    def start_paused(command, *positional, **keywords):
        nonlocal launched
        if launched or len(command) != 2:
            raise RuntimeError('Expected exactly one native solver child')
        exe, config = map(lambda value: Path(value).resolve(strict=True), command)
        if exe.name != 'simple_channel.exe':
            raise ValueError('Unexpected native executable')
        for other in psutil.process_iter(['name']):
            try:
                if other.info['name'] in ('simple_channel.exe', 'native_compact_gpu_audit.exe') or (
                    other.info['name'] == 'twisted_extended.exe' and
                    other.environ().get('APHROS_TWISTED_CUDA_AMG') is not None):
                    raise RuntimeError('Another native GPU experiment is live')
            except psutil.NoSuchProcess:
                pass
        case = json.loads(config.read_text(encoding='utf-8-sig'))
        keywords['creationflags'] = keywords.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW | 0x00000004
        child = original_popen(command, *positional, **keywords)
        process = psutil.Process(child.pid)
        owner = psutil.Process()
        initial_threads = process.threads()
        if len(initial_threads) != 1:
            raise RuntimeError(f'Expected one primary thread in newly suspended child {child.pid}')
        identity = {'pid': child.pid, 'creation_time': process.create_time(),
                    'primary_thread_id': initial_threads[0].id,
                    'executable': str(exe), 'executable_sha256': sha(exe),
                    'config': str(config), 'config_sha256': sha(config),
                    'command': list(map(str, command)), 'output': case['output'],
                    'runner_pid': owner.pid, 'runner_creation_time': owner.create_time(),
                    'initially_suspended': True, 'source_sha256': sources,
                    'solver_state_modified': False}
        # Exclusive creation prevents a previous run identity from being replaced.
        # If writing fails, the child is still at its initial suspended entry.
        try:
            with record.open('x') as stream:
                json.dump(identity, stream, indent=2)
                stream.write('\n')
            if process.status() != psutil.STATUS_STOPPED:
                raise RuntimeError('Native child was not created suspended')
        except BaseException:
            if process.status() != psutil.STATUS_STOPPED:
                process.suspend()
            print(json.dumps({'unrecorded_parked_native': identity}), file=sys.stderr, flush=True)
            raise
        launched = True
        print(json.dumps({'native_process_record': str(record), 'pid': child.pid,
                          'initially_suspended': True}), flush=True)
        return child

    ordinary.subprocess.Popen = start_paused
    sys.argv = [ordinary.__file__, *remaining]
    try:
        ordinary.main()
    finally:
        ordinary.subprocess.Popen = original_popen
        if any(sha(path) != digest for path, digest in sources.items()):
            raise RuntimeError('Persistent runner source changed')


if __name__ == '__main__':
    main()
