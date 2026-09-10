"""Build an isolated CUDA AMGCL C-ABI library for optional Aphros validation.

This compiles no Aphros code and changes no running reference. The produced
library still needs actual operator and flow comparisons before adoption.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amgcl', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--vcvars', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.drive.lower() != 'd:':
        raise ValueError('Use D drive for this experimental build')
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('cuda_amg_bridge.cu')
    cuda, amgcl, vcvars = args.cuda.resolve(), args.amgcl.resolve(), args.vcvars.resolve()
    nvcc = cuda/'bin/nvcc.exe'
    inputs = {str(p): sha(p) for p in [Path(__file__).resolve(), source, nvcc, vcvars]}
    headers = sorted((amgcl/'amgcl').rglob('*.hpp'))
    if not headers:
        raise ValueError('Missing AMGCL headers')
    inputs.update({str(p): sha(p) for p in headers})
    for name in ['include/cuda_runtime_api.h', 'include/cusparse.h',
                 'lib/x64/cudart.lib', 'lib/x64/cusparse.lib']:
        inputs[str(cuda/name)] = sha(cuda/name)
    compiled = out/'cuda_amg_bridge.cu'
    shutil.copyfile(source, compiled)
    shutil.copyfile(__file__, out/'build_script.py.txt')
    library = out/'twisted_cuda_amg.dll'
    command = [str(nvcc), '-std=c++17', '-O2', '-arch=sm_86',
               '--expt-relaxed-constexpr', '--allow-unsupported-compiler',
               '-Xcompiler', '/MD,/EHsc,/openmp', '--cudart', 'shared', '-shared',
               '-I'+str(amgcl), str(compiled), '-lcusparse', '-o', str(library)]
    # No user input is interpreted as a batch command. Require literal paths
    # without batch expansion characters before producing this fixed script.
    for item in [*command, str(vcvars), str(out)]:
        if any(value in item for value in ['%', '!', '\r', '\n', '"', '&', '|', '<', '>']):
            raise ValueError('Unsupported batch metacharacter in build argument')
    batch = out/'build.cmd'
    quote = lambda value: '"'+value+'"'
    batch.write_text('@echo off\ncall '+quote(str(vcvars))+'\n'
                     'if errorlevel 1 exit /b %errorlevel%\n'
                     +' '.join(map(quote, command))+'\n'
                     'exit /b %errorlevel%\n')
    now = lambda: datetime.now(timezone.utc).isoformat()
    manifest = {'scope': __doc__, 'started_utc': now(), 'command': command,
                'cuda': str(cuda), 'amgcl': str(amgcl), 'source_sha256': inputs,
                'compiled_source_sha256': sha(compiled), 'batch_sha256': sha(batch),
                'minimum_launch_available_bytes': 5*2**30,
                'minimum_build_available_bytes': int(2.5*2**30),
                'passed': False, 'stopped_only_build': False}
    report = out/'build_manifest.json'
    report.write_text(json.dumps(manifest, indent=2)+'\n')
    error = None
    try:
        deadline = time.monotonic()+1200
        while psutil.virtual_memory().available < manifest['minimum_launch_available_bytes']:
            (out/'status.json').write_text(json.dumps({'utc': now(), 'phase': 'waiting_memory',
                'available_bytes': psutil.virtual_memory().available})+'\n')
            if time.monotonic() > deadline:
                raise RuntimeError('Compiler not launched: insufficient headroom')
            time.sleep(2)
        manifest['launch_available_bytes'] = psutil.virtual_memory().available
        with (out/'build.log').open('w') as log:
            process = subprocess.Popen(['cmd.exe', '/d', '/c', str(batch)], cwd=out,
                env=dict(os.environ, OMP_NUM_THREADS='2'), stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            owner = psutil.Process(process.pid)
            birth = owner.create_time()
            manifest.update(pid=process.pid, creation_time=birth)
            report.write_text(json.dumps(manifest, indent=2)+'\n')
            peak = 0
            minimum = psutil.virtual_memory().available
            while process.poll() is None:
                try:
                    owner = psutil.Process(process.pid)
                    assert owner.create_time() == birth
                    children = owner.children(recursive=True)
                    identities = [(p.pid, p.create_time()) for p in children]
                    rss = owner.memory_info().rss
                    for child in children:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, rss)
                    available = psutil.virtual_memory().available
                    minimum = min(minimum, available)
                    if available < manifest['minimum_build_available_bytes']:
                        for pid, stamp in reversed(identities):
                            try:
                                child = psutil.Process(pid)
                                if child.create_time() == stamp:
                                    child.terminate()
                            except psutil.NoSuchProcess:
                                pass
                        if owner.create_time() == birth:
                            owner.terminate()
                        manifest['stopped_only_build'] = True
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.25)
            manifest.update(exit_code=process.wait(), maximum_tree_rss=peak,
                            minimum_observed_available_bytes=minimum)
        if manifest['exit_code'] or manifest['stopped_only_build']:
            raise RuntimeError('CUDA bridge build did not complete; inspect build.log')
        manifest['library_sha256'] = sha(library)
        manifest['passed'] = True
    except Exception as exception:
        error = repr(exception)
        raise
    finally:
        unchanged = all(sha(p) == value for p, value in inputs.items())
        manifest.update(completed_utc=now(), inputs_unchanged=unchanged, error=error)
        manifest['passed'] = manifest['passed'] and unchanged
        report.write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({k: v for k, v in manifest.items() if k != 'source_sha256'}), flush=True)
        if not unchanged:
            raise RuntimeError('Build input changed')


if __name__ == '__main__':
    main()
