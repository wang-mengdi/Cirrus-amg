"""Compare MGS2 and batched CGS2 on identical native pressure/viscosity audits.

Alternating fresh processes retain all inputs, known solutions and true residuals.
Elapsed times are observations under the current concurrent machine load, not an
exclusive GPU benchmark or a claim of whole-flow acceleration.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from compare_twisted import read
from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    exe, base_path, out = args.exe.resolve(), args.config.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Preserve earlier benchmarks')
    base = json.loads(base_path.read_text(encoding='utf-8-sig'))
    if base.get('gpu_preconditioner') != 'native_amg':
        raise ValueError('Require native AMG/FGMRES')
    build_path = exe.parent/'build_manifest.json'
    build = json.loads(build_path.read_text())
    if build['exit_code'] or not build['source_unchanged'] or sha(exe) != build['executable_sha256'][exe.name]:
        raise ValueError('Unverified audit executable')
    sources = {str(p): sha(p) for p in (exe, base_path, build_path, Path(__file__).resolve())}
    geometry = Path(base['embedded_geometry'])
    geometry = geometry if geometry.is_absolute() else repo/geometry
    metadata = geometry.with_suffix('.meta.json')
    spec = json.loads((metadata if metadata.exists() else geometry).read_text(encoding='utf-8-sig'))
    geometry_paths = [geometry]
    if metadata.exists():
        geometry_paths.append(metadata)
    if spec['format'] == 'aphros_cut_geometry_v2':
        if spec != json.loads(geometry.read_text(encoding='utf-8-sig')):
            raise ValueError('Packed geometry sidecar differs from actual input')
        for table in spec['tables'].values():
            path = (geometry.parent/table['file']).resolve(strict=True)
            if sha(path) != table['sha256']:
                raise ValueError('Packed geometry payload changed')
            geometry_paths.append(path)
    sources.update({str(p.resolve()): sha(p) for p in geometry_paths})
    environment = dict(os.environ, OMP_NUM_THREADS='2', OMP_WAIT_POLICY='PASSIVE')
    out.mkdir(parents=True, exist_ok=False)
    reports, fields = [], []
    for number, method in enumerate(('mgs2', 'cgs2', 'cgs2', 'mgs2'), 1):
        config = dict(base, gpu_orthogonalization=method, audit_test_failed_solve_reuse=True)
        config_path = out/f'config_{number}_{method}.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        config_hash = sha(config_path)
        run = out/f'run_{number}_{method}'
        command = [str(exe), str(config_path), str(run)]
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(json.dumps({'started': number, 'method': method}), flush=True)
        begin = time.perf_counter()
        with (out/f'run_{number}_{method}.log').open('w') as stream:
            code = subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
        entry = {'method': method, 'command': command, 'exit_code': code, 'started_utc': started,
                 'process_seconds': time.perf_counter()-begin, 'config_sha256': config_hash,
                 'config_unchanged': sha(config_path) == config_hash, 'executable_unchanged': sha(exe) == sources[str(exe)]}
        reports.append(entry)
        if code or not entry['config_unchanged'] or not entry['executable_unchanged']:
            (out/'failed_run.json').write_text(json.dumps(entry, indent=2)+'\n')
            raise ValueError('Audit process failed; preserve its actual output')
        report = json.loads((run/'audit.json').read_text())
        if not report['passed'] or report['gpu_orthogonalization'] != method:
            raise ValueError('Audit did not pass using the selected method')
        for linear in report['linear_tests']:
            if linear['gpu_true_relative_residual'] > 1e-13 or not linear['forced_failure_and_reuse']['reuse_passed']:
                raise ValueError('Original true-residual or failed-solve recovery check failed')
        entry['linear_tests'] = report['linear_tests']
        fields.append({kind: read(run/f'solve_{kind}.csv') for kind in ('pressure', 'diffusion')})
        for path in run.iterdir():
            if path.is_file():
                sources[str(path)] = sha(path)
        sources[str(config_path)] = config_hash
        sources[str(out/f'run_{number}_{method}.log')] = sha(out/f'run_{number}_{method}.log')
        print(json.dumps({'completed': number, 'method': method,
                          'linear_seconds': {v['operator']: v['seconds'] for v in report['linear_tests']}}), flush=True)
    comparisons = []
    for i in range(1, len(fields)):
        for kind in ('pressure', 'diffusion'):
            a, b = fields[0][kind], fields[i][kind]
            if not np.array_equal(a[['id', 'known', 'rhs']], b[['id', 'known', 'rhs']]):
                raise ValueError('Changed manufactured linear system')
            difference = float(np.linalg.norm(a['gpu_solution']-b['gpu_solution'])/np.linalg.norm(a['known']))
            if difference >= 1e-8:
                raise ValueError('Orthogonalization changed the recovered known solution')
            comparisons.append({'run': i+1, 'operator': kind, 'solution_relative_l2_difference': difference})
    timing = {}
    for kind in ('pressure', 'implicit_diffusion'):
        values = {method: [next(v['seconds'] for v in r['linear_tests'] if v['operator'] == kind)
                           for r in reports if r['method'] == method] for method in ('mgs2', 'cgs2')}
        timing[kind] = {'seconds': values, 'observed_median_mgs2_over_cgs2': float(np.median(values['mgs2'])/np.median(values['cgs2']))}
    if any(sha(Path(path)) != value for path, value in sources.items()):
        raise ValueError('Benchmark inputs or outputs changed')
    final = {'passed': True, 'scope': __doc__, 'runs': reports, 'solution_comparisons': comparisons,
             'timing': timing, 'source_sha256': sources, 'goal_complete': False}
    (out/'benchmark.json').write_text(json.dumps(final, indent=2)+'\n')
    print(json.dumps({'passed': True, 'timing': timing}), flush=True)


if __name__ == '__main__':
    main()
