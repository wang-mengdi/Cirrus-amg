"""Verify the minimal driver against a completed full Aphros tube run.

Checks prescribed inputs, discrete geometry, the whole outer-iteration error
history, physical-step diagnostics, all available SIMPLE intermediate dumps,
and final velocity/pressure/wall shear/shared face flux. This is driver
equivalence evidence, not a mesh-convergence or continuum-accuracy test.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from compare_twisted import read


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--build-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [args.reference.resolve(), args.candidate.resolve()]
    configs = [json.loads((p/'case_manifest.json').read_text()) for p in roots]
    if configs[0] != configs[1] or sha(roots[0]/'a.conf') != sha(roots[1]/'a.conf'):
        raise ValueError('Driver audit requires identical input cases')
    runtimes = [json.loads((p/'run_manifest.json').read_text(encoding='utf-8-sig')) for p in roots]
    build = json.loads(args.build_manifest.read_text(encoding='utf-8-sig'))
    if build['exit_code'] or build['executable_sha256'] != runtimes[1]['executable_sha256']:
        raise ValueError('Candidate differs from the recorded driver build')
    for field in ('source', 'library'):
        if sha(Path(build[field])) != build[field+'_sha256']:
            raise ValueError(f'Driver build input has changed: {field}')
    logs, hashes = [], {str(args.build_manifest.resolve()): sha(args.build_manifest)}
    for root, runtime in zip(roots, runtimes):
        if runtime['config_sha256'] != sha(root/'a.conf'):
            raise ValueError('Executed configuration differs from audited input')
        if sha(Path(runtime['executable'])) != runtime['executable_sha256']:
            raise ValueError('Executed binary has changed')
        completion = json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))
        log = (root/'run.log').read_text()
        if completion['exit_code'] or 'End of simulation' not in log:
            raise ValueError('Incomplete flow run')
        if runtime['environment'].get('APHROS_TWISTED_GEOMETRY_ONLY') is not None:
            raise ValueError('Geometry-only execution is not a flow comparison')
        if configs[0]['convection'] and runtime['environment']['APHROS_TWISTED_FIX_FLUX_HALO'] != '1':
            raise ValueError('Both convection cases must use the documented halo repair')
        logs.append(log)
        for name in ('a.conf', 'case_manifest.json', 'run_manifest.json', 'run_completion.json', 'run.log'):
            hashes[str(root/name)] = sha(root/name)
    checks, differences = {}, {}
    for name in ('cells', 'faces', 'walls', 'polygons'):
        paths = [p/f'tube_b0_geometry_{name}.csv' for p in roots]
        checks['identical_geometry_'+name] = sha(paths[0]) == sha(paths[1])
        for path in paths:
            hashes[str(path)] = sha(path)
    histories = [np.array(re.findall(r'iter=(\d+), diff=([\d.eE+\-]+)', log), dtype=float) for log in logs]
    if any(h.ndim != 2 or not len(h) for h in histories):
        raise ValueError('Missing outer-iteration history')
    checks['same_iteration_sequence'] = histories[0].shape == histories[1].shape and np.array_equal(histories[0][:, 0], histories[1][:, 0])
    if checks['same_iteration_sequence']:
        checks['same_iteration_changes'] = bool(np.allclose(histories[0][:, 1], histories[1][:, 1], rtol=1e-10, atol=1e-13))
        differences['iteration_change_absolute_max'] = float(np.max(np.abs(histories[0][:, 1]-histories[1][:, 1])))
    else:
        checks['same_iteration_changes'] = False
    steps = [read(p/'tube_b0_time.csv') for p in roots]
    checks['complete_physical_steps'] = all(len(s) == configs[0]['time_steps'] and np.allclose(
        s['time'], np.arange(1, len(s)+1)*configs[0]['time_step'], rtol=1e-13, atol=0) and np.all(
        s['time_step'] == configs[0]['time_step']) for s in steps)
    reference_dumps = {p.name for p in roots[0].glob('simple_*_b0_*.csv')}
    candidate_dumps = {p.name for p in roots[1].glob('simple_*_b0_*.csv')}
    checks['same_intermediate_dump_set'] = reference_dumps == candidate_dumps
    if not {'simple_final_b0_cells.csv', 'simple_final_b0_faces.csv'} <= reference_dumps:
        raise ValueError('Missing final field dumps')
    for filename in sorted(reference_dumps & candidate_dumps | {'tube_final_b0_walls.csv', 'tube_b0_time.csv'}):
        paths = [p/filename for p in roots]
        arrays = [read(p) for p in paths]
        if arrays[0].dtype.names != arrays[1].dtype.names or arrays[0].shape != arrays[1].shape:
            raise ValueError(f'Different dump layout: {filename}')
        field_errors = {}
        for name in arrays[0].dtype.names:
            a, b = arrays[0][name], arrays[1][name]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f'Nonfinite dump: {filename}:{name}')
            delta = float(np.max(np.abs(b-a)))
            # Same gauge is fixed by the same pressure system in this audit.
            # Tight absolute floor also checks small face flux and RHS fields.
            limit = 1e-15 + 1e-10*float(np.max(np.abs(a)))
            checks[filename+':'+name] = delta <= limit
            field_errors[name] = {'absolute_max': delta, 'limit': limit}
        differences[filename] = field_errors
        for path in paths:
            hashes[str(path)] = sha(path)
    result = {'passed': all(checks.values()),
              'scope': 'Full versus minimal Aphros driver on identical prescribed inputs; same Embed/Simple library algorithms, with disclosed baseline repairs',
              'checks': checks, 'differences': differences, 'runtime': runtimes,
              'driver_build': build, 'source_sha256': hashes,
              'outer_iterations': [len(h) for h in histories], 'physical_steps': configs[0]['time_steps']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'outer_iterations': result['outer_iterations'],
                      'physical_steps': result['physical_steps'], 'failed_checks': [k for k,v in checks.items() if not v]}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
