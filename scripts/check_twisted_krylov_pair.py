"""Compare full native trajectories differing only in FGMRES storage dimension.

The physical field, fixed-point, actual mass and strict linear gates remain the
existing ones. This internal regression does not establish Aphros agreement or
steady/spatial convergence.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from check_twisted_prefix_time_pair import read
from compare_twisted import vector, error
from run_twisted_solver import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous comparisons')
    roots = [a.candidate.resolve(strict=True), a.reference.resolve(strict=True)]
    configs = [json.loads((r/'case.json').read_text()) for r in roots]
    normalized = [{k: v for k, v in c.items() if k not in ('output', 'gpu_krylov_dimension')} for c in configs]
    if normalized[0] != normalized[1] or any(c.get('restart_checkpoint') or c.get('operator_only') for c in configs):
        raise ValueError('Only output and Krylov dimension may differ; use complete zero-start flow runs')
    for c in configs:
        if c['linear_backend'] != 'native_gpu' or c['gpu_preconditioner'] != 'native_amg' or c['fluid_solver'] != 'proj':
            raise ValueError('Expected native AMG projection')
        if not 2 <= c.get('gpu_krylov_dimension', 20) <= 64 or c.get('output_stride', 1) != 1:
            raise ValueError('Expected a valid dimension and every physical field dump')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    hashes, native_checks, builds = {}, [], []
    for index, (root, config) in enumerate(zip(roots, configs)):
        method = json.loads((root/'projection_method.json').read_text())
        if method.get('gpu_krylov_dimension', 20) != config.get('gpu_krylov_dimension', 20):
            raise ValueError('Actual Krylov dimension differs')
        runtime = json.loads((root/'run_manifest.json').read_text())
        build = Path(runtime['executable']).parent/'build_manifest.json'
        compiled = json.loads(build.read_text())
        if compiled['exit_code'] or not compiled['source_unchanged'] or compiled['executable_sha256']['simple_channel.exe'] != runtime['executable_sha256']:
            raise ValueError('Invalid compiled provenance')
        builds.append(compiled)
        hashes[str(build)] = sha(build)
        for name, expected in compiled['source_sha256'].items():
            snapshot = build.parent/'sources'/(name+'.txt')
            if sha(snapshot) != expected:
                raise ValueError('Actual compiled source snapshot changed')
            hashes[str(snapshot)] = expected
        check = a.output.with_name(a.output.stem+f'_native_{index}.json')
        subprocess.run([sys.executable, str(Path(__file__).with_name('check_twisted_prefix_native_run.py')),
                        '--run', str(root), '--output', str(check)], check=True, stdout=subprocess.DEVNULL)
        native = json.loads(check.read_text())
        hashes.update(native['source_sha256'])
        hashes.update(native['executed_input_sha256'])
        hashes[str(check)] = sha(check)
        hashes[str(root/'projection_method.json')] = sha(root/'projection_method.json')
        native_checks.append(native)
    rows = []
    for step in range(1, configs[0]['time_steps']+1):
        folders = [r/f'step_{step:04d}' for r in roots]
        cells = [read(f/'solution.csv') for f in folders]
        walls = [read(f/'walls.csv') for f in folders]
        fluxes = [read(f/'flux.csv') for f in folders]
        for data, columns in ((cells, ('id', 'x', 'y', 'z', 'volume')), (walls, ('face_id', 'x', 'y', 'z', 'area')), (fluxes, ('id',))):
            if not np.array_equal(vector(data[0], columns), vector(data[1], columns)):
                raise ValueError('Different grid geometry or ordering')
        v = cells[0]['volume']
        velocity = [vector(c, ('u', 'v', 'w')) for c in cells]
        pressure = [c['p']-np.average(c['p'], weights=v) for c in cells]
        cut = np.isin(cells[0]['id'], walls[0]['owner'])
        quantities = {'velocity': error(*velocity, v), 'pressure': error(*pressure, v),
                      'cut_cell_velocity': error(velocity[0][cut], velocity[1][cut], v[cut]),
                      'wall_shear': error(*[vector(w, ('tau_x', 'tau_y', 'tau_z')) for w in walls], walls[0]['area']),
                      'shared_face_flux': error(fluxes[0]['flux'], fluxes[1]['flux'], np.ones(len(fluxes[0])))}
        rows.append({'step': step, 'passed': all(q['relative_l2'] < 1e-6 for q in quantities.values()), **quantities})
        for f in folders:
            for name in ('solution.csv', 'walls.csv', 'flux.csv'):
                hashes[str(f/name)] = sha(f/name)
    result = {'scope': __doc__, 'passed': all(r['passed'] for r in rows) and all(c['passed'] for c in native_checks),
              'relative_l2_limit': 1e-6, 'dimensions': [c.get('gpu_krylov_dimension', 20) for c in configs],
              'complete_sequence_checked': True, 'steps': rows, 'native_checks': native_checks,
              'executables': [b['executable_sha256']['simple_channel.exe'] for b in builds],
              'source_sha256': hashes, 'checker_sha256': sha(Path(__file__)), 'goal_complete': False}
    if not all(sha(Path(path)) == value for path, value in hashes.items()):
        raise ValueError('Inputs changed during comparison')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'dimensions': result['dimensions'], 'steps': len(rows)}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
