"""Compare immutable raw projection-iteration dumps with bounded memory.

This verifies a recorded intermediate update, not convergence of either run.
The following dump or a flushed post-dump acceleration record is the barrier.
"""
import argparse
import csv
import hashlib
import itertools
import io
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pressure-operator-pair', action='store_true',
                        help='Compare full versus compact GPU pressure solves with all outer settings fixed')
    args = parser.parse_args()
    if args.output.exists() or args.step < 1 or args.iteration < 1:
        raise ValueError('Choose a fresh output and positive step/iteration')
    roots = [args.candidate.resolve(), args.reference.resolve()]
    configs = [json.loads((r/'case.json').read_text()) for r in roots]
    ignored = ('output', 'gpu_pressure_operator') if args.pressure_operator_pair else ('output',)
    normalized = [{k: v for k, v in c.items() if k not in ignored} for c in configs]
    if normalized[0] != normalized[1]:
        raise ValueError('Different solver configurations')
    if args.pressure_operator_pair:
        if ([c.get('gpu_pressure_operator', 'compact') for c in configs] != ['full', 'compact'] or
                any(c.get('linear_backend') != 'native_gpu' or c.get('gpu_preconditioner') != 'native_amg' or
                    c.get('gpu_pressure_gauge') != 'mean_zero' for c in configs)):
            raise ValueError('Expected full versus compact native-AMG pressure operators')
    paths, hashes, runtimes, barriers = [], {}, [], []
    for root in roots:
        if args.pressure_operator_pair:
            method = json.loads((root/'projection_method.json').read_text())
            expected = configs[len(paths)]['gpu_pressure_operator'] if 'gpu_pressure_operator' in configs[len(paths)] else 'compact'
            if method.get('pressure_operator', 'compact') != expected:
                raise ValueError('Actual pressure operator differs from the configuration')
            hashes[str(root/'projection_method.json')] = sha(root/'projection_method.json')
        step = root/f'step_{args.step:04d}'
        if (step/f'iter_{args.iteration+1}').is_dir():
            barriers.append({'kind': 'following iteration dump directory', 'path': str(step/f'iter_{args.iteration+1}')})
        else:
            data = (step/'acceleration.csv').read_bytes()
            data = data[:data.rfind(b'\n')+1]
            rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
            if not any(int(r['after_iteration']) >= args.iteration for r in rows):
                raise ValueError('No flushed post-dump acceleration record')
            barriers.append({'kind': 'acceleration record flushed after the raw dump',
                             'source': str(step/'acceleration.csv'),
                             'snapshot_sha256': hashlib.sha256(data).hexdigest(),
                             'snapshot_utf8': data.decode('utf-8')})
        paths.append(step/f'iter_{args.iteration}')
        runtime = json.loads((root/'run_manifest.json').read_text())
        if sha(Path(runtime['executable'])) != runtime['executable_sha256']:
            raise ValueError('Executable changed')
        runtimes.append(runtime)
        for p in (root/'case.json', root/'run_manifest.json', root/'mesh_cells.csv',
                  paths[-1]/'cells.csv', paths[-1]/'faces.csv'):
            hashes[str(p)] = sha(p)
    if hashes[str(roots[0]/'mesh_cells.csv')] != hashes[str(roots[1]/'mesh_cells.csv')]:
        raise ValueError('Different cell geometry or ordering')
    geometry_path = Path(configs[0]['embedded_geometry']).resolve()
    hashes[str(geometry_path)] = sha(geometry_path)
    extent_y = json.loads(geometry_path.read_text())['extent'][1]
    result = {};volume_chunks = [];speed = [0., 0.];nets = []
    quantities = [('cells.csv', ['id', 'x', 'y', 'z'], {
        'velocity': ['u', 'v', 'w'], 'pressure': ['p'],
        'advected_velocity': ['u_adv', 'v_adv', 'w_adv'],
        'diffused_velocity': ['u_diff', 'v_diff', 'w_diff'],
        'acceleration': ['source_x', 'source_y', 'source_z']}),
        ('faces.csv', ['id', 'owner', 'neighbor', 'axis', 'area'],
         {'shared_flux': ['flux'], 'predicted_flux': ['predicted_flux']})]
    for name, geometry, groups in quantities:
        if name == 'faces.csv':
            volumes = np.concatenate(volume_chunks)
            nets = [np.zeros(len(volumes)), np.zeros(len(volumes))]
        accum = {key: [0., 0., 0.] for key in groups}
        count = 0
        with (paths[0]/name).open() as a, (paths[1]/name).open() as b, (roots[0]/'mesh_cells.csv').open() as mesh:
            readers = [csv.DictReader(a), csv.DictReader(b)]
            weights = csv.DictReader(mesh)
            while True:
                chunks = [list(itertools.islice(r, 8192)) for r in readers]
                if len(chunks[0]) != len(chunks[1]):
                    raise ValueError('Different row counts')
                if not chunks[0]:
                    break
                size = len(chunks[0]);count += size
                coords = [np.array([[float(r[k]) for k in geometry] for r in c]) for c in chunks]
                if not np.array_equal(*coords):
                    raise ValueError('Different intermediate geometry or ordering')
                if name == 'cells.csv':
                    cells = list(itertools.islice(weights, size))
                    if len(cells) != size or any(int(r['id']) != int(c['id']) for r, c in zip(chunks[0], cells)):
                        raise ValueError('Cell-volume indexing mismatch')
                    weight = np.array([float(r['volume']) for r in cells])
                    volume_chunks.append(weight)
                else:
                    weight = np.ones(size)
                if not np.isfinite(weight).all() or np.any(weight <= 0):
                    raise ValueError('Invalid comparison weights')
                for key, columns in groups.items():
                    values = [np.array([[float(r[k]) for k in columns] for r in c]) for c in chunks]
                    if not all(np.isfinite(v).all() for v in values):
                        raise ValueError('Nonfinite intermediate state')
                    if key == 'velocity':
                        for i, value in enumerate(values):
                            speed[i] = max(speed[i], float(np.linalg.norm(value, axis=1).max()))
                    if key == 'shared_flux':
                        owner, neighbor = coords[0][:, 1].astype(int), coords[0][:, 2].astype(int)
                        inner = neighbor >= 0
                        for net, value in zip(nets, values):
                            if np.any(value[~inner] != 0):
                                raise ValueError('Nonzero impermeable-wall flux')
                            np.add.at(net, owner, value[:, 0])
                            np.add.at(net, neighbor[inner], -value[inner, 0])
                    delta = values[0]-values[1]
                    accum[key][0] += np.sum(weight[:, None]*delta**2)
                    accum[key][1] += np.sum(weight[:, None]*values[1]**2)
                    accum[key][2] = max(accum[key][2], float(abs(delta).max()))
        if not count:
            raise ValueError('Empty intermediate dump')
        for key, (difference, reference, maximum) in accum.items():
            result[key] = {'relative_l2': float(np.sqrt(difference/max(reference, 1e-300))),
                           'absolute_linf': maximum, 'rows': count}
    if any(sha(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('An input changed during comparison')
    if any(s <= 0 for s in speed):
        raise ValueError('Zero reference speed for continuity normalization')
    mass = [float(np.max(abs(net)/volumes)/(s/extent_y)) for net, s in zip(nets, speed)]
    passed = all(v['relative_l2'] < 1e-10 for v in result.values()) and all(v < 1e-7 for v in mass)
    report = {'passed': passed, 'scope': 'One completed raw outer iteration; neither physical time-step convergence nor steady/grid accuracy is certified',
              'pressure_operator_pair': args.pressure_operator_pair,
              'step': args.step, 'iteration': args.iteration, 'relative_l2_limit': 1e-10,
              'quantities': result, 'runtime_manifests': runtimes, 'write_completion_barriers': barriers, 'source_sha256': hashes,
              'independent_divergence_relative_linf_candidate_reference': mass,
              'checker_sha256': sha(Path(__file__))}
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': passed, 'quantities': result, 'independent_divergence_relative_linf': mass}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
