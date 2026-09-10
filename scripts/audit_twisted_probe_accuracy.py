"""Separate smooth-field probe interpolation from cut-wall closure variation.

Use the unchanged 32/64-neighbor refinement sampler on an analytic no-slip
field, its exact traction, the previously verified wall-operator replay, and
actual completed fields. This diagnoses sampling; it does not accept a flow
solution, change a spatial-convergence gate, or replace an Aphros comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_twisted_refinement import probes, sample
from audit_twisted_wall_accuracy import analytic, tangential
from check_twisted_time_pair import read
from compare_twisted import error, vector
from run_twisted_solver import sha


def exact_traction(points, spec, speed, mu):
    """Map by the recorded wall audit's same-x, same-polar-angle convention."""
    _, _, (y, z, yp, zp) = analytic(points, spec, speed)
    radius = np.hypot(y, z)
    if np.any(radius == 0):
        raise ValueError('A wall point lies on the tube centerline')
    cosine, sine = y/radius, z/radius
    slope = yp*cosine+zp*sine
    normal = np.column_stack((-slope, cosine, sine))/np.sqrt(1+slope*slope)[:, None]
    surface = points.copy()
    surface[:, 1] += (spec['radius']-radius)*cosine
    surface[:, 2] += (spec['radius']-radius)*sine
    u, jacobian, _ = analytic(surface, spec, speed)
    traction = -2*mu*speed/spec['radius']*np.sqrt(1+slope*slope)[:, None]*np.column_stack((np.ones(len(points)), yp, zp))
    stress = mu*np.einsum('nij,nj->ni', jacobian+jacobian.transpose(0, 2, 1), normal)
    if np.max(abs(u)) > speed*1e-12 or np.max(abs(traction-tangential(stress, normal))) > np.max(abs(traction))*1e-12:
        raise ValueError('Analytic no-slip or symmetric-stress identity failed')
    return traction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-audit', type=Path, required=True)
    parser.add_argument('--refinement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous probe audits')
    sources = {}

    def keep(path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError('Recorded input changed: '+str(path))
        sources[str(path)] = value
        return path

    audit = json.loads(keep(args.wall_audit/'wall_accuracy.json').read_text())
    if not audit['diagnostic_replay_passed']:
        raise ValueError('Require verified original wall-operator replay')
    # The archived receipt binds the replay CSV bytes to the completed audit.
    receipt_path = Path(__file__).resolve().parents[1]/'validation/twisted/results/wall_closure_accuracy_checkpoint/receipt.json'
    receipt = json.loads(keep(receipt_path).read_text())
    snapshots = {}
    scripts = Path(__file__).resolve().parent
    for path, value in audit['source_sha256'].items():
        original = Path(path).resolve()
        if original.is_relative_to(scripts) and sha(original) != value:
            matches = []
            for historical_receipt in sorted(receipt_path.parent.parent.glob('*/receipt.json')):
                records = json.loads(historical_receipt.read_text())
                for saved in records.get('files', []) if isinstance(records, dict) else []:
                    if isinstance(saved, dict) and saved.get('source_sha256') == value and saved.get('gzip') is False and Path(saved['source']).resolve() == original:
                        matches.append((historical_receipt, saved))
            if not matches:
                raise ValueError('Historical audit source has no archived snapshot: '+str(original))
            historical_receipt, saved = matches[0]
            keep(historical_receipt)
            snapshot = keep(historical_receipt.parent/saved['path'], value)
            snapshots[str(original)] = {'snapshot': str(snapshot), 'receipt': str(historical_receipt), 'sha256': value}
        else:
            keep(original, value)
    retained = {str(Path(row['source']).resolve()): row['source_sha256']
                for row in receipt['retained_runtime_files']+receipt['files']}
    refinement = json.loads(keep(args.refinement).read_text())
    spec = audit['geometry_spec']
    speed = audit['centerline_axial_speed']
    surface, normals, weights, distances, points = probes(spec)
    exact_velocity = analytic(points, spec, speed)[0]
    rows, arrays = [], []
    for row in audit['grids']:
        step = Path(row['step'])
        cfg = json.loads(keep(step/'case.json').read_text())
        geometry = Path(cfg['embedded_geometry'])
        meta = geometry.with_suffix('.meta.json')
        metadata = json.loads(keep(meta if meta.exists() else geometry).read_text())
        if metadata['geometry_spec'] != spec:
            raise ValueError('Different analytic geometry')
        shift = np.array(metadata.get('reference_translation', [0, 0, 0]))
        wall_path = (args.wall_audit/f'wall_accuracy_n{row["ny"]}.csv').resolve()
        if str(wall_path) not in retained:
            raise ValueError('Wall replay CSV lacks completed archive provenance')
        replay = read(keep(wall_path, retained[str(wall_path)]))
        wall = read(keep(step/'walls.csv'))
        cells = read(keep(step/'solution.csv'))
        xc = vector(cells, 'xyz')-shift
        xw = vector(wall, 'xyz')-shift
        if not np.array_equal(replay['face_id'], wall['face_id']) or not np.array_equal(replay['area'], wall['area']) or not np.array_equal(vector(replay, 'xyz'), xw):
            raise ValueError('Replay wall ordering, areas or coordinates differ')
        xc[:, 0] %= spec['period']
        xw[:, 0] %= spec['period']
        mu = cfg['rho']*cfg['nu']
        exact_wall = exact_traction(xw, spec, speed, mu)
        exact_at_probes = exact_traction(surface, spec, speed, mu)
        stored_exact = vector(replay, ('exact_tau_x', 'exact_tau_y', 'exact_tau_z'))
        analytic_replay = float(np.max(abs(exact_wall-stored_exact))/np.max(abs(stored_exact)))
        if analytic_replay > 1e-12:
            raise ValueError('Analytic wall mapping differs from verified audit')
        names = ('exact_traction', 'discrete_manufactured_traction', 'physical_traction',
                 'boundary_offset', 'fit_derivative', 'geometry_transport')
        wall_values = np.column_stack((stored_exact,
            vector(replay, ('numeric_tau_x', 'numeric_tau_y', 'numeric_tau_z')),
            vector(wall, ('tau_x', 'tau_y', 'tau_z')),
            *[vector(replay, tuple(prefix+'_'+d for d in 'xyz')) for prefix in names[3:]]))
        w32, cond32 = sample(xw, wall_values, surface, spec['period'], 32, normals)
        w64, cond64 = sample(xw, wall_values, surface, spec['period'], 64, normals)
        wall_results = {}
        for j, name in enumerate(names[:3]):
            a, b = w32[:, 3*j:3*j+3], w64[:, 3*j:3*j+3]
            wall_results[name] = {'neighbor_sensitivity': error(a, b, weights)}
            if name != 'physical_traction':
                wall_results[name]['neighbors32_error_to_exact'] = error(a, exact_at_probes, weights)
                wall_results[name]['neighbors64_error_to_exact'] = error(b, exact_at_probes, weights)
        delta = w32-w64
        decomposition = delta[:, 3:6]-delta[:, :3]-delta[:, 9:12]-delta[:, 12:15]-delta[:, 15:18]
        decomposition_relative_linf = float(np.max(abs(decomposition))/np.max(abs(exact_at_probes)))
        if decomposition_relative_linf > 1e-12:
            raise ValueError('Linear sampling of the wall-error decomposition failed')
        scale = float(np.sqrt(np.average(np.sum(exact_at_probes**2, axis=1), weights=weights)))
        component_rms = {name: float(np.sqrt(np.average(np.sum(delta[:, j*3:j*3+3]**2, axis=1), weights=weights))/scale)
                         for j, name in enumerate(names) if j != 2}

        smooth_velocity = analytic(xc, spec, speed)[0]
        # A Cartesian y/z quadratic is also periodic in x. This exercises the
        # unchanged volume fit and true inward probe coordinates exactly.
        polynomial = (1+2*xc[:, 1]-3*xc[:, 2]+xc[:, 1]**2+xc[:, 1]*xc[:, 2]+2*xc[:, 2]**2)[:, None]
        target_poly = 1+2*points[:, 1]-3*points[:, 2]+points[:, 1]**2+points[:, 1]*points[:, 2]+2*points[:, 2]**2
        volume_values = np.column_stack((smooth_velocity, vector(cells, 'uvw'), polynomial))
        v32, vcond32 = sample(xc, volume_values, points, spec['period'], 32)
        v64, vcond64 = sample(xc, volume_values, points, spec['period'], 64)
        poly_error = max(float(np.max(abs(v[:, 6]-target_poly))) for v in (v32, v64))
        if poly_error > 1e-11:
            raise ValueError('Volume sampler failed exact quadratic reproduction')
        velocity_results = {}
        for j, distance in enumerate(distances):
            sl = slice(j, None, len(distances))
            velocity_results[str(distance)] = {
                'smooth_neighbors32_error_to_exact': error(v32[sl, :3], exact_velocity[sl], weights),
                'smooth_neighbors64_error_to_exact': error(v64[sl, :3], exact_velocity[sl], weights),
                'smooth_neighbor_sensitivity': error(v32[sl, :3], v64[sl, :3], weights),
                'physical_neighbor_sensitivity': error(v32[sl, 3:6], v64[sl, 3:6], weights)}
        regression = None
        for index, name in enumerate(('coarse', 'fine')):
            if step.resolve() == Path(refinement[name]).resolve():
                diffs = [abs(wall_results['physical_traction']['neighbor_sensitivity']['relative_l2']-
                             refinement['sampling_sensitivity'][index]['wall_shear']['relative_l2'])]
                diffs += [abs(velocity_results[str(d)]['physical_neighbor_sensitivity']['relative_l2']-
                              refinement['sampling_sensitivity_by_distance_m'][index][str(d)]['velocity']['relative_l2']) for d in distances]
                regression = float(max(diffs))
                if regression > 1e-12:
                    raise ValueError('Existing physical sampling metrics changed')
        result = {'ny': row['ny'], 'step': str(step), 'physical_flow_is_steady': row['physical_flow_is_steady'],
                  'wall': wall_results, 'velocity_by_distance_m': velocity_results,
                  'wall_sensitivity_components_scaled_rms': component_rms,
                  'wall_sensitivity_decomposition_relative_linf': decomposition_relative_linf,
                  'analytic_traction_replay_relative_linf': analytic_replay,
                  'quadratic_volume_reproduction_absolute_linf': poly_error,
                  'existing_refinement_sensitivity_maximum_difference': regression,
                  'fit_condition_numbers': {'wall': max(cond32, cond64), 'volume': max(vcond32, vcond64)}}
        rows.append(result)
        arrays.append((row['ny'], np.column_stack((surface, exact_at_probes, w32, w64))))
        print(json.dumps({'ny': row['ny'], 'wall_smooth_sensitivity': wall_results['exact_traction']['neighbor_sensitivity']['relative_l2'],
                          'wall_discrete_sensitivity': wall_results['discrete_manufactured_traction']['neighbor_sensitivity']['relative_l2'],
                          'wall_physical_sensitivity': wall_results['physical_traction']['neighbor_sensitivity']['relative_l2']}), flush=True)
    for name in ('analyze_twisted_refinement.py', 'audit_twisted_wall_accuracy.py', 'check_twisted_time_pair.py',
                 'compare_twisted.py', 'run_twisted_solver.py', Path(__file__).name):
        keep(Path(__file__).with_name(name))
    if any(sha(Path(path)) != value for path, value in sources.items()):
        raise ValueError('Probe audit sources changed during execution')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': __doc__, 'diagnostic_checks_passed': True, 'goal_complete': False,
              'geometry_spec': spec, 'centerline_axial_speed': speed, 'grids': rows,
              'limitations': ['The smooth manufactured velocity is not the constant-force physical solution.',
                             'The wall mapping is the same-x, same-polar-angle convention of the existing wall audit.',
                             'Neighbor sensitivity is not an error bound; component RMS values need not add.',
                             'Physical 128-grid samples are transient and are not compared as a steady fine-grid solution.',
                             'No sampler, physical field, solver or acceptance threshold is changed.'],
              'historical_source_snapshots': snapshots, 'source_sha256': sources}
    for ny, values in arrays:
        header = ['x', 'y', 'z']+['exact_'+d for d in 'xyz']
        header += [f'{n}_{name}_{d}' for n in (32, 64) for name in names for d in 'xyz']
        np.savetxt(args.output/f'wall_probes_n{ny}.csv', values, delimiter=',', header=','.join(header), comments='')
    (args.output/'probe_accuracy.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
