"""Diagnose pressure sensitivity using completed uniform-grid projection fields.

Replays the saved cell-pressure gradient and the exact collocated face-flux
identity. This is a diagnostic, not a temporal/spatial convergence certificate
or proof that one term alone causes the observed time-step sensitivity.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from analyze_twisted_refinement import completed_trajectory
from compare_twisted import error, read, vector
from run_twisted_solver import sha


def rms(a, weight):
    return float(np.sqrt(np.average(np.sum(a*a, axis=1) if a.ndim == 2 else a*a, weights=weight)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--operators', type=Path, required=True, help='Completed operator-only run root')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous pressure diagnostics')
    if len(args.runs) < 2:
        raise ValueError('At least two completed trajectories are required')
    sources = {str(Path(__file__).resolve()): sha(Path(__file__))}

    def keep(path):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return path

    def js(path):
        return json.loads(keep(path).read_text(encoding='utf-8-sig'))

    def data(path):
        return read(keep(path))

    for name in ('analyze_twisted_refinement.py', 'compare_twisted.py', 'run_twisted_solver.py', 'check_twisted_mass.py'):
        keep(Path(__file__).with_name(name))

    # This capture's convection/diffusion settings do not enter I or Gc.
    # Confirm actual geometry and replay Gc against every final iteration dump.
    capture = args.operators.resolve()
    operator_config = js(capture/'case.json')
    if not operator_config.get('operator_only'):
        raise ValueError('Require an operator-only capture')
    runtime = js(capture/'run_manifest.json')
    done = js(capture/'run_completion.json')
    if done['exit_code'] or not all(done[k] for k in ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
        raise ValueError('Operator capture failed or inputs changed')
    for path, value in {runtime['executable']: runtime['executable_sha256'],
                        runtime['config']: runtime['config_sha256'], **runtime['geometry_input_sha256']}.items():
        if sha(keep(path)) != value:
            raise ValueError('Operator capture input changed: '+path)
    op = capture/'operators'
    manifest = js(op/'manifest.json')
    mesh, faces = data(op/'mesh_cells.csv'), data(op/'mesh_faces.csv')
    nc, nf = len(mesh), len(faces)
    inside = faces['neighbor'] >= 0
    ids = np.flatnonzero(inside)
    owner, neighbor = faces['owner'][inside].astype(int), faces['neighbor'][inside].astype(int)
    axis, area, distance = faces['axis'][inside].astype(int), faces['area'][inside], faces['distance'][inside]
    if not np.all(mesh['h'] == mesh['h'][0]) or not np.all(faces['sign'][inside] == 1):
        raise ValueError('Diagnostic currently requires a uniform Cartesian cut grid')

    def matrix(name, shape):
        a = data(op/(name+'.csv'))
        spec = manifest['shapes'][name]
        if shape != (spec['rows'], spec['columns']) or not np.all(np.isfinite(a['value'])):
            raise ValueError('Invalid captured operator: '+name)
        return coo_matrix((a['value'], (a['row'].astype(int), a['column'].astype(int))), shape=shape).tocsr()

    gradient = [matrix('cell_gradient_'+str(d), (nc, nc)) for d in range(3)]
    interpolation = matrix('interpolation', (nf, nc))
    if np.max(abs((interpolation@np.ones(nc))[inside]-1)) > 1e-12:
        raise ValueError('This identity assumes constant material interpolation on all open faces')

    states, records, trajectories = [], [], []
    config0 = None
    for root in args.runs:
        root = root.resolve()
        config = js(root/'case.json')
        step = root/f'step_{config["time_steps"]:04d}'
        step_config, metrics = js(step/'case.json'), js(step/'metrics.json')
        trajectory = completed_trajectory(step, step_config)
        sources.update(trajectory['source_sha256'])
        trajectories.append(trajectory)
        run = js(root/'run_manifest.json')
        if config.get('fluid_solver') != 'proj' or config.get('adaptive') or metrics['coarse_fine_faces']:
            raise ValueError('Require uniform projection flows')
        if not metrics['converged'] or max(metrics['steady_momentum_relative_l2'], metrics['temporal_acceleration_relative_l2']) >= 1e-8:
            raise ValueError('Require physically steady completed fields')
        if config0 is None:
            config0 = config
        for key in ('rho', 'nu', 'force', 'ny', 'adaptive', 'embedded_geometry', 'periodic_x', 'periodic_z',
                    'fluid_solver', 'convection', 'convection_scheme', 'momentum_mode', 'wall_reconstruction'):
            if config.get(key) != config0.get(key):
                raise ValueError('Different physical/spatial problem: '+key)
        a, flux, local_mesh, local_faces, wall, sections = [data(step/name) for name in
            ('solution.csv', 'flux.csv', 'mesh_cells.csv', 'mesh_faces.csv', 'walls.csv', 'sections.csv')]
        if not np.array_equal(local_mesh, mesh) or not np.array_equal(local_faces, faces):
            raise ValueError('Captured and executed operator geometry differs')
        if not np.array_equal(vector(a, ['id', 'x', 'y', 'z', 'h', 'volume']), vector(mesh, ['id', 'x', 'y', 'z', 'h', 'volume'])):
            raise ValueError('Solution geometry differs')
        if not np.array_equal(flux['id'], faces['id']):
            raise ValueError('Flux ordering differs')
        final = data(step/f'iter_{metrics["iterations"]}/cells.csv')
        if not np.array_equal(vector(final, ['id', 'u', 'v', 'w', 'p']), vector(a, ['id', 'u', 'v', 'w', 'p'])):
            raise ValueError('Final iteration dump and final solution differ')
        volume, u = a['volume'], vector(a, 'uvw')
        p = a['p']-np.average(a['p'], weights=volume)
        gp = np.column_stack([g@p for g in gradient])
        source = vector(final, ['source_x', 'source_y', 'source_z'])
        gradient_replay = float(np.max(abs(gp/config['rho']-(np.array(config['force'])-source))))
        # source was formed before the final inner update; steady iteration
        # convergence bounds this comparison, so it is not a bitwise assertion.
        if gradient_replay > 1e-9*max(1., np.linalg.norm(config['force'])):
            raise ValueError('Captured Gc does not reproduce the actual pressure source')
        q, vi = flux['flux'][inside], (interpolation@u)[ids, axis]
        igp, gfp = (interpolation@gp)[ids, axis], (p[neighbor]-p[owner])/distance
        correction = config['time_step']/config['rho']*area*(igp-gfp)
        replay = area*vi+correction
        flux_identity = float(np.linalg.norm(replay-q)/np.linalg.norm(q))
        if flux_identity >= 1e-12:
            raise ValueError('Saved flux does not satisfy the reconstructed projection identity')
        cut = np.isin(a['id'], wall['owner'])

        def net(face_flux):
            result = np.zeros(nc)
            np.add.at(result, owner, face_flux)
            np.add.at(result, neighbor, -face_flux)
            return result

        raw_net, actual_net = net(area*vi), net(q)
        through = float(np.mean(sections['volume_flux']))
        speed = float(np.linalg.norm(u, axis=1).max())
        # Use the prescribed domain height, not the fluid bounding box.
        geometry = Path(config['embedded_geometry'])
        metadata_path = geometry.with_suffix('.meta.json')
        metadata = js(metadata_path if metadata_path.exists() else geometry)
        extent_y = metadata['extent'][1]
        balance = {'divergence_relative_linf': float(np.max(abs(actual_net)/volume)/(speed/extent_y)),
                   'absolute_cell_flux_over_throughflow': float(abs(actual_net).sum()/abs(through)),
                   'section_flux_relative_spread': float(np.ptp(sections['volume_flux'])/abs(through)),
                   'wall_flux_linf': float(np.max(abs(flux['flux'][~inside])))}
        if balance['divergence_relative_linf'] >= 1e-7 or balance['absolute_cell_flux_over_throughflow'] >= 1e-8 or balance['section_flux_relative_spread'] >= 1e-8 or balance['wall_flux_linf'] != 0:
            raise ValueError('Actual conservative face flux failed the unchanged mass gates')
        records.append({'root': str(root), 'dt': config['time_step'], 'steps': config['time_steps'],
                        'executable': run['executable'], 'executable_sha256': run['executable_sha256'],
                        'anderson_depth': config.get('anderson_depth', 0),
                        'pressure_volume_rms': rms(p, volume), 'pressure_gradient_volume_rms': rms(gp, volume),
                        'gradient_source_replay_absolute_linf': gradient_replay,
                        'face_flux_identity_relative_l2': flux_identity,
                        'pressure_flux_correction_relative_l2': float(np.linalg.norm(correction)/np.linalg.norm(q)),
                        'interpolated_velocity_divergence_volume_rms': rms(raw_net/volume, volume),
                        'actual_flux_divergence_volume_rms': rms(actual_net/volume, volume),
                        'steady_momentum_relative_l2': metrics['steady_momentum_relative_l2'],
                        'temporal_acceleration_relative_l2': metrics['temporal_acceleration_relative_l2'],
                        'mass': balance})
        states.append((p, gp, u, cut, volume))

    pairs, differences = [], []
    for i, (a, b) in enumerate(zip(states, states[1:])):
        ra, rb = records[i:i+2]
        if not np.isclose(ra['dt'], 2*rb['dt'], rtol=1e-13, atol=0) or not np.isclose(ra['dt']*ra['steps'], rb['dt']*rb['steps'], rtol=1e-13, atol=0):
            raise ValueError('Require time-step halvings at equal final physical times')
        p, gp, u, cut, v = a
        q, gq, w, _, _ = b
        dp = q-p
        energy = v*dp*dp
        bands = []
        fraction = v/mesh['h']**3
        for label, mask in [('volume_fraction_lt_0.001', fraction < .001),
                            ('volume_fraction_0.001_to_0.1', (fraction >= .001) & (fraction < .1)),
                            ('volume_fraction_0.1_to_1_cut', (fraction >= .1) & cut), ('uncut', ~cut)]:
            bands.append({'region': label, 'cells': int(mask.sum()), 'fluid_volume_fraction': float(v[mask].sum()/v.sum()),
                          'pressure_difference_energy_fraction': float(energy[mask].sum()/energy.sum())})
        pairs.append({'dt': [ra['dt'], rb['dt']],
                      'same_executable': ra['executable_sha256'] == rb['executable_sha256'],
                      'same_anderson_depth': ra['anderson_depth'] == rb['anderson_depth'],
                      'pressure': error(p, q, v), 'cell_pressure_gradient': error(gp, gq, v), 'velocity': error(u, w, v),
                      'pressure_difference_volume_rms': rms(dp, v),
                      'cut_pressure_difference_energy_fraction': float(energy[cut].sum()/energy.sum()),
                      'cut_fluid_volume_fraction': float(v[cut].sum()/v.sum()), 'regions': bands})
        differences.append(dp)
    for i in range(1, len(pairs)):
        pairs[i]['successive_absolute_pressure_difference_ratio'] = pairs[i]['pressure_difference_volume_rms']/pairs[i-1]['pressure_difference_volume_rms']
    report = {'scope': __doc__, 'diagnostic_replay_passed': True, 'goal_complete': False,
              'pressure_time_step_gate': {'relative_l2_limit': .0025,
                                          'passed': all(p['pressure']['relative_l2'] < .0025 for p in pairs)},
              'flux_identity': 'q_f = area_f * (I u)_f + dt/rho * area_f * ((I Gc p)_f - (Gf p)_f)',
              'identity_scope': 'Uniform cut grid, constant density/body force, I(1)=1 on every open face; Gf is the two-cell pressure-projection gradient. Wall flux is zero.',
              'limitations': ['The pressure source dump precedes the final inner update; its replay is checked to 1e-9 absolute acceleration scale.',
                             'Different archived executables/Anderson settings are recorded. Only same-executable pairs constitute a controlled binary comparison.',
                             'The identity proves an explicit dt-dependent pressure/velocity coupling remains at steady state. BCG also depends on dt; this is not a causal isolation experiment.',
                             'Interpolated cell velocity flux is not the conservative solver flux. Its divergence is not a mass-gate failure.',
                             'These coarse-grid diagnostics do not establish fine-grid temporal or spatial convergence.'],
              'runs': records, 'pairs': pairs, 'trajectories': trajectories, 'source_sha256': sources}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    with (out/'pressure_differences.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'x', 'y', 'z', 'volume', 'cut']+[f'delta_p_pair_{i+1}' for i in range(len(pairs))])
        for i in range(nc):
            writer.writerow([int(mesh['id'][i]), mesh['x'][i], mesh['y'][i], mesh['z'][i], mesh['volume'][i], int(states[0][3][i])]+[a[i] for a in differences])
    print(json.dumps({'diagnostic_replay_passed': True, 'pressure_changes': [p['pressure']['relative_l2'] for p in pairs],
                      'gradient_changes': [p['cell_pressure_gradient']['relative_l2'] for p in pairs],
                      'same_executable_pairs': [p['same_executable'] for p in pairs]}))


if __name__ == '__main__':
    main()
