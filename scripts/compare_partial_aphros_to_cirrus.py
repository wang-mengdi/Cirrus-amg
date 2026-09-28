"""Diagnose a completed Aphros physical step against a Cirrus steady state.

This is an exploratory comparison of states at different points in their
trajectories. It deliberately does not issue a steady-alignment verdict.
"""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import numpy as np

from analyze_twisted_refinement import completed_steady_iteration
from check_aphros_coupled_precision_probe import hex_decimal
from check_twisted_mass import check_case
from check_twisted_projection_flux import check_native
from compare_twisted import error, ordered, read, transfer, vector
from validate_aphros_extended_reference import verify_git_commit


def completed_step(reference, cfg):
    history = read(reference / 'tube_b0_time.csv')
    count = len(history)
    if not 1 <= count < cfg['time_steps']:
        raise ValueError('Require a partial trajectory with a completed physical step')
    for index, row in enumerate(history, 1):
        if not np.isclose(row['time'], index * cfg['time_step'], rtol=1e-12, atol=0):
            raise ValueError('Aphros physical time history is incomplete')
    exact = json.loads((reference / 'proj_final_b0_exact.json').read_text())
    physical_time = hex_decimal(exact['physical_time_hex'])
    expected_time = Decimal.from_float(cfg['time_step']) * count
    if abs(physical_time - expected_time) > abs(expected_time) * Decimal('1e-12'):
        raise ValueError('Stored Aphros field is not the last completed step')

    log = (reference / 'run.log').read_text(errors='replace')
    entries = [(int(i), float(e)) for i, e in re.findall(r'iter=(\d+), diff=([^\s]+)', log)]
    starts = [i for i, (iteration, _) in enumerate(entries) if iteration == 1]
    if len(starts) not in (count, count + 1):
        raise ValueError('Aphros iteration groups disagree with completed steps')
    for i in range(count):
        begin = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else len(entries)
        if [iteration for iteration, _ in entries[begin:end]] != list(range(1, end - begin + 1)):
            raise ValueError('Incomplete Aphros inner iteration sequence')
        if not 0 <= entries[end - 1][1] < cfg['iteration_tolerance']:
            raise ValueError('A completed Aphros step did not meet its inner tolerance')
    last_error = entries[(starts[count] if len(starts) > count else len(entries)) - 1][1]
    return count, {name: float(history[-1][name]) for name in history.dtype.names}, last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours', type=Path, required=True)
    parser.add_argument('--aphros', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--git-commit', help='Require the specified Cirrus Git HEAD')
    args = parser.parse_args()
    ours_root = args.ours.resolve()
    reference = args.aphros.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Preserve existing comparison reports')
    if output.drive.lower() != 'd:':
        raise ValueError('Keep experiment reports on D')
    verify_git_commit(args.git_commit)
    native_proof = completed_steady_iteration(ours_root, trust_local_files=True)
    case = json.loads((ours_root / 'case.json').read_text())
    cfg = json.loads((reference / 'case_manifest.json').read_text())
    if case['fluid_solver'] != 'proj_steady' or cfg['fluid_solver'] != 'proj':
        raise ValueError('Require Cirrus pseudo steady Proj and original Aphros Proj')
    if case['momentum_mode'] != 'imp' or cfg['momentum_mode'] != 'imp':
        raise ValueError('Different momentum modes')
    if case['convection_scheme'] != cfg['convection_scheme'] or not case['convection'] or not cfg['convection']:
        raise ValueError('Different convection equations')
    if case['projection_iteration_tolerance'] != cfg['iteration_tolerance']:
        raise ValueError('Different projection inner tolerances')
    if not np.isclose(case['time_step'], cfg['time_step'], rtol=1e-12, atol=0):
        raise ValueError('Different momentum time steps')
    if any(not np.isclose(case[name], cfg['spec'][name], rtol=1e-13, atol=0) for name in ('rho', 'nu')):
        raise ValueError('Different density or viscosity')
    if not np.allclose(case['force'], cfg['spec']['force'], rtol=1e-13, atol=0):
        raise ValueError('Different driving force')
    if cfg['projection_parameters'] != {'bcg': 1, 'redistr_adv': 0, 'diffusion_iters': 8, 'diffusion_consistent_guess': 1}:
        raise ValueError('Unexpected original Aphros projection settings')
    count, temporal, last_error = completed_step(reference, cfg)
    completion_path = reference / 'run_completion.json'
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
    if completion is None or completion['exit_code'] == 0:
        raise ValueError('Require the intentionally stopped, incomplete reference run')

    repo = Path(__file__).resolve().parents[1]
    geometry = Path(case['embedded_geometry'])
    if not geometry.is_absolute():
        geometry = repo / geometry
    meta = geometry.with_suffix('.meta.json')
    metadata = json.loads((meta if meta.exists() else geometry).read_text())
    if metadata['geometry_spec'] != cfg['spec']:
        raise ValueError('Different prescribed geometry')
    shift = np.asarray(metadata.get('reference_translation', [0, 0, 0]))
    h = cfg['spec']['extent'][1] / cfg['ny']
    nx = cfg['shape'][0]

    def translated(data):
        for axis, name in enumerate('xyz'):
            data[name] += shift[axis]
        data['x'] %= cfg['spec']['extent'][0]
        return ordered(data)

    ours = ordered(read(ours_root / 'solution.csv'))
    reference_cells = translated(read(reference / 'proj_final_b0_cells.csv'))
    sampled, transfer_report = transfer(ours, reference_cells, cfg['shape'], h)
    volume = ours['volume']
    fields = {
        'velocity': error(vector(ours, 'uvw'), sampled[:, :3], volume),
        'pressure': error(ours['p'] - np.average(ours['p'], weights=volume),
                          sampled[:, 3] - np.average(sampled[:, 3], weights=volume), volume),
    }
    wall = ordered(read(ours_root / 'walls.csv'))
    wall_reference = translated(read(reference / 'tube_final_b0_walls.csv'))
    if len(wall) != len(wall_reference) or not np.allclose(vector(wall, 'xyz'), vector(wall_reference, 'xyz'), atol=1e-13, rtol=0):
        raise ValueError('Wall locations differ')
    if not np.allclose(wall['area'], wall_reference['area'], rtol=1e-12, atol=0):
        raise ValueError('Wall areas differ')
    if not np.allclose(vector(wall, ('nx', 'ny', 'nz')), vector(wall_reference, ('nx', 'ny', 'nz')), rtol=0, atol=1e-12):
        raise ValueError('Wall normals differ')
    fields['wall_shear'] = error(vector(wall, ('tau_x', 'tau_y', 'tau_z')),
                                 vector(wall_reference, ('tau_x', 'tau_y', 'tau_z')), wall_reference['area'])
    cut = np.isin(ours['id'], wall['owner'])
    fields['cut_cell_velocity'] = error(vector(ours, 'uvw')[cut], sampled[cut, :3], volume[cut])

    faces = read(reference / 'proj_final_b0_faces.csv')
    sections = {}
    for face in faces:
        if int(face['axis']) != 0:
            continue
        ix = int(round(face['x'] / h))
        if ix == nx:
            continue
        ix = (ix + int(round(shift[0] / h))) % nx
        q, area = sections.get(ix, (0., 0.))
        sections[ix] = (q + face['flux'], area + face['area'])
    native_q, reference_q = [], []
    for section in read(ours_root / 'sections.csv'):
        q, area = sections[int(round(section['x'] / h)) % nx]
        if abs(section['area'] - area) > 1e-10 * area:
            raise ValueError('Incomplete cross section coverage')
        native_q.append(section['volume_flux'])
        reference_q.append(q)
    fields['section_flux'] = error(np.asarray(native_q), np.asarray(reference_q), np.ones(len(reference_q)))

    mass = check_case(reference, trust_local_files=True)
    projection_flux = check_native(ours_root, reference, cfg, metadata)
    result = {
        'scope': __doc__,
        'verdict': 'exploratory_diagnostic_only',
        'complete_reference_trajectory': False,
        'steady_alignment_established': False,
        'reason': 'Aphros is an early physical transient; Cirrus is a converged pseudo steady state.',
        'aphros_completed_steps': count,
        'aphros_configured_steps': cfg['time_steps'],
        'aphros_last_completed_time': temporal['time'],
        'aphros_last_completed_inner_change': last_error,
        'aphros_temporal_diagnostic': temporal,
        'aphros_stop_completion': completion,
        'native_steady_validated': native_proof['passed'],
        'native_final_metrics': native_proof['final_metrics'],
        'reference_mass': mass,
        'projection_flux': projection_flux,
        'transfer': transfer_report,
        'fields': fields,
        'input_paths': [str(p) for p in (ours_root, reference, meta if meta.exists() else geometry)],
        'git_commit': args.git_commit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verdict': result['verdict'], 'aphros_completed_steps': count,
                      'relative_l2': {name: data['relative_l2'] for name, data in fields.items()}}))


if __name__ == '__main__':
    main()
