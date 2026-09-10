"""Validate completed pseudo steady iterations, optionally against an ordinary steady control.

The candidate is not a physical trajectory. State-only mode checks original
equation/mass gates, native topology and actual final raw-map dumps, and makes
no field-equivalence claim. The ordinary-control mode additionally compares
same-grid fields. Aphros alignment and time/grid convergence remain separate.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from analyze_twisted_refinement import completed_trajectory
from check_twisted_native_run import columns
from compare_twisted import error
from run_twisted_solver import sha


def validate(run, reference=None):
    root = run.resolve(strict=True)
    if reference is not None:
        reference = reference.resolve(strict=True)
    repo = Path(__file__).resolve().parents[1]
    hashes = {}

    def retain(path):
        path = path.resolve(strict=True)
        hashes[str(path)] = sha(path)
        return path

    def load(path):
        return json.loads(retain(path).read_text(encoding='utf-8-sig'))

    def table(path, names):
        value = columns(retain(path), names)
        if not len(value) or not np.isfinite(value).all():
            raise ValueError('Empty or nonfinite numerical table: '+str(path))
        return value

    def rows(path):
        with retain(path).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def local(path):
        path = Path(path)
        return path if path.is_absolute() else repo/path

    cfg = load(root/'case.json')
    runtime = load(root/'run_manifest.json')
    done = load(root/'run_completion.json')
    summary = load(root/'steady_summary.json')
    if done['exit_code'] or not all(done[k] for k in
            ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
        raise ValueError('Candidate did not complete with unchanged inputs')
    depth = cfg.get('steady_anderson_depth', 0)
    if type(depth) is not int or not 1 <= depth <= 10 or cfg.get('restart_checkpoint'):
        raise ValueError('Expected a fresh, explicitly requested steady iteration')
    required = {'fluid_solver': 'proj', 'linear_backend': 'native_gpu',
                'gpu_preconditioner': 'native_amg', 'gpu_pressure_gauge': 'mean_zero',
                'gpu_pressure_operator': 'full', 'gpu_viscosity_operator': 'full',
                'linear_tolerance': 1e-13}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Steady candidate changed the required original native operators/tolerance')
    if (summary.get('fluid_solver') != 'proj_steady' or not summary['converged'] or
            not summary['steady_converged'] or summary.get('physical_trajectory_claimed') is not False or
            summary.get('steady_anderson_depth') != depth or
            summary.get('maximum_steady_iterations') != cfg['time_steps']):
        raise ValueError('Candidate is not a completed, explicitly labelled steady iteration')
    if any((root/name).exists() for name in ('transient_summary.json', 'time_history.csv')) or list(root.glob('step_*')):
        raise ValueError('Pseudo iterations must not be labelled as a physical trajectory')
    inputs = {runtime['executable']: runtime['executable_sha256'],
              runtime['config']: runtime['config_sha256'], **runtime['geometry_input_sha256']}
    for path, value in inputs.items():
        if sha(Path(path)) != value:
            raise ValueError('Executed input changed: '+path)
    if load(Path(runtime['config'])) != cfg:
        raise ValueError('Root configuration differs from actual executed input')
    exe = Path(runtime['executable'])
    build = load(exe.parent/'build_manifest.json')
    if build['exit_code'] or not build['source_unchanged'] or build['executable_sha256'][exe.name] != runtime['executable_sha256']:
        raise ValueError('Executed binary lacks a verified successful source snapshot')
    for name, value in build['source_sha256'].items():
        if sha(retain(exe.parent/'sources'/(name+'.txt'))) != value:
            raise ValueError('Compiled source changed: '+name)
    method = load(root/'projection_method.json')
    for key, value in {'linear_backend': 'native_gpu_fgmres_amg', 'pressure_operator': 'full',
                       'viscosity_operator': 'full', 'pressure_gauge': 'mean_zero'}.items():
        if method.get(key) != value:
            raise ValueError('Actual method differs from selected native algorithm')
    if method['native_amg_levels'] < 2:
        raise ValueError('Missing multilevel native AMG')
    trace = rows(root/'gpu_linear.csv')
    if {row['operator'] for row in trace} != {'pressure', 'diffusion'}:
        raise ValueError('Missing actual GPU pressure or diffusion calls')
    for row in trace:
        for key in ('true_relative_residual', 'compatibility_relative_l2'):
            value = float(row[key])
            if not math.isfinite(value) or not 0 <= value <= 1e-13:
                raise ValueError('Accepted native linear solve exceeded original tolerance')
    history = rows(root/'steady_history.csv')
    count = summary['iterations_completed']
    if type(count) is not int or not 2 <= count <= cfg['time_steps'] or len(history) != count:
        raise ValueError('Missing or inconsistent completed steady iterations')
    expected_folders = {f'iterate_{i:04d}' for i in range(1, count+1)}
    if {p.name for p in root.glob('iterate_*')} != expected_folders:
        raise ValueError('Noncontiguous or extra pseudo iteration folders')
    geometry = load(local(cfg['embedded_geometry']))
    height = geometry['extent'][1]
    limit = min(cfg['tolerance'], 1e-8)
    topology = table(root/'mesh_cells.csv', ('id', 'level', 'h', 'volume'))
    adjacency = table(root/'mesh_faces.csv', ('id', 'owner', 'neighbor', 'area'))
    if (not np.array_equal(topology[:, 0], np.arange(len(topology))) or
            not np.array_equal(adjacency[:, 0], np.arange(len(adjacency))) or
            np.any(topology[:, 2:] <= 0) or np.any(adjacency[:, 3] <= 0) or
            not np.equal(topology[:, 1], np.floor(topology[:, 1])).all()):
        raise ValueError('Invalid native topology IDs, levels, sizes or measures')
    if (not np.equal(adjacency[:, 1:3], np.floor(adjacency[:, 1:3])).all() or
            np.any(adjacency[:, 1] < 0) or np.any(adjacency[:, 1] >= len(topology)) or
            np.any(adjacency[:, 2] < -1) or np.any(adjacency[:, 2] >= len(topology))):
        raise ValueError('Invalid native shared-face topology')
    inside = adjacency[:, 2] >= 0
    owner = adjacency[:, 1].astype(int)
    neighbor = adjacency[inside, 2].astype(int)
    level_gap = abs(topology[owner[inside], 1]-topology[neighbor, 1])
    if np.any(level_gap > 1):
        raise ValueError('Native octree does not satisfy 2:1 face balance')
    interfaces = int(np.count_nonzero(level_gap))
    custom_viscosity = ~inside
    h = topology[owner, 2]
    custom_viscosity |= abs(adjacency[:, 3]-h*h) > 1e-12*h*h
    custom_viscosity[inside] |= level_gap != 0
    if (method['full_pressure_interface_faces'] != interfaces or
            method['full_viscosity_faces'] != int(np.count_nonzero(custom_viscosity)) or
            (cfg['adaptive'] and interfaces == 0)):
        raise ValueError('Actual native pressure/viscosity stencils or fluid refinement differ')
    native_grid = {'cells': len(topology), 'faces': len(adjacency),
                   'coarse_fine_faces': interfaces, 'wall_faces': int(np.count_nonzero(~inside)),
                   'full_viscosity_faces': int(np.count_nonzero(custom_viscosity))}

    def mass(folder):
        cells = table(folder/'solution.csv', ('id', 'volume', 'u', 'v', 'w'))
        faces = table(folder/'mesh_faces.csv', ('id', 'owner', 'neighbor'))
        flux = table(folder/'flux.csv', ('id', 'flux'))
        sections = table(folder/'sections.csv', ('volume_flux',))[:, 0]
        if (not np.array_equal(cells[:, 0], np.arange(len(cells))) or
                not np.array_equal(faces[:, 0], np.arange(len(faces))) or
                not np.array_equal(faces[:, 0], flux[:, 0]) or np.any(cells[:, 1] <= 0)):
            raise ValueError('Invalid mass table IDs or volumes')
        if (not np.array_equal(cells[:, :2], topology[:, [0, 3]]) or
                not np.array_equal(faces, adjacency[:, :3])):
            raise ValueError('Retained fields differ from the actual native topology')
        owner, neighbor = faces[:, 1], faces[:, 2]
        if (not np.equal(owner, np.floor(owner)).all() or not np.equal(neighbor, np.floor(neighbor)).all() or
                np.any(owner < 0) or np.any(owner >= len(cells)) or np.any(neighbor < -1) or np.any(neighbor >= len(cells))):
            raise ValueError('Invalid shared-face adjacency')
        interior = neighbor >= 0
        q = flux[:, 1]
        if np.any(q[~interior] != 0):
            raise ValueError('Nonzero impermeable-wall flux')
        net = np.zeros(len(cells))
        np.add.at(net, owner.astype(int), q)
        np.add.at(net, neighbor[interior].astype(int), -q[interior])
        rate = np.linalg.norm(cells[:, 2:], axis=1).max()/height
        through = float(sections.mean())
        if not rate > 0 or through == 0:
            raise ValueError('Degenerate driven-flow mass normalization')
        result = {'divergence_relative_linf': float(np.max(abs(net)/cells[:, 1])/rate),
                  'global_absolute_cell_flux_over_throughflow': float(abs(net).sum()/abs(through)),
                  'section_flux_relative_spread': float(np.ptp(sections)/abs(through))}
        result['passed'] = (result['divergence_relative_linf'] < 1e-7 and
                            result['global_absolute_cell_flux_over_throughflow'] < 1e-8 and
                            result['section_flux_relative_spread'] < 1e-8)
        return result

    fields, checks = [], []
    for iteration, row in enumerate(history, 1):
        folder = root/f'iterate_{iteration:04d}'
        conf, metrics = load(folder/'case.json'), load(folder/'metrics.json')
        expected = dict(cfg, output=str(folder), fluid_solver='proj_steady',
                        steady_iteration=iteration, pseudo_time=iteration*cfg['time_step'])
        if conf != expected or metrics.get('fluid_solver') != 'proj_steady' or metrics.get('steady_iteration') != iteration:
            raise ValueError('Pseudo iteration mislabeled or configuration changed')
        if int(row['iteration']) != iteration or row['inner_converged'] != 'true' or not metrics['converged']:
            raise ValueError('Unconverged pseudo iteration')
        if any(metrics[k] != native_grid[k] for k in ('cells', 'faces', 'coarse_fine_faces')):
            raise ValueError('Reported grid dimensions differ from native topology')
        if not math.isclose(float(row['pseudo_time']), iteration*cfg['time_step'], rel_tol=1e-12, abs_tol=0):
            raise ValueError('Inconsistent pseudo time')
        inner = rows(folder/'history.csv')
        if len(inner) != metrics['iterations'] or len(inner) != int(row['inner_iterations']):
            raise ValueError('Missing inner fixed-point history')
        for index, item in enumerate(inner, 1):
            if int(item['iteration']) != index or any(not math.isfinite(float(v)) for v in item.values()):
                raise ValueError('Nonfinite or incomplete inner iteration history')
        for key, bound in [('complete_inner_fixed_point_residual', limit),
                           ('implicit_diffusion_relative_l2', limit), ('continuity_relative_linf', 1e-7),
                           ('velocity_change_absolute_linf', cfg['projection_iteration_tolerance']),
                           ('cross_section_flux_relative_spread', 1e-8)]:
            if not 0 <= metrics[key] < bound:
                raise ValueError('Original convergence gate failed: '+key)
        for key in ('steady_map_velocity_defect_relative_l2', 'steady_momentum_relative_l2'):
            if not math.isfinite(metrics[key]) or metrics[key] < 0 or float(row[key]) != metrics[key]:
                raise ValueError('Invalid or inconsistent stationary residual')
        raw = table(folder/f'iter_{len(inner)}'/'cells.csv',
                    ('id', 'u', 'v', 'w', 'p', 'u_diff', 'v_diff', 'w_diff'))
        if metrics['field_output_written']:
            fields.append(iteration)
            actual = table(folder/'solution.csv', ('id', 'u', 'v', 'w', 'p'))
            if not np.array_equal(raw[:, :5], actual):
                raise ValueError('Final fields differ from the fresh unaccelerated map dump')
            raw_q = table(folder/f'iter_{len(inner)}'/'faces.csv', ('id', 'flux'))
            actual_q = table(folder/'flux.csv', ('id', 'flux'))
            if not np.array_equal(raw_q, actual_q):
                raise ValueError('Final flux differs from fresh unaccelerated map dump')
            wall = table(folder/'walls.csv', ('face_id', 'owner', 'area', 'tau_x', 'tau_y', 'tau_z'))
            expected_wall = adjacency[~inside][:, [0, 1, 3]]
            if (not np.array_equal(wall[:, :3], expected_wall) or len(wall) != metrics['cut_cells']):
                raise ValueError('Wall traction locations/areas differ from native boundary faces')
            mean_shear = float(np.average(np.linalg.norm(wall[:, 3:], axis=1), weights=wall[:, 2]))
            if not math.isclose(mean_shear, metrics['mean_wall_shear_magnitude'], rel_tol=1e-12, abs_tol=0):
                raise ValueError('Stored wall traction differs from the reported area-weighted mean')
            checks.append({'iteration': iteration, **mass(folder)})
    if fields != summary['field_output_iterations'] or not {1, count} <= set(fields):
        raise ValueError('Missing first/final full fields or inconsistent field schedule')
    final = root/f'iterate_{count:04d}'
    if Path(summary['final_output']).resolve() != final:
        raise ValueError('Summary points to another final result')
    final_metrics = load(final/'metrics.json')
    for key in ('steady_map_velocity_defect_relative_l2', 'steady_momentum_relative_l2', 'steady_complete_fixed_point_residual'):
        if not 0 <= final_metrics[key] < limit:
            raise ValueError('Final original stationary equation gate failed: '+key)
    acceleration = rows(root/'steady_acceleration.csv')
    if [int(r['after_iteration']) for r in acceleration] != list(range(2, count)):
        raise ValueError('Missing outer acceleration decisions')
    accepted = 0
    for item in acceleration:
        if item['accepted'] not in ('0', '1'):
            raise ValueError('Invalid outer acceleration decision')
        if item['accepted'] == '1':
            accepted += 1
            if (not all(math.isfinite(float(v)) for v in item.values()) or
                    not 0 <= float(item['candidate_equation_residual']) < float(item['raw_equation_residual']) or
                    not 0 <= float(item['candidate_continuity']) < 1e-7 or
                    not 0 <= float(item['roundoff_repair_relative_change']) < 1e-10 or
                    float(item['backtracking_factor']) not in (1., .5, .25, .125)):
                raise ValueError('Accepted outer proposal failed its safeguards')
        elif float(item['backtracking_factor']) != 0:
            raise ValueError('Rejected outer proposal has a nonzero applied factor')
    failures = []
    failure_log = root/'steady_acceleration_failures.csv'
    if failure_log.exists():
        failures = rows(failure_log)
        if any(item['failed_solution_accepted'] != 'false' for item in failures):
            raise ValueError('Failed linear solution accepted by outer acceleration')

    if reference is None:
        for path, value in {**hashes, **inputs}.items():
            if sha(Path(path)) != value:
                raise ValueError('Input/output changed during state validation: '+path)
        return {'passed': all(item['passed'] for item in checks), 'scope': __doc__,
                'validation_mode': 'state_only', 'physical_trajectory_claimed': False,
                'same_grid_equivalence_checked': False, 'independent_reference_alignment_checked': False,
                'spatial_convergence_checked': False, 'candidate': str(root), 'native_grid': native_grid,
                'steady_iterations': count, 'accepted_outer_proposals': accepted,
                'rejected_or_warmup_proposals': len(acceleration)-accepted,
                'outer_linear_trial_failures': failures, 'final_metrics': final_metrics,
                'retained_field_mass': checks, 'successful_gpu_calls': len(trace),
                'maximum_accepted_linear_residual': max(float(r['true_relative_residual']) for r in trace),
                'executed_input_sha256': inputs, 'source_sha256': hashes,
                'checker_sha256': sha(Path(__file__))}

    control_cfg, control_metrics = load(reference/'case.json'), load(reference/'metrics.json')
    if control_cfg.get('fluid_solver') != 'proj' or control_cfg.get('steady_anderson_depth', 0):
        raise ValueError('Control must be an ordinary physical projection trajectory')
    proof = completed_trajectory(reference, control_cfg)
    hashes.update(proof['source_sha256'])
    if not control_metrics['converged'] or any(not 0 <= control_metrics[k] < limit for k in
            ('steady_momentum_relative_l2', 'temporal_acceleration_relative_l2')):
        raise ValueError('Ordinary control is not steady')
    for key in ('rho', 'nu', 'force', 'convection', 'convection_scheme', 'adaptive', 'ny',
                'periodic_x', 'periodic_z', 'time_step', 'tolerance', 'linear_tolerance',
                'projection_iteration_tolerance', 'momentum_mode'):
        if cfg.get(key) != control_cfg.get(key):
            raise ValueError('Different discrete physical problem: '+key)
    if sha(retain(local(cfg['embedded_geometry']))) != sha(retain(local(control_cfg['embedded_geometry']))):
        raise ValueError('Different geometry inputs')
    for name in ('mesh_cells.csv', 'mesh_faces.csv'):
        if sha(retain(final/name)) != sha(retain(reference/name)):
            raise ValueError('Different actual native mesh ordering or geometry')
    solutions = [table(p/'solution.csv', ('id', 'volume', 'u', 'v', 'w', 'p')) for p in (final, reference)]
    walls = [table(p/'walls.csv', ('face_id', 'owner', 'x', 'y', 'z', 'area', 'tau_x', 'tau_y', 'tau_z')) for p in (final, reference)]
    fluxes = [table(p/'flux.csv', ('id', 'flux')) for p in (final, reference)]
    if (not np.array_equal(solutions[0][:, :2], solutions[1][:, :2]) or
            not np.array_equal(walls[0][:, :6], walls[1][:, :6]) or
            not np.array_equal(fluxes[0][:, 0], fluxes[1][:, 0])):
        raise ValueError('Different solution/traction/shared-face ordering')
    volume = solutions[0][:, 1]
    velocity = [s[:, 2:5] for s in solutions]
    pressure = [s[:, 5]-np.average(s[:, 5], weights=volume) for s in solutions]
    cut = np.isin(solutions[0][:, 0], walls[0][:, 1])
    differences = {'velocity': error(*velocity, volume), 'pressure': error(*pressure, volume),
                   'cut_cell_velocity': error(velocity[0][cut], velocity[1][cut], volume[cut]),
                   'wall_shear': error(walls[0][:, 6:], walls[1][:, 6:], walls[0][:, 5]),
                   'shared_face_flux': error(fluxes[0][:, 1], fluxes[1][:, 1], np.ones(len(fluxes[0])))}
    control_mass = mass(reference)
    passed = (all(item['passed'] for item in checks) and control_mass['passed'] and
              all(value['relative_l2'] < 1e-6 for value in differences.values()))
    for path, value in {**hashes, **inputs}.items():
        if sha(Path(path)) != value:
            raise ValueError('Input/output changed during validation: '+path)
    return {'passed': passed, 'scope': __doc__, 'physical_trajectory_claimed': False,
            'validation_mode': 'ordinary_control', 'same_grid_equivalence_checked': True,
            'independent_reference_alignment_checked': False, 'spatial_convergence_checked': False,
            'native_grid': native_grid,
            'candidate': str(root), 'ordinary_reference': str(reference),
            'steady_iterations': count, 'ordinary_physical_steps': proof['completed_steps'],
            'accepted_outer_proposals': accepted, 'rejected_or_warmup_proposals': len(acceleration)-accepted,
            'outer_linear_trial_failures': failures, 'final_metrics': final_metrics,
            'retained_field_mass': checks, 'ordinary_final_mass': control_mass,
            'relative_field_limit': 1e-6, 'field_differences': differences,
            'successful_gpu_calls': len(trace),
            'maximum_accepted_linear_residual': max(float(r['true_relative_residual']) for r in trace),
            'executed_input_sha256': inputs, 'source_sha256': hashes,
            'checker_sha256': sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ordinary-final', type=Path)
    group.add_argument('--state-only', action='store_true', help='Check original state gates without claiming ordinary or Aphros field agreement')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous steady checks')
    result = validate(args.run, args.ordinary_final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('passed', 'validation_mode', 'native_grid',
                      'steady_iterations', 'accepted_outer_proposals', 'successful_gpu_calls',
                      'maximum_accepted_linear_residual', 'field_differences') if key in result}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
