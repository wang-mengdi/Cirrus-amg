"""Exercise steady-mode option guards and corrupt synthetic result fixtures.

Copies on D are deliberately synthetic checker tests, never solver evidence.
The original completed run and its ordinary reference are read only.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from check_twisted_steady_iteration import validate
from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ordinary-final', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--option-exe', type=Path, help='Test a parser-only rebuild against the same real field proof')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.drive.lower() != 'd:':
        raise ValueError('Put synthetic result copies on D')
    out.mkdir(parents=True, exist_ok=False)
    root = args.run.resolve(strict=True)
    control = args.ordinary_final.resolve(strict=True)
    positive = validate(root, control)
    if not positive['passed']:
        raise ValueError('Real positive control must pass first')
    runtime = json.loads((root/'run_manifest.json').read_text())
    option_exe = args.option_exe.resolve(strict=True) if args.option_exe else Path(runtime['executable'])
    option_exe_hash = sha(option_exe)
    config = json.loads((root/'case.json').read_text())
    repo = Path(__file__).resolve().parents[1]
    records = []

    def write(path, data):
        path.write_text(json.dumps(data, indent=2)+'\n')

    options = {
        'bool_depth': {'steady_anderson_depth': True},
        'float_depth': {'steady_anderson_depth': 1.5},
        'string_depth': {'steady_anderson_depth': '5'},
        'negative_depth': {'steady_anderson_depth': -1},
        'excess_depth': {'steady_anderson_depth': 11},
        'overflow_zero_depth': {'steady_anderson_depth': 2**32},
        'overflow_positive_depth': {'steady_anderson_depth': 2**32+5},
        'underflow_depth': {'steady_anderson_depth': -(2**32)},
        'simple_method': {'fluid_solver': 'simple'},
        'cpu_backend': {'linear_backend': 'cpu'},
        'compact_pressure': {'gpu_pressure_operator': 'compact'},
        'compact_viscosity': {'gpu_viscosity_operator': 'compact'},
        'pinned_gauge': {'gpu_pressure_gauge': 'pinned'},
        'jacobi_preconditioner': {'gpu_preconditioner': 'jacobi'},
        'physical_restart': {'restart_checkpoint': 'must_not_be_opened.json'},
        'one_iteration': {'time_steps': 1},
        'zero_step': {'time_step': 0.},
        'operator_probe': {'operator_only': True},
        'geometry_probe': {'geometry_only': True},
    }
    for name, change in options.items():
        path = out/(name+'.json')
        selected = copy.deepcopy(config)
        selected.update(change, output=str(out/(name+'_unexpected_output')),
                        embedded_geometry=str(out/'must_not_exist_geometry.json'))
        # If a malformed option slips through, nonexistent geometry prevents
        # mesh/GPU creation; creation of the output folder still exposes it.
        write(path, selected)
        result = subprocess.run([str(option_exe), str(path)], cwd=repo,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        (out/(name+'.log')).write_text(result.stdout)
        records.append({'kind': 'parser', 'name': name, 'exit_code': result.returncode,
                        'passed': result.returncode != 0 and not Path(selected['output']).exists(),
                        'diagnostic': result.stdout.strip()})

    def fixture(name):
        target = out/name
        shutil.copytree(root, target)
        cfg = json.loads((target/'case.json').read_text())
        cfg['output'] = str(target)
        write(target/'case.json', cfg)
        input_path = out/(name+'_input.json')
        write(input_path, cfg)
        manifest = json.loads((target/'run_manifest.json').read_text())
        manifest.update(config=str(input_path), config_sha256=sha(input_path),
                        synthetic_checker_fixture=True, original_run=str(root))
        write(target/'run_manifest.json', manifest)
        for folder in target.glob('iterate_*'):
            cfg = json.loads((folder/'case.json').read_text())
            cfg['output'] = str(folder)
            write(folder/'case.json', cfg)
        summary = json.loads((target/'steady_summary.json').read_text())
        summary['final_output'] = str(target/Path(summary['final_output']).name)
        write(target/'steady_summary.json', summary)
        return target, Path(summary['final_output'])

    def mutate_json(path, key, value):
        data = json.loads(path.read_text())
        data[key] = value
        write(path, data)

    def mutate_csv(path, column, value):
        lines = path.read_text().splitlines()
        index = lines[0].split(',').index(column)
        parts = lines[1].split(',')
        parts[index] = value
        lines[1] = ','.join(parts)
        path.write_text('\n'.join(lines)+'\n')

    def corrupt_accepted_proposal(target, final):
        path = target/'steady_acceleration.csv'
        lines = path.read_text().splitlines()
        header = lines[0].split(',')
        for index in range(1, len(lines)):
            parts = lines[index].split(',')
            if parts[header.index('accepted')] == '1':
                parts[header.index('candidate_equation_residual')] = '12345'
                lines[index] = ','.join(parts)
                path.write_text('\n'.join(lines)+'\n')
                return
        raise ValueError('Real control needs an accepted proposal for this test')

    def corrupt_matching_flux(target, final):
        inner = json.loads((final/'metrics.json').read_text())['iterations']
        mutate_csv(final/'flux.csv', 'flux', '0.0001')
        mutate_csv(final/f'iter_{inner}'/'faces.csv', 'flux', '0.0001')

    corruptions = {
        'physical_summary_claim': lambda t, f: mutate_json(t/'steady_summary.json', 'physical_trajectory_claimed', True),
        'physical_history': lambda t, f: (t/'time_history.csv').write_text('not a physical trajectory\n'),
        'loose_stationary_gate': lambda t, f: mutate_json(f/'metrics.json', 'steady_complete_fixed_point_residual', 1e-3),
        'unconverged_inner': lambda t, f: mutate_json(f/'metrics.json', 'complete_inner_fixed_point_residual', 1e-3),
        'unconverged_viscosity': lambda t, f: mutate_json(f/'metrics.json', 'implicit_diffusion_relative_l2', 1e-3),
        'nan_metric': lambda t, f: mutate_json(f/'metrics.json', 'steady_momentum_relative_l2', float('nan')),
        'different_final_velocity': lambda t, f: mutate_csv(f/'solution.csv', 'u', '123'),
        'different_final_flux': lambda t, f: mutate_csv(f/'flux.csv', 'flux', '123'),
        'wrong_wall_shear': lambda t, f: mutate_csv(f/'walls.csv', 'tau_x', '123'),
        'corrupt_independent_mass': corrupt_matching_flux,
        'nonfinite_raw_diffusion': lambda t, f: mutate_csv(f/f'iter_{json.loads((f/"metrics.json").read_text())["iterations"]}'/'cells.csv', 'u_diff', 'nan'),
        'bad_accepted_proposal': corrupt_accepted_proposal,
        'missing_raw_dump': lambda t, f: (f/f'iter_{json.loads((f/"metrics.json").read_text())["iterations"]}'/'cells.csv').rename(f/'synthetic_removed_raw_cells.csv'),
        'missing_iteration': lambda t, f: (t/'iterate_0002').rename(t/'synthetic_removed_iteration'),
        'loose_linear_residual': lambda t, f: mutate_csv(t/'gpu_linear.csv', 'true_relative_residual', '1e-9'),
        'unrecorded_physical_time': lambda t, f: mutate_json(f/'case.json', 'physical_time', .005),
        'false_completion': lambda t, f: mutate_json(t/'run_completion.json', 'exit_code', 3),
    }
    # Establish that rebasing the copied paths itself still passes. Otherwise
    # unrelated path errors could make every corruption look detected.
    template, _ = fixture('synthetic_uncorrupted_control')
    if not validate(template, control)['passed']:
        raise ValueError('Uncorrupted synthetic fixture must pass before corruption tests')
    for name, change in corruptions.items():
        target, final = fixture(name)
        change(target, final)
        try:
            result = validate(target, control)
            passed, reason = not result['passed'], 'Validation returned passed='+str(result['passed'])
        except (ValueError, OSError, KeyError) as error:
            passed, reason = True, str(error)
        records.append({'kind': 'synthetic_corruption', 'name': name, 'passed': passed, 'diagnostic': reason})
    output = {'passed': all(r['passed'] for r in records), 'scope': __doc__,
              'positive_real_run': str(root), 'positive_control_passed': positive['passed'],
              'synthetic_uncorrupted_control_passed': True,
              'option_executable': str(option_exe), 'option_executable_sha256': option_exe_hash,
              'cases': records, 'source_sha256': positive['source_sha256'],
              'checker_sha256': sha(Path(__file__)),
              'validator_sha256': sha(Path(__file__).with_name('check_twisted_steady_iteration.py'))}
    if sha(option_exe) != option_exe_hash:
        raise ValueError('Parser executable changed during tests')
    write(out/'result.json', output)
    print(json.dumps({'passed': output['passed'], 'cases': len(records),
                      'failed_cases': [r for r in records if not r['passed']]}))
    if not output['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
