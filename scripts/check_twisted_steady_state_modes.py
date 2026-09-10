"""Regress explicit steady state-only scope, ordinary equivalence, and damaged topology/fields.

Synthetic copies are checker tests on D, never claimed as executed CFD runs.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from check_twisted_steady_iteration import validate
from run_twisted_solver import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'ordinary-final', 'previous-check', 'negative-fixtures', 'budget-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if out.drive.lower() != 'd:': raise ValueError('Keep synthetic fields on D')
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads(a.previous_check.read_text())
    state = validate(a.run)
    controlled = validate(a.run, a.ordinary_final)
    assert old['passed'] and state['passed'] and controlled['passed']
    assert state['validation_mode'] == 'state_only' and controlled['validation_mode'] == 'ordinary_control'
    assert state['same_grid_equivalence_checked'] is False and controlled['same_grid_equivalence_checked'] is True
    assert all(state[k] is False for k in ('physical_trajectory_claimed', 'independent_reference_alignment_checked', 'spatial_convergence_checked'))
    assert 'field_differences' not in state and 'ordinary_reference' not in state
    for key in ('field_differences', 'retained_field_mass', 'ordinary_final_mass', 'final_metrics',
                'steady_iterations', 'successful_gpu_calls', 'maximum_accepted_linear_residual'):
        assert controlled[key] == old[key], 'Existing ordinary-control evidence changed: '+key
    (out/'state.json').write_text(json.dumps(state, indent=2)+'\n')
    (out/'ordinary.json').write_text(json.dumps(controlled, indent=2)+'\n')
    cases = []

    def rejected(root, name):
        try:
            result = validate(root)
            passed, message = not result['passed'], 'returned passed='+str(result['passed'])
        except (ValueError, KeyError, OSError) as error:
            passed, message = True, str(error)
        cases.append({'name': name, 'passed': passed, 'diagnostic': message, 'input': str(root)})
        if not passed: raise AssertionError('Accepted invalid state-only input: '+name)

    template = a.negative_fixtures/'synthetic_uncorrupted_control'
    assert validate(template)['passed'], 'Uncorrupted rebased fixture must pass first'
    fixture_index = json.loads((a.negative_fixtures/'result.json').read_text())
    for row in fixture_index['cases']:
        if row['kind'] == 'synthetic_corruption':
            rejected(a.negative_fixtures/row['name'], row['name'])
    rejected(a.budget_run, 'actual_exhausted_two_iteration_run')

    def write(path, value):
        path.write_text(json.dumps(value, indent=2)+'\n')

    def clone(name):
        target = out/name
        shutil.copytree(template, target)
        cfg = json.loads((target/'case.json').read_text());cfg['output'] = str(target)
        write(target/'case.json', cfg)
        input_path = out/(name+'_input.json');write(input_path, cfg)
        runtime = json.loads((target/'run_manifest.json').read_text())
        runtime.update(config=str(input_path), config_sha256=sha(input_path), synthetic_checker_fixture=True)
        write(target/'run_manifest.json', runtime)
        for folder in target.glob('iterate_*'):
            cfg = json.loads((folder/'case.json').read_text());cfg['output'] = str(folder);write(folder/'case.json', cfg)
        summary = json.loads((target/'steady_summary.json').read_text())
        summary['final_output'] = str(target/Path(summary['final_output']).name);write(target/'steady_summary.json', summary)
        return target

    for name, column, value in [('unbalanced_native_levels', 'level', '3'), ('nonpositive_native_volume', 'volume', '0')]:
        target = clone(name)
        path = target/'mesh_cells.csv';lines = path.read_text().splitlines();header = lines[0].split(',')
        row = lines[1].split(',');row[header.index(column)] = value;lines[1] = ','.join(row)
        path.write_text('\n'.join(lines)+'\n')
        rejected(target, name)
    for name, key in [('false_native_pressure_stencils', 'full_pressure_interface_faces'),
                      ('false_native_viscosity_stencils', 'full_viscosity_faces')]:
        target = clone(name);path = target/'projection_method.json';method = json.loads(path.read_text())
        method[key] += 1;write(path, method);rejected(target, name)

    for name, choices in [('missing_explicit_scope', []), ('conflicting_scopes', ['--state-only', '--ordinary-final', str(a.ordinary_final)])]:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('check_twisted_steady_iteration.py')),
            '--run', str(a.run), '--output', str(out/(name+'.json')), *choices], capture_output=True, text=True)
        assert result.returncode == 2 and not (out/(name+'.json')).exists()
        cases.append({'name': name, 'passed': True, 'exit_code': result.returncode})
        (out/(name+'.log')).write_text(result.stdout+result.stderr)
    inputs = {str(path.resolve()): sha(path) for path in (a.previous_check, a.negative_fixtures/'result.json',
              Path(__file__), Path(__file__).with_name('check_twisted_steady_iteration.py'))}
    report = {'passed': all(row['passed'] for row in cases), 'scope': __doc__,
              'actual_steady_state_passed': True, 'ordinary_equivalence_numbers_unchanged': True,
              'state_only_does_not_claim_equivalence': True, 'cases': cases,
              'source_sha256': inputs, 'checker_sha256': sha(Path(__file__))}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed', 'ordinary_equivalence_numbers_unchanged', 'state_only_does_not_claim_equivalence')}))


if __name__ == '__main__': main()
