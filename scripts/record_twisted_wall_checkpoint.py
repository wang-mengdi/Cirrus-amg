"""Archive wall diagnostics, rejected experiments, and completed NS evidence."""
import datetime
import json
from pathlib import Path
import subprocess
from run_twisted_solver import sha


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo/'output/twisted'
    target = repo/'validation/twisted/results/wall_consistency_checkpoint'
    if target.exists():
        raise ValueError('Preserve the historical snapshot; choose a new checkpoint')
    paths = {
        'steady_ns32': ('compare_ns32_steady_time_checked.json', True),
        'accelerated_ns16_transient': ('compare_ns16_transient_aa5.json', True),
        'accelerated_ns32_steady': ('compare_ns32_steady_aa5.json', True),
        'acceleration_ns16_pair': ('anderson_ns16_time_pair.json', True),
        'acceleration_ns32_pair': ('anderson_ns32_time_pair.json', True),
        'acceleration_failed_step': ('time_failure_aa5_check.json', True),
        'default_stokes_regression': ('compare_stokes16_wall_audit.json', True),
        'current_binary_quick': ('straight_regression_wall_audit_v2/suite_results.json', True),
        'paraview_ns32_time_readback': ('ours_ns32_steady_v2/paraview_time_readback.json', True),
        'paraview_ns32_accelerated_time_readback': ('ours_ns32_steady_aa5_v1/paraview_time_readback.json', True),
        'mismatched_time_step_failure': ('compare_ns16_mismatched_time_checked.json', False),
        'large_step_first_steady_residual_failure': ('compare_ns16_large_dt_matched.json', False),
        'large_step_ns16_matched': ('compare_ns16_large_dt2_matched.json', True),
        'large_step_ns32_matched': ('compare_ns32_large_dt85.json', True),
        'quadratic_flow_refinement_failure': ('refinement_quad5_32_64/refinement.json', False),
        'linear_flow_refinement_failure': ('refinement_uniform64_128/refinement.json', False),
        'aphros_manufactured_check_n16': ('wall_consistency_v1/aphros_check_n16.json', True),
        'aphros_manufactured_check_n64': ('wall_consistency_v1/aphros_check_n64.json', True),
        'manufactured_quad3': ('wall_consistency_quad_candidate/wall_consistency.json', None),
        'manufactured_quad5': ('wall_consistency_quad5_candidate/wall_consistency.json', None)}
    values = {}
    for name, (relative, expected) in paths.items():
        value = json.loads((source/relative).read_text(encoding='utf-8-sig'))
        if expected is not None and value['passed'] != expected:
            raise ValueError(f'Unexpected result: {relative}')
        values[name] = value
    current_exe = source/'simple_channel_wall_audit_v2.exe'
    for name in ('current_binary_quick', 'acceleration_failed_step'):
        if values[name]['executable_sha256'] != sha(current_exe):
            raise ValueError('Current binary validation hash mismatch')
    if not values['current_binary_quick']['strict_current_binary_pass']:
        raise ValueError('Quick validation does not describe the current binary')
    source_files = list((repo/'simple').glob('*'))+list((repo/'scripts').glob('*twisted*'))
    source_files += list((repo/'validation/aphros').glob('twisted*'))
    source_files += [repo/'validation/check_twisted_time_failure.py', repo/'validation/check_twisted_paraview.py',
                     repo/'validation/twisted/IMPLEMENTATION.md', repo/'validation/twisted/ns32_steady.json',
                     repo/'validation/twisted/experiments/quadratic5.patch', repo/'validation/twisted/experiments/README.md']
    runs = ('ours_ns32_steady_v2', 'ours_ns16_transient_aa5_v1', 'ours_ns32_steady_aa5_v1',
            'ours_ns16_large_dt2_quad5_default', 'ours_ns32_large_dt85', 'ours_stokes16_wall_audit',
            'quad5_uniform16_v2', 'quad5_uniform32_v2', 'quad5_uniform64_v2')
    artifact_hashes = {}
    retained_names = {'case.json', 'metrics.json', 'solution.csv', 'walls.csv', 'sections.csv',
                      'run_manifest.json', 'run_completion.json', 'time_history.csv', 'transient_summary.json',
                      'acceleration.csv', 'embedded_operator_checks.json', 'solution.pvd', 'walls.pvd'}
    for name in runs:
        artifact_hashes[name] = {str(p.relative_to(source/name)): sha(p)
                                for p in sorted((source/name).rglob('*')) if p.is_file() and p.name in retained_names}
    # The rejected configuration must fail before emitting any solution.
    rejected = source/'rejected_quad5_config_v2'
    rejection = json.loads((rejected/'run_completion.json').read_text())
    if rejection['exit_code'] == 0 or list(rejected.rglob('solution.csv')):
        raise ValueError('Archived wall scheme was silently accepted')
    if 'rejected quadratic5 experiment is archived' not in (rejected/'run.log').read_text():
        raise ValueError('Configuration failed for an unrelated reason')
    if json.loads((rejected/'run_manifest.json').read_text())['executable_sha256'] != sha(current_exe):
        raise ValueError('Rejected-configuration check used another executable')
    baseline = Path('D:/Dropbox/Agent-simulation/twisted-baseline/aphros')
    record = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'completed_goal': False,
        'scope': 'Wall consistency diagnosis, rejected quadratic reconstruction, and safeguarded NS acceleration',
        'parent_commit_at_capture': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'source_sha256': {str(p.relative_to(repo)): sha(p) for p in sorted(set(source_files)) if p.is_file()},
        'current_executable': str(current_exe), 'current_executable_sha256': sha(current_exe),
        'experiment_base_commit': 'b742956',
        'experiment_executable_sha256': sha(source/'simple_channel_quad5_v1.exe'),
        'experiment_note': 'Archived patch records the rejected compiled variant; default solver excludes it',
        'aphros_revision': 'b60ce3da52c19935fa24c778f62f02141eaf7f80',
        'aphros_executable_sha256': {name: sha(baseline/'src'/name) for name in ('main_amg_v9.exe', 'main_amg_v10.exe')},
        'aphros_modifications': ['prescribed geometry and read-only diagnostics', 'explicit wall-flux initialization fix',
                                'explicit periodic face-flux halo exchange', 'optional validated direct/AMG linear backend'],
        'rejected_configuration_exit_code': rejection['exit_code'],
        'evidence': {name: {'source': str(source/relative), 'source_sha256': sha(source/relative), 'expected_pass': expected}
                     for name, (relative, expected) in paths.items()},
        'run_artifact_sha256': artifact_hashes,
        'remaining': ['finish and compare actual adaptive NS64 against independent Aphros',
                      'finish independent Stokes128 reference and compare fields',
                      'resolve physical near-wall grid convergence and wall-sampling sensitivity']}
    target.mkdir(parents=True)
    for name, value in values.items():
        (target/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    for name in ('quad5_uniform16_v2', 'quad5_uniform32_v2', 'quad5_uniform64_v2'):
        folder = target/name; folder.mkdir()
        for filename in ('case.json', 'metrics.json', 'embedded_operator_checks.json'):
            (folder/filename).write_bytes((source/name/filename).read_bytes())
    (target/'checkpoint.json').write_text(json.dumps(record, indent=2)+'\n')
    print(target/'checkpoint.json')


if __name__ == '__main__':
    main()
