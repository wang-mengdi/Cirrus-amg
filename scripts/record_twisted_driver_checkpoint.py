"""Preserve completed driver/NS64 evidence without claiming the full goal."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
from run_twisted_solver import sha


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo/'output/twisted'
    target = repo/'validation/twisted/results/driver_ns64_checkpoint'
    if target.exists():
        raise ValueError('Preserve the historical checkpoint')
    reports = {
        'driver_stokes16': ('compare_stokes16_minimal_driver.json', True),
        'driver_transient16': ('minimal_driver_ns16_transient.json', True),
        'driver_steady16': ('minimal_driver_ns16_steady.json', True),
        'driver_full_v10': ('minimal_driver_ns16_full_v10.json', True),
        'adaptive_ns64_independent_step1': ('compare_ns64_adaptive_step1_aphros.json', True),
        'adaptive_ns64_acceleration_step1': ('anderson_ns64_completed_step1.json', True),
        'partial_reference_rejects_steady': ('prefix_rejection_check.json', True),
        'full_reference_comparator_regression': ('compare_ns16_full_after_prefix_support.json', True),
        'adaptive_ns64_paraview': ('ours_ns64_adaptive_aa5_v1/paraview_time_readback.json', True),
        'ns32_64_refinement_failure': ('refinement_ns32_64_large_dt/refinement.json', False)}
    values = {name: json.loads((source/path).read_text(encoding='utf-8-sig'))
              for name, (path, _) in reports.items()}
    for name, (_, expected) in reports.items():
        if values[name]['passed'] != expected:
            raise ValueError(f'Unexpected validation outcome: {name}')
    if values['adaptive_ns64_independent_step1']['complete_reference_run_checked']:
        raise ValueError('A completed prefix was mislabeled as the full reference run')
    if values['adaptive_ns64_acceleration_step1']['complete_sequence_checked']:
        raise ValueError('Stopped ordinary run was mislabeled as a complete time sequence')
    selected = []
    for run, steps in (('ours_ns64_adaptive_aa5_v1', 8), ('ours_ns64_adaptive_large_dt2_aa5_v1', 2)):
        root = source/run
        summary = json.loads((root/'transient_summary.json').read_text())
        completion = json.loads((root/'run_completion.json').read_text())
        if not summary['steady_converged'] or summary['steps_completed'] != steps or completion['exit_code']:
            raise ValueError('NS64 sequence is incomplete')
        selected += [root/name for name in ('case.json', 'time_history.csv', 'transient_summary.json',
                                            'run_manifest.json', 'run_completion.json', 'embedded_operator_checks.json')]
        if not json.loads((root/'embedded_operator_checks.json').read_text())['passed']:
            raise ValueError('Embedded operator audit failed')
        for step in range(1, steps+1):
            selected += [root/f'step_{step:04d}'/'metrics.json']
    selected += [source/'ours_ns64_adaptive_amg85/intentional_stop.json']
    baseline = Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    memory = json.loads((baseline/'navier_stokes_n16_full_memory_v10/observed_memory.json').read_text(encoding='utf-8-sig'))
    record = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'completed_goal': False,
        'scope': 'Minimal driver equivalence, completed native NS64 time sequence, independent transient first step, and failed physical refinement',
        'parent_commit_at_capture': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'source_sha256': {str(p.relative_to(repo)): sha(p) for p in sorted(
            list((repo/'scripts').glob('*twisted*'))+list((repo/'validation/aphros').glob('*twisted*'))+
            [repo/'validation/twisted/IMPLEMENTATION.md', repo/'validation/twisted/ns64_steady.json']) if p.is_file()},
        'solver_executable_sha256': sha(source/'simple_channel_wall_audit_v2.exe'),
        'evidence': {name: {'source': str(source/path), 'sha256': sha(source/path), 'passed': expected}
                     for name, (path, expected) in reports.items()},
        'full_driver_memory': memory,
        'remaining': ['Complete independent NS64 reference and compare the final steady fields',
                      'Complete independent Stokes128 reference and compare uniform/adaptive fields',
                      'Complete n32 minimal-driver equivalence check',
                      'Meet physical near-wall grid-convergence and sampling-sensitivity requirements']}
    record['artifact_sha256'] = {}
    for path in selected:
        if not path.is_file():
            raise ValueError(f'Missing checkpoint input: {path}')
    for run in ('ours_ns64_adaptive_aa5_v1', 'ours_ns64_adaptive_large_dt2_aa5_v1'):
        for p in sorted((source/run).rglob('*')):
            if p.is_file() and p.name in {'solution.csv', 'walls.csv', 'sections.csv', 'solution.pvd', 'walls.pvd', 'cut_geometry.png'}:
                record['artifact_sha256'][str(p.relative_to(source))] = sha(p)
    target.mkdir(parents=True)
    for name, value in values.items():
        (target/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    for path in selected:
        destination = target/path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (target/'minimal_driver_memory.csv').write_bytes((baseline/'navier_stokes_n16_steady_minimal_v1/driver_memory.csv').read_bytes())
    (target/'driver_build.json').write_bytes((baseline/'driver_build_v1/build_manifest.json').read_bytes())
    (target/'checkpoint.json').write_text(json.dumps(record, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
