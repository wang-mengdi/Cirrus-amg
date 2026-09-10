"""Replay a retained failed pressure RHS through the actual projection constructor.

Each run retains its own geometry/operator checks and binary readbacks. A failed
linear solve is a diagnostic result, never a completed physical flow step.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--failure', type=int, default=1)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--dimension', type=int, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.dimension <= 64 or not 1 <= args.repetitions <= 16 or args.failure < 1:
        parser.error('Invalid Krylov dimension, repetitions or failure index')
    repo = Path(__file__).resolve().parents[1]
    parent, build = args.parent.resolve(strict=True), args.build.resolve(strict=True)
    output, record = args.output.resolve(), args.record.resolve()
    if output.drive.lower() != 'd:' or record.drive.lower() != 'd:':
        raise ValueError('Use D drive for new replay data')
    if output.exists() or record.exists():
        raise ValueError('Preserve existing replay results')
    failure = parent/f'linear_failure_{args.failure}'
    old = json.loads((parent/'case.json').read_text())
    info = json.loads((failure/'failure.json').read_text())
    done = json.loads((parent/'run_completion.json').read_text())
    manifest = json.loads((parent/'run_manifest.json').read_text())
    compiled = json.loads((build/'build_manifest.json').read_text())
    executable = build/'simple_channel.exe'
    if done['exit_code'] != 1 or not all(done[k] for k in
            ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
        raise ValueError('Expected a terminated numerical-failure parent with unchanged inputs')
    if info['operator'] != 'pressure' or info['failed_solution_accepted'] or info['tolerance'] != old['linear_tolerance']:
        raise ValueError('Expected an unaccepted pressure failure with the original tolerance')
    if old['linear_tolerance'] != 1e-13 or info['true_relative_residual'] <= info['tolerance']:
        raise ValueError('Expected the strict 1e-13 pressure failure')
    if compiled['exit_code'] != 0 or not compiled['source_unchanged'] or sha(executable) != compiled['executable_sha256']['simple_channel.exe']:
        raise ValueError('Unverified replay build')
    inputs = [parent/name for name in ('case.json', 'run_manifest.json', 'run_completion.json',
              'mesh_cells.csv', 'mesh_faces.csv', 'projection_method.json', 'material_faces.csv',
              'material_cells.csv', 'embedded_operator_checks.json')]
    inputs += [failure/'rhs.bin', failure/'failure.json', build/'build_manifest.json', executable,
               Path(__file__), repo/'scripts/run_twisted_solver.py']
    for name, expected in compiled['source_sha256'].items():
        snapshot = build/'sources'/(name+'.txt')
        if sha(snapshot) != expected:
            raise ValueError('Compiled source snapshot changed: '+name)
        inputs.append(snapshot)
    for name, expected in manifest['geometry_input_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Parent geometry input changed')
        inputs.append(Path(name))
    if sha(Path(manifest['executable'])) != manifest['executable_sha256'] or sha(Path(manifest['config'])) != manifest['config_sha256']:
        raise ValueError('Actual parent executable or configuration changed')
    inputs += [Path(manifest['executable']), Path(manifest['config'])]
    before = {str(p): sha(p) for p in inputs}
    if (failure/'rhs.bin').stat().st_size != info['rhs_count']*8:
        raise ValueError('Wrong retained RHS length')
    config = dict(old)
    config.pop('restart_checkpoint', None)
    config.update(output=str(output), operator_only=True, gpu_krylov_dimension=args.dimension,
                  pressure_replay={'rhs': str(failure/'rhs.bin'), 'repetitions': args.repetitions})
    record.mkdir(parents=True, exist_ok=False)
    case = record/'case.json'
    case.write_text(json.dumps(config, indent=2)+'\n')
    now = lambda: datetime.now(timezone.utc).isoformat()
    command = [sys.executable, str(repo/'scripts/run_twisted_solver.py'), '--config', str(case),
               '--exe', str(executable), '--threads', '2', '--measure-memory']
    report = {'scope': __doc__, 'started_utc': now(), 'parent': str(parent), 'output': str(output),
              'command': command, 'input_sha256': before, 'config_sha256': sha(case),
              'parent_failure': info, 'goal_complete': False}
    (record/'launch.json').write_text(json.dumps(report, indent=2)+'\n')
    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '1'
    with (record/'runner.log').open('w') as log:
        process = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    report.update(completed_utc=now(), runner_exit_code=process.returncode,
                  inputs_unchanged=all(sha(Path(p)) == value for p, value in before.items()))
    names = ('mesh_cells.csv', 'mesh_faces.csv', 'material_faces.csv', 'material_cells.csv',
             'embedded_operator_checks.json')
    report['operator_files_identical'] = {name: (output/name).exists() and sha(output/name) == sha(parent/name) for name in names}
    replay_path = output/'pressure_replay.json'
    report['complete_replay'] = False
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        method = json.loads((output/'projection_method.json').read_text())
        original = json.loads((parent/'projection_method.json').read_text())
        method.pop('gpu_krylov_dimension', None)
        original.pop('gpu_krylov_dimension', None)
        report['projection_method_identical_except_dimension'] = method == original
        report['rhs_echo_identical'] = sha(output/'pressure_replay_rhs.bin') == sha(failure/'rhs.bin')
        report['replay'] = replay
        report['complete_replay'] = (process.returncode in (0, 3) and len(replay['trials']) == args.repetitions
            and report['inputs_unchanged'] and all(report['operator_files_identical'].values())
            and report['projection_method_identical_except_dimension'] and report['rhs_echo_identical'])
        report['no_physical_step_output'] = not (output/'time_history.csv').exists() and not list(output.glob('step_*'))
        report['complete_replay'] &= report['no_physical_step_output']
        report['output_sha256'] = {str(p): sha(p) for p in output.glob('pressure_replay*') if p.is_file()}
    (record/'completion.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_sha256', 'output_sha256')}), flush=True)
    if not report['complete_replay']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
