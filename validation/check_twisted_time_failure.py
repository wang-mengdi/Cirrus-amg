"""A failed physical time step must not advance or claim a steady result."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=repo/'build/windows/x64/release/simple_channel.exe')
    parser.add_argument('--geometry', type=Path, default=repo/'output/twisted/geometry_n16.json')
    parser.add_argument('--output', type=Path, default=repo/'output/twisted/time_failure_check.json')
    parser.add_argument('--anderson-depth', type=int, default=0)
    parser.add_argument('--max-iterations', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.anderson_depth <= 10 or args.max_iterations < 1:
        parser.error('Invalid acceleration depth or iteration count')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='failed_step_', dir=args.output.parent.resolve()))
    cfg = {'ny': 16, 'embedded_geometry': str(args.geometry.resolve()), 'adaptive': False,
           'periodic_x': True, 'periodic_z': False, 'convection': True, 'convection_scheme': 'fou',
           'rho': 1., 'nu': .01, 'force': [1., 0., 0.], 'alpha_u': .7, 'alpha_p': .3,
           'anderson_depth': args.anderson_depth, 'time_step': .25, 'time_steps': 2,
           'max_iterations': args.max_iterations,
           'tolerance': 1e-10, 'linear_tolerance': 1e-13, 'output': str(run)}
    case = run/'input.json'
    case.write_text(json.dumps(cfg, indent=2))
    with (run/'run.log').open('w') as log:
        proc = subprocess.run([str(args.exe.resolve()), str(case)], cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    summary = json.loads((run/'transient_summary.json').read_text())
    history = (run/'time_history.csv').read_text().splitlines()
    checks = {'nonzero_solver_exit': proc.returncode == 3,
              'no_completed_step': summary['steps_completed'] == 0,
              'no_false_convergence': summary['converged'] is False and summary['steady_converged'] is False,
              'no_second_physical_step': not (run/'step_0002').exists() and len(history) == 2}
    if args.anderson_depth:
        with (run/'step_0001/acceleration.csv').open() as stream:
            checks['accepted_acceleration_exercised'] = any(row['accepted'] == '1' for row in csv.DictReader(stream))
    result = {'passed': all(checks.values()), 'checks': checks, 'run': str(run),
              'executable_sha256': hashlib.sha256(args.exe.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
