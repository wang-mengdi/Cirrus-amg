"""Verify that invalid replay and Krylov settings are rejected before mesh setup."""
import argparse
import json
from pathlib import Path
import subprocess
from run_twisted_solver import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text())
    base.update(operator_only=True, pressure_replay={'rhs': 'unused.bin', 'repetitions': 1})
    cases = [('dimension_low', {'gpu_krylov_dimension': 1}),
             ('dimension_high', {'gpu_krylov_dimension': 65}),
             ('dimension_fraction', {'gpu_krylov_dimension': 20.5}),
             ('dimension_cpu', {'gpu_krylov_dimension': 40, 'linear_backend': 'cpu'}),
             ('missing_operator_only', {'operator_only': False}),
             ('wrong_pressure_gauge', {'gpu_pressure_gauge': 'pin'}),
             ('wrong_pressure_operator', {'gpu_pressure_operator': 'compact'}),
             ('with_restart', {'restart_checkpoint': 'unused.json'}),
             ('missing_rhs', {'pressure_replay': {'repetitions': 1}}),
             ('zero_repetitions', {'pressure_replay': {'rhs': 'unused.bin', 'repetitions': 0}}),
             ('fraction_repetitions', {'pressure_replay': {'rhs': 'unused.bin', 'repetitions': 1.5}})]
    before = {str(a.exe.resolve()): sha(a.exe), str(a.config.resolve()): sha(a.config), str(Path(__file__)): sha(Path(__file__))}
    rows = []
    for label, changes in cases:
        config = dict(base)
        config.update(changes)
        output = root/(label+'_run')
        config['output'] = str(output)
        case = root/(label+'.json')
        case.write_text(json.dumps(config, indent=2)+'\n')
        process = subprocess.run([str(a.exe.resolve()), str(case)], cwd=Path(__file__).resolve().parents[1],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (root/(label+'.log')).write_text(process.stdout)
        rows.append({'case': label, 'exit_code': process.returncode,
                     'passed': process.returncode == 1 and 'SIMPLE error:' in process.stdout and not output.exists()})
    result = {'scope': __doc__, 'passed': all(r['passed'] for r in rows), 'cases': rows,
              'input_sha256': before, 'inputs_unchanged': all(sha(Path(p)) == h for p, h in before.items())}
    result['passed'] &= result['inputs_unchanged']
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
