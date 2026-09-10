"""Require byte-identical numerical dumps after assembly-storage changes."""
import argparse
import json
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry-format-change', action='store_true', help='Verify every geometry value before comparing different input containers')
    args = parser.parse_args()
    roots = [args.reference.resolve(), args.candidate.resolve()]
    cases = [json.loads((root/'case.json').read_text(encoding='utf-8-sig')) for root in roots]
    for case in cases:
        case.pop('output')
    geometry_equivalence = None
    if args.geometry_format_change:
        import numpy as np
        from twisted_geometry import load_geometry
        geometry = [load_geometry(case.pop('embedded_geometry')) for case in cases]
        keys = ('extent', 'finest_ny', 'finest_h', 'geometry_spec', 'refine_root_tiles')
        if any(geometry[0][key] != geometry[1][key] for key in keys):
            raise ValueError('Different geometry metadata')
        if geometry[0].get('reference_translation_cells', [0,0,0]) != geometry[1].get('reference_translation_cells', [0,0,0]):
            raise ValueError('Different geometry translation')
        geometry_equivalence = {name: bool(np.array_equal(geometry[0][name], geometry[1][name])) for name in ('cells', 'faces', 'walls')}
        if not all(geometry_equivalence.values()):
            raise ValueError('Different geometry values')
    if cases[0] != cases[1]:
        raise ValueError('Assembly check needs identical solver settings and geometry')
    runs = []
    for root in roots:
        completion = json.loads((root/'run_completion.json').read_text())
        if completion['exit_code'] or not completion['executable_unchanged'] or not completion['config_unchanged']:
            raise ValueError('Run did not complete with immutable inputs')
        folders = ([root/f'step_{step:04d}' for step in range(1, cases[0].get('time_steps', 1)+1)]
                   if cases[0].get('time_step', 0) > 0 else [root])
        for folder in folders:
            if any(not (folder/name).is_file() for name in ('solution.csv', 'walls.csv', 'sections.csv', 'metrics.json')):
                raise ValueError('Missing solved physical-step fields')
        for metric in root.rglob('metrics.json'):
            if not json.loads(metric.read_text())['converged']:
                raise ValueError('Unconverged flow input')
        if cases[0].get('time_steps', 1) > 1:
            summary = json.loads((root/'transient_summary.json').read_text())
            if not summary['converged'] or summary['steps_completed'] != cases[0]['time_steps']:
                raise ValueError('Incomplete physical sequence')
        runs.append({'manifest': json.loads((root/'run_manifest.json').read_text()), 'completion': completion})
    def paths(root):
        return {str(p.relative_to(root)): p for p in root.rglob('*') if p.is_file() and
                (p.suffix == '.csv' or p.name in {'metrics.json', 'embedded_operator_checks.json'})}
    files = [paths(root) for root in roots]
    if files[0].keys() != files[1].keys() or not files[0]:
        raise ValueError('Numerical dump sets differ')
    hashes = {name: [sha(files[0][name]), sha(files[1][name])] for name in files[0]}
    different = []
    for name, pair in hashes.items():
        if pair[0] == pair[1]:
            continue
        if args.geometry_format_change and Path(name).name == 'metrics.json':
            metrics = [json.loads(f[name].read_text()) for f in files]
            for metric in metrics:
                metric.pop('geometry_source')
            if metrics[0] == metrics[1]:
                continue
        different.append(name)
    result = {'passed': not different, 'scope': 'Byte-identical final fields, all dumped linear systems/intermediates, residual/acceleration histories, and operator audits',
              'numerical_files_checked': len(hashes), 'different_files': different,
              'geometry_values_identical': geometry_equivalence,
              'runtime': runs, 'sha256_reference_candidate': hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'numerical_files_checked': len(hashes), 'different_files': different}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
