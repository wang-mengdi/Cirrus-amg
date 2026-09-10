"""Audit a linear-backend change on otherwise identical Aphros flow cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_twisted import read, ordered, vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = (args.reference, args.candidate)
    configs = [json.loads((p/'case_manifest.json').read_text()) for p in roots]
    if configs[0] != configs[1]:
        raise ValueError('Backend audit requires identical case manifests')
    runtime = [json.loads((p/'run_manifest.json').read_text(encoding='utf-8-sig')) for p in roots]
    for p, r in zip(roots, runtime):
        if hashlib.sha256((p/'a.conf').read_bytes()).hexdigest() != r['config_sha256']:
            raise ValueError('Executed configuration differs from audited file')
        if r['environment']['APHROS_TWISTED_FIX_FLUX_HALO'] != '1':
            raise ValueError('Both cases must use the same documented halo repair')
        completion = json.loads((p/'run_completion.json').read_text(encoding='utf-8-sig'))
        if completion['exit_code'] or 'End of simulation' not in (p/'run.log').read_text():
            raise ValueError('Incomplete reference run')
    checks, errors, hashes = {}, {}, {}
    families = [('cells', 'simple_final_b0_cells.csv', ('u', 'v', 'w', 'p'), 1e-11),
                ('walls', 'tube_final_b0_walls.csv', ('tau_x', 'tau_y', 'tau_z'), 1e-11),
                ('faces', 'simple_final_b0_faces.csv', ('flux',), 1e-15)]
    for label, filename, fields, limit in families:
        # Axis precedes coordinates when several faces share a location.
        arrays = [read(p/filename) for p in roots]
        arrays = [a[np.lexsort((a['z'], a['y'], a['x'], a['axis']))] if label == 'faces' else ordered(a) for a in arrays]
        a, b = arrays
        if len(a) != len(b) or not np.allclose(vector(a, ['x', 'y', 'z']), vector(b, ['x', 'y', 'z']), rtol=0, atol=1e-13):
            raise ValueError('Backend comparison geometry differs')
        errors[label] = {}
        for name in fields:
            delta = b[name]-a[name]
            if name == 'p':
                delta -= np.average(delta, weights=a['volume'])
            value = float(np.max(np.abs(delta)))
            errors[label][name] = value
            checks[f'{label}_{name}'] = value < limit
        for p in roots:
            hashes[str((p/filename).resolve())] = hashlib.sha256((p/filename).read_bytes()).hexdigest()
    result = {'passed': all(checks.values()), 'checks': checks, 'absolute_max_errors': errors,
              'scope': 'Aphros direct versus AMG linear backend; same spatial equations and halo repair',
              'runtime': runtime, 'source_sha256': hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
