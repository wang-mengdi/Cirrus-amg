"""Cross-check the manufactured diagnostic with Aphros' actual Gradient()."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from check_twisted_mass import read, vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--aphros', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ours_path = args.analysis/'wall_consistency.csv'
    reference_path = args.aphros/'tube_b0_wall_consistency.csv'
    ours, reference = read(ours_path), read(reference_path)
    def ordered(data):
        return data[np.lexsort((data['z'], data['y'], data['x']))]
    ours, reference = ordered(ours), ordered(reference)
    if len(ours) != len(reference) or not np.allclose(vector(ours, ['x', 'y', 'z']), vector(reference, ['x', 'y', 'z']), rtol=0, atol=1e-13):
        raise ValueError('Different wall locations')
    quantities = {'zero_boundary_derivative': ours['closure_derivative'],
                  'exact_boundary_derivative': ours['closure_derivative']-ours['geometry_error']}
    errors = {}
    for name, values in quantities.items():
        errors[name] = {'absolute_max': float(np.max(np.abs(values-reference[name]))),
                        'relative_l2': float(np.linalg.norm(values-reference[name])/np.linalg.norm(reference[name]))}
    passed = all(row['relative_l2'] < 1e-10 for row in errors.values())
    result = {'passed': passed, 'relative_l2_limit': 1e-10, 'wall_faces': len(ours), 'errors': errors,
              'scope': 'Independent Aphros Gradient() on a manufactured scalar; no flow state was computed',
              'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ours_path, reference_path)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
