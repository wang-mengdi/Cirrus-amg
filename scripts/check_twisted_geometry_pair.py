"""Check exact geometry values across JSON and packed containers."""
import argparse
import json
from pathlib import Path
import numpy as np
from twisted_geometry import load_geometry
from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = [load_geometry(path) for path in (args.reference, args.candidate)]
    keys = ('extent', 'finest_ny', 'finest_h', 'geometry_spec', 'refine_root_tiles')
    checks = {key: data[0][key] == data[1][key] for key in keys}
    checks['translation'] = data[0].get('reference_translation_cells', [0,0,0]) == data[1].get('reference_translation_cells', [0,0,0])
    counts = {}
    for name in ('cells', 'faces', 'walls'):
        arrays = [np.asarray(d[name], dtype=np.float64) for d in data]
        checks[name+'_columns'] = data[0][name+'_columns'] == data[1][name+'_columns']
        checks[name+'_bits'] = all(np.isfinite(a).all() for a in arrays) and np.array_equal(arrays[0].view(np.uint64), arrays[1].view(np.uint64))
        counts[name] = [list(a.shape) for a in arrays]
    report = {'passed': all(checks.values()), 'scope': 'Every float64 bit pattern and grid/refinement metadata; no CFD solution inferred',
              'checks': checks, 'table_shapes': counts,
              'source_sha256': {str(path.resolve()): sha(path) for path in (args.reference, args.candidate)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'table_shapes': counts}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
