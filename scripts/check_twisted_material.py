"""Compare immutable setup-time material/probe dumps, not completed flow fields."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_twisted_solver import sha


def read(path):
    return np.genfromtxt(path, delimiter=',', names=True, ndmin=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assembled', action='store_true', help='Also require the actual assembled cell pressure and viscosity operators')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier audit evidence')
    roots = [args.native.resolve(), args.reference.resolve()]
    hashes = {}
    def record(path):
        hashes[str(path)] = sha(path)
        return path
    runs = [json.loads(record(p/'run_manifest.json').read_text(encoding='utf-8-sig')) for p in roots]
    for run in runs:
        if sha(Path(run['executable'])) != run['executable_sha256']:
            raise ValueError('An audit executable has changed')
    completion = json.loads(record(roots[1]/'run_completion.json').read_text())
    if completion['exit_code'] != 0 or not completion['executable_unchanged'] or not completion['config_unchanged']:
        raise ValueError('Original Aphros audit did not complete unchanged')
    build_path = Path(runs[1]['executable']).parent/'build_manifest.json'
    build = json.loads(record(build_path).read_text())
    if sha(Path(build['library'])) != build['library_sha256']:
        raise ValueError('Original Aphros library has changed')
    case = json.loads(record(roots[0]/'case.json').read_text())
    if case['rho'] != 1 or case['force'] != [1, 0, 0]:
        raise ValueError('Prescribed acceleration probe requires rho=1 and force=(1,0,0)')
    method = json.loads(record(roots[0]/'projection_method.json').read_text())
    reference = json.loads(record(roots[1]/'material_audit.json').read_text())
    if not reference['passed']:
        raise ValueError('Original material relation check failed')
    h = .125/case['ny']
    checks = {}
    def compare(name, fields, grouped, tolerance):
        tables = [read(record(p/name)) for p in roots]
        a, b = tables
        if len(a) == 0 or len(a) != len(b):
            raise ValueError('Expected matching nonempty exceptional-face probes')
        matched = np.empty(len(a), dtype=int)
        max_distance = 0.
        for i, row in enumerate(a):
            mask = np.ones(len(b), dtype=bool)
            for key in grouped:
                mask &= b[key] == row[key]
            ids = np.flatnonzero(mask)
            if not len(ids):
                raise ValueError('Unmatched face orientation or cell side')
            distance = np.sqrt(sum((b[k][ids]-row[k])**2 for k in ('x', 'y', 'z')))
            j = int(np.argmin(distance)); matched[i] = ids[j]
            max_distance = max(max_distance, float(distance[j]))
        if len(set(matched)) != len(a) or max_distance > h*1e-10:
            raise ValueError('Probe geometry differs')
        b = b[matched]
        results = {}
        for key in fields:
            if not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
                raise ValueError('Nonfinite material probe')
            error = float(np.max(np.abs(a[key]-b[key]))/max(float(np.max(np.abs(b[key]))), 1e-300))
            results[key] = {'relative_linf': error, 'limit': tolerance[key], 'passed': error <= tolerance[key]}
        checks[name] = {'rows': len(a), 'maximum_coordinate_distance': max_distance, 'fields': results}
    face_fields = ['area', 'weight', 'mu', 'rho', 'compact_gradient', 'full_gradient', 'pressure_flux', 'viscosity_flux', 'face_acceleration']
    compare('material_faces.csv', face_fields, ['axis'], {k: 1e-12 if k in ('area', 'weight', 'mu', 'rho') else 1e-10 for k in face_fields})
    cell_fields = ['accel_x', 'accel_y', 'accel_z']
    if args.assembled:
        cell_fields += ['pressure_operator', 'viscosity_operator']
    compare('material_cells.csv', cell_fields, ['axis', 'side'], {k: 1e-10 for k in cell_fields})
    count = checks['material_faces.csv']['rows']
    if method['material_weighted_open_faces'] != count or reference['changed_open_faces'] != count:
        raise ValueError('Exceptional face coverage is incomplete')
    passed = all(f['passed'] for test in checks.values() for f in test['fields'].values())
    report = {'passed': passed, 'scope': 'Setup-time material and prescribed-scalar operators at every exceptional open face; no flow convergence claim',
              'assembled_cell_operators_checked': args.assembled,
              'native_flow_completion_required': False, 'native_runtime_state_note': 'Only immutable constructor diagnostics are inspected; active or failed flow does not become a completed result',
              'checks': checks, 'source_sha256': hashes, 'aphros_library_sha256': build['library_sha256']}
    if any(sha(Path(p)) != value for p, value in hashes.items()):
        raise ValueError('A setup diagnostic changed during comparison')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': passed, 'exceptional_faces': count}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
