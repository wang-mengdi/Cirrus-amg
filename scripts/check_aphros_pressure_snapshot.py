"""Compare actual Aphros pressure equations with final shared-face continuity.

Decimal evaluates the captured double coefficients without double dot-product
rounding. The edge-difference expression is diagnostic only; no field is edited.
"""
import argparse
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from scipy import sparse
from check_twisted_mass import read, vector, check_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh diagnostic report')
    root = args.aphros.resolve()
    config = json.loads((root/'case_manifest.json').read_text())
    runtime = json.loads((root/'run_manifest.json').read_text(encoding='utf-8-sig'))
    snapshot = json.loads((root/'proj_final_b0_pressure_snapshot.json').read_text())
    names = ['proj_final_b0_pressure_rows.csv', 'proj_final_b0_pressure_snapshot.json',
             'proj_final_b0_cells.csv', 'proj_final_b0_faces.csv', 'tube_b0_geometry_faces.csv',
             'tube_b0_time.csv', 'case_manifest.json', 'run_manifest.json', 'a.conf']
    volume_compatibility = runtime['environment'].get('APHROS_TWISTED_VOLUME_COMPATIBILITY') == '1'
    if volume_compatibility:
        names.append('pressure_compatibility.csv')
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    if hashlib.sha256(Path(runtime['executable']).read_bytes()).hexdigest() != runtime['executable_sha256']:
        raise ValueError('Captured-system executable changed')
    if runtime['environment'].get('APHROS_TWISTED_CAPTURE_PRESSURE') != '1':
        raise ValueError('Pressure capture was not enabled')
    if (root/'run_completion.json').exists():
        if json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code'] != 0:
            raise ValueError('Reference run failed')
        completion = 'completed configured reference run'
    else:
        from snapshot_twisted_reference import validate
        validate(root)
        completion = 'verified physical-step checkpoint; whole run not complete'
    rows = read(root/names[0]);cells = read(root/names[2]);faces = read(root/names[3]);geometry = read(root/names[4])
    times = read(root/'tube_b0_time.csv')
    if snapshot['physical_time'] != times[-1]['time'] or snapshot['fluid_rows'] != len(cells):
        raise ValueError('Pressure snapshot and final fields are from different steps')
    for keys in (['x', 'y', 'z'], ['volume', 'p']):
        if not np.array_equal(vector(rows, keys), vector(cells, keys)):
            raise ValueError('Pressure equation and final-field identity failed')
    h = config['spec']['extent'][1]/config['ny'];shape = np.array(config['shape'])
    indices = np.rint(vector(cells, ['x', 'y', 'z'])/h-.5).astype(int)
    lookup = np.full(shape, -1, dtype=int);lookup[tuple(indices.T)] = np.arange(len(cells))
    a = vector(rows, ['a'+str(i) for i in range(7)])
    p = vector(rows, ['p']+['p'+str(i) for i in range(1, 7)])
    adjacent = np.empty((len(rows), 6), dtype=int)
    for q in range(1, 7):
        key = np.rint(vector(rows, [d+str(q) for d in 'xyz'])/h-.5).astype(int)%shape
        adjacent[:, q-1] = lookup[tuple(key.T)]
    nonzero = a[:, 1:] != 0
    if np.any(adjacent[nonzero] < 0):
        raise ValueError('Pressure row couples to an excluded cell')
    owner = np.repeat(np.arange(len(rows))[:, None], 6, axis=1)
    off = sparse.csr_matrix((a[:, 1:][nonzero], (owner[nonzero], adjacent[nonzero])), shape=(len(rows), len(rows)))
    skew = (off-off.T).tocoo()
    coordinates = vector(rows, ['x', 'y', 'z'])
    asymmetry = [{'owner': coordinates[i].tolist(), 'neighbor': coordinates[j].tolist(),
                 'coefficient_ij': float(off[i, j]), 'coefficient_ji': float(off[j, i]),
                 'difference': float(v)} for i, j, v in zip(skew.row, skew.col, skew.data)]
    halo_difference = np.max(abs(p[:, 1:][nonzero]-rows['p'][adjacent[nonzero]]), initial=0)
    exact, edge, row_sum = (np.zeros(len(rows)) for _ in range(3))
    ordinary = rows['b']+a[:, 0]*p[:, 0]
    for q in range(1, 7):
        ordinary += a[:, q]*p[:, q]
    with localcontext() as ctx:
        ctx.prec = 100
        for i in range(len(rows)):
            da = [Decimal.from_float(float(v)) for v in a[i]]
            dp = [Decimal.from_float(float(v)) for v in p[i]]
            b = Decimal.from_float(float(rows['b'][i]))
            exact[i] = float(b+sum(x*y for x, y in zip(da, dp)))
            edge[i] = float(b+sum(da[q]*(dp[q]-dp[0]) for q in range(1, 7)))
            row_sum[i] = float(sum(da))
    # Reconstruct the shared Cartesian face balance, keeping one periodic seam.
    fkey = vector(geometry, ['i', 'j', 'k']).astype(int);axis = geometry['axis'].astype(int)
    keep = ~((axis == 0)&(fkey[:, 0] == shape[0]))
    axis = axis[keep];positive = fkey[keep].copy();negative = positive.copy()
    negative[np.arange(len(axis)), axis] -= 1
    positive %= shape;negative %= shape
    owners, neighbors = lookup[tuple(negative.T)], lookup[tuple(positive.T)]
    if np.any(owners < 0) or np.any(neighbors < 0):
        raise ValueError('Open face touches an excluded cell')
    flux = faces['flux'][keep];terms = [[] for _ in rows]
    for i, j, value in zip(owners, neighbors, flux):
        terms[i].append(float(value));terms[j].append(float(-value))
    net = np.array([math.fsum(v) for v in terms])
    volume = rows['volume'];regular_volume = snapshot['regular_cell_volume']
    worst = np.argsort(abs(net)/volume)[-12:][::-1]
    probes = []
    for i in worst:
        probes.append({'xyz': vector(rows, ['x', 'y', 'z'])[i].tolist(), 'volume': float(volume[i]),
                       'pressure': float(rows['p'][i]), 'is_gauge': bool(rows['is_gauge'][i]),
                       'flux_net': float(net[i]), 'ordinary_equation_residual': float(ordinary[i]),
                       'exact_equation_residual': float(exact[i]), 'edge_equation_residual': float(edge[i]),
                       'diagonal_row_sum': float(row_sum[i]),
                       'diagonal_rounding_term': float(row_sum[i]*rows['p'][i]),
                       'face_divergence': float(net[i]/volume[i]),
                       'face_minus_exact_equation': float(net[i]-exact[i]),
                       'face_minus_edge_equation': float(net[i]-edge[i])})
    mass = check_case(root)
    compatibility = None
    if volume_compatibility:
        trace = read(root/'pressure_compatibility.csv')
        total_volume = math.fsum(volume)
        expected_density = trace['rhs_sum']/total_volume
        expected_old_mean = trace['rhs_sum']/len(rows)/regular_volume
        trace_valid = bool(len(trace) and
                           np.array_equal(trace['call'], np.arange(1, len(trace)+1)) and
                           np.all(trace['cell_count'] == len(rows)) and
                           np.all(trace['regular_cell_volume'] == regular_volume) and
                           np.all(trace['minimum_cell_volume'] == min(volume)) and
                           np.allclose(trace['fluid_volume'], total_volume, rtol=2e-15, atol=0) and
                           np.allclose(trace['compatibility_divergence'], expected_density, rtol=3e-15, atol=0) and
                           np.all(trace['effective_tolerance'] == snapshot['effective_tolerance']) and
                           np.all(abs(expected_old_mean) <= trace['effective_tolerance']) and
                           np.all(abs(expected_density) <= trace['effective_tolerance']))
        compatibility = {'passed': trace_valid, 'calls': len(trace),
                         'scope': 'Captured pressure compatibility trace; a live snapshot may include later-step calls. Original RHS residual and actual final face mass are checked separately',
                         'divergence_linf': float(np.max(abs(expected_density), initial=0)),
                         'maximum_fraction_of_tolerance': float(np.max(abs(expected_density)/trace['effective_tolerance'], initial=0))}
    reported_match = math.isclose(float(abs(ordinary).max()/regular_volume), snapshot['reported_original_residual'], rel_tol=1e-12, abs_tol=1e-30)
    result = {'diagnostic_consistent': bool(reported_match and halo_difference == 0 and
                                          (compatibility is None or compatibility['passed'])),
              'scope': 'Actual last pressure equation versus its final face flux; no solver or field modifications',
              'completion_scope': completion, 'snapshot': snapshot,
              'reported_residual_reproduced': reported_match,
              'pressure_halo_difference_linf': float(halo_difference), 'off_diagonal_asymmetry_entries': skew.nnz,
              'off_diagonal_asymmetry_linf': float(np.max(abs(skew.data), initial=0)),
              'off_diagonal_asymmetry_relative_linf': float(np.max(abs(skew.data), initial=0)/np.max(abs(off.data))),
              'all_asymmetry_on_periodic_x_seam': bool(np.all(abs(coordinates[skew.row, 0]-coordinates[skew.col, 0]) > config['spec']['extent'][0]*.5)),
              'asymmetric_edges': asymmetry,
              'ordinary_equation_residual_regular_volume_linf': float(abs(ordinary).max()/regular_volume),
              'exact_equation_residual_regular_volume_linf': float(abs(exact).max()/regular_volume),
              'exact_equation_residual_actual_volume_linf': float(np.max(abs(exact)/volume)),
              'edge_residual_actual_volume_linf': float(np.max(abs(edge)/volume)),
              'rhs_compatibility_mean': math.fsum(rows['b'])/len(rows),
              'diagonal_row_sum_linf': float(abs(row_sum).max()),
              'mean_diagonal_rounding_term': math.fsum(row_sum*rows['p'])/len(rows),
              'mean_edge_residual': math.fsum(edge)/len(rows),
              'volume_compatibility': compatibility,
              'mass': mass, 'worst_mass_cells': probes,
              'source_sha256': hashes, 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest() != expected for name, expected in hashes.items()):
        raise ValueError('Snapshot changed during diagnosis')
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'worst_mass_cells', 'mass', 'asymmetric_edges')}, indent=2))
    print(json.dumps({'mass_passed': mass['passed'], 'worst_mass_cell': probes[0]}, indent=2))
    if not result['diagnostic_consistent']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
