"""Replay a captured pressure equation with two roundoff compatibility weights.

This is an offline linear-solver experiment, not a replacement flow result.
Both modes use the same captured matrix, pin, starting pressure and LU factors.
Neither the captured files nor Aphros' numerical discretization are changed.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from check_twisted_mass import read, vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing reports; use a fresh output')
    root = args.aphros.resolve()
    diagnostic = json.loads(args.diagnostic.read_text())
    if not diagnostic['diagnostic_consistent']:
        raise ValueError('Captured pressure equation was not validated')
    for name, expected in diagnostic['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Captured input changed: '+name)
    case = json.loads((root/'case_manifest.json').read_text())
    rows = read(root/'proj_final_b0_pressure_rows.csv')
    n = len(rows)
    h = case['spec']['extent'][1]/case['ny']
    shape = np.array(case['shape'])
    index = np.rint(vector(rows, ['x', 'y', 'z'])/h-.5).astype(int)
    lookup = np.full(shape, -1, dtype=int)
    lookup[tuple(index.T)] = np.arange(n)
    neighbors = np.empty((n, 6), dtype=int)
    for q in range(1, 7):
        key = np.rint(vector(rows, [d+str(q) for d in 'xyz'])/h-.5).astype(int)%shape
        neighbors[:, q-1] = lookup[tuple(key.T)]
    a = vector(rows, ['a'+str(q) for q in range(7)])
    mask = a[:, 1:] != 0
    owner = np.broadcast_to(np.arange(n)[:, None], neighbors.shape)
    if np.any(neighbors[mask] < 0):
        raise ValueError('Equation references an excluded cell')
    off = sparse.coo_matrix((a[:, 1:][mask], (owner[mask], neighbors[mask])), shape=(n, n)).tocsc()
    original = off+sparse.diags(a[:, 0])
    gauges = np.flatnonzero(rows['is_gauge'])
    if len(gauges) != 1:
        raise ValueError('Require one captured pressure pin')
    gauge = gauges[0]
    coo = original.tocoo()
    keep = (coo.row != gauge)&(coo.col != gauge)
    pinned = sparse.coo_matrix((np.r_[coo.data[keep], 1.],
                               (np.r_[coo.row[keep], gauge], np.r_[coo.col[keep], gauge])), shape=(n, n)).tocsc()
    factor = splu(pinned)
    volume = rows['volume']
    rhs = -rows['b']

    def measures(x):
        exact = np.empty(n)
        with localcontext() as context:
            context.prec = 80
            for i in range(n):
                value = Decimal.from_float(float(rows['b'][i]))+Decimal.from_float(float(a[i, 0]))*Decimal.from_float(float(x[i]))
                for q in range(6):
                    if a[i, q+1]:
                        value += Decimal.from_float(float(a[i, q+1]))*Decimal.from_float(float(x[neighbors[i, q]]))
                exact[i] = float(value)
        return {'equation_residual_actual_volume_linf': float(np.max(abs(exact)/volume)),
                'equation_residual_regular_volume_linf': float(np.max(abs(exact))/(h**3)),
                'pressure_change_linf': float(np.max(abs(x-rows['p']))),
                'worst_cell': vector(rows, ['x', 'y', 'z'])[np.argmax(abs(exact)/volume)].tolist()}

    results = {}
    for mode in ('cell_count', 'fluid_volume'):
        weights = np.full(n, 1/n) if mode == 'cell_count' else volume/math.fsum(volume)
        x = rows['p'].copy()
        trace = []
        for iteration in range(6):
            residual = rhs-original@x
            total = math.fsum(residual)
            correction = total*weights
            b = residual-correction
            b[gauge] = 0
            delta = factor.solve(b)
            trace.append({'iteration': iteration+1, 'residual_sum': total,
                          'compatibility_divergence_linf': float(np.max(abs(correction)/volume)),
                          'correction_pressure_linf': float(np.max(abs(delta)))})
            x += delta
        results[mode] = {'trace': trace, **measures(x)}
    result = {'scope': __doc__, 'starting_pressure': measures(rows['p']), 'modes': results,
              'captured_input_sha256': diagnostic['source_sha256'],
              'diagnostic_sha256': hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('captured_input_sha256',)}, indent=2))


if __name__ == '__main__':
    main()
