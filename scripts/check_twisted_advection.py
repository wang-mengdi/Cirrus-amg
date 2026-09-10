"""Check actual C++ face interpolation against Aphros' pre-assembly FOU dump.

The first SIMPLE update starts from rest and supplies effectively identical
velocity and flux inputs. Compare convection at the second update, before
pressure corrections can hide a local discrepancy. A separate diagnostic
tests the observed upstream zero-halo-flux behavior; it never makes the main
advection gate pass. Only the actual face operator comparison can pass it.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix


def read(path):
    data = np.genfromtxt(path, delimiter=',', names=True, ndmin=1)
    if not len(data) or any(not np.isfinite(data[k]).all() for k in data.dtype.names):
        raise ValueError(f'Empty or nonfinite dump: {path}')
    return data


def norms(delta, reference):
    return {'absolute_max': float(np.max(np.abs(delta))),
            'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(reference), 1e-300))}


def check(ours, aphros):
    case = json.loads((ours/'case.json').read_text(encoding='utf-8-sig'))
    spec = json.loads((aphros/'case_manifest.json').read_text())
    length = spec['spec']['extent'][0]
    if case.get('adaptive') or not case.get('convection'):
        raise ValueError('This stage audit requires a uniform convection case')
    cells = read(ours/'step_0001/iter_1/cells.csv')
    faces = read(ours/'step_0001/iter_1/faces.csv')
    terms = read(ours/'advection_face_interpolation.csv')
    mix = coo_matrix((terms['value'], (terms['row'].astype(int), terms['column'].astype(int))),
                     shape=(len(faces), len(faces))).tocsr()
    def key(row):
        return (int(row['axis']), round(row['x'] % length, 12), round(row['y'], 12), round(row['z'], 12))
    active = faces['neighbor'] >= 0
    lookup = {key(row): i for i, row in enumerate(faces) if row['neighbor'] >= 0}
    if len(lookup) != np.count_nonzero(active):
        raise ValueError('Duplicate native shared faces')
    owners, neighbors = faces['owner'].astype(int), faces['neighbor'].astype(int)
    flux = faces['flux']
    cartesian_lookup = {}
    for i in np.flatnonzero(active):
        axis = int(faces['axis'][i])
        x = [float(cells[n][owners[i]]) for n in ('x', 'y', 'z')]
        x[axis] += .5*cells['h'][owners[i]]
        cartesian_lookup[(axis, round(x[0] % length, 12), round(x[1], 12), round(x[2], 12))] = i
    output = {}
    seam_terms = [(target, source, float(mix[target, source]))
                  for target in range(len(faces)) for source in mix[target].indices
                  if abs(faces['x'][target]-faces['x'][source]) > length/2]
    if not seam_terms:
        raise ValueError('Audit lacks a cut interpolation stencil crossing the periodic seam')
    for d, name in enumerate(('u', 'v', 'w')):
        reference = read(aphros/f'adv_velocity_{"xyz"[d]}_1_b0_faces.csv')
        indices = np.array([lookup[key(row)] for row in reference])
        if len(np.unique(indices)) != np.count_nonzero(active):
            raise ValueError('Incomplete reference face coverage')
        if not np.allclose(faces['area'][indices], reference['area'], rtol=1e-12, atol=0):
            raise ValueError('Face apertures differ')
        upwind = np.zeros(len(faces))
        central = .5*(cells[name][owners]+cells[name][np.maximum(neighbors, 0)])
        upwind[active] = np.where(flux[active] > 0, cells[name][owners[active]],
                                  np.where(flux[active] < 0, cells[name][neighbors[active]], central[active]))
        interpolated = mix @ upwind
        stale_halo = interpolated.copy()
        for target, source, weight in seam_terms:
            stale_halo[target] += weight*(central[source]-upwind[source])
        expected = interpolated[indices]*flux[indices]
        delta = expected-reference['evaluated_flux']
        sources = read(aphros/f'adv_velocity_{"xyz"[d]}_1_b0_sources.csv')
        halo_rows = [j for j, row in enumerate(sources)
                     if (row['x'] < 0 or row['x'] > length or
                         (row['axis'] != 0 and row['x'] >= length)) and key(row) in cartesian_lookup]
        if not halo_rows:
            raise ValueError('Supporting face dump lacks periodic fluid halo faces')
        halo = sources[halo_rows]
        halo_indices = np.array([cartesian_lookup[key(row)] for row in halo])
        halo_flux = norms(halo['volume_flux']-flux[halo_indices], flux[halo_indices])
        worst = int(np.argmax(np.abs(delta)))
        output[name] = {
            'volume_flux_input': norms(flux[indices]-reference['volume_flux'], reference['volume_flux']),
            'actual_advection': norms(delta, reference['evaluated_flux']),
            'zero_halo_flux_hypothesis': norms(stale_halo[indices]*flux[indices]-reference['evaluated_flux'], reference['evaluated_flux']),
            'support_halo_faces': len(halo_rows),
            'support_halo_zero_flux_count': int(np.count_nonzero(halo['volume_flux'] == 0)),
            'support_halo_flux_periodicity': halo_flux,
            'worst_face': {'id': int(indices[worst]), 'axis': int(reference['axis'][worst]),
                           'center': [float(reference[n][worst]) for n in ('x', 'y', 'z')]}}
    checks = {name: row['actual_advection']['relative_l2'] < 1e-10 and
                    row['volume_flux_input']['relative_l2'] < 1e-10 and
                    row['support_halo_flux_periodicity']['relative_l2'] < 1e-10
              for name, row in output.items()}
    return {'passed': all(checks.values()), 'checks': checks, 'relative_l2_limit': 1e-10,
            'scope': 'Second SIMPLE iteration face convection; first update from rest',
            'periodic_seam_interpolation_terms': len(seam_terms), 'components': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours', type=Path, required=True)
    parser.add_argument('--aphros', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.ours, args.aphros)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
