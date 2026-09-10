"""Decompose wall-gradient consistency error on a manufactured scalar.

F=1-((y-yc(x))^2+(z-zc(x))^2)/R^2 vanishes on the analytic tube.
This is an operator diagnostic, not a flow solution or a wall-shear estimate.
It evaluates the documented linear-fit closure on the actual included cube
centers and separates geometry, one-sided difference, and fit contributions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from check_twisted_mass import read, vector


def exact(points, spec):
    k = 2*np.pi/spec['period']
    angle = k*points[:, 0]
    dy = points[:, 1]-spec['center_y']-spec['amplitude']*np.sin(angle)
    dz = points[:, 2]-spec['center_z']-spec['amplitude']*np.cos(angle)
    yp = spec['amplitude']*k*np.cos(angle)
    zp = -spec['amplitude']*k*np.sin(angle)
    r2 = spec['radius']**2
    value = 1-(dy*dy+dz*dz)/r2
    gradient = np.column_stack((2*(dy*yp+dz*zp), -2*dy, -2*dz))/r2
    return value, gradient


def analyze(root, output, quadratic_radius):
    cfg = json.loads((root/'case.json').read_text(encoding='utf-8-sig'))
    geometry = Path(cfg['embedded_geometry'])
    meta = geometry.with_suffix('.meta.json')
    specdata = json.loads((meta if meta.exists() else geometry).read_text())
    spec = specdata['geometry_spec']
    h, length = specdata['finest_h'], spec['period']
    cells, walls = read(root/'solution.csv'), read(root/'walls.csv')
    if not np.allclose(cells['h'], h, rtol=1e-12, atol=0):
        raise ValueError('This consistency audit requires the uniform reference grid')
    center = vector(cells, ['x', 'y', 'z'])
    wallcenter = vector(walls, ['x', 'y', 'z'])
    shift = np.asarray(specdata.get('reference_translation', [0, 0, 0]))
    center -= shift; wallcenter -= shift
    center[:, 0] %= length; wallcenter[:, 0] %= length
    shape = np.rint(np.asarray(spec['extent'])/h).astype(int)
    lookup = np.full(shape, -1, dtype=np.int32)
    keys = np.rint(center/h-.5).astype(int)
    lookup[tuple(keys.T)] = np.arange(len(cells))
    owner = walls['owner'].astype(int)
    if not np.array_equal(cells['id'].astype(int), np.arange(len(cells))):
        raise ValueError('Cell ordering differs from stored IDs')
    n = vector(walls, ['nx', 'ny', 'nz'])
    offsets = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    design = np.column_stack((offsets, np.ones(27)))
    qo = range(-quadratic_radius, quadratic_radius+1)
    quadratic_offsets = np.array([(i, j, k) for i in qo for j in qo for k in qo])
    fitted = np.empty(len(walls)); affine_error = 0.; condition = 0.
    quadratic_zero = np.empty(len(walls)); quadratic_exact = np.empty(len(walls)); quadratic_condition = 0.
    # Batched normal systems are only 4x4; report conditioning and verify affine
    # exactness. The solver itself retains its QR-based linear fit.
    for start in range(0, len(walls), 1024):
        end = min(start+1024, len(walls))
        ids = owner[start:end]
        q = keys[ids, None, :]+offsets
        q[:, :, 0] %= shape[0]
        validbox = ((q >= 0) & (q < shape)).all(axis=2)
        indices = lookup[tuple(np.clip(q, 0, shape-1).transpose(2, 0, 1))]
        valid = validbox & (indices >= 0)
        samples = center[ids, None, :]+h*offsets
        field, _ = exact(samples.reshape(-1, 3), spec)
        field = field.reshape(len(ids), -1)*valid
        normal = np.einsum('bi,ij,ik->bjk', valid, design, design)
        singular = np.linalg.svd(normal, compute_uv=False)
        if np.any(singular[:, -1] <= 1e-12*singular[:, 0]):
            raise ValueError('Rank deficient wall fit')
        condition = max(condition, float(np.max(singular[:, 0]/singular[:, -1])))
        rhs = field@design
        coeff = np.linalg.solve(normal, rhs[..., None])[..., 0]
        target = wallcenter[start:end]-h*n[start:end]-center[ids]
        target[:, 0] -= np.rint(target[:, 0]/length)*length
        evaluation = np.column_stack((target/h, np.ones(len(ids))))
        fitted[start:end] = np.sum(evaluation*coeff, axis=1)
        affine = np.array([.3, -.7, 1.1, .4])
        affine_rhs = (valid*(design@affine))@design
        computed = np.linalg.solve(normal, affine_rhs[..., None])[..., 0]
        affine_error = max(affine_error, float(np.max(np.abs(computed-affine))))
        # Candidate 3D quadratic fit constrained by its value at the wall.
        # This is evaluated only as an operator diagnostic here.
        qkeys = keys[ids, None, :]+quadratic_offsets
        qkeys[:, :, 0] %= shape[0]
        qbox = ((qkeys >= 0) & (qkeys < shape)).all(axis=2)
        qindices = lookup[tuple(np.clip(qkeys, 0, shape-1).transpose(2, 0, 1))]
        qvalid = qbox & (qindices >= 0)
        qsamples = center[ids, None, :]+h*quadratic_offsets
        qfield, _ = exact(qsamples.reshape(-1, 3), spec)
        qfield = qfield.reshape(len(ids), -1)*qvalid
        r = qsamples-wallcenter[start:end, None, :]
        r[:, :, 0] -= np.rint(r[:, :, 0]/length)*length
        x, y, z = (r/h).transpose(2, 0, 1)
        quadratic = np.stack((x, y, z, x*x, y*y, z*z, x*y, x*z, y*z), axis=2)
        qnormal = np.einsum('bi,bij,bik->bjk', qvalid, quadratic, quadratic)
        singular = np.linalg.svd(qnormal, compute_uv=False)
        if np.any(singular[:, -1] <= 1e-12*singular[:, 0]):
            raise ValueError('Rank deficient quadratic wall prototype')
        quadratic_condition = max(quadratic_condition, float(np.max(singular[:, 0]/singular[:, -1])))
        wall_value, _ = exact(wallcenter[start:end], spec)
        qr0 = np.einsum('bi,bij->bj', qfield, quadratic)
        qr1 = np.einsum('bi,bij->bj', (qfield-wall_value[:, None])*qvalid, quadratic)
        qc = np.linalg.solve(qnormal, np.stack((qr0, qr1), axis=2))
        result = np.sum(qc[:, :3, :]*n[start:end, :, None], axis=1)/h
        quadratic_zero[start:end], quadratic_exact[start:end] = result.T
    value_wall, gradient_wall = exact(wallcenter, spec)
    inward, _ = exact(wallcenter-h*n, spec)
    derivative = np.sum(gradient_wall*n, axis=1)
    approximation = -fitted/h
    geometry_error = -value_wall/h
    difference_error = (value_wall-inward)/h-derivative
    fit_error = (inward-fitted)/h
    error = approximation-derivative
    closure = float(np.max(np.abs(error-geometry_error-difference_error-fit_error)))
    weight = walls['area']; denominator = np.sum(weight*derivative**2)
    def norm(values):
        return {'relative_l2': float(np.sqrt(np.sum(weight*values**2)/denominator)),
                'absolute_linf': float(np.max(np.abs(values))),
                'area_weighted_signed_mean': float(np.average(values, weights=weight))}
    report = {'run': str(root.resolve()), 'h': h, 'wall_faces': len(walls),
              'scope': 'Manufactured scalar operator consistency; not computed flow or physical wall shear',
              'field': '1-((y-yc(x))^2+(z-zc(x))^2)/radius^2',
              'derivative_units': '1/m', 'total_zero_wall_error': norm(error),
              'wall_position_contribution': norm(geometry_error),
              'one_sided_difference_contribution': norm(difference_error),
              'linear_fit_contribution': norm(fit_error),
              'exact_boundary_value_closure_error': norm(difference_error+fit_error),
              'quadratic_zero_boundary_error': norm(quadratic_zero-derivative),
              'quadratic_exact_boundary_error': norm(quadratic_exact-derivative),
              'quadratic_normal_system_max_condition': quadratic_condition,
              'quadratic_stencil_radius_cells': quadratic_radius,
              'affine_coefficient_max_error': affine_error, 'normal_system_max_condition': condition,
              'decomposition_absolute_defect': closure,
              'source_sha256': {str((root/f).resolve()): hashlib.sha256((root/f).read_bytes()).hexdigest()
                                for f in ('case.json', 'solution.csv', 'walls.csv')}}
    if affine_error > 1e-12 or closure > 1e-10:
        raise ValueError('Diagnostic algebra check failed')
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output/'wall_consistency.csv', np.column_stack((wallcenter, n, weight, derivative, approximation,
                  geometry_error, difference_error, fit_error, quadratic_zero, quadratic_exact)), delimiter=',',
               header='x,y,z,nx,ny,nz,area,exact_derivative,closure_derivative,geometry_error,difference_error,fit_error,quadratic_zero_derivative,quadratic_exact_derivative', comments='')
    (output/'consistency.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quadratic-radius', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    reports = [analyze(run, args.output/run.name, args.quadratic_radius) for run in args.runs]
    orders = []
    for a, b in zip(reports, reports[1:]):
        orders.append({'coarse_h': a['h'], 'fine_h': b['h'],
                       'observed_order': float(np.log(a['total_zero_wall_error']['relative_l2']/b['total_zero_wall_error']['relative_l2'])/np.log(a['h']/b['h']))})
    result = {'scope': 'Manufactured wall-gradient consistency only', 'grids': reports, 'orders': orders}
    (args.output/'wall_consistency.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'grids': [{'h': r['h'], 'relative_l2': r['total_zero_wall_error']['relative_l2']} for r in reports], 'orders': orders}, indent=2))


if __name__ == '__main__':
    main()
