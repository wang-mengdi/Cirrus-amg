"""Measure the existing cut-wall closure on an analytic divergence-free tube field.

This is an operator/geometry accuracy audit, not a solved Navier--Stokes case.
First reproduce actual C++ wall derivatives from each completed flow's velocity;
then apply the same closure to an analytic no-slip field on the prescribed tube.
No physical-flow acceptance threshold or solver output is changed.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from compare_twisted import error, read, vector
from run_twisted_solver import sha


def analytic(x, spec, speed):
    """u = U (1-r^2/R^2) (1, yc', zc'), with its Cartesian Jacobian."""
    k = 2*np.pi/spec['period']
    a, radius = spec['amplitude'], spec['radius']
    phase = k*x[:, 0]
    y = x[:, 1]-spec['center_y']-a*np.sin(phase)
    z = x[:, 2]-spec['center_z']-a*np.cos(phase)
    yp, zp = a*k*np.cos(phase), -a*k*np.sin(phase)
    tangent = np.column_stack((np.ones(len(x)), yp, zp))
    tangent_derivative = np.column_stack((np.zeros(len(x)), -a*k*k*np.sin(phase), -a*k*k*np.cos(phase)))
    f = speed*(1-(y*y+z*z)/radius**2)
    df = 2*speed/radius**2*np.column_stack((y*yp+z*zp, -y, -z))
    jacobian = tangent[:, :, None]*df[:, None, :]
    jacobian[:, :, 0] += f[:, None]*tangent_derivative
    return f[:, None]*tangent, jacobian, (y, z, yp, zp)


def tangential(a, normal):
    return a-np.sum(a*normal, axis=1)[:, None]*normal


def replay(cells, wall, fields, extent):
    """Replay the production 3x3x3 unweighted affine fit in bounded batches.

    The local design uses only small integer offsets. Check conditioning before
    a batched normal-equation solve, and independently verify results against
    the actual C++ QR-generated derivative dump in main().
    """
    centers = vector(cells, 'xyz')
    h = float(cells['h'].min())
    fine = np.flatnonzero(cells['h'] == h)
    keys = np.rint(centers/h-.5).astype(np.int64)
    dims = np.rint(np.array(extent)/h).astype(np.int64)
    if np.max(abs(centers[fine]/h-.5-keys[fine])) > 1e-8:
        raise ValueError('Fine centers are not on the expected Cartesian grid')

    def encode(key):
        return (key[..., 0]*dims[1]+key[..., 1])*dims[2]+key[..., 2]

    encoded = encode(keys[fine])
    order = np.argsort(encoded)
    encoded = encoded[order]
    ordered_ids = fine[order]
    if np.any(np.diff(encoded) <= 0):
        raise ValueError('Duplicate fine-cell keys')
    offsets = np.array(list(itertools.product(range(-1, 2), repeat=3)), dtype=np.int64)
    design = np.column_stack((offsets, np.ones(27)))
    owner = wall['owner'].astype(int)
    if not np.all(cells['h'][owner] == h):
        raise ValueError('A cut wall is not at the finest level')
    normal, location = vector(wall, ['nx', 'ny', 'nz']), vector(wall, 'xyz')
    if np.max(abs(np.linalg.norm(normal, axis=1)-1)) > 1e-10:
        raise ValueError('Wall normals are not unit vectors')
    result = np.empty((len(wall), fields.shape[1]))
    max_condition, min_neighbors, max_neighbors = 0., 27, 0
    for begin in range(0, len(wall), 2048):
        end = min(begin+2048, len(wall))
        ids = owner[begin:end]
        candidates = keys[ids, None, :]+offsets[None, :, :]
        candidates[:, :, 0] %= dims[0]
        valid = np.all((candidates >= 0) & (candidates < dims), axis=2)
        query = encode(candidates)
        index = np.searchsorted(encoded, query)
        clipped = np.minimum(index, len(encoded)-1)
        present = valid & (index < len(encoded)) & (encoded[clipped] == query)
        mask = present.astype(float)
        count = present.sum(axis=1)
        min_neighbors, max_neighbors = min(min_neighbors, int(count.min())), max(max_neighbors, int(count.max()))
        gram = np.einsum('nk,ki,kj->nij', mask, design, design)
        condition = float(np.linalg.cond(gram).max())
        max_condition = max(max_condition, condition)
        if not np.isfinite(condition) or condition > 1e5:
            raise ValueError('Ill-conditioned wall neighborhood requires QR replay')
        rhs = np.einsum('nk,ki,nkj->nij', mask, design, fields[ordered_ids[clipped]])
        coefficients = np.linalg.solve(gram, rhs)
        evaluate = np.column_stack(((location[begin:end]-h*normal[begin:end]-centers[ids])/h, np.ones(end-begin)))
        result[begin:end] = -np.einsum('ni,nij->nj', evaluate, coefficients)/h
    return result, {'h': h, 'minimum_neighbors': min_neighbors, 'maximum_neighbors': max_neighbors,
                    'maximum_normal_equation_condition': max_condition}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=Path, nargs='+', required=True, help='Existing completed physical-step directories')
    parser.add_argument('--centerline-speed', type=float, default=.02)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous wall-accuracy audits')
    if not 0 < args.centerline_speed < 1e6:
        raise ValueError('Positive finite speed required')
    sources = {}

    def keep(path):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return path

    def js(path):
        return json.loads(keep(path).read_text(encoding='utf-8-sig'))

    for name in ('audit_twisted_wall_accuracy.py', 'compare_twisted.py', 'run_twisted_solver.py'):
        keep(Path(__file__).with_name(name))
    records, outputs = [], []
    prescribed = None
    for step in args.steps:
        step = step.resolve()
        config, metrics = js(step/'case.json'), js(step/'metrics.json')
        runtime, done, summary = [js(step.parent/name) for name in ('run_manifest.json', 'run_completion.json', 'transient_summary.json')]
        if done['exit_code'] or not all(done[k] for k in ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
            raise ValueError('Require an unchanged completed native run')
        if not summary['converged'] or not metrics['converged'] or not 1 <= config['physical_step'] <= summary['steps_completed']:
            raise ValueError('The selected field did not complete its physical step')
        for source, value in {runtime['executable']: runtime['executable_sha256'], runtime['config']: runtime['config_sha256'],
                              **runtime['geometry_input_sha256']}.items():
            if sha(keep(source)) != value:
                raise ValueError('Executed input changed: '+source)
        geometry = Path(config['embedded_geometry'])
        meta = geometry.with_suffix('.meta.json')
        metadata = js(meta if meta.exists() else geometry)
        spec = metadata['geometry_spec']
        if prescribed is None:
            prescribed = spec
        if prescribed != spec:
            raise ValueError('Different prescribed analytic tube geometry')
        cells, wall = [read(keep(step/name)) for name in ('solution.csv', 'walls.csv')]
        if not np.array_equal(cells['id'], np.arange(len(cells))) or not np.all(np.isfinite(vector(cells, ['u', 'v', 'w']))):
            raise ValueError('Invalid solution ordering or values')
        translation = np.array(metadata.get('reference_translation', [0, 0, 0]))
        xc, xw = vector(cells, 'xyz')-translation, vector(wall, 'xyz')-translation
        manufactured, jacobian, _ = analytic(xc, spec, args.centerline_speed)
        div = float(np.max(abs(np.trace(jacobian, axis1=1, axis2=2))))
        values = np.column_stack((vector(cells, 'uvw'), manufactured))
        derivative, stencil = replay(cells, wall, values, metadata['extent'])
        native = vector(wall, ['du_dn', 'dv_dn', 'dw_dn'])
        derivative_replay = float(np.max(abs(derivative[:, :3]-native))/max(1., np.max(abs(native))))
        normal, area = vector(wall, ['nx', 'ny', 'nz']), wall['area']
        mu = config['rho']*config['nu']
        native_tau = vector(wall, ['tau_x', 'tau_y', 'tau_z'])
        tau_replay = float(np.max(abs(mu*tangential(derivative[:, :3], normal)-native_tau))/max(1e-30, np.max(abs(native_tau))))
        if derivative_replay >= 1e-11 or tau_replay >= 1e-11:
            raise ValueError('The replay does not match the actual C++ wall output')
        value_wall, jac_wall, (y, z, yp, zp) = analytic(xw, spec, args.centerline_speed)
        radial = np.hypot(y, z)
        cosine, sine = y/radial, z/radial
        slope = yp*cosine+zp*sine
        true_normal = np.column_stack((-slope, cosine, sine))/np.sqrt(1+slope*slope)[:, None]
        exact_wall = xw.copy()
        exact_wall[:, 1] += (spec['radius']-radial)*cosine
        exact_wall[:, 2] += (spec['radius']-radial)*sine
        boundary_value, boundary_jacobian, _ = analytic(exact_wall, spec, args.centerline_speed)
        no_slip = float(np.max(abs(boundary_value)))
        exact_tau = -2*mu*args.centerline_speed/spec['radius']*np.sqrt(1+slope*slope)[:, None]*np.column_stack((np.ones(len(wall)), yp, zp))
        # Independently evaluate the full Newtonian symmetric-gradient stress
        # at the analytic wall, rather than reusing the closed-form expression.
        stress = mu*np.einsum('nij,nj->ni', boundary_jacobian+boundary_jacobian.transpose(0, 2, 1), true_normal)
        stress = tangential(stress, true_normal)
        stress_check = float(np.max(abs(stress-exact_tau))/np.max(abs(exact_tau)))
        if div >= 1e-12 or no_slip >= args.centerline_speed*1e-12 or stress_check >= 1e-12:
            raise ValueError('Analytic field or wall-stress identity failed')
        numeric = mu*tangential(derivative[:, 3:], normal)
        exact_at_discrete_wall = mu*tangential(np.einsum('nij,nj->ni', jac_wall, normal), normal)
        boundary_offset = -mu*tangential(value_wall, normal)/stencil['h']
        fit_derivative = numeric-boundary_offset-exact_at_discrete_wall
        geometry_transport = exact_at_discrete_wall-exact_tau
        decomposition = float(np.max(abs(numeric-exact_tau-(boundary_offset+fit_derivative+geometry_transport))))
        e = error(numeric, exact_tau, area)
        scale = float(np.sqrt(np.average(np.sum(exact_tau*exact_tau, axis=1), weights=area)))

        def scaled_rms(a):
            return float(np.sqrt(np.average(np.sum(a*a, axis=1), weights=area))/scale)

        relative_point_error = np.linalg.norm(numeric-exact_tau, axis=1)/np.linalg.norm(exact_tau, axis=1)
        sort = np.argsort(relative_point_error)
        cumulative_area = np.cumsum(area[sort])/area.sum()
        quantiles = {str(q): float(relative_point_error[sort[min(np.searchsorted(cumulative_area, q), len(sort)-1)]])
                     for q in (.5, .9, .95, .99)}
        angle = np.degrees(np.arccos(np.clip(np.sum(normal*true_normal, axis=1), -1, 1)))
        energy = area*np.sum((numeric-exact_tau)**2, axis=1)
        regions = {}
        for label, mask in [('wall_area_lt_0.01_h2', area < .01*stencil['h']**2),
                            ('owner_volume_lt_0.001_h3', cells['volume'][wall['owner'].astype(int)] < .001*stencil['h']**3),
                            ('normal_angle_gt_5_degrees', angle > 5)]:
            regions[label] = {'faces': int(mask.sum()), 'wall_area_fraction': float(area[mask].sum()/area.sum()),
                              'traction_error_energy_fraction': float(energy[mask].sum()/energy.sum())}
        record = {'step': str(step), 'ny': config['ny'], 'cells': len(cells), 'wall_faces': len(wall), 'stencil': stencil,
                  'physical_step_time': config['physical_time'], 'physical_flow_is_steady': summary['steady_converged'],
                  'native_executable_sha256': runtime['executable_sha256'],
                  'actual_cpp_derivative_replay_scaled_linf': derivative_replay,
                  'actual_cpp_traction_replay_relative_linf': tau_replay,
                  'analytic_divergence_linf': div, 'analytic_wall_velocity_linf': no_slip,
                  'analytic_stress_check_relative_linf': stress_check, 'manufactured_wall_shear': e,
                  'pointwise_relative_error': {'maximum': float(relative_point_error.max()), 'area_weighted_quantiles': quantiles,
                      'wall_area_fraction_above': {str(t): float(area[relative_point_error > t].sum()/area.sum()) for t in (.01, .05, .1, .25)},
                      'scope': 'Descriptive area distribution, not new physical-flow acceptance gates'},
                  'wall_geometry_regions': regions,
                  'error_decomposition': {'definition': 'numeric - analytic-wall traction = boundary_offset + fit_derivative + geometry_transport; vector terms may cancel',
                      'boundary_value_offset_scaled_rms': scaled_rms(boundary_offset),
                      'fit_and_one_sided_derivative_scaled_rms': scaled_rms(fit_derivative),
                      'discrete_point_normal_to_analytic_wall_scaled_rms': scaled_rms(geometry_transport),
                      'sum_replay_absolute_linf': decomposition}}
        records.append(record)
        outputs.append((config['ny'], np.column_stack((wall['face_id'], xw, area, numeric, exact_tau, relative_point_error,
                                                      boundary_offset, fit_derivative, geometry_transport))))
        print(json.dumps({'ny': config['ny'], 'actual_cpp_replay': derivative_replay, 'manufactured_wall_shear_relative_l2': e['relative_l2']}), flush=True)
    refinement = []
    for a, b in zip(records, records[1:]):
        ratio = a['stencil']['h']/b['stencil']['h']
        if ratio <= 1:
            raise ValueError('Pass grids in increasing resolution order')
        ea, eb = a['manufactured_wall_shear']['absolute_l2'], b['manufactured_wall_shear']['absolute_l2']
        refinement.append({'ny': [a['ny'], b['ny']], 'h_ratio': ratio, 'absolute_error_ratio': ea/eb,
                           'observed_order': float(np.log(ea/eb)/np.log(ratio))})
    report = {'scope': __doc__, 'diagnostic_replay_passed': True, 'goal_complete': False,
              'analytic_field': 'U*(1-(Y^2+Z^2)/R^2)*(1,yc_prime,zc_prime), Y=y-yc(x), Z=z-zc(x)',
              'centerline_axial_speed': args.centerline_speed, 'geometry_spec': prescribed,
              'analytic_wall_mapping': 'Same x and polar angle relative to the analytic centerline; compare Cartesian traction vectors at that analytic wall point.',
              'limitations': ['The manufactured field is not the solution of the constant-force physical case; it is an operator accuracy probe.',
                             'The replay uses the production affine wall fit and zero velocity at the discrete planar wall. The analytic field vanishes on the analytic curved wall.',
                             'Geometry and wall-closure errors are both included. Error-decomposition RMS values cannot be added because their vector terms can cancel.',
                             'Transient native fields are used only to verify the wall-operator replay; no steady-flow or independent Aphros match is claimed here.',
                             'Errors are integrated over actual discrete wall-face areas, not fitted at the fixed spatial-convergence probes. Existing physical gates remain separate.'],
              'grids': records, 'refinement': refinement, 'source_sha256': sources}
    if any(sha(Path(p)) != value for p, value in sources.items()):
        raise ValueError('An audit input changed during the calculation')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'wall_accuracy.json').write_text(json.dumps(report, indent=2)+'\n')
    for ny, values in outputs:
        np.savetxt(out/f'wall_accuracy_n{ny}.csv', values, delimiter=',', comments='',
                   header='face_id,x,y,z,area,numeric_tau_x,numeric_tau_y,numeric_tau_z,exact_tau_x,exact_tau_y,exact_tau_z,relative_point_error,boundary_offset_x,boundary_offset_y,boundary_offset_z,fit_derivative_x,fit_derivative_y,fit_derivative_z,geometry_transport_x,geometry_transport_y,geometry_transport_z')


if __name__ == '__main__':
    main()
