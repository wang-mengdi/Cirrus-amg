"""Compare accelerated and ordinary Cirrus time sequences at every physical step.

SIMPLE and projection are supported, but the two inputs must use the same method.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from compare_twisted import vector, error


def read(path):
    """Load every numeric CSV field without genfromtxt's large token lists."""
    with Path(path).open(encoding='utf-8-sig') as stream:
        names=next(csv.reader(stream),[])
        if not names or len(set(names))!=len(names) or any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',name) for name in names):
            raise ValueError('Invalid native numeric CSV header: '+str(path))
        data=np.loadtxt(stream,delimiter=',',comments=None,
                        dtype=[(name,np.float64) for name in names],ndmin=1)
    if not len(data) or any(not np.isfinite(data[name]).all() for name in names):
        raise ValueError('Nonfinite or empty CSV: '+str(path))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accelerated', '--candidate', type=Path, required=True)
    parser.add_argument('--ordinary', '--reference', type=Path, required=True)
    parser.add_argument('--linear-backend-pair',action='store_true',help='Compare GPU versus CPU with identical outer algorithm and acceleration')
    parser.add_argument('--implicit-viscosity-pair',action='store_true',
                        help='Compare GPU full implicit viscosity with CPU deferred iteration at the same converged discrete fixed point')
    parser.add_argument('--orthogonalization-pair', action='store_true',
                        help='Compare CGS2 versus MGS2 using the same native GPU executable and identical physical settings')
    parser.add_argument('--native-build-pair', action='store_true',
                        help='Compare two native GPU builds with identical physical settings; preserve both compiled source snapshots')
    parser.add_argument('--restart-pair', action='store_true',
                        help='With --native-build-pair, compare a verified restarted candidate against an uninterrupted reference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, help='Compare only this completed prefix; never counts as a full time-sequence check')
    args = parser.parse_args()
    native_pair = args.orthogonalization_pair or args.native_build_pair
    if args.output.exists():
        raise ValueError('Preserve earlier time-pair checks')
    if native_pair and (args.linear_backend_pair or args.implicit_viscosity_pair or args.steps is not None or (args.orthogonalization_pair and args.native_build_pair)):
        raise ValueError('Native pair requires two complete trajectories and one explicit comparison scope')
    if args.restart_pair and not args.native_build_pair:
        raise ValueError('Restart comparison requires a complete native build pair')
    if args.implicit_viscosity_pair and not args.linear_backend_pair:
        raise ValueError('Full implicit viscosity comparison requires the GPU/CPU backend checks')
    roots = [args.accelerated.resolve(), args.ordinary.resolve()]
    configs = [json.loads((p/'case.json').read_text(encoding='utf-8-sig')) for p in roots]
    if args.restart_pair and (not configs[0].get('restart_checkpoint') or configs[1].get('restart_checkpoint')):
        raise ValueError('Expected restarted candidate versus uninterrupted reference')
    if not args.restart_pair and any(c.get('restart_checkpoint') for c in configs):
        raise ValueError('Restarted time sequences require an explicit restart-pair comparison')
    fluid_solver=configs[0].get('fluid_solver','simple')
    if configs[1].get('fluid_solver','simple')!=fluid_solver:
        raise ValueError('Different fluid solvers')
    for key in ('rho', 'nu', 'force', 'convection', 'convection_scheme', 'adaptive',
                'alpha_u', 'alpha_p', 'time_step', 'time_steps', 'tolerance', 'linear_tolerance'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Different time-discrete problem: {key}')
    geometry_hashes = [hashlib.sha256(Path(c['embedded_geometry']).read_bytes()).hexdigest() for c in configs]
    if geometry_hashes[0] != geometry_hashes[1]:
        raise ValueError('Different geometry')
    if args.linear_backend_pair:
        if configs[0].get('linear_backend','cpu')!='native_gpu' or configs[1].get('linear_backend','cpu')!='cpu':
            raise ValueError('Expected native GPU versus CPU backend')
        ignored=('output','linear_backend','gpu_preconditioner','gpu_pressure_gauge','gpu_pressure_operator','gpu_orthogonalization')
        if args.implicit_viscosity_pair:
            if [c.get('gpu_viscosity_operator','compact') for c in configs]!=['full','compact']:
                raise ValueError('Expected full GPU viscosity versus compact/deferred CPU viscosity')
            ignored+=('gpu_viscosity_operator',)
        normalized=[{k:v for k,v in c.items() if k not in ignored} for c in configs]
        if normalized[0]!=normalized[1]:raise ValueError('Backend comparison must retain identical outer algorithm and parameters')
    elif native_pair:
        if any(c.get('linear_backend')!='native_gpu' or c.get('gpu_preconditioner')!='native_amg' for c in configs):
            raise ValueError('Orthogonalization comparison requires native GPU AMG on both sides')
        if args.orthogonalization_pair and [c.get('gpu_orthogonalization','mgs2') for c in configs]!=['cgs2','mgs2']:
            raise ValueError('Expected candidate CGS2 and reference MGS2')
        if any(c.get('gpu_orthogonalization','mgs2') not in ('mgs2','cgs2') for c in configs):
            raise ValueError('Unknown native orthogonalization')
        ignored=('output','gpu_orthogonalization')+(('restart_checkpoint',) if args.restart_pair else ())
        normalized=[{k:v for k,v in c.items() if k not in ignored} for c in configs]
        if normalized[0]!=normalized[1]:raise ValueError('Only orthogonalization and output may change')
        manifests=[json.loads((r/'run_manifest.json').read_text()) for r in roots]
        if args.orthogonalization_pair and manifests[0]['executable_sha256']!=manifests[1]['executable_sha256']:
            raise ValueError('Orthogonalization pair must use the same executable')
    elif configs[0]['anderson_depth'] <= 0 or configs[1]['anderson_depth'] != 0:
        raise ValueError('Expected accelerated versus ordinary solver')
    total_steps = configs[0]['time_steps']
    compared_steps = total_steps if args.steps is None else args.steps
    if not 1 <= compared_steps <= total_steps:
        raise ValueError('Requested prefix must be within the configured time sequence')
    complete_sequence = compared_steps == total_steps
    completion_evidence = {}
    for root in roots:
        if complete_sequence:
            summary = json.loads((root/'transient_summary.json').read_text())
            if not summary['converged'] or summary['steps_completed'] != total_steps:
                raise ValueError('Incomplete time sequence')
            if args.linear_backend_pair or native_pair:
                completion=json.loads((root/'run_completion.json').read_text())
                run=json.loads((root/'run_manifest.json').read_text())
                if completion['exit_code']!=0 or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
                    raise ValueError('Backend run completion or provenance failed')
                if hashlib.sha256(Path(run['executable']).read_bytes()).hexdigest()!=run['executable_sha256']:
                    raise ValueError('Recorded backend executable has changed')
        with (root/'time_history.csv').open() as stream:
            history = list(csv.DictReader(stream))[:compared_steps]
        if len(history) != compared_steps or any(
                row['inner_converged'] != 'true' or int(row['step']) != step or
                not np.isclose(float(row['time']), step*configs[0]['time_step'], rtol=1e-13, atol=0)
                for step, row in enumerate(history, 1)):
            raise ValueError('Requested physical steps are not recorded as complete')
        completion_evidence[str(root)] = history
    rows, hashes = [], {}
    for step in range(1, compared_steps+1):
        folders = [root/f'step_{step:04d}' for root in roots]
        metrics = [json.loads((p/'metrics.json').read_text()) for p in folders]
        if not all(m['converged'] for m in metrics):
            raise ValueError('Unconverged physical step')
        if fluid_solver=='proj' and configs[0]['anderson_depth'] and metrics[0].get('complete_inner_fixed_point_residual',float('inf'))>=configs[0]['tolerance']:
            raise ValueError('Accelerated projection did not satisfy all inner fixed-point blocks')
        if native_pair and any(m.get('complete_inner_fixed_point_residual',float('inf'))>=min(c['tolerance'],1e-8) for m,c in zip(metrics,configs)):
            raise ValueError('An orthogonalization trajectory failed the complete fixed point')
        cells = [read(p/'solution.csv') for p in folders]
        walls = [read(p/'walls.csv') for p in folders]
        faces = [read(p/f'iter_{m["iterations"]}/faces.csv') for p, m in zip(folders, metrics)]
        for data, columns in ((cells, ['id', 'x', 'y', 'z', 'volume']),
                              (walls, ['face_id', 'x', 'y', 'z', 'area']),
                              (faces, ['id', 'owner', 'neighbor', 'area'])):
            if not np.array_equal(vector(data[0], columns), vector(data[1], columns)):
                raise ValueError('Different grid ordering or geometry')
        volume = cells[0]['volume']
        pressure = [c['p']-np.average(c['p'], weights=volume) for c in cells]
        cut = np.isin(cells[0]['id'], walls[0]['owner'])
        velocities = [vector(c, ['u', 'v', 'w']) for c in cells]
        quantities = {'velocity': error(*velocities, volume),
                      'pressure': error(*pressure, volume),
                      'cut_cell_velocity': error(velocities[0][cut], velocities[1][cut], volume[cut]),
                      'wall_shear': error(*[vector(w, ['tau_x', 'tau_y', 'tau_z']) for w in walls], walls[0]['area']),
                      'shared_face_flux': error(faces[0]['flux'], faces[1]['flux'], np.ones(len(faces[0])))}
        passed = all(q['relative_l2'] < 1e-6 for q in quantities.values())
        mass=[]
        if args.linear_backend_pair or native_pair:
            for folder,cell,u in zip(folders,cells,velocities):
                geometry=read(folder/'mesh_faces.csv');q=read(folder/'flux.csv')['flux'];inner=geometry['neighbor']>=0
                if not np.array_equal(geometry['id'],np.arange(len(q))) or np.any(q[~inner]!=0):
                    raise ValueError('Invalid face IDs or nonzero impermeable-wall flux')
                net=np.zeros(len(cell));np.add.at(net,geometry['owner'].astype(int),q)
                np.add.at(net,geometry['neighbor'][inner].astype(int),-q[inner])
                shape=json.loads(Path(configs[0]['embedded_geometry']).read_text())
                rate=np.linalg.norm(u,axis=1).max()/shape['extent'][1]
                sections=read(folder/'sections.csv')['volume_flux'];through=float(np.mean(sections))
                item={'divergence_relative_linf':float(np.max(abs(net)/cell['volume'])/rate),
                      'section_flux_relative_spread':float(np.ptp(sections)/abs(through)),
                      'global_absolute_cell_flux_over_throughflow':float(np.sum(abs(net))/abs(through))}
                item['passed']=item['divergence_relative_linf']<1e-7 and item['section_flux_relative_spread']<1e-8 and item['global_absolute_cell_flux_over_throughflow']<1e-8
                mass.append(item)
            passed=passed and all(m['passed'] for m in mass)
        accepted_accelerations=0
        if configs[0]['anderson_depth']:
            with (folders[0]/'acceleration.csv').open() as stream:
                accepted_accelerations = sum(int(row['accepted']) for row in csv.DictReader(stream))
        rows.append({'step': step, 'time': step*configs[0]['time_step'], 'passed': passed,
                     'iterations_accelerated_ordinary': [m['iterations'] for m in metrics],
                     'accepted_accelerations': accepted_accelerations, **quantities})
        if args.linear_backend_pair or native_pair:rows[-1]['independent_mass_candidate_reference']=mass
        for folder, m in zip(folders, metrics):
            for path in [folder/'solution.csv', folder/'walls.csv', folder/'metrics.json',
                         folder/f'iter_{m["iterations"]}/faces.csv']:
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'passed': all(row['passed'] for row in rows), 'relative_l2_limit': 1e-6,
              'scope': ('Internal acceleration check at every physical step' if complete_sequence else
                        'Internal acceleration check of a completed prefix only; later steps remain unverified')+
                       '; independent Aphros validation is separate',
              'complete_sequence_checked': complete_sequence, 'configured_steps': total_steps,
              'completion_evidence': completion_evidence,
              'steps': rows, 'source_sha256': hashes}
    if args.linear_backend_pair:
        with (roots[0]/'gpu_linear.csv').open() as stream:trace=list(csv.DictReader(stream))
        method=json.loads((roots[0]/'projection_method.json').read_text())
        native_amg=configs[0].get('gpu_preconditioner','jacobi')=='native_amg'
        expected='native_gpu_fgmres_amg' if native_amg else 'native_gpu_pcg_jacobi'
        if method['linear_backend']!=expected or not len(trace):raise ValueError('No actual GPU linear solve evidence')
        orthogonalization=configs[0].get('gpu_orthogonalization','mgs2')
        if orthogonalization not in ('mgs2','cgs2') or method.get('gpu_orthogonalization','mgs2')!=orthogonalization:
            raise ValueError('Actual GPU orthogonalization differs from configuration')
        if native_amg and method.get('native_amg_levels',0)<2:raise ValueError('No native multilevel hierarchy evidence')
        if set(row['operator'] for row in trace)!={'pressure','diffusion'}:raise ValueError('Missing pressure or viscosity GPU solve')
        r=np.array([float(row['true_relative_residual']) for row in trace])
        if not np.isfinite(r).all() or np.max(r)>configs[0]['linear_tolerance']:raise ValueError('GPU true residual exceeded the requested linear tolerance')
        gauge=configs[0].get('gpu_pressure_gauge','pin')
        if method.get('pressure_gauge','pin')!=gauge:raise ValueError('Actual GPU pressure gauge differs from configuration')
        pressure_operator=configs[0].get('gpu_pressure_operator','compact')
        if pressure_operator not in ('compact','full') or method.get('pressure_operator','compact')!=pressure_operator:
            raise ValueError('Actual GPU pressure operator differs from configuration')
        if pressure_operator=='full' and (not native_amg or gauge!='mean_zero'):
            raise ValueError('Full pressure operator requires native AMG and mean-zero pressure')
        viscosity_operator=configs[0].get('gpu_viscosity_operator','compact')
        if viscosity_operator not in ('compact','full') or method.get('viscosity_operator','compact')!=viscosity_operator:
            raise ValueError('Actual GPU viscosity operator differs from configuration')
        if viscosity_operator=='full' and (not native_amg or not args.implicit_viscosity_pair):
            raise ValueError('Full viscosity comparison requires native AMG and explicit implicit-viscosity-pair scope')
        if pressure_operator=='full' or viscosity_operator=='full':
            topology=read(roots[0]/'mesh_cells.csv');connectivity=read(roots[0]/'mesh_faces.csv')
            inner=connectivity['neighbor']>=0
            owners=connectivity['owner'][inner].astype(int);neighbors=connectivity['neighbor'][inner].astype(int)
            count=int(np.count_nonzero(topology['level'][owners]!=topology['level'][neighbors]))
            if pressure_operator=='full' and method.get('full_pressure_interface_faces')!=count:
                raise ValueError('Full GPU pressure stencil count differs from actual fluid interfaces')
            if viscosity_operator=='full':
                all_owners=connectivity['owner'].astype(int)
                expected_custom=~inner
                h=topology['h'][all_owners]
                expected_custom|=abs(connectivity['area']-h*h)>1e-12*h*h
                expected_custom[inner]|=topology['level'][owners]!=topology['level'][neighbors]
                if method.get('full_viscosity_faces')!=int(np.count_nonzero(expected_custom)):
                    raise ValueError('Full GPU viscosity stencil count differs from actual cut, wall and coarse/fine faces')
            for name in ('mesh_cells.csv','mesh_faces.csv'):
                p=roots[0]/name;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        compatibility=np.array([float(row.get('compatibility_relative_l2',0.)) for row in trace])
        if gauge=='mean_zero' and any('compatibility_relative_l2' not in row for row in trace):
            raise ValueError('Missing pressure compatibility evidence')
        if not np.isfinite(compatibility).all() or np.min(compatibility)<0 or np.max(compatibility)>configs[0]['linear_tolerance']:
            raise ValueError('Pressure compatibility correction exceeded the requested linear tolerance')
        result['scope']=('Complete time sequence' if complete_sequence else 'Completed prefix only')+'; GPU versus CPU linear backend with identical outer algorithm; independent Aphros and physical convergence checks remain separate'
        if args.implicit_viscosity_pair:
            result['scope']=('Complete time sequence' if complete_sequence else 'Completed prefix only')+'; GPU full implicit viscosity versus CPU deferred iteration, same converged discrete fixed point; independent Aphros and physical convergence checks remain separate'
        result['gpu_linear']={'calls_observed':len(trace),'maximum_true_relative_residual':float(np.max(r)),
                              'orthogonalization':orthogonalization,
                              'iteration_vectors':'GPU resident',
                              'preconditioner':'native_float_amg' if native_amg else 'jacobi',
                              'pressure_gauge':gauge,'maximum_compatibility_relative_l2':float(np.max(compatibility)),
                              'pressure_operator':pressure_operator,
                              'full_pressure_interface_faces':method.get('full_pressure_interface_faces',0),
                              'viscosity_operator':viscosity_operator,'full_viscosity_faces':method.get('full_viscosity_faces',0),
                              'native_amg_levels':method.get('native_amg_levels',0)}
        for root in roots:
            for name in ('case.json','run_manifest.json','projection_method.json'):
                p=root/name;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        p=roots[0]/'gpu_linear.csv';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    if native_pair:
        native_checks=[]
        builds=[]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for index,(root,config) in enumerate(zip(roots,configs)):
            method=json.loads((root/'projection_method.json').read_text())
            if method.get('gpu_orthogonalization','mgs2')!=config.get('gpu_orthogonalization','mgs2') or method['linear_backend']!='native_gpu_fgmres_amg' or method['native_amg_levels']<2:
                raise ValueError('Missing actual selected GPU orthogonalization or native AMG evidence')
            for key,option in (('pressure_operator','gpu_pressure_operator'),('viscosity_operator','gpu_viscosity_operator')):
                if method[key]!=config[option]:raise ValueError('Actual GPU operator differs from configuration')
            check_path=args.output.with_name(args.output.stem+f'_native_{index}.json')
            subprocess.run([sys.executable,str(Path(__file__).with_name('check_twisted_native_run.py')),
                            '--run',str(root),'--output',str(check_path)],check=True,stdout=subprocess.DEVNULL)
            check=json.loads(check_path.read_text());native_checks.append(check)
            hashes.update(check['source_sha256']);hashes.update(check['executed_input_sha256'])
            for name in ('case.json','run_manifest.json','run_completion.json','projection_method.json'):
                path=root/name;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(check_path.resolve())]=hashlib.sha256(check_path.read_bytes()).hexdigest()
            if args.native_build_pair:
                runtime=json.loads((root/'run_manifest.json').read_text())
                exe=Path(runtime['executable']);build_path=exe.parent/'build_manifest.json'
                build=json.loads(build_path.read_text())
                if build['exit_code'] or not build['source_unchanged'] or build['executable_sha256'][exe.name]!=runtime['executable_sha256']:
                    raise ValueError('Executed build does not match its source manifest')
                for name,value in build['source_sha256'].items():
                    path=exe.parent/'sources'/(name+'.txt')
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=value:
                        raise ValueError('Compiled source snapshot changed: '+str(path))
                    hashes[str(path.resolve())]=value
                hashes[str(build_path.resolve())]=hashlib.sha256(build_path.read_bytes()).hexdigest()
                builds.append(build)
        result['native_trajectory_checks']=native_checks
        result['scope']='Complete native GPU time sequences, CGS2 versus MGS2, identical executable and physical settings; independent Aphros, steady and spatial convergence remain separate'
        result['passed']=result['passed'] and all(c['passed'] for c in native_checks)
        if args.native_build_pair:
            a,b=[v['source_sha256'] for v in builds]
            result['changed_compiled_source_paths']=sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
            result['executables_candidate_reference']=[v['executable_sha256']['simple_channel.exe'] for v in builds]
            result['scope']='Complete native GPU time sequences from recorded builds with identical physical settings; orthogonalization may differ; physical fields, actual mass and full fixed points are compared; independent Aphros, steady and grid convergence remain separate'
        if args.restart_pair:
            result['scope']='Verified completed parent prefix plus restarted native GPU continuation compared at every physical step with an uninterrupted trajectory; physical settings and gates unchanged; independent Aphros and time/grid convergence remain separate'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'steps': len(rows),
                      'iterations_accelerated_ordinary': np.sum([r['iterations_accelerated_ordinary'] for r in rows], axis=0).tolist()}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
