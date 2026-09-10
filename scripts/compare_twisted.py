"""Compare converged curved-tube states with an explicit checked grid transfer.

Pressure is aligned by its volume-weighted gauge. Local wall traction is
compared as a vector with area weights; values near zero are not used as
pointwise relative-error denominators. Coarse samples require --adaptive and
use tensor cubic interpolation of 64 included reference centers. Cut cells and
wall points must still coincide exactly; coordinate translations are explicit.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from check_twisted_mass import check_case


def read(path):
    data=np.genfromtxt(path,delimiter=',',names=True,ndmin=1)
    if not len(data) or any(not np.isfinite(data[k]).all() for k in data.dtype.names):
        raise ValueError(f'Nonfinite or empty CSV: {path}')
    return data


def vector(data,names):return np.column_stack([data[k] for k in names])
def ordered(data):return data[np.lexsort((data['z'],data['y'],data['x']))]


def error(a,b,weights):
    if a.ndim==1:a=a[:,None];b=b[:,None]
    difference=np.linalg.norm(a-b,axis=1);reference=np.linalg.norm(b,axis=1)
    denominator=float(np.sum(weights*reference**2))
    return {'relative_l2':float(np.sqrt(np.sum(weights*difference**2)/max(denominator,1e-300))),
            'absolute_l2':float(np.sqrt(np.sum(weights*difference**2)/np.sum(weights))),
            'absolute_max':float(np.max(difference)),
            'absolute_p99':float(np.quantile(difference,.99))}


def transfer(ours,reference,shape,h):
    """Sample a uniform reference at native fine/coarse cube centers."""
    lookup=np.full(shape,-1,dtype=np.int32)
    positions=vector(reference,['x','y','z'])
    keys=np.rint(positions/h-.5).astype(int)
    if np.any(keys<0) or np.any(keys>=shape):raise ValueError('Reference coordinates outside translated box')
    lookup[tuple(keys.T)]=np.arange(len(reference))
    locations=vector(ours,['x','y','z'])
    fine=np.isclose(ours['h'],h,rtol=1e-12,atol=0)
    if not np.all(fine | np.isclose(ours['h'],2*h,rtol=1e-12,atol=0)):
        raise ValueError('Transfer supports only reference spacing or one coarser octree level')
    values=vector(reference,['u','v','w','p']);out=np.zeros((len(ours),4));linear=out.copy()
    fk=np.rint(locations[fine]/h-.5).astype(int);indices=lookup[tuple(fk.T)]
    if np.any(indices<0):raise ValueError('Fine sample absent from independent reference')
    if not np.allclose(ours['volume'][fine],reference['volume'][indices],rtol=1e-12,atol=0):
        raise ValueError('Fine cut volumes differ')
    out[fine]=linear[fine]=values[indices]
    coarse=np.flatnonzero(~fine)
    ck=np.rint(locations[coarse]/h).astype(int)
    if not np.allclose(locations[coarse]/h,ck,rtol=0,atol=1e-12):raise ValueError('Coarse center not aligned')
    weights=np.array([-1,9,9,-1])/16
    volume=np.zeros(len(coarse))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                q=ck+np.array([i,j,k])-2;q[:,0]%=shape[0]
                if np.any(q[:,1:]<0) or np.any(q[:,1:]>=np.array(shape[1:])):raise ValueError('Transfer stencil leaves box')
                ids=lookup[tuple(q.T)]
                if np.any(ids<0):raise ValueError('Cubic transfer stencil includes excluded fluid')
                out[coarse]+=weights[i]*weights[j]*weights[k]*values[ids]
                if i in (1,2) and j in (1,2) and k in (1,2):
                    linear[coarse]+=values[ids]/8
                    volume+=reference['volume'][ids]
    if not np.allclose(ours['volume'][coarse],volume,rtol=1e-12,atol=0):raise ValueError('Coarse coverage volume mismatch')
    sensitivity=error(linear[:,:3],out[:,:3],ours['volume'])
    return out,{'coarse_samples':len(coarse),'fine_samples':int(fine.sum()),
                'method':'Tensor cubic at coarse centers; exact identity at fine centers',
                'linear_vs_cubic_velocity':sensitivity}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours',type=Path,required=True)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adaptive',action='store_true',help='Require actual fluid coarse/fine faces and permit checked cubic transfer')
    parser.add_argument('--transient',action='store_true',help='Compare the same physical time; do not claim a steady state')
    parser.add_argument('--aphros-diffusion-iterations',type=int,default=1,
                        help='Explicit expected original Proj inner diffusion count; compares converged equations, not identical iteration maps')
    parser.add_argument('--reference-checkpoint',action='store_true',help='Use a verified snapshot of a completed physical step; never counts as a completed baseline run')
    parser.add_argument('--native-step-prefix',action='store_true',help='Validate one completed Proj step of a longer trajectory; requires --transient and never asserts full native completion')
    parser.add_argument('--extended-reference',action='store_true',help='Explicitly validate the isolated extended-scalar Aphros build, original equation bodies and original geometry snapshot')
    args=parser.parse_args()
    if args.native_step_prefix and not args.transient:parser.error('A native step prefix requires --transient')
    if args.aphros_diffusion_iterations<1:parser.error('Aphros diffusion iterations must be positive')
    baseline_metadata=json.loads((args.aphros/'case_manifest.json').read_text(encoding='utf-8'))
    fluid_solver=baseline_metadata.get('fluid_solver','simple')
    if fluid_solver not in ('simple','proj'):raise ValueError('Unsupported baseline fluid solver')
    if args.extended_reference and fluid_solver!='proj':parser.error('Extended reference requires Proj')
    extended_reference=None
    if args.native_step_prefix and fluid_solver!='proj':parser.error('Native step prefixes require Proj')
    if fluid_solver!='proj' and args.aphros_diffusion_iterations!=1:
        raise ValueError('Diffusion iteration option requires the original Proj baseline')
    checkpoint=None
    if args.reference_checkpoint:
        if not args.transient:parser.error('A partial reference checkpoint requires --transient')
        from snapshot_twisted_reference import validate
        checkpoint,checkpoint_error=validate(args.aphros)
    native_checkpoint=None
    if args.native_step_prefix:
        from validate_twisted_native_step import validate as validate_native
        native_checkpoint=validate_native(args.ours)
    paths=[args.ours/'solution.csv',args.aphros/f'{fluid_solver}_final_b0_cells.csv',
           args.ours/'walls.csv',args.aphros/'tube_final_b0_walls.csv']
    ours,reference=map(lambda p:ordered(read(p)),paths[:2])
    case=json.loads((args.ours/'case.json').read_text(encoding='utf-8-sig'))
    if case.get('fluid_solver','simple')!=fluid_solver:raise ValueError('Different native/reference fluid solvers')
    if fluid_solver=='proj':
        if args.reference_checkpoint:raise ValueError('Projection comparison currently requires a complete independent reference')
        expected={'bcg':1,'redistr_adv':0,'diffusion_iters':args.aphros_diffusion_iterations,'diffusion_consistent_guess':1}
        if baseline_metadata.get('projection_parameters')!=expected or case.get('momentum_mode')!='imp':
            raise ValueError('Projection algorithm parameters differ')
        configured=re.findall(r'^set int proj_diffusion_iters (\d+)\s*$',(args.aphros/'a.conf').read_text(),flags=re.M)
        if configured!=[str(args.aphros_diffusion_iterations)]:
            raise ValueError('Actual Proj diffusion iteration setting differs from the explicitly requested count')
        if case.get('projection_iteration_tolerance',1e-11)!=baseline_metadata['iteration_tolerance']:
            raise ValueError('Projection iteration tolerances differ')
        runroot=args.ours.parent
        complete=None if native_checkpoint else json.loads((runroot/'run_completion.json').read_text())
        executed=json.loads((runroot/'run_manifest.json').read_text())
        if complete is not None and (complete['exit_code']!=0 or not all(complete[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged'))):
            raise ValueError('Native projection run did not complete with unchanged inputs')
        if hashlib.sha256(Path(executed['executable']).read_bytes()).hexdigest()!=executed['executable_sha256']:
            raise ValueError('Native projection executable changed')
        reference_runtime=json.loads((args.aphros/'run_manifest.json').read_text(encoding='utf-8-sig'))
        reference_executable=Path(reference_runtime['executable'])
        reference_build_path=reference_executable.parent/'build_manifest.json'
        reference_build=json.loads(reference_build_path.read_text(encoding='utf-8-sig'))
        if (hashlib.sha256(reference_executable.read_bytes()).hexdigest()!=reference_runtime['executable_sha256'] or
                reference_build['executable_sha256']!=reference_runtime['executable_sha256']):
            raise ValueError('Executed reference binary differs from its build')
        if args.extended_reference:
            from validate_aphros_extended_reference import validate as validate_extended
            extended_reference=validate_extended(args.aphros)
        elif hashlib.sha256(Path(reference_build['library']).read_bytes()).hexdigest()!=reference_build['library_sha256']:
            raise ValueError('Linked reference library changed')
    geometry=Path(case['embedded_geometry']);meta=geometry.with_suffix('.meta.json')
    metadata=json.loads((meta if meta.exists() else geometry).read_text(encoding='utf-8'))
    shift=np.array(metadata.get('reference_translation',[0,0,0]))
    cfg=json.loads((args.aphros/'case_manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256((args.aphros/'a.conf').read_bytes()).hexdigest()!=cfg['config_sha256']:
        raise ValueError('Baseline config does not match its manifest')
    physical_parameters=[('rho',cfg['spec']['rho']),('nu',cfg['spec']['nu'])]
    if fluid_solver=='simple':
        physical_parameters += [('alpha_u',cfg.get('velocity_relaxation',.7)),
                                ('alpha_p',cfg.get('pressure_relaxation',.3))]
    for field,expected in physical_parameters:
        if not np.isclose(case[field],expected,rtol=1e-13,atol=0):raise ValueError(f'Mismatched {field}')
    convection=case.get('convection',True)
    if case.get('momentum_mode','imp')!=cfg.get('momentum_mode','imp'):
        raise ValueError('Different implicit/explicit momentum discretizations')
    if case.get('wall_reconstruction','linear')!='linear':
        raise ValueError('This same-discretization Aphros audit requires the linear wall closure; experimental quadratic walls need a separate physical-grid comparison')
    if convection!=cfg.get('convection',False):raise ValueError('Different Stokes/Navier-Stokes equations')
    if args.transient and not convection:raise ValueError('Transient comparison requires the time-dependent Navier-Stokes cases')
    if not np.allclose(case['force'],cfg['spec']['force'],rtol=1e-13,atol=0):raise ValueError('Mismatched driving force')
    if metadata['geometry_spec']!=cfg['spec']:raise ValueError('Mismatched prescribed geometry or physics')
    h=cfg['spec']['extent'][1]/cfg['ny'];nx=cfg['shape'][0]
    def translated(data):
        data=data.copy()
        for d,name in enumerate(('x','y','z')):data[name]+=shift[d]
        data['x']%=cfg['spec']['extent'][0]
        return ordered(data)
    reference=translated(reference)
    if not args.adaptive and (len(ours)!=len(reference) or not np.allclose(vector(ours,['x','y','z']),vector(reference,['x','y','z']),rtol=0,atol=1e-13)):
        raise ValueError('Different sampling locations require --adaptive')
    sampled,transfer_report=transfer(ours,reference,cfg['shape'],h)
    volume=ours['volume']
    velocity=error(vector(ours,['u','v','w']),sampled[:,:3],volume)
    p=ours['p']-np.average(ours['p'],weights=volume)
    pref=sampled[:,3]-np.average(sampled[:,3],weights=volume)
    pressure=error(p,pref,volume)
    wall=ordered(read(paths[2]));wallref=translated(read(paths[3]))
    if len(wall)!=len(wallref) or not np.allclose(vector(wall,['x','y','z']),vector(wallref,['x','y','z']),atol=1e-13,rtol=0):
        raise ValueError('Wall points differ')
    if not np.allclose(wall['area'],wallref['area'],rtol=1e-12,atol=0):raise ValueError('Wall areas differ')
    if not np.allclose(vector(wall,['nx','ny','nz']),vector(wallref,['nx','ny','nz']),rtol=0,atol=1e-12):raise ValueError('Wall normal directions differ')
    shear=error(vector(wall,['tau_x','tau_y','tau_z']),vector(wallref,['tau_x','tau_y','tau_z']),wallref['area'])
    cut=np.isin(ours['id'],wall['owner'])
    near=error(vector(ours,['u','v','w'])[cut],sampled[cut,:3],volume[cut])
    # A repeated x=0/L periodic seam is counted only once.
    faces=read(args.aphros/f'{fluid_solver}_final_b0_faces.csv')
    sections={}
    for face in faces:
        if int(face['axis'])!=0:continue
        ix=int(round(face['x']/h))
        if ix==nx:continue
        ix=(ix+int(round(shift[0]/h)))%nx
        q,area=sections.get(ix,(0.,0.));sections[ix]=(q+face['flux'],area+face['area'])
    actual=read(args.ours/'sections.csv');oursq=[];refq=[]
    for section in actual:
        ix=int(round(section['x']/h))%nx;q,area=sections[ix]
        if abs(section['area']-area)>1e-10*area:raise ValueError('Incomplete section coverage')
        oursq.append(section['volume_flux']);refq.append(q)
    flux=error(np.array(oursq),np.array(refq),np.ones(len(refq)))
    log=(args.aphros/'run.log').read_text(encoding='utf-8',errors='replace')
    iterations=[float(line.split('diff=')[1]) for line in log.splitlines() if 'diff=' in line]
    oursmetrics=json.loads((args.ours/'metrics.json').read_text(encoding='utf-8'))
    baseline_converged=bool(iterations and iterations[-1]<1e-11 and 'End of simulation' in log)
    if checkpoint is not None:baseline_converged=checkpoint_error<1e-11
    temporal=None
    runtime=None
    if convection:
        runtime=json.loads((args.aphros/'run_manifest.json').read_text(encoding='utf-8-sig'))
        if checkpoint is None:
            completion=json.loads((args.aphros/'run_completion.json').read_text(encoding='utf-8-sig'))
            if completion['exit_code']!=0 or not completion['completed_utc']:
                raise ValueError('Reference process did not complete successfully')
        if runtime['config_sha256']!=cfg['config_sha256']:
            raise ValueError('Executed baseline configuration differs from the case manifest')
        env=runtime['environment']
        if (fluid_solver=='simple' and env.get('APHROS_TWISTED_FIX_FLUX_HALO')!='1') or env.get('APHROS_TWISTED_GEOMETRY_ONLY') is not None:
            raise ValueError('Navier-Stokes reference requires the documented flux halo repair and an actual flow run')
        if oursmetrics['convection_scheme']!=cfg['convection_scheme']:raise ValueError('Different convection schemes')
        times=read(args.aphros/'tube_b0_time.csv');last=times[-1]
        reference_steps=cfg['time_steps'] if checkpoint is None else checkpoint['completed_step']
        if len(times)!=reference_steps or not np.isclose(last['time'],cfg['time_step']*reference_steps,rtol=1e-12,atol=0):
            raise ValueError('Baseline physical time sequence is incomplete')
        temporal={name:float(last[name]) for name in times.dtype.names}
        if args.transient:
            if not (np.isclose(case.get('time_step',0),cfg['time_step'],rtol=1e-12,atol=0) and
                    np.isclose(case.get('physical_time',0),last['time'],rtol=1e-12,atol=0)):
                raise ValueError('Transient comparison requires identical physical time and time-step size')
    baseline_mass=check_case(args.aphros)
    if args.extended_reference and (args.aphros/'proj_final_b0_exact.json').exists():
        from check_aphros_exact_mass import calculate
        exact_mass=calculate(args.aphros)
        exact_mass['double_import_diagnostic']=baseline_mass
        baseline_mass=exact_mass
    limits=({'velocity':.005,'pressure':.01,'near_wall':.01,'wall_shear':.01,'flux':.005} if args.adaptive else
            {key:1e-6 for key in ('velocity','pressure','near_wall','wall_shear','flux')})
    checks={'ours_converged':oursmetrics['converged'],'aphros_converged':baseline_converged,
            'aphros_mass_conservation':baseline_mass['passed'],
            'velocity':velocity['relative_l2']<limits['velocity'],'pressure':pressure['relative_l2']<limits['pressure'],
            'cut_cell_velocity':near['relative_l2']<limits['near_wall'],'wall_shear':shear['relative_l2']<limits['wall_shear'],
            'flux':flux['relative_l2']<limits['flux']}
    if case.get('momentum_mode','imp')=='exp':
        checks['aphros_explicit_residual_halo']=runtime['environment'].get('APHROS_TWISTED_FIX_EXPLICIT_RESIDUAL_HALO')=='1'
    if convection and not args.transient:
        # The reciprocal momentum diagonal enters Rhie--Chow even at a steady
        # fixed point. A different BE step can change its finite-grid result.
        checks['matching_momentum_time_step']=np.isclose(case.get('time_step',0),cfg['time_step'],rtol=1e-12,atol=0)
        checks['ours_steady_momentum']=oursmetrics['steady_momentum_relative_l2']<min(1e-8,case['tolerance'])
        checks['ours_temporally_steady']=oursmetrics['temporal_acceleration_relative_l2']<1e-8
        checks['aphros_temporally_steady']=temporal['temporal_acceleration_volume_l2']<1e-8*np.linalg.norm(cfg['spec']['force'])
    if args.adaptive:checks['actual_fluid_interfaces']=oursmetrics['coarse_fine_faces']>0 and transfer_report['coarse_samples']>0
    projection_flux=None
    projection_runtime=None
    if fluid_solver=='proj':
        from check_twisted_projection_flux import check_native
        projection_flux=check_native(args.ours,args.aphros,cfg,metadata)
        checks['native_projection_mass']=projection_flux['native_mass']['passed']
        checks['all_face_fluxes']=projection_flux['face_normal_velocity']['relative_l2']<limits['flux']
        if case.get('anderson_depth',0):
            checks['complete_projection_fixed_point']=oursmetrics.get('complete_inner_fixed_point_residual',float('inf'))<case['tolerance']
        projection_runtime={'manifest':executed,'completion':complete,'reference_build':reference_build,
                            'complete_native_run_checked':native_checkpoint is None,'native_step_prefix':native_checkpoint}
        paths += [args.ours/'flux.csv',args.ours/'mesh_faces.csv',args.ours/'mesh_cells.csv',
                  args.ours/'sections.csv',args.ours/'case.json',args.ours/'metrics.json',
                  runroot/'run_manifest.json',
                  args.aphros/'proj_final_b0_faces.csv',args.aphros/'tube_b0_geometry_faces.csv',
                  args.aphros/'tube_b0_geometry_cells.csv',args.aphros/'tube_b0_geometry_walls.csv',
                  args.aphros/'tube_b0_time.csv',args.aphros/'case_manifest.json',args.aphros/'a.conf',
                  args.aphros/'run_manifest.json',args.aphros/'run_completion.json',
                  reference_build_path,meta if meta.exists() else geometry,Path(__file__),
                  Path(__file__).with_name('check_twisted_projection_flux.py')]
        if native_checkpoint is None:paths.append(runroot/'run_completion.json')
        else:
            from validate_twisted_native_step import verify_unchanged
            verify_unchanged(native_checkpoint)
            paths.append(Path(__file__).with_name('validate_twisted_native_step.py'))
    checks={key:bool(value) for key,value in checks.items()}
    if checkpoint is not None:checks['aphros_checkpoint_converged']=checks.pop('aphros_converged')
    result={'passed':all(checks.values()),'checks':checks,'comparison':'checked coarse-center transfer; exact cut-cell and wall locations' if args.adaptive else 'identical cell and wall locations',
            'limits':limits,'transfer':transfer_report,'reference_translation':shift.tolist(),
            'flow_regime':'transient Navier-Stokes checkpoint' if args.transient else ('steady Navier-Stokes' if convection else 'Stokes'),
            'aphros_temporal_diagnostic':temporal,
            'aphros_runtime':runtime,
            'complete_reference_run_checked':checkpoint is None,'reference_checkpoint':checkpoint,
            'cells':len(ours),'wall_faces':len(wall),'velocity':velocity,'pressure':pressure,
            'cut_cell_velocity':near,'wall_shear':shear,'section_flux':flux,
            'aphros_final_iteration_change':checkpoint_error if checkpoint is not None else (iterations[-1] if iterations else None),
            'aphros_mass_conservation':baseline_mass,
            'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if projection_flux is not None:
        result['projection_flux']=projection_flux
        result['native_projection_runtime']=projection_runtime
        result['aphros_diffusion_iterations']=args.aphros_diffusion_iterations
        result['projection_comparison_scope']='Same converged discrete equations; original Aphros inner diffusion count is recorded explicitly'
    if extended_reference is not None:
        result['extended_reference_provenance']=extended_reference
        result['source_sha256'].update(extended_reference['source_sha256'])
        result['source_sha256'].update(baseline_mass['source_sha256'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
