"""Compare converged tube grids at common physical wall and inward probes.

Reports interpolation sensitivity separately. A same-grid solver match is not
used as evidence of grid convergence. Wall probes follow the analytic surface;
near-wall velocity probes follow its true inward normal, in SI coordinates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from check_twisted_mass import read,vector
from compare_twisted import error
from run_twisted_solver import sha


def completed_trajectory(folder,config):
    """Check the actual completed trajectory behind a final Navier--Stokes field."""
    root=folder.resolve().parent
    if not folder.name.startswith('step_') or config.get('physical_step')!=config.get('time_steps'):
        raise ValueError('Navier-Stokes refinement requires a final physical-step folder')
    paths=[root/name for name in ('case.json','run_manifest.json','run_completion.json','transient_summary.json','time_history.csv')]
    hashes={str(p):sha(p) for p in paths}
    parent=json.loads(paths[0].read_text());runtime=json.loads(paths[1].read_text());done=json.loads(paths[2].read_text())
    summary=json.loads(paths[3].read_text());times=list(csv.DictReader(paths[4].open()))
    normalize=lambda c:{k:v for k,v in c.items() if k not in ('output','physical_step','physical_time')}
    if normalize(parent)!=normalize(config):raise ValueError('Final field configuration differs from trajectory')
    if done['exit_code'] or not all(done[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
        raise ValueError('Run did not complete with unchanged inputs')
    if not summary['steady_converged'] or not summary['converged'] or summary['steps_completed']!=config['time_steps']:
        raise ValueError('Run summary does not establish a completed steady trajectory')
    if len(times)!=config['time_steps']:raise ValueError('Incomplete physical time sequence')
    for step,row in enumerate(times,1):
        if int(row['step'])!=step or row['inner_converged']!='true' or not np.isclose(float(row['time']),step*config['time_step'],rtol=1e-12,atol=0):
            raise ValueError('Missing or unconverged physical step')
    if not np.isclose(config['physical_time'],config['time_step']*config['time_steps'],rtol=1e-12,atol=0):
        raise ValueError('Final field has the wrong physical time')
    inputs={runtime['executable']:runtime['executable_sha256'],runtime['config']:runtime['config_sha256'],**runtime['geometry_input_sha256']}
    for path,value in inputs.items():
        if sha(Path(path))!=value:raise ValueError('Executed trajectory input changed: '+path)
        hashes[str(Path(path).resolve())]=value
    return {'completed_steps':len(times),'final_physical_time':config['physical_time'],'source_sha256':hashes}


def periodic_neighbors(tree,centers,points,period,count):
    query=np.concatenate([points+np.array([s,0,0]) for s in (0,-period,period)])
    _,ids=tree.query(query,k=min(count,len(centers)))
    ids=np.asarray(ids).reshape(3,len(points),-1).transpose(1,0,2).reshape(len(points),-1)
    selected=[];offsets=[]
    for point,candidate in zip(points,ids):
        candidate=np.unique(candidate);delta=centers[candidate]-point
        delta[:,0]-=np.rint(delta[:,0]/period)*period
        order=np.argsort(np.sum(delta*delta,axis=1))[:count]
        selected.append(candidate[order]);offsets.append(delta[order])
    return selected,offsets


def sample(centers,values,points,period,count,normals=None,tree=None):
    ids,offsets=periodic_neighbors(cKDTree(centers) if tree is None else tree,centers,points,period,count)
    output=[];condition=[]
    for i,(index,delta) in enumerate(zip(ids,offsets)):
        scale=np.max(np.linalg.norm(delta,axis=1))
        if scale<=0:raise ValueError('Degenerate probe neighborhood')
        local=delta/scale
        if normals is None:
            x,y,z=local.T
            design=np.column_stack((np.ones(len(x)),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z))
        else:
            # Fit along the two analytic wall tangents; no normal extrapolation
            # of traction is implied by a curved surface's chord coordinates.
            normal=normals[i];axis=np.eye(3)[np.argmin(abs(normal))]
            t=np.cross(normal,axis);t/=np.linalg.norm(t);b=np.cross(normal,t)
            x,y=local@t,local@b
            design=np.column_stack((np.ones(len(x)),x,y,x*x,x*y,y*y))
        weight=1/np.sqrt(.05+np.sum(local*local,axis=1))
        coef,_,rank,singular=np.linalg.lstsq(design*weight[:,None],values[index]*weight[:,None],rcond=1e-12)
        if rank!=design.shape[1]:raise ValueError('Rank-deficient physical probe fit')
        condition.append(float(singular[0]/singular[-1]));output.append(coef[0])
    return np.asarray(output),max(condition)


def probes(spec):
    # Fixed physical probes, independent of either requested grid resolution.
    xs=np.arange(8)*spec['period']/8
    angles=np.arange(16)*2*np.pi/16
    surface=[];normals=[];weights=[]
    k=2*np.pi/spec['period'];a=spec['amplitude'];r=spec['radius']
    for x in xs:
        yc=spec['center_y']+a*np.sin(k*x);zc=spec['center_z']+a*np.cos(k*x)
        for theta in angles:
            tangent=a*k*np.cos(k*x+theta)
            normal=np.array([-tangent,np.cos(theta),np.sin(theta)])
            weights.append(np.linalg.norm(normal));normal/=np.linalg.norm(normal)
            surface.append([x,yc+r*np.cos(theta),zc+r*np.sin(theta)]);normals.append(normal)
    surface=np.array(surface);normals=np.array(normals)
    distances=np.array([.001953125,.00390625,.0078125,.015625])
    if distances[-1]>=r:raise ValueError('Probe distances require the standard thin-tube radius')
    points=(surface[:,None,:]-distances[None,:,None]*normals[:,None,:]).reshape(-1,3)
    points[:,0]%=spec['period']
    return surface,normals,np.array(weights),distances,points


def load(root,steady_iteration=False):
    config=json.loads((root/'case.json').read_text(encoding='utf-8-sig'))
    if (config.get('fluid_solver')=='proj_steady') != steady_iteration:
        raise ValueError('Pseudo steady input requires its explicit steady-iteration option; physical input must not use it')
    path=Path(config['embedded_geometry']);meta=path.with_suffix('.meta.json')
    metadata=json.loads((meta if meta.exists() else path).read_text(encoding='utf-8'))
    metrics=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    if not metrics['converged']:raise ValueError('Unconverged Cirrus input')
    residuals=['steady_momentum_relative_l2',
               'steady_map_velocity_defect_relative_l2' if steady_iteration else 'temporal_acceleration_relative_l2']
    if config.get('convection',True) and any(not 0<=metrics.get(k,float('inf'))<1e-8 for k in residuals):
        raise ValueError('Navier-Stokes grid refinement requires a converged steady solution')
    cells=read(root/'solution.csv');walls=read(root/'walls.csv')
    shift=np.array(metadata.get('reference_translation',[0,0,0]));period=metadata['geometry_spec']['period']
    for data in (cells,walls):
        for d,name in enumerate(('x','y','z')):data[name]-=shift[d]
        data['x']%=period
    cells['p']-=np.average(cells['p'],weights=cells['volume'])
    return config,metadata,metrics,cells,walls


def completed_steady_iteration(folder, trust_local_files=False):
    # Import lazily: the state checker also uses completed_trajectory above.
    # Revalidate actual outputs and all original gates, rather than trusting a
    # supplied summary or relabeling pseudo iterations as physical time steps.
    from check_twisted_steady_iteration import validate
    proof=validate(folder.resolve().parent, trust_local_files=trust_local_files)
    if not proof['passed']:raise ValueError('Steady state validation failed')
    summary=json.loads((folder.parent/'steady_summary.json').read_text())
    if Path(summary['final_output']).resolve()!=folder.resolve():
        raise ValueError('Refinement requires the actual final steady iteration')
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse',type=Path,required=True)
    parser.add_argument('--fine',type=Path,required=True)
    parser.add_argument('--coarse-steady-iteration',action='store_true')
    parser.add_argument('--fine-steady-iteration',action='store_true')
    parser.add_argument('--same-grid-regression',action='store_true',
                        help='Exercise probe analysis on an identical native mesh without claiming spatial convergence')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous refinement results')
    roots=(args.coarse.resolve(),args.fine.resolve())
    modes=(args.coarse_steady_iteration,args.fine_steady_iteration)
    data=[load(p,mode) for p,mode in zip(roots,modes)]
    trajectories=[completed_trajectory(root,d[0]) if d[0].get('convection',True) and not mode else None
                  for root,d,mode in zip(roots,data,modes)]
    steady_proofs=[completed_steady_iteration(root) if mode else None for root,mode in zip(roots,modes)]
    c,f=data
    if c[1]['geometry_spec']!=f[1]['geometry_spec']:raise ValueError('Different prescribed problem')
    # Both modes have passed their own complete acceptance checks. Their
    # iteration schedules differ; the discrete projection equations do not.
    solvers=['proj' if mode else d[0].get('fluid_solver','simple') for d,mode in zip(data,modes)]
    if solvers[0]!=solvers[1]:
        raise ValueError('Different fluid solvers; this is not a pure grid-refinement comparison')
    if args.same_grid_regression:
        if c[1]['finest_h']!=f[1]['finest_h'] or any(sha(roots[0]/name)!=sha(roots[1]/name)
                for name in ('mesh_cells.csv','mesh_faces.csv')):
            raise ValueError('Same-grid regression requires the identical native mesh')
    elif not c[1]['finest_h']>f[1]['finest_h']:
        raise ValueError('Spatial refinement requires a strictly finer grid; identical grids need --same-grid-regression')
    for name in ('rho','nu','force','alpha_u','alpha_p','convection'):
        if c[0][name]!=f[0][name]:raise ValueError(f'Different {name}; grid comparison would be confounded')
    if c[0].get('momentum_mode','imp')!=f[0].get('momentum_mode','imp'):
        raise ValueError('Different momentum modes; this is not a pure grid-refinement comparison')
    if c[0].get('wall_reconstruction','linear')!=f[0].get('wall_reconstruction','linear'):
        raise ValueError('Different wall closures; this is not a pure grid-refinement comparison')
    if c[0]['convection']:
        for name in ('time_step','convection_scheme'):
            if c[0].get(name)!=f[0].get(name):
                raise ValueError(f'Different {name}; finite-grid momentum/flux discretization differs')
    spec=c[1]['geometry_spec'];surface,normals,weights,distances,points=probes(spec)
    velocity=[];traction=[];sensitivity=[];conditions=[];sensitivity_by_distance=[]
    for config,metadata,metrics,cells,walls in data:
        centers=vector(cells,['x','y','z']);values=vector(cells,['u','v','w','p'])
        v32,cv32=sample(centers,values,points,spec['period'],32)
        v64,cv64=sample(centers,values,points,spec['period'],64)
        wc=vector(walls,['x','y','z']);wv=vector(walls,['tau_x','tau_y','tau_z'])
        t32,ct32=sample(wc,wv,surface,spec['period'],32,normals)
        t64,ct64=sample(wc,wv,surface,spec['period'],64,normals)
        velocity.append(v32);traction.append(t32)
        sensitivity.append({'velocity':error(v32[:,:3],v64[:,:3],np.repeat(weights,len(distances))),
                            'pressure':error(v32[:,3],v64[:,3],np.repeat(weights,len(distances))),
                            'wall_shear':error(t32,t64,weights)})
        sensitivity_by_distance.append({str(d):{
            'velocity':error(v32[j::len(distances),:3],v64[j::len(distances),:3],weights),
            'pressure':error(v32[j::len(distances),3],v64[j::len(distances),3],weights)}
            for j,d in enumerate(distances)})
        conditions.append({'volume_fit':max(cv32,cv64),'surface_fit':max(ct32,ct64)})
    by_distance={str(d):error(velocity[0][j::len(distances),:3],velocity[1][j::len(distances),:3],weights)
                 for j,d in enumerate(distances)}
    pressure_by_distance={str(d):error(velocity[0][j::len(distances),3],velocity[1][j::len(distances),3],weights)
                          for j,d in enumerate(distances)}
    shear=error(traction[0],traction[1],weights)
    qdifference=abs(c[2]['volume_flux']/f[2]['volume_flux']-1)
    # Prospective refinement targets, not inferred from same-grid matching.
    limits={'flow_relative':.005,'near_wall_velocity_relative_l2':.01,'wall_shear_relative_l2':.01,
            'pressure_relative_l2':.01,'sampling_pressure_relative_l2':.0025,
            'sampling_velocity_relative_l2':.0025,'sampling_shear_relative_l2':.0025}
    checks={'flow':qdifference<limits['flow_relative'],
            'near_wall_velocity':all(v['relative_l2']<limits['near_wall_velocity_relative_l2'] for v in by_distance.values()),
            'wall_shear':shear['relative_l2']<limits['wall_shear_relative_l2'],
            'pressure':all(v['relative_l2']<limits['pressure_relative_l2'] for v in pressure_by_distance.values()),
            'sampling_sensitivity':all(s['wall_shear']['relative_l2']<limits['sampling_shear_relative_l2'] for s in sensitivity) and
                all(s['velocity']['relative_l2']<limits['sampling_velocity_relative_l2'] and
                    s['pressure']['relative_l2']<limits['sampling_pressure_relative_l2']
                    for grid in sensitivity_by_distance for s in grid.values())}
    report={'passed':all(checks.values()),'scope':('Same-grid probe regression; no spatial convergence claim' if args.same_grid_regression else
            'Cirrus grid refinement at fixed physical probes; not independent Aphros matching'),
            'spatial_convergence_checked':not args.same_grid_regression,
            'spatial_convergence_passed':all(checks.values()) and not args.same_grid_regression,
            'independent_aphros_alignment_checked':False,
            'input_iteration_modes':['steady_pseudo_iteration' if mode else 'physical_trajectory' if d[0].get('convection',True) else 'stokes'
                                     for d,mode in zip(data,modes)],
            'flow_regime':'steady Navier-Stokes' if c[0]['convection'] else 'Stokes',
            'fluid_solver':solvers[0],
            'coarse':str(args.coarse.resolve()),'fine':str(args.fine.resolve()),'checks':checks,'limits':limits,
            'finest_h':[d[1]['finest_h'] for d in data],'fluid_cells':[len(d[3]) for d in data],
            'coarse_fine_faces':[d[2]['coarse_fine_faces'] for d in data],
            'volume_flux':[d[2]['volume_flux'] for d in data],'flow_relative_difference_to_fine':qdifference,
            'near_wall_velocity_by_distance_m':by_distance,'wall_shear':shear,'sampling_sensitivity':sensitivity,
            'sampling_sensitivity_by_distance_m':sensitivity_by_distance,
            'pressure_by_distance_m':pressure_by_distance,
            'fit_condition_numbers':conditions,
            'probe_definition':'8 axial by 16 angular analytic wall locations; inward along true analytic normal; quadratic least squares with 32 vs 64 neighbors',
            'source_sha256':{str((root/name).resolve()):hashlib.sha256((root/name).read_bytes()).hexdigest()
                              for root in (args.coarse,args.fine) for name in ('solution.csv','walls.csv','metrics.json','case.json')}}
    report['trajectory_provenance']=trajectories
    report['steady_iteration_provenance']=steady_proofs
    report['checker_sha256']=sha(Path(__file__))
    for item in trajectories:
        if item is not None:report['source_sha256'].update(item['source_sha256'])
    for item in steady_proofs:
        if item is not None:
            report['source_sha256'].update(item['source_sha256'])
            report['source_sha256'].update(item['executed_input_sha256'])
    if any(sha(Path(path))!=value for path,value in report['source_sha256'].items()):raise ValueError('Refinement sources changed during comparison')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'refinement.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    np.savetxt(args.output/'velocity_probes.csv',np.column_stack((points,np.tile(distances,len(surface)),*velocity)),delimiter=',',
               header='x,y,z,inward_distance,u_coarse,v_coarse,w_coarse,p_coarse,u_fine,v_fine,w_fine,p_fine',comments='')
    np.savetxt(args.output/'wall_probes.csv',np.column_stack((surface,normals,*traction)),delimiter=',',
               header='x,y,z,nx,ny,nz,tau_x_coarse,tau_y_coarse,tau_z_coarse,tau_x_fine,tau_y_fine,tau_z_fine',comments='')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
