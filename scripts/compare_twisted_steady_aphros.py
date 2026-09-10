"""Compare a fully validated native pseudo steady state directly with completed original Aphros.

Pseudo iterations are not physical time steps. Both states must independently
meet their original steady and mass gates. The common momentum time step is
still required because it enters the finite-grid Rhie--Chow discretization.
The field limits and coarse-center transfer match the ordinary comparison;
this does not establish spatial convergence or an error bound to the true flow.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from analyze_twisted_refinement import completed_steady_iteration
from check_aphros_exact_mass import calculate
from check_twisted_mass import check_case
from check_twisted_projection_flux import check_native
from compare_twisted import read,ordered,vector,transfer,error
from run_twisted_solver import sha
from validate_aphros_extended_reference import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ours','aphros','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--adaptive',action='store_true')
    p.add_argument('--aphros-diffusion-iterations',type=int,default=8)
    a=p.parse_args();ours_root=a.ours.resolve();reference=a.aphros.resolve();out=a.output.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep new comparison reports on D')
    if out.exists():raise ValueError('Preserve prior comparisons')
    if a.aphros_diffusion_iterations<1:p.error('Require a positive original diffusion iteration count')
    repo=Path(__file__).resolve().parents[1]
    case=json.loads((ours_root/'case.json').read_text())
    if case.get('fluid_solver')!='proj_steady':raise ValueError('This entry point requires a native pseudo steady iteration')
    # Validates every original state gate and requires the actual final folder.
    native_proof=completed_steady_iteration(ours_root)
    reference_proof=validate(reference)
    cfg=json.loads((reference/'case_manifest.json').read_text())
    if cfg.get('fluid_solver')!='proj' or not cfg.get('convection') or not case.get('convection'):
        raise ValueError('Require the original steady Navier-Stokes Proj problem')
    expected={'bcg':1,'redistr_adv':0,'diffusion_iters':a.aphros_diffusion_iterations,'diffusion_consistent_guess':1}
    if cfg.get('projection_parameters')!=expected or case.get('momentum_mode')!='imp' or cfg.get('momentum_mode')!='imp':
        raise ValueError('Original implicit projection parameters differ')
    configured=re.findall(r'^set int proj_diffusion_iters (\d+)\s*$',(reference/'a.conf').read_text(),flags=re.M)
    if configured!=[str(a.aphros_diffusion_iterations)]:raise ValueError('Actual reference diffusion setting differs')
    if case.get('projection_iteration_tolerance')!=cfg['iteration_tolerance'] or case.get('convection_scheme')!=cfg['convection_scheme']:
        raise ValueError('Convection scheme or inner convergence tolerance differs')
    if case.get('wall_reconstruction','linear')!='linear':raise ValueError('Original Aphros alignment requires its linear wall closure')
    for name in ('rho','nu'):
        if not np.isclose(case[name],cfg['spec'][name],rtol=1e-13,atol=0):raise ValueError('Different '+name)
    if not np.allclose(case['force'],cfg['spec']['force'],rtol=1e-13,atol=0):raise ValueError('Different driving force')
    geometry=Path(case['embedded_geometry'])
    if not geometry.is_absolute():geometry=repo/geometry
    meta=geometry.with_suffix('.meta.json');meta=meta if meta.exists() else geometry
    metadata=json.loads(meta.read_text())
    if metadata['geometry_spec']!=cfg['spec']:raise ValueError('Different prescribed geometry or physics')
    shift=np.asarray(metadata.get('reference_translation',[0,0,0]));h=cfg['spec']['extent'][1]/cfg['ny'];nx=cfg['shape'][0]

    def translated(data):
        for d,name in enumerate('xyz'):data[name]+=shift[d]
        data['x']%=cfg['spec']['extent'][0]
        return ordered(data)

    ours=ordered(read(ours_root/'solution.csv'));ref=translated(read(reference/'proj_final_b0_cells.csv'))
    if not a.adaptive and (len(ours)!=len(ref) or not np.allclose(vector(ours,'xyz'),vector(ref,'xyz'),rtol=0,atol=1e-13)):
        raise ValueError('Different sample locations require --adaptive')
    sampled,transfer_report=transfer(ours,ref,cfg['shape'],h);volume=ours['volume']
    fields={'velocity':error(vector(ours,'uvw'),sampled[:,:3],volume),
            'pressure':error(ours['p']-np.average(ours['p'],weights=volume),
                             sampled[:,3]-np.average(sampled[:,3],weights=volume),volume)}
    wall=ordered(read(ours_root/'walls.csv'));wallref=translated(read(reference/'tube_final_b0_walls.csv'))
    if len(wall)!=len(wallref) or not np.allclose(vector(wall,'xyz'),vector(wallref,'xyz'),atol=1e-13,rtol=0):
        raise ValueError('Wall points differ')
    if not np.allclose(wall['area'],wallref['area'],rtol=1e-12,atol=0):raise ValueError('Wall areas differ')
    if not np.allclose(vector(wall,('nx','ny','nz')),vector(wallref,('nx','ny','nz')),rtol=0,atol=1e-12):
        raise ValueError('Wall normals differ')
    fields['wall_shear']=error(vector(wall,('tau_x','tau_y','tau_z')),vector(wallref,('tau_x','tau_y','tau_z')),wallref['area'])
    cut=np.isin(ours['id'],wall['owner'])
    fields['cut_cell_velocity']=error(vector(ours,'uvw')[cut],sampled[cut,:3],volume[cut])
    faces=read(reference/'proj_final_b0_faces.csv');sections={}
    for face in faces:
        if int(face['axis'])!=0:continue
        ix=int(round(face['x']/h))
        if ix==nx:continue
        ix=(ix+int(round(shift[0]/h)))%nx
        q,area=sections.get(ix,(0.,0.));sections[ix]=(q+face['flux'],area+face['area'])
    oursq=[];refq=[]
    for section in read(ours_root/'sections.csv'):
        q,area=sections[int(round(section['x']/h))%nx]
        if abs(section['area']-area)>1e-10*area:raise ValueError('Incomplete section coverage')
        oursq.append(section['volume_flux']);refq.append(q)
    fields['section_flux']=error(np.array(oursq),np.array(refq),np.ones(len(refq)))
    temporal_row=read(reference/'tube_b0_time.csv')[-1]
    temporal={name:float(temporal_row[name]) for name in temporal_row.dtype.names}
    iteration_errors=[float(value) for value in re.findall(r'iter=\d+, diff=([^\s]+)',(reference/'run.log').read_text())]
    final_iteration_error=iteration_errors[-1] if iteration_errors else None
    metrics=native_proof['final_metrics']
    baseline_mass=calculate(reference);baseline_mass['double_import_diagnostic']=check_case(reference)
    projection_flux=check_native(ours_root,reference,cfg,metadata)
    limits=({'velocity':.005,'pressure':.01,'near_wall':.01,'wall_shear':.01,'flux':.005} if a.adaptive else
            {key:1e-6 for key in ('velocity','pressure','near_wall','wall_shear','flux')})
    checks={'native_complete_steady_state':native_proof['passed'],'aphros_complete_trajectory':reference_proof['passed'],
            'aphros_final_inner_converged':final_iteration_error is not None and 0<=final_iteration_error<min(1e-11,cfg['iteration_tolerance']),
            'matching_momentum_time_step':np.isclose(case['time_step'],cfg['time_step'],rtol=1e-12,atol=0),
            'native_steady_momentum':metrics['steady_momentum_relative_l2']<min(1e-8,case['tolerance']),
            'native_steady_map_defect':metrics['steady_map_velocity_defect_relative_l2']<min(1e-8,case['tolerance']),
            'native_complete_stationary_fixed_point':metrics['steady_complete_fixed_point_residual']<min(1e-8,case['tolerance']),
            'aphros_temporally_steady':temporal['temporal_acceleration_volume_l2']<1e-8*np.linalg.norm(cfg['spec']['force']),
            'aphros_mass_conservation':baseline_mass['passed'],'native_projection_mass':projection_flux['native_mass']['passed'],
            'all_face_fluxes':projection_flux['face_normal_velocity']['relative_l2']<limits['flux']}
    for key,limit in (('velocity','velocity'),('pressure','pressure'),('wall_shear','wall_shear'),
                      ('cut_cell_velocity','near_wall'),('section_flux','flux')):
        checks[key]=fields[key]['relative_l2']<limits[limit]
    if a.adaptive:checks['actual_fluid_interfaces']=metrics['coarse_fine_faces']>0 and transfer_report['coarse_samples']>0
    paths=[ours_root/name for name in ('solution.csv','walls.csv','flux.csv','mesh_cells.csv','mesh_faces.csv','sections.csv','case.json','metrics.json')]
    paths += [reference/name for name in ('proj_final_b0_cells.csv','proj_final_b0_faces.csv','tube_final_b0_walls.csv','case_manifest.json','a.conf','tube_b0_time.csv','run_completion.json')]
    paths += [meta,Path(__file__),*[Path(__file__).with_name(name) for name in (
        'analyze_twisted_refinement.py','check_twisted_steady_iteration.py','check_twisted_native_run.py','compare_twisted.py',
        'check_twisted_projection_flux.py','check_twisted_mass.py','check_aphros_exact_mass.py',
        'check_aphros_coupled_precision_probe.py','validate_aphros_extended_reference.py','run_twisted_solver.py')]]
    hashes={str(path.resolve()):sha(path) for path in paths}
    for proof in (native_proof,reference_proof,baseline_mass):hashes.update(proof['source_sha256'])
    hashes.update(native_proof['executed_input_sha256'])
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Compared inputs changed')
    checks={key:bool(value) for key,value in checks.items()}
    result={'passed':all(checks.values()),'scope':__doc__,'checks':checks,'limits':limits,
            'comparison':'checked coarse-center transfer; exact cut-cell and wall locations' if a.adaptive else 'identical cell and wall locations',
            'flow_regime':'steady Navier-Stokes','native_iteration_mode':'steady_pseudo_iteration',
            'native_physical_trajectory_claimed':False,'spatial_convergence_checked':False,
            'native_state_provenance':native_proof,'extended_reference_provenance':reference_proof,
            'complete_native_steady_run_checked':True,'complete_reference_run_checked':True,'reference_checkpoint':None,
            'cells':len(ours),'wall_faces':len(wall),'transfer':transfer_report,'reference_translation':shift.tolist(),
            'aphros_temporal_diagnostic':temporal,'aphros_diffusion_iterations':a.aphros_diffusion_iterations,
            'aphros_final_iteration_change':final_iteration_error,
            'aphros_mass_conservation':baseline_mass,'projection_flux':projection_flux,**fields,'source_sha256':hashes}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':checks,'fields':{key:value['relative_l2'] for key,value in fields.items()}}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
