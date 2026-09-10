"""Independently balance every retained replay face and compare paired repairs.

This checks an operator-only replay, never a physical time step or steady flow.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

from check_twisted_prefix_native_run import columns
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-cycle',action='store_true')
    args=parser.parse_args();root=args.run.resolve()
    if args.output.exists():raise ValueError('Preserve earlier replay checks')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    hashes={}
    def keep(path,expected=None):
        path=Path(path).resolve();value=sha(path)
        if expected is not None and value!=expected:raise ValueError('Replay input changed: '+str(path))
        hashes[str(path)]=value;return path
    def load(path):return json.loads(keep(path).read_bytes())
    prepared=load(args.preparation);checkpoint=load(prepared['parent_checkpoint'])
    for path,digest in prepared['inputs'].items():keep(path,digest)
    runtime=load(root/'run_manifest.json');completion=load(root/'run_completion.json')
    config=load(root/'case.json');report=load(root/'projection_replay.json')
    if completion['exit_code'] or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
        raise ValueError('Replay did not terminate successfully with unchanged inputs')
    if not config.get('operator_only') or config.get('restart_checkpoint') or not prepared['no_physical_time_advanced']:
        raise ValueError('Require the explicit operator-only paired projection replay')
    if not report['all_projection_checks_passed'] or config['linear_tolerance']!=1e-13:
        raise ValueError('Replay did not meet its original pressure bounds')
    for path,digest in {runtime['executable']:runtime['executable_sha256'],runtime['config']:runtime['config_sha256'],**runtime['geometry_input_sha256']}.items():keep(path,digest)
    build=Path(runtime['executable']).parent/'build_manifest.json';compiled=load(build)
    if compiled['exit_code'] or not compiled['source_unchanged'] or compiled['executable_sha256']['simple_channel.exe']!=runtime['executable_sha256']:
        raise ValueError('Replay executable lacks a verified build')
    for name,digest in compiled['source_sha256'].items():keep(build.parent/'sources'/(name+'.txt'),digest)
    parent=Path(checkpoint['parent_run'])
    for name in ('mesh_cells.csv','mesh_faces.csv'):
        keep(root/name,sha(parent/name));keep(parent/name)
    cells=columns(root/'mesh_cells.csv',('id','volume'))
    faces=columns(root/'mesh_faces.csv',('id','owner','neighbor','area'))
    nc,nf=len(cells),len(faces)
    if nc!=report['cells'] or nf!=report['faces'] or not np.array_equal(cells[:,0],np.arange(nc)) or not np.array_equal(faces[:,0],np.arange(nf)):
        raise ValueError('Replay native topology or ordering differs')
    if np.any(cells[:,1]<=0) or np.any(faces[:,3]<=0):raise ValueError('Invalid native volumes or areas')
    volume=cells[:,1];area=faces[:,3];inside=faces[:,2]>=0
    ids=np.flatnonzero(inside)
    owner=faces[inside,1].astype(np.int64);neighbor=faces[inside,2].astype(np.int64)
    incidence=np.r_[owner,neighbor]
    order=np.argsort(incidence,kind='stable')
    offsets=np.r_[0,np.cumsum(np.bincount(incidence,minlength=nc))]
    dt=report['time_step']
    if dt!=prepared['probe_dt']:raise ValueError('Different intended replay time scale')
    if 'predictor_state' in config['projection_replay']:
        state=keep(config['projection_replay']['predictor_state'],checkpoint['state_sha256'])
        if report['predictor_source_state']!=str(state) or report['predictor_full_time_step']!=config['time_step'] or dt!=.5*config['time_step']:
            raise ValueError('Wrong source state or BCG predictor time step')
        flux_path=root/'predictor_before_projection.bin'
        if Path(report['input_flux']).resolve()!=flux_path:raise ValueError('Unexpected predictor output path')
    else:
        flux_path=Path(config['projection_replay']['flux'])
        if dt!=config['time_step']:raise ValueError('Flux replay time step differs')
    original=np.fromfile(keep(flux_path),dtype='<f8')
    if len(original)!=nf or not np.isfinite(original).all() or np.any(original[~inside]!=0):
        raise ValueError('Invalid source flux')
    velocity_scale=float(np.max(np.abs(original)/area))
    drive=max(math.sqrt(math.fsum(v*v for v in config['force'])),1e-30)*checkpoint['extent'][0]*dt
    through=float(columns(keep(parent/f"step_{checkpoint['physical_step']:04d}/sections.csv"),('volume_flux',)).mean())
    def balance(q):
        if len(q)!=nf or not np.isfinite(q).all() or np.any(q[~inside]!=0):raise ValueError('Invalid repaired flux')
        signed=np.r_[q[ids],-q[ids]][order]
        # math.fsum rounds once after accurately summing all incident doubles;
        # it is independent of Eigen multiplication and the C++ Neumaier loop.
        net=np.fromiter((math.fsum(signed[offsets[i]:offsets[i+1]]) for i in range(nc)),dtype=np.float64,count=nc)
        divergence=float(np.max(np.abs(net)/volume))
        relative=divergence/(float(np.max(np.abs(q)/area))/checkpoint['extent'][1])
        global_absolute=float(np.abs(net).sum()/abs(through))
        if divergence>1e-8 or relative>=1e-7 or global_absolute>=1e-8:
            raise ValueError('Actual independently summed face balance failed')
        return {'compensated_divergence_linf':divergence,'relative_linf':relative,
                'global_absolute_cell_flux_over_source_throughflow':global_absolute}
    rows=report['rows'];repetitions=config['projection_replay']['repetitions']
    if len(rows)!=2*repetitions:raise ValueError('Missing paired projection runs')
    pairs=[]
    for repeat in range(1,repetitions+1):
        pair=rows[2*(repeat-1):2*repeat]
        if [r['repeat'] for r in pair]!=[repeat,repeat] or [r['cycle_exit_enabled'] for r in pair]!=[False,True]:
            raise ValueError('Wrong scalar/cycle pair order')
        fluxes=[];pressures=[];mass=[]
        for row in pair:
            q=np.fromfile(keep(root/(row['output_prefix']+'_flux.bin')),dtype='<f8')
            p=np.fromfile(keep(root/(row['output_prefix']+'_pressure.bin')),dtype='<f8')
            if len(p)!=nc or not np.isfinite(p).all():raise ValueError('Invalid pressure repair')
            fluxes.append(q);pressures.append(p);mass.append(balance(q))
            if not math.isclose(mass[-1]['compensated_divergence_linf'],row['compensated_divergence_linf'],rel_tol=1e-12,abs_tol=0):
                raise ValueError('Independent balance differs from actual projection readback')
        difference=(fluxes[1]-fluxes[0])/area
        linf=float(np.max(np.abs(difference))/velocity_scale)
        l2=float(np.sqrt(np.sum(area*difference**2)/np.sum(area*(original/area)**2)))
        impulse=float(np.max(np.abs(pressures[1]-pressures[0]))*dt/config['rho']/drive)
        passed=linf<=64*sys.float_info.epsilon and l2<=1e-12 and impulse<=128*sys.float_info.epsilon
        if not passed:raise ValueError('Paired physical flux or pressure impulse differs beyond the roundoff envelope')
        pairs.append({'repeat':repeat,'passed':passed,'face_velocity_relative_linf':linf,
                      'area_weighted_face_velocity_relative_l2':l2,'impulse_difference_over_drive':impulse,
                      'mass':mass,'gpu_calls':[r['gpu_calls'] for r in pair],'cycle_exits':pair[1]['cycle_exits']})
    with keep(root/'gpu_linear.csv').open() as stream:linear=list(csv.DictReader(stream))
    if len(linear)!=sum(row['gpu_calls'] for row in rows):raise ValueError('Incomplete replay linear trace')
    for row in linear:
        if row['operator']!='pressure' or any(not 0<=float(row[k])<=1e-13 for k in ('true_relative_residual','compatibility_relative_l2')):
            raise ValueError('Replay linear solve violates the unchanged bound')
    with keep(root/'projection_cycle_exits.csv').open() as stream:events=list(csv.DictReader(stream))
    groups={}
    for row in events:groups.setdefault((int(row['call']),int(row['exit_pass'])),[]).append(row)
    for (call,last),window in groups.items():
        period=int(window[0]['period']);r=[float(w['divergence_linf']) for w in window]
        if period not in (2,3,4) or len(window)!=2*period or r[:period]!=r[period:] or len(set(r))<2:
            raise ValueError('Invalid cycle period in the actual exit trace')
        if [int(w['pass']) for w in window]!=list(range(last-2*period+1,last+1)):
            raise ValueError('Incomplete cycle history')
        for w in window:
            if not all(math.isfinite(float(v)) for v in w.values()):raise ValueError('Nonfinite cycle observation')
            if not (float(w['epsilon'])==sys.float_info.epsilon and 1e-10<=float(w['divergence_linf'])<=1e-8 and
                    float(w['flux_velocity_scale'])>0 and float(w['impulse_scale'])>0 and
                    0<=float(w['flux_velocity_change_linf'])<=.5*sys.float_info.epsilon*float(w['flux_velocity_scale']) and
                    0<=float(w['impulse_change_upper_linf'])<=.5*sys.float_info.epsilon*float(w['impulse_scale']) and
                    0<=float(w['exit_compensated_divergence_linf'])<=1e-8):
                raise ValueError('Actual cycle window violates its tighter bounds')
        matching=[r for r in rows if r['pressure_call']==call and r['cycle_exit_enabled'] and r['cycle_exits']==1]
        if len(matching)!=1 or float(window[-1]['exit_compensated_divergence_linf'])!=matching[0]['compensated_divergence_linf']:
            raise ValueError('Cycle trace does not match its completed replay')
    if len(groups)!=sum(r['cycle_exits'] for r in rows):raise ValueError('Missing cycle-exit events')
    if args.require_cycle and not groups:raise ValueError('Replay did not exercise the additional cycle rule')
    if any(sha(Path(p))!=digest for p,digest in hashes.items()):raise ValueError('Replay input changed during the check')
    result={'passed':True,'scope':__doc__,'pairs':pairs,'cycle_exits_verified':len(groups),
            'successful_pressure_calls':len(linear),'linear_relative_residual_max':max((float(r['true_relative_residual']) for r in linear),default=None),
            'source_sha256':hashes,'checker_sha256':sha(Path(__file__)),'goal_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
