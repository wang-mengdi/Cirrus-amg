"""Compare a completed projection diagnostic with independent original Aphros.

This validates the uniform-grid diagnostic prototype only. It does not certify
native C++ projection, adaptive interfaces, a steady state, or grid convergence.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from check_twisted_mass import read,vector,check_case
from compare_twisted import error
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve old comparison reports')
    audit=json.loads((args.audit/'projection_audit.json').read_text())
    cfg=json.loads((args.aphros/'case_manifest.json').read_text())
    runtime=json.loads((args.aphros/'run_manifest.json').read_text(encoding='utf-8-sig'))
    completion=json.loads((args.aphros/'run_completion.json').read_text(encoding='utf-8-sig'))
    if completion['exit_code']!=0 or not audit['completed_requested_transient'] or cfg['fluid_solver']!='proj':
        raise ValueError('Both independent projection runs must be complete')
    if sha(args.aphros/'a.conf')!=cfg['config_sha256'] or runtime['config_sha256']!=cfg['config_sha256']:
        raise ValueError('Reference configuration changed')
    executable=Path(runtime['executable']);build_path=executable.parent/'build_manifest.json'
    build=json.loads(build_path.read_text(encoding='utf-8-sig'))
    if sha(executable)!=runtime['executable_sha256'] or build['executable_sha256']!=runtime['executable_sha256']:
        raise ValueError('Executed reference binary differs from the build')
    if sha(Path(build['library']))!=build['library_sha256']:raise ValueError('Linked baseline library changed')
    recovered_sources={}
    for path,value in audit['source_sha256'].items():
        if sha(Path(path))!=value:
            snapshot=args.audit/'executed_source.py'
            if Path(path).name=='audit_twisted_projection.py' and snapshot.exists() and sha(snapshot)==value:
                recovered_sources[path]=str(snapshot.resolve())
            else:raise ValueError('Prototype input or executed script changed: '+path)
    paths=[Path(p) for p in audit['source_sha256']]
    meshpath=next(p for p in paths if p.name=='mesh_faces.csv')
    faces=read(meshpath);history=audit['history'];steps=cfg['time_steps']
    if len(history)!=steps:raise ValueError('Wrong number of completed physical steps')
    temporal=read(args.aphros/'tube_b0_time.csv')
    if len(temporal)!=steps or not np.allclose(temporal['time'],[r['time'] for r in history],atol=1e-13,rtol=1e-13):
        raise ValueError('Physical times differ')
    counts=[];last=0
    for value in re.findall(r'\.\.\.\.\.iter=(\d+),', (args.aphros/'run.log').read_text()):
        current=int(value)
        if current==1 and last:counts.append(last)
        last=current
    counts.append(last)
    folder=args.audit/f'step_{steps:04d}'
    ours=read(folder/'solution.csv');reference=read(args.aphros/'proj_final_b0_cells.csv')
    distance,index=cKDTree(vector(reference,['x','y','z'])).query(vector(ours,['x','y','z']))
    if max(distance)>1e-13 or len(set(index))!=len(ours) or len(ours)!=len(reference):raise ValueError('Cell coordinates differ')
    reference=reference[index];V=ours['volume']
    if not np.allclose(V,reference['volume'],rtol=1e-12,atol=0):raise ValueError('Cut volumes differ')
    u,ref_u=vector(ours,'uvw'),vector(reference,'uvw')
    p=ours['p']-np.average(ours['p'],weights=V);ref_p=reference['p']-np.average(reference['p'],weights=V)
    wall=faces['neighbor']<0;cut=faces['owner'][wall].astype(int)
    a,b=read(folder/'walls.csv'),read(args.aphros/'tube_final_b0_walls.csv')
    distance,index=cKDTree(vector(b,['x','y','z'])).query(vector(a,['x','y','z']))
    if max(distance)>1e-13 or len(a)!=len(b) or len(set(index))!=len(a):raise ValueError('Wall points differ')
    b=b[index]
    if not np.allclose(a['area'],b['area'],rtol=1e-12,atol=0):raise ValueError('Wall areas differ')
    fields={'velocity':error(u,ref_u,V),'pressure':error(p,ref_p,V),
        'cut_cell_velocity':error(u[cut],ref_u[cut],V[cut]),
        'wall_shear':error(vector(a,['tau_x','tau_y','tau_z']),vector(b,['tau_x','tau_y','tau_z']),a['area'])}
    flux=read(folder/'flux.csv')['flux'];ref_faces=read(args.aphros/'proj_final_b0_faces.csv')
    if len(flux)!=len(faces) or np.any(flux[wall]!=0):raise ValueError('Prototype wall flux is not zero')
    inner=np.flatnonzero(~wall);ref_flux=np.zeros(len(faces))
    for d in range(3):
        rows=inner[faces['axis'][inner]==d];refs=np.flatnonzero(ref_faces['axis']==d)
        x,y=vector(faces,['x','y','z'])[rows],vector(ref_faces,['x','y','z'])[refs]
        x[:,0]%=cfg['spec']['extent'][0];y[:,0]%=cfg['spec']['extent'][0]
        distance,index=cKDTree(y).query(x);refs=refs[index]
        if max(distance)>1e-13 or not np.allclose(faces['area'][rows],ref_faces['area'][refs],rtol=1e-12,atol=0):
            raise ValueError('Face aperture correspondence failed')
        ref_flux[rows]=ref_faces['flux'][refs]
    fields['face_normal_velocity']=error(flux[inner]/faces['area'][inner],ref_flux[inner]/faces['area'][inner],faces['area'][inner])
    mass=check_case(args.aphros)
    net=np.zeros(len(ours));absolute=net.copy()
    np.add.at(net,faces['owner'].astype(int),flux)
    np.add.at(net,faces['neighbor'][inner].astype(int),-flux[inner])
    np.add.at(absolute,faces['owner'].astype(int),abs(flux))
    np.add.at(absolute,faces['neighbor'][inner].astype(int),abs(flux[inner]))
    axial=inner[faces['axis'][inner]==0];positions=np.mod(np.rint(faces['x'][axial]/ours['h'][0]).astype(int),cfg['shape'][0])
    section=np.bincount(positions,weights=flux[axial],minlength=cfg['shape'][0]);Q=float(np.mean(section))
    rate=float(np.linalg.norm(u,axis=1).max()/cfg['spec']['extent'][1])
    native_mass={'divergence_relative_linf':float(max(abs(net)/V)/rate),
        'divergence_linf':float(max(abs(net)/V)),'section_flux_mean':Q,
        'section_flux_relative_spread':float(np.ptp(section)/abs(Q)),
        'global_absolute_cell_flux_over_throughflow':float(abs(net).sum()/abs(Q))}
    native_mass['passed']=native_mass['divergence_relative_linf']<1e-7 and native_mass['section_flux_relative_spread']<1e-8 and native_mass['global_absolute_cell_flux_over_throughflow']<1e-8
    flow_error=abs(Q/mass['section_flux_mean']-1)
    time_error=float(np.max(abs(np.array([r['temporal_acceleration_l2'] for r in history])-temporal['temporal_acceleration_volume_l2'])))
    passed=all(r['relative_l2']<1e-6 for r in fields.values()) and flow_error<1e-6 and mass['passed'] and native_mass['passed'] and counts==[r['iterations'] for r in history] and time_error<1e-9
    report={'scope':__doc__,'passed':bool(passed),'fields':fields,'flow_relative_difference':flow_error,
        'prototype_mass':native_mass,'aphros_mass':mass,'iteration_counts_equal':counts==[r['iterations'] for r in history],
        'iteration_counts':counts,'temporal_acceleration_absolute_max_difference':time_error,
        'reference_runtime':runtime,'reference_build':build,'verified_executed_source_snapshots':recovered_sources}
    sources=[args.audit/'projection_audit.json',args.aphros/'case_manifest.json',args.aphros/'run_completion.json',args.aphros/'run.log',args.aphros/'tube_b0_time.csv',
        folder/'solution.csv',folder/'flux.csv',folder/'walls.csv',args.aphros/'proj_final_b0_cells.csv',args.aphros/'proj_final_b0_faces.csv',args.aphros/'tube_final_b0_walls.csv',build_path,Path(__file__)]
    report['source_sha256']={str(p.resolve()):sha(p) for p in sources}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','reference_runtime','reference_build','aphros_mass')},indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
