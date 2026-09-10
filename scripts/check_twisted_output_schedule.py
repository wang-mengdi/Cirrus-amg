"""Verify sparse field output preserves all step diagnostics and retained fields.

Only retained steps permit independent field/flux comparisons. Omitted fields
are never presented as checked, and this is not a steady or grid accuracy test.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from check_twisted_mass import read, vector
from compare_twisted import error
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sparse',type=Path,required=True)
    parser.add_argument('--dense',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous schedule checks')
    roots=[args.sparse.resolve(),args.dense.resolve()]
    configs=[json.loads((r/'case.json').read_text()) for r in roots]
    stride=configs[0].get('output_stride',1)
    if stride<=1 or configs[1].get('output_stride',1)!=1:raise ValueError('Expected sparse versus dense output')
    normalized=[{k:v for k,v in c.items() if k not in ('output','output_stride')} for c in configs]
    if normalized[0]!=normalized[1]:
        raise ValueError('Different physical or numerical configurations')
    steps=configs[0]['time_steps'];kept=[i for i in range(1,steps+1) if i==1 or i==steps or i%stride==0]
    sources=[];histories=[];trace_checks=[];executables=[]
    for root in roots:
        summary=json.loads((root/'transient_summary.json').read_text())
        completion=json.loads((root/'run_completion.json').read_text())
        runtime=json.loads((root/'run_manifest.json').read_text())
        if not summary['converged'] or summary['steps_completed']!=steps:raise ValueError('Incomplete time sequence')
        if completion['exit_code'] or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
            raise ValueError('Changed input or failed run')
        if sha(Path(runtime['executable']))!=runtime['executable_sha256']:raise ValueError('Actual executable changed')
        executables.append(runtime['executable_sha256'])
        history=list(csv.DictReader((root/'time_history.csv').open()));histories.append(history)
        if len(history)!=steps or any(r['inner_converged']!='true' for r in history):raise ValueError('Incomplete step history')
        sources += [root/n for n in ('case.json','transient_summary.json','run_manifest.json','run_completion.json','time_history.csv')]
        if configs[0].get('linear_backend')=='native_gpu':
            trace=list(csv.DictReader((root/'gpu_linear.csv').open()))
            if set(v['operator'] for v in trace)!={'pressure','diffusion'}:raise ValueError('Missing GPU operator trace')
            residual=np.array([float(v['true_relative_residual']) for v in trace])
            compatibility=np.array([float(v['compatibility_relative_l2']) for v in trace])
            if not np.isfinite(residual).all() or not np.isfinite(compatibility).all() or min(residual)<0 or min(compatibility)<0 or\
                    max(residual)>configs[0]['linear_tolerance'] or max(compatibility)>configs[0]['linear_tolerance']:
                raise ValueError('GPU true residual or compatibility failed')
            trace_checks.append({'calls':len(trace),'max_true_residual':float(max(residual)),'max_compatibility':float(max(compatibility))})
            sources.append(root/'gpu_linear.csv')
    summaries=[json.loads((r/'transient_summary.json').read_text()) for r in roots]
    if len(set(executables))!=1:raise ValueError('Output schedule comparison requires the same actual executable')
    if [s['output_stride'] for s in summaries]!=[stride,1]:raise ValueError('Summary and configured output stride differ')
    if summaries[0]['field_output_steps']!=kept or summaries[1]['field_output_steps']!=list(range(1,steps+1)):
        raise ValueError('Incorrect declared field schedule')
    for step,rows in enumerate(zip(*histories),1):
        for row in rows:
            if int(row['step'])!=step or not np.isclose(float(row['time']),step*configs[0]['time_step'],rtol=1e-13,atol=0):
                raise ValueError('Incorrect physical step/time')
        if rows[0]['inner_iterations']!=rows[1]['inner_iterations']:raise ValueError('Output schedule changed iteration count')
        for name in ('temporal_acceleration_relative_l2','steady_momentum_relative_l2'):
            if not np.isclose(float(rows[0][name]),float(rows[1][name]),rtol=1e-10,atol=1e-12):
                raise ValueError('Output schedule changed step dynamics')
    fields=[];links=[];metric_comparisons=[]
    for step in range(1,steps+1):
        folders=[r/f'step_{step:04d}' for r in roots]
        metrics=[json.loads((p/'metrics.json').read_text()) for p in folders]
        for m in metrics:
            if not m['converged'] or m['implicit_diffusion_relative_l2']>=configs[0]['tolerance'] or\
                    m.get('complete_inner_fixed_point_residual',float('inf'))>=configs[0]['tolerance'] or\
                    m['continuity_relative_linf']>=1e-7 or m['cross_section_flux_relative_spread']>=1e-8 or\
                    m['velocity_change_absolute_linf']>=configs[0]['projection_iteration_tolerance']:
                raise ValueError('Original physical-step gates failed')
        if metrics[0]['field_output_written']!=(step in kept) or not metrics[1]['field_output_written']:
            raise ValueError('Per-step field flag differs from schedule')
        differences={}
        for name in ('volume','wall_area','speed_max','volume_flux','mean_wall_shear_magnitude'):
            a,b=[m[name] for m in metrics]
            if not np.isclose(a,b,rtol=1e-10,atol=1e-14):raise ValueError('Output schedule changed physical metric: '+name)
            differences[name]=abs(a-b)
        metric_comparisons.append({'step':step,'absolute_differences':differences})
        for folder in folders:sources += [folder/'metrics.json',folder/'sections.csv']
        if step not in kept:
            if any((folders[0]/name).exists() for name in ('solution.csv','walls.csv','flux.csv','native_fields.bin','mesh_cells.csv','mesh_faces.csv')):
                raise ValueError('Unexpected unscheduled full field')
            continue
        cells=[read(p/'solution.csv') for p in folders];walls=[read(p/'walls.csv') for p in folders]
        flux=[read(p/'flux.csv') for p in folders];face=read(folders[0]/'mesh_faces.csv')
        for data,keys in ((cells,('id','level','x','y','z','h','volume')),(walls,('face_id','owner','x','y','z','area')),(flux,('id',))):
            if not np.array_equal(vector(data[0],keys),vector(data[1],keys)):raise ValueError('Different field geometry or ordering')
        volume=cells[0]['volume'];cut=np.isin(cells[0]['id'],walls[0]['owner'])
        velocities=[vector(c,'uvw') for c in cells]
        pressure=[c['p']-np.average(c['p'],weights=volume) for c in cells]
        checks={'velocity':error(*velocities,volume),'pressure':error(*pressure,volume),
                'cut_cell_velocity':error(velocities[0][cut],velocities[1][cut],volume[cut]),
                'wall_shear':error(*[vector(w,('tau_x','tau_y','tau_z')) for w in walls],walls[0]['area']),
                'face_flux':error(*[f['flux'] for f in flux],face['area'])}
        if any(v['relative_l2']>=1e-11 for v in checks.values()):raise ValueError('Output schedule changed retained fields')
        mass=[];inner=face['neighbor']>=0
        for c,f,u in zip(cells,flux,velocities):
            net=np.zeros(len(c));np.add.at(net,face['owner'].astype(int),f['flux'])
            np.add.at(net,face['neighbor'][inner].astype(int),-f['flux'][inner])
            geometry=json.loads(Path(configs[0]['embedded_geometry']).read_text())
            div=float(np.max(abs(net)/volume)/(np.linalg.norm(u,axis=1).max()/geometry['extent'][1]))
            if div>=1e-7 or np.any(f['flux'][~inner]!=0):raise ValueError('Independent retained face mass failed')
            mass.append(div)
        fields.append({'step':step,'quantities':checks,'independent_divergence_relative_linf':mass})
        for root,folder in zip(roots,folders):
            for name in ('mesh_cells.csv','mesh_faces.csv'):
                if sha(root/name)!=sha(folder/name):raise ValueError('Shared mesh differs from immutable root')
                links.append({'path':str(folder/name),'hard_link':(root/name).samefile(folder/name)})
            sources += [folder/n for n in ('solution.csv','walls.csv','flux.csv','mesh_cells.csv','mesh_faces.csv')]
    report={'passed':True,'scope':__doc__,'physical_steps':steps,'retained_field_steps':kept,
            'diagnostics_compared_at_every_step':metric_comparisons,'retained_fields':fields,
            'relative_l2_limit':1e-11,'mesh_storage':links,'gpu_linear':trace_checks,
            'source_sha256':{str(p):sha(p) for p in sources}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'physical_steps':steps,'retained_field_steps':kept,
                      'mesh_hard_links':sum(v['hard_link'] for v in links)}))


if __name__=='__main__':main()
