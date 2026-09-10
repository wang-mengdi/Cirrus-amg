"""Measure dt versus dt/2 sensitivity of two completed, steady projection runs.

This compares identical cut-cell samples, so no spatial transfer is used.
Passing is a two-step-size sensitivity check, not a spatial accuracy claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_twisted import read, vector, error
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse',type=Path,required=True)
    parser.add_argument('--fine',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing timestep reports')
    roots=[args.coarse.resolve(),args.fine.resolve()]
    configs=[];data=[];sources=[Path(__file__)];mass=[]
    for root in roots:
        config=json.loads((root/'case.json').read_text())
        summary=json.loads((root/'transient_summary.json').read_text())
        completion=json.loads((root/'run_completion.json').read_text())
        runtime=json.loads((root/'run_manifest.json').read_text())
        if config.get('fluid_solver')!='proj' or not summary['steady_converged'] or summary['steps_completed']!=config['time_steps']:
            raise ValueError('Require completed steady projection runs')
        if completion['exit_code']!=0 or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
            raise ValueError('Run inputs changed or process failed')
        if sha(Path(runtime['executable']))!=runtime['executable_sha256']:raise ValueError('Executable changed')
        folder=root/f'step_{config["time_steps"]:04d}'
        metrics=json.loads((folder/'metrics.json').read_text())
        if max(metrics['steady_momentum_relative_l2'],metrics['temporal_acceleration_relative_l2'])>=1e-8:
            raise ValueError('Physical state is not steady')
        names=['solution.csv','walls.csv','flux.csv','mesh_faces.csv','sections.csv']
        cells,walls,flux,faces,sections=[read(folder/n) for n in names]
        if not np.array_equal(cells['id'],np.arange(len(cells))) or not np.array_equal(flux['id'],faces['id']):
            raise ValueError('Invalid native field ordering')
        q=flux['flux'];inner=faces['neighbor']>=0;net=np.zeros(len(cells))
        np.add.at(net,faces['owner'].astype(int),q)
        np.add.at(net,faces['neighbor'][inner].astype(int),-q[inner])
        through=float(np.mean(sections['volume_flux']))
        speed=float(np.linalg.norm(vector(cells,'uvw'),axis=1).max())
        geometry=Path(config['embedded_geometry']);meta=geometry.with_suffix('.meta.json')
        metadata=json.loads((meta if meta.exists() else geometry).read_text())
        div=float(np.max(abs(net)/cells['volume'])/(speed/metadata['extent'][1]))
        balance={'divergence_relative_linf':div,'absolute_cell_flux_over_throughflow':float(abs(net).sum()/abs(through)),
                 'section_flux_relative_spread':float(np.ptp(sections['volume_flux'])/abs(through))}
        balance['passed']=div<1e-7 and balance['absolute_cell_flux_over_throughflow']<1e-8 and balance['section_flux_relative_spread']<1e-8 and np.all(q[~inner]==0)
        mass.append(balance);configs.append(config);data.append((cells,walls,flux,faces,sections))
        sources += [root/n for n in ('case.json','transient_summary.json','run_manifest.json','run_completion.json')]
        sources += [folder/n for n in names+['metrics.json']]
    for k in ('rho','nu','force','ny','adaptive','embedded_geometry','fluid_solver','convection','convection_scheme','momentum_mode','wall_reconstruction'):
        if configs[0].get(k)!=configs[1].get(k):raise ValueError('Different spatial/physical problem: '+k)
    if not np.isclose(configs[0]['time_step'],2*configs[1]['time_step'],rtol=1e-13,atol=0):raise ValueError('Require a time-step halving')
    if not np.isclose(configs[0]['time_steps']*configs[0]['time_step'],configs[1]['time_steps']*configs[1]['time_step'],rtol=1e-13,atol=0):
        raise ValueError('Different final physical times')
    for slot,columns in [(0,['id','x','y','z','volume']),(1,['face_id','owner','x','y','z','area']),
                         (3,['id','owner','neighbor','area']),(4,['x','area'])]:
        if not np.array_equal(vector(data[0][slot],columns),vector(data[1][slot],columns)):raise ValueError('Different sample geometry')
    a,b=data[0][0],data[1][0];V=a['volume'];u,v=vector(a,'uvw'),vector(b,'uvw')
    cut=np.isin(a['id'],data[0][1]['owner']);p=a['p']-np.average(a['p'],weights=V);r=b['p']-np.average(b['p'],weights=V)
    wa,wb=data[0][1],data[1][1]
    fields={'velocity':error(u,v,V),'pressure':error(p,r,V),'cut_cell_velocity':error(u[cut],v[cut],V[cut]),
            'wall_shear':error(vector(wa,['tau_x','tau_y','tau_z']),vector(wb,['tau_x','tau_y','tau_z']),wa['area']),
            'section_flux':error(data[0][4]['volume_flux'],data[1][4]['volume_flux'],np.ones(len(data[0][4])))}
    faces=data[0][3];inside=faces['neighbor']>=0
    fields['face_normal_velocity']=error(data[0][2]['flux'][inside]/faces['area'][inside],data[1][2]['flux'][inside]/faces['area'][inside],faces['area'][inside])
    limits={k:.0025 for k in fields};limits['section_flux']=.00125
    passed=all(m['passed'] for m in mass) and all(fields[k]['relative_l2']<limits[k] for k in fields)
    report={'passed':bool(passed),'scope':__doc__,'time_steps':[c['time_step'] for c in configs],
            'relative_l2_limits':limits,'fields':fields,'mass':mass,'source_sha256':{str(p.resolve()):sha(p) for p in sources}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,default=lambda x:bool(x))+'\n')
    print(json.dumps({'passed':bool(passed),'fields':{k:v['relative_l2'] for k,v in fields.items()}}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
