"""Compare all retained physical fields of a chain continuation and a continuous control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from check_twisted_time_pair import read
from compare_twisted import vector,error
from twisted_chain_restart import physical_config
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();roots=[a.candidate.resolve(),a.reference.resolve()]
    if a.output.exists():raise ValueError('Preserve previous comparisons')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cfg=[json.loads((r/'case.json').read_text()) for r in roots]
    if physical_config(cfg[0])!=physical_config(cfg[1]) or cfg[0]['time_steps']!=cfg[1]['time_steps']:
        raise ValueError('Different physical configurations or final times')
    if not cfg[0].get('restart_checkpoint') or cfg[1].get('restart_checkpoint'):
        raise ValueError('Require chained candidate and continuous control')
    hashes={};native=[]
    for i,r in enumerate(roots):
        path=a.output.with_name(a.output.stem+f'_native_{i}.json')
        with path.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('check_twisted_chain_native_run.py')),
                            '--run',str(r),'--output',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
        j=json.loads(path.read_text());native.append(j)
        if not j['passed']:raise ValueError('Native trajectory validation failed')
        hashes.update(j['source_sha256']);hashes.update(j['executed_input_sha256']);hashes[str(path.resolve())]=sha(path)
    if native[0]['restart_prefix_validation']['ancestry_depth']<1:raise ValueError('No actual chained ancestry')
    results=[]
    for step in range(1,cfg[0]['time_steps']+1):
        folders=[r/f'step_{step:04d}' for r in roots]
        if not all(json.loads((r/'metrics.json').read_text())['field_output_written'] for r in folders):
            raise ValueError('This regression requires every physical step retained')
        cells=[read(r/'solution.csv') for r in folders];walls=[read(r/'walls.csv') for r in folders]
        flux=[read(r/'flux.csv') for r in folders]
        for arrays,cols in ((cells,('id','x','y','z','volume')),(walls,('face_id','x','y','z','area')),(flux,('id',))):
            if not np.array_equal(vector(arrays[0],cols),vector(arrays[1],cols)):raise ValueError('Field geometry or ordering differs')
        volume=cells[0]['volume'];u=[vector(c,('u','v','w')) for c in cells]
        pressure=[c['p']-np.average(c['p'],weights=volume) for c in cells]
        cut=np.isin(cells[0]['id'],walls[0]['owner'])
        fields={'velocity':error(*u,volume),'pressure':error(*pressure,volume),
                'cut_cell_velocity':error(u[0][cut],u[1][cut],volume[cut]),
                'wall_shear':error(*[vector(w,('tau_x','tau_y','tau_z')) for w in walls],walls[0]['area']),
                'shared_face_flux':error(flux[0]['flux'],flux[1]['flux'],np.ones(len(flux[0])))}
        results.append({'step':step,'passed':all(f['relative_l2']<1e-6 for f in fields.values()),**fields})
        for folder in folders:
            for name in ('solution.csv','walls.csv','flux.csv'):hashes[str(folder/name)]=sha(folder/name)
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Comparison inputs changed')
    result={'passed':all(row['passed'] for row in results),'scope':__doc__,'steps':results,'relative_l2_limit':1e-6,
            'native_checks':native,'source_sha256':hashes,'checker_sha256':sha(Path(__file__)),'goal_complete':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'steps':len(results),'ancestry_depth':native[0]['restart_prefix_validation']['ancestry_depth']}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
