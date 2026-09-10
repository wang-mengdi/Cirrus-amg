"""Compare complete restarted tails using the same verified physical checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from check_twisted_prefix_time_pair import read
from compare_twisted import vector,error
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('candidate','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();roots=[a.candidate.resolve(),a.reference.resolve()]
    if a.output.exists():raise ValueError('Preserve previous tail comparisons')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    configs=[json.loads((r/'case.json').read_bytes()) for r in roots]
    if [{k:v for k,v in c.items() if k!='output'} for c in configs][0]!={k:v for k,v in configs[1].items() if k!='output'}:
        raise ValueError('Both tails must use identical physical and restart settings')
    if any(not c.get('restart_checkpoint') or c.get('operator_only') for c in configs):
        raise ValueError('Require actual physical continuation on both sides')
    methods=[json.loads((r/'projection_method.json').read_bytes()) for r in roots]
    if methods[0].get('pressure_roundoff_cycle_exit','disabled')=='disabled' or methods[1].get('pressure_roundoff_cycle_exit','disabled')!='disabled':
        raise ValueError('Expected the new cycle rule versus the original scalar-repeat rule')
    if any(m.get('pressure_iterate_storage')!='twofold' for m in methods):raise ValueError('Both sides must retain twofold pressure')
    hashes={};native=[]
    for index,root in enumerate(roots):
        runtime=json.loads((root/'run_manifest.json').read_bytes());build=Path(runtime['executable']).parent/'build_manifest.json'
        compiled=json.loads(build.read_bytes())
        if compiled['exit_code'] or not compiled['source_unchanged'] or compiled['executable_sha256']['simple_channel.exe']!=runtime['executable_sha256']:
            raise ValueError('Unverified native build')
        hashes[str(build)]=sha(build)
        for name,digest in compiled['source_sha256'].items():
            path=build.parent/'sources'/(name+'.txt')
            if sha(path)!=digest:raise ValueError('Compiled source snapshot changed')
            hashes[str(path)]=digest
        report=a.output.with_name(a.output.stem+f'_native_{index}.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('check_twisted_prefix_native_run.py')),
                        '--run',str(root),'--output',str(report)],check=True,stdout=subprocess.DEVNULL)
        result=json.loads(report.read_bytes());native.append(result)
        hashes.update(result['source_sha256']);hashes.update(result['executed_input_sha256']);hashes[str(report.resolve())]=sha(report)
    start=native[0]['restart_prefix_validation']['completed_parent_steps_used']
    if native[1]['restart_prefix_validation']['completed_parent_steps_used']!=start:raise ValueError('Different prefix lengths')
    comparisons=[]
    for step in range(start+1,configs[0]['time_steps']+1):
        folders=[r/f'step_{step:04d}' for r in roots]
        cells=[read(r/'solution.csv') for r in folders];walls=[read(r/'walls.csv') for r in folders];flux=[read(r/'flux.csv') for r in folders]
        for data,cols in ((cells,('id','x','y','z','volume')),(walls,('face_id','x','y','z','area')),(flux,('id',))):
            if not np.array_equal(vector(data[0],cols),vector(data[1],cols)):raise ValueError('Tail geometry differs')
        volume=cells[0]['volume'];u=[vector(c,('u','v','w')) for c in cells]
        pressure=[c['p']-np.average(c['p'],weights=volume) for c in cells]
        cut=np.isin(cells[0]['id'],walls[0]['owner'])
        fields={'velocity':error(*u,volume),'pressure':error(*pressure,volume),
                'cut_cell_velocity':error(u[0][cut],u[1][cut],volume[cut]),
                'wall_shear':error(*[vector(w,('tau_x','tau_y','tau_z')) for w in walls],walls[0]['area']),
                'shared_face_flux':error(flux[0]['flux'],flux[1]['flux'],np.ones(len(flux[0])))}
        comparisons.append({'step':step,'passed':all(f['relative_l2']<1e-6 for f in fields.values()),**fields})
        for r in folders:
            for name in ('solution.csv','walls.csv','flux.csv'):hashes[str(r/name)]=sha(r/name)
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Tail comparison inputs changed')
    result={'passed':all(r['passed'] for r in comparisons),'scope':__doc__,'parent_steps':start,
            'new_physical_steps':len(comparisons),'complete_restarted_tail_checked':True,
            'relative_l2_limit':1e-6,'steps':comparisons,'native_checks':native,'source_sha256':hashes,
            'checker_sha256':sha(Path(__file__)),'goal_complete':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'parent_steps':start,'new_steps':len(comparisons)}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
