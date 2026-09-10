"""Preserve projection acceleration, exact baseline cache checks, and failed accuracy gates."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    runs=repo/'output/twisted';base=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    target=args.output.resolve();target.mkdir(parents=True,exist_ok=False);receipt=[]
    def save(source,destination,snapshot=False):
        source=Path(source);data=source.read_bytes();name=Path(destination)
        packed=len(data)>2_000_000 and source.suffix in ('.csv','.json','.log','.mtx')
        content=gzip.compress(data,mtime=0) if packed else data
        if packed:name=Path(str(name)+'.gz')
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('Duplicate archive target')
        path.write_bytes(content)
        receipt.append({'path':name.as_posix(),'source':str(source.resolve()),'source_sha256':digest(data),
                        'sha256':digest(content),'size':len(content),'gzip':packed,'diagnostic_snapshot':snapshot})
    reports=('compare_native_proj16_aa5_v4.json','projection_ns16_aa5_time_pair_v4.json',
             'compare_native_proj16_aa5_v5.json','compare_native_proj32_aa5_v5.json',
             'projection_ns16_aa5_time_pair_v5.json','projection_ns32_aa5_time_pair_v5.json',
             'projection_v4_default_regression.json','aphros_projection_ns64_mass_v1.json',
             'aphros_projection_cache3_original_v1.json','aphros_projection_cache1_original_v1.json',
             'compare_native_proj64_adaptive_aa5_seamnorm_final_v4.json',
             'projection_ns64_seam_diagnostic_v1.json','projection_ns64_adaptive_aa5_time_pair_v4.json')
    for name in reports:save(runs/name,'reports/'+name)
    for version in (4,5):
        save(runs/f'projection_v{version}_build.json',f'builds/projection_v{version}_build.json')
        for source in (runs/f'projection_v{version}_sources').iterdir():
            save(source,f'builds/projection_v{version}_sources/'+source.name+'.txt')
    def native(name,steps,full=True):
        root=runs/name
        if json.loads((root/'run_completion.json').read_text())['exit_code']!=0:raise ValueError('Incomplete native run '+name)
        for source in root.iterdir():
            if source.is_file() and (source.suffix=='.json' or source.name in ('run.log','time_history.csv')):
                save(source,f'native/{name}/'+source.name)
        for step in steps:
            folder=root/f'step_{step:04d}';metrics=json.loads((folder/'metrics.json').read_text())
            names=['case.json','metrics.json','history.csv','acceleration.csv']
            if full:names+=['solution.csv','walls.csv','flux.csv','sections.csv','mesh_cells.csv','mesh_faces.csv',
                            f'iter_{metrics["iterations"]}/faces.csv','paraview_export.json','paraview_step_readback.json']
            for filename in names:
                if (folder/filename).exists():save(folder/filename,f'native/{name}/{folder.name}/'+filename)
    for n,ordinary in ((16,'ours_proj16_steps8_v3'),(32,'ours_proj32_steps8_v1')):
        native(f'ours_proj{n}_steps8_aa5_v5',range(1,9));native(ordinary,range(1,9))
    native('ours_proj32_steps8_aa5_v4',range(1,9),False)
    native('ours_proj32_steady_v1',(128,))
    native('ours_proj64_adaptive_probe_v3',(1,));native('ours_proj64_adaptive_aa5_v4',(1,))
    for name in ('projection_ns16_pressure_mechanism_v1','projection_ns16_32_refinement_v1'):
        for path in (runs/name).iterdir():
            if path.is_file():save(path,f'diagnostics/{name}/'+path.name)
    for name in ('navier_stokes_proj_n16_cache1_steps8_v1','navier_stokes_proj_n16_cache3_steps8_v1',
                 'navier_stokes_proj_n64_adaptive_probe_v1','navier_stokes_proj_n64_pressuretol16_v1'):
        root=base/name
        completion=json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))
        if name.endswith('pressuretol16_v1'):
            if completion['exit_code']==0:raise ValueError('Expected preserved tighter-tolerance failure')
        elif completion['exit_code']!=0:raise ValueError('Reference did not finish')
        for path in root.iterdir():
            if path.is_file() and path.suffix in ('.json','.csv','.conf','.log'):
                save(path,f'aphros/{name}/'+path.name)
    save(base/'projection_cache_build_v1/build_manifest.json','builds/projection_cache_build_v1.json')
    for path in (base/'linear_cache_sources_v1').iterdir():
        if path.is_file():save(path,'builds/linear_cache_sources_v1/'+path.name+'.txt')
    for name in ('ours_proj64_adaptive_probe_v3','ours_proj64_adaptive_aa5_v4','ours_proj16_eighthdt_steady_aa5_v5'):
        root=runs/name
        for filename in ('case.json','run_manifest.json','run.log','run_completion.json','projection_method.json',
                         'projection_pressure.csv','step_0001/acceleration.csv'):
            if (root/filename).exists():save(root/filename,f'continuing_snapshots/{name}/'+filename,True)
    for name in ('navier_stokes_proj_n32_steady_cache3_v1','navier_stokes_proj_n64_pressuretol15_cache3_v1'):
        for filename in ('a.conf','case_manifest.json','run_manifest.json'):
            save(base/name/filename,f'continuing_inputs/{name}/'+filename,True)
    scripts=('check_twisted_time_pair.py','compare_twisted.py','analyze_twisted_projection_pressure.py',
             'analyze_twisted_refinement.py','check_twisted_projection_cache_pair.py','check_twisted_paraview_step.py',
             'run_twisted_baseline.ps1','record_twisted_acceleration_checkpoint.py','check_twisted_projection_flux.py')
    for name in scripts:save(repo/'scripts'/name,'sources/scripts/'+name+'.txt')
    for name in ('twisted_direct.h','prepare_twisted_linear_cache.py','build_twisted_driver.ps1',
                 'twisted_projection_driver.cpp','LICENSE.aphros','LICENSE.amgcl'):
        save(repo/'validation/aphros'/name,'sources/aphros/'+name+'.txt')
    report={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':__doc__,'accuracy_status':'Incomplete: n16/32 spatial refinement and n16 timestep pressure gates fail; original Aphros64 cut-volume mass gate fails.',
            'files':receipt,'bytes':sum(row['size'] for row in receipt)}
    (target/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in receipt:
        content=(target/row['path']).read_bytes()
        if digest(content)!=row['sha256'] or digest(gzip.decompress(content) if row['gzip'] else content)!=row['source_sha256']:
            raise ValueError('Archive readback mismatch')
    print(json.dumps({'files':len(receipt),'bytes':report['bytes'],'output':str(target)}))


if __name__=='__main__':main()
