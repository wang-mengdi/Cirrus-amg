"""Archive native projection evidence, preserving exact runtime bytes and hashes.

Large CSVs are stored as deterministic gzip streams; the receipt records both
the archived bytes and the decompressed source bytes. Live-run files are marked
as diagnostic snapshots and never certify a completed physical step.
"""
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
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];runs=repo/'output/twisted'
    baseline=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    target=args.output.resolve();target.mkdir(parents=True,exist_ok=False)
    receipt=[]
    def save(source,destination,snapshot=False):
        source=Path(source);data=source.read_bytes();name=Path(destination)
        packed=len(data)>2_000_000 and source.suffix in ('.csv','.json','.log')
        output=gzip.compress(data,mtime=0) if packed else data
        if packed:name=Path(str(name)+'.gz')
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('Duplicate archive target '+str(name))
        path.write_bytes(output)
        receipt.append({'path':name.as_posix(),'source':str(source.resolve()),'source_sha256':digest(data),
                        'sha256':digest(output),'size':len(output),'gzip':packed,'diagnostic_snapshot':snapshot})
    for name in ('compare_native_proj16_steps8_verified_v2.json','compare_native_proj16_steady_verified_v2.json',
                 'compare_native_proj32_steps8_v1.json','simple_projection_v1_regression.json',
                 'projection_v2_regression.json','projection_v3_uniform_regression.json',
                 'projection_ns16_timestep_v1.json','projection_incremental_pressure_equivalence_v3.json'):
        save(runs/name,'reports/'+name)
    for version in (1,2,3):
        save(runs/f'projection_v{version}_build.json',f'builds/projection_v{version}_build.json')
        for source in (runs/f'projection_v{version}_sources').iterdir():
            save(source,f'builds/projection_v{version}_sources/'+source.name+'.txt')
    def native(name,steps):
        root=runs/name
        completion=json.loads((root/'run_completion.json').read_text())
        if completion['exit_code']!=0:raise ValueError('Expected successful native run '+name)
        for source in root.iterdir():
            if source.is_file() and (source.suffix=='.json' or source.name in ('time_history.csv','run.log')):
                save(source,f'native/{name}/'+source.name)
        for step in steps:
            folder=root/f'step_{step:04d}'
            for filename in ('case.json','metrics.json','history.csv','solution.csv','walls.csv','flux.csv','sections.csv','mesh_cells.csv','mesh_faces.csv'):
                save(folder/filename,f'native/{name}/{folder.name}/'+filename)
    native('ours_proj16_steps8_v1',range(1,9))
    native('ours_proj32_steps8_v1',(1,8))
    native('ours_proj16_steady_v1',(1,128))
    native('ours_proj16_halfdt_steady_v1',(256,))
    native('ours_proj16_steps8_v3',(8,))
    if (runs/'projection_ns16_timestep_v2.json').exists():
        save(runs/'projection_ns16_timestep_v2.json','reports/projection_ns16_timestep_v2.json')
        native('ours_proj16_quarterdt_steady_v3',(512,))
    for name in ('navier_stokes_proj_n16_implicit_probe_v1','navier_stokes_proj_n16_steady_probe_v1','navier_stokes_proj_n32_implicit_probe_v1'):
        root=baseline/name
        if json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code']!=0:raise ValueError('Incomplete reference')
        for filename in ('a.conf','case_manifest.json','run_manifest.json','run_completion.json','run.log',
                         'proj_final_b0_cells.csv','proj_final_b0_faces.csv','tube_final_b0_walls.csv','tube_b0_time.csv',
                         'tube_b0_geometry_cells.csv','tube_b0_geometry_faces.csv','tube_b0_geometry_walls.csv'):
            save(root/filename,f'aphros/{name}/'+filename)
    save(baseline/'projection_driver_build_v3/build_manifest.json','aphros/build_manifest.json')
    for name in ('ours_proj64_adaptive_probe_v1','ours_proj64_pressure_trace_v2','ours_proj64_adaptive_probe_v3'):
        root=runs/name
        for filename in ('case.json','run_manifest.json','run_completion.json','intentional_termination.json','run.log',
                         'projection_pressure.csv','embedded_operator_checks.json','projection_interface_checks.json','projection_method.json'):
            if (root/filename).exists():save(root/filename,f'interface_diagnostics/{name}/'+filename,True)
        if name!='ours_proj64_pressure_trace_v2':
            for filename in ('mesh_cells.csv','step_0001/iter_1/cells.csv','step_0001/iter_1/faces.csv'):
                save(root/filename,f'interface_diagnostics/{name}/'+filename,True)
    # Prescribed live reference inputs are immutable; a running log is omitted.
    root=baseline/'navier_stokes_proj_n64_adaptive_probe_v1'
    for filename in ('a.conf','case_manifest.json','run_manifest.json'):
        save(root/filename,'pending_aphros64/'+filename,True)
    for name in ('ours_proj32_steady_v1','ours_proj16_quarterdt_steady_v3'):
        root=runs/name
        for filename in ('case.json','run_manifest.json'):
            save(root/filename,f'pending_inputs/{name}/'+filename,True)
    for filename in ('compare_twisted.py','check_twisted_projection_flux.py','check_twisted_projection_timestep.py',
                     'check_twisted_projection_iteration.py','record_twisted_projection_checkpoint.py','run_twisted_solver.py'):
        save(repo/'scripts'/filename,'scripts/'+filename+'.txt')
    save(repo/'validation/aphros/twisted_projection_driver.cpp','aphros/twisted_projection_driver.cpp.txt')
    save(repo/'validation/aphros/LICENSE.aphros','aphros/LICENSE.aphros')
    report={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Native projection uniform transient/steady agreement; failed dt/2 pressure sensitivity; improved interface pressure iteration. Adaptive Aphros agreement and spatial convergence remain incomplete.',
            'files':receipt,'bytes':sum(r['size'] for r in receipt)}
    (target/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in receipt:
        content=(target/row['path']).read_bytes()
        if digest(content)!=row['sha256'] or digest(gzip.decompress(content) if row['gzip'] else content)!=row['source_sha256']:
            raise ValueError('Archive readback mismatch')
    print(json.dumps({'files':len(receipt),'bytes':report['bytes'],'output':str(target)}))


if __name__=='__main__':main()
