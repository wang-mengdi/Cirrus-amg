"""Archive output scheduling, uniform steady alignment, and fine-grid progress.

Only terminal run data are archived. Live runs contribute immutable inputs only.
Large audit vectors and failed 128 reference fields remain on disk with hashes.
This checkpoint does not establish adaptive steady, grid, or time convergence.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    runs=repo/'output/twisted'
    baseline=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    target=args.output.resolve()
    native_names=('ours_proj16_output_dense_v1','ours_proj16_output_stride3_v1')
    reference_names=('navier_stokes_proj_n32_steady_volume_pressure_v1',
                     'navier_stokes_proj_n32_volume_pressure_diff8_steps2_v1',
                     'navier_stokes_proj_n32_volume_pressure_default_steps2_v1')
    for root in [runs/n for n in native_names]+[runs/'ours_proj32_steady_v1']+[baseline/n for n in reference_names]:
        complete=json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))
        if complete['exit_code'] or any(not complete.get(k,True) for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
            raise ValueError('Incomplete or changed run: '+str(root))
    target.mkdir(parents=True,exist_ok=False)
    records=[];retained=[]

    def save(source,destination,scope):
        before=sha(source)
        compress=source.stat().st_size>2_000_000 and source.suffix in ('.csv','.json','.log')
        path=target/(str(destination)+('.gz' if compress else ''))
        path.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,path.open('xb') as dst:
            if compress:
                with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as zipped:
                    shutil.copyfileobj(src,zipped,1024*1024)
            else:shutil.copyfileobj(src,dst,1024*1024)
        if sha(source)!=before:raise ValueError('Source changed while archiving: '+str(source))
        records.append({'path':path.relative_to(target).as_posix(),'source':str(source.resolve()),
                        'source_sha256':before,'sha256':sha(path),'size':path.stat().st_size,'gzip':compress,'scope':scope})

    def keep(source,scope):
        retained.append({'source':str(source.resolve()),'source_sha256':sha(source),
                         'size':source.stat().st_size,'scope':scope+'; runtime artifact retained on disk'})

    def tree(root,destination,scope,operator=False):
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            if p.suffix in ('.exe','.vtu','.vtp','.bin') or (operator and p.suffix=='.csv'):
                keep(p,scope)
            elif p.suffix in ('.csv','.json','.log','.txt','.py','.ps1','.cmd','.h','.cpp','.conf','.pvd'):
                save(p,Path(destination)/p.relative_to(root),scope)

    tree(runs/'output_schedule_build_v1','build','exact compiled source bytes and immutable executable')
    for name in native_names:tree(runs/name,Path('native_output')/name,'complete eight-step output-schedule test')
    for name in reference_names:tree(baseline/name,Path('aphros32')/name,'complete original Proj/Embed trajectory; scope recorded in each report')
    native32=runs/'ours_proj32_steady_v1'
    for p in sorted(native32.iterdir()):
        if p.is_file():save(p,Path('native32')/p.name,'complete native32 run metadata and root mesh')
    for step in (2,128):
        folder=native32/f'step_{step:04d}'
        for p in sorted(folder.iterdir()):
            if not p.is_file():continue
            if p.suffix in ('.bin','.vtu','.vtp'):keep(p,'native32 compared final fields and ParaView')
            else:save(p,Path('native32')/folder.name/p.name,'native32 compared completed physical step')
    for step in range(1,129):
        if step in (2,128):continue
        folder=native32/f'step_{step:04d}'
        for name in ('case.json','metrics.json','sections.csv'):
            save(folder/name,Path('native32')/folder.name/name,'complete native32 per-step diagnostics')
    tree(runs/'full_viscosity_operator128_v1','operator128','all-face operator and manufactured-solution audit; not full flow',operator=True)
    for suffix in ('.launch.json','.log'):
        save(runs/f'full_viscosity_operator128_v1{suffix}',Path('operator128')/f'launch{suffix}','complete operator audit launch and log')
    old=runs/'ours_proj128_full_pressure_gpu_v1'
    for p in sorted(old.glob('*')):
        if p.is_file() and p.suffix in ('.json','.log'):save(p,Path('interrupted_native128')/p.name,'interrupted before completing any physical step')
    for name in ('acceleration.csv','history.csv'):
        p=old/'step_0001'/name
        if p.exists():save(p,Path('interrupted_native128/step_0001')/name,'terminal interrupted-run history')
    for iteration in (1,2):
        for name in ('cells.csv','faces.csv'):keep(old/'step_0001'/f'iter_{iteration}'/name,'immutable interrupted intermediate fields')
    failed=baseline/'navier_stokes_proj_n128_implicit_amg_material_v1'
    if json.loads((failed/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code']:
        raise ValueError('Expected completed but physically failed reference')
    for p in sorted(failed.iterdir()):
        if not p.is_file():continue
        if p.suffix=='.csv':keep(p,'original128 reference failed actual mass gate')
        else:save(p,Path('failed_aphros128')/p.name,'reference process completed but actual mass gate failed')
    for name in ('config_ours_proj16_output_dense_v1.json','config_ours_proj16_output_stride3_v1.json',
                 'config_full_viscosity_gpu_audit128_v1.json','config_ours_proj128_full_viscosity_gpu_v1.json'):
        save(runs/name,Path('configs')/name,'exact tested or launched configuration')
    for name in ('ours_proj64_steady_full_viscosity_gpu_v1','ours_proj128_full_viscosity_gpu_v1'):
        for field in ('case.json','run_manifest.json'):
            p=runs/name/field
            # A large case may still be in mesh construction before writing case.json.
            if p.exists():save(p,Path('pending_native')/name/field,'immutable input only; no completion claim')
    for name in ('navier_stokes_proj_n64_steady_volume_pressure_v1','navier_stokes_proj_n128_volume_pressure_diff8_v1'):
        for field in ('a.conf','case_manifest.json','run_manifest.json'):
            save(baseline/name/field,Path('pending_aphros')/name/field,'immutable input only; no completion claim')
    report_stems=('output_schedule_n16_pair_v1','output_schedule_default_n16_cpu_pair_v1',
                  'aphros_volume_pressure_n32_steady_pair_v1','aphros_volume_pressure_n32_steady_pair_v2',
                  'aphros_volume_pressure_n32_steady_pair_v3','aphros_volume_pressure_n32_steady_pair_v4',
                  'aphros_volume_pressure_n32_steady_diagnostic_v1','aphros_diffusion8_n32_steps2_pair_v1',
                  'aphros_diffusion8_n32_native_pair_v1','aphros_diffusion8_n32_native_pair_v2',
                  'aphros_diffusion8_explicit_guard_v1','aphros_projection_n128_material_mass_v1')
    for stem in report_stems:
        for extension in ('.json','.log'):
            p=runs/(stem+extension)
            if p.exists():save(p,Path('reports')/p.name,'complete report or preserved failed attempt; inspect passed/scope')
    for name in ('compare_twisted.py','check_twisted_mass.py','check_twisted_projection_flux.py',
                 'check_twisted_time_pair.py','check_aphros_pressure_snapshot.py','check_aphros_diffusion_iterations.py',
                 'check_twisted_output_schedule.py','run_twisted_solver.py','run_twisted_baseline.ps1',
                 'make_twisted_baseline.py','export_twisted_paraview.py','check_twisted_paraview_step.py',
                 'export_twisted_time_series.py','check_twisted_paraview.py','record_twisted_output_checkpoint.py'):
        save(repo/'scripts'/name,Path('checkers')/name,'reproduction source')
    for name in ('LICENSE.aphros','LICENSE.amgcl'):
        save(repo/'validation/aphros'/name,Path('licenses')/name,'upstream license')
    library=baseline/'aphros/src/libaphros_static.lib'
    if sha(library)!='b8f2bbf83c69678da30f99c79874de2989e9eef1b4a91a13f2b68a4382952732':
        raise ValueError('Original shared Aphros library changed')
    result={'scope':__doc__,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'goal_complete':False,'original_aphros_library_sha256':sha(library),
            'remaining':'Independent adaptive steady alignment and physical grid/time convergence.',
            'files':records,'retained_runtime_files':retained}
    (target/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(records),'bytes':sum(r['size'] for r in records),'retained_runtime_files':len(retained)}),flush=True)


if __name__=='__main__':main()
