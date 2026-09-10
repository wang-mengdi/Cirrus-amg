"""Preserve pressure-precision diagnostics and independent transient comparisons without claiming steady convergence."""
import argparse,datetime,gzip,json,shutil
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    runs=repo/'output/twisted';base=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[];retained=[]
    def save(p,dest,scope):
        before=sha(p);compress=p.stat().st_size>2_000_000 and p.suffix in ('.csv','.log','.json')
        target=out/(str(dest)+('.gz' if compress else ''));target.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,target.open('xb') as dst:
            if compress:
                with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as z:shutil.copyfileobj(src,z,1024*1024)
            else:shutil.copyfileobj(src,dst,1024*1024)
        if sha(p)!=before:raise ValueError('Archived source changed')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(p.resolve()),
                      'source_sha256':before,'sha256':sha(target),'size':target.stat().st_size,'gzip':compress,'scope':scope})
    def keep(p,scope):
        retained.append({'source':str(p.resolve()),'source_sha256':sha(p),'size':p.stat().st_size,'scope':scope})
    def tree(root,dest,scope,large=False):
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            if p.suffix in ('.exe','.bin','.vtu','.vtp','.o') or (large and p.suffix=='.csv'):
                keep(p,scope+'; retained runtime bytes')
            elif p.suffix in ('.json','.csv','.log','.txt','.py','.ps1','.conf','.cpp','.h','.ipp'):
                save(p,Path(dest)/p.relative_to(root),scope)
    tree(runs/'original_rhs_build_v1','build','current native compiled source and executable')
    for name,dest in (('navier_stokes_proj_n64_faces_capture_diff8_v1','reference64_capture'),
                      ('navier_stokes_proj_n128_volume_pressure_diff8_v1','reference128')):
        root=base/name
        if json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code']:
            raise ValueError('Required reference trajectory has not completed')
        tree(root,dest,'one completed physical step only; actual mass remains a separate check',large=True)
    for name in ('projection_faces_build_v1','projection_volume_build_v1'):
        tree(base/name,Path('reference_builds')/name,'immutable compiled reference driver/backend and source hashes')
    for name in ('aphros_extended_precision_probe64_v1','aphros_extended_precision_probe64_v2'):
        tree(runs/name,Path('local_precision_probes')/name,'local fixed-neighbor arithmetic only; v1 failed at runtime, v2 completed')
    for version in (1,3):
        tree(base/f'extended_template_probe_v{version}',Path('template_probes')/f'v{version}',
             'original Mesh and Embed Proj compile feasibility only; v1 failed, v3 compiled two objects; no linked solver')
    save(base/'extended_template_probe_v2/probe_source.py.txt','template_probes/v2/probe_source.py.txt',
         'actual preparation attempt; failed its literal-count guard before compilation')
    for name in ('aphros_faces_n16_cells_v1','aphros_faces_n16_cells_v2','aphros_faces_n64_diagnostic_v1',
                 'aphros_faces_n64_rows_v1','aphros_faces_n64_cells_v1','aphros_faces_n64_native_prefix_pair_v1',
                 'aphros_faces_n64_native_prefix_pair_v2','native_prefix_default_n16_pair_v1',
                 'native_prefix_no_transient_guard_v1','native_prefix_scope_guard_v1',
                 'native_prefix_default_pending_guard_v1','native_prefix_default_guard_v1',
                 'aphros_extended_precision_probe64_v1','aphros_extended_precision_probe64_v2',
                 'aphros_extended_templates_v1','aphros_extended_templates_v2','aphros_extended_templates_v3',
                 'aphros_volume_n128_full_gpu_pair_v1'):
        for ext in ('.json','.log'):
            p=runs/(name+ext)
            if p.exists():save(p,Path('reports')/p.name,'actual diagnostic/comparison attempt; failed reports are retained explicitly')
    required=('aphros_faces_n64_diagnostic_v1.json','aphros_faces_n64_rows_v1.json',
              'aphros_faces_n64_cells_v1.json','aphros_faces_n64_native_prefix_pair_v2.json',
              'native_prefix_default_n16_pair_v1.json','native_prefix_scope_guard_v1.json',
              'native_prefix_default_guard_v1.json','aphros_volume_n128_full_gpu_pair_v1.json')
    for name in required:
        if not (runs/name).exists():raise ValueError('Missing required evidence: '+name)
    if not json.loads((runs/'aphros_extended_precision_probe64_v2/report.json').read_text())['passed']:
        raise ValueError('Extended local arithmetic feasibility not established')
    if not json.loads((base/'extended_template_probe_v3/report.json').read_text())['passed']:
        raise ValueError('Extended original-template compilation not established')
    for name in ('ours_proj64_steady_original_rhs_v1','ours_proj128_steady_original_rhs_v1'):
        for field in ('case.json','run_manifest.json'):
            save(runs/name/field,Path('pending_native')/name/field,'immutable running-trajectory inputs only; no completion claim')
    save(runs/'config_ours_proj128_steady_original_rhs_v1.json','configs/native128_steady.json','new actual steady-trajectory configuration')
    for name in ('compare_twisted.py','validate_twisted_native_step.py','inspect_aphros_pressure_cells.py',
                 'check_aphros_pressure_faces.py','check_aphros_pressure_snapshot.py','check_twisted_mass.py',
                 'check_twisted_projection_flux.py','run_aphros_precision_probe.py','run_twisted_solver.py',
                 'run_twisted_baseline.ps1','record_twisted_precision_checkpoint.py','check_twisted_checkpoint.py'):
        save(repo/'scripts'/name,Path('checkers')/name,'reproduction and verification source')
    for name in ('pressure_precision_probe.cpp','probe_extended_templates.py','LICENSE.aphros','LICENSE.amgcl'):
        save(repo/'validation/aphros'/name,Path('reproduction')/name,'local arithmetic source or upstream license')
    receipt={'scope':__doc__,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'goal_complete':False,'remaining':'Reference cut-cell arithmetic precision; independent adaptive steady comparison; physical grid/time convergence.',
             'files':files,'retained_runtime_files':retained}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files':len(files),'retained':len(retained),'bytes':sum(f['size'] for f in files)}),flush=True)


if __name__=='__main__':main()
