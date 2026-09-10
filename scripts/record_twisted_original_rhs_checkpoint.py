"""Preserve complete original-RHS stopping tests and separately scoped reference progress."""
import argparse,datetime,gzip,json,shutil
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    runs=repo/'output/twisted';base=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    target=args.output.resolve();target.mkdir(parents=True,exist_ok=False)
    records=[];retained=[]
    def save(p,dest,scope):
        before=sha(p);compress=p.stat().st_size>2_000_000 and p.suffix in ('.csv','.json','.log')
        out=target/(str(dest)+('.gz' if compress else ''));out.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,out.open('xb') as dst:
            if compress:
                with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as zipped:shutil.copyfileobj(src,zipped,1024*1024)
            else:shutil.copyfileobj(src,dst,1024*1024)
        if sha(p)!=before:raise ValueError('Source changed while archiving')
        records.append({'path':out.relative_to(target).as_posix(),'source':str(p.resolve()),'source_sha256':before,
                        'sha256':sha(out),'size':out.stat().st_size,'gzip':compress,'scope':scope})
    def keep(p,scope):
        retained.append({'source':str(p.resolve()),'source_sha256':sha(p),'size':p.stat().st_size,'scope':scope})
    def tree(root,dest,scope,large_csv=False):
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            if p.suffix in ('.exe','.bin','.vtu','.vtp') or (large_csv and p.suffix=='.csv'):keep(p,scope+'; retained on disk')
            elif p.suffix in ('.json','.csv','.log','.txt','.py','.ps1','.conf'):
                save(p,Path(dest)/p.relative_to(root),scope)
    tree(runs/'original_rhs_build_v1','build','exact immutable compiled sources and binary hashes')
    for n in (16,64):
        for name in (f'original_rhs_operator{n}_v1',f'ours_proj{n}_original_rhs_v1'):
            root=runs/name;c=json.loads((root/'run_completion.json').read_text())
            if c['exit_code'] or not all(c[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
                raise ValueError('Required native test incomplete')
            tree(root,Path('native')/name,'complete configured operator/flow test; physical steady convergence is separate',name.startswith('original_rhs_operator'))
        for suffix in ('.launch.json','.log'):
            save(runs/f'original_rhs_operator{n}_v1{suffix}',Path('operator_launches')/f'{n}{suffix}','complete audit provenance')
        for stem in (f'config_ours_proj{n}_original_rhs_v1',f'config_full_viscosity_gpu_audit{n}_v1'):
            save(runs/(stem+'.json'),Path('configs')/(stem+'.json'),'exact tested configuration')
        pair=runs/f'original_rhs_n{n}_cpu_pair_v1.json'
        if not json.loads(pair.read_text())['passed']:raise ValueError('Required CPU pair failed')
        for extension in ('.json','.log'):
            save(pair.with_suffix(extension),Path('reports')/pair.with_suffix(extension).name,'complete CPU field and mass comparison')
    tree(runs/'original_rhs_stopping_evidence_v1','stopping_evidence','immutable completed-line prefix only')
    for name in ('original_rhs_work_comparison_v1.json','aphros64_volume_steady_step1_diagnostic_v1.json',
                 'aphros64_volume_steady_step1_diagnostic_v1.log','aphros64_volume_steady_step1_snapshot_v1.log'):
        save(runs/name,Path('reports')/name,'actual work counts or failed reference mass diagnostic; inspect scope')
    snapshot=base/'navier_stokes_proj_n64_steady_volume_pressure_step1_checkpoint_v1'
    tree(snapshot,'reference64_step1','verified prefix; not a completed reference run',large_csv=True)
    old=base/'navier_stokes_proj_n64_steady_volume_pressure_v1'
    for name in ('run_interruption.json','run_completion.json','run_manifest.json','a.conf','case_manifest.json'):
        save(old/name,Path('interrupted_reference64')/name,'interrupted original N1 sequence; not a successful steady run')
    for name in ('ours_proj64_steady_original_rhs_v1',):
        for field in ('case.json','run_manifest.json'):
            p=runs/name/field
            if p.exists():save(p,Path('pending_native')/name/field,'immutable input only; no completion claim')
    save(runs/'config_ours_proj64_steady_original_rhs_v1.json','configs/config_ours_proj64_steady_original_rhs_v1.json','new steady run input')
    reference=base/'navier_stokes_proj_n64_steady_volume_pressure_diff8_v1'
    for name in ('a.conf','case_manifest.json','run_manifest.json'):
        save(reference/name,Path('pending_reference64')/name,'original N8 complete steady trajectory remains pending')
    for name in ('build_original_rhs_v1.py','run_original_rhs_audits_v1.py','record_original_rhs_evidence_v1.py','original_rhs_work_v1.py'):
        save(runs/name,Path('reproduction')/name,'actual build, audit and work-count scripts')
    for name in ('check_twisted_time_pair.py','check_twisted_mass.py','check_twisted_projection_flux.py','compare_twisted.py',
                 'check_twisted_checkpoint.py','run_twisted_solver.py','run_twisted_baseline.ps1','snapshot_twisted_reference.py',
                 'check_aphros_pressure_snapshot.py','record_twisted_original_rhs_checkpoint.py'):
        save(repo/'scripts'/name,Path('checkers')/name,'reproduction and integrity source')
    for name in ('LICENSE.aphros','LICENSE.amgcl'):save(repo/'validation/aphros'/name,Path('licenses')/name,'upstream license')
    receipt={'scope':__doc__,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'goal_complete':False,
             'remaining':'Independent adaptive steady alignment and physical grid/time convergence; original64/128 mass issues remain open.',
             'files':records,'retained_runtime_files':retained}
    (target/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'files':len(records),'bytes':sum(r['size'] for r in records),'retained_runtime_files':len(retained)}),flush=True)


if __name__=='__main__':main()
