"""Preserve completed native128 recovery and full extended-reference cache equality."""
import argparse,gzip,json,shutil
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];runs=repo/'output/twisted'
    base=Path('D:/Dropbox/Agent-simulation/twisted-baseline');out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=[];retained={}
    def keep(path,scope):
        path=Path(path).resolve();retained[str(path)]={'source':str(path),'source_sha256':sha(path),'size':path.stat().st_size,'scope':scope}
    def save(path,dest,scope,compress=False):
        path=Path(path).resolve();target=out/dest;target.parent.mkdir(parents=True,exist_ok=True);value=sha(path)
        if compress:
            with path.open('rb') as a,gzip.open(target,'wb',compresslevel=6) as b:shutil.copyfileobj(a,b)
        else:shutil.copyfile(path,target)
        if sha(path)!=value:raise ValueError('Input changed during archive')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(path),'source_sha256':value,
                      'sha256':sha(target),'size':target.stat().st_size,'gzip':compress,'scope':scope})
    save(runs/'anderson_failure_build_v1/build_manifest.json','build/build_manifest.json','native build used in completed two-step run')
    for name in ('native128_recovered_terminal_checks_v1.json','aphros_extended_cache_full_pair_v1.json'):
        path=runs/name;report=json.loads(path.read_text())
        if not report['passed']:raise ValueError('Completed scoped check did not pass')
        save(path,'checks/'+name,'completed scoped check, not global goal success')
        for field in ('source_sha256','executed_input_sha256'):
            for source,value in report.get(field,{}).items():
                if sha(Path(source))!=value:raise ValueError('Verified evidence changed')
                keep(source,'actual completed run field, input or build source used by this check')
    native=runs/'ours_proj128_anderson_failure_2steps_v1'
    for name in ('run_manifest.json','run_completion.json','case.json','run.log','time_history.csv','gpu_linear.csv','transient_summary.json'):
        save(native/name,'native128/'+name,'completed two-step native trajectory')
    for step in (1,2):save(native/f'step_{step:04d}/metrics.json',f'native128/step_{step:04d}/metrics.json','completed physical-step diagnostics')
    for path in (base/'navier_stokes_proj_n16_extended_cache_v7').iterdir():
        if path.is_file():save(path,'reference_cache16/'+path.name+('.gz' if path.suffix=='.csv' else ''),'completed full reference cache comparison',path.suffix=='.csv')
    for root,label in ((runs/'ours_proj128_steady_anderson_recovery_v1','native128_steady'),
                       (base/'navier_stokes_proj_n64_extended_cache_v7','reference64_extended')):
        for name in ('run_manifest.json','case.json','case_manifest.json','a.conf'):
            if (root/name).exists():save(root/name,'new_launches/'+label+'/'+name,'launch provenance only; no final-field claim')
    for name in ('check_twisted_native_run.py','check_aphros_precision_pair.py','record_twisted_recovery_completion.py'):
        save(repo/'scripts'/name,'sources/'+name,'actual verification or preservation source')
    save(repo/'validation/twisted/RECOVERY_COMPLETION.md','scope.md','completion update superseding earlier pending status')
    (out/'receipt.json').write_text(json.dumps({'scope':__doc__,'goal_complete':False,'files':files,
        'retained_runtime_files':list(retained.values())},indent=2)+'\n')
    print(json.dumps({'files':len(files),'retained':len(retained),'goal_complete':False}))


if __name__=='__main__':main()
