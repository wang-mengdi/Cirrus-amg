"""Preserve verified trial-failure infrastructure and coupled pressure precision evidence."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1];runs=repo/'output/twisted'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[];retained={}
    def keep(path,scope):
        path=path.resolve();retained[str(path)]={'source':str(path),'source_sha256':sha(path),'size':path.stat().st_size,'scope':scope}
    def save(path,dest,scope):
        path=path.resolve();value=sha(path);target=out/dest;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        if sha(path)!=value or sha(target)!=value:raise ValueError('Archive copy changed')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(path),'source_sha256':value,
                      'sha256':value,'size':target.stat().st_size,'gzip':False,'scope':scope})
    build=runs/'anderson_failure_build_v1'
    for source in build.rglob('*'):
        if source.is_file():
            if source.suffix=='.exe':keep(source,'immutable executable from this native build')
            else:save(source,Path('build')/source.relative_to(build),'completed build and actual source snapshots')
    for n in (16,64):
        audit=runs/f'anderson_failure_audit{n}_v1'
        data=json.loads((audit/'audit.json').read_text())
        if not data['passed'] or not all(t['forced_failure_and_reuse']['reuse_passed'] for t in data['linear_tests']):
            raise ValueError('Failed-solve reuse audit failed')
        for path in audit.iterdir():
            if path.suffix=='.json':save(path,Path(f'audits/n{n}')/path.name,'completed full operator and solver-reuse audit')
            else:keep(path,'actual operator and manufactured-solution samples')
        root=runs/f'ours_proj{n}_anderson_failure_v1'
        completion=json.loads((root/'run_completion.json').read_text())
        if completion['exit_code']:raise ValueError('Regression trajectory incomplete')
        for name in ('case.json','run_manifest.json','run_completion.json','transient_summary.json','run.log','gpu_linear.csv','time_history.csv','projection_method.json'):
            save(root/name,Path(f'native{n}')/name,'completed normal-path regression')
        for step in sorted(root.glob('step_*')):
            for name in ('case.json','metrics.json','history.csv','acceleration.csv'):
                save(step/name,Path(f'native{n}')/step.name/name,'all completed physical-step diagnostics')
    reports=['anderson_failure_n16_cpu_pair_v1.json','anderson_failure_n64_cpu_pair_v1.json',
             'anderson_failure_n16_aphros_pair_v1.json','aphros_coupled_precision64_decimal_checks_v1.json']
    for name in reports:
        path=runs/name;data=json.loads(path.read_text())
        if not data['passed']:raise ValueError('Required validation failed: '+name)
        save(path,Path('checks')/name,'completed scoped verification; does not establish the full goal')
        for source in data.get('source_sha256',{}):
            # These checkers record absolute paths, unlike the original Aphros face checker.
            path=Path(source)
            if path.is_absolute():keep(path,'verified input retained for reproducing this scoped check')
    path=runs/'anderson_failure_n16_regression_v1.json'
    if json.loads(path.read_text())['passed']:raise ValueError('Expected recorded non-bitwise regression outcome')
    save(path,Path('checks')/path.name,'FAILED bitwise check, preserved explicitly; numerical/physical checks are separate')
    probe=runs/'aphros_coupled_precision64_v1'
    for path in probe.iterdir():
        if path.suffix in ('.bin','.exe') or path.name=='coupled_pressure.csv':keep(path,'coupled pressure replay payload retained unchanged')
        else:save(path,Path('coupled_pressure_probe')/path.name,'completed offline coupled pressure replay')
    for name in ('config_ours_proj16_anderson_failure_v1.json','config_ours_proj64_anderson_failure_v1.json',
                 'config_ours_proj128_anderson_failure_2steps_v1.json','config_anderson_failure_audit16_v1.json','config_anderson_failure_audit64_v1.json'):
        save(runs/name,Path('configs')/name,'actual executed configuration')
    pending=runs/'ours_proj128_anderson_failure_2steps_v1'
    for name in ('case.json','run_manifest.json'):
        save(pending/name,Path('pending_native128')/name,'launch provenance only; not a completion claim')
    for name in ('build_twisted_native.py','check_twisted_native_regression.py','check_twisted_time_pair.py',
                 'run_aphros_coupled_precision_probe.py','check_aphros_coupled_precision_probe.py',
                 'record_twisted_anderson_checkpoint.py','check_twisted_checkpoint.py'):
        save(repo/'scripts'/name,Path('checkers')/name,'reproduction source')
    save(repo/'validation/twisted/ANDERSON_FAILURE.md',Path('scope.md'),'explicit validation scope and remaining requirements')
    receipt={'scope':__doc__,'goal_complete':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'files':files,'retained_runtime_files':list(retained.values())}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'archive_files':len(files),'retained_files':len(retained),'goal_complete':False}))


if __name__=='__main__':main()
