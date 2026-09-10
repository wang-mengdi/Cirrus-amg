"""Preserve exact reference-flow checks and bounded cache evidence without a steady claim."""
import argparse,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    repo=Path(__file__).resolve().parents[1];runs=repo/'output/twisted';base=Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);files=[];retained={}
    def keep(path,scope):
        path=Path(path).resolve();retained[str(path)]={'source':str(path),'source_sha256':sha(path),'size':path.stat().st_size,'scope':scope}
    def save(path,dest,scope,compress=False):
        path=Path(path).resolve();target=out/dest;target.parent.mkdir(parents=True,exist_ok=True);value=sha(path)
        if compress:
            with path.open('rb') as a,gzip.open(target,'wb',compresslevel=6) as b:shutil.copyfileobj(a,b)
        else:shutil.copyfile(path,target)
        if sha(path)!=value:raise ValueError('Archive source changed')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(path),'source_sha256':value,
            'sha256':sha(target),'size':target.stat().st_size,'gzip':compress,'scope':scope})
    save(runs/'anderson_failure_build_v1/build_manifest.json','build/build_manifest.json','unchanged native build in the field comparisons')
    for version in (5,6,7):
        build=base/f'extended_projection_build_v{version}';m=json.loads((build/'build_manifest.json').read_text())
        if not m['passed'] or not m['source_unchanged']:raise ValueError('Incomplete build')
        for path in build.iterdir():
            if path.suffix in ('.o','.exe'):keep(path,'actual compiled object or executable')
            elif path.is_file():save(path,f'builds/v{version}/{path.name}','actual build log or manifest')
        if version==6:continue
        source=base/f'extended_projection_sources_v{version}';package=runs/f'exact_reference_sources_v{version}.zip'
        with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,value in m['source_sha256'].items():
                if sha(Path(name))!=value:raise ValueError('Actual compilation source changed')
                z.write(name,Path(name).relative_to(source).as_posix())
        with zipfile.ZipFile(package) as z:
            for name,value in m['source_sha256'].items():
                if hashlib.sha256(z.read(Path(name).relative_to(source).as_posix())).hexdigest()!=value:raise ValueError('Source package differs')
        save(package,f'builds/v{version}/sources.zip','verified actual complete source package')
    for name in ('navier_stokes_proj_n16_extended_exact_v5','navier_stokes_proj_n16_extended_exact_o3_v6',
                 'navier_stokes_proj_n16_extended_amg_v4'):
        root=base/name;c=json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))
        if c['exit_code']:raise ValueError('Completed flow failed')
        for path in root.iterdir():
            if path.is_file():save(path,f'flows/{name}/{path.name}'+('.gz' if path.suffix=='.csv' else ''),'completed independent reference flow',path.suffix=='.csv')
    reports=('aphros_n16_extended_exact_native_pair_v1.json','aphros_n16_extended_exact_mass_v1.json',
        'aphros_n16_extended_steps8_native_pair_v1.json','aphros_extended_o1_o3_pair_v1.json',
        'extended_reference_checker_original16_regression_v1.json','exact_mass_controls_checks_v3.json',
        'extended_cache_trace_checks_v1.json')
    for name in reports:
        path=runs/name;r=json.loads(path.read_text())
        if not r['passed']:raise ValueError('Expected scoped check failed: '+name)
        save(path,'checks/'+name,'completed scoped check; no global goal claim')
        maps=[r.get('source_sha256',{})]
        for row in r.get('checks',[]) if isinstance(r.get('checks'),list) else []:
            maps.extend([row['exact']['source_sha256'],row['ordinary']['source_sha256']])
        for mapping in maps:
            for source,value in mapping.items():
                if sha(Path(source))!=value:raise ValueError('Checked evidence changed: '+source)
                keep(source,'verified source or field used by this check')
    for version in (6,7):
        for path in (base/f'navier_stokes_proj_n16_extended_cache_trace_v{version}').iterdir():
            if path.is_file():save(path,f'bounded_cache_traces/v{version}/{path.name}','deliberately capped three-iteration diagnostic; NOT completed flow')
    for name in ('audit_exact_mass_controls_v2.py','recheck_exact_mass_controls_v3.py','check_extended_cache_trace_v1.py',
                 'check_aphros_exact_mass_initial_v1.py.txt','exact_mass_controls_v1.log'):
        save(runs/name,'control_sources/'+name,'control producer or explicitly preserved initial failure')
    for name in ('compare_twisted.py','check_aphros_exact_mass.py','validate_aphros_extended_reference.py',
                 'check_aphros_precision_pair.py','record_twisted_exact_flow_checkpoint.py'):
        save(repo/'scripts'/name,'sources/scripts/'+name,'current reproduction or verification source')
    for name in ('twisted_exact_flow.h','prepare_extended_reference.py','build_extended_reference.py'):
        save(repo/'validation/aphros'/name,'sources/aphros/'+name,'current reproduction source')
    save(repo/'validation/twisted/EXACT_REFERENCE_FLOW.md','scope.md','completed scope and remaining requirements')
    pending=base/'navier_stokes_proj_n16_extended_cache_v7'
    for name in ('a.conf','case_manifest.json','run_manifest.json'):
        save(pending/name,'pending_cache_flow/'+name,'launch provenance only; no final-field claim')
    receipt={'scope':__doc__,'goal_complete':False,'files':files,'retained_runtime_files':list(retained.values())}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'archive_files':len(files),'retained_files':len(retained),'goal_complete':False}))


if __name__=='__main__':main()
