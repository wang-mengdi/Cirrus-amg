"""Preserve isolated extended-reference builds, unchanged geometry, and actual candidate recovery."""
import argparse
import csv
import datetime
import gzip
import json
from pathlib import Path
import shutil
import zipfile
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1];runs=repo/'output/twisted'
    base=Path('D:/Dropbox/Agent-simulation/twisted-baseline');out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False);files=[];retained={}
    def keep(path,scope):
        path=path.resolve();retained[str(path)]={'source':str(path),'source_sha256':sha(path),'size':path.stat().st_size,'scope':scope}
    def save(path,dest,scope,compress=False):
        path=path.resolve();value=sha(path);target=out/dest;target.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with path.open('rb') as a,gzip.open(target,'wb',compresslevel=6) as b:shutil.copyfileobj(a,b)
        else:shutil.copyfile(path,target)
        if sha(path)!=value:raise ValueError('Archive input changed')
        files.append({'path':target.relative_to(out).as_posix(),'source':str(path),'source_sha256':value,
                      'sha256':sha(target),'size':target.stat().st_size,'gzip':compress,'scope':scope})
    save(runs/'anderson_failure_build_v1/build_manifest.json',Path('build/build_manifest.json'),'unchanged native build used in the actual 128 recovery')
    for version in range(1,5):
        build=base/f'extended_projection_build_v{version}';manifest=json.loads((build/'build_manifest.json').read_text())
        if not manifest['source_unchanged']:raise ValueError('Extended build sources changed during compilation')
        if version>=3 and not manifest['passed']:raise ValueError('Linked extended build missing')
        for name,value in manifest['source_sha256'].items():
            if sha(Path(name))!=value:raise ValueError('Prepared build input changed: '+name)
        for path in build.iterdir():
            if path.suffix in ('.o','.exe'):keep(path,'actual completed object or executable from this isolated attempt')
            elif path.is_file():save(path,Path(f'builds/v{version}')/path.name,'complete build attempt, including failures')
        src=base/f'extended_projection_sources_v{version}'
        package=runs/f'extended_reference_sources_v{version}.zip'
        if package.exists():raise ValueError('Preserve prior source package')
        with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in manifest['source_sha256']:z.write(name,Path(name).relative_to(src).as_posix())
        # Verify every packed byte against the actual compilation inputs.
        import hashlib
        with zipfile.ZipFile(package) as z:
            for name,value in manifest['source_sha256'].items():
                if hashlib.sha256(z.read(Path(name).relative_to(src).as_posix())).hexdigest()!=value:
                    raise ValueError('Source package differs from actual compilation inputs')
        save(package,Path(f'builds/v{version}/sources.zip'),'verified complete source package used by this actual build')
    double_build=base/'geometry_state_double_build_v1'
    for path in double_build.iterdir():
        if path.suffix in ('.obj','.exe'):keep(path,'double snapshot driver linked against the unchanged original static library')
        elif path.is_file():save(path,Path('double_geometry_build')/path.name,'actual original-library driver build')
    names=['geometry_state_double_n16_checks_v2.json','geometry_state_double_n64_checks_v1.json',
           'geometry_state_extended_n16_checks_v2.json','geometry_state_extended_n64_checks_v1.json']
    for name in names:
        path=runs/name;report=json.loads(path.read_text())
        if not report['passed']:raise ValueError('Geometry check failed: '+name)
        save(path,Path('checks')/name,'completed scoped geometry check')
        for source in report['source_sha256']:keep(Path(source),'verified original/captured/reloaded geometry payload')
    save(runs/'geometry_state_extended_n16_checks_v1.json',Path('checks/failed_strict_polygon_comparison.json'),'FAILED strict equality of recomputed wall display vertices; retained explicitly')
    save(runs/'check_aphros_geometry_state_strict_v1.py.txt',Path('checkers/strict_geometry_checker_v1.py.txt'),'actual checker used for the strict comparison')
    guard=base/'geometry_state_wrong_mesh_guard_v1'
    completion=json.loads((guard/'run_completion.json').read_text())
    if not completion['exit_code'] or 'Geometry snapshot mesh differs' not in (guard/'run.log').read_text():
        raise ValueError('Wrong-mesh geometry guard did not reject the load')
    for path in guard.iterdir():
        if path.is_file():save(path,Path('wrong_mesh_guard')/path.name,'expected rejected geometry load; not a flow result')
    native=runs/'ours_proj128_anderson_failure_2steps_v1';failures=[]
    for i in range(1,5):
        folder=native/f'linear_failure_{i}';data=json.loads((folder/'failure.json').read_text())
        if data['physical_step']!=2 or data['iteration']!=2 or data['phase']!='anderson_fixed_point' or data['anderson_attempt']!=i:
            raise ValueError('Unexpected actual candidate failure phase')
        if data['failed_solution_accepted'] or not data['true_relative_residual']>data['tolerance']==1e-13:
            raise ValueError('Candidate failure tolerance changed or failed solution accepted')
        failures.append(data)
        save(folder/'failure.json',Path(f'native128/linear_failure_{i}/failure.json'),'actual rejected optional candidate')
        save(folder/'rhs.bin',Path(f'native128/linear_failure_{i}/rhs.bin.gz'),'actual failed linear RHS, native cell ID order',True)
    lines=(native/'step_0002/acceleration.csv').read_text().splitlines(keepends=True)
    prefix=''.join(lines[:4]);rows=list(csv.DictReader(prefix.splitlines()))
    if len(rows)!=3 or rows[1]['after_iteration']!='2' or rows[1]['accepted']!='0' or rows[2]['after_iteration']!='3':
        raise ValueError('Missing recovery after all four rejected candidates')
    snapshot=runs/'native128_candidate_recovery_prefix_v1.csv'
    with snapshot.open('x') as f:f.write(prefix)
    save(snapshot,Path('native128/recovery_prefix.csv'),'first three acceleration rows only; not a completed physical step')
    for root,label in ((native,'native128'),(base/'navier_stokes_proj_n16_extended_cg_v3','extended_cg16'),
                       (base/'navier_stokes_proj_n16_extended_amg_v4','extended_amg16')):
        for name in ('run_manifest.json','case.json','case_manifest.json','a.conf'):
            if (root/name).exists():save(root/name,Path('pending_runs')/label/name,'launch provenance only; no completion claim')
    for name in ('run_twisted_baseline.ps1','run_aphros_geometry_state.py','check_aphros_geometry_state.py',
                 'record_twisted_extended_checkpoint.py','check_twisted_checkpoint.py'):
        save(repo/'scripts'/name,Path('checkers')/name,'reproduction source')
    for name in ('build_extended_reference.py','prepare_extended_reference.py','build_twisted_driver.ps1','twisted_geometry_state.h'):
        save(repo/'validation/aphros'/name,Path('checkers')/name,'reproduction source')
    save(repo/'validation/twisted/EXTENDED_REFERENCE.md',Path('scope.md'),'explicit scope and remaining work')
    check={'passed':True,'scope':__doc__,'native128_recovered_past_failed_candidate':True,
           'native128_completed_two_steps':False,'extended_flow_alignment_proven':False,'goal_complete':False,
           'source_packages_verified':4,'actual_rejected_candidate_residuals':[r['true_relative_residual'] for r in failures]}
    check_path=runs/'extended_reference_checkpoint_checks_v1.json'
    with check_path.open('x') as f:json.dump(check,f,indent=2)
    save(check_path,Path('checks.json'),'scoped checkpoint verification only')
    receipt={'scope':__doc__,'goal_complete':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'files':files,'retained_runtime_files':list(retained.values())}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'archive_files':len(files),'retained_files':len(retained),'goal_complete':False}))


if __name__=='__main__':main()
