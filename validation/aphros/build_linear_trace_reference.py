"""Relink a verified extended Aphros build with only its driver logging wired up.

All other translation units and headers must remain byte-identical. Reuse their
verified object files without rebuilding or modifying a live reference build.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess

from build_extended_reference import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.reference_build.resolve();out=args.output.resolve()
    manifest=json.loads((base/'build_manifest.json').read_text())
    if not manifest['passed'] or not manifest['source_unchanged']:
        raise ValueError('Require a completed unchanged reference build')
    for path,value in manifest['source_sha256'].items():
        if sha(Path(path))!=value:raise ValueError('Reference build source changed: '+path)
    if sha(base/'twisted_extended.exe')!=manifest['executable_sha256']:
        raise ValueError('Reference executable changed')
    drivers=[row for row in manifest['results'] if Path(row['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Require one driver translation unit')
    driver=drivers[0];old=Path(driver['source']).parent
    out.mkdir(parents=True,exist_ok=False);src=out/'sources';shutil.copytree(old,src)
    body=(src/'driver.cpp').read_bytes()
    anchor=b'  if(sem("solver")) {'
    if body.count(anchor)!=1:raise ValueError('Unknown driver solver initialization')
    ending=b'\r\n' if b'\r\n' in body else b'\n'
    inserted=anchor+ending+b'    m.flags.linreport=var.Int["linreport"]!=0;'
    (src/'driver.cpp').write_bytes(body.replace(anchor,inserted))
    changes=[]
    for path in old.rglob('*'):
        if path.is_file() and sha(path)!=sha(src/path.relative_to(old)):
            changes.append(path.relative_to(old).as_posix())
    if changes!=['driver.cpp']:raise ValueError('Unexpected dependency change')
    prepared=json.loads((src/'prepare_manifest.json').read_text())
    prepared['source_sha256'][str(Path(__file__).resolve())]=sha(Path(__file__))
    prepared['prepared_source_sha256']['driver.cpp']=sha(src/'driver.cpp')
    prepared['linear_trace_scope']='Only wire Vars linreport into mesh.flags.linreport; original equations and linear backend remain unchanged'
    (src/'prepare_manifest.json').write_text(json.dumps(prepared,indent=2)+'\n')
    before={str(p):sha(p) for p in src.rglob('*') if p.is_file()}
    shutil.copyfile(__file__,out/'build_source.py.txt')
    rows=[]
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    for original in manifest['results']:
        obj=Path(original['object'])
        if original['exit_code'] or sha(obj)!=original['object_sha256']:
            raise ValueError('Reference object changed')
        target=out/obj.name
        row=dict(original)
        if original is driver:
            command=[arg.replace(str(old),str(src)) for arg in original['command']]
            output_index=command.index('-o')+1;command[output_index]=str(target)
            start=now()
            with (out/'driver.log').open('w') as log:
                code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
            row.update(source=str(src/'driver.cpp'),command=command,started_utc=start,completed_utc=now(),
                       exit_code=code,object=str(target),object_sha256=sha(target) if target.exists() else None,
                       log_sha256=sha(out/'driver.log'))
            if code:raise ValueError('Diagnostic driver compilation failed; retain output')
        else:
            shutil.copyfile(obj,target)
            row.update(object=str(target),reused_from=str(obj),object_sha256=sha(target))
        rows.append(row)
    compiler=Path(driver['command'][0]);exe=out/'twisted_extended.exe'
    if sha(compiler)!=manifest['compiler_sha256']:raise ValueError('Reference compiler changed')
    command=[str(compiler),'-fopenmp','-static','-Wl,--gc-sections',*[r['object'] for r in rows],'-lpsapi','-o',str(exe)]
    with (out/'link.log').open('w') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    unchanged=all(sha(Path(path))==value for path,value in before.items())
    original_unchanged=all(sha(Path(path))==value for path,value in manifest['source_sha256'].items())
    result={'scope':__doc__,'passed':code==0 and unchanged and original_unchanged,
            'source_unchanged':unchanged,'original_source_unchanged':original_unchanged,
            'source_sha256':before,'compiler_sha256':sha(compiler),'results':rows,'link_exit_code':code,
            'executable_sha256':sha(exe) if exe.exists() else None,'completed_utc':now(),
            'reference_build':str(base),'reference_build_manifest_sha256':sha(base/'build_manifest.json'),
            'recompiled_translation_units':['driver.cpp'],'reused_objects':len(rows)-1}
    (out/'build_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','results')}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
