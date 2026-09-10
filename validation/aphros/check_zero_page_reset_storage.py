"""Compile and run storage reset semantics checks with demand-zero optimization off and on."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_extended_reference import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();tree=args.sources.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    compiler=Path('C:/ProgramData/mingw64/mingw64/bin/g++.exe')
    source=Path(__file__).with_name('zero_page_reset_storage_test.cpp')
    exe=out/'storage.exe'
    shutil.copyfile(source,out/'source.cpp')
    shutil.copyfile(__file__,out/'producer.py.txt')
    command=[str(compiler),'-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fopenmp','-static',
             '-I'+str(tree),'-I'+str(tree/'src'),str(source),'-lpsapi','-o',str(exe)]
    inputs={str(p.resolve()):sha(p) for p in (compiler,source,Path(__file__),
            tree/'src/geom/field.h',tree/'twisted_zero_pages_reset.h',tree/'src/geom/vect.h')}
    with (out/'compile.log').open('w') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    rows=[]
    if code==0:
        for enabled in (False,True):
            env=os.environ.copy();env.pop('APHROS_TWISTED_ZERO_PAGES',None)
            if enabled:env['APHROS_TWISTED_ZERO_PAGES']='1'
            run=subprocess.run([str(exe)],env=env,capture_output=True,text=True)
            (out/('enabled.log' if enabled else 'disabled.log')).write_text(run.stdout+run.stderr)
            rows.append({'enabled':enabled,'exit_code':run.returncode,
                         'result':json.loads(run.stdout) if run.returncode==0 else None})
    unchanged=all(sha(Path(p))==h for p,h in inputs.items())
    result={'passed':code==0 and unchanged and all(r['exit_code']==0 and r['result']['passed'] for r in rows),
            'scope':__doc__+' Working-set counters describe the storage probe, not a completed CFD run.',
            'command':command,'compile_exit_code':code,'rows':rows,'inputs_unchanged':unchanged,
            'executable_sha256':sha(exe) if exe.exists() else None,'source_sha256':inputs}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','command')}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
