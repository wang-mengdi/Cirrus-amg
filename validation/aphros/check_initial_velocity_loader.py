"""Compile and verify velocity input, periodic halos, byte echo, and malformed-input rejection."""
import argparse
import json
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
    source=Path(__file__).with_name('initial_velocity_loader_test.cpp')
    helper=Path(__file__).with_name('twisted_initial_velocity.h')
    if sha(helper)!=sha(tree/helper.name):raise ValueError('Test header differs from compiled driver header')
    exe=out/'test.exe'
    inputs={str(p.resolve()):sha(p) for p in (compiler,source,helper,Path(__file__),
        tree/'src/geom/field.h',tree/'src/geom/vect.h',tree/'twisted_zero_pages_reset.h')}
    command=[str(compiler),'-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-static',
             '-I'+str(source.parent),'-I'+str(tree),'-I'+str(tree/'src'),str(source),'-o',str(exe)]
    shutil.copyfile(source,out/'source.cpp');shutil.copyfile(__file__,out/'producer.py.txt')
    with (out/'compile.log').open('w') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    run=None;result=None
    if code==0:
        run=subprocess.run([str(exe)],cwd=out,capture_output=True,text=True)
        (out/'run.log').write_text(run.stdout+run.stderr)
        if run.returncode==0:result=json.loads(run.stdout)
    unchanged=all(sha(Path(p))==h for p,h in inputs.items())
    report={'passed':code==0 and run.returncode==0 and result['passed'] and result['malformed_inputs_rejected']==9 and unchanged,
        'scope':__doc__,'compile_exit_code':code,'run_exit_code':run.returncode if run else None,
        'result':result,'inputs_unchanged':unchanged,'command':command,'source_sha256':inputs,
        'executable_sha256':sha(exe) if exe.exists() else None}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','command')}))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
