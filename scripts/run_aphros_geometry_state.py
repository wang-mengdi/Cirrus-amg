"""Capture or round-trip complete Aphros geometry without advancing a flow state."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path)
    args=p.parse_args();source=args.case.resolve();exe=args.exe.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name in ('a.conf','case_manifest.json'):shutil.copyfile(source/name,out/name)
    env=dict(os.environ,OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE',APHROS_TWISTED_GEOMETRY='tube',
             APHROS_TWISTED_GEOMETRY_STATE_ONLY='1',APHROS_TWISTED_GEOMETRY_STATE_OUT=str(out/'geometry.bin'))
    env.pop('APHROS_TWISTED_GEOMETRY_STATE_IN',None)
    if args.input:env['APHROS_TWISTED_GEOMETRY_STATE_IN']=str(args.input.resolve(strict=True))
    inputs={str(source/'a.conf'):sha(source/'a.conf'),str(exe):sha(exe)}
    if args.input:inputs[str(args.input.resolve())]=sha(args.input)
    manifest={'scope':__doc__,'source_sha256':inputs,'executable':str(exe),'executable_sha256':sha(exe),
              'environment':{k:v for k,v in env.items() if k.startswith('APHROS_TWISTED_GEOMETRY') or k.startswith('OMP_')},
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (out/'run.log').open('w') as log:
        code=subprocess.run([str(exe),'a.conf'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    unchanged=all(sha(Path(name))==value for name,value in inputs.items())
    completion={'exit_code':code,'source_unchanged':unchanged,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'geometry_sha256':sha(out/'geometry.bin') if (out/'geometry.bin').exists() else None,
                'no_flow_trajectory':True}
    (out/'run_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion))
    if code or not unchanged:raise SystemExit(code or 1)


if __name__=='__main__':main()
