"""Build and preserve fresh native flow/audit executables with source provenance."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-inventory',type=Path,required=True,help='Prior native build manifest defining source paths')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    inventory=json.loads(args.source_inventory.read_text())
    before={name:sha(repo/name) for name in inventory['source_sha256']}
    for name in before:
        snapshot=out/'sources'/(name+'.txt');snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,snapshot)
    shutil.copyfile(__file__,out/'build_script.py.txt')
    def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
    targets=('simple_channel','native_compact_gpu_audit')
    report={'source_sha256':before,'build_started_utc':now(),
            'commands':[['xmake','build','-j','1',target] for target in targets]}
    manifest=out/'build_manifest.json';manifest.write_text(json.dumps(report,indent=2)+'\n')
    code=0
    with (out/'build.log').open('w') as log:
        for command in report['commands']:
            code=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:break
    report.update(exit_code=code,source_unchanged=all(sha(repo/name)==value for name,value in before.items()),build_completed_utc=now())
    if not code and report['source_unchanged']:
        report['executable_sha256']={}
        for target in targets:
            exe=out/(target+'.exe')
            shutil.copyfile(repo/'output/build_compact_rows/windows/x64/release'/(target+'.exe'),exe)
            report['executable_sha256'][exe.name]=sha(exe)
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)
    if code or not report['source_unchanged']:raise SystemExit(code or 1)


if __name__=='__main__':main()
