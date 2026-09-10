"""Verify that allowing equivalent seed paths cannot admit changed inputs or loader bytes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','accepted-pair','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep test fixtures on D')
    pair=json.loads(a.accepted_pair.read_text())
    if not pair['passed'] or not pair['equivalent_seed_paths_checked']:
        raise ValueError('Require the actual completed positive comparison first')
    out.mkdir(parents=True,exist_ok=False)
    original=json.loads((a.candidate/'case_manifest.json').read_text())
    script=Path(__file__).with_name('check_aphros_precision_pair.py')
    copied=('run_manifest.json','run_completion.json','a.conf','case_manifest.json','run.log',
            'tube_b0_geometry_cells.csv','tube_b0_geometry_faces.csv','tube_b0_geometry_walls.csv',
            'tube_b0_time.csv','initial_velocity_echo.bin')
    cases=[]
    for name,expected in (
        ('paths_without_explicit_flag','Require identical input configurations'),
        ('changed_loader_echo','Actual initial loader differs'),
        ('different_verified_payload','Equivalent seed paths contain different initial velocities'),
        ('changed_viscosity','Require identical input configurations'),
        ('missing_seed_metadata','Equivalent seed paths require two actually seeded runs')):
        candidate=a.candidate;flags=[] if name=='paths_without_explicit_flag' else ['--equivalent-seed-paths']
        if name!='paths_without_explicit_flag':
            candidate=out/name;candidate.mkdir()
            for filename in copied:shutil.copyfile(a.candidate/filename,candidate/filename)
            config=copy.deepcopy(original)
            if name=='changed_loader_echo':
                path=candidate/'initial_velocity_echo.bin';data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            elif name=='different_verified_payload':
                path=candidate/'different_velocity.bin'
                data=bytearray(Path(config['initial_velocity']['file']).read_bytes());data[-1]^=1;path.write_bytes(data)
                shutil.copyfile(path,candidate/'initial_velocity_echo.bin')
                metadata=json.loads(Path(config['initial_velocity']['manifest']).read_text())
                metadata['initial_velocity_sha256']=sha(path)
                meta=candidate/'different_seed_manifest.json';meta.write_text(json.dumps(metadata,indent=2)+'\n')
                config['initial_velocity'].update(file=str(path),sha256=sha(path),manifest=str(meta),manifest_sha256=sha(meta))
            elif name=='changed_viscosity':config['spec']['nu']*=2
            elif name=='missing_seed_metadata':del config['initial_velocity']
            (candidate/'case_manifest.json').write_text(json.dumps(config,indent=2)+'\n')
        output=out/(name+'.json')
        command=[sys.executable,str(script),'--reference',str(a.reference),'--candidate',str(candidate),
                 '--output',str(output),*flags]
        result=subprocess.run(command,capture_output=True,text=True)
        log=out/(name+'.log');log.write_text(result.stdout+result.stderr)
        if result.returncode==0 or expected not in result.stderr or output.exists():
            raise ValueError('Comparison scope regression: '+name+' '+result.stderr)
        cases.append({'name':name,'passed':True,'exit_code':result.returncode,'expected_rejection':expected,
                      'command':command,'log_sha256':sha(log)})
    inputs={str(path.resolve()):sha(path) for path in (Path(__file__),script,a.accepted_pair)}
    for root in (a.reference,a.candidate):
        for name in copied:inputs[str((root/name).resolve())]=sha(root/name)
    report={'passed':True,'scope':__doc__,'cases':cases,'source_sha256':inputs,'new_flow_run_claimed':False}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'rejections_checked':len(cases),'output':str(out)}))


if __name__=='__main__':main()
