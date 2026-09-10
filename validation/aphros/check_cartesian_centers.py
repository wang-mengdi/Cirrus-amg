"""Compare actual stored/lazy Cartesian getters against a separately linked old build."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from build_extended_reference import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-build',type=Path,required=True);p.add_argument('--candidate-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    test=Path(__file__).with_name('cartesian_centers_test.cpp').resolve();shutil.copyfile(test,out/test.name)
    builds=[];inputs={str(test):sha(test),str(Path(__file__).resolve()):sha(Path(__file__))}
    for label,base in [('reference',a.reference_build.resolve()),('candidate',a.candidate_build.resolve())]:
        manifest=base/'build_manifest.json';b=json.loads(manifest.read_text());inputs[str(manifest)]=sha(manifest)
        if not b['passed'] or not b['source_unchanged']:raise ValueError('Require complete verified builds')
        for path,digest in b['source_sha256'].items():
            if sha(Path(path))!=digest:raise ValueError('Compiled source changed: '+path)
        drivers=[r for r in b['results'] if Path(r['source']).name=='driver.cpp']
        if len(drivers)!=1:raise ValueError('Require one actual driver object')
        driver=drivers[0];obj=out/(label+'.o');exe=out/(label+'.exe')
        command=[str(test) if arg==driver['source'] else str(obj) if arg==driver['object'] else arg for arg in driver['command']]
        assert str(test) in command and str(obj) in command
        with (out/(label+'_compile.log')).open('w') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('Coordinate test compilation failed; retain log')
        objects=[]
        for row in b['results']:
            if row['exit_code'] or sha(Path(row['object']))!=row['object_sha256']:raise ValueError('Original object changed')
            if row is not driver:objects.append(row['object'])
        link=[command[0],'-fopenmp','-static','-Wl,--gc-sections',str(obj),*objects,'-lpsapi','-o',str(exe)]
        with (out/(label+'_link.log')).open('w') as log:
            code=subprocess.call(link,stdout=log,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('Coordinate test link failed; retain log')
        builds.append({'label':label,'reference_build':str(base),'command':command,'link_command':link,
            'executable':str(exe),'executable_sha256':sha(exe),'object_sha256':sha(obj)})
    runs=[]
    for label,index,lazy in [('reference',0,False),('candidate_stored',1,False),('candidate_lazy',1,True)]:
        env=os.environ.copy();env.pop('APHROS_TWISTED_LAZY_CENTERS',None);env['OMP_NUM_THREADS']='2';env['OMP_WAIT_POLICY']='PASSIVE'
        if lazy:env['APHROS_TWISTED_LAZY_CENTERS']='1'
        output=out/(label+'.csv');command=[builds[index]['executable'],str(output)]
        run=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True)
        (out/(label+'_run.log')).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError('Coordinate test failed; retain output')
        counts=json.loads(run.stdout)
        if counts['cases']!=9 or counts['scalar_bytes']!=16 or counts['scalar_digits']!=64:raise ValueError('Unexpected coordinate test coverage or scalar precision')
        runs.append({'label':label,'lazy_centers':lazy,'command':command,'counts':counts,'output':str(output),'sha256':sha(output)})
    same=len({r['sha256'] for r in runs})==1
    unchanged=all(sha(Path(path))==h for path,h in inputs.items())
    result={'passed':same and unchanged,'scope':__doc__+' Nine domains/halo widths, including shifted indices and nonbinary spacing. Coordinates are emitted as lossless hexadecimal long doubles.',
        'all_coordinate_bytes_equal':same,'inputs_unchanged':unchanged,'builds':builds,'runs':runs,'source_sha256':inputs,'goal_complete':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('builds','source_sha256')}))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
