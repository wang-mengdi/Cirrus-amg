"""Run the optional pressure-AMG/direct-velocity validation backend on unchanged original Aphros equations, geometry and residual acceptance.

The original full-domain geometry, scalar precision, equations and run controls
are retained. A completed run still needs the independent field/mass comparison.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil
from run_twisted_solver import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release-cold-storage',action='store_true')
    p.add_argument('--factor-cache-entries',type=int,choices=range(1,9))
    p.add_argument('--lazy-centers',action='store_true')
    p.add_argument('--linear-backend',choices=('amg','direct','amg-pressure-direct-velocity'),required=True)
    a=p.parse_args();reference=a.reference.resolve();exe=a.executable.resolve();out=a.output.resolve()
    runtime=json.loads((reference/'run_manifest.json').read_text(encoding='utf-8-sig'))
    build_path=exe.parent/'build_manifest.json';build=json.loads(build_path.read_text())
    if not build['passed'] or not build['source_unchanged'] or sha(exe)!=build['executable_sha256']:
        raise ValueError('Require a completed verified build')
    for path,h in build['source_sha256'].items():
        if sha(Path(path))!=h:raise ValueError('Compiled source changed: '+path)
    drivers=[Path(row['source']) for row in build['results'] if Path(row['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Require one compiled driver')
    prepared=json.loads((drivers[0].parent/'prepare_manifest.json').read_text())
    if 'lazy_centers_scope' not in prepared or 'cold_storage_scope' not in prepared:raise ValueError('Build does not contain the guarded cold-storage release')
    settings=runtime['environment'].copy()
    required={'OMP_WAIT_POLICY':'PASSIVE','APHROS_TWISTED_AMG':'1',
        'APHROS_TWISTED_DIRECT':'1','APHROS_TWISTED_ZERO_PAGES':'1','APHROS_TWISTED_MEMORY_TRACE':'1'}
    if any(settings.get(k)!=v for k,v in required.items()):raise ValueError('Require the verified original zero-storage AMG controls')
    if settings.get('APHROS_TWISTED_INITIAL_VELOCITY') is not None:
        raise ValueError('Use an unchanged zero-start reference for this storage comparison')
    if 'set string vel_init zero' not in (reference/'a.conf').read_text():raise ValueError('Require zero initial velocity')
    geometry=Path(runtime['geometry_state']).resolve()
    if sha(geometry)!=runtime['geometry_state_sha256'] or sha(reference/'a.conf')!=runtime['config_sha256']:
        raise ValueError('Reference input changed')
    settings['APHROS_TWISTED_AMG']='1' if a.linear_backend!='direct' else None
    settings['APHROS_TWISTED_DIRECT_VELOCITY']='1' if a.linear_backend=='amg-pressure-direct-velocity' else None
    if a.linear_backend=='amg-pressure-direct-velocity' and 'velocity_direct_scope' not in prepared:
        raise ValueError('Build lacks the optional velocity backend selector')
    settings['APHROS_TWISTED_RELEASE_COLD_STORAGE']='1' if a.release_cold_storage else None
    settings['APHROS_TWISTED_LAZY_CENTERS']='1' if a.lazy_centers else None
    if a.factor_cache_entries is not None:
        settings['APHROS_TWISTED_FACTOR_CACHE']=str(a.factor_cache_entries)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('a.conf','case_manifest.json'):shutil.copyfile(reference/name,out/name)
    inputs={str(path):sha(path) for path in (Path(__file__).resolve(),build_path,exe,geometry,
        reference/'run_manifest.json',reference/'a.conf',reference/'case_manifest.json',out/'a.conf',out/'case_manifest.json')}
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('APHROS_'):env.pop(key)
    for key,value in settings.items():
        if value is None:env.pop(key,None)
        else:env[key]=value
    now=lambda:datetime.now(timezone.utc).isoformat()
    manifest={'executable':str(exe),'executable_sha256':sha(exe),'config_sha256':sha(out/'a.conf'),
        'geometry_state':str(geometry),'geometry_state_sha256':sha(geometry),
        'environment':settings,'started_utc':now()}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'cold_run_preparation.json').write_text(json.dumps({'scope':__doc__,
        'source_sha256':inputs,'reference':str(reference),'linear_backend':a.linear_backend,'release_cold_storage':a.release_cold_storage,'lazy_centers':a.lazy_centers,
        'factor_cache_entries':int(settings['APHROS_TWISTED_FACTOR_CACHE']),
        'reference_factor_cache_entries':int(runtime['environment']['APHROS_TWISTED_FACTOR_CACHE']),
        'minimum_available_memory_bytes':2*1024**3,'required_low_memory_seconds':10},indent=2)+'\n')
    start=time.perf_counter();samples=0;peak=private=0;minimum=None;low_since=None;stopped=False
    with (out/'run.log').open('w') as log,(out/'memory.csv').open('x',newline='') as memlog:
        writer=csv.writer(memlog);writer.writerow(['elapsed_seconds','pid','working_set_bytes','private_bytes','peak_working_set_bytes','available_system_bytes'])
        process=subprocess.Popen([str(exe),'a.conf'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        native=psutil.Process(process.pid)
        identity={'pid':native.pid,'creation_time':native.create_time(),'executable':native.exe(),
            'creation_utc':datetime.fromtimestamp(native.create_time(),timezone.utc).isoformat()}
        if Path(native.exe()).resolve()!=exe or native.cmdline()[-1]!='a.conf' or Path(native.cwd()).resolve()!=out:
            raise ValueError('Unexpected actual child identity')
        (out/'process_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        print(json.dumps({'native_process':identity,'output':str(out)}),flush=True)
        while process.poll() is None:
            try:
                current=psutil.Process(identity['pid'])
                if current.create_time()!=identity['creation_time'] or Path(current.exe()).resolve()!=exe:
                    raise ValueError('Native child identity changed')
                memory=current.memory_info();available=psutil.virtual_memory().available;elapsed=time.perf_counter()-start
                samples+=1;peak=max(peak,memory.peak_wset);private=max(private,memory.private)
                minimum=available if minimum is None else min(minimum,available)
                writer.writerow([elapsed,current.pid,memory.rss,memory.private,memory.peak_wset,available]);memlog.flush()
                low_since=(low_since if low_since is not None else elapsed) if available<2*1024**3 else None
                if not stopped and low_since is not None and elapsed-low_since>=10:
                    if current.cmdline()[-1]!='a.conf' or Path(current.cwd()).resolve()!=out:
                        raise ValueError('Child configuration changed before resource stop')
                    current.terminate();stopped=True
                    print(json.dumps({'stopped_only_new_cpu_reference_for_memory':identity,'available_bytes':available}),flush=True)
            except psutil.NoSuchProcess:pass
            time.sleep(.5)
        code=process.wait()
    completion={'exit_code':code,'completed_utc':now(),'geometry_state_unchanged':sha(geometry)==manifest['geometry_state_sha256'],
        'executable_unchanged':sha(exe)==manifest['executable_sha256'],'config_unchanged':sha(out/'a.conf')==manifest['config_sha256']}
    (out/'run_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    unchanged=all(sha(Path(path))==h for path,h in inputs.items())
    result={'scope':__doc__,'passed':code==0 and unchanged and not stopped and all(completion[k] for k in
        ('geometry_state_unchanged','executable_unchanged','config_unchanged')),
        'stopped_for_sustained_low_memory':stopped,'native_process':identity,'memory_samples':samples,
        'peak_working_set_bytes':peak,'maximum_observed_private_bytes':private,'minimum_available_bytes':minimum,
        'wall_seconds':time.perf_counter()-start,'inputs_unchanged':unchanged,
        'cold_storage_record_written':(out/'cold_storage.csv').exists(),'goal_complete':False}
    if result['cold_storage_record_written']!=a.release_cold_storage:result['passed']=False
    center_storage=out/'mesh_center_storage.json'
    if center_storage.exists():
        centers=json.loads(center_storage.read_text())
        result['center_storage']=centers
        result['passed']=result['passed'] and centers['lazy_centers']==a.lazy_centers and (centers['coordinate_array_bytes']==0 if a.lazy_centers else centers['coordinate_array_bytes']>0)
    else:
        result['passed']=False
    (out/'cold_run_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
