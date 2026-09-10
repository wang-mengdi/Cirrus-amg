"""Run original Aphros from a recorded velocity guess with verified memory controls.

Only the initial velocity and optionally the requested physical duration change
from the source case. The original geometry, equations, scalar precision,
linear AMG backend and tolerances are retained. Completion alone is not a
steady, reference-alignment or spatial-convergence claim.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import psutil
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-case','seed','executable','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--factor-cache-entries',type=int,choices=range(1,9),default=1)
    p.add_argument('--release-cold-storage',action='store_true')
    p.add_argument('--lazy-centers',action='store_true')
    a=p.parse_args()
    if a.steps is not None and a.steps<1:p.error('Require positive physical step count')
    source=a.reference_case.resolve();seed=a.seed.resolve();exe=a.executable.resolve();out=a.output.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep new flow outputs on D')
    if out.exists():raise ValueError('Preserve earlier flow runs')
    cfg=json.loads((source/'case_manifest.json').read_text())
    runtime=json.loads((source/'run_manifest.json').read_text(encoding='utf-8-sig'))
    build_path=exe.parent/'build_manifest.json';build=json.loads(build_path.read_text())
    if not build['passed'] or not build['source_unchanged'] or sha(exe)!=build['executable_sha256']:
        raise ValueError('Require a completed verified original-equation build')
    for path,digest in build['source_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Compiled source changed: '+path)
    drivers=[Path(row['source']) for row in build['results'] if Path(row['source']).name=='driver.cpp']
    if len(drivers)!=1:raise ValueError('Require one compiled driver')
    prepared=json.loads((drivers[0].parent/'prepare_manifest.json').read_text())
    if 'initial_velocity_scope' not in prepared:raise ValueError('Build lacks the recorded velocity-only loader')
    if a.release_cold_storage and 'cold_storage_scope' not in prepared:raise ValueError('Build lacks the verified storage lifetime controls')
    if a.lazy_centers and 'lazy_centers_scope' not in prepared:raise ValueError('Build lacks the verified Cartesian center formulas')
    seedmeta=seed.with_name('seed_manifest.json');info=json.loads(seedmeta.read_text())
    if (sha(seed)!=info['initial_velocity_sha256'] or info['shape']!=cfg['shape'] or
            not info['only_velocity_initialized'] or info['steady_alignment_proven']):
        raise ValueError('Seed bytes, shape or velocity-only scope differ')
    for path,digest in info['source_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Seed preparation input changed: '+path)
    geometry=Path(runtime['geometry_state']).resolve()
    if sha(geometry)!=runtime['geometry_state_sha256'] or sha(source/'a.conf')!=cfg['config_sha256']:
        raise ValueError('Original geometry or case changed')
    settings=runtime['environment'].copy()
    for name,value in {'APHROS_TWISTED_AMG':'1','APHROS_TWISTED_DIRECT':'1',
                       'OMP_NUM_THREADS':'2','OMP_WAIT_POLICY':'PASSIVE'}.items():
        if settings.get(name)!=value:raise ValueError('Require recorded original AMG and thread controls')
    if any(settings.get(name) is not None for name in ('APHROS_TWISTED_GEOMETRY_ONLY','APHROS_TWISTED_GEOMETRY_STATE_ONLY')):
        raise ValueError('The source must describe an actual flow run')
    settings.update(APHROS_TWISTED_ZERO_PAGES='1',APHROS_TWISTED_MEMORY_TRACE='1',
                    APHROS_TWISTED_DIRECT_VELOCITY=None,APHROS_TWISTED_INITIAL_VELOCITY=str(seed),
                    APHROS_TWISTED_GEOMETRY_STATE_IN=str(geometry),
                    APHROS_TWISTED_FACTOR_CACHE=str(a.factor_cache_entries),APHROS_TWISTED_FACTOR_TRACE='1',
                    APHROS_TWISTED_RELEASE_COLD_STORAGE='1' if a.release_cold_storage else None,
                    APHROS_TWISTED_LAZY_CENTERS='1' if a.lazy_centers else None)
    text=(source/'a.conf').read_text()
    text,count=re.subn(r'^set string vel_init (zero|twisted_seed)$','set string vel_init twisted_seed',text,flags=re.M)
    if count!=1:raise ValueError('Require one known initial-velocity setting')
    if a.steps is not None:
        cfg['time_steps']=a.steps
        text,count=re.subn(r'^set double tmax .*$',f"set double tmax {cfg['time_step']*a.steps:.17g}",text,flags=re.M)
        if count!=1:raise ValueError('Require one physical duration setting')
    if psutil.virtual_memory().available<2.5*2**30:raise ValueError('Insufficient available memory to start a new reference')
    out.mkdir(parents=True,exist_ok=False)
    (out/'a.conf').write_text(text);cfg['config_sha256']=sha(out/'a.conf')
    cfg['initial_velocity']={'mode':'twisted_seed','file':str(seed),'sha256':sha(seed),
                             'manifest':str(seedmeta),'manifest_sha256':sha(seedmeta)}
    (out/'case_manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    inputs={str(path.resolve()):sha(path) for path in (Path(__file__),exe,build_path,seed,seedmeta,geometry,
            source/'a.conf',source/'case_manifest.json',source/'run_manifest.json',out/'a.conf',out/'case_manifest.json')}
    now=lambda:datetime.now(timezone.utc).isoformat()
    command=[str(exe),'a.conf']
    preparation={'scope':__doc__,'command':command,'reference_case':str(source),'source_sha256':inputs,
                 'expected_time_steps':cfg['time_steps'],'factor_cache_entries':a.factor_cache_entries,
                 'release_cold_storage':a.release_cold_storage,'lazy_centers':a.lazy_centers,
                 'minimum_available_bytes':int(2.5*2**30),'required_low_memory_seconds':10,'goal_complete':False}
    (out/'initial_run_preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    manifest={'executable':str(exe),'executable_sha256':sha(exe),'config_sha256':sha(out/'a.conf'),
              'geometry_state':str(geometry),'geometry_state_sha256':sha(geometry),'environment':settings,
              'initial_velocity_sha256':sha(seed),'started_utc':now()}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('APHROS_')};env['OPENBLAS_NUM_THREADS']='1'
    for key,value in settings.items():
        if value is not None:env[key]=value
        else:env.pop(key,None)
    start=time.monotonic();peak=0;private=0;minimum=psutil.virtual_memory().available;low=None;stopped=False
    with (out/'run.log').open('w') as log,(out/'memory.csv').open('w',newline='') as memlog:
        writer=csv.writer(memlog);writer.writerow(['elapsed_seconds','pid','working_set_bytes','private_bytes','peak_working_set_bytes','available_system_bytes'])
        child=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        process=psutil.Process(child.pid)
        identity={'pid':process.pid,'creation_time':process.create_time(),'executable':process.exe(),
                  'creation_utc':datetime.fromtimestamp(process.create_time(),timezone.utc).isoformat()}
        if Path(process.exe()).resolve()!=exe or process.cmdline()[-1]!='a.conf' or Path(process.cwd()).resolve()!=out:
            raise ValueError('Unexpected actual reference child identity')
        (out/'process_identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps(identity),flush=True)
        while child.poll() is None:
            try:
                process=psutil.Process(identity['pid'])
                if (process.create_time()!=identity['creation_time'] or Path(process.exe()).resolve()!=exe or
                        process.cmdline()[-1]!='a.conf' or Path(process.cwd()).resolve()!=out):
                    raise ValueError('Actual reference child identity changed')
                memory=process.memory_info();available=psutil.virtual_memory().available;elapsed=time.monotonic()-start
                peak=max(peak,memory.peak_wset);private=max(private,memory.private);minimum=min(minimum,available)
                writer.writerow([elapsed,process.pid,memory.rss,memory.private,memory.peak_wset,available]);memlog.flush()
                low=(low if low is not None else elapsed) if available<2.5*2**30 else None
                if not stopped and low is not None and elapsed-low>=10:
                    process.terminate();stopped=True
            except psutil.NoSuchProcess:pass
            time.sleep(.5)
        code=child.wait()
    completion={'exit_code':code,'completed_utc':now(),'executable_unchanged':sha(exe)==manifest['executable_sha256'],
                'config_unchanged':sha(out/'a.conf')==manifest['config_sha256'],
                'geometry_state_unchanged':sha(geometry)==manifest['geometry_state_sha256'],
                'initial_velocity_unchanged':sha(seed)==manifest['initial_velocity_sha256']}
    (out/'run_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    echo=out/'initial_velocity_echo.bin';centers=out/'mesh_center_storage.json'
    storage=json.loads(centers.read_text()) if centers.exists() else None
    checks={'exit_zero':code==0,'unchanged_inputs':all(sha(Path(path))==digest for path,digest in inputs.items()),
            'actual_initial_storage_bitwise':echo.exists() and sha(echo)==sha(seed),
            'no_resource_stop':not stopped,'cold_storage_record_matches':(out/'cold_storage.csv').exists()==a.release_cold_storage,
            'center_storage_matches':storage is not None and storage['lazy_centers']==a.lazy_centers and
                (storage['coordinate_array_bytes']==0 if a.lazy_centers else storage['coordinate_array_bytes']>0),
            'unchanged_execution_inputs':all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_state_unchanged','initial_velocity_unchanged'))}
    result={'passed':all(checks.values()),'scope':__doc__,'checks':checks,'source_sha256':inputs,'native_process':identity,
            'wall_seconds':time.monotonic()-start,'peak_working_set_bytes':peak,'maximum_observed_private_bytes':private,
            'minimum_available_bytes':minimum,'stopped_for_sustained_low_memory':stopped,'goal_complete':False}
    (out/'initial_run_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
