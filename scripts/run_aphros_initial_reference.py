"""Run original Aphros from a recorded velocity-only initial guess."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import time
import psutil
from run_twisted_solver import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-case','seed','executable','runner','geometry','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int)
    args=parser.parse_args();source=args.reference_case.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    seed=args.seed.resolve();seedmeta=seed.with_name('seed_manifest.json')
    sm=json.loads(seedmeta.read_text());cfg=json.loads((source/'case_manifest.json').read_text())
    exe=args.executable.resolve();runner=args.runner.resolve();geometry=args.geometry.resolve()
    build=json.loads((exe.parent/'build_manifest.json').read_text())
    if not build['passed'] or sha(exe)!=build['executable_sha256']:raise ValueError('Build not verified')
    if sha(seed)!=sm['initial_velocity_sha256'] or sm['shape']!=cfg['shape']:raise ValueError('Seed does not match the case')
    for p,h in sm['source_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('Seed provenance changed: '+p)
    text=(source/'a.conf').read_text()
    if sha(source/'a.conf')!=cfg['config_sha256']:raise ValueError('Reference config changed')
    text,n=re.subn(r'^set string vel_init zero$', 'set string vel_init twisted_seed',text,flags=re.M)
    if n!=1:raise ValueError('Require one original zero initial condition')
    if args.steps is not None:
        if args.steps<1:raise ValueError('Require positive steps')
        cfg['time_steps']=args.steps
        text,n=re.subn(r'^set double tmax .*$',f"set double tmax {cfg['time_step']*args.steps:.17g}",text,flags=re.M)
        if n!=1:raise ValueError('Unknown physical duration')
    (out/'a.conf').write_text(text)
    cfg['config_sha256']=sha(out/'a.conf')
    cfg['initial_velocity']={'mode':'twisted_seed','file':str(seed),'sha256':sha(seed),
                             'manifest':str(seedmeta),'manifest_sha256':sha(seedmeta)}
    (out/'case_manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    pwsh='C:/Users/bear/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
    command=[pwsh,'-NoProfile','-File',str(runner),'-CaseDirectory',str(out),'-Executable',str(exe),
             '-Threads','2','-UseAmg','-FactorCacheEntries','3','-TraceFactorCache',
             '-GeometryState',str(geometry),'-InitialVelocity',str(seed)]
    # Preserve the original 64-grid volume compatibility control. The original
    # 16-grid reference does not enable it.
    runtime=json.loads((source/'run_manifest.json').read_text(encoding='utf-8-sig'))
    if runtime['environment'].get('APHROS_TWISTED_VOLUME_COMPATIBILITY')=='1':command.append('-VolumePressureCompatibility')
    inputs={str(p):sha(p) for p in (Path(__file__).resolve(),seed,seedmeta,exe,runner,geometry,
            exe.parent/'build_manifest.json',source/'a.conf',source/'case_manifest.json',
            source/'run_manifest.json',out/'a.conf',out/'case_manifest.json')}
    (out/'initial_run_preparation.json').write_text(json.dumps({'scope':__doc__,'command':command,
        'source_sha256':inputs,'expected_time_steps':cfg['time_steps'],'goal_complete':False},indent=2)+'\n')
    start=time.perf_counter();native=None;identity=None
    with (out/'wrapper.log').open('w') as log:
        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        wrapper=psutil.Process(process.pid)
        print(json.dumps({'wrapper_pid':process.pid,'case':str(out)}),flush=True)
        while process.poll() is None:
            if native is None:
                try:matches=[p for p in wrapper.children(recursive=True) if Path(p.exe()).resolve()==exe]
                except psutil.NoSuchProcess:matches=[]
                if len(matches)>1:raise ValueError('Multiple solver children')
                if matches:
                    native=matches[0]
                    identity={'pid':native.pid,'creation_time':native.create_time(),
                              'creation_utc':datetime.fromtimestamp(native.create_time(),timezone.utc).isoformat(),
                              'executable':str(exe)}
                    (out/'process_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
                    print(json.dumps({'native_process':identity}),flush=True)
            time.sleep(.5)
        code=process.wait()
    completion=json.loads((out/'run_completion.json').read_text(encoding='utf-8-sig'))
    runtime=json.loads((out/'run_manifest.json').read_text(encoding='utf-8-sig'))
    checks={'exit_zero':code==0 and completion['exit_code']==0,
            'geometry_unchanged':completion.get('geometry_state_unchanged') is True,
            'initial_velocity_unchanged':completion.get('initial_velocity_unchanged') is True,
            'input_hash_matches_runtime':runtime.get('initial_velocity_sha256')==sha(seed),
            'input_path_matches_runtime':runtime['environment'].get('APHROS_TWISTED_INITIAL_VELOCITY')==str(seed),
            'actual_initial_storage_bitwise':(out/'initial_velocity_echo.bin').exists() and sha(out/'initial_velocity_echo.bin')==sha(seed),
            'inputs_unchanged':all(sha(Path(p))==h for p,h in inputs.items())}
    result={'scope':__doc__+' Completion does not establish steady alignment; a separate field and mass check is required.',
            'passed':all(checks.values()),'checks':checks,'wall_seconds':time.perf_counter()-start,
            'source_sha256':inputs,'native_process':identity,'goal_complete':False}
    (out/'initial_run_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
