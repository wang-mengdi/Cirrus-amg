"""Evaluate a captured failed device iterate using actual exported face coefficients.

The standalone compiler must supply at least 64 long-double mantissa bits. A
completed arithmetic diagnostic does not accept a failed solve or advance flow.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-record',type=Path,required=True)
    p.add_argument('--oracle-build',type=Path,required=True)
    p.add_argument('--trial',type=int,default=1)
    p.add_argument('--accepted',action='store_true',help='Check the returned twofold solution against the original physical RHS')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    record=a.replay_record.resolve(strict=True);root=a.output.resolve()
    if root.exists():raise ValueError('Preserve previous arithmetic diagnostics')
    completed=json.loads((record/'completion.json').read_text())
    if not completed['complete_replay'] or not completed['inputs_unchanged']:
        raise ValueError('Unverified operator/RHS replay')
    run=Path(completed['output'])
    info=json.loads((run/'pressure_replay.json').read_text())
    trial=info['trials'][a.trial-1]
    if trial['trial']!=a.trial:raise ValueError('Wrong replay trial')
    if a.accepted:
        if not trial['solver_accepted'] or trial.get('solution_storage')!='twofold':
            raise ValueError('Expected an actual accepted twofold solution')
    elif trial['solver_accepted'] or trial['failed_iterate_accepted']:
        raise ValueError('Expected an actual unaccepted iterate')
    build=a.oracle_build.resolve(strict=True);manifest=json.loads((build/'build_manifest.json').read_text())
    exe=build/'pressure_arithmetic_oracle.exe'
    if manifest['exit_code'] or not manifest['source_unchanged'] or sha(exe)!=manifest['executable_sha256']:
        raise ValueError('Unverified arithmetic executable')
    source=build/'native_pressure_arithmetic_oracle.cpp'
    if sha(source)!=manifest['source_sha256']:raise ValueError('Oracle source snapshot changed')
    prefix=run/f'pressure_replay_{a.trial}'
    if a.accepted:
        state,ax,low=Path(str(prefix)+'_solution.bin'),Path(str(prefix)+'_gpu_ax.bin'),Path(str(prefix)+'_solution_low.bin')
        scale=1.;expectedResidual=trial['returned_solution_gpu_residual']
    else:
        state,ax,low=Path(str(prefix)+'_failed_scaled_iterate.bin'),Path(str(prefix)+'_failed_scaled_ax.bin'),Path(str(prefix)+'_failed_scaled_low.bin')
        scale=trial['rhs_scale'];expectedResidual=trial['true_relative_residual']
    paths=[run/'pressure_replay_operator.bin',run/'pressure_replay_rhs.bin',state,ax]
    inputs=paths+[record/'completion.json',run/'pressure_replay.json',run/'run_manifest.json',run/'run_completion.json',
                  build/'build_manifest.json',source,exe,Path(__file__)]
    if low.exists():inputs.append(low)
    before={str(path):sha(path) for path in inputs}
    for path,expected in completed['output_sha256'].items():
        if sha(Path(path))!=expected:raise ValueError('Actual replay dump changed')
    root.mkdir(parents=True,exist_ok=False)
    command=[str(exe),*map(str,paths),repr(scale),str(root/'oracle'),'accepted-twofold-pressure' if a.accepted else 'failed-device-pressure']
    if low.exists():command.append(str(low))
    env=os.environ.copy();runtime=str(Path(manifest['compiler']).parent)
    env['PATH']=runtime+os.pathsep+env.get('PATH','')
    (root/'launch.json').write_text(json.dumps({'scope':__doc__,'command':command,'inputs':before,'runtime_path_prefix':runtime},indent=2)+'\n')
    with (root/'oracle.log').open('w') as log:
        process=subprocess.run(command,cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError('Arithmetic oracle failed')
    result=json.loads((root/'oracle/result.json').read_text())
    rhs=np.fromfile(paths[1],dtype='<f8')/scale
    actual=np.fromfile(paths[3],dtype='<f8')
    if len(rhs)!=info['cells'] or len(actual)!=len(rhs):raise ValueError('Wrong failed state length')
    residual=float(np.linalg.norm(rhs-actual)/np.linalg.norm(rhs))
    checked=(result['long_double_mantissa_bits']>=64 and result['cells']==info['cells']
        and result['interface_faces']==info['full_pressure_interface_faces']
        and abs(residual-expectedResidual)<1e-25 and abs(residual-result['gpu_residual'])<1e-25)
    data={'scope':__doc__,'arithmetic_check_complete':checked,'oracle':result,'replay_trial':trial,
          'numpy_gpu_residual':residual,'input_sha256':before,
          'inputs_unchanged':all(sha(Path(path))==h for path,h in before.items()),'failed_solution_accepted':False,
          'goal_complete':False,'output_sha256':{str(path):sha(path) for path in (root/'oracle').iterdir() if path.is_file()}}
    data['accepted_solution_checked']=a.accepted
    if a.accepted:
        data['returned_twofold_residuals_passed']=result['twofold_input'] and residual<=info['linear_tolerance'] and result['accurate_residual']<=info['linear_tolerance']
        checked &= data['returned_twofold_residuals_passed']
        data['arithmetic_check_complete']=checked
    data['arithmetic_check_complete'] &= data['inputs_unchanged']
    (root/'result.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('input_sha256','output_sha256')}))
    if not data['arithmetic_check_complete']:raise SystemExit(1)


if __name__=='__main__':main()
