"""Capture one fully written original Aphros step, retaining its configured trajectory and exact scalar dumps.

A snapshot is not a new solver run. Later-step logs or real terminal evidence
establish the write barrier; all copied state bytes are checked before and after.
"""
import argparse
import csv
from datetime import datetime,timezone
from decimal import Decimal,localcontext
import json
import math
from pathlib import Path
import re
import shutil
import psutil
from run_twisted_solver import sha
from check_aphros_coupled_precision_probe import hex_decimal

FORMAT='aphros_extended_completed_step_v1'
FILES=('a.conf','case_manifest.json','run_manifest.json','tube_b0_time.csv',
       'proj_final_b0_cells.csv','proj_final_b0_faces.csv','tube_final_b0_walls.csv',
       'proj_final_b0_exact.json','proj_final_b0_exact_cells.csv','proj_final_b0_exact_faces.csv',
       'tube_b0_geometry_cells.csv','tube_b0_geometry_faces.csv','tube_b0_geometry_walls.csv','tube_b0_geometry_polygons.csv')


def step_barrier(log,step,total,tolerance,terminal):
    entries=[(int(i),float(e)) for i,e in re.findall(r'iter=(\d+), diff=([^\s]+)',log)]
    previous=0
    for iteration,error in entries:
        if iteration not in (1,previous+1) or not math.isfinite(error) or error<0:
            raise ValueError('Incomplete or invalid inner iteration sequence')
        previous=iteration
    starts=[i for i,(iteration,_) in enumerate(entries) if iteration==1]
    expected=step if terminal else step+1
    if len(starts)!=expected or not starts or starts[0]!=0:raise ValueError('No exact completed-step write barrier')
    if terminal and (step!=total or 'End of simulation: original Aphros Proj' not in log):
        raise ValueError('Terminal evidence does not cover the prescribed trajectory')
    ends=[starts[i]-1 for i in range(1,len(starts))]
    if terminal:ends.append(len(entries)-1)
    errors=[entries[i][1] for i in ends]
    if len(errors)!=step or any(not math.isfinite(e) or not 0<=e<min(tolerance,1e-11) for e in errors):
        raise ValueError('A selected physical step did not converge')
    return errors[-1]


def validate_capture(root):
    root=Path(root).resolve();audit=json.loads((root/'extended_step_checkpoint.json').read_text())
    if audit['format']!=FORMAT or audit['new_solver_run_claimed'] or audit['complete_reference_run']:
        raise ValueError('Unsupported or falsely relabelled reference snapshot')
    if not (set(FILES)|{'run.log'})<=set(audit['snapshot_sha256']):raise ValueError('Missing required snapshot inputs')
    if sha(root/'producer.py.txt')!=audit['producer_sha256']:raise ValueError('Captured producer bytes changed')
    for name,digest in audit['snapshot_sha256'].items():
        path=(root/name).resolve()
        if root not in path.parents or sha(path)!=digest:raise ValueError('Snapshot field changed: '+name)
    cfg=json.loads((root/'case_manifest.json').read_text());runtime=json.loads((root/'run_manifest.json').read_text())
    if cfg['fluid_solver']!='proj' or not cfg['convection']:raise ValueError('Require original Navier-Stokes Proj')
    if audit['configured_steps']!=cfg['time_steps']:raise ValueError('Configured trajectory was relabelled')
    expected_inputs={runtime['executable']:runtime['executable_sha256'],runtime['geometry_state']:runtime['geometry_state_sha256']}
    if audit['source_input_sha256']!=expected_inputs:raise ValueError('Captured source inputs differ from actual run metadata')
    step=audit['completed_step']
    if type(step) is not int or not 1<=step<=cfg['time_steps']:raise ValueError('Wrong physical step')
    if sha(root/'a.conf')!=cfg['config_sha256'] or runtime['config_sha256']!=cfg['config_sha256']:
        raise ValueError('Actual executed configuration differs')
    if audit['physical_time']!=step*cfg['time_step']:raise ValueError('Wrong snapshot physical time')
    terminal=audit['source_run_complete_proven']
    if terminal:
        if 'parent_run_completion.json' not in audit['snapshot_sha256']:raise ValueError('Missing actual terminal evidence')
        done=json.loads((root/'parent_run_completion.json').read_text())
        if done['exit_code'] or not done.get('geometry_state_unchanged') or any(
                k in done and not done[k] for k in ('executable_unchanged','config_unchanged','initial_velocity_unchanged')):
            raise ValueError('Source flow did not finish with unchanged inputs')
        # Older producers record executable/configuration hashes in their run
        # metadata instead of duplicating boolean flags in the completion file.
        if sha(Path(runtime['executable']))!=runtime['executable_sha256']:
            raise ValueError('Source executable differs from its actual run')
    elif step>=cfg['time_steps'] or not audit['source_observation'].get('confirmed_live'):
        raise ValueError('Missing live source observation for a partial trajectory')
    error=step_barrier((root/'run.log').read_text(),step,cfg['time_steps'],cfg['iteration_tolerance'],terminal)
    with (root/'tube_b0_time.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=step:raise ValueError('Physical history length differs')
    for i,row in enumerate(rows,1):
        if any(not math.isfinite(float(v)) for v in row.values()):raise ValueError('Nonfinite physical history')
        if not math.isclose(float(row['time']),i*cfg['time_step'],rel_tol=1e-12,abs_tol=0):raise ValueError('Physical time order differs')
        if not math.isclose(float(row['time_step']),cfg['time_step'],rel_tol=1e-12,abs_tol=0):raise ValueError('Physical dt differs')
        if any(float(row[k])<0 for k in ('velocity_change_linf','velocity_change_volume_l2','temporal_acceleration_volume_l2')):
            raise ValueError('Negative physical change diagnostic')
    exact=json.loads((root/'proj_final_b0_exact.json').read_text())
    with localcontext() as c:
        c.prec=100;actual=hex_decimal(exact['physical_time_hex']);expected=Decimal.from_float(cfg['time_step'])*step
        if abs(actual-expected)>abs(expected)*Decimal('1e-12'):raise ValueError('Exact stored field time differs from history')
    if audit['completed_step_iteration_change']!=error:raise ValueError('Captured final iteration error differs')
    return audit,error


def capture(source,output,step,pid=None,creation_time=None):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Preserve existing snapshots')
    cfg=json.loads((source/'case_manifest.json').read_text());runtime=json.loads((source/'run_manifest.json').read_text())
    terminal=step==cfg['time_steps'];observation=None
    if not terminal:
        if pid is None or creation_time is None:raise ValueError('Require the exact live source identity')
        p=psutil.Process(pid)
        if abs(p.create_time()-creation_time)>1e-5 or Path(p.exe()).resolve()!=Path(runtime['executable']).resolve():
            raise ValueError('Live Aphros identity differs')
        if Path(p.cwd()).resolve()!=source or p.cmdline()[-1]!='a.conf':raise ValueError('Live Aphros configuration differs')
        observation={'pid':pid,'created':p.create_time(),'executable':p.exe(),'command':p.cmdline(),'confirmed_live':True}
    log=(source/'run.log').read_bytes();log=log[:log.rfind(b'\n')+1]
    error=step_barrier(log.decode('utf-8-sig'),step,cfg['time_steps'],cfg['iteration_tolerance'],terminal)
    names=list(FILES)
    for name in ('initial_velocity_echo.bin','initial_run_preparation.json','proj_final_b0_pressure_rows.csv',
                 'proj_final_b0_pressure_snapshot.json','proj_final_b0_pressure_faces.csv','proj_final_b0_pressure_faces.json'):
        if (source/name).is_file():names.append(name)
    originals={name:source/name for name in names}
    if terminal:originals['parent_run_completion.json']=source/'run_completion.json'
    inputs={runtime['executable']:runtime['executable_sha256'],runtime['geometry_state']:runtime['geometry_state_sha256']}
    if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('Source executable or geometry changed')
    before={name:sha(path) for name,path in originals.items()}
    output.mkdir(parents=True,exist_ok=False)
    for name,path in originals.items():
        shutil.copyfile(path,output/name)
        if sha(output/name)!=before[name]:raise ValueError('Snapshot copy changed: '+name)
    (output/'run.log').write_bytes(log);before['run.log']=sha(output/'run.log')
    if any(sha(path)!=before[name] for name,path in originals.items()):raise ValueError('Aphros advanced during field capture')
    if not (source/'run.log').read_bytes().startswith(log):raise ValueError('Source log prefix changed')
    if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('Source inputs changed during capture')
    audit={'format':FORMAT,'scope':__doc__,'source':str(source),'captured_utc':datetime.now(timezone.utc).isoformat(),
           'completed_step':step,'physical_time':step*cfg['time_step'],'configured_steps':cfg['time_steps'],
           'completed_step_iteration_change':error,'source_run_complete_proven':terminal,'complete_reference_run':False,
           'new_solver_run_claimed':False,'source_observation':observation,'snapshot_sha256':before,
           'source_input_sha256':inputs,'producer_sha256':sha(Path(__file__))}
    (output/'extended_step_checkpoint.json').write_text(json.dumps(audit,indent=2)+'\n')
    shutil.copyfile(__file__,output/'producer.py.txt')
    validate_capture(output)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--pid',type=int);p.add_argument('--creation-time',type=float)
    a=p.parse_args();j=capture(a.source,a.output,a.step,a.pid,a.creation_time)
    print(json.dumps({k:j[k] for k in ('completed_step','physical_time','source_run_complete_proven','new_solver_run_claimed')}))
