"""Validate an immutable completed Proj step inside a longer native trajectory.

The flushed time_history row follows writeStep in ProjectionSolver::run. This
barrier proves the selected fields were written, without inventing run completion.
"""
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from run_twisted_solver import sha


def validate(folder):
    folder=folder.resolve();root=folder.parent
    matched=re.fullmatch(r'step_(\d+)',folder.name)
    if not matched:raise ValueError('Expected an explicit native physical-step directory')
    step=int(matched[1]);case=json.loads((folder/'case.json').read_text())
    parent=json.loads((root/'case.json').read_text());runtime=json.loads((root/'run_manifest.json').read_text())
    normalize=lambda c:{k:v for k,v in c.items() if k not in ('output','physical_time','physical_step')}
    if normalize(case)!=normalize(parent) or case.get('fluid_solver')!='proj' or not 1<=step<parent['time_steps']:
        raise ValueError('Expected a matching completed prefix of a longer Proj trajectory')
    if case['physical_step']!=step or not math.isclose(case['physical_time'],step*case['time_step'],rel_tol=1e-12,abs_tol=0):
        raise ValueError('Physical step time differs from its index')
    inputs={runtime['executable']:runtime['executable_sha256'],runtime['config']:runtime['config_sha256']}
    inputs.update(runtime['geometry_input_sha256'])
    if any(sha(Path(p))!=value for p,value in inputs.items()):raise ValueError('Executed native input changed')
    trace=(root/'time_history.csv').read_bytes();lines=trace.splitlines(keepends=True)
    if len(lines)<step+1 or not lines[step].endswith(b'\n'):raise ValueError('Selected step has no completed write barrier')
    prefix=b''.join(lines[:step+1]);records=list(csv.DictReader(io.StringIO(prefix.decode())))
    for i,r in enumerate(records,1):
        if int(r['step'])!=i or r['inner_converged']!='true' or not math.isclose(float(r['time']),i*case['time_step'],rel_tol=1e-12,abs_tol=0):
            raise ValueError('Prefix contains missing or unconverged physical steps')
    metrics=json.loads((folder/'metrics.json').read_text())
    if not metrics['converged'] or metrics['iterations']!=int(records[-1]['inner_iterations']):
        raise ValueError('Selected-step diagnostics differ from the completed write barrier')
    paths=[root/'run_manifest.json',root/'case.json']+[folder/n for n in (
        'case.json','metrics.json','solution.csv','walls.csv','flux.csv','mesh_cells.csv','mesh_faces.csv','sections.csv')]
    hashes={str(p):sha(p) for p in paths}
    return {'scope':__doc__,'completed_step':step,'physical_time':case['physical_time'],
            'complete_native_run':False,'time_history_prefix':prefix.decode(),
            'time_history_prefix_sha256':hashlib.sha256(prefix).hexdigest(),
            'source_sha256':hashes,'executed_input_sha256':inputs,'validator_sha256':sha(Path(__file__))}


def verify_unchanged(checkpoint):
    for group in ('source_sha256','executed_input_sha256'):
        if any(sha(Path(p))!=value for p,value in checkpoint[group].items()):
            raise ValueError('Completed native step or runtime input changed during comparison')
