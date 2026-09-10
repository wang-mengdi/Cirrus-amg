"""Independently verify a completed twofold replay, preserving the initial metadata-check failure.

Only pressure_iterate_storage and Krylov dimension may differ in method metadata.
Actual topology, material weights, RHS bytes, executable and compiled sources
remain checked against the retained run and original failed parent.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-snapshot',type=Path,help='Preserved initial replay-helper bytes; cannot replace geometry, configuration or executable inputs')
    a=p.parse_args();root=a.record.resolve(strict=True);output=a.output.resolve()
    if output.exists():raise ValueError('Preserve earlier verification')
    original=json.loads((root/'completion.json').read_text())
    run=Path(original['output']);parent=Path(original['parent'])
    resolved_inputs={}
    for path,expected in original['input_sha256'].items():
        actual=Path(path)
        if a.source_snapshot is not None and actual.name=='replay_twisted_pressure_twofold.py':
            actual=a.source_snapshot.resolve(strict=True)
        if sha(actual)!=expected:raise ValueError('Original replay input changed: '+path)
        resolved_inputs[str(actual)]=expected
    for path,expected in original['output_sha256'].items():
        if sha(Path(path))!=expected:raise ValueError('Original replay output changed: '+path)
    runtime=json.loads((run/'run_manifest.json').read_text());done=json.loads((run/'run_completion.json').read_text())
    if done['exit_code']!=0 or not all(done[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
        raise ValueError('Replay did not complete successfully')
    for path,expected in {runtime['executable']:runtime['executable_sha256'],runtime['config']:runtime['config_sha256'],**runtime['geometry_input_sha256']}.items():
        if sha(Path(path))!=expected:raise ValueError('Actual executed input changed')
    config=json.loads((run/'case.json').read_text());replay=json.loads((run/'pressure_replay.json').read_text())
    method=json.loads((run/'projection_method.json').read_text());old_method=json.loads((parent/'projection_method.json').read_text())
    if method.get('pressure_iterate_storage')!='twofold' or method.get('gpu_krylov_dimension')!=config.get('gpu_krylov_dimension',20):
        raise ValueError('Missing actual twofold pressure storage or configured dimension')
    for value in (method,old_method):
        value.pop('pressure_iterate_storage',None);value.pop('gpu_krylov_dimension',None)
    if method!=old_method:raise ValueError('A different projection method field changed')
    for name in ('mesh_cells.csv','mesh_faces.csv','material_faces.csv','material_cells.csv','embedded_operator_checks.json'):
        if sha(run/name)!=sha(parent/name):raise ValueError('Actual geometry or material operator changed')
    if sha(run/'pressure_replay_rhs.bin')!=sha(Path(config['pressure_replay']['rhs'])):
        raise ValueError('Different pressure RHS')
    if config.get('restart_checkpoint') or not config['operator_only'] or (run/'time_history.csv').exists() or list(run.glob('step_*')):
        raise ValueError('Expected diagnostic-only zero-flow replay')
    if len(replay['trials'])!=config['pressure_replay']['repetitions'] or not replay['all_solve_checks_passed']:
        raise ValueError('Missing accepted replay trials')
    for trial in replay['trials']:
        if not trial['solver_accepted'] or not trial['original_rhs_accepted'] or trial['solution_storage']!='twofold' or trial['true_relative_residual']>config['linear_tolerance']:
            raise ValueError('Replay failed the original strict pressure equation')
    result=dict(original)
    result.update(scope=__doc__,complete_replay=True,pressure_iterate_storage='twofold',
        projection_method_identical_except_storage_and_dimension=True,
        independently_verified_utc=datetime.now(timezone.utc).isoformat(),
        original_verification={'record':str(root/'completion.json'),'sha256':sha(root/'completion.json'),
                               'complete_replay':original['complete_replay']},
        verifier_sha256=sha(Path(__file__)),failed_original_check_preserved=True)
    result['original_input_sha256']=original['input_sha256']
    result['input_sha256']=resolved_inputs
    if a.source_snapshot is not None:result['initial_replay_helper_snapshot']=str(a.source_snapshot.resolve())
    output.mkdir(parents=True,exist_ok=False)
    (output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete_replay':True,'original_failed_metadata_check_preserved':not original['complete_replay'],
                      'accepted_trials':len(replay['trials'])}))


if __name__=='__main__':main()
