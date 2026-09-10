"""Preserve a completed Aphros physical step while later steps are running.

The next step's iteration log is the barrier proving FinishStep completed all
field writes. The time record and field bytes must remain unchanged throughout
the copy. The original configured time sequence is retained without inventing
a successful process-completion record.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from compare_twisted import read


def digest(data):
    return hashlib.sha256(data).hexdigest()


def completed_step(log, step):
    iterations = [(int(i), float(e)) for i, e in re.findall(r'iter=(\d+), diff=([\d.eE+\-]+)', log)]
    starts = [i for i, (iteration, _) in enumerate(iterations) if iteration == 1]
    if len(starts) != step+1 or starts[0] != 0:
        raise ValueError('Need a log from the next physical step, after all checkpoint writes')
    errors = [iterations[starts[i]-1][1] for i in range(1, len(starts))]
    if any(not 0 <= error < 1e-11 for error in errors):
        raise ValueError('A preceding physical step did not converge')
    return errors[-1]


def validate(root):
    audit = json.loads((root/'reference_checkpoint.json').read_text())
    for name, expected in audit['snapshot_sha256'].items():
        if digest((root/name).read_bytes()) != expected:
            raise ValueError(f'Checkpoint artifact changed: {name}')
    config = json.loads((root/'case_manifest.json').read_text())
    step = audit['completed_step']
    if not 1 <= step < config['time_steps'] or audit['complete_reference_run']:
        raise ValueError('Expected a completed prefix of a longer reference run')
    times = read(root/'tube_b0_time.csv')
    if len(times) != step or not math.isclose(times[-1]['time'], step*config['time_step'], rel_tol=1e-12, abs_tol=0):
        raise ValueError('Physical checkpoint time is inconsistent')
    return audit, completed_step((root/'run.log').read_text(), step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Preserve existing snapshots; choose a fresh directory')
    config = json.loads((source/'case_manifest.json').read_text())
    solver = config.get('fluid_solver', 'simple')
    if solver not in ('simple', 'proj'):
        raise ValueError('Unsupported reference fluid solver')
    if not config['convection'] or not 1 <= args.step < config['time_steps']:
        raise ValueError('Expected a prefix of the configured Navier-Stokes sequence')
    names = ['a.conf', 'case_manifest.json', 'run_manifest.json', 'tube_b0_time.csv',
             f'{solver}_final_b0_cells.csv', f'{solver}_final_b0_faces.csv', 'tube_final_b0_walls.csv']
    names += [f'tube_b0_geometry_{family}.csv' for family in ('cells', 'faces', 'walls', 'polygons')]
    if (source/'proj_final_b0_pressure_rows.csv').exists():
        names += ['proj_final_b0_pressure_rows.csv', 'proj_final_b0_pressure_snapshot.json']
    log = (source/'run.log').read_bytes()
    error = completed_step(log.decode('utf-8-sig'), args.step)
    captured = {name: (source/name).read_bytes() for name in names}
    captured['run.log'] = log
    if (source/'pressure_compatibility.csv').exists():
        # This append-only trace can already contain calls from the next step.
        # Preserve complete lines as a separately scoped prefix, never claim
        # that its last call belongs to the captured physical pressure field.
        trace = (source/'pressure_compatibility.csv').read_bytes()
        captured['pressure_compatibility.csv'] = trace[:trace.rfind(b'\n')+1]
    # The log continues to grow; completed-step artifacts must stay stable.
    if any(captured[name] != (source/name).read_bytes() for name in names):
        raise ValueError('Reference advanced during capture; retry using its new completed step')
    output.mkdir(parents=True)
    for name, data in captured.items():
        (output/name).write_bytes(data)
    audit = {'source': str(source), 'completed_step': args.step,
             'physical_time': args.step*config['time_step'], 'complete_reference_run': False,
             'completed_step_iteration_change': error,
             'barrier': 'Next physical step has started; copied field and time files unchanged across capture',
             'compatibility_trace_scope': 'Complete-line prefix; may include calls from the next physical step' if 'pressure_compatibility.csv' in captured else None,
             'snapshot_sha256': {name: digest(data) for name, data in captured.items()}}
    (output/'reference_checkpoint.json').write_text(json.dumps(audit, indent=2)+'\n')
    validate(output)
    print(json.dumps({'output': str(output), 'completed_step': args.step, 'complete_reference_run': False}))


if __name__ == '__main__':
    main()
