"""Lossless projection restarts from a verified, completed native trajectory.

Only a single parent from a zero-start run is supported. A restart preserves
its completed prefix and absolute physical step; it is not a new initial flow.
"""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from check_twisted_time_pair import read, vector
from run_twisted_solver import sha


FORMAT = 'cirrus_projection_restart_v1'
REPO = Path(__file__).resolve().parents[1]
IGNORED = ('output', 'time_steps', 'output_stride', 'dump_iterations', 'restart_checkpoint')


def physical_config(config):
    return {k: v for k, v in config.items() if k not in IGNORED}


def restored_values(folder):
    """Reconstruct every state byte from the actual final unaccelerated dump."""
    metrics = json.loads((folder/'metrics.json').read_text())
    cells = read(folder/'solution.csv')
    final = read(folder/f'iter_{metrics["iterations"]}'/'cells.csv')
    flux = read(folder/'flux.csv')
    if not np.array_equal(cells['id'], np.arange(len(cells))) or not np.array_equal(flux['id'], np.arange(len(flux))):
        raise ValueError('Unordered restart cell or face IDs')
    for name in ('id', 'x', 'y', 'z', 'u', 'v', 'w', 'p'):
        if not np.array_equal(cells[name], final[name]):
            raise ValueError('Final solution and final iteration differ: '+name)
    fields = vector(final, ('u', 'v', 'w', 'p', 'u_diff', 'v_diff', 'w_diff'))
    return np.concatenate((fields.ravel(), flux['flux'])).astype('<f8', copy=False)


def create_checkpoint(run, step, output):
    run, output = run.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Preserve existing checkpoints')
    config = json.loads((run/'case.json').read_text())
    if config.get('restart_checkpoint'):
        raise ValueError('Nested restart parents are not supported')
    if config.get('fluid_solver') != 'proj' or config.get('linear_backend') != 'native_gpu' or any(
            config.get(k) != v for k, v in (('gpu_preconditioner', 'native_amg'), ('gpu_pressure_operator', 'full'), ('gpu_viscosity_operator', 'full'))):
        raise ValueError('Require a complete native AMG projection parent')
    if not 1 <= step <= config['time_steps']:
        raise ValueError('Restart step is outside the parent trajectory')
    output.mkdir(parents=True, exist_ok=False)
    proof_path = output/'parent_native_check.json'
    checker = REPO/'scripts/check_twisted_native_run.py'
    subprocess.run([sys.executable, str(checker), '--run', str(run), '--output', str(proof_path)], check=True)
    shutil.copyfile(checker, output/'parent_checker.py.txt')
    proof = json.loads(proof_path.read_text())
    if not proof['passed']:
        raise ValueError('Parent native trajectory failed')
    summary = json.loads((run/'transient_summary.json').read_text())
    field_steps = [s for s in summary['field_output_steps'] if s <= step]
    if not field_steps or field_steps[0] != 1 or field_steps[-1] != step:
        raise ValueError('Selected restart step has no complete retained fields')
    with (run/'time_history.csv').open() as stream:
        history = list(csv.DictReader(stream))[:step]
    parsed = []
    for i, row in enumerate(history, 1):
        if int(row['step']) != i or row['inner_converged'] != 'true' or float(row['time']) != i*config['time_step']:
            raise ValueError('Invalid completed physical prefix')
        parsed.append({key: (True if key == 'inner_converged' else int(value) if key in ('step', 'inner_iterations') else float(value)) for key, value in row.items()})
    prefix_files = {}
    for i in range(1, step+1):
        for path in sorted((run/f'step_{i:04d}').rglob('*')):
            if path.is_file():
                prefix_files[path.relative_to(run).as_posix()] = {'source': str(path), 'sha256': sha(path)}
    selected = run/f'step_{step:04d}'
    metrics = json.loads((selected/'metrics.json').read_text())
    geometry = Path(config['embedded_geometry'])
    if not geometry.is_absolute():
        geometry = REPO/geometry
    shape = json.loads(geometry.read_text())
    values = restored_values(selected)
    if len(values) != 7*metrics['cells']+metrics['faces']:
        raise ValueError('Incomplete restored state')
    values.tofile(output/'state.bin')
    sources = {**proof['source_sha256'], **proof['executed_input_sha256']}
    sources.update({row['source']: row['sha256'] for row in prefix_files.values()})
    for name in ('mesh_cells.csv', 'mesh_faces.csv', 'projection_method.json'):
        sources[str(run/name)] = sha(run/name)
    report = {'format': FORMAT, 'parent_run': str(run), 'parent_config': config,
              'physical_step': step, 'physical_time': step*config['time_step'], 'time_step': config['time_step'],
              'cells': metrics['cells'], 'faces': metrics['faces'], 'extent': shape['extent'],
              'state_file': 'state.bin', 'state_sha256': sha(output/'state.bin'),
              'state_layout': 'little-endian float64: seven values per cell [u,v,w,p,u_diff,v_diff,w_diff], then all face fluxes in ID order',
              'time_history': parsed, 'field_output_steps': field_steps, 'prefix_files': prefix_files,
              'parent_check_file': proof_path.name, 'parent_check_sha256': sha(proof_path),
              'parent_checker_sha256': sha(output/'parent_checker.py.txt'),
              'source_sha256': sources, 'producer_sha256': sha(Path(__file__))}
    shutil.copyfile(__file__, output/'producer.py.txt')
    (output/'checkpoint.json').write_text(json.dumps(report, indent=2)+'\n')
    load_checkpoint(output/'checkpoint.json')
    return report


def load_checkpoint(path, config=None):
    path = Path(path).resolve()
    meta = json.loads(path.read_text())
    if meta['format'] != FORMAT or meta['state_file'] != 'state.bin' or meta['parent_check_file'] != 'parent_native_check.json':
        raise ValueError('Unsupported restart format or file layout')
    parent = Path(meta['parent_run'])
    if meta['parent_config'].get('restart_checkpoint'):
        raise ValueError('Nested restart parents are not supported')
    if config is not None and (physical_config(config) != physical_config(meta['parent_config']) or config['time_steps'] <= meta['physical_step']):
        raise ValueError('Restart changes the physical/numerical problem or has no remaining steps')
    inputs = dict(meta['source_sha256'])
    inputs.update({str(path): sha(path), str(path.parent/'state.bin'): meta['state_sha256'],
                   str(path.parent/'parent_native_check.json'): meta['parent_check_sha256'],
                   str(path.parent/'parent_checker.py.txt'): meta['parent_checker_sha256'],
                   str(path.parent/'producer.py.txt'): meta['producer_sha256']})
    for source, value in inputs.items():
        if sha(Path(source)) != value:
            raise ValueError('Restart input changed: '+source)
    proof = json.loads((path.parent/'parent_native_check.json').read_text())
    if not proof['passed'] or proof['checker_sha256'] != meta['parent_checker_sha256'] or proof['steps_completed'] != meta['parent_config']['time_steps']:
        raise ValueError('Missing verified complete parent trajectory')
    for source, value in {**proof['source_sha256'], **proof['executed_input_sha256']}.items():
        if inputs.get(source) != value:
            raise ValueError('Restart omitted a verified parent input')
    if meta['parent_config'] != json.loads((parent/'case.json').read_text()):
        raise ValueError('Parent configuration differs')
    step = meta['physical_step']
    if not isinstance(step, int) or not 1 <= step <= proof['steps_completed'] or len(meta['time_history']) != step:
        raise ValueError('Wrong restart physical prefix')
    if meta['physical_time'] != step*meta['time_step'] or meta['time_step'] != meta['parent_config']['time_step']:
        raise ValueError('Wrong restart physical time')
    with (parent/'time_history.csv').open() as stream:
        parent_time = list(csv.DictReader(stream))[:step]
    for i, (row, actual) in enumerate(zip(meta['time_history'], parent_time), 1):
        if row['step'] != i or row['time'] != i*meta['time_step'] or not row['inner_converged'] or actual['inner_converged'] != 'true':
            raise ValueError('Unconverged or misnumbered restart prefix')
        for key in ('step', 'time', 'inner_iterations', 'temporal_acceleration_relative_l2', 'steady_momentum_relative_l2'):
            if row[key] != float(actual[key]):
                raise ValueError('Restart prefix history changed')
    fields = json.loads((parent/'transient_summary.json').read_text())['field_output_steps']
    if meta['field_output_steps'] != [s for s in fields if s <= step] or step not in fields:
        raise ValueError('Restart field schedule changed')
    metrics = json.loads((parent/f'step_{step:04d}'/'metrics.json').read_text())
    geometry = Path(meta['parent_config']['embedded_geometry'])
    if not geometry.is_absolute():
        geometry = REPO/geometry
    if any(meta[key] != metrics[key] for key in ('cells', 'faces')) or meta['extent'] != json.loads(geometry.read_text())['extent']:
        raise ValueError('Restart geometry dimensions differ from the completed parent')
    expected = {p.relative_to(parent).as_posix(): p for i in range(1, step+1) for p in (parent/f'step_{i:04d}').rglob('*') if p.is_file()}
    if set(meta['prefix_files']) != set(expected):
        raise ValueError('Missing or unexpected preserved prefix files')
    for relative, row in meta['prefix_files'].items():
        if Path(row['source']).resolve() != expected[relative].resolve() or inputs.get(row['source']) != row['sha256']:
            raise ValueError('Prefix file provenance differs')
    if (path.parent/'state.bin').stat().st_size != 8*(7*meta['cells']+meta['faces']):
        raise ValueError('Checkpoint byte length differs from its full state layout')
    state = np.fromfile(path.parent/'state.bin', dtype='<f8')
    original = restored_values(parent/f'step_{step:04d}')
    if not np.isfinite(state).all() or len(state) != 7*meta['cells']+meta['faces'] or state.tobytes() != original.tobytes():
        raise ValueError('Checkpoint does not exactly restore the completed parent state')
    return meta, inputs


def copy_prefix(meta, output):
    for relative, row in meta['prefix_files'].items():
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Do not overwrite a restart prefix')
        shutil.copyfile(row['source'], target)
        if sha(target) != row['sha256']:
            raise ValueError('Restart prefix copy differs')


def verify_resumed_run(root, config, runtime, completion, summary):
    """Verify state, lineage and copied physical prefix before checking the tail."""
    checkpoint = Path(config['restart_checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = REPO/checkpoint
    meta, inputs = load_checkpoint(checkpoint, config)
    step = meta['physical_step']
    if not completion.get('restart_unchanged') or runtime.get('restart_input_sha256') != inputs or runtime.get('restart_step') != step:
        raise ValueError('Restart execution provenance differs')
    if summary.get('restart_step') != step or summary.get('steps_computed_this_run') != config['time_steps']-step:
        raise ValueError('Restart tail is incomplete or counted as a full new trajectory')
    echoed = root/'restart_loaded.bin'
    if sha(echoed) != meta['state_sha256']:
        raise ValueError('C++ restored state differs from the exact parent state')
    for relative, row in meta['prefix_files'].items():
        target = root/relative
        if sha(target) != row['sha256']:
            raise ValueError('Copied physical prefix changed: '+relative)
        inputs[str(target)] = row['sha256']
    for name in ('mesh_cells.csv', 'mesh_faces.csv'):
        parent = Path(meta['parent_run'])/name
        if sha(root/name) != sha(parent):
            raise ValueError('Restart rebuilt a different native grid or face ordering')
    proof = json.loads((checkpoint.parent/'parent_native_check.json').read_text())
    inputs[str(echoed)] = sha(echoed)
    return {'passed': True, 'completed_parent_steps_used': step, 'new_steps_computed': config['time_steps']-step,
            'parent_run': meta['parent_run'], 'exact_state_roundtrip': True,
            'parent_native_check': proof, 'source_sha256': inputs,
            'scope': 'Completed zero-start parent prefix plus separately executed continuation; parent proof may cover later unused parent steps as well'}
