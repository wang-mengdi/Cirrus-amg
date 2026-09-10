"""Audit captured pressure updates without declaring a flow result converged.

Snapshots complete CSV lines from a live or completed trace. Reconstructs the
stored worst-cell flux balance independently with math.fsum, and distinguishes
the intended correction from changes representable in the stored doubles.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier diagnoses; choose a fresh report')
    root = args.run.resolve()
    inputs = args.output.with_name(args.output.stem+'_inputs')
    inputs.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def snapshot(name, csv_file=False):
        data = (root/name).read_bytes()
        if csv_file:
            # A live writer may be between bytes of its final record.
            data = data[:data.rfind(b'\n')+1]
            if not data:
                raise ValueError(f'No complete header in {name}')
        (inputs/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        return data.decode('utf-8-sig')

    runtime = json.loads(snapshot('run_manifest.json'))
    case = json.loads(snapshot('case.json'))
    method = json.loads(snapshot('projection_method.json'))
    if hashlib.sha256(Path(runtime['executable']).read_bytes()).hexdigest() != runtime['executable_sha256']:
        raise ValueError('Recorded executable changed')
    if case.get('gpu_pressure_gauge') != 'mean_zero' or case.get('linear_backend') != 'native_gpu':
        raise ValueError('This audit requires the compensated mean-zero GPU pressure path')
    updates = list(csv.DictReader(io.StringIO(snapshot('projection_updates.csv', True))))
    faces = list(csv.DictReader(io.StringIO(snapshot('projection_floor_faces.csv', True))))
    if not updates:
        raise ValueError('No pressure-update records')
    for row in updates+faces:
        if None in row or any(v is None or not math.isfinite(float(v)) for v in row.values()):
            raise ValueError('Malformed or nonfinite complete trace record')
    lookup = {(int(r['call']), int(r['pass'])): r for r in updates}
    if len(lookup) != len(updates):
        raise ValueError('Duplicate pressure-update record')
    groups = {}
    for row in faces:
        key = int(row['call']), int(row['pass']), int(row['cell'])
        groups.setdefault(key, []).append(row)
    episodes = []
    for (call, iteration, cell), records in groups.items():
        update = lookup.get((call, iteration))
        if update is None:
            # The face dump is flushed immediately before its update record.
            continue
        volume = float(records[0]['cell_volume'])
        if volume <= 0 or len({r['face'] for r in records}) != len(records):
            raise ValueError('Invalid cell volume or duplicate incident face')
        before, after, correction, details = [], [], [], []
        for r in records:
            owner, neighbor = int(r['owner']), int(r['neighbor'])
            if cell not in (owner, neighbor) or float(r['cell_volume']) != volume:
                raise ValueError('Incident-face geometry mismatch')
            sign = -1 if cell == owner else 1
            a, d, b = (float(r[k]) for k in ('flux_before', 'intended_delta', 'flux_after'))
            if b != a+d:
                raise ValueError('Stored flux update differs from the intended double addition')
            before.append(sign*a); correction.append(sign*d); after.append(sign*b)
            details.append({'face': int(r['face']), 'flux_changed': b != a,
                            'intended_change_in_ulps': abs(d)/math.ulp(a)})
        net = math.fsum(after)
        recorded = float(records[0]['cell_defect'])
        if any(float(r['cell_defect']) != recorded for r in records) or not math.isclose(
                net, recorded, rel_tol=1e-12, abs_tol=math.ulp(recorded)*4):
            raise ValueError('Independent compensated face balance does not match the dump')
        flux_scale = float(update['flux_velocity_scale'])
        pressure_scale = float(update['impulse_range'])
        episodes.append({'call': call, 'pass': iteration, 'cell': cell, 'volume': volume,
                         'divergence_before': math.fsum(before)/volume,
                         'divergence_after': net/volume,
                         'intended_divergence_after_without_per_face_rounding': math.fsum(before+correction)/volume,
                         'global_flux_velocity_correction_relative_linf': float(update['intended_flux_velocity_change_linf'])/max(flux_scale, 1e-300),
                         'global_impulse_correction_relative_linf': float(update['intended_impulse_change_linf'])/max(pressure_scale, 1e-300),
                         'changed_flux_faces': int(update['changed_flux_faces']),
                         'changed_pressure_cells': int(update['changed_pressure_cells']),
                         'incident_faces': details})
    exits = []
    if (root/'projection_roundoff_exits.csv').exists():
        exit_records = list(csv.DictReader(io.StringIO(snapshot('projection_roundoff_exits.csv', True))))
        for row in exit_records:
            if None in row or any(v is None or not math.isfinite(float(v)) for v in row.values()):
                raise ValueError('Malformed or nonfinite roundoff exit')
            call, iteration = int(row['call']), int(row['pass'])
            # An exit can be written after the earlier update-file snapshot.
            # Certify only records covered by that same captured trajectory.
            current = lookup.get((call, iteration))
            if current is None:
                continue
            epsilon = float(row['epsilon'])
            if epsilon != sys.float_info.epsilon or iteration < 3:
                raise ValueError('Invalid roundoff precision or pass')
            residual = float(row['divergence_linf'])
            vscale, pscale = float(row['flux_velocity_scale']), float(row['impulse_scale'])
            if not (1e-10 <= residual <= 1e-8 and vscale > 0 and pscale > 0 and
                    0 <= float(row['flux_velocity_change_linf']) <= epsilon*vscale and
                    0 <= float(row['impulse_change_linf']) <= epsilon*pscale):
                raise ValueError('Exit does not satisfy both correction bounds and the existing pressure bound')
            previous = lookup.get((call, iteration-1))
            if previous is None or float(previous['divergence_linf']) != residual:
                raise ValueError('Exit lacks a repeated divergence residual')
            if float(current['divergence_linf']) != residual:
                raise ValueError('Exit and ordinary pressure trace disagree')
            exits.append(row)
    (inputs/'checker.py').write_bytes(Path(__file__).read_bytes())
    report = {'diagnostic_consistent': True,
              'scope': 'Captured pressure-update arithmetic only; not a completed flow or physical accuracy check',
              'flow_convergence_claimed': False,
              'run': str(root), 'linear_backend': method['linear_backend'],
              'captured_update_records': len(updates), 'stagnation_episodes': episodes,
              'verified_roundoff_exits': exits,
              'input_snapshot_directory': str(inputs.resolve()), 'input_snapshot_sha256': hashes,
              'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'diagnostic_consistent': True, 'updates': len(updates),
                      'episodes': len(episodes), 'roundoff_exits': len(exits),
                      'last_episode': episodes[-1] if episodes else None}, indent=2))


if __name__ == '__main__':
    main()
