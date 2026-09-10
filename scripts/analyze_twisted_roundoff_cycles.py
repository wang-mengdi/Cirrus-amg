"""Profile repeated pressure-correction cycles in a captured trace prefix.

This is a read-only screening diagnostic, not a new solver stopping rule. Timing
after a candidate point is measured work in the old trajectory; it is not a
prediction for a rerun whose later nonlinear iterates could differ.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys


EPSILON = sys.float_info.epsilon
TRACE_NAMES = ('projection_pressure.csv', 'projection_updates.csv',
               'projection_roundoff_exits.csv')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def complete_lines(data):
    prefix = data[:data.rfind(b'\n')+1]
    if not prefix:
        raise ValueError('No complete CSV header')
    return prefix


def parse(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError('Missing or duplicate CSV columns')
    result = []
    for record in reader:
        if None in record or any(v is None for v in record.values()):
            raise ValueError('Malformed complete CSV record')
        row = {k: float(v) for k, v in record.items()}
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite complete CSV record')
        for name in ('call', 'pass'):
            if row[name] < 1 or not row[name].is_integer():
                raise ValueError('Invalid call or pass index')
            row[name] = int(row[name])
        result.append(row)
    keys = [(r['call'], r['pass']) for r in result]
    if keys != sorted(set(keys)):
        raise ValueError('Duplicate or out-of-order trace record')
    return result


def small_update(row, drive_per_dt):
    # The ordinary update trace records BEFORE-update scales; the solver uses
    # AFTER-update scales. Screen with a factor-two margin and only consider
    # episodes where the unchanged driving term dominates the impulse range.
    # This remains a proxy, not a certification of an executable stopping rule.
    drive = row['time_step'] * drive_per_dt
    vscale = row['flux_velocity_scale']
    dv = row['intended_flux_velocity_change_linf']
    dp = row['intended_impulse_change_linf']
    return (1e-10 <= row['divergence_linf'] <= 1e-8 and
            0 <= row['compensated_divergence_linf'] <= 1e-8 and
            drive > 0 and vscale > 0 and
            0 <= dv <= 0.5 * EPSILON * vscale and
            0 <= dp <= 0.5 * EPSILON * drive and
            0 <= row['impulse_range'] and
            row['impulse_range'] + 2 * dp <= 0.5 * drive)


def cycle_period(rows, drive_per_dt):
    for period in (2, 3, 4):
        if len(rows) < 2 * period:
            continue
        window = rows[-2*period:]
        if not all(small_update(r, drive_per_dt) for r in window):
            continue
        values = [r['divergence_linf'] for r in window]
        # Constant sequences are already covered by the current scalar-repeat
        # rule. Require two entire identical periods of a nonconstant cycle.
        if len(set(values)) > 1 and values[:period] == values[period:]:
            return period
    return None


def analyze(timing, updates, exits, drive_per_dt, pass_limit):
    if not timing or not updates or drive_per_dt <= 0 or not math.isfinite(drive_per_dt):
        raise ValueError('Require observed pressure updates and a positive drive scale')
    # The last call could still be running. Only a subsequent call proves that
    # a traced call has ended. Calls with no correction produce no trace rows.
    cutoff = min(timing[-1]['call'], updates[-1]['call'])
    keyed = lambda rows: {(r['call'], r['pass']): r for r in rows if r['call'] < cutoff}
    t, u, x = keyed(timing), keyed(updates), keyed(exits)
    if not t or set(t) != set(u):
        raise ValueError('Incomplete paired records in the closed-call prefix')
    groups = collections.defaultdict(list)
    for key in t:
        if any(t[key][k] != u[key][k] for k in ('time_step', 'divergence_linf')):
            raise ValueError('Paired pressure traces disagree')
        if t[key]['time_step'] <= 0 or t[key]['pass_seconds'] < 0:
            raise ValueError('Invalid pressure timing')
        groups[key[0]].append(u[key])
    for call, rows in groups.items():
        if [r['pass'] for r in rows] != list(range(1, len(rows)+1)) or len(rows) > pass_limit:
            raise ValueError('Missing pressure pass or exceeded configured pass limit')
        if len({r['time_step'] for r in rows}) != 1:
            raise ValueError('Time step changed within a pressure call')
    for key, row in x.items():
        call, iteration = key
        previous = u.get((call, iteration-1))
        if key not in u or previous is None or iteration != len(groups[call]):
            raise ValueError('Roundoff exit does not end an observed call')
        r = row['divergence_linf']
        if not (iteration >= 3 and row['epsilon'] == EPSILON and
                1e-10 <= r <= 1e-8 and r == previous['divergence_linf'] == u[key]['divergence_linf'] and
                row['time_step'] == u[key]['time_step'] and
                row['flux_velocity_scale'] > 0 and row['impulse_scale'] > 0 and
                0 <= row['flux_velocity_change_linf'] <= EPSILON*row['flux_velocity_scale'] and
                0 <= row['impulse_change_linf'] <= EPSILON*row['impulse_scale']):
            raise ValueError('Recorded exit violates the existing stopping rule')
    candidates = []
    for call, rows in groups.items():
        for index in range(len(rows)):
            period = cycle_period(rows[:index+1], drive_per_dt)
            if period is None:
                continue
            tail = rows[index+1:]
            if tail:
                window = rows[index+1-2*period:index+1]
                candidates.append({
                    'call': call, 'time_step': rows[index]['time_step'],
                    'candidate_pass': index+1, 'period': period, 'actual_last_pass': len(rows),
                    'actual_scalar_repeat_exit': (call, len(rows)) in x,
                    'window_divergence': [r['divergence_linf'] for r in window],
                    'window_compensated_divergence': [r['compensated_divergence_linf'] for r in window],
                    'maximum_velocity_correction_in_epsilon_scales': max(
                        r['intended_flux_velocity_change_linf']/(EPSILON*r['flux_velocity_scale']) for r in window),
                    'maximum_impulse_correction_in_epsilon_drive_scales': max(
                        r['intended_impulse_change_linf']/(EPSILON*r['time_step']*drive_per_dt) for r in window),
                    'later_observed_passes': len(tail),
                    'later_observed_seconds': math.fsum(t[(call, r['pass'])]['pass_seconds'] for r in tail),
                    'later_minimum_divergence': min(r['divergence_linf'] for r in tail),
                    'later_maximum_divergence': max(r['divergence_linf'] for r in tail)})
            break
    return {
        'trace_consistent': True, 'flow_convergence_claimed': False,
        'stopping_rule_implemented': False, 'predicted_flow_speedup': None,
        'scope': __doc__, 'excluded_call_at_or_above': cutoff,
        'observed_closed_calls': len(groups), 'observed_closed_passes': len(t),
        'actual_roundoff_exits_checked': len(x),
        'closed_call_pass_histogram': dict(sorted(collections.Counter(len(v) for v in groups.values()).items())),
        'closed_pass_seconds': math.fsum(r['pass_seconds'] for r in t.values()),
        'candidate_calls': len(candidates),
        'later_observed_passes': sum(r['later_observed_passes'] for r in candidates),
        'later_observed_seconds': math.fsum(r['later_observed_seconds'] for r in candidates),
        'drive_per_dt': drive_per_dt,
        'screening_rule': 'Two exact nonconstant periods of length 2..4; every row has raw divergence in [1e-10,1e-8], compensated divergence <=1e-8, corrections <=0.5 epsilon times before-velocity/driving-impulse scales, and before-impulse range plus twice correction <=0.5 driving scale. This proxy does not replace the existing final mass or fixed-point gates.',
        'candidates': candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, build, out = (p.resolve() for p in (args.run, args.build, args.output))
    out.mkdir(parents=True, exist_ok=False)
    inputs = out/'inputs'
    inputs.mkdir()
    snapshots = {}

    def capture(source, name, trace=False):
        data = source.read_bytes()
        size = len(data)
        if trace:
            data = complete_lines(data)
        (inputs/name).write_bytes(data)
        snapshots[name] = {'source': str(source), 'sha256': hashlib.sha256(data).hexdigest(),
                           'captured_bytes': len(data), 'discarded_partial_bytes': size-len(data)}
        return data

    runtime = json.loads(capture(root/'run_manifest.json', 'run_manifest.json'))
    case = json.loads(capture(root/'case.json', 'case.json'))
    method = json.loads(capture(root/'projection_method.json', 'projection_method.json'))
    manifest = json.loads(capture(build/'build_manifest.json', 'build_manifest.json'))
    executable = Path(runtime['executable'])
    if (executable.resolve() != (build/'simple_channel.exe').resolve() or
            sha(executable) != runtime['executable_sha256'] or
            runtime['executable_sha256'] != manifest['executable_sha256']['simple_channel.exe']):
        raise ValueError('Run/build executable provenance differs')
    if (manifest['exit_code'] != 0 or not manifest['source_unchanged'] or
            any(r['exit_code'] != 0 for r in manifest['processes'])):
        raise ValueError('Native build did not complete with unchanged sources')
    configuration = Path(runtime['config'])
    if sha(configuration) != runtime['config_sha256'] or json.loads(configuration.read_bytes()) != case:
        raise ValueError('Run configuration changed or differs from the recorded case')
    source = capture(build/'sources/simple/ProjectionSolver.cpp.txt', 'ProjectionSolver.cpp.txt')
    if hashlib.sha256(source).hexdigest() != manifest['source_sha256']['simple\\ProjectionSolver.cpp']:
        raise ValueError('Compiled projection source snapshot changed')
    capture(Path(__file__).resolve(), 'analyzer.py')
    if case.get('linear_backend') != 'native_gpu' or case.get('gpu_pressure_gauge') != 'mean_zero':
        raise ValueError('Require the native mean-zero pressure path')
    if method.get('pressure_iterate_storage') != 'twofold':
        raise ValueError('This profile is scoped to the current twofold pressure path')
    geometry_path = Path(case['embedded_geometry'])
    if not geometry_path.is_absolute():
        geometry_path = Path(__file__).resolve().parents[1]/geometry_path
    geometry = json.loads(capture(geometry_path.resolve(), 'geometry.json'))
    expected_geometry = runtime['geometry_input_sha256'].get(str(geometry_path.resolve()))
    if snapshots['geometry.json']['sha256'] != expected_geometry:
        raise ValueError('Geometry metadata differs from the run launch')
    # Read timing first and updates second; the solver flushes updates before
    # timing. The final call remains excluded even if a live snapshot advances.
    tables = [parse(capture(root/name, name, True)) for name in TRACE_NAMES]
    drive = max(math.sqrt(math.fsum(v*v for v in case['force'])), 1e-30)*geometry['extent'][0]
    report = analyze(*tables, drive, case['nonorth_iterations'])
    report.update({'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'run': str(root), 'executable_sha256': runtime['executable_sha256'],
                   'input_snapshot_sha256': snapshots})
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('candidates', 'input_snapshot_sha256', 'scope', 'screening_rule')}, indent=2))


if __name__ == '__main__':
    main()
