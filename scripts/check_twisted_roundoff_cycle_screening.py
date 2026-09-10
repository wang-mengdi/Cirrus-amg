"""Exercise cycle-screening counterexamples and replay an immutable real prefix."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import analyze_twisted_roundoff_cycles as profile


def main():
    global profile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analyzer', type=Path,
                        help='Use the captured analyzer bytes when replaying an archived snapshot')
    args = parser.parse_args()
    if args.analyzer is not None:
        spec = importlib.util.spec_from_file_location('captured_cycle_analyzer', args.analyzer.resolve())
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []

    def check(name, passed):
        if not passed:
            raise ValueError('Counterexample failed: '+name)
        results.append(name)

    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            results.append(name)
        else:
            raise ValueError('Invalid input was accepted: '+name)

    def rows(values):
        return [{'call': 1, 'pass': i+1, 'time_step': 0.0025, 'divergence_linf': v,
                 'compensated_divergence_linf': 1e-9,
                 'intended_flux_velocity_change_linf': 1e-18, 'flux_velocity_scale': 0.025,
                 'intended_impulse_change_linf': 1e-21, 'impulse_range': 1e-5}
                for i, v in enumerate(values)]

    for period in (2, 3, 4):
        values = [1e-9*(i+1) for i in range(period)]*2
        check('recognize_period_'+str(period), profile.cycle_period(rows(values), .25) == period)
    valid = rows([1e-9, 2e-9]*2)
    for name, data in (
            ('single_period', rows([1e-9, 2e-9])),
            ('constant_has_existing_rule', rows([1e-9]*8)),
            ('decreasing_no_cycle', rows([8e-9, 7e-9, 6e-9, 5e-9, 4e-9, 3e-9, 2e-9, 1e-9])),
            ('period_outside_bound', rows([1e-9, 2e-9, 3e-9, 4e-9, 5e-9]*2))):
        check('reject_'+name, profile.cycle_period(data, .25) is None)
    for field, value in (
            ('intended_flux_velocity_change_linf', 1e-15),
            ('intended_impulse_change_linf', 1e-17),
            ('compensated_divergence_linf', 1.01e-8),
            ('impulse_range', 0.001),
            ('flux_velocity_scale', 0.),
            ('divergence_linf', 1.01e-8)):
        data = copy.deepcopy(valid)
        # A small LAST update alone cannot qualify the preceding cycle.
        data[0][field] = value
        check('reject_window_'+field, profile.cycle_period(data, .25) is None)

    timing = [{'call': r['call'], 'pass': r['pass'], 'time_step': r['time_step'],
               'divergence_linf': r['divergence_linf'], 'pass_seconds': 1.}
              for r in valid]
    # A subsequent call closes the first one; an extra update in the open call
    # is legitimate when the writer advanced between the two file reads.
    t = timing + [dict(timing[0], call=2)]
    u = valid + [dict(valid[0], call=2), dict(valid[1], call=2)]
    a = profile.analyze(t, u, [], .25, 64)
    check('exclude_open_call_and_racing_update', a['observed_closed_passes'] == 4 and a['observed_closed_calls'] == 1)
    rejects('missing_closed_record', lambda: profile.analyze(t, u[1:], [], .25, 64))
    mismatched = copy.deepcopy(u)
    mismatched[0]['divergence_linf'] *= 1.1
    rejects('mismatched_pair', lambda: profile.analyze(t, mismatched, [], .25, 64))
    rejects('exceeded_pass_limit', lambda: profile.analyze(t, u, [], .25, 3))
    invalid_exit = dict(call=1, **{'pass': 4}, time_step=.0025, divergence_linf=2e-9,
                        epsilon=profile.EPSILON, flux_velocity_scale=.025, impulse_scale=.000625,
                        flux_velocity_change_linf=1e-18, impulse_change_linf=1e-21)
    rejects('cycle_is_not_current_scalar_exit', lambda: profile.analyze(t, u, [invalid_exit], .25, 64))
    header = b'call,pass,value\n'
    check('drop_partial_last_line', profile.complete_lines(header+b'1,1,2\n1,2,') == header+b'1,1,2\n')
    rejects('no_complete_header', lambda: profile.complete_lines(b'call,pass'))
    for name, body in (
            ('duplicate', b'1,1,2\n1,1,2\n'), ('order', b'1,2,2\n1,1,2\n'),
            ('nonfinite', b'1,1,nan\n'), ('noninteger_pass', b'1,1.5,2\n'),
            ('missing_value', b'1,1\n'), ('extra_value', b'1,1,2,3\n')):
        rejects('parse_'+name, lambda body=body: profile.parse(header+body))

    original = json.loads((args.snapshot/'result.json').read_bytes())
    inputs = args.snapshot/'inputs'
    for name, info in original['input_snapshot_sha256'].items():
        check('snapshot_hash_'+name, profile.sha(inputs/name) == info['sha256'])
    check('same_analyzer_bytes', profile.sha(Path(profile.__file__)) == profile.sha(inputs/'analyzer.py'))
    tables = [profile.parse((inputs/name).read_bytes()) for name in profile.TRACE_NAMES]
    case = json.loads((inputs/'case.json').read_bytes())
    replay = profile.analyze(*tables, original['drive_per_dt'], case['nonorth_iterations'])
    check('real_prefix_replay_identical', all(original[k] == v for k, v in replay.items()
                                           if k != 'closed_call_pass_histogram'))
    check('histogram_replay_identical', original['closed_call_pass_histogram'] ==
          {str(k): v for k, v in replay['closed_call_pass_histogram'].items()})
    (args.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    report = {'passed': True, 'scope': __doc__, 'flow_convergence_claimed': False,
              'checks': results, 'snapshot': str(args.snapshot.resolve()),
              'snapshot_result_sha256': profile.sha(args.snapshot/'result.json'),
              'analyzer_sha256': profile.sha(Path(profile.__file__)),
              'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(results), 'scope': 'Screening and snapshot replay only'}))


if __name__ == '__main__':
    main()
