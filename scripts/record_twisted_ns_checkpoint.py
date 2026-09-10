"""Archive completed NS evidence while retaining failed accuracy/halo audits."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo/'output/twisted'
    target = repo/'validation/twisted/results/navier_stokes_checkpoint'
    target.mkdir(parents=True, exist_ok=True)
    if (target/'checkpoint.json').exists():
        raise ValueError('Preserve the existing historical checkpoint; choose a new snapshot for later changes')
    paths = {
        'transient_n16': ('compare_ns16_transient_v2.json', True),
        'steady_n16_direct': ('compare_ns16_steady_v2.json', True),
        'steady_n16_amg': ('compare_ns16_steady_amg_v2.json', True),
        'stokes_regression_n16': ('compare_stokes16_ns_v2.json', True),
        'advection_fixed': ('ns16_advection_fixed_v2.json', True),
        'advection_upstream_failure': ('ns16_advection_stale_v2.json', False),
        'linear_backend_pair': ('backend_ns16_amg_vs_direct.json', True),
        'failed_step_handling': ('time_failure_check.json', True),
        'straight_quick': ('straight_regression_ns_v2/suite_results.json', True),
        'paraview_time_readback': ('ours_ns16_steady_v2/paraview_time_readback.json', True),
        'uniform64_to_128_refinement': ('refinement_uniform64_128/refinement.json', False),
        'adaptive128_internal_comparison': ('ours_adaptive128_amg98p03_v2/internal_uniform_comparison.json', None)}
    exe = repo/'build/windows/x64/release/simple_channel.exe'
    executable_sha = sha(exe)
    evidence = {}
    for name, (relative, expected) in paths.items():
        path = source/relative
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if expected is not None and value['passed'] != expected:
            raise ValueError(f'Unexpected result: {path}')
        if name in ('straight_quick', 'failed_step_handling') and value['executable_sha256'] != executable_sha:
            raise ValueError('Validation binary differs from the checkpoint executable')
        if name == 'straight_quick' and not value['strict_current_binary_pass']:
            raise ValueError('Current-binary regression did not pass')
        (target/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
        evidence[name] = {'source': str(path), 'source_sha256': sha(path), 'expected_pass': expected}
    artifacts = {}
    for run_name in ('ours_ns16_transient_v2', 'ours_ns16_steady_v2', 'ours_stokes16_ns_v2'):
        run = source/run_name
        artifacts[run_name] = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    files = list((repo/'simple').glob('*')) + list((repo/'scripts').glob('*twisted*'))
    files += list((repo/'validation/aphros').glob('twisted*'))
    files += [repo/'validation/aphros/prepare_twisted.py', repo/'validation/aphros/build_baseline.ps1',
              repo/'validation/check_twisted_time_failure.py', repo/'validation/twisted/ns16_steady.json',
              repo/'validation/twisted/IMPLEMENTATION.md', repo/'xmake.lua']
    baseline = Path('D:/Dropbox/Agent-simulation/twisted-baseline/aphros')
    record = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'completed_goal': False,
        'scope': 'Embedded FOU/backward-Euler implementation and periodic halo repair; near-wall accuracy goal remains open',
        'parent_commit_at_capture': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'cirrus_executable_sha256': executable_sha,
        'source_sha256': {str(p.relative_to(repo)): sha(p) for p in sorted(set(files)) if p.is_file()},
        'aphros_revision': 'b60ce3da52c19935fa24c778f62f02141eaf7f80',
        'aphros_executable_sha256': {name: sha(baseline/'src'/name) for name in ('main_amg_v5.exe', 'main_amg_v8.exe', 'main_amg_v9.exe')},
        'aphros_modifications': ['prescribed geometry and read-only diagnostics', 'explicit uninitialized wall-flux fix',
            'explicit volume-flux halo exchange using existing CommFieldFace',
            'optional direct/AMG linear solvers checked against every original equation row'],
        'evidence': evidence, 'cirrus_artifact_sha256': artifacts,
        'remaining': ['finish and compare ongoing n32 NS time sequences', 'validate inertia on actual wall-refined octree interfaces',
                      'finish independent n128 Stokes reference and compare uniform/adaptive fields',
                      'resolve near-wall grid convergence and wall-sampling sensitivity']}
    (target/'checkpoint.json').write_text(json.dumps(record, indent=2)+'\n')
    print(target/'checkpoint.json')


if __name__ == '__main__':
    main()
