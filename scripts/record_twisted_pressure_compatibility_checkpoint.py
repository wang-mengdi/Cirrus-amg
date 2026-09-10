"""Archive actual pressure equations, compatibility experiments and flow checks."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runs = repo/'output/twisted'
    baseline = Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    target = args.output.resolve()
    names = ('navier_stokes_proj_n16_pressure_capture_v1',
             'navier_stokes_proj_n32_pressure_capture_steps2_v1',
             'navier_stokes_proj_n16_volume_pressure_default_v1',
             'navier_stokes_proj_n32_volume_pressure_steps2_v1')
    for name in names:
        if json.loads((baseline/name/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code'] != 0:
            raise ValueError('Incomplete reference: '+name)
    target.mkdir(parents=True, exist_ok=False)
    receipt = []

    def save(source, destination, scope):
        raw = source.read_bytes()
        compress = len(raw) > 2_000_000 and source.suffix in ('.csv', '.json', '.log')
        data = gzip.compress(raw, mtime=0) if compress else raw
        name = str(destination)+('.gz' if compress else '')
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError('Duplicate target: '+name)
        path.write_bytes(data)
        receipt.append({'path': Path(name).as_posix(), 'source': str(source.resolve()),
                        'source_sha256': hashlib.sha256(raw).hexdigest(),
                        'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data),
                        'gzip': compress, 'scope': scope})

    def tree(root, destination, scope):
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.csv', '.json', '.log', '.txt', '.py', '.ps1', '.cmd', '.h', '.ipp', '.cpp', '.conf'):
                save(p, Path(destination)/p.relative_to(root), scope)

    for name in names:
        tree(baseline/name, Path('aphros')/name, 'completed configured original Proj run; inspect separate mass and field checks')
    for name in ('projection_snapshot_build_v1', 'projection_volume_build_v1'):
        tree(baseline/name, Path('builds')/name, 'immutable validation driver and linear backend sources; original library unchanged')
    reports = ('aphros_pressure_capture_n16_regression_v1.json',
               'aphros_pressure_original_discretization_v1.json',
               'aphros_pressure_capture_n32_regression_v1.json',
               'aphros_pressure_snapshot_n16_v1.json',
               'aphros_pressure_snapshot_n16_v2.json',
               'aphros_pressure_snapshot_n32_steps2_v1.json',
               'aphros_pressure_compatibility_probe_n32_v1.json',
               'aphros_volume_pressure_default_n16_regression_v1.json',
               'aphros_volume_pressure_n32_step1_diagnostic_v1.json',
               'aphros_volume_pressure_n32_steps2_diagnostic_v1.json',
               'aphros_volume_pressure_n32_steps2_pair_v1.json')
    for name in reports:
        save(runs/name, Path('reports')/name, 'diagnostic or completed comparison; failures retained and not relabeled')
    tree(runs/'aphros_proj32_volume_pressure_step1_v1', 'aphros/volume_pressure_step1',
         'verified first completed physical step of the two-step run; pressure trace is a separately scoped prefix')
    tree(runs/'aphros_proj32_pressure_tol2e15_before_volume_v1', 'aphros/failed_pressure_tolerance_step35',
         'verified completed step 35 of an interrupted reference; actual mass failed')
    interrupted = baseline/'navier_stokes_proj_n32_steady_pressure_tol2e15_v1'
    if json.loads((interrupted/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code'] == 0:
        raise ValueError('Expected the documented interrupted pressure-tolerance experiment')
    for name in ('a.conf', 'case_manifest.json', 'run_manifest.json', 'run_completion.json',
                 'run_interruption.json', 'run.log', 'tube_b0_time.csv'):
        save(interrupted/name, Path('interrupted')/name, 'superseded unweighted compatibility run; no completed steady-flow claim')
    for version in (1, 2):
        name = f'aphros_pressure_snapshot_checker_v{version}.py'
        save(runs/name, Path('checkers')/name, 'exact checker bytes for earlier captured reports')
    for name in ('check_aphros_pressure_snapshot.py', 'probe_aphros_pressure_compatibility.py',
                 'check_twisted_mass.py', 'compare_twisted.py', 'check_twisted_projection_flux.py',
                 'snapshot_twisted_reference.py', 'run_twisted_baseline.ps1',
                 'record_twisted_pressure_compatibility_checkpoint.py'):
        save(repo/'scripts'/name, Path('checkers')/name, 'reproducibility source')
    for name in ('LICENSE.aphros', 'LICENSE.amgcl', 'prepare_twisted_linear_cache.py', 'build_twisted_driver.ps1'):
        save(repo/'validation/aphros'/name, Path('reproduce')/name, 'licenses and independent driver build tools')
    native = runs/'ours_proj32_steady_v1'
    for name in ('run_manifest.json', 'run_completion.json', 'case.json'):
        save(native/name, Path('native')/name, 'completed native run provenance; only compared step 2 fields included here')
    for p in sorted((native/'step_0002').iterdir()):
        if p.is_file() and p.suffix in ('.csv', '.json'):
            save(p, Path('native/step_0002')/p.name, 'native step 2 compared against independently restarted reference')
    for pending_name in ('navier_stokes_proj_n64_volume_pressure_amg_v1',
                         'navier_stokes_proj_n32_volume_pressure_default_steps2_v1'):
        pending = baseline/pending_name
        for name in ('a.conf', 'case_manifest.json', 'run_manifest.json'):
            save(pending/name, Path('pending')/pending.name/name,
                 'immutable launch inputs only; no completed result claimed from these files')
    receipt_data = {'scope': __doc__, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'goal_complete': False,
                    'remaining': 'Complete steady adaptive GPU/reference agreement and physical grid/time convergence; this checkpoint does not claim them.',
                    'files': receipt}
    (target/'receipt.json').write_text(json.dumps(receipt_data, indent=2)+'\n')
    print(json.dumps({'files': len(receipt), 'bytes': sum(p['size'] for p in receipt), 'target': str(target)}))


if __name__ == '__main__':
    main()
