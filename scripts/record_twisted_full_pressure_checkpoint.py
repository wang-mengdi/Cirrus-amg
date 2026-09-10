"""Archive full GPU pressure validation, retaining failed independent mass checks.

Large manufactured 128-level vectors stay in the runtime directory with hashes;
their generating source, exact configuration, geometry and executable identities
are recorded. Flow evidence includes complete 16/64 runs, not a steady-flow claim.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runs = repo/'output/twisted'
    baseline = Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    target = args.output.resolve()
    native_names = ('ours_proj16_full_pressure_default_v1', 'ours_proj64_full_pressure_gpu_v1')
    reference_names = ('navier_stokes_proj_n32_volume_pressure_default_steps2_v1',
                       'navier_stokes_proj_n64_volume_pressure_amg_v1')
    for root in [runs/n for n in native_names]+[baseline/n for n in reference_names]:
        if json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))['exit_code'] != 0:
            raise ValueError('Incomplete run: '+str(root))
    target.mkdir(parents=True, exist_ok=False)
    receipt, runtime = [], []

    def retain(source, scope):
        runtime.append({'source': str(source.resolve()), 'source_sha256': sha(source),
                        'size': source.stat().st_size, 'scope': scope})

    def save(source, destination, scope):
        before = sha(source)
        compress = source.stat().st_size > 2_000_000 and source.suffix in ('.csv', '.json', '.log')
        path = target/(str(destination)+('.gz' if compress else ''))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as output, source.open('rb') as original:
            if compress:
                with gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as zipped:
                    shutil.copyfileobj(original, zipped, 1024*1024)
            else:
                shutil.copyfileobj(original, output, 1024*1024)
        if sha(source) != before:
            raise ValueError('Source changed while archiving: '+str(source))
        receipt.append({'path': path.relative_to(target).as_posix(), 'source': str(source.resolve()),
                        'source_sha256': before, 'sha256': sha(path), 'size': path.stat().st_size,
                        'gzip': compress, 'scope': scope})

    def tree(root, destination, scope, only_metadata=False):
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            if p.suffix in ('.exe', '.vtu', '.vtp') or (only_metadata and p.suffix == '.csv'):
                retain(p, scope+'; retained runtime artifact')
            elif p.suffix in ('.csv', '.json', '.log', '.txt', '.py', '.ps1', '.cmd', '.h', '.ipp', '.cpp', '.conf'):
                save(p, Path(destination)/p.relative_to(root), scope)

    tree(runs/'full_pressure_gpu_build_v1', 'build', 'immutable native executable build and exact compiled sources')
    for n in (16, 64, 128):
        root = runs/f'full_pressure_operator{n}_v1'
        if not json.loads((root/'audit.json').read_text())['passed']:
            raise ValueError('Failed operator audit')
        tree(root, Path('operators')/str(n), 'Ax and manufactured linear systems only', only_metadata=n == 128)
    for name in native_names:
        tree(runs/name, Path('native')/name, 'complete configured time sequence; physical steady/grid convergence separate')
    for name in reference_names:
        tree(baseline/name, Path('aphros')/name, 'original Proj/Embed with volume-weighted linear compatibility; n64 actual mass fails')
    for name in ('full_pressure_default_n16_cpu_pair_v1.json', 'full_pressure_n64_cpu_pair_v1.json',
                 'full_pressure_n64_iteration1_pair_v1.json', 'full_pressure_n64_work_comparison_v1.json',
                 'full_pressure_n64_work_comparison_v2.json', 'pressure_roundoff_n64_cpu_pair_v2.json',
                 'full_pressure_n64_volume_aphros_pair_v1.json', 'pressure_roundoff_n64_volume_aphros_pair_v1.json',
                 'aphros_volume_pressure_n64_diagnostic_v1.json',
                 'aphros_volume_pressure_default_n32_steps2_pair_v1.json',
                 'aphros_volume_pressure_default_n32_steps2_diagnostic_v1.json'):
        save(runs/name, Path('reports')/name, 'unchanged actual comparison or diagnostic, including failures')
    for name in ('config_ours_proj16_full_pressure_default_v1.json', 'config_ours_proj16_full_pressure_gpu_v1.json',
                 'config_ours_proj64_full_pressure_gpu_v1.json', 'config_ours_proj128_full_pressure_gpu_v1.json',
                 'config_full_pressure_gpu_audit64_v1.json', 'config_full_pressure_gpu_audit128_v1.json'):
        save(runs/name, Path('configs')/name, 'exact tested or pending configuration')
    old = runs/'ours_proj128_pressure_roundoff_v2'
    if json.loads((old/'run_completion.json').read_text())['exit_code'] == 0:
        raise ValueError('Expected documented superseded run')
    for name in ('run_manifest.json', 'run_completion.json', 'run_interruption.json', 'case.json', 'run.log',
                 'projection_method.json', 'gpu_linear.csv', 'projection_updates.csv',
                 'projection_floor_faces.csv', 'projection_roundoff_exits.csv', 'step_0001/acceleration.csv'):
        save(old/name, Path('interrupted_native128')/name, 'stopped after 22 outer iterations; no completed physical step')
    for p in sorted((old/'step_0001').rglob('*.csv')):
        if p.parent.name.startswith('iter_'):
            retain(p, 'immutable raw prefix of superseded 128-level run; retained on disk')
    for name in ('case.json', 'run_manifest.json'):
        save(runs/'ours_proj128_full_pressure_gpu_v1'/name, Path('pending_native128')/name,
             'immutable launched full-pressure configuration; no flow completion claim')
    for name in ('a.conf', 'case_manifest.json', 'run_manifest.json'):
        save(baseline/'navier_stokes_proj_n32_steady_volume_pressure_v1'/name,
             Path('pending_aphros32_steady')/name, 'immutable steady-run inputs only')
    for name in ('check_twisted_projection_iteration.py', 'check_twisted_time_pair.py',
                 'check_aphros_pressure_snapshot.py', 'check_twisted_mass.py', 'compare_twisted.py',
                 'check_twisted_projection_flux.py', 'run_twisted_solver.py', 'run_twisted_baseline.ps1',
                 'export_twisted_paraview.py', 'check_twisted_paraview_step.py',
                 'record_twisted_full_pressure_checkpoint.py'):
        save(repo/'scripts'/name, Path('checkers')/name, 'reproduction source')
    for name in ('paraview_export.json', 'paraview_step_readback.json'):
        save(runs/'ours_proj32_steady_v1/step_0002'/name, Path('native32_step2_visualization')/name,
             'actual ParaView readback of already archived native step 2')
    for name in ('LICENSE.aphros', 'LICENSE.amgcl'):
        save(repo/'validation/aphros'/name, Path('licenses')/name, 'upstream license')
    result = {'scope': __doc__, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'goal_complete': False,
              'remaining': 'Independent steady adaptive GPU/reference agreement, actual mass and physical grid/time convergence remain required.',
              'files': receipt, 'retained_runtime_files': runtime}
    (target/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'target': str(target), 'files': len(receipt), 'bytes': sum(v['size'] for v in receipt),
                      'retained_runtime_files': len(runtime)}), flush=True)


if __name__ == '__main__':
    main()
