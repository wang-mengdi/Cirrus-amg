"""Archive complete GPU viscosity checks and immutable ongoing-run inputs.

All archived runs must be terminal. Large operator vectors and ParaView files
remain available on disk with hashes. This is not a steady/grid accuracy claim.
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
    names = ('ours_proj16_full_viscosity_default_v1', 'ours_proj16_full_viscosity_gpu_v1',
             'ours_proj64_full_viscosity_gpu_v1')
    for name in names:
        complete = json.loads((runs/name/'run_completion.json').read_text())
        if complete['exit_code'] or not all(complete[k] for k in ('executable_unchanged', 'config_unchanged', 'geometry_unchanged')):
            raise ValueError('Incomplete or changed run: '+name)
    target.mkdir(parents=True, exist_ok=False)
    receipt, retained = [], []

    def save(source, destination, scope):
        before = sha(source)
        compress = source.stat().st_size > 2_000_000 and source.suffix in ('.csv', '.json', '.log')
        path = target/(str(destination)+('.gz' if compress else ''))
        path.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, path.open('xb') as dst:
            if compress:
                with gzip.GzipFile(filename='', fileobj=dst, mode='wb', mtime=0) as zipped:
                    shutil.copyfileobj(src, zipped, 1024*1024)
            else:
                shutil.copyfileobj(src, dst, 1024*1024)
        if sha(source) != before:
            raise ValueError('Source changed while archiving: '+str(source))
        receipt.append({'path': path.relative_to(target).as_posix(), 'source': str(source.resolve()),
                        'source_sha256': before, 'sha256': sha(path), 'size': path.stat().st_size,
                        'gzip': compress, 'scope': scope})

    def tree(root, destination, scope, operator=False):
        for p in sorted(root.rglob('*')):
            if not p.is_file():
                continue
            if p.suffix in ('.exe', '.vtu', '.vtp') or (operator and p.suffix == '.csv'):
                retained.append({'source': str(p.resolve()), 'source_sha256': sha(p),
                                 'size': p.stat().st_size, 'scope': scope+'; runtime artifact retained on disk'})
            elif p.suffix in ('.csv', '.json', '.log', '.txt', '.py', '.ps1', '.cmd', '.h', '.cpp', '.conf'):
                save(p, Path(destination)/p.relative_to(root), scope)

    tree(runs/'full_viscosity_gpu_build_v1', 'build', 'exact compiled sources and immutable executable identities')
    for n in (16, 64):
        tree(runs/f'full_viscosity_operator{n}_v1', Path('operators')/str(n),
             'all-face CPU reference versus native full viscosity Ax and manufactured solve', operator=True)
        for suffix in ('.launch.json', '.log'):
            save(runs/f'full_viscosity_operator{n}_v1{suffix}', Path('operators')/f'{n}{suffix}',
                 'audit launch manifest and complete process log')
    for name in names:
        tree(runs/name, Path('native')/name, 'complete configured flow sequence; steady and grid convergence remain separate')
    for name in ('full_viscosity_n16_cpu_pair_v1.json', 'full_viscosity_n64_cpu_pair_v1.json',
                 'full_viscosity_default_n16_cpu_pair_v1.json', 'full_viscosity_work_comparison_v1.json'):
        save(runs/name, Path('reports')/name, 'actual full-sequence comparison or work counts; no wall-clock speed claim')
    for name in ('config_ours_proj16_full_viscosity_default_v1.json', 'config_ours_proj16_full_viscosity_gpu_v1.json',
                 'config_ours_proj64_full_viscosity_gpu_v1.json', 'config_ours_proj64_steady_full_viscosity_gpu_v1.json',
                 'config_full_viscosity_gpu_audit16_v1.json', 'config_full_viscosity_gpu_audit64_v1.json'):
        save(runs/name, Path('configs')/name, 'exact tested or launched configuration')
    for name in ('case.json', 'run_manifest.json'):
        save(runs/'ours_proj64_steady_full_viscosity_gpu_v1'/name, Path('pending_native64_steady')/name,
             'immutable launched input only; no steady completion claim')
    for name in ('a.conf', 'case_manifest.json', 'run_manifest.json'):
        save(baseline/'navier_stokes_proj_n64_steady_volume_pressure_v1'/name, Path('pending_aphros64_steady')/name,
             'independent original Proj/Embed launch; no steady completion claim')
    for name in ('check_twisted_time_pair.py', 'compare_twisted.py', 'check_twisted_mass.py',
                 'check_twisted_projection_flux.py', 'run_twisted_solver.py', 'run_twisted_baseline.ps1',
                 'export_twisted_paraview.py', 'check_twisted_paraview_step.py',
                 'record_twisted_viscosity_checkpoint.py', 'make_twisted_baseline.py'):
        save(repo/'scripts'/name, Path('checkers')/name, 'reproduction source')
    original = baseline/'aphros/src/libaphros_static.lib'
    if sha(original) != 'b8f2bbf83c69678da30f99c79874de2989e9eef1b4a91a13f2b68a4382952732':
        raise ValueError('Original shared Aphros library changed')
    for name in ('LICENSE.aphros', 'LICENSE.amgcl'):
        save(repo/'validation/aphros'/name, Path('licenses')/name, 'upstream license')
    result = {'scope': __doc__, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'goal_complete': False, 'original_aphros_library_sha256': sha(original),
              'remaining': 'Independent steady adaptive agreement and physical grid/time convergence; complete GPU128 and reference runs.',
              'files': receipt, 'retained_runtime_files': retained}
    (target/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'files': len(receipt), 'bytes': sum(v['size'] for v in receipt),
                      'retained_runtime_files': len(retained)}), flush=True)


if __name__ == '__main__':
    main()
