"""Preserve the completed extended Aphros64 step and its independent native match."""
import argparse
import gzip
import json
import shutil
from pathlib import Path

from run_twisted_solver import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runs = repo/'output/twisted'
    base = Path('D:/Dropbox/Agent-simulation/twisted-baseline')
    reference = base/'navier_stokes_proj_n64_extended_cache_v7'
    launch = base/'navier_stokes_proj_n64_steady_extended_cache_v7'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files, retained = [], {}

    def save(path, destination, scope, compress=False):
        path = Path(path).resolve()
        target = out/destination
        target.parent.mkdir(parents=True, exist_ok=True)
        value = sha(path)
        if compress:
            with path.open('rb') as source, target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=6) as sink:
                shutil.copyfileobj(source, sink)
        else:
            shutil.copyfile(path, target)
        if sha(path) != value:
            raise ValueError('Archive source changed')
        files.append({'path': target.relative_to(out).as_posix(), 'source': str(path), 'source_sha256': value,
                      'sha256': sha(target), 'size': target.stat().st_size, 'gzip': compress, 'scope': scope})

    def collect(node):
        if isinstance(node, dict):
            for field, content in node.items():
                if field == 'source_sha256':
                    for source, value in content.items():
                        path = Path(source).resolve()
                        if str(path) in retained:
                            if retained[str(path)]['source_sha256'] != value:
                                raise ValueError('Conflicting source hashes')
                            continue
                        if sha(path) != value:
                            raise ValueError('Verified input changed: '+source)
                        retained[str(path)] = {'source': str(path), 'source_sha256': value, 'size': path.stat().st_size,
                                               'scope': 'actual completed comparison, exact stored flux, geometry or full build provenance'}
                else:
                    collect(content)
        elif isinstance(node, list):
            for content in node:
                collect(content)

    for name in ('aphros_n64_extended_cache_exact_mass_v1.json', 'aphros_n64_extended_cache_native_pair_v1.json'):
        path = runs/name
        report = json.loads(path.read_text())
        if not report['passed']:
            raise ValueError('Completed scoped check failed')
        collect(report)
        save(path, 'checks/'+name, 'completed transient independent comparison; not a steady-state claim')
    for name in ('a.conf', 'case_manifest.json', 'run_manifest.json', 'run_completion.json', 'run.log',
                 'tube_b0_time.csv', 'proj_final_b0_exact.json', 'proj_final_b0_exact_cells.csv', 'proj_final_b0_exact_faces.csv'):
        compress = name.endswith('.csv')
        save(reference/name, 'reference64/'+name+('.gz' if compress else ''), 'actual completed reference and exact stored flow', compress)
    a, b = [json.loads((root/'run_manifest.json').read_text(encoding='utf-8-sig')) for root in (reference, launch)]
    for key in ('executable_sha256', 'geometry_state_sha256', 'environment'):
        if a[key] != b[key]:
            raise ValueError('Steady continuation changed reference solver settings')
    if (launch/'a.conf').read_text().replace('set double tmax 0.64\n', 'set double tmax 0.005\n') != (reference/'a.conf').read_text():
        raise ValueError('Steady reference changed more than the final physical time')
    for name in ('a.conf', 'case_manifest.json', 'run_manifest.json'):
        save(launch/name, 'steady_reference_launch/'+name, 'launch only; no completed steady trajectory claim')
    save(runs/'anderson_failure_build_v1/build_manifest.json', 'build/build_manifest.json', 'current native build; compared native run records its own earlier immutable build')
    for name in ('check_aphros_exact_mass.py', 'compare_twisted.py', 'validate_aphros_extended_reference.py',
                 'make_twisted_baseline.py', 'run_twisted_baseline.ps1', 'record_twisted_reference64_completion.py'):
        save(repo/'scripts'/name, 'sources/'+name, 'actual checking, launching or preservation source')
    save(repo/'validation/twisted/REFERENCE64_EXTENDED_COMPLETION.md', 'scope.md', 'completed physical-step findings and outstanding work')
    receipt = {'scope': __doc__, 'independent_transient_match_passed': True, 'goal_complete': False,
               'files': files, 'retained_runtime_files': list(retained.values())}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'archive_files': len(files), 'retained_files': len(retained), 'goal_complete': False}))


if __name__ == '__main__':
    main()
