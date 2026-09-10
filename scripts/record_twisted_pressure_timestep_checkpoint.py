"""Archive the pressure time-step diagnostic and its actual completed inputs."""
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
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files, retained = [], {}

    def keep(path, scope):
        path = Path(path).resolve()
        retained[str(path)] = {'source': str(path), 'source_sha256': sha(path), 'size': path.stat().st_size, 'scope': scope}

    def save(path, destination, scope, compress=False):
        path = Path(path).resolve()
        target = out/destination
        target.parent.mkdir(parents=True, exist_ok=True)
        value = sha(path)
        if compress:
            with path.open('rb') as source, target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as sink:
                shutil.copyfileobj(source, sink)
        else:
            shutil.copyfile(path, target)
        if sha(path) != value:
            raise ValueError('Archive input changed')
        files.append({'path': target.relative_to(out).as_posix(), 'source': str(path), 'source_sha256': value,
                      'sha256': sha(target), 'size': target.stat().st_size, 'gzip': compress, 'scope': scope})

    root = runs/'pressure_timestep_diagnostic16_v2'
    report = json.loads((root/'diagnostic.json').read_text())
    if not report['diagnostic_replay_passed'] or report['pressure_time_step_gate']['passed'] or report['goal_complete']:
        raise ValueError('Expected successful replay and explicitly failed pressure time-step acceptance')
    for source, value in report['source_sha256'].items():
        path = Path(source)
        if sha(path) != value:
            raise ValueError('Diagnostic source changed: '+source)
        if path.suffix == '.exe':
            keep(path, 'actual immutable historical executable; hash is not a claim this used the current native build')
        else:
            relative = path.relative_to(repo).as_posix()
            save(path, 'inputs/'+relative+'.gz', 'actual diagnostic input: completed trajectory, field, captured operator or producer', True)
    for path in root.iterdir():
        if path.is_file():
            save(path, 'diagnostic/'+path.name, 'completed pressure diagnostic; temporal pressure gate FAILED')
    plot = runs/'pressure_timestep_plot16_v2'
    manifest = json.loads((plot/'plot_manifest.json').read_text())
    for source, value in manifest['source_sha256'].items():
        if sha(Path(source)) != value:
            raise ValueError('Plot source changed')
    for name, value in manifest['output_sha256'].items():
        if sha(plot/name) != value:
            raise ValueError('Plot changed')
    for path in plot.iterdir():
        if path.is_file():
            save(path, 'plot/'+path.name, 'scientific plot, visually inspected; not a convergence certificate')
    for name in ('plot_twisted_pressure_timestep.py', 'record_twisted_pressure_timestep_checkpoint.py'):
        save(repo/'scripts'/name, 'sources/'+name, 'actual plot/archive producer')
    save(repo/'validation/twisted/PRESSURE_TIMESTEP_DIAGNOSTIC.md', 'scope.md', 'physical interpretation and limitations')
    aphros = Path('D:/Dropbox/Agent-simulation/twisted-baseline/aphros/src/solver/proj.ipp')
    save(aphros, 'sources/aphros_proj.ipp.txt', 'read-only original projection source supporting the flux identity')
    save(runs/'anderson_failure_build_v1/build_manifest.json', 'build/build_manifest.json',
         'current native solver build for ongoing main runs; historical diagnostic executables are identified separately')
    receipt = {'scope': __doc__, 'pressure_time_step_gate_passed': False, 'goal_complete': False,
               'files': files, 'retained_runtime_files': list(retained.values())}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'archive_files': len(files), 'retained_files': len(retained), 'pressure_time_step_gate_passed': False}))


if __name__ == '__main__':
    main()
