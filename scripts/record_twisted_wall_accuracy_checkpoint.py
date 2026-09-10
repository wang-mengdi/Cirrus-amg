"""Preserve the manufactured wall-closure audit and its verified physical-field replay."""
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
    files, retained = [], []

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
            raise ValueError('Input changed while archiving')
        files.append({'path': target.relative_to(out).as_posix(), 'source': str(path), 'source_sha256': value,
                      'sha256': sha(target), 'size': target.stat().st_size, 'gzip': compress, 'scope': scope})

    root = runs/'wall_accuracy32_64_128_v2'
    report = json.loads((root/'wall_accuracy.json').read_text())
    if not report['diagnostic_replay_passed'] or report['goal_complete']:
        raise ValueError('Expected a completed diagnostic, not a goal-completion claim')
    if [g['ny'] for g in report['grids']] != [32, 64, 128]:
        raise ValueError('Unexpected audited grids')
    for source, value in report['source_sha256'].items():
        path = Path(source)
        if sha(path) != value:
            raise ValueError('Audited source changed: '+source)
        retained.append({'source': str(path), 'source_sha256': value, 'size': path.stat().st_size,
                         'scope': 'actual completed physical field used for replay, executed input or audit producer'})
    for path in root.iterdir():
        if path.is_file():
            compress = path.suffix == '.csv'
            save(path, 'audit/'+path.name+('.gz' if compress else ''), 'manufactured operator/geometry error, not constant-force flow accuracy', compress)
    plot = runs/'wall_accuracy_plot_v3'
    manifest = json.loads((plot/'plot_manifest.json').read_text())
    for source, value in manifest['source_sha256'].items():
        if sha(Path(source)) != value:
            raise ValueError('Plot source changed')
    for name, value in manifest['output_sha256'].items():
        if sha(plot/name) != value:
            raise ValueError('Rendered plot changed')
    for path in plot.iterdir():
        if path.is_file():
            save(path, 'plot/'+path.name, 'visually inspected scientific plot of the manufactured wall audit')
    for name in ('audit_twisted_wall_accuracy.py', 'plot_twisted_wall_accuracy.py', 'record_twisted_wall_accuracy_checkpoint.py'):
        save(repo/'scripts'/name, 'sources/'+name, 'actual diagnostic or preservation source')
    save(repo/'simple/EmbeddedOperators.cpp', 'sources/EmbeddedOperators.cpp.txt', 'inspected current production wall definition; executed historical binaries identified separately')
    save(Path('D:/Dropbox/Agent-simulation/twisted-baseline/aphros/src/solver/approx_eb.ipp'), 'sources/aphros_approx_eb.ipp.txt', 'read-only upstream wall closure and its stated first-order accuracy')
    save(repo/'validation/twisted/WALL_CLOSURE_ACCURACY.md', 'scope.md', 'interpretation, limitations and outstanding physical acceptance')
    save(runs/'anderson_failure_build_v1/build_manifest.json', 'build/build_manifest.json', 'current native build and actual build behind the completed 128 two-step run')
    receipt = {'scope': __doc__, 'diagnostic_replay_passed': True, 'goal_complete': False,
               'files': files, 'retained_runtime_files': retained}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'archive_files': len(files), 'retained_files': len(retained), 'goal_complete': False}))


if __name__ == '__main__':
    main()
