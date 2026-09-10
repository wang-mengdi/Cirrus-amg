"""A truncated packed table must fail before any CFD solution is emitted."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import hashlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='truncated_geometry_', dir=args.output.parent.resolve()))
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    source = Path(config['embedded_geometry'])
    if not source.is_absolute():
        source = repo/source
    geometry = json.loads(source.read_text())
    if geometry['format'] != 'aphros_cut_geometry_v2':
        raise ValueError('Expected packed geometry')
    for table in geometry['tables'].values():
        shutil.copyfile(source.parent/table['file'], root/table['file'])
    binary = root/geometry['tables']['cells']['file']
    with binary.open('r+b') as stream:
        stream.truncate(binary.stat().st_size-1)
    (root/'geometry.json').write_text(json.dumps(geometry))
    config['embedded_geometry'] = str(root/'geometry.json')
    config['output'] = str(root/'solver')
    (root/'input.json').write_text(json.dumps(config))
    process = subprocess.run([str(args.exe.resolve()), str(root/'input.json')], cwd=repo,
                             capture_output=True, text=True, timeout=120)
    (root/'run.log').write_text(process.stdout+process.stderr)
    passed = process.returncode == 1 and 'Invalid packed cut-geometry header or file size' in process.stderr and not list(root.rglob('solution.csv'))
    report = {'passed': passed, 'exit_code': process.returncode,
              'scope': 'Actual C++ input reader rejects a truncated float64 table before solving',
              'executable_sha256': hashlib.sha256(args.exe.read_bytes()).hexdigest(), 'run': str(root)}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
