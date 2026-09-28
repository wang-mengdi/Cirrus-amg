"""Pack, verify or restore the frozen Windows experiment inputs (never run a solver)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path('D:/CirrusExperiments/cirrus-amg/handoff/inputs.json')
ROOTS = ('C:/Code/Cirrus-amg/', 'D:/CirrusExperiments/cirrus-amg/',
         'D:/Dropbox/Agent-simulation/twisted-baseline/')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def destination(row):
    name = row['path'].replace('\\', '/')
    if not any(name.lower().startswith(root.lower()) for root in ROOTS):
        raise ValueError('Destination outside the three experiment roots: ' + name)
    if '..' in name.split('/') or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
        raise ValueError('Invalid manifest entry')
    target = Path(name).resolve()
    if not any(target.is_relative_to(Path(root).resolve()) for root in ROOTS):
        raise ValueError('Resolved destination escaped the experiment roots')
    return target


def check_file(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']


def check_git_file(path, row, commit):
    if not path.is_relative_to(REPO):
        raise ValueError('Git source outside the repository: ' + str(path))
    relative = path.relative_to(REPO).as_posix()
    result = subprocess.run(
        ['git', '-C', str(REPO), 'show', f'{commit}:{relative}'],
        capture_output=True,
    )
    return (result.returncode == 0 and len(result.stdout) == row['bytes']
            and hashlib.sha256(result.stdout).hexdigest() == row['sha256'])


def transfer(source, target, expected):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(4 * 1024**2), b''):
        target.write(block)
        digest.update(block)
        size += len(block)
    if digest.hexdigest() != expected['sha256'] or size != expected['bytes']:
        raise ValueError('Content checksum/size differs: ' + expected['path'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('pack', 'verify', 'restore', 'verify-local'))
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--git-commit', help='Verify Git-tracked sources in this exact HEAD commit instead of the working tree')
    a = p.parse_args()
    if a.mode != 'verify-local' and a.bundle is None:
        p.error('--bundle is required')
    if a.git_commit:
        if not re.fullmatch(r'[0-9a-f]{40}', a.git_commit):
            p.error('--git-commit requires a full 40-character commit SHA')
        head = subprocess.run(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
            capture_output=True, text=True,
        )
        if head.returncode != 0 or head.stdout.strip() != a.git_commit:
            raise ValueError('Current Git HEAD differs from --git-commit')
    raw = a.manifest.read_bytes()
    manifest = json.loads(raw)
    rows = manifest['files']
    # Validate all destinations before writing anything. Existing files must match.
    targets = [destination(row) for row in rows]
    if len({str(x).lower() for x in targets}) != len(targets):
        raise ValueError('Duplicate destination')
    def source_matches(row, target):
        if row['git'] and a.git_commit:
            return check_git_file(target, row, a.git_commit)
        return check_file(target, row)
    if a.mode == 'verify-local':
        failures = [str(t) for r, t in zip(rows, targets) if not source_matches(r, t)]
        print(json.dumps({'passed': not failures, 'checked': len(rows), 'missing_or_changed': failures,
                          'git_commit': a.git_commit}), flush=True)
        raise SystemExit(bool(failures))
    if a.mode == 'pack':
        a.bundle.parent.mkdir(parents=True, exist_ok=True)
        written = set()
        with zipfile.ZipFile(a.bundle, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            z.writestr('inputs.json', raw)
            for i, (row, target) in enumerate(zip(rows, targets)):
                if row['git'] or row['sha256'] in written:
                    continue
                with target.open('rb') as src, z.open('objects/' + row['sha256'], 'w', force_zip64=True) as dst:
                    transfer(src, dst, row)
                written.add(row['sha256'])
                if row['bytes'] > 64 * 1024**2 or i % 100 == 0:
                    print(json.dumps({'packed': i + 1, 'total': len(rows), 'path': str(target)}), flush=True)
        print(json.dumps({'bundle': str(a.bundle), 'bytes': a.bundle.stat().st_size,
                          'sha256': sha(a.bundle), 'unique_objects': len(written)}), flush=True)
        return
    with zipfile.ZipFile(a.bundle) as z:
        if z.read('inputs.json') != raw:
            raise ValueError('Bundle manifest differs from the checked-out manifest')
        checked = set()
        for i, (row, target) in enumerate(zip(rows, targets)):
            if row['git']:
                if not source_matches(row, target):
                    raise ValueError('git pull/checkout is missing the required source: ' + str(target))
                continue
            key = 'objects/' + row['sha256']
            if a.mode == 'verify':
                if key not in checked:
                    digest = hashlib.sha256()
                    size = 0
                    with z.open(key) as src:
                        for block in iter(lambda: src.read(4 * 1024**2), b''):
                            digest.update(block)
                            size += len(block)
                    if digest.hexdigest() != row['sha256'] or size != row['bytes']:
                        raise ValueError('Corrupt object: ' + key)
                    checked.add(key)
            elif target.exists():
                if not check_file(target, row):
                    raise ValueError('Refusing to replace an existing different file: ' + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.reproduction-partial')
                with z.open(key) as src, tmp.open('xb') as dst:
                    transfer(src, dst, row)
                # Windows rename refuses replacement if another writer created target.
                tmp.rename(target)
            if row['bytes'] > 256 * 1024**2:
                print(json.dumps({'checked': i + 1, 'total': len(rows)}), flush=True)
    print(json.dumps({'passed': True, 'mode': a.mode, 'files': len(rows),
                      'git_commit': a.git_commit}), flush=True)


if __name__ == '__main__':
    main()
