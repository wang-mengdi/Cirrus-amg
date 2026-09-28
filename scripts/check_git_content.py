"""Reject generated data in the index or in commits added by this development branch."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess

REPO = Path(__file__).resolve().parents[1]
CODE = {'.c', '.cpp', '.h', '.cu', '.cuh', '.ipp', '.inc', '.py', '.ps1', '.sh', '.bat', '.lua', '.patch'}
EXACT = {'.gitignore', '.gitattributes', '.editorconfig', 'CMakeLists.txt', 'pyproject.toml', 'requirements.txt'}
GENERATED = ('validation/twisted/results/', 'validation/results/', 'validation/figures/')
CASES = {f'validation/twisted/reproduction/native{n}.json' for n in (16, 64, 128, 256)}
APHROS_CONFIGS = {'validation/aphros/sim_base.conf'}


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory='+REPO.as_posix(), *args], cwd=REPO)


def allowed(name):
    p = PurePosixPath(name)
    if name.startswith(GENERATED):
        return False
    if name.startswith('validation/twisted/reproduction/'):
        return name in CASES
    if name in APHROS_CONFIGS:
        return True
    if p.name in EXACT or p.name.startswith('LICENSE') or p.suffix in CODE | {'.md'}:
        return True
    return p.suffix == '.json' and p.parent.as_posix() in {'scenes', 'validation/twisted'}


def check(rows):
    errors = []
    for name, oid in rows:
        if not allowed(name):
            errors.append(name + ': data or unsupported file type')
            continue
        size = int(git('cat-file', '-s', oid))
        limit = 64*1024 if name.endswith(('.json', '.conf')) else 2*1024**2
        if size > limit:
            errors.append(name + ': exceeds source/configuration size limit')
            continue
        raw = git('cat-file', 'blob', oid)
        if b'\x00' in raw:
            errors.append(name + ': binary content')
        if name.endswith('.json'):
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict) or {'passed', 'source_sha256', 'tables', 'receipt'} & obj.keys():
                    errors.append(name + ': report/data manifest is not a solver configuration')
            except (ValueError, UnicodeError):
                errors.append(name + ': invalid configuration JSON')
    return errors


def tree(ref):
    rows = []
    for entry in git('ls-tree', '-rz', ref).split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        if kind != 'blob' or mode not in {'100644', '100755'}:
            raise ValueError('Unexpected non-file Git entry: ' + name.decode())
        rows.append((name.decode(), oid))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged', action='store_true')
    p.add_argument('--ref', default='HEAD')
    p.add_argument('--base', help='Additionally inspect every new commit after this existing base')
    a = p.parse_args()
    errors = []
    if a.staged:
        rows = []
        for entry in git('ls-files', '--stage', '-z').split(b'\0'):
            if entry:
                meta, name = entry.split(b'\t', 1)
                mode, oid, stage = meta.decode().split()
                if stage != '0':
                    raise ValueError('Resolve index conflicts before checking')
                rows.append((name.decode(), oid))
        errors += check(rows)
    else:
        refs = [a.ref]
        if a.base:
            refs += git('rev-list', a.base+'..'+a.ref).decode().splitlines()
        seen = set()
        for ref in refs:
            rows = tree(ref)
            fresh = [(n, h) for n, h in rows if (n, h) not in seen]
            seen.update(rows)
            errors += check(fresh)
    print(json.dumps({'passed': not errors, 'errors': errors}, ensure_ascii=True))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
