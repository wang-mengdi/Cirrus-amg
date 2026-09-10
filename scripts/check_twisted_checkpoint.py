"""Verify archived payloads, retained files, and optionally actual staged Git blobs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from run_twisted_solver import sha


def digest(stream,size=None):
    result=hashlib.sha256()
    while size is None or size:
        block=stream.read(1024*1024 if size is None else min(size,1024*1024))
        if not block:
            if size:raise ValueError('Truncated Git blob')
            break
        result.update(block)
        if size is not None:size-=len(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--staged',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve prior integrity checks')
    repo=Path(__file__).resolve().parents[1]
    archive=args.archive.resolve()
    receipt=json.loads((archive/'receipt.json').read_text())
    expected={}
    for row in receipt['files']:
        p=(archive/row['path']).resolve()
        if not p.is_relative_to(archive):raise ValueError('Archive path escapes its directory')
        if p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:
            raise ValueError('Archive bytes differ: '+str(p))
        with (gzip.open(p,'rb') if row['gzip'] else p.open('rb')) as stream:
            if digest(stream)!=row['source_sha256']:raise ValueError('Uncompressed payload differs: '+str(p))
        if sha(Path(row['source']))!=row['source_sha256']:raise ValueError('Retained source changed: '+row['source'])
        expected[p.relative_to(repo).as_posix()]=row['sha256']
    for row in receipt['retained_runtime_files']:
        p=Path(row['source'])
        if p.stat().st_size!=row['size'] or sha(p)!=row['source_sha256']:
            raise ValueError('Retained runtime file changed: '+str(p))
    expected[(archive/'receipt.json').relative_to(repo).as_posix()]=sha(archive/'receipt.json')
    if {p.relative_to(repo).as_posix() for p in archive.rglob('*') if p.is_file()}!=set(expected):
        raise ValueError('Unrecorded or missing archive files')
    build=json.loads((archive/'build/build_manifest.json').read_text())
    if build['exit_code'] or not build['source_unchanged']:raise ValueError('Build did not complete with unchanged source')
    for name,value in build['source_sha256'].items():
        if sha(repo/name)!=value:raise ValueError('Compiled source differs from current worktree: '+name)
    staged_count=0
    if args.staged:
        listing=subprocess.check_output(['git','ls-files','--stage','--',str(archive.relative_to(repo))],cwd=repo,text=True)
        entries={}
        for row in listing.splitlines():
            header,name=row.split('\t',1);_,oid,stage=header.split()
            if stage!='0' or name in entries:raise ValueError('Unexpected staged index entry')
            entries[name]=oid
        if set(entries)!=set(expected):raise ValueError('Staged archive file set differs')
        with subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as process:
            for name,oid in entries.items():
                process.stdin.write((oid+'\n').encode());process.stdin.flush()
                actual,kind,size=process.stdout.readline().decode().split()
                if actual!=oid or kind!='blob':raise ValueError('Unexpected Git object')
                if digest(process.stdout,int(size))!=expected[name]:raise ValueError('Actual staged Git blob differs: '+name)
                if process.stdout.read(1)!=b'\n':raise ValueError('Invalid Git batch delimiter')
                staged_count+=1
            process.stdin.close()
            if process.wait()!=0:raise ValueError('Git blob read failed')
    report={'passed':True,'archive':str(archive),'archive_data_files':len(receipt['files']),
            'retained_runtime_files':len(receipt['retained_runtime_files']),
            'compiled_source_paths':len(build['source_sha256']),'actual_staged_blobs_checked':staged_count,
            'receipt_sha256':sha(archive/'receipt.json'),'checker_sha256':sha(Path(__file__))}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
