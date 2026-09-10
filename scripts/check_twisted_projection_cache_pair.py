"""Verify exact retained Aphros dumps and iteration history after factor caching.

The baseline library and driver must be identical. The isolated linear sources
are checked against the original worktree, and only the optional backend header
may differ. Retained dumps are checked byte for byte; this does not claim that
fields overwritten by the original driver were independently retained.
"""
import argparse
import datetime
import json
import re
from pathlib import Path
from run_twisted_solver import sha


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();roots=[args.reference.resolve(),args.candidate.resolve()]
    if args.output.exists():raise ValueError('Preserve the previous comparison')
    runtimes=[];builds=[];hashes={};seconds=[]
    for root in roots:
        runtime=read(root/'run_manifest.json');completion=read(root/'run_completion.json')
        executable=Path(runtime['executable']);build=read(executable.parent/'build_manifest.json')
        if completion['exit_code']!=0:raise ValueError('Incomplete reference run')
        if sha(executable)!=runtime['executable_sha256'] or sha(executable)!=build['executable_sha256']:
            raise ValueError('Executable differs from recorded build/run')
        if sha(Path(build['library']))!=build['library_sha256']:raise ValueError('Baseline library changed')
        if sha(root/'a.conf')!=runtime['config_sha256']:raise ValueError('Configuration changed')
        for path in [root/'a.conf',root/'case_manifest.json',root/'run_manifest.json',root/'run_completion.json',
                     executable.parent/'build_manifest.json',root/'run.log']:
            hashes[str(path)]=sha(path)
        seconds.append((datetime.datetime.fromisoformat(completion['completed_utc'].replace('Z','+00:00'))-
                        datetime.datetime.fromisoformat(runtime['started_utc'].replace('Z','+00:00'))).total_seconds())
        runtimes.append(runtime);builds.append(build)
    if runtimes[0]['config_sha256']!=runtimes[1]['config_sha256']:raise ValueError('Different prescribed configuration')
    if any(builds[0][key]!=builds[1][key] for key in ('source_sha256','library_sha256','flags')):
        raise ValueError('Driver, original library, or common compiler flags differ')
    allowed={'APHROS_TWISTED_FACTOR_CACHE','APHROS_TWISTED_FACTOR_TRACE'}
    envs=[{k:v for k,v in r['environment'].items() if k not in allowed} for r in runtimes]
    if envs[0]!=envs[1]:raise ValueError('Different runtime environment beyond cache/trace settings')
    extras=builds[1]['extra_sources']
    if len(extras)!=1 or Path(extras[0]['path']).name!='linear.cpp':raise ValueError('Unexpected override')
    folder=Path(extras[0]['path']).parent;manifest=read(folder/'override_manifest.json')
    original=Path(builds[1]['library']).parent/'linear'
    for name in ('linear.cpp','linear.ipp','linear.h'):
        if sha(folder/name)!=sha(original/name):raise ValueError('Original linear algorithm changed: '+name)
    for path,digest in manifest['generated'].items():
        if sha(Path(path))!=digest:raise ValueError('Override source changed')
        hashes[path]=digest
    hashes[str(folder/'override_manifest.json')]=sha(folder/'override_manifest.json')
    names=[{p.name for p in root.glob('*.csv')} for root in roots]
    if names[0]!=names[1] or not names[0]:raise ValueError('Different retained dump coverage')
    files=[]
    for name in sorted(names[0]):
        paths=[root/name for root in roots];digests=[sha(p) for p in paths]
        files.append({'file':name,'identical':digests[0]==digests[1]})
        for path,digest in zip(paths,digests):hashes[str(path)]=digest
    logs=[(root/'run.log').read_text(encoding='utf-8',errors='replace') for root in roots]
    iterations=[re.findall(r'\.\.\.\.\.iter=\d+, diff=[^\r\n]+',log) for log in logs]
    if not iterations[0]:raise ValueError('Missing physical-step iteration history')
    traces=[]
    for log in logs:
        rows=re.findall(r'factor-cache name=(\w+) hit=(\d+) entries=(\d+) hits=(\d+) misses=(\d+) rows=(\d+) nnz=(\d+)',log)
        traces.append({'calls':len(rows),'final':rows[-1] if rows else None})
    result={'passed':all(row['identical'] for row in files) and iterations[0]==iterations[1],
            'scope':__doc__,'retained_csv_files':files,'iteration_history_identical':iterations[0]==iterations[1],
            'iteration_records':len(iterations[0]),'factor_traces':traces,'elapsed_seconds_reference_candidate':seconds,
            'timing_limit':'Observed concurrent runs; elapsed times are not an isolated hardware benchmark.',
            'source_sha256':hashes}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','retained_csv_files','scope')}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
