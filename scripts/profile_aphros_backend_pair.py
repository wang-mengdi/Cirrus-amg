"""Measure recorded phases of completed AMG/direct runs on the same case and executable.

These are observed timings with concurrent jobs and coarse Windows tick counters,
not exclusive hardware benchmarks. This report does not validate field accuracy.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import re
from run_twisted_solver import sha

def profile(root,hashes):
    def load(name):
        path=root/name;hashes[str(path)]=sha(path)
        return json.loads(path.read_text(encoding='utf-8-sig'))
    runtime=load('run_manifest.json');done=load('run_completion.json');resources=load('cold_run_completion.json')
    if done['exit_code'] or not resources['passed'] or resources['stopped_for_sustained_low_memory']:
        raise ValueError('Require successful full runs, without a resource stop')
    for path,h in ((Path(runtime['executable']),runtime['executable_sha256']),
                   (root/'a.conf',runtime['config_sha256']),
                   (Path(runtime['geometry_state']),runtime['geometry_state_sha256'])):
        if sha(path)!=h:raise ValueError('Executed input changed')
        hashes[str(path)]=h
    trace_path=root/'linear_memory.csv';hashes[str(trace_path)]=sha(trace_path)
    phases=defaultdict(list);systems=Counter();active=None;event=0;previous_tick=-1
    with trace_path.open() as stream:
        for row in csv.DictReader(stream):
            event+=1;tick=int(row['tick_milliseconds']);stage=row['stage']
            if int(row['event'])!=event or tick<previous_tick:raise ValueError('Trace order changed')
            previous_tick=tick
            if stage=='entry':
                if active is not None:raise ValueError('Unfinished linear call')
                active={'entry':tick,'system':row['system']}
            if active is None or row['system']!=active['system']:raise ValueError('Unmatched trace event')
            if stage in ('before-solve','after-solve','snapshot-complete'):
                if stage in active:raise ValueError('Duplicate phase event')
                active[stage]=tick
            if stage=='snapshot-complete':
                if not {'before-solve','after-solve'}<=active.keys():raise ValueError('Incomplete solve trace')
                phases['setup_ms'].append(active['before-solve']-active['entry'])
                phases['first_solve_ms'].append(active['after-solve']-active['before-solve'])
                phases['refinement_checks_output_ms'].append(tick-active['after-solve'])
                phases['total_linear_backend_ms'].append(tick-active['entry'])
                systems[active['system']]+=1;active=None
    if active is not None or not systems:raise ValueError('Incomplete backend history')
    log=root/'run.log';hashes[str(log)]=sha(log)
    iterations=re.findall(r'^\.{5}iter=(\d+), diff=([^\r\n]+)',log.read_text(),re.M)
    def stats(values):
        values=sorted(values)
        return {'count':len(values),'sum_ms':sum(values),'mean_ms':sum(values)/len(values),
            'median_ms':values[len(values)//2],'minimum_ms':values[0],'maximum_ms':values[-1]}
    return runtime,{'root':str(root),'linear_calls_by_system':dict(systems),'outer_iterations':len(iterations),
        'phases':{name:stats(values) for name,values in phases.items()},
        'wall_seconds':resources['wall_seconds'],'peak_working_set_bytes':resources['peak_working_set_bytes'],
        'maximum_observed_private_bytes':resources['maximum_observed_private_bytes'],
        'minimum_available_bytes':resources['minimum_available_bytes']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--amg',type=Path,required=True);p.add_argument('--direct',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous timing reports')
    hashes={str(Path(__file__).resolve()):sha(Path(__file__))};runs=[];runtimes=[]
    for root in (a.amg.resolve(),a.direct.resolve()):
        runtime,run=profile(root,hashes);runtimes.append(runtime);runs.append(run)
    if runtimes[0]['environment'].get('APHROS_TWISTED_AMG')!='1' or runtimes[1]['environment'].get('APHROS_TWISTED_AMG') is not None:
        raise ValueError('Require explicit AMG and direct selections')
    clean=lambda env:{k:v for k,v in env.items() if k!='APHROS_TWISTED_AMG'}
    if clean(runtimes[0]['environment'])!=clean(runtimes[1]['environment']):raise ValueError('Other controls differ')
    for key in ('executable_sha256','config_sha256','geometry_state_sha256'):
        if runtimes[0][key]!=runtimes[1][key]:raise ValueError('Different executed '+key)
    if any(sha(Path(path))!=h for path,h in hashes.items()):raise ValueError('Timing inputs changed')
    result={'scope':__doc__,'complete_run_inputs_verified':True,'runs':runs,
        'observed_amg_over_direct_wall_ratio':runs[0]['wall_seconds']/runs[1]['wall_seconds'],
        'source_sha256':hashes,'goal_complete':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))

if __name__=='__main__':main()
