"""Profile a completed-call window captured from one identified live Aphros process.

This is a read-only resource observation with concurrent workloads. It neither
claims physical-step completion nor validates accuracy or exclusive performance.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import statistics
import psutil
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--creation-time',type=float,required=True)
    p.add_argument('--calls',type=int,default=200);a=p.parse_args()
    if a.calls<1:raise ValueError('Positive completed-call window required')
    run=a.run.resolve();out=a.output.resolve();process=psutil.Process(a.pid)
    runtime=json.loads((run/'run_manifest.json').read_bytes())
    if abs(process.create_time()-a.creation_time)>1e-5 or Path(process.exe()).resolve()!=Path(runtime['executable']).resolve():
        raise ValueError('Live reference process identity differs')
    if Path(process.cwd()).resolve()!=run or process.cmdline()[-1]!='a.conf':
        raise ValueError('Live reference work directory or arguments differ')
    inputs={str(run/'run_manifest.json'):sha(run/'run_manifest.json'),
            runtime['executable']:runtime['executable_sha256'],str(run/'a.conf'):runtime['config_sha256']}
    if any(sha(Path(path))!=digest for path,digest in inputs.items()):raise ValueError('Reference executable or configuration changed')
    out.mkdir(parents=True,exist_ok=False)
    data=(run/'linear_memory.csv').read_bytes();data=data[:data.rfind(b'\n')+1]
    marker=data.rfind(b',snapshot-complete,')
    if marker<0:raise ValueError('No completed linear calls')
    end=data.find(b'\n',marker)+1;complete=data[:end]
    trace=out/'linear_memory_prefix.csv';trace.write_bytes(complete)
    calls=[];active=None;previous_tick=-1;event=0
    for row in csv.DictReader(io.StringIO(complete.decode())):
        event+=1;tick=int(row['tick_milliseconds']);stage=row['stage']
        if int(row['event'])!=event or tick<previous_tick:raise ValueError('Trace event order differs')
        previous_tick=tick
        if stage=='entry':
            if active is not None:raise ValueError('Nested linear trace')
            active={'system':row['system'],'ticks':{},'event':event}
        if active is None or row['system']!=active['system'] or stage in active['ticks']:
            raise ValueError('Unmatched or repeated stage')
        active['ticks'][stage]=tick
        if stage=='snapshot-complete':
            t=active['ticks']
            if not {'entry','before-solve','after-solve','refinement-complete','original-rows-checked','snapshot-complete'}<=t.keys():
                raise ValueError('Missing measured phase')
            active['phases_ms']={'setup':t['before-solve']-t['entry'],
                'first_solve':t['after-solve']-t['before-solve'],
                'refinement':t['refinement-complete']-t['after-solve'],
                'original_rows':t['original-rows-checked']-t['refinement-complete'],
                'snapshot':t['snapshot-complete']-t['original-rows-checked']}
            calls.append(active);active=None
    if active or not calls:raise ValueError('Unclosed trace prefix')
    window=calls[-a.calls:];groups=defaultdict(list)
    for row in window:groups[row['system']].append(row)
    def summarize(rows):
        phases={name:[row['phases_ms'][name] for row in rows] for name in rows[0]['phases_ms']}
        total=sum(sum(values) for values in phases.values())
        return {'calls':len(rows),'backend_total_ms':total,'phases':{name:{'sum_ms':sum(values),
            'median_ms':statistics.median(values),'maximum_ms':max(values),
            'fraction':sum(values)/total if total else 0} for name,values in phases.items()}}
    if not (run/'linear_memory.csv').read_bytes().startswith(complete):raise ValueError('Captured trace prefix changed')
    if any(sha(Path(path))!=digest for path,digest in inputs.items()):raise ValueError('Observed inputs changed')
    inputs[str(trace)]=sha(trace)
    result={'scope':__doc__,'observed_utc':datetime.now(timezone.utc).isoformat(),
        'process':{'pid':a.pid,'created':process.create_time(),'executable':process.exe(),'confirmed_live':True},
        'total_completed_calls_in_prefix':len(calls),'window_calls':len(window),
        'first_window_event':window[0]['event'],'window_elapsed_ms':window[-1]['ticks']['snapshot-complete']-window[0]['ticks']['entry'],
        'all_systems':summarize(window),'by_system':{name:summarize(rows) for name,rows in groups.items()},
        'source_sha256':inputs,'checker_sha256':sha(Path(__file__)),'goal_complete':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('total_completed_calls_in_prefix','window_calls','all_systems','by_system')}))


if __name__=='__main__':main()
