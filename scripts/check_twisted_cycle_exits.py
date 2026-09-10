"""Verify actual C++ cycle-exit observations and independent worst-cell flux sums.

Requires a completed native run whose full physical fields have already passed
the existing native checker. This audit does not establish spatial convergence.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--native-check',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-cycle',action='store_true')
    args=parser.parse_args()
    root=args.run.resolve()
    if args.output.exists():raise ValueError('Preserve previous cycle audits')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    hashes={str(args.native_check.resolve()):sha(args.native_check)}
    native=json.loads(args.native_check.read_bytes())
    if not native['passed'] or str(root/'run_completion.json') not in native['source_sha256']:
        raise ValueError('Require a passed complete native check for this actual run')
    hashes.update(native['source_sha256']);hashes.update(native['executed_input_sha256'])
    for path,digest in hashes.items():
        if sha(Path(path))!=digest:raise ValueError('Previously checked native input changed')
    method=json.loads((root/'projection_method.json').read_bytes())
    if method.get('pressure_roundoff_cycle_exit','disabled')=='disabled' or method.get('pressure_iterate_storage')!='twofold':
        raise ValueError('The actual native method does not enable the tested cycle rule')
    hashes[str(root/'projection_method.json')]=sha(root/'projection_method.json')
    floor=args.output.with_name(args.output.stem+'_floor.json')
    subprocess.run([sys.executable,str(Path(__file__).with_name('check_twisted_projection_floor.py')),
                    '--run',str(root),'--output',str(floor)],check=True,stdout=subprocess.DEVNULL)
    diagnostic=json.loads(floor.read_bytes())
    if not diagnostic['diagnostic_consistent']:raise ValueError('Independent face-sum check failed')
    hashes[str(floor.resolve())]=sha(floor)
    captured=Path(diagnostic['input_snapshot_directory'])
    for name,digest in diagnostic['input_snapshot_sha256'].items():
        if sha(root/name)!=digest or sha(captured/name)!=digest:
            raise ValueError('Floor audit does not cover this complete trace')
        hashes[str(root/name)]=digest;hashes[str(captured/name)]=digest

    def table(name):
        path=root/name
        data=path.read_bytes()
        if not data.endswith(b'\n'):raise ValueError('Incomplete final trace line')
        hashes[str(path)]=sha(path)
        with path.open() as stream:rows=list(csv.DictReader(stream))
        for row in rows:
            if None in row or any(v is None or not math.isfinite(float(v)) for v in row.values()):
                raise ValueError('Malformed or nonfinite cycle trace')
        return rows

    updates={(int(r['call']),int(r['pass'])):r for r in table('projection_updates.csv')}
    timing={(int(r['call']),int(r['pass'])):r for r in table('projection_pressure.csv')}
    episodes={(r['call'],r['pass']):r for r in diagnostic['stagnation_episodes']}
    records=table('projection_cycle_exits.csv')
    groups={}
    for r in records:
        key=int(r['call']),int(r['exit_pass'])
        groups.setdefault(key,[]).append(r)
    checks=[]
    for (call,last),rows in groups.items():
        period=int(rows[0]['period'])
        if period not in (2,3,4) or len(rows)!=2*period:
            raise ValueError('Missing complete periods')
        if [int(r['pass']) for r in rows]!=list(range(last-2*period+1,last+1)):
            raise ValueError('Nonconsecutive cycle observations')
        values=[]
        for row in rows:
            if int(row['period'])!=period or row['time_step']!=rows[0]['time_step']:
                raise ValueError('Cycle metadata differs within its window')
            current=updates.get((call,int(row['pass'])))
            if current is None:raise ValueError('Cycle observation lacks an ordinary update')
            residual=float(row['divergence_linf']);epsilon=float(row['epsilon'])
            dv,sv,dp,sp=(float(row[k]) for k in ('flux_velocity_change_linf','flux_velocity_scale','impulse_change_upper_linf','impulse_scale'))
            if not (epsilon==sys.float_info.epsilon and 1e-10<=residual<=1e-8 and sv>0 and sp>0 and
                    0<=dv<=.5*epsilon*sv and 0<=dp<=.5*epsilon*sp):
                raise ValueError('Recorded update does not satisfy the tighter cycle bounds')
            if (residual!=float(current['divergence_linf']) or float(row['time_step'])!=float(current['time_step']) or
                    dv!=float(current['intended_flux_velocity_change_linf']) or
                    dp<float(current['intended_impulse_change_linf'])):
                raise ValueError('Cycle observations disagree with the physical update trace')
            before=float(current['flux_velocity_scale'])
            if abs(sv-before)>dv+4*epsilon*max(sv,before):
                raise ValueError('Post-update velocity scale is inconsistent with the update')
            values.append(residual)
        if values[:period]!=values[period:] or len(set(values))<2:
            raise ValueError('No two complete nonconstant periods')
        if (call,last) not in timing or any(c==call and p>last for c,p in timing):
            raise ValueError('Cycle exit did not end this pressure call')
        compensated=float(rows[-1]['exit_compensated_divergence_linf'])
        if any(float(r['exit_compensated_divergence_linf'])!=compensated for r in rows):
            raise ValueError('Different final compensated values')
        if not 0<=compensated<=1e-8 or compensated!=float(updates[(call,last)]['compensated_divergence_linf']):
            raise ValueError('Compensated actual divergence exceeds its unchanged bound')
        incident=episodes.get((call,last))
        if incident is None or not math.isclose(abs(incident['divergence_after']),compensated,rel_tol=1e-12,abs_tol=0):
            raise ValueError('Independent shared-face sum does not certify the exited worst cell')
        checks.append({'call':call,'exit_pass':last,'period':period,'compensated_divergence_linf':compensated,
                       'worst_cell':incident['cell'],'worst_cell_volume':incident['volume'],'passed':True})
    if args.require_cycle and not checks:raise ValueError('This run did not exercise a cycle exit')
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise ValueError('Cycle audit input changed')
    result={'passed':True,'scope':__doc__,'native_fields_checked':True,'cycle_exits':checks,
            'cycle_count':len(checks),'flow_convergence_claimed':False,'source_sha256':hashes,
            'checker_sha256':sha(Path(__file__)),'goal_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'cycle_count':len(checks),'native_fields_checked':True}))


if __name__=='__main__':main()
