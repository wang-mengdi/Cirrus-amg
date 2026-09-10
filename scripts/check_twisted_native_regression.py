"""Require identical fields and inner histories for complete native GPU regression runs."""
import argparse
import csv
import json
from pathlib import Path
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();roots=[args.reference.resolve(),args.candidate.resolve()]
    if args.output.exists():raise ValueError('Preserve existing regression evidence')
    configs=[json.loads((p/'case.json').read_text()) for p in roots]
    if [{k:v for k,v in c.items() if k!='output'} for c in configs][0]!={k:v for k,v in configs[1].items() if k!='output'}:
        raise ValueError('Regression requires identical numerical and output settings')
    count=configs[0]['time_steps'];hashes={};checks=[]
    def track(p):
        value=sha(p);hashes[str(p)]=value;return value
    for root,config in zip(roots,configs):
        run=json.loads((root/'run_manifest.json').read_text());completion=json.loads((root/'run_completion.json').read_text())
        summary=json.loads((root/'transient_summary.json').read_text())
        if completion['exit_code'] or not all(completion[k] for k in ('executable_unchanged','config_unchanged','geometry_unchanged')):
            raise ValueError('Run did not finish with unchanged inputs')
        if sha(Path(run['executable']))!=run['executable_sha256']:raise ValueError('Executed binary changed')
        if not summary['converged'] or summary['steps_completed']!=count:raise ValueError('Incomplete trajectory')
        if config.get('linear_backend')!='native_gpu':raise ValueError('Expected GPU regression run')
        with (root/'gpu_linear.csv').open() as f:trace=list(csv.DictReader(f))
        if not trace or any(not 0<=float(r['true_relative_residual'])<=config['linear_tolerance'] for r in trace):
            raise ValueError('GPU solve failed the unchanged linear tolerance')
        if list(root.glob('linear_failure_*')):raise ValueError('This regression expects the ordinary path without linear failures')
        for name in ('case.json','run_manifest.json','run_completion.json','transient_summary.json','gpu_linear.csv'):
            track(root/name)
    for step in range(1,count+1):
        folders=[p/f'step_{step:04d}' for p in roots]
        metrics=[json.loads((p/'metrics.json').read_text()) for p in folders]
        for p,m in zip(folders,metrics):
            if not m['converged'] or m['complete_inner_fixed_point_residual']>=configs[0]['tolerance']:
                raise ValueError('A physical step did not reach its full fixed point')
            track(p/'metrics.json')
        files=['solution.csv','walls.csv','flux.csv','sections.csv','history.csv','acceleration.csv']
        if metrics[0]['iterations']!=metrics[1]['iterations']:raise ValueError('Inner iteration count changed')
        files += [f'iter_{metrics[0]["iterations"]}/{name}.csv' for name in ('cells','faces')]
        for name in files:
            values=[track(p/name) for p in folders]
            checks.append({'step':step,'file':name,'bitwise_equal':values[0]==values[1]})
    passed=all(c['bitwise_equal'] for c in checks)
    result={'passed':passed,'scope':__doc__+' Physical accuracy and independent Aphros alignment remain separate.',
            'complete_steps':count,'checks':checks,'source_sha256':hashes,'checker_sha256':sha(Path(__file__))}
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('Regression sources changed during inspection')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':passed,'complete_steps':count,'bitwise_field_and_history_pairs':len(checks)}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
