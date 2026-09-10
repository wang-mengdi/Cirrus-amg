"""Check that corrupted copies of real Aphros snapshots cannot pass step validation.

Only explicitly labelled negative-test copies are mutated. Solver outputs and
the actual live reference are never changed.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
from run_twisted_solver import sha
from snapshot_aphros_extended_step import validate_capture


def check(source, output):
    source=source.resolve();output=output.resolve()
    if output.exists():raise ValueError('Preserve previous rejection evidence')
    original,_=validate_capture(source)
    if not original['source_run_complete_proven']:raise ValueError('Use a real terminal snapshot for these corruptions')
    output.mkdir(parents=True)
    tests=[]

    def reject(name, mutate, rehash=()):
        root=output/name
        shutil.copytree(source,root)
        audit=json.loads((root/'extended_step_checkpoint.json').read_text())
        mutate(root,audit)
        for filename in rehash:audit['snapshot_sha256'][filename]=sha(root/filename)
        (root/'extended_step_checkpoint.json').write_text(json.dumps(audit,indent=2)+'\n')
        try:validate_capture(root)
        except (ValueError,KeyError,TypeError,FileNotFoundError) as e:
            tests.append({'name':name,'rejected':True,'reason':str(e)})
        else:raise AssertionError('Accepted a corrupted snapshot: '+name)

    def replace(root,name,transform):
        p=root/name;p.write_text(transform(p.read_text()))

    reject('field_bytes',lambda r,a:(r/'proj_final_b0_cells.csv').open('a').write('\n'))
    reject('producer_bytes',lambda r,a:(r/'producer.py.txt').open('a').write('\n'))
    reject('wrong_step',lambda r,a:a.update(completed_step=a['completed_step']-1))
    reject('wrong_configured_trajectory',lambda r,a:a.update(configured_steps=a['configured_steps']+1))
    reject('wrong_physical_time',lambda r,a:a.update(physical_time=a['physical_time']+.001))
    reject('fake_complete_reference',lambda r,a:a.update(complete_reference_run=True))
    reject('fake_new_run',lambda r,a:a.update(new_solver_run_claimed=True))
    reject('missing_terminal_evidence',lambda r,a:a['snapshot_sha256'].pop('parent_run_completion.json'))
    reject('source_hash_changed',lambda r,a:a['source_input_sha256'].update({next(iter(a['source_input_sha256'])):'0'*64}))
    reject('missing_history_row',lambda r,a:replace(r,'tube_b0_time.csv',lambda s:'\n'.join(s.splitlines()[:-1])+'\n'),('tube_b0_time.csv',))
    reject('missing_end_marker',lambda r,a:replace(r,'run.log',lambda s:s.replace('End of simulation: original Aphros Proj','Removed by negative test')),('run.log',))
    reject('missing_iteration',lambda r,a:replace(r,'run.log',lambda s:re.sub(r'iter=2, diff=[^\s]+','Removed by negative test',s,count=1)),('run.log',))
    reject('nonfinite_iteration',lambda r,a:replace(r,'run.log',lambda s:re.sub(r'(iter=1, diff=)[^\s]+',r'\g<1>nan',s,count=1)),('run.log',))
    def bad_time(root,audit):
        p=root/'proj_final_b0_exact.json';meta=json.loads(p.read_text());meta['physical_time_hex']='0x0p+0'
        p.write_text(json.dumps(meta)+'\n')
    reject('wrong_exact_field_time',bad_time,('proj_final_b0_exact.json',))
    if validate_capture(source)[0]!=original:raise ValueError('Original snapshot changed during the test')
    result={'passed':True,'scope':__doc__,'tests':tests,'original_snapshot':str(source),
            'original_snapshot_sha256':sha(source/'extended_step_checkpoint.json'),
            'checker_sha256':sha(Path(__file__)),'validator_sha256':sha(Path(__file__).with_name('snapshot_aphros_extended_step.py')),
            'goal_complete':False}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check(args.source,args.output)
    print(json.dumps({'passed':result['passed'],'rejected':len(result['tests'])}))
