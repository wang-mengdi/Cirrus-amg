"""Exercise actual completed-prefix validation with damaged state and provenance."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from run_twisted_solver import sha
from twisted_prefix_restart import load_checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.checkpoint.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    original,_=load_checkpoint(source/'checkpoint.json')
    def clone(name):
        root=out/name;shutil.copytree(source,root)
        meta=json.loads((root/'checkpoint.json').read_text());proof=json.loads((root/'parent_prefix_check.json').read_text())
        for obj in (meta,proof):
            obj['source_sha256']={str(root/Path(k).relative_to(source)) if Path(k).is_relative_to(source) else k:v for k,v in obj['source_sha256'].items()}
        return root,meta,proof
    def write(root,meta,proof):
        (root/'parent_prefix_check.json').write_text(json.dumps(proof,indent=2)+'\n')
        meta['parent_check_sha256']=sha(root/'parent_prefix_check.json')
        meta['state_sha256']=sha(root/'state.bin')
        (root/'checkpoint.json').write_text(json.dumps(meta,indent=2)+'\n')
    root,meta,proof=clone('valid_control');write(root,meta,proof);load_checkpoint(root/'checkpoint.json')
    cases=[]
    def test(name,mutate,config_change=None):
        root,meta,proof=clone(name);config=meta['parent_config'].copy();config['time_steps']=meta['physical_step']+1
        mutate(root,meta,proof)
        if config_change:config_change(config)
        write(root,meta,proof)
        try:load_checkpoint(root/'checkpoint.json',config)
        except (ValueError,KeyError) as e:cases.append({'name':name,'rejected':True,'reason':str(e)});return
        except FileNotFoundError as e:
            if name!='removed_write_barrier' or Path(e.filename)!=Path(meta['parent_run'])/'transient_summary.json':raise
            cases.append({'name':name,'rejected':True,'reason':'Removed later-step barrier; the interrupted parent has no complete-run summary'});return
        cases.append({'name':name,'rejected':False})
    noop=lambda *_:None
    for name,key,value in (('changed_viscosity','nu',.02),('changed_dt','time_step',.002),
                           ('weaker_linear_tolerance','linear_tolerance',1e-12),('unknown_orthogonalization','gpu_orthogonalization','unknown'),
                           ('no_remaining_step','time_steps',original['physical_step'])):
        test(name,noop,lambda c,k=key,v=value:c.__setitem__(k,v))
    test('wrong_absolute_time',lambda r,m,p:m.__setitem__('physical_time',m['physical_time']+.001))
    test('wrong_prefix_history',lambda r,m,p:m['time_history'][0].__setitem__('inner_iterations',9999))
    test('missing_field_step',lambda r,m,p:m['field_output_steps'].pop(0))
    test('missing_prefix_file',lambda r,m,p:m['prefix_files'].pop(next(iter(m['prefix_files']))))
    test('false_parent_proof',lambda r,m,p:p.__setitem__('passed',False))
    def corrupt_state(root,mode):
        data=np.fromfile(root/'state.bin',dtype='<f8')
        if mode=='nan':data[0]=np.nan
        elif mode=='value':data[0]+=1e-4
        else:data=data[:-1]
        data.tofile(root/'state.bin')
    for mode in ('nan','value','truncated'):test('state_'+mode,lambda r,m,p,mode=mode:corrupt_state(r,mode))
    def remove_barrier(root,meta,proof):
        path=root/'parent_run_prefix.log';data=path.read_bytes();marker=b'projection step=5 iter=1'
        if marker not in data:raise ValueError('Expected the live four-step fixture')
        path.write_bytes(data[:data.index(marker)])
        for obj in (meta,proof):obj['source_sha256'][str(path)]=sha(path)
    test('removed_write_barrier',remove_barrier)
    # Changing this implementation choice is permitted explicitly by format v2;
    # the live parent/tail flow comparison checks the resulting numerical fields.
    allowed=original['parent_config'].copy();allowed['gpu_orthogonalization']='cgs2';allowed['time_steps']=original['physical_step']+1
    load_checkpoint(source/'checkpoint.json',allowed)
    result={'passed':all(c['rejected'] for c in cases),'scope':__doc__,'valid_relocated_control_passed':True,
        'known_orthogonalization_change_allowed':True,'cases':cases,'rejected_cases':sum(c['rejected'] for c in cases),
        'source_sha256':{str(source/'checkpoint.json'):sha(source/'checkpoint.json'),str(Path(__file__).resolve()):sha(Path(__file__))},
        'goal_complete':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
