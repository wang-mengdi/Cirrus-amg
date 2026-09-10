"""Exercise chain validation against actual copied native histories and damaged checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from twisted_chain_restart import load_checkpoint, check_prefix, complete_lines, hash_session, sha as checked_sha
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.checkpoint.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    original,inputs=load_checkpoint(source/'checkpoint.json')
    if original['ancestry_depth']<1:raise ValueError('Require an actual resumed parent')
    shutil.copyfile(__file__,out/'executed_checker.py.txt')
    results=[]

    def rejected(name,action):
        try:action()
        except (ValueError,KeyError) as e:
            results.append({'case':name,'rejected':True,'reason':str(e)})
        else:results.append({'case':name,'rejected':False})

    # Relocation must remain valid before testing altered checkpoint metadata.
    def clone(name):
        root=out/name;shutil.copytree(source,root)
        meta=json.loads((root/'checkpoint.json').read_text())
        proof=json.loads((root/'parent_prefix_check.json').read_text())
        for obj in (meta,proof):
            obj['source_sha256']={str(root/Path(k).relative_to(source)) if Path(k).is_relative_to(source) else k:v
                                  for k,v in obj['source_sha256'].items()}
        return root,meta,proof

    def write(root,meta,proof):
        (root/'parent_prefix_check.json').write_text(json.dumps(proof,indent=2)+'\n')
        meta['parent_check_sha256']=sha(root/'parent_prefix_check.json')
        meta['state_sha256']=sha(root/'state.bin')
        (root/'checkpoint.json').write_text(json.dumps(meta,indent=2)+'\n')

    root,meta,proof=clone('relocated_control');write(root,meta,proof)
    load_checkpoint(root/'checkpoint.json')

    def test(name,change,config_change=lambda c:None):
        root,meta,proof=clone(name)
        cfg=meta['parent_config'].copy();cfg['time_steps']=meta['physical_step']+1
        change(root,meta,proof);config_change(cfg);write(root,meta,proof)
        rejected(name,lambda:load_checkpoint(root/'checkpoint.json',cfg))

    noop=lambda *_:None
    for name,key,value in [('changed_dt','time_step',.002),('changed_viscosity','nu',.02),
                           ('weaker_linear_tolerance','linear_tolerance',1e-12),
                           ('no_remaining_tail','time_steps',original['physical_step'])]:
        test(name,noop,lambda c,k=key,v=value:c.__setitem__(k,v))
    test('false_depth',lambda r,m,p:(m.__setitem__('ancestry_depth',99),p.__setitem__('ancestry_depth',99)))
    test('omitted_ancestor_state',lambda r,m,p:m['source_sha256'].pop(next(k for k in m['source_sha256'] if k.endswith('state.bin'))))
    test('forged_ancestor_input',lambda r,m,p:p['ancestry']['input_sha256'].__setitem__(next(iter(p['ancestry']['input_sha256'])),'0'*64))
    test('missing_prefix_file',lambda r,m,p:m['prefix_files'].pop(next(iter(m['prefix_files']))))
    def corrupt(root,meta,proof):
        state=root/'state.bin';data=bytearray(state.read_bytes());data[0]^=1;state.write_bytes(data)
    test('state_changed_with_updated_hash',corrupt)

    # These test actual resumed-parent evidence, rather than only its stored proof.
    parent=Path(original['parent_run'])
    def copied_parent(name,change,selected=None):
        root=out/name/'parent';shutil.copytree(parent,root)
        evidence=out/name/'evidence';evidence.mkdir()
        for original_name,captured in [('run.log','parent_run_prefix.log'),('time_history.csv','parent_time_prefix.csv'),('gpu_linear.csv','parent_gpu_prefix.csv')]:
            (evidence/captured).write_bytes(complete_lines(root/original_name))
        change(root)
        rejected(name,lambda:check_prefix(root,selected or original['physical_step'],evidence))

    def mutate_json(path,key,value):
        j=json.loads(path.read_text());j[key]=value;path.write_text(json.dumps(j,indent=2)+'\n')
    def damage_loaded(root):
        p=root/'restart_loaded.bin';data=bytearray(p.read_bytes());data[0]^=1;p.write_bytes(data)
    copied_parent('changed_actual_restored_state',damage_loaded)
    copied_parent('omitted_executed_ancestor',lambda r:mutate_json(r/'run_manifest.json','restart_input_sha256',{}))
    copied_parent('changed_copied_prefix',lambda r:(r/'step_0001/flux.csv').write_bytes((r/'step_0001/flux.csv').read_bytes()+b'\n'))
    def change_history(root):
        path=root/'time_history.csv'
        with path.open() as f:rows=list(csv.DictReader(f))
        rows[0]['temporal_acceleration_relative_l2']='123.0'
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (root.parent/'evidence/parent_time_prefix.csv').write_bytes(complete_lines(path))
    copied_parent('changed_inherited_time_history',change_history)
    ancestor=json.loads(Path(original['parent_config']['restart_checkpoint']).read_text())
    copied_parent('no_physical_advance',noop,ancestor['physical_step'])
    # The recursion guard is exercised on an actual, valid checkpoint path.
    rejected('ancestry_cycle',lambda:load_checkpoint(source/'checkpoint.json',_seen=frozenset({source/'checkpoint.json'})))
    root,meta,proof=clone('cyclic_parent_configuration')
    copied=root/'parent';shutil.copytree(parent,copied)
    config=json.loads((copied/'case.json').read_text());config['restart_checkpoint']=str(root/'checkpoint.json')
    (copied/'case.json').write_text(json.dumps(config,indent=2)+'\n')
    meta['parent_run']=str(copied);meta['parent_config']=config
    write(root,meta,proof)
    rejected('cyclic_parent_configuration',lambda:load_checkpoint(root/'checkpoint.json'))
    def changed_cached_file():
        path=out/'cache_input.bin';path.write_bytes(b'original')
        with hash_session():
            checked_sha(path);path.write_bytes(b'changed length')
            checked_sha(path)
    rejected('changed_cached_input',changed_cached_file)
    def changed_after_last_lookup():
        path=out/'cache_final_input.bin';path.write_bytes(b'original')
        with hash_session():
            checked_sha(path);path.write_bytes(b'changed length')
    rejected('changed_after_last_hash_lookup',changed_after_last_lookup)
    if any(sha(Path(path))!=digest for path,digest in inputs.items()):raise ValueError('Original test inputs changed')
    result={'passed':all(r['rejected'] for r in results),'scope':__doc__,'relocated_control_passed':True,
            'rejections':results,'count':len(results),'source_sha256':inputs,'checker_sha256':sha(Path(__file__)),
            'goal_complete':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'cases':len(results)}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
