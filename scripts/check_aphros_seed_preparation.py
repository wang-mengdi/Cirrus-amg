"""Check real velocity-seed preservation, explicit steady-source scope and fine-grid file layout.

These are initial-condition preparation checks, not new Aphros flow results.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
from run_twisted_solver import sha


def seed(root):
    info=json.loads((root/'seed_manifest.json').read_text())
    path=root/'velocity.bin'
    if sha(path)!=info['initial_velocity_sha256']:raise ValueError('Seed bytes differ from manifest')
    with path.open('rb') as stream:magic,*header=struct.unpack('<8s3Q3d',stream.read(56))
    if magic!=b'TWVEL01\0' or header[:3]!=info['shape'] or header[3:]!=info['spacing']:
        raise ValueError('Seed header does not describe the target uniform mesh')
    n=int(np.prod(info['shape']))
    if path.stat().st_size!=56+25*n:raise ValueError('Wrong dense seed size')
    mask=np.memmap(path,mode='r',dtype=np.uint8,offset=56,shape=(n,))
    values=np.memmap(path,mode='r',dtype='<f8',offset=56+n,shape=(n,3))
    if not np.isin(mask,[0,1]).all() or np.count_nonzero(mask)!=info['fluid_cells']:
        raise ValueError('Wrong included-cell mask')
    if not np.isfinite(values).all() or np.any(values[mask==0]!=0):raise ValueError('Invalid seed values or nonzero excluded cells')
    if (not info['only_velocity_initialized'] or info['steady_alignment_proven'] or info['goal_complete']):
        raise ValueError('Initial guess incorrectly claims a solved reference')
    return info,mask,values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','batched','steady','fine','geometry-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep validation on D')
    out.mkdir(parents=True,exist_ok=False)
    old,om,ov=seed(a.previous);batch,bm,bv=seed(a.batched);steady,sm,sv=seed(a.steady);fine,fm,fv=seed(a.fine)
    assert old['initial_velocity_sha256']==batch['initial_velocity_sha256']
    assert batch['interpolation_batch_size']==128 and batch['interpolation_batches']==32
    assert steady['native_source_iteration_mode']=='steady_pseudo_iteration'
    proof_path=Path(steady['native_steady_source_check']['file'])
    assert sha(proof_path)==steady['native_steady_source_check']['sha256']
    proof=json.loads(proof_path.read_text());assert proof['passed'] and not proof['physical_trajectory_claimed']
    assert np.array_equal(om,sm)
    active=om>0
    difference=float(np.linalg.norm(sv[active]-ov[active])/np.linalg.norm(ov[active]))
    assert difference<1e-6
    assert fine['shape']==[256,128,128] and fine['fluid_cells']==1076936
    assert fine['interpolation_batches']>1 and fine['interpolation_batch_size']==4096
    cases=[];script=Path(__file__).with_name('prepare_aphros_initial_velocity.py')
    base=[sys.executable,str(script),'--source',steady['source_run'],'--kind','native',
          '--target-case',steady['target_case'],'--geometry-run',str(a.geometry_run)]
    for name,extra,expected in (
        ('pseudo_without_explicit_mode',[],'explicit steady-iteration'),
        ('invalid_zero_batch',['--native-steady-iteration','--interpolation-batch-size','0'],'batch size'),
        ('invalid_large_batch',['--native-steady-iteration','--interpolation-batch-size','65537'],'batch size'),
        ('pseudo_flag_on_aphros_kind',['--native-steady-iteration','--kind','aphros'],'requires --kind native')):
        target=out/name
        command=[*base,'--output',str(target),*extra]
        result=subprocess.run(command,capture_output=True,text=True)
        assert result.returncode and expected in result.stderr and not (target/'velocity.bin').exists(),(name,result.stderr)
        (out/(name+'.log')).write_text(result.stdout+result.stderr)
        cases.append({'name':name,'passed':True,'exit_code':result.returncode})
    paths=[Path(__file__),script,Path(__file__).with_name('analyze_twisted_refinement.py'),proof_path]
    paths.extend(root/name for root in (a.previous,a.batched,a.steady,a.fine) for name in ('seed_manifest.json','velocity.bin'))
    result={'passed':True,'scope':__doc__,'ordinary64_seed_bitwise_unchanged':True,
            'steady64_source_check_passed':True,'ordinary_to_pseudo_seed_relative_l2':difference,
            'fine_seed_target_shape':fine['shape'],'fine_seed_fluid_cells':fine['fluid_cells'],
            'fine_seed_bytes':(a.fine/'velocity.bin').stat().st_size,
            'aphros_solution_claimed':False,'cases':cases,'source_sha256':{str(path.resolve()):sha(path) for path in paths}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','cases','scope')}))


if __name__=='__main__':main()
