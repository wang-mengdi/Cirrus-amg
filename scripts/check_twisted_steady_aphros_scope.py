"""Check direct steady/reference comparison against actual control fields and reject wrong input scopes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from check_twisted_native_run import columns
from compare_twisted import error
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('direct','ordinary-pair','native','ordinary','aphros','snapshot','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.drive.lower()!='d:':raise ValueError('Keep regression outputs on D')
    out.mkdir(parents=True,exist_ok=False)
    direct=json.loads(a.direct.read_text());ordinary=json.loads(a.ordinary_pair.read_text())
    assert direct['passed'] and ordinary['passed'] and direct['native_iteration_mode']=='steady_pseudo_iteration'
    assert direct['complete_native_steady_run_checked'] and not direct['native_physical_trajectory_claimed']
    assert direct['complete_reference_run_checked'] and not direct['spatial_convergence_checked']
    for proof in (direct,ordinary):
        for path,digest in proof['source_sha256'].items():assert sha(Path(path))==digest,path
    assert direct['limits']==ordinary['limits']
    for name in ('mesh_cells.csv','mesh_faces.csv'):
        assert sha(a.native/name)==sha(a.ordinary/name),name
    native,control=[columns(root/'solution.csv',('id','volume','u','v','w','p')) for root in (a.native,a.ordinary)]
    nw,cw=[columns(root/'walls.csv',('face_id','owner','area','tau_x','tau_y','tau_z')) for root in (a.native,a.ordinary)]
    assert np.array_equal(native[:,:2],control[:,:2]) and np.array_equal(nw[:,:3],cw[:,:3])
    volume=native[:,1];cut=np.isin(native[:,0],nw[:,1])
    difference={'velocity':error(native[:,2:5],control[:,2:5],volume),
                'pressure':error(native[:,5]-np.average(native[:,5],weights=volume),control[:,5]-np.average(control[:,5],weights=volume),volume),
                'cut_cell_velocity':error(native[cut,2:5],control[cut,2:5],volume[cut]),
                'wall_shear':error(nw[:,3:],cw[:,3:],nw[:,2])}
    ns,cs=[columns(root/'sections.csv',('volume_flux',))[:,0] for root in (a.native,a.ordinary)]
    difference['section_flux']=error(ns,cs,np.ones(len(ns)))
    triangle={}
    for key,value in difference.items():
        # | ||N-A||/||A|| - ||C-A||/||A|| | <= ||N-C||/||C|| * (1 + ||C-A||/||A||).
        n,c=direct[key]['relative_l2'],ordinary[key]['relative_l2']
        bound=value['relative_l2']*(1+c)
        roundoff=64*np.finfo(float).eps*(1+n+c)
        assert abs(n-c)<=bound+roundoff,key
        triangle[key]={'direct':n,'ordinary_to_aphros':c,'native_to_ordinary':value['relative_l2'],
                       'triangle_bound':bound,'difference':abs(n-c),'roundoff_allowance':roundoff,'passed':True}
    script=Path(__file__).with_name('compare_twisted_steady_aphros.py');cases=[]
    for name,native,reference,expected in (
        ('physical_step_as_pseudo',a.ordinary,a.aphros,'requires a native pseudo steady iteration'),
        ('nonfinal_pseudo_iteration',a.native.parent/'iterate_0004',a.aphros,'actual final steady iteration'),
        ('partial_reference_as_complete',a.native,a.snapshot,'run_completion.json')):
        report=out/(name+'.json')
        command=[sys.executable,str(script),'--ours',str(native),'--aphros',str(reference),
                 '--adaptive','--aphros-diffusion-iterations','8','--output',str(report)]
        result=subprocess.run(command,capture_output=True,text=True)
        log=out/(name+'.log');log.write_text(result.stdout+result.stderr)
        assert result.returncode and expected in result.stderr and not report.exists(),(name,result.stderr)
        cases.append({'name':name,'passed':True,'exit_code':result.returncode,'expected_rejection':expected,'command':command,'log_sha256':sha(log)})
    hashes={str(path.resolve()):sha(path) for path in (Path(__file__),script,a.direct,a.ordinary_pair)}
    for root in (a.native,a.ordinary):
        for name in ('solution.csv','walls.csv','sections.csv','mesh_cells.csv','mesh_faces.csv'):
            hashes[str((root/name).resolve())]=sha(root/name)
    result={'passed':True,'scope':__doc__,'original_limits_unchanged':True,'actual_control_triangle_checks':triangle,
            'scope_rejections':cases,'source_sha256':hashes,'new_physical_trajectory_claimed':False,'spatial_convergence_claimed':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'triangle_checks':len(triangle),'scope_rejections':len(cases)}))


if __name__=='__main__':main()
