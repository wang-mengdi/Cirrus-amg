"""Compare original Aphros runs differing only in implicit diffusion iteration count.

This checks the final fields of complete configured trajectories, all step-stop
records, exact geometry, and actual cut-volume mass. It is not a grid accuracy test.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from check_twisted_mass import read, vector, check_case
from compare_twisted import error
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier comparisons')
    roots=[args.candidate.resolve(),args.reference.resolve()]
    configs=[json.loads((r/'case_manifest.json').read_text()) for r in roots]
    counts=[c['projection_parameters']['diffusion_iters'] for c in configs]
    if counts[0]<=1 or counts[1]!=1:raise ValueError('Expected multiple versus one original diffusion iteration')
    normalized=[]
    for c in configs:
        d=json.loads(json.dumps(c));d.pop('config_sha256');d['projection_parameters'].pop('diffusion_iters');normalized.append(d)
    if normalized[0]!=normalized[1]:raise ValueError('Different physical/numerical configuration')
    text=[re.sub(r'^set int proj_diffusion_iters \d+$','set int proj_diffusion_iters VALUE',(r/'a.conf').read_text(),flags=re.M) for r in roots]
    if text[0]!=text[1]:raise ValueError('Actual configurations differ beyond the diffusion iteration count')
    runtimes=[];iteration_counts=[];sources=[];masses=[];stop_histories=[]
    for root,c in zip(roots,configs):
        runtime=json.loads((root/'run_manifest.json').read_text(encoding='utf-8-sig'));runtimes.append(runtime)
        if sha(root/'a.conf')!=c['config_sha256'] or runtime['config_sha256']!=c['config_sha256']:
            raise ValueError('Executed input changed')
        if sha(Path(runtime['executable']))!=runtime['executable_sha256']:raise ValueError('Actual executable changed')
        completion=json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))
        if completion['exit_code']:raise ValueError('Reference process failed')
        log=(root/'run.log').read_text();rows=[(int(i),float(v)) for i,v in re.findall(r'iter=(\d+), diff=([\d.eE+\-]+)',log)]
        starts=[i for i,(it,_) in enumerate(rows) if it==1]
        if len(starts)!=c['time_steps'] or starts[0]!=0:raise ValueError('Incomplete iteration history')
        endings=[rows[end-1] for end in starts[1:]+[len(rows)]]
        if any(v>=c['iteration_tolerance'] or not np.isfinite(v) for _,v in endings):raise ValueError('Physical step did not converge')
        iteration_counts.append(sum(it for it,_ in endings));stop_histories.append(endings)
        times=read(root/'tube_b0_time.csv')
        if len(times)!=c['time_steps'] or not np.allclose(times['time'],np.arange(1,len(times)+1)*c['time_step'],rtol=1e-12,atol=0):
            raise ValueError('Incomplete or different physical time sequence')
        masses.append(check_case(root))
        sources += [root/n for n in ('a.conf','case_manifest.json','run_manifest.json','run_completion.json','run.log','tube_b0_time.csv')]
    if runtimes[0]['executable_sha256']!=runtimes[1]['executable_sha256'] or runtimes[0]['environment']!=runtimes[1]['environment']:
        raise ValueError('Require identical original executable and linear-backend environment')
    for name in ('tube_b0_geometry_cells.csv','tube_b0_geometry_faces.csv','tube_b0_geometry_walls.csv','tube_b0_geometry_polygons.csv'):
        if sha(roots[0]/name)!=sha(roots[1]/name):raise ValueError('Different discrete geometry')
        sources += [r/name for r in roots]
    cells=[read(r/'proj_final_b0_cells.csv') for r in roots]
    walls=[read(r/'tube_final_b0_walls.csv') for r in roots]
    faces=[read(r/'proj_final_b0_faces.csv') for r in roots]
    for data,keys in ((cells,('x','y','z','volume')),(walls,('x','y','z','area')),(faces,('axis','x','y','z','area'))):
        if not np.array_equal(vector(data[0],keys),vector(data[1],keys)):raise ValueError('Different final sample geometry/order')
    V=cells[0]['volume'];u=[vector(c,'uvw') for c in cells]
    p=[c['p']-np.average(c['p'],weights=V) for c in cells]
    h=configs[0]['spec']['extent'][1]/configs[0]['ny'];cut=V<h**3*(1-1e-12)
    fields={'velocity':error(*u,V),'pressure':error(*p,V),'cut_cell_velocity':error(u[0][cut],u[1][cut],V[cut]),
            'wall_shear':error(*[vector(w,('tau_x','tau_y','tau_z')) for w in walls],walls[0]['area']),
            'shared_face_flux':error(*[f['flux'] for f in faces],faces[0]['area'])}
    for r in roots:sources += [r/n for n in ('proj_final_b0_cells.csv','proj_final_b0_faces.csv','tube_final_b0_walls.csv')]
    passed=all(m['passed'] for m in masses) and all(v['relative_l2']<1e-6 for v in fields.values())
    report={'passed':passed,'scope':__doc__,'diffusion_iterations_candidate_reference':counts,
            'outer_iterations_candidate_reference':iteration_counts,'per_step_stop_records':stop_histories,
            'derived_scalar_diffusion_solves':[3*n*it for n,it in zip(counts,iteration_counts)],
            'fields':fields,'relative_l2_limit':1e-6,'mass':masses,
            'source_sha256':{str(p):sha(p) for p in sources},'checker_sha256':sha(Path(__file__))}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':passed,'outer_iterations':iteration_counts,'fields':{k:v['relative_l2'] for k,v in fields.items()},'mass_passed':[m['passed'] for m in masses]}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
