"""Compare completed extended Aphros runs on identical prescribed inputs."""
import argparse
import datetime
import json
from pathlib import Path
import re
import numpy as np
from compare_twisted import read, vector, error
from check_aphros_exact_mass import calculate
from validate_aphros_extended_reference import validate
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--equivalent-seed-paths',action='store_true',
                   help='Permit different recorded seed file paths only after checking both exact payloads and actual loader echoes')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve prior precision comparisons')
    roots=[args.reference.resolve(),args.candidate.resolve()];provenance=[validate(r) for r in roots]
    cfg=[json.loads((r/'case_manifest.json').read_text()) for r in roots]
    seed_hashes={};normalized=cfg
    if args.equivalent_seed_paths:
        normalized=[];payloads=[]
        for root,config in zip(roots,cfg):
            initial=config.get('initial_velocity')
            if not initial:raise ValueError('Equivalent seed paths require two actually seeded runs')
            for key,digest in (('file','sha256'),('manifest','manifest_sha256')):
                path=Path(initial[key]).resolve()
                if sha(path)!=initial[digest]:raise ValueError('Recorded initial input changed')
                seed_hashes[str(path)]=sha(path)
            echo=root/'initial_velocity_echo.bin'
            if sha(echo)!=initial['sha256']:raise ValueError('Actual initial loader differs from the seed bytes')
            seed_hashes[str(echo)]=sha(echo);payloads.append(initial['sha256'])
            metadata=json.loads(Path(initial['manifest']).read_text())
            if metadata['initial_velocity_sha256']!=initial['sha256'] or metadata['shape']!=config['shape']:
                raise ValueError('Seed manifest differs from actual initial payload or shape')
            normalized.append({**config,'initial_velocity':{k:v for k,v in initial.items()
                               if k not in ('file','manifest','manifest_sha256')}})
        if payloads[0]!=payloads[1]:raise ValueError('Equivalent seed paths contain different initial velocities')
    if normalized[0]!=normalized[1] or sha(roots[0]/'a.conf')!=sha(roots[1]/'a.conf'):
        raise ValueError('Require identical input configurations')
    mass=[calculate(r) for r in roots];hashes=dict(seed_hashes);bitwise={};fields={}
    arrays={}
    for name in ('proj_final_b0_cells.csv','proj_final_b0_faces.csv','tube_final_b0_walls.csv'):
        a,b=[read(r/name) for r in roots]
        if a.shape!=b.shape or a.dtype.names!=b.dtype.names:raise ValueError('Different field layout')
        for column in ('x','y','z'):
            if not np.array_equal(a[column],b[column]):raise ValueError('Different field sampling locations')
        arrays[name]=(a,b)
    a,b=arrays['proj_final_b0_cells.csv'];v=a['volume']
    fields['velocity']=error(vector(b,'uvw'),vector(a,'uvw'),v)
    fields['pressure']=error(b['p']-np.average(b['p'],weights=v),a['p']-np.average(a['p'],weights=v),v)
    a,b=arrays['tube_final_b0_walls.csv']
    fields['wall_shear']=error(vector(b,('tau_x','tau_y','tau_z')),vector(a,('tau_x','tau_y','tau_z')),a['area'])
    a,b=arrays['proj_final_b0_faces.csv']
    fields['face_velocity']=error(b['flux']/b['area'],a['flux']/a['area'],a['area'])
    histories=[re.findall(r'iter=(\d+), diff=([\d.eE+\-]+)',(r/'run.log').read_text()) for r in roots]
    for name in (*arrays,'proj_final_b0_exact_cells.csv','proj_final_b0_exact_faces.csv','tube_b0_time.csv'):
        bitwise[name]=sha(roots[0]/name)==sha(roots[1]/name)
        for root in roots:hashes[str(root/name)]=sha(root/name)
    durations=[]
    for root in roots:
        runtime=json.loads((root/'run_manifest.json').read_text(encoding='utf-8-sig'))
        completion=json.loads((root/'run_completion.json').read_text(encoding='utf-8-sig'))
        stamp=lambda s:datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
        durations.append((stamp(completion['completed_utc'])-stamp(runtime['started_utc'])).total_seconds())
    checks={'both_mass_checks':all(m['passed'] for m in mass),
            'identical_outer_iteration_errors':histories[0]==histories[1],
            'same_discrete_fields':all(value['relative_l2']<1e-6 for value in fields.values())}
    for item in provenance:hashes.update(item['source_sha256'])
    for item in mass:hashes.update(item['source_sha256'])
    result={'passed':all(checks.values()),'scope':__doc__+' This is an implementation comparison, not a steady or grid-convergence result.',
            'equivalent_seed_paths_checked':args.equivalent_seed_paths,
            'checks':checks,'fields':fields,'bitwise_equal_outputs':bitwise,'exact_mass':mass,
            'wall_seconds':durations,'timing_scope':'Observed wall times with other jobs running; not an exclusive performance benchmark',
            'outer_iterations':[len(h) for h in histories],'source_sha256':hashes,'checker_sha256':sha(Path(__file__))}
    if any(sha(Path(path))!=value for path,value in hashes.items()):raise ValueError('Pair inputs changed')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','exact_mass')}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
