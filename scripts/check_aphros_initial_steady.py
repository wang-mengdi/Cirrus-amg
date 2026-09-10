"""Check recorded velocity-only initial guesses against a completed original cold-start steady reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_twisted_time_pair import read
from check_twisted_mass import check_case
from check_aphros_exact_mass import calculate
from compare_twisted import vector,error
from run_twisted_solver import sha
from validate_aphros_extended_reference import validate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cold','one','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--half',type=Path)
    args=parser.parse_args();roots=[args.cold.resolve(),args.one.resolve()]
    if args.half:roots.append(args.half.resolve())
    if args.output.exists():raise ValueError('Preserve prior verification')
    hashes={}
    def keep(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Input changed: '+str(path))
        hashes[str(path)]=actual;return path
    def load(path):return json.loads(keep(path).read_text(encoding='utf-8-sig'))
    configs=[load(r/'case_manifest.json') for r in roots]
    raw=[keep(r/'a.conf').read_text() for r in roots]
    cold=roots[0];runtime=load(cold/'run_manifest.json');done=load(cold/'run_completion.json')
    build=load(Path(runtime['executable']).parent/'build_manifest.json')
    if done['exit_code'] or build['exit_code']:raise ValueError('Cold reference did not complete')
    keep(runtime['executable'],runtime['executable_sha256']);keep(runtime['executable'],build['executable_sha256'])
    keep(build['library'],build['library_sha256'])
    # The historical producer has since evolved. Verify its exact archived
    # bytes against the actual compiled build, rather than trusting the live file.
    receipt_path=Path(__file__).resolve().parents[1]/'validation/twisted/results/native_projection_checkpoint/receipt.json'
    receipt=load(receipt_path)
    matches=[r for r in receipt['files'] if r.get('source_sha256')==build['source_sha256'] and not r.get('gzip')]
    if not matches:raise ValueError('Missing exact historical cold-driver source')
    driver=keep(receipt_path.parent/matches[0]['path'],build['source_sha256'])
    if 'using M=MeshCartesian<double,3>;' not in driver.read_text():raise ValueError('Unexpected historical scalar')
    if 'End of simulation' not in keep(cold/'run.log').read_text():raise ValueError('Incomplete cold run')
    provenance=[validate(r) for r in roots[1:]]
    for proof in provenance:hashes.update(proof['source_sha256'])
    seeded=[]
    for i,root in enumerate(roots[1:],1):
        completion=load(root/'initial_run_completion.json');prep=load(root/'initial_run_preparation.json')
        if not completion['passed'] or not all(completion['checks'].values()):raise ValueError('Initial velocity run not verified')
        for p,h in prep['source_sha256'].items():keep(p,h)
        info=configs[i]['initial_velocity'];seed=keep(info['file'],info['sha256'])
        seedmeta=load(keep(info['manifest'],info['manifest_sha256']))
        for p,h in seedmeta['source_sha256'].items():keep(p,h)
        keep(root/'initial_velocity_echo.bin',sha(seed))
        normalize=lambda c:{k:v for k,v in c.items() if k not in ('initial_velocity','config_sha256')}
        if normalize(configs[i])!=normalize(configs[0]):raise ValueError('Different prescribed physical problem')
        if raw[i].replace('set string vel_init twisted_seed','set string vel_init zero')!=raw[0]:
            raise ValueError('More than initial condition changed in configuration')
        seeded.append(seedmeta)
    if seeded[0]['scale']!=1:raise ValueError('Require the declared full initial guess')
    if args.half:
        if seeded[1]['scale']!=.5 or seeded[0]['source_run']!=seeded[1]['source_run']:
            raise ValueError('Require the declared half initial guess from the same source')
        if seeded[0]['initial_velocity_sha256']==seeded[1]['initial_velocity_sha256']:
            raise ValueError('Initial guesses are identical')
    times=[]
    for root,cfg in zip(roots,configs):
        keep(root/'a.conf',cfg['config_sha256'])
        table=read(keep(root/'tube_b0_time.csv'))
        if len(table)!=cfg['time_steps'] or not np.allclose(table['time'],np.arange(1,len(table)+1)*cfg['time_step'],rtol=1e-12,atol=0):
            raise ValueError('Incomplete physical time sequence')
        times.append({k:float(table[-1][k]) for k in table.dtype.names})
    geometry=[read(keep(r/'tube_b0_geometry_cells.csv')) for r in roots]
    if any(g.shape!=geometry[0].shape or any(not np.array_equal(g[k],geometry[0][k]) for k in g.dtype.names) for g in geometry[1:]):
        raise ValueError('Different computational geometry')
    mass=[check_case(roots[0])]+[calculate(r) for r in roots[1:]]
    for m in mass:hashes.update(m.get('source_sha256',{}))
    cells=[read(keep(r/'proj_final_b0_cells.csv')) for r in roots]
    faces=[read(keep(r/'proj_final_b0_faces.csv')) for r in roots]
    walls=[read(keep(r/'tube_final_b0_walls.csv')) for r in roots]
    cut=geometry[0]['cut']>0;pairs={}
    for a,b in ((a,b) for a in range(len(roots)) for b in range(a+1,len(roots))):
        for data in (cells,faces,walls):
            if data[a].shape!=data[b].shape or not np.array_equal(vector(data[a],('x','y','z')),vector(data[b],('x','y','z'))):
                raise ValueError('Different field sampling locations')
        weight=cells[a]['volume'];u=[vector(c,('u','v','w')) for c in (cells[a],cells[b])]
        p=[c['p']-np.average(c['p'],weights=weight) for c in (cells[a],cells[b])]
        fields={'velocity':error(u[1],u[0],weight),'pressure':error(p[1],p[0],weight),
            'cut_velocity':error(u[1][cut],u[0][cut],weight[cut]),
            'wall_shear':error(vector(walls[b],('tau_x','tau_y','tau_z')),vector(walls[a],('tau_x','tau_y','tau_z')),walls[a]['area']),
            'face_velocity':error(faces[b]['flux']/faces[b]['area'],faces[a]['flux']/faces[a]['area'],faces[a]['area'])}
        pairs[f'{a}-{b}']=fields
    limit=1e-6
    checks={'all_mass_checks':all(m['passed'] for m in mass),
        'all_temporally_steady':all(t['temporal_acceleration_volume_l2']<1e-8*np.linalg.norm(c['spec']['force']) for t,c in zip(times,configs)),
        'all_fields_agree':all(v['relative_l2']<limit for p in pairs.values() for v in p.values())}
    result={'passed':all(checks.values()),'scope':__doc__+' This does not establish finer-grid accuracy or spatial convergence.',
        'checks':checks,'relative_l2_limit':limit,'pairs':pairs,'final_time_records':times,
        'mass':mass,'initial_scales':[s['scale'] for s in seeded],'source_sha256':hashes,
        'checker_sha256':sha(Path(__file__)),'goal_complete':False}
    if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('Inputs changed during verification')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','mass')}))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
