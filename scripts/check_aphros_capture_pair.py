"""Require bitwise unchanged final fields, geometry, and iteration history for a read-only capture."""
import argparse
import json
import re
from pathlib import Path
from check_twisted_mass import check_case, read
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--original-library-pair',action='store_true',
                        help='Compare capture-disabled rebuilt Proj against the original static-library Proj')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a fresh report')
    roots=[args.candidate.resolve(),args.reference.resolve()]
    sources={};runtime=[];configs=[];histories=[];mass=[]
    for root in roots:
        def get(name):
            sources[str(root/name)]=sha(root/name)
            return json.loads((root/name).read_text(encoding='utf-8-sig'))
        case=get('case_manifest.json');run=get('run_manifest.json');completion=get('run_completion.json')
        if completion['exit_code'] or case['fluid_solver']!='proj':raise ValueError('Need successful Proj trajectories')
        if sha(root/'a.conf')!=case['config_sha256'] or run['config_sha256']!=case['config_sha256']:
            raise ValueError('Executed config changed')
        if sha(Path(run['executable']))!=run['executable_sha256']:raise ValueError('Executed binary changed')
        sources[str(root/'a.conf')]=sha(root/'a.conf');sources[str(root/'run.log')]=sha(root/'run.log')
        pairs=[(int(i),float(x)) for i,x in re.findall(r'iter=(\d+), diff=([\d.eE+\-]+)',(root/'run.log').read_text())]
        starts=[i for i,(it,_) in enumerate(pairs) if it==1]
        if not starts or starts[0]!=0 or len(starts)!=case['time_steps']:raise ValueError('Incomplete trajectory')
        for start,end in zip(starts,starts[1:]+[len(pairs)]):
            seq=pairs[start:end]
            if [it for it,_ in seq]!=list(range(1,len(seq)+1)) or not seq[-1][1]<case['iteration_tolerance']:
                raise ValueError('Incomplete or unconverged inner sequence')
        times=read(root/'tube_b0_time.csv')
        if len(times)!=case['time_steps']:raise ValueError('Missing physical step diagnostics')
        for i,t in enumerate(times['time'],1):
            if abs(t-i*case['time_step'])>1e-12:raise ValueError('Wrong physical time')
        runtime.append(run);configs.append(case);histories.append(pairs);mass.append(check_case(root))
    if configs[0]!=configs[1]:raise ValueError('Different physical/numerical configurations')
    env=[dict(r['environment']) for r in runtime]
    flags=[e.pop('APHROS_TWISTED_CAPTURE_PRESSURE_FACES',None) for e in env]
    if env[0]!=env[1]:raise ValueError('Different environment beyond read-only capture')
    same_exe=runtime[0]['executable_sha256']==runtime[1]['executable_sha256']
    if args.original_library_pair:
        if same_exe or flags!=[None,None]:raise ValueError('Expected different executables with capture disabled')
    elif not same_exe or flags!=['1',None]:raise ValueError('Expected same executable, capture enabled versus disabled')
    names=('a.conf','tube_b0_geometry_cells.csv','tube_b0_geometry_faces.csv',
           'tube_b0_geometry_walls.csv','tube_b0_geometry_polygons.csv','tube_b0_time.csv',
           'proj_final_b0_cells.csv','proj_final_b0_faces.csv','tube_final_b0_walls.csv',
           'proj_final_b0_pressure_rows.csv','proj_final_b0_pressure_snapshot.json')
    identity={}
    for name in names:
        values=[sha(r/name) for r in roots];identity[name]=values[0]==values[1]
        for r,value in zip(roots,values):sources[str(r/name)]=value
    passed=all(identity.values()) and histories[0]==histories[1] and all(m['passed'] for m in mass)
    report={'passed':passed,'scope':__doc__,'original_library_pair':args.original_library_pair,
            'bitwise_identity':identity,'iteration_history_identical':histories[0]==histories[1],
            'total_outer_iterations':[len(h) for h in histories],'mass':mass,
            'source_sha256':sources,'checker_sha256':sha(Path(__file__))}
    if any(sha(Path(p))!=value for p,value in sources.items()):raise ValueError('Inputs changed during check')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':passed,'identity':identity,'iterations':report['total_outer_iterations']}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
