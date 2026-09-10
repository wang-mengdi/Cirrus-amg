"""Prepare a traceable velocity-only initial guess; it is not a baseline solution."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from scipy.spatial import cKDTree
from check_twisted_time_pair import read
from compare_twisted import vector
from run_twisted_solver import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--kind',choices=('aphros','native'),required=True)
    parser.add_argument('--native-steady-iteration',action='store_true',
                        help='Explicitly validate a completed native pseudo steady iterate as a velocity-only source')
    parser.add_argument('--target-case',type=Path,required=True)
    parser.add_argument('--geometry-run',type=Path,required=True)
    parser.add_argument('--scale',type=float,default=1)
    parser.add_argument('--interpolation-batch-size',type=int,default=4096,
                        help='Maximum target points in each interpolation query; reuse one source search tree')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();target=args.target_case.resolve()
    if args.native_steady_iteration and args.kind!='native':parser.error('Steady-iteration scope requires --kind native')
    if not 1<=args.interpolation_batch_size<=65536:parser.error('Interpolation batch size must be in 1..65536')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not np.isfinite(args.scale) or not 0<args.scale<=2:raise ValueError('Require a finite scale in (0,2]')
    inputs={}
    def keep(path):
        path=Path(path).resolve();inputs[str(path)]=sha(path);return path
    keep(Path(__file__))
    cfg=json.loads(keep(target/'case_manifest.json').read_text())
    if sha(keep(target/'a.conf'))!=cfg['config_sha256']:raise ValueError('Target config changed')
    shape=np.asarray(cfg['shape'],dtype=np.int64);h=np.array(cfg['spec']['extent'])/shape
    if np.any(shape<=0) or not np.all(h==h[0]):raise ValueError('Require the uniform reference mesh')
    geom=read(keep(args.geometry_run/'tube_b0_geometry_cells.csv'))
    actual=read(keep(target/'tube_b0_geometry_cells.csv'))
    if geom.dtype.names!=actual.dtype.names or geom.shape!=actual.shape or any(
            not np.array_equal(geom[k],actual[k]) for k in geom.dtype.names):
        raise ValueError('Target computational geometry differs from the original capture')
    points=vector(geom,('x','y','z'));ijk=vector(geom,('i','j','k')).astype(np.int64)
    if np.any(ijk<0) or np.any(ijk>=shape) or not np.array_equal((ijk+.5)*h,points):
        raise ValueError('Geometry centers are not on the prescribed grid')
    flat=(ijk[:,2]*shape[1]+ijk[:,1])*shape[0]+ijk[:,0]
    if len(np.unique(flat))!=len(flat):raise ValueError('Duplicate fluid cells')
    condition=None;native_proof=None;batch_count=0
    if args.kind=='aphros':
        done=json.loads(keep(source/'run_completion.json').read_text(encoding='utf-8-sig'))
        source_cfg=json.loads(keep(source/'case_manifest.json').read_text())
        if done['exit_code'] or source_cfg['spec']!=cfg['spec']:raise ValueError('Require a completed source with the same physics')
        data=read(keep(source/'proj_final_b0_cells.csv'))
        if len(data)!=len(geom) or not np.array_equal(vector(data,('x','y','z')),points) or not np.array_equal(data['volume'],geom['volume']):
            raise ValueError('Aphros seed requires identical grid locations and cut volumes')
        values=vector(data,('u','v','w'));matched=len(values);interpolated=0
    else:
        from analyze_twisted_refinement import load,completed_trajectory,completed_steady_iteration,sample
        script=Path(__file__).with_name('analyze_twisted_refinement.py');keep(script)
        native,meta,metrics,cells,_=load(source,args.native_steady_iteration)
        proof=completed_steady_iteration(source) if args.native_steady_iteration else completed_trajectory(source,native)
        if args.native_steady_iteration:
            native_proof=proof
            keep(Path(__file__).with_name('check_twisted_steady_iteration.py'))
            inputs.update(proof['executed_input_sha256'])
        inputs.update(proof['source_sha256'])
        for name in ('solution.csv','case.json','metrics.json'):keep(source/name)
        geo=Path(native['embedded_geometry']);keep(geo)
        if geo.with_suffix('.meta.json').exists():keep(geo.with_suffix('.meta.json'))
        if meta['geometry_spec']!=cfg['spec'] or native['time_step']!=cfg['time_step']:
            raise ValueError('Native seed has different physics or time step')
        centers=vector(cells,('x','y','z'));u=vector(cells,('u','v','w'))
        tree=cKDTree(centers)
        distances,ids=tree.query(points)
        exact=distances<h[0]*1e-10
        values=np.empty((len(points),3));values[exact]=u[ids[exact]]
        matched=int(exact.sum());interpolated=int((~exact).sum())
        if interpolated:
            targets=np.flatnonzero(~exact);condition=0.
            for start in range(0,len(targets),args.interpolation_batch_size):
                selected=targets[start:start+args.interpolation_batch_size]
                result,current=sample(centers,u,points[selected],cfg['spec']['period'],32,tree=tree)
                if not np.isfinite(current) or current>1e8:raise ValueError('Ill-conditioned initial interpolation')
                values[selected]=result;condition=max(condition,current);batch_count+=1
    values*=args.scale
    if not np.all(np.isfinite(values)):raise ValueError('Nonfinite initial velocity')
    count=int(np.prod(shape));mask=np.zeros(count,dtype=np.uint8);mask[flat]=1
    dense=np.zeros((count,3),dtype='<f8');dense[flat]=values
    path=out/'velocity.bin'
    with path.open('xb') as f:
        f.write(struct.pack('<8s3Q3d',b'TWVEL01',*shape.tolist(),*h.tolist()))
        f.write(mask.tobytes());f.write(dense.tobytes())
    result={'scope':__doc__,'source_kind':args.kind,'source_run':str(source),'target_case':str(target),
        'shape':shape.tolist(),'spacing':h.tolist(),'scale':args.scale,'fluid_cells':len(values),
        'exact_source_centers':matched,'interpolated_centers':interpolated,'maximum_fit_condition':condition,
        'interpolation_batch_size':args.interpolation_batch_size,'interpolation_batches':batch_count,
        'native_source_iteration_mode':('steady_pseudo_iteration' if args.native_steady_iteration else 'physical_trajectory') if args.kind=='native' else None,
        'maximum_seed_speed':float(np.max(np.linalg.norm(values,axis=1))),
        'initial_velocity_file':str(path),'initial_velocity_sha256':sha(path),'source_sha256':inputs,
        'only_velocity_initialized':True,'steady_alignment_proven':False,'goal_complete':False}
    if native_proof is not None:
        # Keep the exact complete source check separately; it certifies the
        # source state, never the interpolated initial guess as an Aphros result.
        proof_path=out/'native_steady_source_check.json'
        proof_path.write_text(json.dumps(native_proof,indent=2)+'\n')
        result['native_steady_source_check']={'file':str(proof_path),'sha256':sha(proof_path)}
    if any(sha(Path(p))!=v for p,v in inputs.items()):raise ValueError('Seed source changed during preparation')
    (out/'seed_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))

if __name__=='__main__':main()
