"""Check actual fine Aphros slab geometry for unique entities, fluid adjacency, periodic consistency and area-vector closure."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from run_twisted_solver import sha
from twisted_geometry import load_geometry
from profile_twisted_refinement_layout import classify,layout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    path=args.geometry.resolve();sources={str(p):sha(p) for p in [path,Path(__file__),Path(__file__).with_name('twisted_geometry.py'),Path(__file__).with_name('profile_twisted_refinement_layout.py'),Path(__file__).with_name('run_twisted_solver.py')]}
    passed=False;checks={}
    try:
        geometry=load_geometry(path);ny=geometry['finest_ny'];h=geometry['finest_h']
        assert geometry['extent']==[.25,.125,.125] and geometry['reference_translation_cells']==[0,0,0]
        shape=np.array([2*ny,ny,ny]);count=int(np.prod(shape));node_shape=shape+1
        for name in ['cells','faces','walls']:
            p=path.parent/geometry['tables'][name]['file'];sources[str(p)]=sha(p)
        c,f,w=[geometry[n] for n in ['cells','faces','walls']]
        index=np.full(count,-1,dtype=np.int32)
        area=np.zeros((len(c),3));cut=np.empty(len(c),dtype=np.bool_)
        def cell_code(q):return (q[:,2]*shape[1]+q[:,1])*shape[0]+q[:,0]
        last=-1;volume=0.;minimum_fraction=1.
        for start in range(0,len(c),50000):
            rows=np.asarray(c[start:start+50000]);q=rows[:,:3].astype(np.int64)
            assert np.array_equal(q,rows[:,:3]) and np.all((q>=0)&(q<shape))
            code=cell_code(q);assert code[0]>last and np.all(np.diff(code)>0);last=int(code[-1])
            assert np.isfinite(rows).all() and np.all(rows[:,6]==h)
            assert np.all(rows[:,7]>0) and np.all(rows[:,7]<=h**3*(1+1e-12))
            assert np.all((rows[:,8]==0)|(rows[:,8]==1))
            assert np.array_equal(rows[:,3:6],(q+.5)*h)
            regular=rows[:,8]==0;assert np.all(rows[regular,7]==h**3)
            index[code]=np.arange(start,start+len(rows),dtype=np.int32);cut[start:start+len(rows)]=~regular
            volume+=float(rows[:,7].sum());minimum_fraction=min(minimum_fraction,float(rows[:,7].min()/h**3))
        face_seen=np.zeros(int(np.prod(node_shape))*3,dtype=np.bool_)
        seams=[[],[]];canonical_faces=0
        for start in range(0,len(f),50000):
            rows=np.asarray(f[start:start+50000]);q=rows[:,1:4].astype(np.int64);axis=rows[:,0].astype(np.int64)
            assert np.isfinite(rows).all() and np.array_equal(q,rows[:,1:4]) and np.array_equal(axis,rows[:,0])
            assert np.all((axis>=0)&(axis<3)) and np.all((q>=0)&(q<node_shape))
            assert np.all(rows[:,7]>0) and np.all(rows[:,7]<=h*h*(1+1e-12))
            code=((q[:,2]*node_shape[1]+q[:,1])*node_shape[0]+q[:,0])*3+axis
            assert len(np.unique(code))==len(code) and not face_seen[code].any();face_seen[code]=True
            for side,pos in enumerate([0,shape[0]]):
                mask=(axis==0)&(q[:,0]==pos)
                if mask.any():seams[side].append(rows[mask].copy())
            keep=~((axis==0)&(q[:,0]==shape[0]));q=q[keep];axis=axis[keep];a=rows[keep,7]
            lo=q.copy();lo[np.arange(len(q)),axis]-=1;lo[:,0]%=shape[0]
            hi=q.copy();hi[:,0]%=shape[0]
            assert np.all((lo>=0)&(lo<shape)) and np.all((hi>=0)&(hi<shape))
            owner=index[cell_code(lo)];neighbor=index[cell_code(hi)]
            assert np.all(owner>=0) and np.all(neighbor>=0)
            np.add.at(area,(owner,axis),a);np.add.at(area,(neighbor,axis),-a);canonical_faces+=len(q)
        del face_seen
        seam=[]
        for chunks in seams:
            rows=np.concatenate(chunks);order=np.lexsort((rows[:,2],rows[:,3]));seam.append(rows[order])
        assert np.array_equal(seam[0][:,2:4],seam[1][:,2:4])
        periodic_area_error=float(np.max(np.abs(seam[0][:,7]-seam[1][:,7]))/(h*h))
        assert periodic_area_error<=1e-12
        wall_keys=np.asarray(w[:,:3],dtype=np.int64);wall_code=cell_code(wall_keys)
        assert np.array_equal(wall_keys,w[:,:3]) and len(np.unique(wall_code))==len(w)
        wall_owner=index[wall_code];assert np.all(wall_owner>=0) and cut[wall_owner].all()
        assert len(w)==int(cut.sum()) and np.all(w[:,9]>0)
        normal_error=float(np.max(np.abs(np.linalg.norm(w[:,6:9],axis=1)-1)))
        assert normal_error<=1e-10
        area[wall_owner]+=w[:,9,None]*w[:,6:9]
        closure=0.
        for start in range(0,len(area),50000):closure=max(closure,float(np.max(np.linalg.norm(area[start:start+50000],axis=1))/(h*h)))
        assert closure<=1e-10
        del area,index,cut
        predicted=classify(geometry['geometry_spec'],ny,[0,0,0],retain_keys=True)
        xyz_code=(wall_keys[:,0]*ny+wall_keys[:,1])*ny+wall_keys[:,2]
        assert np.array_equal(np.sort(xyz_code),np.sort(predicted['wall_keys']))
        assert int(predicted['fluid'].sum())==len(c)
        roots=[list(map(int,k)) for k in zip(*np.nonzero(predicted['refined']))]
        assert roots==geometry['refine_root_tiles']
        checks={'fluid_cells':len(c),'positive_faces_with_periodic_duplicates':len(f),'canonical_faces':canonical_faces,
                'cut_wall_cells':len(w),'fluid_volume':volume,'minimum_volume_fraction':minimum_fraction,
                'maximum_area_vector_closure_over_h_squared':closure,'maximum_unit_normal_error':normal_error,
                'periodic_aperture_difference_over_h_squared':periodic_area_error,
                'all_positive_faces_have_two_fluid_neighbors':True,'all_cell_face_wall_keys_unique':True,
                'all_analytic_cut_cell_ids_equal':True,'all_analytic_fluid_counts_equal':True,'refinement_roots_equal':True,
                'predicted_native_layout':layout(predicted,ny,ny//2,20)}
        passed=True
    finally:
        unchanged=all(sha(Path(p))==d for p,d in sources.items())
        result={'scope':__doc__,'passed':passed and unchanged,'completed_utc':datetime.now(timezone.utc).isoformat(),
                'checks':checks,'source_sha256':sources,'inputs_unchanged':unchanged,'native_mesh_constructed':False,
                'flow_accuracy_checked':False,'goal_complete':False}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
