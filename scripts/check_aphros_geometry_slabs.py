"""Require exact keyed equality of original full-domain and slab Aphros cut geometry, including polygon vertices."""
import argparse
from datetime import datetime,timezone
import io
import itertools
import json
from pathlib import Path
import numpy as np
from run_twisted_solver import sha
from twisted_geometry import HEADER,load_geometry


def binary(path,width):
    with path.open('rb') as stream:magic,rows,columns,endian=HEADER.unpack(stream.read(HEADER.size))
    assert (magic,columns,endian)==(b'CIRRCUT1',width,0x01020304)
    assert path.stat().st_size==HEADER.size+rows*width*8
    return np.memmap(path,dtype='<f8',mode='r',offset=HEADER.size,shape=(rows,width))


def keys(data,name,ny):
    face=name in ('faces','polygons');columns=4 if face else 3
    integers=np.asarray(data[:,:columns],dtype=np.int64)
    assert np.array_equal(integers,data[:,:columns])
    q=integers[:,1:] if face else integers
    assert np.all(q>=0) and np.all(q<=np.array([2*ny,ny,ny]))
    code=(q[:,2]*(ny+1)+q[:,1])*(2*ny+1)+q[:,0]
    if face:
        axis=integers[:,0];assert np.all((axis>=0)&(axis<(4 if name=='polygons' else 3)))
        code=code*4+axis
    if name=='polygons':
        vertex=np.asarray(data[:,4],dtype=np.int64)
        assert np.array_equal(vertex,data[:,4]) and np.all((vertex>=0)&(vertex<16))
        code=code*16+vertex
    order=np.argsort(code);ordered=code[order]
    assert np.all(ordered[1:]>ordered[:-1]),'Duplicate geometry keys'
    return ordered,order


def compare(a,b,name,ny):
    assert a.shape==b.shape,(name,a.shape,b.shape)
    ak,ai=keys(a,name,ny);bk,bi=keys(b,name,ny)
    assert np.array_equal(ak,bk),(name,'Missing or changed geometry keys')
    equal=True;different=0;maxabs=np.zeros(a.shape[1])
    first=[]
    for start in range(0,len(a),25000):
        x=np.ascontiguousarray(a[ai[start:start+25000]])
        y=np.ascontiguousarray(b[bi[start:start+25000]])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        different_bits=x.view(np.uint64)!=y.view(np.uint64)
        mask=np.any(different_bits,axis=1);different+=int(mask.sum());equal &= not mask.any()
        maxabs=np.maximum(maxabs,np.max(np.abs(x-y),axis=0))
        for i in np.flatnonzero(mask)[:max(0,5-len(first))]:
            first.append({'key':int(ak[start+i]),'reference':x[i].tolist(),'candidate':y[i].tolist()})
    return {'passed':bool(equal),'rows':len(a),'columns':a.shape[1],'unique_keys_equal':True,
            'bitwise_identical_after_key_ordering':bool(equal),'different_rows':different,
            'maximum_absolute_difference_by_column':maxabs.tolist(),'first_differences':first}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources={};tables={};passed=False
    def keep(path):
        path=Path(path).resolve();sources[str(path)]=sha(path)
    for path in [Path(__file__),Path(__file__).with_name('twisted_geometry.py'),Path(__file__).with_name('run_twisted_solver.py'),args.reference,args.candidate]:keep(path)
    try:
        reference=load_geometry(args.reference);candidate=load_geometry(args.candidate)
        for key in ['extent','geometry_spec','finest_ny','finest_h','reference_translation_cells','reference_translation','wall_refinement_padding_fine_cells','refine_root_tiles']:
            assert candidate[key]==reference[key],key
        ny=reference['finest_ny']
        for name in ['cells','faces','walls']:
            for path,data in [(args.reference,reference),(args.candidate,candidate)]:keep(path.parent/data['tables'][name]['file'])
            tables[name]=compare(reference[name],candidate[name],name,ny)
        source_poly=Path(reference['baseline_polygons']);keep(source_poly)
        converted=out/'reference.polygons.bin';rows=0
        with source_poly.open('r') as stream,converted.open('wb') as target:
            assert stream.readline().strip()=='axis,i,j,k,vertex,x,y,z'
            target.write(HEADER.pack(b'CIRRCUT1',0,8,0x01020304))
            while True:
                lines=list(itertools.islice(stream,25000))
                if not lines:break
                data=np.loadtxt(io.StringIO(''.join(lines)),delimiter=',',ndmin=2,dtype='<f8')
                assert data.shape[1]==8;data.tofile(target);rows+=len(data)
            target.seek(0);target.write(HEADER.pack(b'CIRRCUT1',rows,8,0x01020304))
        keep(converted)
        candidate_poly=args.candidate.parent/candidate['tables']['polygons']['file'];keep(candidate_poly)
        assert sha(candidate_poly)==candidate['tables']['polygons']['sha256']
        tables['polygons']=compare(binary(converted,8),binary(candidate_poly,8),'polygons',ny)
        passed=all(t['passed'] for t in tables.values())
    finally:
        unchanged=all(sha(Path(p))==d for p,d in sources.items())
        result={'scope':__doc__,'passed':passed and unchanged,'completed_utc':datetime.now(timezone.utc).isoformat(),
                'tables':tables,'source_sha256':sources,'inputs_unchanged':unchanged,'flow_accuracy_checked':False,'goal_complete':False}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
