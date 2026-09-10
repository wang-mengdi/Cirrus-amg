"""Package Aphros cut GEOMETRY ONLY as input to the native octree benchmark.

This does not read velocity, pressure, or any solution. The common geometry
isolates discretization from geometry mismatch. Grid refinement and flow
unknowns are constructed independently by Cirrus using its HADeviceGrid.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_twisted_solver import sha
from twisted_geometry import HEADER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--packed',action='store_true',help='Write exact float64 tables plus a small metadata JSON')
    parser.add_argument('--shift-cells',type=int,nargs=3,default=[0,0,0],
                        help='Exact integer-cell translation; x wraps periodically, y/z must stay inside the box')
    args=parser.parse_args()
    if args.packed and (args.output.exists() or any(args.output.with_suffix(f'.{name}.bin').exists() for name in ('cells','faces','walls'))):
        raise ValueError('Preserve existing packed geometry; use a fresh output name')
    root=args.baseline.resolve()
    manifest=json.loads((root/'case_manifest.json').read_text(encoding='utf-8'))
    result={'format':'aphros_cut_geometry_v1','extent':manifest['spec']['extent'],
            'finest_ny':manifest['ny'],'geometry_spec':manifest['spec'],'source_sha256':{}}
    for name in ('cells','faces','walls'):
        path=root/f'tube_b0_geometry_{name}.csv'
        with path.open(encoding='utf-8') as f: columns=f.readline().strip().split(',')
        data=np.loadtxt(path,skiprows=1,delimiter=',',ndmin=2)
        if not len(data) or not np.isfinite(data).all(): raise ValueError(f'Invalid geometry: {path}')
        result[name+'_columns']=columns
        result[name]=data if args.packed else data.tolist()
        result['source_sha256'][str(path)]=sha(path)
    hf=manifest['spec']['extent'][1]/manifest['ny']
    if not np.all(np.array(result['cells'])[:,6]==hf): raise ValueError('Nonuniform reference spacing')
    result['finest_h']=hf
    shape=np.array(manifest['shape'],dtype=int)
    shift=np.array(args.shift_cells,dtype=int)
    result['reference_translation_cells']=shift.tolist()
    result['reference_translation']=(shift*hf).tolist()
    if np.any(shift):
        for name in ('cells','faces','walls'):
            data=np.array(result[name])
            cols=result[name+'_columns']
            for axis,label in enumerate(('i','j','k')):
                data[:,cols.index(label)]+=shift[axis]
            # Every entity is translated as a whole in its adjacent periodic
            # image; face x=L may canonicalize to x=0 and keeps duplicate checks.
            ix=cols.index('i')
            wrap=np.floor_divide(data[:,ix].astype(int),shape[0])
            data[:,ix]-=wrap*shape[0]
            for axis,label in enumerate(('x','y','z')):
                data[:,cols.index(label)]+=shift[axis]*hf
                if axis==0:data[:,cols.index(label)]-=wrap*manifest['spec']['extent'][0]
                glabel='geometry_'+label
                if glabel in cols:
                    data[:,cols.index(glabel)]+=shift[axis]*hf
                    if axis==0:data[:,cols.index(glabel)]-=wrap*manifest['spec']['extent'][0]
            if name=='cells' and (np.any(data[:,1:3]<0) or np.any(data[:,1:3]>=shape[1:])):
                raise ValueError('Translation moves fluid outside the transverse box')
            result[name]=data if args.packed else data.tolist()
    # Two fine cells of padding keep wall-fit and bilinear face stencils fine.
    roots=set()
    for row in result['walls']:
        key=np.array(row[:3],dtype=int)
        for i in (-2,0,2):
            for j in (-2,0,2):
                for k in (-2,0,2):
                    q=key+np.array([i,j,k]); q[0]%=shape[0]
                    if ((q>=0)&(q<shape)).all(): roots.add(tuple(int(x) for x in q//16))
    result['refine_root_tiles']=[list(k) for k in sorted(roots)]
    result['wall_refinement_padding_fine_cells']=2
    result['baseline_polygons']=str(root/'tube_b0_geometry_polygons.csv')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fluid_cells=len(result['cells']);wall_faces=len(result['walls'])
    if args.packed:
        result['format']='aphros_cut_geometry_v2';result['tables']={}
        for name in ('cells','faces','walls'):
            data=np.asarray(result.pop(name),dtype='<f8')
            binary=args.output.with_suffix(f'.{name}.bin')
            with binary.open('wb') as stream:
                stream.write(HEADER.pack(b'CIRRCUT1',len(data),data.shape[1],0x01020304))
                data.tofile(stream)
            result['tables'][name]={'file':binary.name,'rows':len(data),'columns':data.shape[1],'sha256':sha(binary)}
    args.output.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
    metadata={key:value for key,value in result.items() if key not in ('cells','faces','walls')}
    args.output.with_suffix('.meta.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output.resolve()),'fluid_cells':fluid_cells,
                      'walls':wall_faces,'refined_root_tiles':len(roots),'format':result['format']}))


if __name__=='__main__': main()
