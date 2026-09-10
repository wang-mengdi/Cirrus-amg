"""Forecast wall-refined tile layouts after validating the classifier on actual native geometry.

This is a bounded CPU planning calculation, not a new native mesh or flow run.
Future cut-cell identities are predicted from the eight analytic corner signs.
The supplied completed mesh checks the classifier, refinement mask and leaf
counts at its actual resolution. Multi-level forecasts enforce a conservative
26-neighbor 2:1 balance; native ghost/parent storage and solver workspaces are
excluded from the explicitly labeled memory lower bounds.
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from run_twisted_solver import sha
from twisted_geometry import load_geometry

TILE_DIM = 8
TILE_CHANNEL_SIZE = 736


def coarsen_sum(array):
    x,y,z=array.shape
    if any(n%2 for n in (x,y,z)):
        raise ValueError('Tile hierarchy must have even dimensions')
    return array.reshape(x//2,2,y//2,2,z//2,2).sum(axis=(1,3,5))


def dilate(array):
    # x is periodic. Transverse neighbors outside the background box are
    # excluded from physical leaves; native ghost allocations remain extra.
    padded=np.pad(array,((1,1),(1,1),(1,1)),mode='constant')
    padded[0,1:-1,1:-1]=array[-1]
    padded[-1,1:-1,1:-1]=array[0]
    result=np.zeros_like(array,dtype=bool)
    for offset in itertools.product(range(3),repeat=3):
        result|=padded[tuple(slice(d,d+n) for d,n in zip(offset,array.shape))]
    return result


def classify(spec,ny,shift_cells,retain_keys=False):
    h=spec['extent'][1]/ny
    shape=np.rint(np.array(spec['extent'])/h).astype(int)
    if np.any(shape%16) or not np.allclose(shape*h,spec['extent'],rtol=0,atol=1e-14):
        raise ValueError('Forecast requires whole dyadic root tiles')
    nx,ny,nz=shape
    shift=np.asarray(shift_cells)*h
    y=np.arange(ny+1)*h-shift[1]
    z=np.arange(nz+1)*h-shift[2]
    omega=2*np.pi/spec['period']
    def plane(x):
        phase=omega*(x-shift[0])
        return spec['radius']-np.hypot(y[:,None]-spec['center_y']-spec['amplitude']*np.sin(phase),
                                      z[None,:]-spec['center_z']-spec['amplitude']*np.cos(phase))
    fluid=np.zeros(tuple(shape//8),dtype=np.int64)
    walls=np.zeros_like(fluid)
    refined=np.zeros(tuple(shape//16),dtype=bool)
    keys=[];previous=plane(0)
    for i in range(nx):
        following=plane((i+1)*h)
        lo=np.minimum(previous,following);hi=np.maximum(previous,following)
        minimum=np.minimum.reduce((lo[:-1,:-1],lo[1:,:-1],lo[:-1,1:],lo[1:,1:]))
        maximum=np.maximum.reduce((hi[:-1,:-1],hi[1:,:-1],hi[:-1,1:],hi[1:,1:]))
        active=maximum>0;cut=(minimum<0)&active
        fluid[i//8]+=active.reshape(ny//8,8,nz//8,8).sum(axis=(1,3))
        walls[i//8]+=cut.reshape(ny//8,8,nz//8,8).sum(axis=(1,3))
        jj,kk=np.nonzero(cut)
        if retain_keys:keys.append((i*ny+jj)*nz+kk)
        for dx,dy,dz in itertools.product((-2,0,2),repeat=3):
            yq,zq=jj+dy,kk+dz
            valid=(yq>=0)&(yq<ny)&(zq>=0)&(zq<nz)
            refined[((i+dx)%nx)//16,yq[valid]//16,zq[valid]//16]=True
        previous=following
    return {'shape':shape,'fluid':fluid,'walls':walls,'refined':refined,
            'wall_keys':np.concatenate(keys) if retain_keys else None}


def layout(classified,ny,base_ny,krylov_dimension):
    levels=(ny//base_ny).bit_length()-1
    if base_ny*(2**levels)!=ny or levels<1:
        raise ValueError('Base and finest resolutions must form a refined dyadic hierarchy')
    refine=[None]*levels
    refine[-1]=classified['refined']
    for level in range(levels-2,-1,-1):
        refine[level]=coarsen_sum(dilate(refine[level+1]))>0
    fluid=[classified['fluid']];walls=[classified['walls']]
    for _ in range(levels):
        fluid.insert(0,coarsen_sum(fluid[0]));walls.insert(0,coarsen_sum(walls[0]))
    existing=np.ones(refine[0].shape,dtype=bool)
    rows=[]
    for level in range(levels+1):
        split=refine[level] if level<levels else np.zeros_like(existing)
        if np.any(split&~existing):raise ValueError('Refinement target has no ancestor')
        leaves=existing&~split
        if level<levels and np.any(walls[level][leaves]):
            raise ValueError('Coarse leaf contains a predicted cut wall')
        divisor=8**(levels-level)
        if np.any(fluid[level][leaves]%divisor):
            raise ValueError('Coarse leaf does not aggregate complete fluid cells')
        rows.append({'level':level,'ny':base_ny*2**level,'refined_tiles':int(split.sum()),
                     'leaf_tiles':int(leaves.sum()),'fluid_cells':int(fluid[level][leaves].sum()//divisor)})
        existing=np.repeat(np.repeat(np.repeat(split,2,axis=0),2,axis=1),2,axis=2)
    leaves=sum(row['leaf_tiles'] for row in rows);cells=sum(row['fluid_cells'] for row in rows)
    return {'finest_ny':ny,'base_ny':base_ny,'levels':rows,'leaf_tiles':leaves,'fluid_cells':cells,
            'background_leaf_cells':leaves*TILE_DIM**3,
            'one_double_field_leaf_storage_lower_bound_bytes':leaves*TILE_CHANNEL_SIZE*8,
            'packed_fgmres_basis_and_images_bytes':(2*krylov_dimension+1)*cells*8,
            'memory_scope':'Leaf scalar slots plus packed Krylov storage only; excludes native parents/ghosts, AMG, face stencils and CPU matrices/state/history',
            'constructed_native_mesh':False,'flow_accuracy_checked':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--native-state-check',type=Path,required=True)
    parser.add_argument('--resolutions',type=int,nargs='+',default=[128,256,512])
    parser.add_argument('--base-resolutions',type=int,nargs='+',default=[32,64])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or out.drive.lower()!='d:':raise ValueError('Use a fresh D-drive output directory')
    if any(n<16 or n&(n-1) for n in args.resolutions) or any(n<8 or n&(n-1) for n in args.base_resolutions):
        raise ValueError('Use dyadic resolutions of at least 16 and base resolutions of at least 8')
    repo=Path(__file__).resolve().parents[1]
    tile_source=(repo/'src/PoissonTile.h').read_text()
    # The classifier's tile indexing and scalar-slot forecast depend on these
    # actual native constants. Fail explicitly if the layout has changed.
    if not re.search(r'LOG2DIM\s*=\s*3\s*;',tile_source) or not re.search(r'CHNLSIZE\s*=\s*736\s*;',tile_source):
        raise ValueError('Native tile constants changed; update the layout forecast')
    geometry=load_geometry(args.geometry,tables=('walls',))
    proof=json.loads(args.native_state_check.read_text());root=Path(proof['candidate'])
    if not proof['passed'] or proof['validation_mode']!='state_only':raise ValueError('Require a completed native state check')
    case_path=root/'case.json';case=json.loads(case_path.read_text())
    if Path(case['embedded_geometry']).resolve()!=args.geometry.resolve() or case['ny']!=geometry['finest_ny']:
        raise ValueError('Native validation belongs to a different geometry')
    all_hashes={**proof['source_sha256'],**proof['executed_input_sha256']}
    selected=[case_path,root/'mesh_cells.csv',args.geometry.resolve()]
    for path in selected:
        if all_hashes.get(str(path.resolve()))!=sha(path):raise ValueError('Validated native layout input changed: '+str(path))
    source_paths=[Path(__file__),args.geometry,args.native_state_check,case_path,root/'mesh_cells.csv',root/'run.log']
    source_paths += [repo/p for p in ('scripts/twisted_geometry.py','scripts/run_twisted_solver.py',
        'scripts/import_twisted_geometry.py','simple/EmbeddedMesh.cpp','simple/OctreeMesh.cu',
        'src/PoissonTile.h','simple/NativeCompactGpu.cu')]
    source={str(p.resolve()):sha(p) for p in source_paths}
    if geometry['format']=='aphros_cut_geometry_v2':
        binary=args.geometry.parent/geometry['tables']['walls']['file'];source[str(binary.resolve())]=sha(binary)
    actual_ny=geometry['finest_ny'];shift=np.array(geometry.get('reference_translation_cells',[0,0,0]))
    actual=classify(geometry['geometry_spec'],actual_ny,shift,retain_keys=True)
    wall=np.asarray(geometry['walls'])[:,:3].astype(np.int64)
    actual_keys=(wall[:,0]*actual['shape'][1]+wall[:,1])*actual['shape'][2]+wall[:,2]
    if not np.array_equal(np.sort(actual_keys),np.sort(actual['wall_keys'])):
        raise ValueError('Analytic corner classifier differs from actual Aphros cut-cell identities')
    targets=np.argwhere(actual['refined'])
    if not np.array_equal(targets,np.asarray(geometry['refine_root_tiles'])):
        raise ValueError('Forecast refinement mask differs from actual one-level geometry')
    observed_levels=np.loadtxt(root/'mesh_cells.csv',delimiter=',',skiprows=1,usecols=(1,),dtype=np.int64,ndmin=1)
    current=layout(actual,actual_ny,actual_ny//2,20)
    if len(observed_levels)!=current['fluid_cells'] or any(int(np.sum(observed_levels==r['level']))!=r['fluid_cells'] for r in current['levels']):
        raise ValueError('Forecast active leaf counts differ from actual native topology')
    match=re.search(r'Native octree: leafTiles=(\d+) allTiles=(\d+) maxLevel=(\d+) coarseNy=(\d+)',(root/'run.log').read_text())
    if not match or tuple(map(int,(match[1],match[3],match[4])))!=(current['leaf_tiles'],1,actual_ny//2):
        raise ValueError('Forecast background leaves differ from actual native construction')
    out.mkdir(parents=True,exist_ok=False)
    forecast=[]
    for ny in sorted(set(args.resolutions+[actual_ny])):
        if (shift*ny%actual_ny).any():raise ValueError('Requested grid does not preserve the exact geometry translation')
        data=actual if ny==actual_ny else classify(geometry['geometry_spec'],ny,shift*ny//actual_ny)
        for base in sorted(set(args.base_resolutions+[ny//2])):
            if base>ny//2:continue
            row=layout(data,ny,base,20)
            row['predicted_finest_uniform_fluid_cells']=int(data['fluid'].sum())
            row['predicted_cut_wall_cells']=int(data['walls'].sum())
            forecast.append(row)
        print(json.dumps({'finest_ny':ny,'forecasts':[r for r in forecast if r['finest_ny']==ny]}),flush=True)
    if any(sha(Path(p))!=digest for p,digest in source.items()):raise ValueError('Planning input changed')
    savings=[]
    for row in forecast:
        one=next(r for r in forecast if r['finest_ny']==row['finest_ny'] and r['base_ny']==row['finest_ny']//2)
        savings.append({'finest_ny':row['finest_ny'],'base_ny':row['base_ny'],
                        'leaf_tile_fraction_saved':1-row['leaf_tiles']/one['leaf_tiles'],
                        'fluid_cell_fraction_saved':1-row['fluid_cells']/one['fluid_cells']})
    result={'scope':__doc__,'classifier_and_current_layout_verified':True,
            'actual_native_run':str(root),'actual_native_ny':actual_ny,'actual_native_all_tiles':int(match[2]),
            'actual_wall_cells':len(actual_keys),'actual_fluid_cells':len(observed_levels),
            'future_native_meshes_created':False,'flow_accuracy_checked':False,'krylov_dimension':20,
            'balancing':'All 26 neighboring tiles balanced by refining their parents; x periodic, transverse box bounded',
            'tile_dimension':TILE_DIM,'tile_channel_size':TILE_CHANNEL_SIZE,
            'forecasts':forecast,'savings_against_one_refinement_level':savings,
            'source_sha256':source,'goal_complete':False}
    (out/'layout_forecast.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
