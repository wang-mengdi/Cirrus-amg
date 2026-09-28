"""Reconstruct Aphros cell continuity from final shared Cartesian face fluxes.

Uses indexed geometry, verifies its correspondence with the solution dump,
checks both copies of the periodic seam, and divides by actual cut volumes.
The prescribed stationary impermeable embedded wall contributes zero flux.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open(encoding='utf-8') as stream:
        names=stream.readline().strip().split(',')
        data=np.loadtxt(stream,delimiter=',',dtype=[(name,float) for name in names],ndmin=1)
    if not len(data) or any(not np.isfinite(data[k]).all() for k in names):
        raise ValueError(f'Empty or nonfinite input: {path}')
    return data


def vector(data,names):return np.column_stack([data[k] for k in names])


def mass_metrics(cells,faces,geometry,shape,h):
    shape=np.asarray(shape,dtype=int)
    if len(faces)!=len(geometry):raise ValueError('Geometry/flux face count differs')
    if (not np.array_equal(faces['axis'],geometry['axis']) or
        not np.allclose(vector(faces,['x','y','z']),vector(geometry,['x','y','z']),rtol=0,atol=h*1e-11) or
        not np.allclose(faces['area'],geometry['area'],rtol=1e-12,atol=0)):
        raise ValueError('Indexed geometry and final face dump do not correspond')
    xyz=vector(cells,['x','y','z'])/h-.5
    keys=np.rint(xyz).astype(int)
    if not np.allclose(xyz,keys,rtol=0,atol=1e-10):raise ValueError('Cell center not on reference lattice')
    if np.any(keys<0) or np.any(keys>=shape):raise ValueError('Cell index outside box')
    lookup=np.full(shape,-1,dtype=np.int32)
    lookup[tuple(keys.T)]=np.arange(len(cells))
    if np.count_nonzero(lookup>=0)!=len(cells):raise ValueError('Duplicate fluid cell')
    if np.any(cells['volume']<=0):raise ValueError('Invalid fluid volume')
    fkey=vector(geometry,['i','j','k']).astype(int)
    axis=geometry['axis'].astype(int)
    if np.any((axis<0)|(axis>2)):raise ValueError('Invalid Cartesian axis')
    low=np.flatnonzero((axis==0)&(fkey[:,0]==0))
    high=np.flatnonzero((axis==0)&(fkey[:,0]==shape[0]))
    low=low[np.lexsort((fkey[low,2],fkey[low,1]))]
    high=high[np.lexsort((fkey[high,2],fkey[high,1]))]
    if len(low)!=len(high) or not np.array_equal(fkey[low,1:],fkey[high,1:]):
        raise ValueError('Periodic seam coverage differs')
    seam_delta=float(np.max(np.abs(faces['flux'][low]-faces['flux'][high]),initial=0))
    seam_scale=float(np.max(np.abs(faces['flux'][low]),initial=0))
    if seam_delta>1e-12*max(seam_scale,1e-300):raise ValueError('Periodic seam flux copies differ')
    keep=~((axis==0)&(fkey[:,0]==shape[0]))
    axis=axis[keep];positive=fkey[keep].copy();negative=positive.copy()
    negative[np.arange(len(axis)),axis]-=1
    positive[:,0]%=shape[0];negative[:,0]%=shape[0]
    if any(np.any(q<0) or np.any(q>=shape) for q in (positive,negative)):
        raise ValueError('Open fluid face touches nonperiodic outer box')
    p=lookup[tuple(negative.T)];n=lookup[tuple(positive.T)]
    if np.any(p<0) or np.any(n<0):raise ValueError('Open face touches excluded fluid')
    flux=faces['flux'][keep]
    net=np.zeros(len(cells));absolute=np.zeros(len(cells))
    np.add.at(net,p,flux);np.add.at(net,n,-flux)
    np.add.at(absolute,p,abs(flux));np.add.at(absolute,n,abs(flux))
    divergence=net/cells['volume']
    u=vector(cells,['u','v','w'])
    speed=float(np.linalg.norm(u,axis=1).max())
    rate_scale=speed/(shape[1]*h)
    if rate_scale<=0:raise ValueError('Zero velocity scale')
    q=np.bincount(positive[axis==0,0],weights=flux[axis==0],minlength=shape[0])
    qmean=float(np.mean(q))
    if abs(qmean)<1e-300:raise ValueError('Zero mean axial flow')
    roundoff=8*np.finfo(float).eps*absolute/cells['volume']
    result={'fluid_cells':len(cells),'unique_internal_faces':len(flux),
            'wall_flux_policy':'Prescribed stationary impermeable wall: zero',
            'divergence_units':'1/s; net shared face volume flux divided by actual fluid volume',
            'divergence_linf':float(abs(divergence).max()),
            'divergence_volume_weighted_l2':float(np.sqrt(np.average(divergence**2,weights=cells['volume']))),
            'normalization_rate_max_speed_over_box_y':rate_scale,
            'divergence_relative_linf':float(abs(divergence).max()/rate_scale),
            'roundoff_estimate_linf':float(roundoff.max()),
            'global_signed_flux':float(net.sum()),
            'global_absolute_cell_flux_over_throughflow':float(abs(net).sum()/abs(qmean)),
            'section_flux_mean':qmean,
            'section_flux_relative_spread':float(np.ptp(q)/abs(qmean)),
            'periodic_seam_flux_max_difference':seam_delta}
    # Absolute cut-volume divergence is never waived by a roundoff estimate.
    result['passed']=bool(result['divergence_relative_linf']<1e-7 and
                          result['section_flux_relative_spread']<1e-8 and
                          result['global_absolute_cell_flux_over_throughflow']<1e-8)
    return result


def check_case(root, trust_local_files=False):
    manifest=json.loads((root/'case_manifest.json').read_text(encoding='utf-8'))
    if not trust_local_files and hashlib.sha256((root/'a.conf').read_bytes()).hexdigest()!=manifest['config_sha256']:
        raise ValueError('Baseline config differs from manifest')
    solver=manifest.get('fluid_solver','simple')
    if solver not in ('simple','proj'):raise ValueError('Unsupported baseline fluid solver')
    paths=[root/f'{solver}_final_b0_cells.csv',root/f'{solver}_final_b0_faces.csv',root/'tube_b0_geometry_faces.csv']
    result=mass_metrics(*(read(p) for p in paths),manifest['shape'],manifest['spec']['extent'][1]/manifest['ny'])
    result['source_sha256']={} if trust_local_files else {
        str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result['fluid_solver']=solver
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check_case(args.aphros)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
