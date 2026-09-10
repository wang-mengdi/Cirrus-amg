"""Reconstruct continuity from the actual stored hexadecimal Aphros face fluxes.

Decimal arithmetic prevents the verifier from truncating extended scalars back
to double. This checks the stored flux, not a recomputed pressure expression.
The physical conservation thresholds are the same as check_twisted_mass.py.
"""
import argparse
import csv
from decimal import Decimal, localcontext
import json
from pathlib import Path
import numpy as np
from check_twisted_mass import read
from check_aphros_coupled_precision_probe import hex_decimal
from run_twisted_solver import sha


def calculate(root):
    cfg=json.loads((root/'case_manifest.json').read_text())
    meta=json.loads((root/'proj_final_b0_exact.json').read_text())
    if meta['mantissa_bits'] not in (53,64):raise ValueError('Unsupported recorded scalar precision')
    if cfg['fluid_solver']!='proj':raise ValueError('Exact dump requires the Proj driver')
    shape=tuple(cfg['shape']);h=cfg['spec']['extent'][1]/cfg['ny']
    geometry=read(root/'tube_b0_geometry_cells.csv');gfaces=read(root/'tube_b0_geometry_faces.csv')
    cells=read(root/'proj_final_b0_cells.csv');faces=read(root/'proj_final_b0_faces.csv')
    paths=[root/name for name in ('case_manifest.json','a.conf','proj_final_b0_exact.json',
        'proj_final_b0_exact_cells.csv','proj_final_b0_exact_faces.csv','tube_b0_geometry_cells.csv',
        'tube_b0_geometry_faces.csv','proj_final_b0_cells.csv','proj_final_b0_faces.csv')]
    hashes={str(p.resolve()):sha(p) for p in paths}
    if sha(root/'a.conf')!=cfg['config_sha256']:raise ValueError('Configuration changed')
    if meta['cells']!=len(cells) or len(cells)!=len(geometry) or meta['faces']!=len(faces) or len(faces)!=len(gfaces):
        raise ValueError('Exact and ordinary dump sizes differ')
    if not np.allclose(np.column_stack([cells[d] for d in 'xyz']),
                       np.column_stack([geometry[d] for d in 'xyz']),rtol=0,atol=h*1e-12):
        raise ValueError('Final cells differ from indexed geometry')
    for d in ('x','y','z','area','axis'):
        if not np.allclose(faces[d],gfaces[d],rtol=1e-12,atol=0):raise ValueError('Final faces differ from indexed geometry')
    # The existing decimal output deliberately remains unchanged. Its 17
    # significant digits can round an extended scalar to a neighboring double.
    def ordinary_matches(exact,ordinary):
        if abs(float(exact)-ordinary)>2*np.finfo(float).eps*max(abs(ordinary),1e-300):
            raise ValueError('Exact and ordinary final fields do not correspond')
    with localcontext() as ctx:
        ctx.prec=100
        physical_time=hex_decimal(meta['physical_time_hex'])
        expected_time=Decimal.from_float(float(cfg['time_step']))*cfg['time_steps']
        if abs(physical_time-expected_time)>abs(expected_time)*Decimal('1e-12'):
            raise ValueError('Exact dump is not at the final prescribed physical time')
        D=lambda value:Decimal.from_float(float(value))
        index={};volumes=[];net=[];speed2=Decimal(0)
        with (root/'proj_final_b0_exact_cells.csv').open() as f:
            count=0
            for i,row in enumerate(csv.DictReader(f)):
                if i>=len(cells):raise ValueError('Extra exact cell')
                key=tuple(int(row[d]) for d in ('i','j','k'))
                if key!=tuple(int(geometry[d][i]) for d in ('i','j','k')) or key in index:
                    raise ValueError('Exact cell index or order differs')
                if any(k<0 or k>=n for k,n in zip(key,shape)):raise ValueError('Cell outside box')
                volume=hex_decimal(row['volume_hex'])
                if volume<=0 or volume!=D(geometry['volume'][i]):raise ValueError('Original cut volume changed')
                ordinary_matches(volume,cells['volume'][i])
                velocity=[hex_decimal(row[d+'_hex']) for d in 'uvw']
                for d,value in zip('uvw',velocity):ordinary_matches(value,cells[d][i])
                ordinary_matches(hex_decimal(row['p_hex']),cells['p'][i])
                speed2=max(speed2,sum(v*v for v in velocity))
                index[key]=i;volumes.append(volume);net.append(Decimal(0));count+=1
        if count!=len(cells):raise ValueError('Truncated exact cells')
        sections=[Decimal(0) for _ in range(shape[0])];seam={};unique=0;keys=set()
        with (root/'proj_final_b0_exact_faces.csv').open() as f:
            count=0
            for i,row in enumerate(csv.DictReader(f)):
                if i>=len(faces):raise ValueError('Extra exact face')
                axis=int(row['axis']);key=tuple(int(row[d]) for d in ('i','j','k'))
                if (axis,*key)!=tuple(int(gfaces[d][i]) for d in ('axis','i','j','k')):
                    raise ValueError('Exact face index or order differs')
                if axis not in (0,1,2) or (axis,*key) in keys:raise ValueError('Invalid or duplicate face')
                keys.add((axis,*key));count+=1
                area=hex_decimal(row['area_hex']);q=hex_decimal(row['flux_hex'])
                if area!=D(gfaces['area'][i]):raise ValueError('Original face area changed')
                ordinary_matches(q,faces['flux'][i])
                if axis==0 and key[0] in (0,shape[0]):
                    seam.setdefault(key[1:],{})[key[0]]=q
                if axis==0 and key[0]==shape[0]:continue
                positive=list(key);negative=list(key);negative[axis]-=1
                positive[0]%=shape[0];negative[0]%=shape[0]
                p=index.get(tuple(negative));n=index.get(tuple(positive))
                if p is None or n is None:raise ValueError('Open face touches excluded fluid')
                net[p]+=q;net[n]-=q;unique+=1
                if axis==0:sections[positive[0]]+=q
        if count!=len(faces):raise ValueError('Truncated exact faces')
        if not seam or any(set(s)!={0,shape[0]} for s in seam.values()):raise ValueError('Periodic seam coverage differs')
        seam_delta=max(abs(s[0]-s[shape[0]]) for s in seam.values())
        seam_scale=max(abs(s[0]) for s in seam.values())
        seam_relative=seam_delta/max(seam_scale,Decimal('1e-300'))
        if seam_relative>Decimal('1e-12'):raise ValueError('Periodic seam copies exceed the established relative tolerance')
        # The original code evaluates the two seam copies independently. Also
        # check the balance with the opposite copy, so choosing a seam cannot
        # conceal a large divergence in a tiny boundary cut cell.
        other=net.copy()
        for (j,k),s in seam.items():
            delta=s[shape[0]]-s[0]
            other[index[(shape[0]-1,j,k)]]+=delta;other[index[(0,j,k)]]-=delta
        rate=speed2.sqrt()/D(cfg['spec']['extent'][1]);mean=sum(sections)/len(sections)
        if not rate or not mean:raise ValueError('Zero flow scale')
        divergence=[abs(q)/v for q,v in zip(net,volumes)]
        other_relative=max(abs(q)/v for q,v in zip(other,volumes))/rate
        worst=max(range(len(net)),key=divergence.__getitem__)
        relative=divergence[worst]/rate
        spread=(max(sections)-min(sections))/abs(mean)
        total=sum(abs(q) for q in net)/abs(mean)
        result={'passed':max(relative,other_relative)<Decimal('1e-7') and spread<Decimal('1e-8') and total<Decimal('1e-8'),
            'scope':__doc__,'mantissa_bits':meta['mantissa_bits'],'decimal_precision':ctx.prec,
            'physical_time':str(physical_time),
            'fluid_cells':len(net),'unique_internal_faces':unique,'worst_cell_index':worst,
            'divergence_linf':str(divergence[worst]),'divergence_relative_linf':str(relative),
            'section_flux_mean':str(mean),'section_flux_relative_spread':str(spread),
            'global_absolute_cell_flux_over_throughflow':str(total),'global_signed_flux':str(sum(net)),
            'periodic_seam_exactly_equal':seam_delta==0,'periodic_seam_relative_difference':str(seam_relative),
            'opposite_seam_divergence_relative_linf':str(other_relative),'wall_flux_policy':'Stationary impermeable wall: zero',
            'source_sha256':hashes,'checker_sha256':sha(Path(__file__))}
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('Inputs changed during exact mass inspection')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous exact mass checks')
    completion=json.loads((args.aphros/'run_completion.json').read_text(encoding='utf-8-sig'))
    if completion['exit_code'] or 'End of simulation' not in (args.aphros/'run.log').read_text():
        raise ValueError('Require a successfully completed reference flow run')
    result=calculate(args.aphros)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
