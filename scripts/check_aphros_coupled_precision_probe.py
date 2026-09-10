"""Independently check extended pressure against all captured face equations with Decimal."""
import argparse
import csv
from decimal import Decimal,localcontext
import json
from pathlib import Path
import struct
import numpy as np
from run_twisted_solver import sha


def hex_decimal(text):
    sign=-1 if text.startswith('-') else 1
    mantissa,exponent=text.lstrip('+-').lower().split('p');mantissa=mantissa[2:]
    integer,_,fraction=mantissa.partition('.')
    return Decimal(sign*int(integer+fraction,16))*Decimal(2)**(int(exponent)-4*len(fraction))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.probe.resolve()
    if args.output.exists():raise ValueError('Preserve prior checks')
    manifest=json.loads((root/'manifest.json').read_text());report=json.loads((root/'coupled_report.json').read_text())
    if manifest['compile_exit_code'] or manifest['run_exit_code'] or not manifest['source_unchanged']:
        raise ValueError('Probe did not complete with unchanged sources')
    hashes={str(path):sha(path) for path in root.iterdir() if path.is_file()}
    if sha(root/'input.bin')!=manifest['input_sha256'] or sha(root/'pressure_probe.exe')!=manifest['executable_sha256']:
        raise ValueError('Probe input or executable changed')
    with (root/'input.bin').open('rb') as f:
        n,nf,rate=struct.unpack('<qqd',f.read(24));cells=np.fromfile(f,dtype='<f8',count=n*2).reshape(n,2)
        faces=np.fromfile(f,dtype=[('a','<i4'),('c','<i4'),('e0','<f8'),('e1','<f8'),('b','<f8')])
    if len(faces)!=nf:raise ValueError('Face count differs')
    with (root/'coupled_pressure.csv').open() as f:rows=list(csv.DictReader(f))
    if [int(r['id']) for r in rows]!=list(range(n)):raise ValueError('Cell ordering differs')
    with localcontext() as ctx:
        ctx.prec=100
        pressure=[hex_decimal(r['pressure_hex']) for r in rows]
        net=[Decimal(0) for _ in range(n)]
        D=lambda x:Decimal.from_float(float(x))
        for f in faces:
            a,c=int(f['a']),int(f['c'])
            q=(pressure[a]*D(f['e0'])+pressure[c]*D(f['e1']))+D(f['b'])
            net[a]+=q;net[c]-=q
        divergence=[abs(q)/D(v)/D(rate) for q,v in zip(net,cells[:,1])]
        worst=max(range(n),key=divergence.__getitem__);maximum=divergence[worst]
        global_net=sum(net)
        # Independently read the solver's actually rounded arithmetic result,
        # while Decimal above tests its extended pressure in exact expressions.
        rounded=max(Decimal(r['relative_divergence']) for r in rows)
        total_volume=sum(D(v) for v in cells[:,1])
        mean=sum((p-D(old))*D(v) for p,(old,v) in zip(pressure,cells))/total_volume
        change=max(abs(p-D(old)-mean) for p,(old,v) in zip(pressure,cells))
        passed=report['passed'] and report['mantissa_bits']==64 and maximum<Decimal('1e-7') and rounded<Decimal('1e-7')
        result={'passed':bool(passed),'scope':__doc__+' Offline coupled pressure replay; no independent flow trajectory or steady alignment claim.',
                'cells':n,'faces':nf,'decimal_precision':100,'decimal_relative_divergence_linf':str(maximum),
                'rounded_extended_relative_divergence_linf':str(rounded),'exact_expression_global_net':str(global_net),
                'worst_cell':worst,'pressure_change_modulo_gauge_linf':str(change),
                'source_sha256':hashes,'checker_sha256':sha(Path(__file__))}
    if any(sha(Path(name))!=value for name,value in hashes.items()):raise ValueError('Probe changed during verification')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
