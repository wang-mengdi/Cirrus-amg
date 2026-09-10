"""Audit captured original pressure face expressions without modifying any field.

Alternative arithmetic is evaluated only in memory and labelled counterfactual.
Its continuity is not a baseline success or a new converged flow solution.
"""
import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from check_twisted_mass import read, mass_metrics


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.aphros.resolve()
    if args.output.exists():raise ValueError('Choose a fresh report')
    names=('case_manifest.json','a.conf','run_manifest.json','run_completion.json',
           'proj_final_b0_cells.csv','proj_final_b0_faces.csv','tube_b0_geometry_faces.csv',
           'proj_final_b0_pressure_faces.csv','proj_final_b0_pressure_faces_snapshot.json',
           'proj_final_b0_pressure_snapshot.json')
    hashes={name:sha(root/name) for name in names}
    get=lambda name:json.loads((root/name).read_text(encoding='utf-8-sig'))
    case=get('case_manifest.json');run=get('run_manifest.json')
    if get('run_completion.json')['exit_code']:raise ValueError('Reference run failed')
    if case['fluid_solver']!='proj' or case['config_sha256']!=hashes['a.conf'] or run['config_sha256']!=hashes['a.conf']:
        raise ValueError('Wrong solver or changed reference configuration')
    if sha(Path(run['executable']))!=run['executable_sha256']:
        raise ValueError('Reference executable changed')
    if run['environment'].get('APHROS_TWISTED_CAPTURE_PRESSURE_FACES')!='1':
        raise ValueError('Face capture was not enabled')
    captured=read(root/'proj_final_b0_pressure_faces.csv')
    cells=read(root/'proj_final_b0_cells.csv');faces=read(root/'proj_final_b0_faces.csv')
    geometry=read(root/'tube_b0_geometry_faces.csv')
    meta=get('proj_final_b0_pressure_faces_snapshot.json');pressure_meta=get('proj_final_b0_pressure_snapshot.json')
    identity=(len(captured)==len(faces)==meta['open_faces'] and
              meta['physical_time']==pressure_meta['physical_time'] and
              meta['pressure_calls']==pressure_meta['pressure_calls'] and
              abs(meta['physical_time']-case['time_step']*case['time_steps'])<1e-12 and
              all(np.array_equal(captured[k],faces[k]) for k in ('x','y','z','axis','area','flux')))
    if not identity:raise ValueError('Capture does not describe the final face field')
    h=case['spec']['extent'][1]/case['ny']
    def mass(q):
        sample=faces.copy();sample['flux']=q
        return mass_metrics(cells,sample,geometry,case['shape'],h)
    ordinary=captured['p0']*captured['e0']+captured['p1']*captured['e1']+captured['b']
    opposite=np.array_equal(captured['e0'],-captured['e1'])
    if not opposite:raise ValueError('This diagnostic requires opposite internal-face coefficients')
    difference=captured['e1']*(captured['p1']-captured['p0'])+captured['b']
    exact=np.empty(len(captured))
    with localcontext() as context:
        context.prec=100
        for i,row in enumerate(captured):
            d={k:Decimal.from_float(float(row[k])) for k in ('e0','e1','b','p0','p1')}
            exact[i]=float(d['e0']*d['p0']+d['e1']*d['p1']+d['b'])
    actual=captured['flux']
    result={'scope':__doc__,'diagnostic_consistent':bool(identity and np.array_equal(ordinary,actual)),
            'original_expression_bitwise_reproduced':bool(np.array_equal(ordinary,actual)),
            'original_expression_max_flux_difference':float(np.max(abs(ordinary-actual))),
            'opposite_pressure_coefficients':opposite,'physical_time':meta['physical_time'],
            'actual_mass':mass(actual),'counterfactual_arithmetic_only':{
                'pressure_difference':{'mass':mass(difference),'changed_faces':int(np.count_nonzero(difference!=actual)),
                                       'max_flux_change':float(np.max(abs(difference-actual)))},
                'decimal_products_rounded_once':{'mass':mass(exact),'changed_faces':int(np.count_nonzero(exact!=actual)),
                                               'max_flux_change':float(np.max(abs(exact-actual)))}}
            ,'source_sha256':hashes,'reference_executable_sha256':run['executable_sha256']}
    if hashes!={name:sha(root/name) for name in names}:raise ValueError('Inputs changed during diagnostic')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if not result['diagnostic_consistent']:raise SystemExit(1)


if __name__=='__main__':main()
