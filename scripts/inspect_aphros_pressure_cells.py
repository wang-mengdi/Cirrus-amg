"""Inspect pressure representability at the worst original cut-cell balances.

Local trial pressures hold neighboring pressures fixed and are diagnostics only.
They neither solve the coupled system nor replace any reference field.
"""
import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from check_twisted_mass import read, vector, mass_metrics
from check_aphros_pressure_faces import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aphros',type=Path,required=True)
    parser.add_argument('--face-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.aphros.resolve()
    if args.output.exists():raise ValueError('Preserve earlier reports')
    report=json.loads(args.face_report.read_text())
    if not report['diagnostic_consistent']:raise ValueError('Need a consistent original face capture')
    hashes=dict(report['source_sha256'])
    hashes['proj_final_b0_pressure_rows.csv']=sha(root/'proj_final_b0_pressure_rows.csv')
    if any(sha(root/name)!=value for name,value in hashes.items()):raise ValueError('Inputs differ from face diagnostic')
    case=json.loads((root/'case_manifest.json').read_text())
    cells=read(root/'proj_final_b0_cells.csv');geometry=read(root/'tube_b0_geometry_faces.csv')
    expressions=read(root/'proj_final_b0_pressure_faces.csv');rows=read(root/'proj_final_b0_pressure_rows.csv')
    if any(not np.array_equal(cells[k],rows[k]) for k in ('x','y','z','volume','p')):
        raise ValueError('Pressure equations and final cells differ')
    h=case['spec']['extent'][1]/case['ny'];shape=np.array(case['shape'])
    lookup=np.full(shape,-1,dtype=int)
    keys=np.rint(vector(cells,('x','y','z'))/h-.5).astype(int)
    lookup[tuple(keys.T)]=np.arange(len(cells))
    fkey=vector(geometry,('i','j','k')).astype(int);axis=geometry['axis'].astype(int)
    keep=~((axis==0)&(fkey[:,0]==shape[0]));axis=axis[keep]
    positive=fkey[keep].copy();negative=positive.copy()
    negative[np.arange(len(axis)),axis]-=1;positive[:,0]%=shape[0];negative[:,0]%=shape[0]
    owners=lookup[tuple(negative.T)];neighbors=lookup[tuple(positive.T)]
    if np.any(owners<0) or np.any(neighbors<0):raise ValueError('Missing adjacent fluid cell')
    all_expressions=expressions
    expressions=expressions[keep]
    if not np.array_equal(expressions['p0'],cells['p'][owners]) or not np.array_equal(expressions['p1'],cells['p'][neighbors]):
        raise ValueError('Captured halo pressures do not match adjacent physical pressures')
    # Group signed faces per cell; fsum then removes summation-order ambiguity.
    nf=len(expressions);ids=np.concatenate((owners,neighbors));order=np.argsort(ids,kind='stable')
    edges=np.concatenate((np.arange(nf),np.arange(nf)))[order]
    signs=np.concatenate((np.ones(nf),-np.ones(nf)))[order]
    offsets=np.r_[0,np.cumsum(np.bincount(ids,minlength=len(cells)))]
    net=np.array([math.fsum(float(s*expressions['flux'][j]) for j,s in
                  zip(edges[offsets[c]:offsets[c+1]],signs[offsets[c]:offsets[c+1]])) for c in range(len(cells))])
    rate=report['actual_mass']['normalization_rate_max_speed_over_box_y']
    selected=np.argsort(abs(net)/cells['volume'])[-12:][::-1]
    probes=[]
    with localcontext() as context:
        context.prec=100
        D=lambda v:Decimal.from_float(float(v))
        for c in selected:
            p=float(cells['p'][c]);v=float(cells['volume'][c])
            local=list(zip(edges[offsets[c]:offsets[c+1]],signs[offsets[c]:offsets[c+1]]))
            diagonal=Decimal(0);constant=Decimal(0);details=[]
            for j,sign in local:
                f=expressions[j];is_owner=sign==1
                ac=f['e0'] if is_owner else f['e1'];an=f['e1'] if is_owner else f['e0']
                pn=f['p1'] if is_owner else f['p0']
                diagonal+=D(sign)*D(ac);constant+=D(sign)*(D(an)*D(pn)+D(f['b']))
                exact=D(f['e0'])*D(f['p0'])+D(f['e1'])*D(f['p1'])+D(f['b'])
                details.append({'face_raw':int(f['face_raw']),'outward_sign':int(sign),
                                'e0':float(f['e0']),'e1':float(f['e1']),'b':float(f['b']),
                                'p0':float(f['p0']),'p1':float(f['p1']),'flux':float(f['flux']),
                                'exact_expression':str(exact),'evaluation_rounding':float(D(f['flux'])-exact)})
            target=-constant/diagonal
            def trial(q):
                values=[]
                for j,sign in local:
                    f=expressions[j];p0=q if sign==1 else float(f['p0']);p1=q if sign==-1 else float(f['p1'])
                    values.append(float(sign)*(p0*float(f['e0'])+p1*float(f['e1'])+float(f['b'])))
                return math.fsum(values)
            q=float(target);candidates={q,p};lo=hi=q
            for _ in range(8):
                lo=math.nextafter(lo,-math.inf);hi=math.nextafter(hi,math.inf);candidates.update((lo,hi))
            candidates=sorted(candidates);tested=[(q,trial(q)) for q in candidates]
            best=min(tested,key=lambda t:abs(t[1]));e=rows[c]
            exact_row=D(e['b'])+D(e['a0'])*D(p)
            for k in range(1,7):exact_row+=D(e['a'+str(k)])*D(e['p'+str(k)])
            probes.append({'cell_raw':int(e['cell_raw']),'xyz':[float(cells[k][c]) for k in ('x','y','z')],
                           'volume':v,'pressure':p,'pressure_ulp':math.ulp(p),'actual_flux_net_fsum':float(net[c]),
                           'actual_relative_divergence':float(abs(net[c])/v/rate),
                           'net_flux_limit_at_relative_1e_minus7':v*rate*1e-7,
                           'unrounded_face_net':str(diagonal*D(p)+constant),
                           'exact_stored_matrix_row':str(exact_row),'exact_face_diagonal':str(diagonal),
                           'stored_diagonal_minus_face_diagonal':str(D(e['a0'])-diagonal),
                           'local_exact_pressure_target':str(target),'target_pressure_delta_in_ulps':float((target-D(p))/D(math.ulp(p))),
                           'local_trial_best_pressure':best[0],'local_trial_best_flux_net':best[1],
                           'local_trial_best_relative_divergence':abs(best[1])/v/rate,
                           'trials':[{'pressure':q,'flux_net':b} for q,b in tested],'faces':details})
    # Pressure zero is arbitrary for this periodic tube. Test geometric gauge
    # choices only in memory, retaining the original expressions and RHS.
    gauges={'volume_mean':float(np.average(cells['p'],weights=cells['volume'])),
            'minimum_volume_cell':float(cells['p'][np.argmin(cells['volume'])]),
            'maximum_diagonal_over_volume_cell':float(cells['p'][np.argmax(rows['a0']/cells['volume'])])}
    gauge_trials={}
    raw_faces=read(root/'proj_final_b0_faces.csv')
    for name,shift in gauges.items():
        sample=raw_faces.copy()
        sample['flux']=(all_expressions['p0']-shift)*all_expressions['e0']+(all_expressions['p1']-shift)*all_expressions['e1']+all_expressions['b']
        residual=rows['b']+rows['a0']*(rows['p']-shift)
        for k in range(1,7):residual+=rows['a'+str(k)]*(rows['p'+str(k)]-shift)
        gauge_trials[name]={'pressure_shift':shift,'counterfactual_mass':mass_metrics(cells,sample,geometry,shape,h),
                            'original_row_residual_per_regular_volume':float(np.max(abs(residual))/h**3)}
    result={'scope':__doc__,'diagnostic_consistent':True,'physical_time':report['physical_time'],
            'fsum_actual_mass_relative_linf':float(np.max(abs(net)/cells['volume'])/rate),
            'worst_cells':probes,'counterfactual_global_pressure_gauges':gauge_trials,
            'source_sha256':hashes,'face_report_sha256':sha(args.face_report),
            'checker_sha256':sha(Path(__file__))}
    if any(sha(root/name)!=value for name,value in hashes.items()):raise ValueError('Inputs changed during inspection')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'mass_relative_linf':result['fsum_actual_mass_relative_linf'],
                     'gauge_trials':{k:v['counterfactual_mass']['divergence_relative_linf'] for k,v in gauge_trials.items()},
                     'probes':[{'xyz':p['xyz'],'actual':p['actual_relative_divergence'],
                                'trial_only':p['local_trial_best_relative_divergence'],
                                'target_shift_ulps':p['target_pressure_delta_in_ulps']} for p in probes]}),flush=True)


if __name__=='__main__':main()
