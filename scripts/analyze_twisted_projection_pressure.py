"""Isolate the dt-dependent collocated flux correction at a steady projection state.

The momentum source is reconstructed from a completed native state and frozen.
Changing dt then changes only the pressure/face-flux constraint in a coupled
linear solve. This is a mechanism diagnostic, never an independent CFD baseline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from audit_twisted_diffusion_spectrum import load_operators
from compare_twisted import read,vector,error
from run_twisted_solver import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operators',type=Path,required=True)
    p.add_argument('--states',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=args.output/'executed_source.py';source.write_bytes(Path(__file__).read_bytes())
    ops,cells,faces=load_operators(args.operators);n=len(cells);nf=len(faces);V=cells['volume']
    if n>4000 or not np.all(cells['h']==cells['h'][0]):raise ValueError('Bounded uniform small-grid diagnostic only')
    inner=np.flatnonzero(faces['neighbor']>=0);wall=np.flatnonzero(faces['neighbor']<0)
    owner=faces['owner'][inner].astype(int);neighbor=faces['neighbor'][inner].astype(int)
    B=sparse.coo_matrix((np.r_[np.ones(len(inner)),-np.ones(len(inner))],(np.r_[owner,neighbor],np.r_[inner,inner])),shape=(n,nf)).tocsr()
    W=sparse.coo_matrix((np.ones(len(wall)),(faces['owner'][wall].astype(int),wall)),shape=(n,nf)).tocsr()
    N=sparse.coo_matrix((np.r_[-1/faces['distance'][inner],1/faces['distance'][inner]],(np.r_[inner,inner],np.r_[owner,neighbor])),shape=(nf,n)).tocsr()
    I=ops['interpolation'];G=[ops['cell_gradient_'+str(d)] for d in range(3)]
    if max(abs((I@np.ones(n))[inner]-1))>1e-12:raise ValueError('Requires constant material interpolation')
    A=sparse.diags(faces['area'])
    D=-B@A@ops['face_gradient']-W@A@ops['wall_gradient']
    F=[sparse.diags(np.where((faces['neighbor']>=0)&(faces['axis']==d),faces['area'],0.))@I for d in range(3)]
    C=[B@f for f in F];dtP=sum((F[d]@G[d] for d in range(3)),-A@N)
    data=[];paths=[source]+list((args.operators/'operators').glob('*'))+[args.operators/'mesh_cells.csv',args.operators/'mesh_faces.csv']
    for root in args.states:
        cfg=json.loads((root/'case.json').read_text());metrics=json.loads((root/'metrics.json').read_text())
        c=read(root/'solution.csv');q=read(root/'flux.csv')
        if cfg.get('fluid_solver')!='proj' or not metrics['converged'] or max(metrics['steady_momentum_relative_l2'],metrics['temporal_acceleration_relative_l2'])>=1e-8:
            raise ValueError('Require steady native projection states')
        if any(not np.array_equal(c[k],cells[k]) for k in ('id','x','y','z','volume')) or not np.array_equal(q['id'],faces['id']):raise ValueError('Geometry or ordering differs')
        u=vector(c,'uvw');pressure=c['p']-np.average(c['p'],weights=V);rho=cfg['rho'];dt=cfg['time_step']
        predicted=sum((F[d]@u[:,d] for d in range(3)),np.zeros(nf))+dt/rho*(dtP@pressure)
        identity=error(predicted[inner]/faces['area'][inner],q['flux'][inner]/faces['area'][inner],faces['area'][inner])
        data.append((cfg,u,pressure,identity));paths += [root/f for f in ('case.json','metrics.json','solution.csv','flux.csv')]
    cfg,u0,p0,_=data[0];rho=cfg['rho'];H=rho*cfg['nu']*D
    source_force=[H@u0[:,d]+V*(G[d]@p0) for d in range(3)]
    rhs=np.r_[*source_force,np.zeros(n)];scale=np.r_[1/(rho*V),1/(rho*V),1/(rho*V),1/V]
    gauge=3*n+int(np.argmax(V));rows=[];solved=[]
    for index,(case,u,pres,identity) in enumerate(data):
        if any(case[k]!=cfg[k] for k in ('rho','nu','force','embedded_geometry','convection_scheme')):raise ValueError('Different physical or spatial problem')
        dt=case['time_step']
        matrix=sparse.bmat([[H,None,None,sparse.diags(V)@G[0]],[None,H,None,sparse.diags(V)@G[1]],
                           [None,None,H,sparse.diags(V)@G[2]],[C[0],C[1],C[2],dt/rho*(B@dtP)]],format='csr')
        scaled=(sparse.diags(scale)@matrix).tocsc();b=rhs*scale
        coo=scaled.tocoo();keep=(coo.row!=gauge)&(coo.col!=gauge)
        fixed=sparse.coo_matrix((np.r_[coo.data[keep],1.],(np.r_[coo.row[keep],gauge],np.r_[coo.col[keep],gauge])),shape=scaled.shape).tocsc()
        b[gauge]=0;solver=splu(fixed);x=solver.solve(b)
        for _ in range(3):x+=solver.solve(b-fixed@x)
        if not np.isfinite(x).all():raise ValueError('Nonfinite coupled solution')
        velocity=x[:3*n].reshape(3,n).T;pressure=x[3*n:];pressure-=np.average(pressure,weights=V)
        solved.append(pressure)
        defect=(matrix@np.r_[velocity.T.ravel(),pressure]-rhs)*scale
        row={'dt':dt,'observed_flux_identity':identity,'frozen_source_velocity_vs_observed':error(velocity,u,V),
             'frozen_source_pressure_vs_observed':error(pressure,pres,V),
             'observed_pressure_l2':float(np.sqrt(np.average(pres**2,weights=V))),
             'coupled_momentum_l2':float(np.sqrt(np.average(np.sum(defect[:3*n].reshape(3,n).T**2,axis=1),weights=V))),
             'coupled_divergence_linf':float(max(abs(defect[3*n:])))}
        if index:
            measured=pres-p0;predicted=pressure-solved[0]
            row['pressure_change_explained_by_flux_constraint']=error(predicted,measured,V)
            row['pressure_change_weighted_cosine']=float(np.sum(V*predicted*measured)/np.sqrt(np.sum(V*predicted**2)*np.sum(V*measured**2)))
        rows.append(row)
        np.savetxt(args.output/f'pressure_{index}.csv',np.column_stack((cells['id'],pres,pressure)),delimiter=',',header='id,observed,frozen_momentum_source',comments='',fmt='%.17g')
        print(json.dumps(row),flush=True)
    report={'scope':__doc__,'rows':rows,'source_sha256':{str(p.resolve()):sha(p) for p in paths if p.is_file()}}
    (args.output/'pressure_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
