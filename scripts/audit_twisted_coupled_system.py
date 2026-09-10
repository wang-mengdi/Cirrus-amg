"""Solve the dumped uniform exp-mode equations as a coupled diagnostic system.

This does not advance SIMPLE or replace the independent Aphros baseline. Its
transient solution can be checked against completed SIMPLE runs; a stationary
algebraic solution alone is not evidence of temporal stability or grid accuracy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from audit_twisted_diffusion_spectrum import load_operators
from run_twisted_solver import sha
from check_twisted_mass import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operators',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='Completed same-step native SIMPLE solution CSV')
    parser.add_argument('--previous',type=Path,help='Previous physical-step velocity CSV; otherwise starts from zero')
    parser.add_argument('--steady',action='store_true',help='Omit physical time derivative, retain specified Rhie-Chow diagonal')
    parser.add_argument('--stokes',action='store_true',help='Omit advection in this diagnostic')
    parser.add_argument('--unredistributed-diffusion',action='store_true',
                        help='Diagnostic control: assemble raw face/wall diffusion fluxes, without redistribution; not an Aphros run')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    executed_source=args.output/'executed_source.py'
    executed_source.write_bytes(Path(__file__).read_bytes())
    config=json.loads(args.case.read_text());ops,cells,faces=load_operators(args.operators)
    n=len(cells);nf=len(faces);V=cells['volume'];rho=config['rho'];nu=config['nu'];force=np.array(config['force'])
    if config.get('adaptive') or config.get('momentum_mode')!='exp' or config.get('flux_relaxation_memory',False):
        raise ValueError('Diagnostic currently requires uniform exp and no flux memory')
    if not np.all(cells['h']==cells['h'][0]):raise ValueError('Expected uniform cell sizes')
    if not np.array_equal(cells['id'],np.arange(n)) or not np.array_equal(faces['id'],np.arange(nf)):
        raise ValueError('Unordered mesh IDs')
    previous=np.zeros((n,3));sources=[args.case,executed_source]
    if args.previous:
        previous_rows=read(args.previous)
        if len(previous_rows)!=n or any(not np.array_equal(previous_rows[d],cells[d]) for d in ('x','y','z')):
            raise ValueError('Previous physical-step mesh differs')
        previous=np.column_stack([previous_rows[d] for d in 'uvw']);sources.append(args.previous)
    inner=np.where(faces['neighbor']>=0)[0];owner=faces['owner'][inner].astype(int);neighbor=faces['neighbor'][inner].astype(int)
    B=sparse.coo_matrix((np.r_[np.ones(len(inner)),-np.ones(len(inner))],(np.r_[owner,neighbor],np.r_[inner,inner])),shape=(n,nf)).tocsr()
    mean=sparse.coo_matrix((np.full(2*len(inner),.5),(np.r_[inner,inner],np.r_[owner,neighbor])),shape=(nf,n)).tocsr()
    normal=sparse.coo_matrix((np.r_[-1/faces['distance'][inner],1/faces['distance'][inner]],
        (np.r_[inner,inner],np.r_[owner,neighbor])),shape=(nf,n)).tocsr()
    gamma=config['alpha_u']*config['time_step']/rho
    df=ops['interpolation']@np.full(n,gamma)
    G=[ops['cell_gradient_'+str(d)] for d in range(3)]
    flux_u=[];wide=sparse.csr_matrix((nf,n));balanced=np.zeros(nf)
    for d in range(3):
        selector=np.where((faces['neighbor']>=0)&(faces['axis']==d),faces['area']*faces['sign'],0.)
        flux_u.append(sparse.diags(selector)@ops['interpolation'])
        wide+=gamma*sparse.diags(selector)@mean@G[d]
        balanced+=selector*rho*force[d]*(df-gamma)
    flux_p=wide-sparse.diags(faces['area']*df)@normal
    C=[B@f for f in flux_u];P=B@flux_p
    diffusion=rho*nu*(ops['compact_diffusion']+ops['deferred_diffusion'])
    operator_control=None
    if args.unredistributed_diffusion:
        wall=np.where(faces['neighbor']<0)[0]
        W=sparse.coo_matrix((np.ones(len(wall)),(faces['owner'][wall].astype(int),wall)),shape=(n,nf)).tocsr()
        full=-B@sparse.diags(faces['area'])@ops['face_gradient']-W@sparse.diags(faces['area'])@ops['wall_gradient']
        reconstructed=ops['redistribution']@full
        expected=ops['compact_diffusion']+ops['deferred_diffusion']
        delta=reconstructed-expected
        norm=float(np.asarray(abs(expected).sum(axis=1)).max())
        reconstruction_error=float(np.asarray(abs(delta).sum(axis=1)).max()/norm)
        if reconstruction_error>1e-12:raise ValueError('Raw diffusion flux reconstruction does not reproduce dumped exp operator')
        volume_ratio=(ops['redistribution']@V)/V
        operator_control={'scope':'Raw diffusion only; other coupled equations remain the SIMPLE exp fixed point',
            'redistributed_reconstruction_relative_infinity_norm':reconstruction_error,
            'redistributed_constant_volume_source_ratio_min':float(volume_ratio.min()),
            'redistributed_constant_volume_source_ratio_max':float(volume_ratio.max())}
        diffusion=rho*nu*full
    mass=np.zeros(n) if args.steady else rho*V/config['time_step']
    rhs=np.concatenate([V*rho*force[d]+mass*previous[:,d] for d in range(3)]+[-B@balanced])
    q=np.zeros(nf);u=previous.copy();p=np.zeros(n);history=[];converged=False
    R=ops['redistribution'];mix=ops['advection_face_interpolation']
    scale=np.r_[1/(rho*V),1/(rho*V),1/(rho*V),1/V]
    def advection(flux):
        picks=np.where(flux[inner]>=0,owner,neighbor)
        pick=sparse.coo_matrix((np.ones(len(inner)),(inner,picks)),shape=(nf,n)).tocsr()
        return R@B@sparse.diags(rho*flux)@mix@pick
    for iteration in range(100):
        adv=sparse.csr_matrix((n,n)) if args.stokes else advection(q)
        H=diffusion+adv+sparse.diags(mass)
        A=sparse.bmat([[H,None,None,sparse.diags(V)@G[0]],
                      [None,H,None,sparse.diags(V)@G[1]],
                      [None,None,H,sparse.diags(V)@G[2]],
                      [C[0],C[1],C[2],P]],format='csr')
        scaled=(sparse.diags(scale)@A).tocsc();b=rhs*scale
        gauge=3*n
        coo=scaled.tocoo();keep=(coo.row!=gauge)&(coo.col!=gauge)
        fixed=sparse.coo_matrix((np.r_[coo.data[keep],1.],(np.r_[coo.row[keep],gauge],np.r_[coo.col[keep],gauge])),shape=scaled.shape).tocsc()
        b[gauge]=0
        result=spsolve(fixed,b)
        if not np.isfinite(result).all():raise ValueError('Nonfinite coupled diagnostic solution')
        u=result[:3*n].reshape(3,n).T;p=result[3*n:]
        q=sum((flux_u[d]@u[:,d] for d in range(3)),balanced.copy())+flux_p@p
        actual_adv=sparse.csr_matrix((n,n)) if args.stokes else advection(q)
        momentum=np.column_stack([(diffusion+actual_adv+sparse.diags(mass))@u[:,d]+V*(G[d]@p)-rhs[d*n:(d+1)*n] for d in range(3)])
        acceleration=momentum/(rho*V[:,None])
        residual=float(np.sqrt(np.sum(V*np.sum(acceleration*acceleration,axis=1))/V.sum()))
        continuity=float(np.max(abs(B@q)/V))
        history.append({'iteration':iteration+1,'momentum_acceleration_l2':residual,'continuity_linf':continuity,'speed_max':float(np.linalg.norm(u,axis=1).max())})
        print(json.dumps(history[-1]),flush=True)
        if residual<1e-9 and continuity<1e-8:converged=True;break
    p-=np.average(p,weights=V)
    np.savetxt(args.output/'diagnostic_solution.csv',np.column_stack([cells[d] for d in ('id','x','y','z','h','volume')]+[u[:,0],u[:,1],u[:,2],p]),
        delimiter=',',header='id,x,y,z,h,volume,u,v,w,p',comments='',fmt='%.17g')
    np.savetxt(args.output/'diagnostic_flux.csv',np.column_stack((faces['id'],q)),delimiter=',',header='id,flux',comments='',fmt='%.17g')
    report={'scope':__doc__,'converged_algebraic_equations':converged,'steady_equations':args.steady,'stokes':args.stokes,'history':history,'rhie_chow_reciprocal_diagonal':gamma,
            'unredistributed_diffusion_control':operator_control}
    if args.reference:
        ref=read(args.reference)
        if len(ref)!=n or any(not np.array_equal(ref[d],cells[d]) for d in ('x','y','z')):raise ValueError('Reference mesh differs')
        refu=np.column_stack([ref[d] for d in 'uvw']);refp=ref['p']-np.average(ref['p'],weights=V)
        report['reference_velocity_relative_l2']=float(np.sqrt(np.sum(V*np.sum((u-refu)**2,axis=1))/np.sum(V*np.sum(refu**2,axis=1))))
        report['reference_pressure_relative_l2']=float(np.sqrt(np.sum(V*(p-refp)**2)/np.sum(V*refp**2)))
        sources.append(args.reference)
    report['source_sha256']={str(p.resolve()):sha(p) for p in sources+list((args.operators/'operators').glob('*')) if p.is_file()}
    (args.output/'coupled_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if not converged:raise SystemExit(1)


if __name__=='__main__':main()
