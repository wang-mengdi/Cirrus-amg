"""Independent small-grid projection audit using dumped native octree operators.

Implements the original Aphros Proj ordering, BCG advection and implicit raw
diffusion on a uniform cut grid. This is a diagnostic prototype, not the native
production solver or independent baseline, and cannot certify wall accuracy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from audit_twisted_diffusion_spectrum import load_operators
from check_twisted_mass import read
from run_twisted_solver import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operators',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bcg-reference',type=Path,help='Compare only the original prescribed-field BCG operator audit')
    parser.add_argument('--stationary-state',type=Path,help='Read an existing prototype step and evaluate its steady projection momentum equation')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    executed_source=args.output/'executed_source.py'
    executed_source.write_bytes(Path(__file__).read_bytes())
    meta=json.loads((args.baseline/'case_manifest.json').read_text())
    if meta['fluid_solver']!='proj' or meta['momentum_mode']!='imp' or not meta['convection']:
        raise ValueError('Expected implicit-diffusion projection NS reference')
    par=meta['projection_parameters']
    if par!={'bcg':1,'redistr_adv':0,'diffusion_iters':1,'diffusion_consistent_guess':1}:
        raise ValueError('Only the original default BCG projection parameters are supported')
    op,cells,faces=load_operators(args.operators)
    n=len(cells);nf=len(faces);h=cells['h'][0];V=cells['volume']
    if n>40000 or not np.all(cells['h']==h):raise ValueError('Diagnostic requires a bounded uniform grid')
    if not np.array_equal(cells['id'],np.arange(n)) or not np.array_equal(faces['id'],np.arange(nf)):
        raise ValueError('Expected ordered native mesh IDs')
    spec=meta['spec'];rho=spec['rho'];mu=rho*spec['nu'];force=np.asarray(spec['force'])
    if abs(h-spec['extent'][1]/meta['ny'])>1e-15:raise ValueError('Reference spacing differs')
    native_case=json.loads((args.operators/'case.json').read_text())
    if native_case['rho']!=rho or native_case['nu']!=spec['nu'] or native_case['force']!=spec['force']:
        raise ValueError('Native physical parameters differ')
    geometry=read(args.baseline/'tube_b0_geometry_cells.csv')
    def ordered(table):return table[np.lexsort((table['z'],table['y'],table['x']))]
    ca,cb=ordered(cells),ordered(geometry)
    if len(ca)!=len(cb) or any(not np.allclose(ca[k],cb[k],rtol=1e-13,atol=1e-16) for k in ('x','y','z','volume')):
        raise ValueError('Reference and native cut geometries differ')
    dt=meta['time_step'];tol=meta['iteration_tolerance']
    inner=np.flatnonzero(faces['neighbor']>=0);wall=np.flatnonzero(faces['neighbor']<0)
    p=faces['owner'][inner].astype(int);q=faces['neighbor'][inner].astype(int);axis=faces['axis'][inner].astype(int)
    area=faces['area'];I=op['interpolation'];G=[op['cell_gradient_'+str(d)] for d in range(3)]
    if np.any(faces['sign'][inner]!=1):raise ValueError('Expected positive Cartesian orientation')
    if np.max(abs((I@np.ones(n))[inner]-1))>1e-12:raise ValueError('Constant material interpolation needs explicit boundary handling')
    B=sparse.coo_matrix((np.r_[np.ones(len(inner)),-np.ones(len(inner))],
        (np.r_[p,q],np.r_[inner,inner])),shape=(n,nf)).tocsr()
    area_vectors=np.zeros((nf,3));area_vectors[inner,axis]=area[inner]
    normal=-(B@area_vectors)[faces['owner'][wall].astype(int)]/area[wall,None]
    if np.max(abs(np.linalg.norm(normal,axis=1)-1))>1e-10:
        raise ValueError('Wall area-vector closure failed')
    W=sparse.coo_matrix((np.ones(len(wall)),(faces['owner'][wall].astype(int),wall)),shape=(n,nf)).tocsr()
    N=sparse.coo_matrix((np.r_[-1/faces['distance'][inner],1/faces['distance'][inner]],
        (np.r_[inner,inner],np.r_[p,q])),shape=(nf,n)).tocsr()
    pressure_matrix=(-B@sparse.diags(area)@N).tocsc()
    coo=pressure_matrix.tocoo();keep=(coo.row!=0)&(coo.col!=0)
    fixed=sparse.coo_matrix((np.r_[coo.data[keep],1],(np.r_[coo.row[keep],0],np.r_[coo.col[keep],0])),shape=(n,n)).tocsc()
    pressure_lu=splu(fixed)
    compact=pressure_matrix+sparse.diags(np.bincount(faces['owner'][wall].astype(int),weights=2*area[wall]/h,minlength=n))
    full=-B@sparse.diags(area)@op['face_gradient']-W@sparse.diags(area)@op['wall_gradient']
    deferred=mu*(full-compact)
    mass=rho*V/dt
    diffusion_lu=splu((mu*compact+sparse.diags(mass)).tocsc())
    adjacent=np.full((n,3,2),-1,dtype=int)
    adjacent[p,axis,1]=inner;adjacent[q,axis,0]=inner
    # Aphros initializes is_boundary=false, then flags cut faces by iterating
    # eb.SuFaces(), which excludes excluded faces. The missing-face sentinel
    # therefore stays unflagged, with zero gradient, flux and area. Treating it
    # as a flagged boundary changes BCG even when all gradients already match.
    unflagged=np.zeros(nf+1,dtype=bool);unflagged[-1]=True
    unflagged[inner]=abs(area[inner]-h*h)<1e-12*h*h
    areas=np.r_[area,0]
    def project(flux,step):
        rhs=-rho/step*(B@flux);rhs[0]=0
        pressure=pressure_lu.solve(rhs)
        corrected=flux-step/rho*area*(N@pressure)
        return pressure,corrected
    def velocity_flux(velocity):
        result=np.zeros(nf);sample=I@velocity
        result[inner]=area[inner]*sample[inner,axis]
        return result
    def acceleration(pressure):return force-np.column_stack([g@pressure for g in G])/rho
    faceforce=np.zeros(nf);faceforce[inner]=force[axis]*rho
    def bcg(velocity,flux,source):
        gradient=np.vstack((op['face_gradient']@velocity,np.zeros((1,3))))
        flux_ext=np.r_[flux,0]
        sign=np.where(flux[inner]>0,1.,-1.)
        up=np.where(sign>0,p,q)
        fm=adjacent[up,axis,0];fp=adjacent[up,axis,1]
        slope=np.where((unflagged[fm]&unflagged[fp])[:,None],.5*(gradient[fm]+gradient[fp]),gradient[inner])
        temporal=.5*(source[p]+source[q])-slope*(flux[inner]/area[inner])[:,None]
        for offset in (1,2):
            tang=(axis+offset)%3;fm=adjacent[up,tang,0];fp=adjacent[up,tang,1]
            valid=unflagged[fm]&unflagged[fp]
            speed=np.zeros(len(inner));speed[valid]=(flux_ext[fm[valid]]+flux_ext[fp[valid]])/(areas[fm[valid]]+areas[fp[valid]])
            temporal-=gradient[np.where(speed>0,fm,fp)]*speed[:,None]
        result=np.zeros((nf,3))
        result[inner]=velocity[up]+.5*h*sign[:,None]*slope+.5*dt*temporal
        return result
    if args.bcg_reference:
        from scipy.spatial import cKDTree
        reference_cells=read(args.bcg_reference/'bcg_cells.csv')
        reference_faces=read(args.bcg_reference/'bcg_faces.csv')
        def xyz(data):return np.column_stack([data[k] for k in ('x','y','z')])
        distance,indices=cKDTree(xyz(reference_cells)).query(xyz(cells))
        if max(distance)>h*1e-10:raise ValueError('BCG reference cell coordinates differ')
        values=np.repeat(reference_cells['value'][indices,None],3,axis=1)
        sources=np.repeat(reference_cells['source'][indices,None],3,axis=1)
        flux=np.zeros(nf);reference_gradient=np.zeros(nf);reference_value=np.zeros(nf)
        for d in range(3):
            rows=inner[axis==d];refs=np.flatnonzero(reference_faces['axis']==d)
            x,y=xyz(faces)[rows],xyz(reference_faces)[refs]
            x[:,0]%=spec['extent'][0];y[:,0]%=spec['extent'][0]
            distance,indices=cKDTree(y).query(x)
            if max(distance)>h*1e-10:raise ValueError('BCG reference face coordinates differ')
            refs=refs[indices]
            if not np.allclose(area[rows],reference_faces['area'][refs],rtol=1e-12,atol=0):
                raise ValueError('BCG reference aperture differs')
            flux[rows]=reference_faces['flux'][refs]
            reference_gradient[rows]=reference_faces['gradient'][refs]
            reference_value[rows]=reference_faces['value'][refs]
        computed=bcg(values,flux,sources)[:,0]
        gradient=op['face_gradient']@values[:,0]
        gradient_error=float(max(abs(gradient[inner]-reference_gradient[inner])))
        value_error=float(max(abs(computed[inner]-reference_value[inner])))
        report={'scope':'Prescribed nonzero-field BCG operator audit, not a flow solution',
            'gradient_absolute_max':gradient_error,'bcg_absolute_max':value_error,
            'passed':gradient_error<1e-10 and value_error<1e-10}
        np.savetxt(args.output/'bcg_comparison.csv',np.column_stack([faces[k][inner] for k in ('id','x','y','z','axis')]+[computed[inner],reference_value[inner],gradient[inner],reference_gradient[inner]]),
            delimiter=',',header='id,x,y,z,axis,bcg,reference_bcg,gradient,reference_gradient',comments='',fmt='%.17g')
        report['source_sha256']={str(p.resolve()):sha(p) for p in [executed_source,args.bcg_reference/'bcg_cells.csv',args.bcg_reference/'bcg_faces.csv']}
        (args.output/'bcg_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        if not report['passed']:raise SystemExit(1)
        return
    if args.stationary_state:
        state=read(args.stationary_state/'solution.csv');saved_flux=read(args.stationary_state/'flux.csv')
        if len(state)!=n or any(not np.array_equal(state[k],cells[k]) for k in ('id','x','y','z','h','volume')):
            raise ValueError('Stationarity state mesh differs')
        if len(saved_flux)!=nf or not np.array_equal(saved_flux['id'],faces['id']):raise ValueError('Stationarity flux topology differs')
        velocity=np.column_stack([state[k] for k in 'uvw']);pressure=state['p'];flux=saved_flux['flux']
        accel=acceleration(pressure)
        predicted=bcg(velocity,flux,accel)
        predicted_flux=np.zeros(nf);predicted_flux[inner]=predicted[inner,axis]*area[inner]
        _,predicted_flux=project(predicted_flux,dt*.5)
        advected=bcg(velocity,predicted_flux,accel)
        advective=op['redistribution']@(-B@(predicted_flux[:,None]*advected))
        residual=accel+advective/V[:,None]-mu*(full@velocity)/(rho*V[:,None])
        l2=float(np.sqrt(np.average(np.sum(residual**2,axis=1),weights=V)))
        maximum=float(np.linalg.norm(residual,axis=1).max())
        np.savetxt(args.output/'steady_momentum.csv',np.column_stack([cells[k] for k in ('id','x','y','z','volume')]+[residual[:,d] for d in range(3)]),
            delimiter=',',header='id,x,y,z,volume,residual_x,residual_y,residual_z',comments='',fmt='%.17g')
        sources=[executed_source,args.stationary_state/'solution.csv',args.stationary_state/'flux.csv',args.baseline/'case_manifest.json']+list((args.operators/'operators').glob('*'))
        report={'scope':'Steady momentum of the original projection spatial operators at the specified dt; not time-step or grid convergence',
            'time_step':dt,'momentum_acceleration_volume_l2':l2,'momentum_acceleration_max':maximum,
            'continuity_linf':float(np.max(abs(B@flux)/V)),'momentum_gate_passed':l2<1e-8,
            'source_sha256':{str(p.resolve()):sha(p) for p in sources if p.is_file()}}
        (args.output/'stationarity.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
        if not report['momentum_gate_passed']:raise SystemExit(1)
        return
    u=np.zeros((n,3));pressure=np.zeros(n);flux=np.zeros(nf);guess=u.copy();history=[]
    converged=False
    for step in range(1,meta['time_steps']+1):
        old=u.copy();oldflux=flux.copy()
        for iteration in range(1,5001):
            previous=u.copy()
            if step==1:pressure,flux=project(velocity_flux(u)+faceforce*area*dt/rho,dt)
            accel=acceleration(pressure)
            predicted=bcg(old,oldflux,accel)
            predicted_flux=np.zeros(nf);predicted_flux[inner]=predicted[inner,axis]*area[inner]
            _,predicted_flux=project(predicted_flux,dt*.5)
            advected=bcg(old,predicted_flux,accel)
            advection=op['redistribution']@(-B@(predicted_flux[:,None]*advected))
            intermediate=old+dt*advection/V[:,None]+dt*accel
            diffused=diffusion_lu.solve(mass[:,None]*intermediate-deferred@guess)
            guess=diffused.copy()
            intermediate=diffused-dt*accel
            pressure,flux=project(velocity_flux(intermediate)+faceforce*area*dt/rho,dt)
            u=intermediate+dt*acceleration(pressure)
            error=float(np.max(abs(u-previous)))
            if not np.isfinite(u).all():raise ValueError('Nonfinite projection audit')
            if iteration in (1,2,10) or iteration%100==0:
                print(json.dumps({'step':step,'iteration':iteration,'difference':error}),flush=True)
            if error<tol:break
        else:raise ValueError('Projection audit did not converge within a physical step')
        folder=args.output/f'step_{step:04d}';folder.mkdir()
        gauge=pressure-np.average(pressure,weights=V)
        np.savetxt(folder/'solution.csv',np.column_stack([cells[k] for k in ('id','x','y','z','h','volume')]+[u[:,0],u[:,1],u[:,2],gauge]),
            delimiter=',',header='id,x,y,z,h,volume,u,v,w,p',comments='',fmt='%.17g')
        np.savetxt(folder/'flux.csv',np.column_stack((faces['id'],flux)),delimiter=',',header='id,flux',comments='',fmt='%.17g')
        derivative=op['wall_gradient']@u
        shear=mu*(derivative[wall]-normal*np.sum(normal*derivative[wall],axis=1)[:,None])
        np.savetxt(folder/'walls.csv',np.column_stack([faces[k][wall] for k in ('id','x','y','z','area')]+[shear[:,0],shear[:,1],shear[:,2]]),
            delimiter=',',header='face_id,x,y,z,area,tau_x,tau_y,tau_z',comments='',fmt='%.17g')
        history.append({'step':step,'time':step*dt,'iterations':iteration,'difference':error,
            'temporal_acceleration_l2':float(np.sqrt(np.average(np.sum(((u-old)/dt)**2,axis=1),weights=V))),
            'continuity_linf':float(np.max(abs(B@flux)/V)),'speed_max':float(np.linalg.norm(u,axis=1).max())})
        print(json.dumps(history[-1]),flush=True)
        (args.output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
    converged=True
    sources=[executed_source,args.baseline/'case_manifest.json',args.baseline/'a.conf',args.baseline/'tube_b0_geometry_cells.csv',args.operators/'case.json']+list((args.operators/'operators').glob('*'))
    report={'scope':__doc__,'completed_requested_transient':converged,'history':history,
        'source_sha256':{str(p.resolve()):sha(p) for p in sources if p.is_file()}}
    (args.output/'projection_audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
