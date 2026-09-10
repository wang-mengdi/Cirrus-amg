"""Independent native projection mass balance and indexed reference face transfer."""
import numpy as np
from check_twisted_mass import read
from compare_twisted import error


def check_native(root,reference,cfg,metadata):
    cells=read(root/'solution.csv');faces=read(root/'mesh_faces.csv');values=read(root/'flux.csv')
    if not np.array_equal(cells['id'],np.arange(len(cells))) or not np.array_equal(faces['id'],values['id']):
        raise ValueError('Native projection IDs are not ordered and corresponding')
    q=values['flux'];inner=np.flatnonzero(faces['neighbor']>=0);wall=faces['neighbor']<0
    if np.any(q[wall]!=0):raise ValueError('Native projection wall flux is not exactly zero')
    net=np.zeros(len(cells))
    np.add.at(net,faces['owner'].astype(int),q)
    np.add.at(net,faces['neighbor'][inner].astype(int),-q[inner])
    h=cfg['spec']['extent'][1]/cfg['ny'];shape=np.asarray(cfg['shape']);shift=np.asarray(metadata.get('reference_translation_cells',[0,0,0]))
    r=read(reference/'proj_final_b0_faces.csv');g=read(reference/'tube_b0_geometry_faces.csv')
    if len(r)!=len(g) or not np.array_equal(r['axis'],g['axis']):raise ValueError('Reference face geometry and values differ')
    # Use the same relative infinity norm as check_twisted_mass. Dividing by
    # each near-zero seam flux misclassifies tiny cancellation roundoff.
    seam=(g['axis']==0)&(g['i']==0)
    seam_scale=float(np.max(abs(r['flux'][seam]),initial=0));seam_delta=0.
    lookup={}
    for i in range(len(g)):
        ijk=np.array([g[k][i] for k in ('i','j','k')],dtype=int)+shift;ijk[0]%=shape[0]
        key=(int(g['axis'][i]),*ijk)
        if key in lookup:
            old=lookup[key]
            if key[0]!=0 or key[1]!=int(shift[0]%shape[0]) or {int(g['i'][old]),int(g['i'][i])}!={0,int(shape[0])}:
                raise ValueError('Unexpected duplicate reference face')
            difference=abs(r['flux'][old]-r['flux'][i]);seam_delta=max(seam_delta,difference)
            if difference>1e-12*max(seam_scale,1e-300):raise ValueError('Inconsistent reference seam copies')
        else:lookup[key]=i
    sampled=[];normals=[];weights=[];coarse=0
    for j in inner:
        f=faces[j];p,n=int(f['owner']),int(f['neighbor']);axis=int(f['axis']);hp,hn=cells['h'][p],cells['h'][n]
        center=np.array([f[k] for k in ('x','y','z')]);area=0;flow=0
        if abs(min(hp,hn)-h)<1e-12*h:
            fine=p if hp<=hn else n
            center_cell=np.array([cells[k][fine] for k in ('x','y','z')])
            key=np.rint(center_cell/h-.5).astype(int);key[axis]=round(center[axis]/h);key[0]%=shape[0]
            ids=[lookup[(axis,*key)]]
        elif abs(hp-2*h)<1e-12*h and abs(hn-2*h)<1e-12*h:
            coarse+=1;key=np.rint(center/h-1).astype(int);key[axis]=round(center[axis]/h)
            a,b=(axis+1)%3,(axis+2)%3;ids=[]
            for i in range(2):
                for k in range(2):
                    child=key.copy();child[a]+=i;child[b]+=k;child[0]%=shape[0]
                    ids.append(lookup[(axis,*child)])
        else:raise ValueError('Unexpected native face resolution')
        area=float(np.sum(r['area'][ids]));flow=float(np.sum(r['flux'][ids]))
        if abs(area/f['area']-1)>1e-10:raise ValueError('Reference subfaces do not cover native face aperture')
        sampled.append(flow/area);normals.append(f['sign']*q[j]/area);weights.append(area)
    sections=read(root/'sections.csv');through=float(np.mean(sections['volume_flux']))
    speed=float(np.sqrt(sum(cells[k]**2 for k in 'uvw')).max());rate=speed/cfg['spec']['extent'][1]
    mass={'divergence_linf':float(max(abs(net)/cells['volume'])),
        'divergence_relative_linf':float(max(abs(net)/cells['volume'])/rate),
        'global_absolute_cell_flux_over_throughflow':float(abs(net).sum()/abs(through)),
        'section_flux_relative_spread':float(np.ptp(sections['volume_flux'])/abs(through)),
        'wall_flux_max':float(abs(q[wall]).max(initial=0))}
    mass['passed']=mass['divergence_relative_linf']<1e-7 and mass['global_absolute_cell_flux_over_throughflow']<1e-8 and mass['section_flux_relative_spread']<1e-8
    return {'native_mass':mass,'face_normal_velocity':error(np.asarray(normals),np.asarray(sampled),np.asarray(weights)),
            'reference_seam_flux_relative_linf':seam_delta/max(seam_scale,1e-300),
            'coarse_full_faces_aggregated':coarse,'scope':'Shared conservative flux on every native Cartesian face; coarse faces aggregate four indexed reference apertures'}
