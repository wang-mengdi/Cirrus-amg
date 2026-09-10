"""Independent flux-loop, tiny cut volume, and periodic-copy regression checks."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_twisted_mass import mass_metrics

shape=[2,2,2];h=1.
cells=np.zeros(8,dtype=[(s,float) for s in ('x','y','z','volume','u','v','w')])
points=(np.indices(shape).reshape(3,-1).T+.5)*h
for d,name in enumerate(('x','y','z')):cells[name]=points[:,d]
cells['volume']=1.;cells['volume'][0]=1e-8;cells['u']=.25
rows=[]
for i in range(3):
    for j in range(2):
        for k in range(2):
            # Throughflow plus a circulation around a four-cell loop.
            flux=.25+(.125*(1-2*j) if i==1 else 0)
            rows.append((0,i,j,k,i,j+.5,k+.5,1.,flux))
for i in range(2):
    for k in range(2):rows.append((1,i,1,k,i+.5,1,k+.5,1.,.125*(2*i-1)))
g=np.array([r[:-1] for r in rows],dtype=float)
geometry=np.zeros(len(g),dtype=[(s,float) for s in ('axis','i','j','k','x','y','z','area')])
for d,name in enumerate(geometry.dtype.names):geometry[name]=g[:,d]
faces=np.zeros(len(g),dtype=[(s,float) for s in ('x','y','z','axis','area','flux')])
for name in faces.dtype.names:
    faces[name]=[r[-1] for r in rows] if name=='flux' else geometry[name]
result=mass_metrics(cells,faces,geometry,shape,h)
assert result['passed'] and result['divergence_linf']==0,result
broken=faces.copy();broken['flux'][4]+=1e-8
result=mass_metrics(cells,broken,geometry,shape,h)
assert not result['passed'] and result['divergence_linf']>.9,result
broken=faces.copy();broken['flux'][0]+=.01
try:mass_metrics(cells,broken,geometry,shape,h)
except ValueError as e:assert 'seam flux' in str(e)
else:raise AssertionError('Mismatched periodic flux copies accepted')
print('PASS: conservative 3D flux loops; tiny-volume divergence; periodic seam mismatch')
