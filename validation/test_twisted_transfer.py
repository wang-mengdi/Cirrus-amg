"""Check cubic reference sampling against analytic polynomials and missing data."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_twisted import transfer

h=.01
points=(np.indices((16,16,16)).reshape(3,-1).T+.5)*h
names=['x','y','z','volume','u','v','w','p']
reference=np.zeros(len(points),dtype=[(name,float) for name in names])
for d,name in enumerate(('x','y','z')):reference[name]=points[:,d]
reference['volume']=h**3
def polynomial(p):
    x,y,z=p.T
    return np.column_stack((x*x+y*z,y**3-3*x*z,x*y*z,x+y*y+z**3))
for d,name in enumerate(('u','v','w','p')):reference[name]=polynomial(points)[:,d]
ours=np.zeros(2,dtype=[(name,float) for name in ('x','y','z','h','volume')])
target=np.array([[7,9,7],[3.5,4.5,5.5]])*h
for d,name in enumerate(('x','y','z')):ours[name]=target[:,d]
ours['h']=[2*h,h];ours['volume']=ours['h']**3
actual,report=transfer(ours,reference,[16,16,16],h)
error=np.max(np.abs(actual-polynomial(target)))
assert error<1e-13,error
key=np.array([6.5,8.5,6.5])*h
missing=reference[np.max(np.abs(points-key),axis=1)>1e-12]
try:transfer(ours,missing,[16,16,16],h)
except ValueError as e:assert 'excluded fluid' in str(e)
else:raise AssertionError('Missing reference stencil was silently accepted')
print(f'PASS: polynomial maximum error={error:.3e}; missing-stencil guard passed')
