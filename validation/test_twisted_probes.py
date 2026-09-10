"""Probe interpolation tests use analytic fields, including the periodic seam."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_twisted_refinement import sample

errors=[]
for n in (16,32):
    # Hold the local stencil phase fixed when measuring interpolation order.
    points=np.array([[.1/n,.5+.2/n,.5-.1/n],[1-.1/n,.5+.2/n,.5-.1/n],
                     [.5+.1/n,.5+.2/n,.5-.1/n]])
    centers=(np.indices((n,n,n)).reshape(3,-1).T+.5)/n
    def value(x):
        a,b,c=x.T
        return np.column_stack((np.sin(2*np.pi*a),b*b+b*c+c*c,np.ones(len(a))))
    result,_=sample(centers,value(centers),points,1.,32)
    assert abs(result[:,1:]-value(points)[:,1:]).max()<1e-12
    errors.append(float(abs(result[:,0]-value(points)[:,0]).max()))
assert errors[0]/errors[1]>3,errors
yz=(np.indices((16,16)).reshape(2,-1).T+.5)/16
centers=np.column_stack((np.full(len(yz),.5),yz))
values=np.column_stack((yz[:,0]**2,yz[:,0]*yz[:,1],yz[:,1]**2))
wallpoints=np.array([[.5,.3,.6],[.5,.6,.4]])
answer,_=sample(centers,values,wallpoints,1.,32,np.tile([1.,0.,0.],(2,1)))
exact=np.column_stack((wallpoints[:,1]**2,wallpoints[:,1]*wallpoints[:,2],wallpoints[:,2]**2))
assert abs(answer-exact).max()<1e-12
print(f'PASS: quadratic volume/surface fields; periodic sine errors {errors}')
