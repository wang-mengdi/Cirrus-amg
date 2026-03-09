# Cirrus: Adaptive Hybrid Particle-Grid Flow Maps on GPU

**Visual Studio**

    $ python makesln.py

Running args: `.\scenes\smokesphere.json`

**xmake**

    $ xmake -v
    $ xmake r cirrus tests


### sl_cutcell
cutcell，流体外插速度场对流，有点类似于IB。
用semi-lagrangian每步对流。
结论是r=1的球或者圆柱都无法在后面算出来尾涡，可能是边界条件太好了速度场过于smooth。

### impulse_cutcell
cutcell解，若干步impulse对流。