# Cirrus: Adaptive Hybrid Particle-Grid Flow Maps on GPU

### Compilation

Run:

    $ xmake

(Alternative) create `Cirrus-amg\build\vsxmake2022\Cirrus-amg.sln` solution file for Visual Studio:

    $ python makesln.py

### Run Tests

### Run Simulations

Use `.json` file in `scenes` folder as the argument. Sphere, tie fighter, delta wing simulations are as follows:

    $ xmake r cirrus_cutcell .\scenes\sphere_circling.json
    $ xmake r cirrus_cutcell .\scenes\tie_fighter.json
    $ xmake r cirrus_cutcell .\scenes\delta_wing.json

Modify the `.json` file for parameters like total number of frames. The object trajectories are programmed to last for 4s or 400 frames in 100FPS.

The simulator will write results under `./output/` folder. 

You can use `Paraview` for visualization. If it's installed, render with the following script:

    $ pvpython --force-offscreen-rendering .\scripts\pararender.py .\output\sphere_circling\ --slice 0:401 --name vorticity --outline --mask-non-finest --mesh .\scenes\sphere0.2r.ply

Rendered images will be saved to `output/sphere_circling/render_vorticity`.

**xmake**




### sl_cutcell
cutcell，流体外插速度场对流，有点类似于IB。
用semi-lagrangian每步对流，对流速度是外插的fluid velocity。
结论是r=1的球或者圆柱都无法在后面算出来尾涡，可能是边界条件太好了速度场过于smooth。

### impulse_cutcell
cutcell解，若干步impulse对流。
对流速度是mix velocity（流体和固体加权）
事实证明，在cutcell的情况下，只有mix velocity能搓出涡，否则太平滑了没有涡。

### cirrus_cutcell
使用cirrus的混合对流格式，但使用cutcell求解。
机头-z, 上方是x