# Matrix-Free GPU Implementation of Algebraic Multigrid Poisson Solver on Adaptive Octree with Irregular Domains

This repo hosts the code for our adaptive AMG solver.

It's developed based on *Cirrus* simulator: 
[![code](https://img.shields.io/badge/Source_Code-Github-blue)](https://github.com/wang-mengdi/Cirrus)


### Compilation

Run:

    $ xmake

(Alternative) create `Cirrus-amg\build\vsxmake2022\Cirrus-amg.sln` solution file for Visual Studio:

    $ python makesln.py

### Run Numerical Tests

    $ xmake r tests

See tests/main.cpp for more details.

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