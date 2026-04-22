# Matrix-Free Multigrid with Algebraically Consistent Coarsening on Adaptive Octrees

[![arXiv](https://img.shields.io/badge/arXiv-2604.18886-b31b1b.svg)](https://arxiv.org/abs/2604.18886)

This repo hosts the code for our paper *Matrix-Free Multigrid with Algebraically Consistent Coarsening on Adaptive Octrees* that has been submitted to the *Journal of Computational Physics*.

It's developed based on *Cirrus* simulator: 
[![code](https://img.shields.io/badge/Source_Code-Github-blue)](https://github.com/wang-mengdi/Cirrus)


### Build Environment

The project uses [xmake](https://xmake.io) as its build system. All dependencies (except CUDA) are fetched and built automatically by xmake.

#### Prerequisites

- **xmake** >= 3.0
- **CUDA toolkit** (system install)
- C++17 compiler: MSVC on Windows, GCC on Linux

#### Tested Configurations

| | Windows | Linux |
|---|---|---|
| **OS** | Windows x64 | Ubuntu 24.04 (x86_64) |
| **Compiler** | MSVC 14.50 | GCC 13.3.0 |
| **CUDA** | 12.9 | 12.0 |
| **xmake** | 3.0.5 | 3.0.8 |

#### Dependency Versions (via xmake)

| Package | Windows | Linux |
|---|---|---|
| Eigen | 5.0.0 | 5.0.1 |
| VTK | 9.5.1 | 9.5.1 |
| fmt | 12.1.0 | 12.1.0 |
| nlohmann_json | v3.12.0 | v3.12.0 |
| polyscope | v2.5.0 | v2.5.0 |
| glm | 0.9.9+8 | 0.9.9+8 |
| tbb | 2022.1.0 | 2022.1.0 (system) |
| libigl | v2.6.0 | v2.6.0 |
| magic_enum | v0.9.7 | v0.9.7 |

### Compilation

Configure and build (xmake fetches all packages automatically):

    $ xmake f -y
    $ xmake

To force a full rebuild:

    $ xmake f -c -y
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