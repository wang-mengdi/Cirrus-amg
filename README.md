# Matrix-Free Multigrid with Algebraically Consistent Coarsening on Adaptive Octrees

[![arXiv](https://img.shields.io/badge/arXiv-2604.18886-b31b1b.svg)](https://arxiv.org/abs/2604.18886)

This repo hosts the code for our paper *Matrix-Free Multigrid with Algebraically Consistent Coarsening on Adaptive Octrees* that has been submitted to the *Journal of Computational Physics*.

It's developed based on *Cirrus* simulator: 
[![code](https://img.shields.io/badge/Source_Code-Github-blue)](https://github.com/wang-mengdi/Cirrus)


### 定常管流实验交接（中文）

实验分支：`feature/simple-octree-pipe`；截至 2026-09-09（本机时间）。目前等待大内存机器，精度验证尚未全部完成。

**目标与方法。** 保留 Cirrus 自适应八叉树和原生 GPU matrix-free AMG，不使用 flow map，准确求解细管中的黏性流动，重点检查近壁速度和壁面剪切应力（WSS）。目前独立基线是 **Aphros**，尚未与 CFX 实测对比。分支包含 SIMPLE 实现，但当前扭曲管道验证采用 **投影法（projection）+ BCG 对流 + 隐式黏性**，并用 Anderson 加速收敛到定常；压力和黏性求解仍使用原生 GPU 算子及 AMG。

**实验流程。**

1. 先用 CPU double SIMPLE 验证平行板通道和方管，覆盖均匀/自适应网格；通过解析解、Aphros 中间状态 dump、守恒和真残差检查定位并修正壁面黏性闭合、压力梯度及线性收敛判定问题。
2. 转向三维周期扭曲细管：无滑移管壁、体力驱动，壁面附近八叉树加密；使用 Aphros 平面切割几何构造 cut-cell，保留流体体积、面开口和壁面信息，粗细交界共享子面通量。在一致的几何与物理条件下，比较速度、压力、切割单元速度、WSS 和截面流量。Aphros 参考保留原有流动离散方程，另加精度增强线性后端、诊断输出及存储优化。
3. 检查定常加速与普通时间推进是否得到同一解，再做网格加密和近壁采样检查。下表中的 64/128/256 表示最细横向分辨率，不是全域均匀网格尺寸。

| 检查 | 结果与结论 |
|---|---|
| 早期 SIMPLE 直管验证 | 11 组验证及三组细化检查通过；细八叉树速度/流量解析误差为 0.0741%/0.1068%。适用范围是这些标准层流算例，详见 [SIMPLE 结果](validation/SIMPLE_RESULTS.md)。 |
| Cirrus 64 定常解 vs 独立 Aphros 64 | 五项既定误差门槛均通过，说明当前实现可在该分辨率上与基线对齐。 |
| Cirrus 128 定常加速 vs 普通时间推进到定常 | 通过，同一网格上的场量相对差约 `1e-9` 或更小。 |
| 64→128 空间收敛 | **未通过**：流量差 0.6063%（门槛 0.5%）；最近壁速度差 2.6509%（门槛 1%）；原采样 WSS 差 1.9673%（门槛 1%）。 |
| 独立 Aphros 128 / Cirrus 256 | **未完成**：前者因内存保护退出，仅完成两个物理步；后者完成几何构建和 93 次 GPU 线性求解，但没有可用于验收的 256 定常场。 |

**现有问题。** 标准投影法与现有八叉树/GPU AMG 的结合已经可行，但粗网格基线对齐和残差收敛不能证明近壁结果已达到网格无关精度。目前仍需解决或确认近壁空间误差、极小切割面引起的 WSS 采样敏感性，以及大算例的内存和计算开销。面积加权壁面拟合仅是诊断候选，尚未替换正式验证器；不能据此宣称整体通过，也不能宣称已达到 CFX 等效精度。

**接手顺序。** 在大内存机器上先构建并通过小算例检查，再完成独立 Aphros 128 和 Cirrus 256，保持原门槛重新检查 128→256 的流量、速度、压力、近壁速度和 WSS，并复查采样/时间步敏感性。旧机器上的暂停进程不是可迁移检查点，256 需从头启动。完整命令、资源建议、输入恢复和 ParaView 查看方法见 [大机器重现实验说明](validation/twisted/REPRODUCE_ON_LARGE_MACHINE.md)；当前重现流程已验证的平台为 Windows x64。

**交付约定。** Git 仅保存代码、脚本、小型算例配置和文档；输入、结果、dump、日志、图及压缩包保存在本地 `D:/CirrusExperiments/cirrus-amg`，通过 U 盘等介质单独转移。只拉取 Git 仓库不会获得全部重现数据。

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

The `.vti` and `.vtu` output files are written directly by the simulator and do
not require the VTK SDK at build or run time.

You can use `Paraview` for visualization. If it's installed, render with the following script:

    $ pvpython --force-offscreen-rendering .\scripts\pararender.py .\output\sphere_circling\ --slice 0:401 --name vorticity --outline --mask-non-finest --mesh .\scenes\sphere0.2r.ply

Rendered images will be saved to `output/sphere_circling/render_vorticity`.

    $ ffmpeg -framerate 25 -i output/sphere_circling/render_vorticity/frame.%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4