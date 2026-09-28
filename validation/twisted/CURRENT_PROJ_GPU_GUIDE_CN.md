# 当前扭曲细管 Proj + GPU 路线：中文代码导读

更新：2026-09-23。本文只描述这次在新电脑上实际构建和运行的
`simple_channel` 扭曲细管配置，以及截至本日已经取得的验证证据。
`validation/twisted/IMPLEMENTATION.md` 按研发时间记录了早期 CPU 原型和后续 GPU
接入；阅读那些历史段落时，不能把中间状态当成当前运行方式。

## 1. 这次究竟运行了什么

- 代码基线：`448284e22a54a506b68d2d90c37b67eea091e1c0`。本次迁移未修改
  `simple/`、`src/`、`xmake.lua` 或 `validation/twisted/reproduction/native128.json`
  中的求解算法。经用户同意，构建脚本固定选用 VS 2022；输入校验脚本增加了
  按 Git 提交核对 tracked 源码的选项。这些修改不改变流动离散。
- 目标：`xmake.lua` 中的 `simple_channel`，编译 `simple/*.cpp` 和 `simple/*.cu`，
  并依赖原有 `src` 目标。名字中的 `simple` 是目标和目录的历史名称；这次配置
  `fluid_solver=proj`，不运行早期的 SIMPLE 流动算法。
- 实际运行配置：`validation/twisted/reproduction/native128.json`。它选择
  `adaptive=true`、BCG 对流、隐式黏性、`linear_backend=native_gpu`、完整压力和
  黏性 GPU 算子、`gpu_preconditioner=native_amg`、两次正交化以及
  `steady_anderson_depth=5`。物理参数与精度门槛见该 JSON 和
  `REPRODUCE_ON_LARGE_MACHINE.md`。
- 运行时的 `projection_method.json` 报告后端为 `native_gpu_fgmres_amg`、
  五层原生 AMG，且 `host_pressure_matrices=not_assembled`。此次成功的 128 检查
  记录了 3170 次 GPU 线性求解调用；不能将这次运行称为 CPU 压力解。

## 2. 代码如何连接

| 入口 | 当前职责 |
| --- | --- |
| `simple/main.cpp` | 读取 JSON；按 `fluid_solver=proj` 创建 `ProjectionSolver`，运行后按收敛状态返回。 |
| `simple/OctreeMesh.cu`、`simple/EmbeddedMesh.cpp` | 使用原生 HA 八叉树和导入的切割几何建立单元、共享面与粗细交界。没有展开成全域最细均匀网格。 |
| `simple/QuadraticReconstruction.cpp`、`simple/EmbeddedOperators.cpp`、`simple/ProjectionAssembly.h` | 构造近壁与粗细交界所需的重构、离散模板和主机端辅助数据。 |
| `simple/ProjectionSolver.cpp` | 执行 Proj 流动推进、BCG 输运、隐式黏性/压力调用、内外层迭代、守恒及诊断；选择实际线性后端。 |
| `simple/NativeCompactGpu.cu`、`simple/NativeAmgPreconditioner.cu` | 在 GPU 上应用完整压力/黏性算子并执行 FGMRES 与原生 AMG 预条件。 |
| `simple/AndersonAcceleration.cpp` | 提供迭代加速所用的历史/组合逻辑；定常输出 `iterate_####` 是伪时间外迭代，不是物理时间步。 |

这是 **CPU/GPU 混合实现**：压力和隐式黏性线性求解使用原生 GPU 算子与 AMG；
几何、模板构造、输运、部分延迟修正和诊断仍在 CPU。它不是“整个时间推进
都在 GPU 上”的版本。GPU 压力算子不组装主机端全局压力矩阵，但主机端仍有
几何、面、重构与辅助存储。详细演进见 `IMPLEMENTATION.md`，当前实际选择
应以配置和每次输出的 `projection_method.json` 为准。

本次运行使用了文档已有的 `SIMPLE_NATIVE_HOST_TILES_FILE=1` 和
`SIMPLE_ANDERSON_FILE_HISTORY=1`：分别将部分主机 Tile 和 Anderson 历史放在
实验盘。这是存储路径选择；两个变量的 `SIMPLE` 前缀不代表运行 SIMPLE 算法。
参见 `NATIVE_HOST_STORAGE.md` 和 `STEADY_OUTER_ANDERSON.md`。

## 3. “128”代表什么；本次验证了什么

这里的 128 是最细横向分辨率，**不是全域均匀 `128³` 网格**。当前八叉树在壁面
附近加密。128 实际包含 790216 个流体单元、2431944 个共享/边界面和
49152 个粗细交界面。原生 Tile 网格也保存未必属于流体的单元位置。

本机重新构建的 `simple_channel.exe` 对 128 配置正常退出；第 23 次定常外迭代
达到程序门槛。`check_twisted_steady_iteration.py --state-only` 通过原方程、
守恒、原生拓扑和 GPU 线性残差检查。峰值工作集约 2.08 GiB，峰值私有提交量
约 3.79 GiB；两者不是同一个内存指标，也不等于显存占用。运行与检查原始记录
分别在 `D:/CirrusExperiments/cirrus-amg/runs/reproduction_native128_v1` 和
`D:/CirrusExperiments/cirrus-amg/checks/reproduction_native128_state_v1.json`。

**这还不能证明 128 与 Aphros 对齐。** 已通过的跨程序定常对比是当前 Proj 路线
的 64 对 64。早期同网格的原生 128 定常加速与普通时间推进场量对比约为
`1e-9` 或更小，它只验证 Cirrus 内部两种推进方式的一致性。独立 Aphros Proj
128 定常参考在旧 32 GiB 机器上触发内存保护，只完成两个物理步，尚无可验收
的最终定常场。本机这次没有运行 Aphros 128，也没有生成 128 对 128 的通过报告。

另外，64→128 的空间收敛检查尚未通过既定近壁门槛。这与“同一分辨率下是否
对齐 Aphros”是不同检查，不能互相代替。

要回答 128 是否对齐，须先完成独立 Aphros **Proj** 128 的定常运行并验证它的
真实收敛、守恒和输入来源，再在相同物理参数与动量时间步下运行
`scripts/compare_twisted_steady_aphros.py` 的直接定常比较。速度、压力、切割
单元速度、壁面剪切、截面流量及守恒均需过原门槛；不得用这次 Cirrus 的
`state_only` 通过报告替代 Aphros 对比。

## 4. 继续阅读

- `HANDOFF_CN.md`、`validation/twisted/REPRODUCE_ON_LARGE_MACHINE.md`：当前任务、
  物理模型、未完成项及重现顺序。
- `validation/twisted/IMPLEMENTATION.md`：按时间记录的推导、代码修改和实验，
  其中早期 CPU/PCG 章节是历史状态。
- `validation/twisted/STEADY_APHROS_COMPARISON.md`：已完成的 64 对 64 直接
  定常比较与验收定义。
- `validation/twisted/NATIVE128_STEADY_COMPLETION.md`：旧机 128 原生定常完成
  证据；仍未声称独立 Aphros 128 对齐。
