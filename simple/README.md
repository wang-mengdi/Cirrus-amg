# Native-octree SIMPLE channel solver

2026-09-10：[大内存机器重现与当前验收状态](../validation/twisted/REPRODUCE_ON_LARGE_MACHINE.md)。

这个目标在现有 `HADeviceGrid<Tile>` 八叉树上实现不可压缩、常物性 Newtonian 流体求解，不使用 flow map。本文主体说明早期直通道和方管的稳态 SIMPLE 路径；曲管另有 `fluid_solver: "proj"` 的标准 BCG/隐式黏性投影时间推进，以及可选的定常迭代加速。

下文介绍的直通道路径使用 CUDA 创建网格、执行原生 tile refinement 和写回结果，使用 **CPU double / Eigen** 求解。曲管的 `linear_backend: "native_gpu"` 已接入原生 GPU FGMRES/AMG 压力和黏性算子，配置与验证见相应曲管文档。`OctreeMesh.cu` 从真实 native leaf holder 提取控制体，没有把 adaptive 网格展开成最细均匀网格。因此运行这个目标仍需要可用的 CUDA GPU。

本文说明实现和复现方法，不记录一次运行的成功结论或误差数字。实际验收结果单独记录在 `validation/SIMPLE_RESULTS.md`，应同时检查对应 `suite_results.json`、算例配置和可执行文件记录。

曲管切割单元的新增实验另见 [实现说明](../validation/twisted/IMPLEMENTATION.md)。
它从 Aphros 导入共享几何，并在原生八叉树上独立求解。投影模式的 GPU 路径及
时间步验证见 [64 网格时间步报告](../validation/twisted/TIMESTEP64_COMPLETION.md)，
定常迭代的新选项见 [外层 Anderson 实现与验证](../validation/twisted/STEADY_OUTER_ANDERSON.md)。
[32→64 近壁空间收敛](../validation/twisted/SPATIAL_CONVERGENCE_32_64.md) 尚未通过，不能把求解器一致性当作网格精度结论。

## 构建与单例运行

以下命令在仓库根目录、Windows PowerShell 中执行。沿用根目录 README 的 xmake、C++17/MSVC、CUDA 和依赖配置：

```powershell
xmake f -m release -y
xmake build -j4 simple_channel
xmake run simple_channel scenes/simple_uniform8.json
```

也可以直接运行构建出的程序：

```powershell
& .\build\windows\x64\release\simple_channel.exe .\scenes\simple_uniform8.json
```

`simple_uniform8.json` 是关闭对流的平行板基线配置；`simple_perturb8.json` 使用有初始散度的速度扰动来激活压力修正。不要把 `simple_channel.json` 的默认开启对流设置与 Stokes 基线配置混为一组实验。

四壁方管的细网格示例使用 `simple_duct16.json`：x 周期、y/z 无滑移、`ny=16`、关闭对流，配置中启用深度 5 的可选 Anderson 加速：

```powershell
xmake run simple_channel scenes/simple_duct16.json
```

将该配置的 `anderson_depth` 设为 0 可运行原始 SIMPLE。此示例配置与完整验证结果应分别查看。

程序接收一个 JSON 路径。返回码 `0` 表示达到求解器收敛判据，`3` 表示耗尽外迭代次数，`1` 表示运行或算子检查异常，`2` 表示命令行用法错误。求解器收敛和精度验收是两件事；最终精度由验证脚本另行检查。

## 网格、物理量和参数

固定域为 `Lx=1, Ly=Lz=0.125`，流动沿 x 方向。y 两侧始终无滑移；z 可选周期或无滑移；x 可选周期体力驱动或压力入口/出口。

| JSON 参数 | 默认值 | 含义 |
|---|---:|---|
| `ny` | `8` | 粗层跨 y 的 cell 数；必须是至少 8 的二次幂。粗网格为 `(8*ny, ny, ny)`，native tile 为 `8^3` cells |
| `adaptive` | `false` | 将 `0.25 <= x < 0.75` 的 native tiles 加密一级，保留 2:1 粗细交界 |
| `periodic_x` | `true` | x 周期；设为 false 则使用压力入口/出口 |
| `periodic_z` | `true` | true 为平行板通道；false 为 y/z 四壁方管 |
| `rho`, `nu` | `1`, `0.01` | 密度、运动黏度；动力黏度为 `mu=rho*nu` |
| `force` | `[1,0,0]` | 加速度，体积力是 `rho*force`；当前只支持轴向分量 |
| `pressure_in`, `pressure_out` | `1`, `0` | 压力口的物理压力。非周期 x 且未显式给 force 时，force 自动置零 |
| `convection` | `true` | 开启一阶迎风动量对流；false 为稳态 Stokes |
| `alpha_u`, `alpha_p` | `0.7`, `0.3` | 速度方程和压力更新的欠松弛系数 |
| `quadratic_interfaces` | `true` | 启用粗细面二次重构；false 保留线性非正交修正，供消融对比 |
| `nonorth_iterations` | `8` | 每个 SIMPLE 步的延迟压力通量修正次数 |
| `flux_relaxation_memory` | `false` | 可选旧面通量松弛记忆项；基线对齐时保持配置一致 |
| `initial_perturbation` | `0` | 初始速度扰动的绝对幅度，单位为速度；初始压力为零 |
| `tolerance` | `1e-8` | 外层真实方程残差、修正一致性及状态变化的归一化阈值 |
| `linear_tolerance` | `1e-11` | 线性求解阈值；另检查原始矩阵上的实际残差 |
| `pressure_solver` | `"auto"` | 压力矩阵线性后端：`"auto"`、`"cg"` 或 `"ldlt"`，选择规则见下文 |
| `max_iterations` | `2000` | SIMPLE 外迭代上限 |
| `anderson_depth` | `0` | 0 关闭；1–10 为历史深度，仅允许 `convection=false` |
| `dump_iterations` | `[0,1,2,5,10]` | 保存中间状态的迭代编号；最终迭代始终保存 |
| `output` | `output/simple_channel` | 输出目录，相对路径按启动时的工作目录解释 |

压力变量单位为 Pa。周期驱动时输出的是压力波动，等效平均压降由 `force` 单独承担。面通量为体积/时间，带 owner 朝外的符号；不能把它当成面速度或质量流量。

## 离散与实现入口

- `SimpleMesh.*` / `OctreeMesh.cu`：一个 leaf cell 一个控制体，内部面只存一次；2:1 交界拆成四个共享子面。owner 加通量，neighbor 减同一通量。周期面保留相邻周期映像的位移和偏心信息。
- `SimpleSolver.*`：单元中心速度/压力与共享面通量，积分形式动量方程、隐式黏性和可选迎风对流、欠松弛、Rhie–Chow 面通量、压力修正及速度校正。
- `QuadraticReconstruction.*`：只在粗细面相邻 cells 缓存两环邻居的二次最小二乘系数；拟合一阶导和 Hessian。周期 stencil 按 `Face.delta` 展开，秩不足明确报错。
- `OperatorChecks.*`：运行几何闭合、仿射重构、随机通量守恒和压力矩阵/通量身份检查，并调用实际二次重构模块验证多项式一致性。这些检查不能替代耦合 CFD 验证。

内部面黏性保留正的两点隐式系数 `mu*area/distance`，非正交和二次曲率项作为延迟通量。二次修正同时考虑 cell 连线的切向偏移，以及连线中点与真实子面中心的偏移；不是针对某一条解析速度曲线硬编码。

无滑移壁面使用与 Aphros 基线一致的二次单边导数。若 `q_wall` 为壁面值，`q1/q2` 为沿内法向第一、第二个同级 cell 的值，外法向导数为 `(8*q_wall-9*q1+q2)/(3*h)`。隐式部分仍是到壁面距离 `h/2` 的两点项，剩余部分延迟处理。网格必须提供这两个对齐的内部 cells。

**壁面压力梯度外推不等于允许流体穿墙。** 用于单元动量及速度校正的压力梯度从内部外推；物理壁面的法向通量和压力修正通量保持零。压力口的速度使用零法向导数，适用于本目标的充分发展直通道。

压力校正矩阵采用紧致两点部分，非正交/曲率通量在内迭代中更新。最终通量使用最后一次压力方程 RHS 中的同一份延迟通量；另外检查重新计算的修正通量与这份通量之间的偏差。无压力口时，对参考 cell 同时处理矩阵行/列以消除压力零空间，压力更新后再去掉体积均值。

`pressure_solver` 只选择同一压力校正矩阵的线性求解后端，不改变 SIMPLE 方程、边界处理或精度验收阈值：

- `"auto"`：`convection=false` 且实际 leaf cell 数不少于 100000 时，使用 Eigen `SimplicialLDLT`；其余情况使用 CG 加不完全 Cholesky 预条件。
- `"cg"`：显式选择 CG 加不完全 Cholesky 预条件。
- `"ldlt"`：显式选择 `SimplicialLDLT`。Stokes 的固定矩阵缓存分解，以复用多个 RHS；`convection=true` 时，每个外迭代按当前压力矩阵重新分解，同一外迭代内仍可复用该分解。

两种后端都检查原始矩阵上的实际线性残差，并继续执行全部外层收敛判据。缓存多 RHS 分解的收益与整体耗时需从对应运行记录判断，不能由后端名称推定全套验证已经提速或通过。

压力 RHS 在后续延迟修正中精确为零时，直接取零压力校正，避免对零向量归一化；仍继续检查原始矩阵残差及通量修正一致性。

外层收敛同时要求：真实未松弛动量残差、面通量散度、延迟修正的散度及面值缺陷，以及速度、面通量、压力的变化都满足阈值。只看压力线性残差或速度变化不足以验收。

## 可选 Anderson 加速

`anderson_depth=0` 是默认设置，也是逐步对齐 Aphros 时的设置。启用后，对固定归一化的速度、压力和全部共享面通量使用 Type-II Anderson 历史混合；正则化、秩检测和系数保护只检查代数可接受性。

加速发生在完整 SIMPLE 步的残差计算和 dump **之后**。候选及其回溯状态必须降低真实动量残差，并使质量残差小于 `max(1e-10,10*原始质量残差)`。通过的候选只作为下一步输入；不通过则保持原始输出并清空相应历史。最终收敛仍由完整、未加速 SIMPLE 步的全部判据决定。

混合包含面通量，因此公共线性守恒约束在舍入误差范围内得以保留，但仍进行实际质量检查。启用加速后，第 k 步 dump 与第 k+1 步输入可能不同，不能继续用相邻原始 dump 做 Aphros 的逐步 delta-RHS 对齐。

独立 Anderson 代数测试位于 [`validation/anderson_test.cpp`](../validation/anderson_test.cpp)，检查真实矩阵残差、加速前后的解、非齐次线性约束以及异常系数保护：

```powershell
xmake build simple_anderson_test
xmake run simple_anderson_test output/anderson_test.json
```

测试同时向 stdout 输出 JSON；省略路径时不写文件。这个测试不调用耦合 SIMPLE 或其物理残差回溯。

## 输出与调试

每次运行保存以下文件：

| 文件 | 内容 |
|---|---|
| `case.json` | 程序收到的配置；未显式填写的默认参数仍以代码为准 |
| `operator_checks.json` | 独立几何/算子检查及二次重构测试，包含适用范围 |
| `history.csv` | 每个原始 SIMPLE 步的真实方程残差、线性残差和状态变化 |
| `iter_k/cells.csv` | 几何、校正后速度/压力、松弛后动量对角和绝对 RHS、预测速度、压力校正 RHS 与解 |
| `iter_k/faces.csv` | 唯一面几何、owner/neighbor、预测及校正后体积通量 |
| `iter_k/momentum_matrix.csv`, `pressure_matrix.csv` | 积分形式稀疏矩阵 triplets；初始 `iter_0` 没有矩阵 |
| `solution.csv` | 最终 double 单元状态及解析参考值 |
| `sections.csv` | 完整横截面的守恒面通量积分及解析流量 |
| `metrics.json` | 收敛状态、物理误差、流量、平均壁面剪切及网格/算法信息 |
| `native_fields.bin` | 把单元中心 `u/v/w` 写入原 native tile 的通道 0/1/2，`p` 写入通道 3 后的快照 |

`native_fields.bin` 使用原来的 float tile，属于补充回写快照；其通道布局是 SIMPLE 单元中心布局，不是原 flow-map/MAC 速度布局。精度比较使用 17 位有效数字的 CSV。周期设置和物理边界还需结合配置读取，不能仅从 native blob 恢复整个算例。

启用 Anderson 时另有 `acceleration.csv` 和 `dump_semantics.json`。同一输出目录会被后续运行覆盖，正式验证应使用独立目录并保留 harness 的来源记录。

区分单元中心速度误差、解析速度在离散采样点上的误差、以及截面实际面通量积分误差。即便单元中心恰好落在解析曲线上，中点积分仍可能产生流量误差。方管的剪切指标是壁面积加权平均，不是完整局部壁面应力验证。

## ParaView 可视化

已完成的 full 运行位于 `output/simple_validation_full`。在仓库根目录执行以下命令，可将每个已收敛算例的最终 CSV 导出为 ParaView 文件，再生成默认视图（第一条使用装有 NumPy、VTK 的 Python 3.11 或更新版本）：

```powershell
python scripts/export_simple_paraview.py --root output/simple_validation_full
& 'C:\Program Files\ParaView 5.13.0\bin\pvpython.exe' --force-offscreen-rendering scripts/paraview_simple_view.py
& 'C:\Program Files\ParaView 5.13.0\bin\paraview.exe' --state=output/simple_validation_full/paraview/overview.pvsm
```

也可在 ParaView 中选择 **File → Load State**，加载 `output/simple_validation_full/paraview/overview.pvsm`。默认显示 `adaptive16` 纵向速度切片和 `duct16` 横截面，网格边线对应真实 leaf cells。完整三维网格保留在 Pipeline Browser 中，打开 `full octree volume` 的眼睛即可显示。移动结果目录后，加载 state 时需要重新指定数据文件位置。

每个算例的 `paraview/<case>/solution.vtu` 保留原始八叉树六面体、粗细界面悬挂节点和 double 单元数据，没有重采样。选择 `Speed`、`Velocity`、`Pressure`、`RefinementLevel` 或 `VelocityErrorMagnitude` 着色；在 Slice 上切换为 `Surface` 可隐藏网格边线。

这里计算的是三维体网格：`adaptive16` 有 147456 个六面体单元，域尺寸为 `1 × 0.125 × 0.125`。其平行板充分发展解析解只有轴向速度，且仅随 y 变化，z 方向周期；因此纵切片看起来像二维问题。`duct16` 的四壁方管解析速度随 y、z 两个横向坐标变化。这些标准算例验证三维离散下的充分发展层流，不能代表复杂三维流动已经验证。

默认色标 `Speed` 是速度大小，单位为 m/s；`2e-01` 表示 `0.2 m/s`。本次平行板参数 `g=1 m/s², H=0.125 m, nu=0.01 m²/s` 对应解析峰值 `g*H²/(8*nu)=0.1953125 m/s`。检查局部误差应选择 `VelocityErrorMagnitude`，它是数值速度与解析速度的向量差的模，单位同为 m/s，不能直接当成相对误差或百分数。体积加权相对 L2 与守恒流量误差记录在各算例的 `metrics.json`。

`faces.vtp` 保存唯一共享子面的几何及原始守恒体积通量。`VolumeFlux` 的正方向是 owner 的外法向，`PositiveAxisVolumeFlux` 是全局正坐标方向。`NormalVelocity` 只有面法向分量，不是完整速度矢量；观察速度场应使用 `solution.vtu` 的 `Velocity`。

周期驱动算例中的 `Pressure` 是压力扰动，平均驱动力单独记录在配置的 `force` 中；查看压力入口到出口的压力分布可打开 `pressure8/solution.vtu`。当前导出是最终收敛状态，不是物理时间序列。

导出器重新读取 VTK 文件，检查所有字段与原 CSV 完全一致、连接关系及坐标保持一致、六面体体积为正且等于原始体积；记录在 `paraview/export_manifest.json`。`paraview/overview.png` 是默认视图预览。导出不修改原求解结果及其验证记录。

## 可重复验证

先按照 [Aphros baseline README](../validation/aphros/README.md) 构建独立基线并运行其算例。基线 root 应包含 `periodic_3d_n8`、`perturb_3d_n8`、`duct_3d_n8` 等结果目录。以下示例使用本次实验目录，可替换为自己的路径：

```powershell
$baselineRoot = 'D:\Dropbox\Agent-simulation\simple-baseline'
python scripts/validate_simple.py --quick --baseline-root $baselineRoot --output output/simple_validation
python scripts/validate_simple.py --full --baseline-root $baselineRoot --output output/simple_validation_full
```

Quick 为 `uniform8, perturb8, pressure8, duct8`；full 共 11 项，另含 `uniform16, adaptive8, adaptive16, duct16, convective8, accelerated8, adaptive_convective8`。

Full 中 `uniform16, adaptive8, adaptive16, duct16, accelerated8` 设置 `anderson_depth=5`。Quick 四项及 `convective8, adaptive_convective8` 保持深度 0；两项 convective 算例开启对流，其中 `adaptive_convective8` 同时保留 native 粗细八叉树交界。这些设置由验证脚本显式生成，不改变求解器默认关闭加速的行为。

单独运行或审计已有 harness 结果：

```powershell
python scripts/validate_simple.py --cases adaptive8,adaptive16 --baseline-root $baselineRoot --output output/adaptive_validation
python scripts/validate_simple.py --analyze-only --baseline-root $baselineRoot --output output/simple_validation
```

脚本生成固定配置及阈值，记录可执行文件 SHA256、运行状态和输出文件 hash；旧结果、缺失基线和进程失败不能被当成成功。`--analyze-only` 只接受 harness 有来源记录的结果。明确允许旧 binary 的历史审计也不代表当前 binary 通过。检查 `suite_results.json` 的 `passed`、`quick_pass`、`full_pass` 和缺失项目；自选子集通过不等于 full 通过。

调试相同均匀网格的首步：

```powershell
python scripts/compare_simple_aphros.py --ours output/simple_uniform8 --aphros "$baselineRoot\periodic_3d_n8" --iteration 1 --atol 1e-11 --rtol 1e-8 --output output/compare_uniform8_iter1.json
python scripts/compare_simple_aphros.py --ours output/simple_uniform8 --aphros "$baselineRoot\periodic_3d_n8" --final --output output/compare_uniform8_final.json
```

比较脚本校验完整单元/面覆盖，规范周期面方向和重复面，移除压力规范常数，并将 Aphros 的 delta RHS 与本实现的 `absolute_rhs - A_relaxed*u_previous` 比较。中间步比较需要前一步 dump，且应关闭 Anderson。粗细八叉树与均匀基线不具备逐项相同的离散 unknown，应比较解析误差、守恒流量及网格加密趋势。

当前 Aphros SIMPLE 压力入口/出口存在配置处理限制，因此压力驱动分支使用解析直通道参照，不能把它标记为已由该 Aphros 压力口基线独立确认。基线 patch 的内容与版本见其 README。

## 适用范围

当前目标是轴对齐、充分发展、层流的直通道/四壁方管，固定静态网格，恒定密度与黏度。解析参考和误差指标都针对这个范围。

上面的固定回归套件不覆盖曲面、弯管和 GPU 求解；曲管切割单元、按壁面距离
加密以及原生 GPU AMG 的扩展与逐项验收记录见
[曲管实现说明](../validation/twisted/IMPLEMENTATION.md)。真实入口发展段、
压力口回流处理、湍流、可压缩和多相模型仍不在本实验的验证范围内。
直管套件的自适应为固定 x 区间的一级 tile refinement；曲管扩展按壁面
距离生成静态加密，也不是随解动态加密。

`convection=true` 提供迎风对流实现；充分发展单向流本身的稳态对流导数为零，即使该算例通过，也不能据此证明一般有惯性输运的流动已经完成验证。独立多项式检查、求解器残差、解析解比较和跨代码对齐各自覆盖不同问题，应分别保留证据。
