# 曲管切割单元：当前实现与验证边界

本实验在 `feature/simple-octree-pipe` 上扩展 CPU double SIMPLE 参考求解器。
拓扑来自 Cirrus 原来的 `HADeviceGrid<Tile>` 八叉树；曲壁几何暂由独立 Aphros
工作目录生成，再只导入几何。没有把 Aphros 的速度或压力作为 Cirrus 的解。
它还不是 Cirrus 内置的任意 STL/level-set 切割器。

## 几何与数据流

`case.json` 定义一个三维周期曲管单元：

```
yc(x) = 0.0625 + 0.015 sin(2 pi x / 0.25)
zc(x) = 0.0625 + 0.015 cos(2 pi x / 0.25)
phi(x,y,z) = 0.035 - sqrt((y-yc(x))² + (z-zc(x))²)
```

`phi >= 0` 为流体；这里的圆截面在常 x 平面内，不能将其解释为垂直于
中心线的截面也严格为圆。x 周期，外侧管壁静止无滑移；密度 1、运动黏度
0.01、轴向体加速度 1，量纲采用 SI。输出压力是周期扰动压力，均匀驱动
对应每周期 0.25 Pa 压降。原先的 16/32/64/128 系列关闭对流项，是三维
Stokes 问题；新增的 Navier–Stokes 时间推进验证及其基线修复见后文。

1. `validation/aphros/prepare_twisted.py` 把该解析 level set 注入基线的
   `InitEmbedHook`，然后调用 Aphros 原来的 `Embed::Init`。
2. Aphros `src/solver/embed.ipp` 的 `InitFaces` 在边上对零交点做线性插值，
   构造流体开口多边形。切割面开口面积被限制在 `[0.001,0.999] h²`。
3. `InitCells` 用开口面积向量的负和确定壁面外法向，用边交点的平均投影
   确定局部切平面常数；通过平面与立方体相交计算体积分数。壁面面积取使
   面积向量闭合的值。这些量是上游数值几何，不能笼统称为解析曲面精确积分。
   特别是面上独立构造的开口多边形与体积所用的局部切平面不必完全一致。
4. `twisted_diagnostics.h` 导出 cells、faces、walls、polygons CSV。
   `scripts/import_twisted_geometry.py` 只读取几何 CSV，写出带来源 SHA256 的 JSON。
5. `simple/OctreeMesh.cu` 用原生 `iterativeRefine` 生成叶单元。
   `simple/EmbeddedMesh.cpp::makeEmbeddedOctree` 将最细几何装到叶单元上。

## 八叉树中具体存什么

`simple/SimpleMesh.h` 的 Cell 保存原立方体中心、h、level、实际流体 volume
及 cut 标志。未知量仍位于原立方体中心，以匹配此 Aphros 路径的采样约定，
不移到切割部分的几何中心；个别极小切割单元的采样点可能在物理流体外侧。

内部 Face 保留 owner/neighbor 连接，并使用开口面积与开口中心。
粗细交界保留四个独立子面，每个子面只存一份守恒通量。全固体单元不进入
线性系统。每个 cut cell 新增一个 boundary=1 的壁面，带独立的
embeddedNormal、壁面中心与面积。底层八叉树存储仍包含外侧的背景格子；
`native_fields.bin` 单独不能表达所有切割几何，必须结合几何数据使用。

当前加密是静态、一级、按 tile 的：标记切割单元及周围两层最细网格保护带
命中的 root tile。粗叶单元只能在全部 8 个参考子单元都是完整流体时保留；
如果壁面仍穿过粗单元，直接报错。低分辨率下所有流体 tile 可能都被加密，
因此必须检查实际 `coarse_fine_faces > 0`，不能仅凭 adaptive=true 声称验证了自适应。

## 离散方程

`simple/EmbeddedOperators.cpp` 组装稀疏插值与梯度算子，
`simple/SimpleSolver.cpp` 将其接到独立的 SIMPLE 更新中。

- 连续性使用开口面积乘面速度得到的共享通量，壁面通量为零。
- 动量体力与压力源使用切割后的真实流体体积。
- 切割开口面的插值/法向差分通过周边对齐笛卡尔面做双线性组合。
- 无滑移壁面梯度：在 3×3×3 邻域有效单元上拟合线性函数，在
  `xw-h*n` 处求拟合速度，然后取 `du/dn = (0-u_fit)/h`。
  这是当前 Aphros 路径的线性闭合，不能当作二阶曲壁重构。
- 黏性项采用紧凑矩阵隐式求解，完整面算子与紧凑算子的差显式修正。
  按上游规则，只把这个显式黏性修正源在小切割单元附近做体积加权重分配；
  不合并单元、不把切割体积替换为 h³，也不把所有动量源一同重分配。
- SIMPLE 保留压力修正和含体力平衡项的 Rhie–Chow 面通量。
- 粗细交界的补偿使用现有二环二次重构展开为线性稀疏行。
  这个扩展需要另外验证，不能从相同均匀网格的匹配推导出其正确性。

`Face.distance=h/2` 在壁面上是匹配上游紧凑隐式项的系数距离，
不是声称壁面到采样点的几何距离等于 h/2；真实位置保存在 center/ownerOffset，
壁面梯度由上面的独立拟合算子计算。

输出壁面量使用 `tau = mu*(du_dn - n*(n·du_dn))`，单位 Pa，并比较向量而非
只比较模长。它是双方共同定义的诊断量；同网格吻合不能独立证明壁面剪切的物理精度。

## 基线修改与现有证据

Aphros 固定在 `b60ce3da52c19935fa24c778f62f02141eaf7f80`。
独立工作目录为 `D:/Dropbox/Agent-simulation/twisted-baseline/aphros`。
原有直管基线目录与结果未覆盖。

- 添加只读几何/中间状态/壁面导数 dump，以及上述新算例的解析几何 hook。
- 发现上游 SIMPLE 的 `FieldFaceb<Scal> fev(m)` 未初始化 embedded wall 槽，
  后续却会读取这些槽。通过显式选项 `--fix-wall-initialization` 初始化全部面，
  包括壁面预测速度通量。未经修复的失败运行保留在 `stokes_n16`；该基线修复
  需要报告，不能把实际执行的程序称为完全未修改的上游。
- `--direct-backend` 与运行环境 `APHROS_TWISTED_DIRECT=1` 可选启用直接线性
  后端；读取 Aphros 组装的同一稀疏矩阵，检查原矩阵残差，不修改空间离散项。
  它目前仅支持单 block 全域。n16 上与原共轭迭代后端的最终速度最大差
  7.3e-14 m/s，消去常数后的压力最大差 3.2e-13 Pa，均为 175 次外迭代。

两组相同几何、相同采样位置的收敛 Stokes 结果：

| 最细 y 分辨率 | 流体单元 | 速度相对 L2 差 | 壁面剪切相对 L2 差 | Cirrus 体积流量 m³/s |
|---|---:|---:|---:|---:|
| 16 | 2,744 | 8.29e-9 | 6.40e-9 | 5.80324484e-5 |
| 32 | 19,000 | 1.68e-8 | 1.32e-8 | 5.14146442e-5 |

详细原始输出在 `output/twisted/ours_n16`、`ours_n32`；比较报告为
`output/twisted/compare_n16.json`、`compare_n32.json`。
**同网格差很小，但两级网格流量相差约 11.4%（相对粗网格），尚未网格收敛。**
因此当前结论是切割离散已对齐，不能宣称复杂曲管的近壁精度已经达标。
此表是 d2ad7fc 的历史检查点；后续实际自适应及细化结果见下文。

64 分辨率上的原生 tile 加密运行也已收敛，Q=4.93715960e-5 m³/s；
但实际流体区域的 coarse_fine_faces=0，所有粗叶单元均在管外。
`output/twisted/ours_adaptive64` 这个目录名记录的是请求配置，**不能据此把
它当成管内自适应成功案例**。

## 后续实际自适应验证（2026-09-07）

为使 64 分辨率在原生 8³ tile 布局中包含管内粗单元，几何整体平移
`[0,-6,-2] * h64`。这是明确记录的整数网格平移，不改变管径、形状或物理参数。
比较时按元数据把 Aphros 坐标同步平移，未平移任何求解结果的数值。

`ours_adaptive64_shift_amg85` 实际包含 136,632 个流体单元，其中 512 个粗单元，
以及 1,536 个流体粗细交界面。与独立 Aphros `stokes_n64_amg85_v3` 比较：

| 指标 | 相对 L2 差 |
|---|---:|
| 全场速度（体积加权） | 0.0265611% |
| 周期扰动压力（去体积加权常数） | 0.271080% |
| 切割单元速度 | 0.0233883% |
| 壁面剪切向量（面积加权） | 0.0220491% |
| 守恒截面流量 | 0.00984843% |

粗单元中心用 4×4×4 参考单元作三次插值；切割单元与壁面位置仍精确对应。
所有采样模板必须完全落在有效参考流体中。线性/三次转移的全场速度差为
0.05228%，单独报告；它不影响位置完全对应的壁面剪切比较。
两边 alpha_u=0.85、alpha_p=0.3，不能在比较中混用松弛参数：当前 Rhie–Chow
有限网格离散会受到这些参数影响。

相同均匀 64 网格上，Cirrus/Aphros 速度差 2.80e-8、壁面剪切差 1.99e-8。
`check_twisted_mass.py` 根据索引几何和最终共享面通量重算 Aphros 每个切割体积的
散度，并核对周期面的两个副本。最大物理散度为 7.95e-11 /s，体积加权 L2 为
7.97e-16 /s；完整截面的流量相对变化为 5.08e-15。比较器现在将此检查纳入门槛。

128 分辨率不需要平移：790,216 个自适应流体单元（40,960 粗单元），49,152 个
流体粗细交界面；相对于均匀网格的 1,076,936 流体单元减少约 26.6%。
`ours_adaptive128_amg98p03_v2` 已在 74 次迭代收敛：动量残差 8.70e-9，连续性
残差 1.09e-9，Q=4.8755785681e-5 m³/s。独立 Aphros 细网格基线现已完成；
均匀和实际自适应两种对比均通过，具体结果见后面的独立 128/稳态 NS64 检查。

以相同 alpha_u=0.98、alpha_p=0.03 的 64 均匀网格作比较，流量差为 1.2628%
（相对 128 结果）。`analyze_twisted_refinement.py` 在固定的解析壁面点及其真正
内法线上采样：距壁 1.953125、3.90625、7.8125、15.625 mm 的速度相对 L2 差
分别约 4.89%、2.47%、1.35%、0.777%。这些数据包含网格细化和内部自适应的
共同影响。均匀 128 解现已收敛：1,076,936 个流体单元、77 次迭代，
Q=4.8744358306e-5 m³/s。与其比较，自适应 128 的速度、剪切和流量差分别为
0.01552%、0.01412%、0.02344%。这是 Cirrus 内部比较，不是独立基线验证。

仅比较均匀 64→128，流量差仍为 1.28655%，上述四个固定距壁位置的速度差
分别为 4.8945%、2.4822%、1.3602%、0.7834%。这证明原来的近壁网格敏感性
不能主要归因于内部粗化。`refinement_uniform64_128/refinement.json` 保持失败。

共同壁面点的剪切差约 4.44%，但 32/64 邻点表面拟合的敏感性约 1.5–1.6%，
所以不能把 4.44% 全部解释为网格误差。脚本保留失败结果、拟合条件数、采样
敏感性和实际探针 CSV，不降低门槛。本阶段尚未通过近壁网格收敛验收。

## 对流、时间推进及周期接缝修复

`convection=true` 的曲壁路径现支持一阶上风（`convection_scheme=fou`）。
每个共享面按其自身守恒通量选择上风单元，切割开口再应用双线性面插值。
紧凑上风矩阵隐式处理，完整切割面算子与紧凑算子的差同黏性修正一起按原有
体积分数规则重分配。粗细共享子面目前使用各自的一阶上风值，不额外作二次
对流重构。守恒算子检查增加了任意标量/面通量下的全局对流通量抵消检查。

`time_step>0`、`time_steps>=1` 启用后向欧拉：动量矩阵加入 rho*V/dt，
右端加入 rho*V*u_old/dt。每个物理步有独立 SIMPLE 内迭代和 `step_NNNN`
输出，上一物理步速度只在进入新步时更新。`time_history.csv` 同时报告内迭代
收敛、时间加速度和去掉时间项的稳态动量残差。一个物理步内迭代失败时立即
返回非零状态，不推进下一步、不计为已完成步，也不报告稳态成功。

对流扩展发现了另一个基线实现问题。Aphros `simple.ipp` 更新内侧面通量，
但下一次迭代的 `InterpolateUpwind` 会读取支持区面通量；原路径没有交换这些
幽灵面。n16 从静止开始，第 1 次迭代的输入速度/面通量吻合，第 2 次迭代的
468 个周期幽灵面通量却全部为零，导致上风插值在这些位置退成中心插值。
离线将对应来源的上风值换成中心值能解释全部差异到约 6e-14 相对 L2；
此诊断假设不会让实际对流匹配检查通过，修复前记录仍为失败。

`prepare_twisted.py --fix-flux-halo` 加入显式运行时开关
`APHROS_TWISTED_FIX_FLUX_HALO=1`：在旋转迭代层之后、使用对流通量之前，
调用上游已有的 `CommFieldFace(fev_.iter_prev,m)`。它补齐幽灵数据交换，保留
原来的插值、动量和压力方程；实际基线必须明确标注这项实现修复，不能称为
未修改的上游程序。修复前/后使用同一 v8 二进制、相同配置和同一 Cirrus 解：

| n16 对流审计 | 未启用交换 | 启用交换 |
|---|---:|---:|
| 周期幽灵面通量相对 L2 差 | 1.0 | 9.66e-14 |
| 第二次迭代 u 对流面通量相对 L2 差 | 1.83e-4 | 2.11e-14 |
| 第二次迭代 v 对流面通量相对 L2 差 | 2.68e-4 | 5.90e-14 |
| 第二次迭代 w 对流面通量相对 L2 差 | 4.70e-5 | 2.73e-14 |

`check_twisted_advection.py` 比较实际 C++ 稀疏面插值 dump 与 Aphros 在单元
组装/重分配之前的对流表达式，覆盖 108 条穿越周期接缝的插值连接。

n16、dt=0.25 s 的单步瞬态比较，速度/压力/切割单元速度/剪切/流量相对 L2
差分别为 2.26e-9、3.09e-10、1.43e-9、1.72e-9、2.10e-9。这个时刻的
时间加速度约 0.065 m/s²，明确不是稳态，必须使用比较器的 `--transient`。

n16、dt=1 s 推进 8 步后，速度/压力/切割单元速度/剪切/流量相对 L2 差分别
为 6.37e-9、9.16e-10、4.14e-9、4.93e-9、5.93e-9。Cirrus 稳态动量残差
7.35e-9；双方末步时间加速度分别为 1.74e-11、4.09e-12 m/s²。比较器默认
要求真实的时间平稳性和稳态动量残差，不能仅用内迭代变化量替代。

以上是均匀 n16 的实现对齐验证，不能替代壁面加密八叉树上的惯性流验证或
近壁精度验收。n32 双方八步计算和 n128 独立 Stokes 参考现已完成，结果见下文。

新增 AMG 非对称线性路径使用 AMGCL smoothed aggregation + SPAI0 + BiCGSTAB；
对称矩阵仍用 CG，所有原始 Aphros 方程行的残差仍逐行检查。n16 八步计算与
SparseLU/LDLT 直接后端的速度最大差 1.31e-17 m/s、去规范压力最大差
2.99e-17 Pa、壁面剪切分量最大差 3.16e-16 Pa，双方总外迭代均为 529。
此检查只验证线性后端替换，不为后续更细网格结果背书。

新 `run_twisted_baseline.ps1` 记录实际二进制哈希、配置哈希、运行环境与终止
状态，并拒绝覆盖已有 run.log。因 PowerShell 空字符串环境变量仍可能被
`getenv` 视为存在，关闭的开关必须显式从 Env: 删除；两次只生成几何的失败
试跑目录 `navier_stokes_n16_transient_v8_{stale,fixed}` 保留，不计入流动结果。

已导出的 `ours_ns16_steady_v2/solution.pvd` 和 `walls.pvd` 包含 t=1..8 s。
ParaView 5.13 的 PVDReader 已逐步读回并检查 Velocity、WallShear、单元数量和
有限值；最后一帧就是上述稳态解。它们仍是粗网格验证图，不能解释为网格无关解。

可复现命令（在仓库根目录执行；需要先按前文准备隔离 Aphros 工作目录）：

```powershell
python validation/aphros/prepare_twisted.py --aphros D:/Dropbox/Agent-simulation/twisted-baseline/aphros --fix-wall-initialization --direct-backend --fix-flux-halo
# 使用 build_baseline.ps1，按前文传 Eigen/AMGCL 路径，输出独立命名的 main_amg_v9.exe。
python scripts/make_twisted_baseline.py --ny 16 --convection --time-step 1 --time-steps 8 --suffix _ns_reproduce
./scripts/run_twisted_baseline.ps1 -CaseDirectory D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_n16_ns_reproduce -Executable D:/Dropbox/Agent-simulation/twisted-baseline/aphros/src/main_amg_v9.exe -FixFluxHalo -UseAmg
python scripts/import_twisted_geometry.py --baseline D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_n16_ns_reproduce --output output/twisted/reproduce_geometry_n16.json
xmake build simple_channel
./build/windows/x64/release/simple_channel.exe validation/twisted/ns16_steady.json
python scripts/compare_twisted.py --ours output/twisted/reproduce_ns16_steady/step_0008 --aphros D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_n16_ns_reproduce --output output/twisted/reproduce_ns16_comparison.json
python scripts/export_twisted_time_series.py --run output/twisted/reproduce_ns16_steady
```

## 后续 n32 对齐、壁面误差诊断与失败实验

`ours_ns32_steady_v2` 与独立 `navier_stokes_n32_steady_v9_amg_fixed`
均以 alpha_u=0.7、alpha_p=0.3、dt=1 s 推进八步。19,000 个流体单元、
5,424 个曲壁单元的最终速度、压力、切割单元速度、壁面剪切、流量相对 L2
差分别为 6.005e-10、7.398e-11、3.847e-10、4.295e-10、5.686e-10。
Cirrus 稳态动量残差 9.202e-9，时间加速度 5.446e-12 m/s²，流量
5.1393983812e-5 m³/s。Aphros 时间加速度 5.161e-12 m/s²，真实切割体积
上的最大散度 3.068e-10 /s。这个均匀网格结果通过实现对齐检查，尚不证明
近壁网格收敛或粗细交界上的 Navier–Stokes 精度。

`solution.pvd`、`walls.pvd` 包含全部八个时刻。
`validation/check_twisted_paraview.py` 使用 ParaView 5.13 PVDReader 逐帧读取，
检查全部数值单元数组的有限性、单元数，并逐项比较实际读回的速度/剪切
与求解器 CSV。n32 检查通过；这比只检查文件存在或范围更严格。

为定位壁面误差，`analyze_twisted_wall_consistency.py` 采用在解析管壁为零
的制造标量 F=1-((y-yc(x))²+(z-zc(x))²)/R²，在真实采样点计算线性壁面
算子的误差，分解为壁面位置、单边差分和线性拟合三部分。n16/64/128 的
相对 L2 误差为 15.031%、3.760%、1.886%，观测阶约 1。n128 的三部分
误差范数分别为 0.4284%、1.4497%、0.9534%；这些范数不能直接相加，因为
误差会相互抵消。该诊断没有使用流动解，也不能把 1.886% 当成实际剪切误差。

Aphros v10 新增只读 `APHROS_TWISTED_WALL_CONSISTENCY` 环境开关，参数是
`0.035 0.015 0.25 0.0625 0.0625`。它在独立标量字段上调用上游真实
`UEmbed::Gradient`，不改速度、压力或方程。n16/64 与 Python 重建诊断的
相对差分别小于 2.4e-14、3.1e-13；这验证了所分析的确为当前壁面闭合。

尝试的 5³ 邻域三维二次壁面拟合虽然改善局部制造解梯度，却使实际曲管
32→64 流量/最近壁面速度差达到 14.28%/40.61%。这个候选没有通过验收，
已从正常求解器移出；补丁和复现说明在 `experiments/`，原结果保留。
正常求解器显式拒绝 `wall_reconstruction=quadratic5`，防止旧实验配置被
静默当作线性壁面执行。默认线性路径与改动前的 n16 Stokes 速度和剪切
CSV 逐字节一致。原来的 64→128 近壁收敛失败仍有效。

另一组试验用很大的后向欧拉步长近似定常极限。n16 的 dt=1e8 s 单步
Aphros 解与 dt=1 s 八步解的速度/压力差为 1.817e-6/3.099e-5。时间项
很小也不能保证这两个有限网格固定点完全一样：动量对角元仍进入
Rhie–Chow 面通量。比较器因此在定常 NS 对齐中也检查相同时间步长。
双方都用 dt=1e8 s 时，Cirrus 第二个校正步与 Aphros 单步的速度/压力/
剪切/流量差为 7.388e-9、1.069e-9、5.704e-9、6.869e-9，且稳态动量、
时间加速度和守恒门槛均通过。Cirrus 首次单步的稳态动量残差
1.0036e-8 略高于 1e-8，失败记录保留，未调整门槛。这组大步长结果仅用于
定常极限求解，不作为实际非定常轨迹的验证。

`scripts/run_twisted_solver.py` 现为新的 Cirrus 运行记录实际二进制和配置
哈希、环境变量、开始/结束时间及退出码，并拒绝覆盖已有输出目录。
历史运行的二进制信息继续沿用历史检查点；不能把当前源码哈希冒充为
历史二进制的编译来源。

带对流的 embedded 路径现允许显式启用 `anderson_depth=5`。每个时间步
单独建立加速历史；候选同时包含速度、压力和共享面通量。接受前用候选
自身的速度/通量重新组装非线性动量残差，并检查真实切割体积的连续性。
最终收敛仍只能由完整、未加速的 SIMPLE 输出确认，未放宽任何门槛。
普通非 embedded 对流路径仍不启用此扩展，默认深度仍为零。

n16、dt=0.25 s 的瞬态迭代数由 190 降为 31，与 Aphros 的速度/剪切差
为 2.345e-9/1.784e-9。n32、dt=1 s 八步总迭代数由 1,724 降为 150；
`check_twisted_time_pair.py` 对所有八步分别核对速度、去规范压力、切割
单元速度、剪切和共享面通量，通过原有 1e-6 相对门槛。最终与独立 Aphros
的速度/压力/剪切/流量差为 7.261e-9、2.088e-9、6.089e-9、6.601e-9。
`check_twisted_time_failure.py --anderson-depth 5 --max-iterations 4` 还确认
实际接受过加速候选后，未收敛物理步仍返回失败、不推进第二步。
复现配置为 `ns32_steady.json`；实际粗细交界的 n64 加速运行仍需独立验收。

上游 `approx_eb.ipp` 虽然留有 `GradDirichletQuadSecond`，但默认调用被
注释掉，且其横向索引使用 `1-argmax(normal)`，属于二维构造，不能直接
切换用于本三维曲管。当前没有把它启用到 Aphros 基线中。

## 细网格算子与线性后端

- 切割面双线性模板可能引用被排除的笛卡尔面。Aphros 先将这些面值与面梯度
  初始化为零，然后只填有效面；Cirrus 现采用同一策略。128 网格有 4 个这样的
  目标面，最大固定零系数为 0.0906032053187411。Aphros 新增的只读
  `APHROS_TWISTED_INTERPOLATION_CHECK=1` dump 已独立确认计数、系数和常量零梯度。
  这不是把缺失单元压力当零，也没有重新归一化插值权重。
- 实际组装后的稀疏行逐项检查粗细交界二次多项式、常量压力梯度、壁面线性场
  法向导数和全局黏性通量平衡。128 的对应误差约 1e-15、1e-15、2.72e-14、8.26e-17。
- 可选 AMGCL smoothed aggregation + SPAI0 + CG 压力后端减少细网格求解代价。
  右端先作标量归一化，避免小 SI 量纲 RHS 被 AMGCL 当成零；解后检查原矩阵
  残差，并可作迭代精化。没有依赖时显式请求 `pressure_solver=amg` 会报错。
- Aphros 可选后端仍读取上游组装的矩阵。仅剥离代数上孤立的标量行并精确求解；
  压力固定点选在大对角元处。v5 只移除已小于请求线性容差的周期 RHS 舍入均值，
  最后仍对全部原始方程（含未固定的压力行、原始 RHS）检查残差。
  n16 与原共轭后端速度最大差 7.65e-14 m/s，去常数压力最大差 3.12e-13 Pa。
- Anderson 候选的连续性限制对 embedded 情况为 `max(1e-10,0.25*tolerance)`
  与十倍原始步残差的较大者，以容纳极小体积的浮点通量合并误差。最终收敛
  仍必须通过全新的未加速 SIMPLE 步和原来的全部残差门槛。

## ParaView 查看

运行 `python scripts/export_twisted_paraview.py --run output/twisted/ours_n32`。
`solution.vtu` 包含局部平面切出的多面体以及 Velocity、Speed、Pressure、
Level、CutCell、FluidFraction 和 UnknownLocation；没有插值或构造解析参考场。
`walls.vtp` 包含壁面多边形及 WallShear / WallShearMagnitude。
使用 `Surface With Edges` 查看切割形状，用 `Slice` 看内部截面。

导出程序检查多边形闭合与一致朝向，并分别用闭合面有向体积分和独立凸包
体积核对求解体积。VTK 通用体积过滤器的结果另行记录：VTK 9.3.1 对 128 网格
中的 8 个近顶点切割单元会漏算细小内部四面体，同一几何改顶点编号也会改变其
结果，最大差为 1.24e-6 h³，总体积差仅约 9.60e-12。几何未为迎合过滤器而修改。
源码路径为 [vtkPolyhedron::Triangulate](https://github.com/Kitware/VTK/blob/v9.3.1/Common/DataModel/vtkPolyhedron.cxx#L1203)，
使用 vtkOrderedTriangulator；可复现几何与朝向检查在 `validation/test_twisted_clip.py`。
壁面可视化与体积采用局部平面，数值开口
面积仍可能因上游独立重构/限幅略有区别，不能把可视化多边形面积直接替代通量系数。

## 来源与许可

切割面/壁面算子的实现依据 Aphros 的 `src/solver/embed.ipp`、
`src/solver/approx_eb.ipp`、`src/solver/simple.ipp` 中的对应路径。相关 MIT 许可与
Copyright (c) 2021 ETH Zurich 见 `../aphros/LICENSE.aphros`。
Cirrus 的八叉树拓扑、粗细交界重构与 SIMPLE 主流程保持为本项目实现。

AMGCL 为可选外部依赖，版本 `f4614a7e9ccfe716c4c96df75dc349157229609a`；
来源 https://github.com/ddemidov/amgcl ，MIT 许可见 `../aphros/LICENSE.amgcl`。

## 复现当前自适应检查

以下命令在仓库根目录执行，Python 需要 NumPy、SciPy、VTK；路径可按机器调整。
先按 `validation/aphros/README.md` 建立固定提交的独立工作目录、应用诊断补丁、
构建依赖，再启用曲管 hook 和明确的壁面初始化修复。

```powershell
python validation/aphros/prepare_twisted.py --aphros D:/Dropbox/Agent-simulation/twisted-baseline/aphros --fix-wall-initialization --direct-backend
./validation/aphros/build_baseline.ps1 -AphrosRoot D:/Dropbox/Agent-simulation/twisted-baseline/aphros -EigenInclude <Eigen头文件目录> -AmgclInclude D:/Dropbox/Agent-simulation/twisted-baseline/amgcl -ExecutableName main_amg_v5.exe
xmake f --simple_amgcl_root=D:/Dropbox/Agent-simulation/twisted-baseline/amgcl -y
xmake build simple_channel
python scripts/make_twisted_baseline.py --ny 64 --velocity-relaxation .85 --pressure-relaxation .3 --pressure-linear-tolerance 1e-12 --suffix _reproduce85
```

进入生成的 `stokes_n64_reproduce85` 目录，设置以下环境变量并运行基线。
环境变量 `APHROS_TWISTED_GEOMETRY_ONLY=1` 只生成几何而不求流场，不能用于通过比较。

```powershell
$env:OMP_NUM_THREADS='4'
$env:OMP_WAIT_POLICY='PASSIVE'
$env:APHROS_SIMPLE_DUMP='simple'
$env:APHROS_TWISTED_GEOMETRY='tube'
$env:APHROS_TWISTED_DIRECT='1'
$env:APHROS_TWISTED_AMG='1'
../aphros/src/main_amg_v5.exe a.conf *> run.log
```

回到 Cirrus 仓库，导入几何，再在配置中指定它并运行。已提交的
`ours_adaptive64.json`/`ours_adaptive128.json` 保存本机验证配置；换目录时相应修改路径。

```powershell
python scripts/import_twisted_geometry.py --baseline D:/Dropbox/Agent-simulation/twisted-baseline/stokes_n64_reproduce85 --shift-cells 0 -6 -2 --output output/twisted/geometry_n64_shift.json
./build/windows/x64/release/simple_channel.exe validation/twisted/ours_adaptive64.json
python scripts/compare_twisted.py --ours output/twisted/ours_adaptive64_shift_amg85 --aphros D:/Dropbox/Agent-simulation/twisted-baseline/stokes_n64_reproduce85 --adaptive --output output/twisted/compare_adaptive64_reproduced.json
python scripts/export_twisted_paraview.py --run output/twisted/ours_adaptive64_shift_amg85
```

128 对照用 `--ny 128 --velocity-relaxation .98 --pressure-relaxation .03
--pressure-linear-tolerance 1e-11`，导入时不平移。运行前应保留已有输出目录，
为重复实验在配置中选新输出路径。

## NS64 八叉树时间序列与独立首步检查

`ours_ns64_adaptive_aa5_v1` 已完成 dt=1 s 的全部八步，总计 412 次内迭代。
使用 136,632 个真实流体叶单元、21,816 个壁面切割单元、1,536 个粗细交界
子面；其中 512 个粗流体单元，136,120 个细流体单元。最终 Q 为
4.93649187338864e-5 m³/s，稳态动量相对残差 2.76052e-9，时间加速度
1.00681e-12，截面流量相对差 4.80441e-15。实际执行程序仍为已验证的
`simple_channel_wall_audit_v2.exe`，没有因为这些新实验改动 C++ 空间离散。

独立 Aphros `navier_stokes_n64_steady_v9_amg85_fixed` 完成第一步后，保存
`navier_stokes_n64_steady_v9_step1_snapshot`。`snapshot_twisted_reference.py`
要求下一物理步已开始，证明上一 FinishStep 的输出已经关闭；复制期间还逐项
检查场文件和时间记录没有变化。它保留原始八步配置和运行来源，不生成虚假的
进程完成记录。比较器必须显式使用 `--transient --reference-checkpoint`，
报告 `complete_reference_run_checked=false`，不能据此通过最终稳态验收。

在 t=1 s，八叉树与独立均匀 Aphros 的相对差为：速度 0.0265370%，压力
0.262558%，切割单元速度 0.0221907%，剪切向量 0.0208992%，守恒流量
0.00886933%，全部通过预先设定的自适应门槛。细单元和壁面位置精确对应；
粗中心采用已检查的三次采样，线性与三次采样的全场速度差为 0.0520300%。
这个检查是瞬态对齐，独立 Aphros 的后续时间序列仍需完成。

首步还与原普通 Cirrus 路径逐场比较，通过 1e-6 门槛，速度差 6.86375e-9。
加速路径首步 130 次、普通路径 930 次。普通运行在这一检查及此前完整 NS32
八步验证后主动停止，`intentional_stop.json` 保存原因；其后七步不算完成。

`solution.pvd`、`walls.pvd` 包含真实 t=1..8 s。已使用 ParaView 的 PVDReader
逐一读取八个时刻，核对所有数值数组有限，并将速度/剪切数组与 CSV 比较。
`step_0008/cut_geometry.png` 展示实际导出的曲管和切割格子。

复现配置为 `validation/twisted/ns64_steady.json`。这些已完成的历史计算使用
`output/twisted/simple_channel_wall_audit_v2.exe`；当前本机 xmake builddir 已切到
`output/build_compact_rows`，包含后文的存储优化。默认 `build` 目录仍保留旧程序。
使用历史不可变程序复现的示例：

```powershell
python scripts/run_twisted_solver.py --config validation/twisted/ns64_steady.json --exe output/twisted/simple_channel_wall_audit_v2.exe --threads 2
python scripts/export_twisted_time_series.py --run output/twisted/reproduce_ns64_steady
& 'C:/Program Files/ParaView 5.13.0/bin/pvpython.exe' validation/check_twisted_paraview.py --run output/twisted/reproduce_ns64_steady
```

## 精简 Aphros 驱动与新增网格检查

`validation/aphros/twisted_driver.cpp` 直接调用原库的 `Embed`、`Simple`、
`ParsePar` 和线性求解器工厂，只初始化本算例所需的单相物性、零初始速度、
平衡体力与无滑移壁面。它省去 Hydro 中本算例不使用的多相、示踪剂、颗粒
调度。构建脚本 `build_twisted_driver.ps1` 链接已准备好的静态库，记录源文件、
库、编译参数和可执行文件 SHA256。该库仍包含明确披露的壁面初始化/周期
通量 halo 修复、只读诊断及可选线性后端，不能称为未打补丁的上游程序。

`check_twisted_driver_pair.py` 要求两边输入完全一致，验证切割几何文件逐字节
相同、整段外迭代历史、各物理步时间诊断、全部现有中间 dump 及最终速度、
压力、壁面剪切和共享面通量。n16 瞬态一步、dt=1 s 八步都已通过，迭代数
分别与完整驱动相同，都是 164 和 529。完整 v10 程序与精简驱动的八步检查
也通过。n32 八步扩展检查现已通过，完整和精简驱动均为 1,736 次外迭代；
几何、迭代历史、时间诊断、中间 dump、最终速度/压力/剪切/通量均通过检查。

n16 八步实测：完整 v10 进程峰值工作集 59,052,032 B，精简驱动
54,128,640 B，降低约 8.3%。这是本档测量，不能据此声称更细网格一定装得下。

另外 `ours_ns64_adaptive_large_dt2_aa5_v1` 完成 dt=1e8 s 的两个稳态极限
步骤（108+1 次内迭代）。在相同 dt、松弛参数和壁面闭合下，与已对齐 Aphros
的 n32 大步长解进行固定物理探针检查，**32→64 仍未通过网格收敛**：Q 差
4.1121%，距壁 1.953125 mm 的速度差 24.4444%，剪切差 4.2437%。剪切采样
敏感性约 1.4–1.6%，也未通过门槛，因此不能把剪切差全部归因于网格误差。
这一检查同时细化壁面并改变内部粗细拓扑；此前 64→128 纯均匀 Stokes
失败仍独立保留。此处大步长是稳态极限研究，不代表真实启动过程的时间精度。
上述失败报告和已有失败实验均保留，目标仍未完成。

## 独立 128 Stokes 与 64 稳态 Navier–Stokes 检查

独立 Aphros `stokes_n128_amg98p03_v5` 已正常结束，533 次迭代，末次变化
9.85717e-12。`compare_uniform128_aphros.json` 在相同单元/壁面位置上通过
1e-6 门槛：速度相对 L2 差 4.61462e-9、压力 7.06626e-8、剪切向量
4.11114e-7、守恒流量 1.65224e-9。独立质量守恒重算也通过。

`compare_adaptive128_aphros.json` 对 790,216 个真实流体叶单元的检查通过：
速度差 0.0155226%、压力 0.195621%、剪切 0.0141221%、流量 0.0234436%。
该网格有 49,152 个真实流体粗细交界子面；切割单元和壁面精确对应，粗中心
采用经过采样敏感性检查的转移。此前内部对比与本次独立基线证据分别保留。

独立 Aphros `navier_stokes_n64_steady_large_dt2_v9_85` 已完成 dt=1e8 s 的
两个步骤，共 980 次外迭代。与 Cirrus `ours_ns64_adaptive_large_dt2_aa5_v1`
末步的完整稳态检查通过：速度差 0.0266902%、压力 0.268167%、剪切向量
0.0212007%、流量 0.00917266%。双方 dt、松弛参数相同，时间平稳性、Cirrus
稳态动量残差和双方质量守恒均通过。这是完整稳态对齐；前面的 dt=1 s 独立
首步快照仍仅代表瞬态前缀，不据此宣称那个独立八步运行已全部完成。

这两项新证据证明实现对齐，没有改变此前失败的物理网格收敛结论。

## 更细网格所需的存储优化与精确几何输入

`EmbeddedOperators.cpp` 用按列排序的小型连续数组代替逐项分配的 map
组装行，保持原来的求和顺序；仅为 cut cell 构造壁面拟合/源重分配所需的
3×3×3 邻域。`check_twisted_storage_pair.py` 要求完整步骤及全部现有数值 dump
一致。Stokes16、NS32 八步和真实自适应 Stokes64 分别有 34、170、35 个文件
逐字节相同，既没有改空间算子，也没有放宽任何收敛门槛。

在同一 Stokes64 配置、相同线程数的两个完整运行中，原程序峰值工作集为
1,128,304,640 B，连续行程序为 797,220,864 B，降低 29.34%；Windows
PeakPagefileUsage 从 1,464,680,448 B 降至 1,170,001,920 B，降低 20.12%。
后者是进程提交内存计数，并不表示实际写到磁盘的字节数。运行期间其他任务的
CPU 负载不同，不据此比较速度。

`import_twisted_geometry.py --packed` 可写出 v2 格式：一个小 JSON 加三个
little-endian float64 表。每表先写 24 字节头：8 字节 CIRRCUT1、uint64 行数、
uint32 列数、uint32 0x01020304；cells/faces/walls 列数分别为 12/8/11。
元数据记录源几何和表内容的 SHA256。C++ 检查格式、大小、有限值及原有几何
不变量，运行包装器在启动前核对表哈希并检查运行期间输入不变。旧 JSON 输入
仍可使用，读取后也先转成连续记录、释放庞大 JSON，再建立原生拓扑。

`check_twisted_geometry_pair.py` 已逐个 float64 位模式核对 64/128 的三张表，
与旧几何完全相同。NS64 两步的新旧格式运行共 42 个数值文件检查通过；仅
metrics 中记录输入路径的 geometry_source 允许不同。实际 C++ 对被截断的
二进制表返回失败且不输出解，验证记录也保留。64 这一对输入格式的求解器峰值
工作集都约 947 MB，因此目前不宣称二进制格式进一步降低了整段求解峰值。
它避免的是大规模 JSON 解析开销；128 的完整实测要等对应运行完成。

新的公平 NS64→128 检查统一使用 alpha_u=.98、alpha_p=.03、dt=1e8 s×2，
复现配置为 `ns64_refinement.json` 和 `ns128_refinement.json`。64 的本地计算
已完成，128 本地两步计算也已完成；对应独立 128 基线验证仍待完成。不能把 .85/.3 的 64
结果与 .98/.03 的 128 结果直接当作只改变网格的收敛证据。

```powershell
python scripts/import_twisted_geometry.py --baseline D:/Dropbox/Agent-simulation/twisted-baseline/stokes_n64_amg85_v3 --shift-cells 0 -6 -2 --packed --output output/twisted/reproduce_packed64/geometry.json
python scripts/import_twisted_geometry.py --baseline D:/Dropbox/Agent-simulation/twisted-baseline/stokes_n128_amg98p03_v5 --packed --output output/twisted/reproduce_packed128/geometry.json
python scripts/run_twisted_solver.py --config validation/twisted/ns64_refinement.json --exe output/build_compact_rows/windows/x64/release/simple_channel.exe --threads 2 --measure-memory
python scripts/run_twisted_solver.py --config validation/twisted/ns128_refinement.json --exe output/build_compact_rows/windows/x64/release/simple_channel.exe --threads 2 --measure-memory
```

两次大规模计算应按可用提交内存安排，保留已有输出并使用新的复现目录。

## NS64→128 完整结果与黏性组合算子诊断

`ours_ns128_large_dt2_98p03_packed_v1` 正常结束，75+1 次内迭代，两步都完成。
最终有 790,216 个流体单元、87,248 个曲壁单元和 49,152 个粗细交界子面。
稳态动量相对残差 9.64164e-9、时间加速度 3.09110e-21、截面通量相对差
9.03532e-15。进程峰值工作集 4,887,072,768 B，峰值提交内存计数
5,863,567,360 B。运行包装器确认配置、实际程序和几何输入没有改变。

64 档新的独立 `navier_stokes_n64_large_dt2_98p03_minimal_v1` 完成 520+1 次
迭代，与同参数本地末步比较通过：速度差 0.0265479%、压力 0.253977%、
切割单元速度 0.0222943%、剪切向量 0.0210168%、流量 0.00913510%。独立
128 档同参数运行已经启动，使用同一精简驱动和原有隐式空间算子。

相同松弛系数、dt、物性、驱动和壁面闭合的 NS64→128 检查仍失败：

| 量 | 网格相对差 | 原门槛 |
|---|---:|---:|
| 守恒流量 | 1.26463% | 0.5% |
| 距壁 1.953125 mm 速度 | 4.88903% | 1% |
| 距壁 3.90625 mm 速度 | 2.47452% | 1% |
| 距壁 7.8125 mm 速度 | 1.34904% | 1% |
| 距壁 15.625 mm 速度 | 0.770613% | 1% |
| 固定壁面位置的剪切向量 | 4.43716% | 1% |

剪切采样敏感性为 1.63298%/1.47710%，仍超过 0.25%。两个网格的全壁面平均
剪切仅从 0.01567699 变到 0.01568427 Pa；这个平均值掩盖局部误差，不能替代
固定位置验收。`refinement_ns64_128_large_dt98` 保存数据和图，绘图现在按实际
方程标注 Stokes 或 steady Navier–Stokes，并读取实际验收状态。

128 末步的 `solution.vtu` 和 `walls.vtp` 已导出。使用实际 ParaView XML
读取器检查所有数值数组，并对速度、压力和剪切逐项核对 CSV，最大差均为零。
几何闭合边界积分和独立凸包体积也通过；VTK 通用体积过滤器的已知差异继续
单独记录，未用于掩盖错误。

为检查为何单个壁面梯度通过线性测试、整体近壁结果却仍敏感，新增只读程序
`validation/aphros/twisted_affine_audit.cpp`。它链接原来的 Aphros 静态库，
直接调用 `GradientImplicit`、`RedistributeConstTerms` 和 `RedistributeCutCells`，
不修改库、不推进流场。测试为平面切割区域上的标量 u=y-ywall：x/z 周期，
嵌入壁面值为零，顶部给出精确 Dirichlet 值；精确梯度已知、拉普拉斯为零。
16/32/64 网格各测试两个单元内壁面偏移 0.27、0.73。这是局部算子诊断，
不替代三维曲管，也不是固定物理管道的网格收敛测试。

结果：所有面梯度误差和重分配前的单元通量残差（除以 h²）都约为 1e-14。
仅重分配表达式的常数项后，两种偏移的最大残差/h² 分别为 0.0884232 和
0.859516，三个分辨率均复现。它仍保持全局残差总和，因此守恒检查无法单独
发现这一问题。这些数值是算子残差指标，不能解释为速度误差百分比。

当前零曲壁值下的黏性组合是 Lc + R(Lf-Lc)，其中 Lc 为紧凑隐式矩阵，Lf
为完整面算子，R 为小切割单元的重分配。即使 Lf 对某个场精确，Lc 与 R 的
组合也可能改变其局部平衡。上述原库调用证实了此平面测试中的线性一致性缺陷；
尚不能将全部曲管误差归因于这一项。

另一个控制是将完整零通量残差传给上游 `RedistributeCutCells`，残差仍约为
1e-14。原有配置 `set string conv exp` 选择 `ConvDiffScalExp`，正是先形成完整
对流/扩散残差再重分配（`convdiffe.ipp`），是下一步应验证的已有路径。
`explviscous` 是另一项力修正开关，不能将它误当作这两种动量求解器的选择。
该候选路径的曲管精度尚未验证，原隐式基线、失败报告和所有验收门槛均保留。

复现原库算子诊断：

```powershell
./validation/aphros/build_twisted_driver.ps1 -Source validation/aphros/twisted_affine_audit.cpp -ExecutableName twisted_affine_audit.exe -OutputDirectory D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_affine_build
python scripts/run_twisted_affine_audit.py --template D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_n16_steady_minimal_v1/a.conf --build-directory D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_affine_build --output output/twisted/reproduce_affine_audit
```

## 完整动量残差路径及周期残差 halo 诊断（2026-09-07）

新增 Cirrus 配置 `momentum_mode=exp`，对应原有 Aphros `conv=exp`。
`EmbeddedOperators` 将紧凑动量空间矩阵置零；对流和黏性完整面残差经过同一 R
重分配。时间对角线、体力/压力源、无滑移壁面拟合与 SIMPLE 压力修正仍保持
各自原来的定义。默认 `imp` 不变，比较脚本拒绝混用两种模式。
本入口要求 embedded Navier–Stokes 和正时间步。这里的“exp”是 SIMPLE 内
上一迭代的空间残差更新；物理旧时刻速度在整个内迭代中固定，因此内迭代
收敛后的物理时间离散仍是后向 Euler，不能直接称为一次前向 Euler 更新。

首个 n16、dt=1e-4 s 的试验中，第 1 次迭代全部数值吻合，而第 2 次动量
预测在周期接缝出现偏差。完成一步后，速度相对 L2 差 3.3844e-4、压力
1.2253e-3，失败输出保存在 `ours_ns16_explicit_step1_v1` 与原 Aphros
`navier_stokes_n16_explicit_step1_v1` 的比较报告中。

上游 `convdiffe.ipp` 只在内部单元组装完整通量残差，随后直接调用
`RedistributeCutCells`；后者会读取一层相邻单元，却没有在此处更新残差 halo。
相反，原隐式路径的 `RedistributeConstTerms` 在重分配前进行了通信。
新诊断 `twisted_explicit_residual.h` 同时计算未填 halo 和填入周期副本后的
原重分配结果。`analyze_twisted_explicit_halo.py` 将它们之差与两边第 2 轮
真实动量矩阵/右端项之差比较：三个分量都只影响 x=0.00390625 和 0.24609375
两列中的 138 个单元，最大未解释残差分别为 1.174e-20、2.818e-20、1.398e-20。
补齐 halo 后，完整残差总和的守恒误差处于浮点舍入量级。

该审计没有覆盖旧 Aphros 库：`prepare_twisted_explicit_audit.py` 在独立目录
复制原始 `convdiffe.cpp/.ipp`，仅插入诊断/可选 halo 修复入口；链接时用新
翻译单元和原静态库生成另一个程序。旧库、旧驱动和正在运行的 128 基线保持
原样。诊断开启而修复关闭时，原运行的全部 16 个数值 CSV 逐字节一致。
修复只支持单 block 覆盖全域，显式开关为
`APHROS_TWISTED_FIX_EXPLICIT_RESIDUAL_HALO=1`；它补齐周期邻居数据，不替换
原梯度、对流、重分配或 SIMPLE 算法。必须在基线来源中披露这项修复。

修复后，n16 的第 1、2、10 轮中间对比通过（第 10 轮未输出上一轮，明确
不比较 delta RHS）。`compare_simple_aphros.py --embedded-walls` 现在逐个核对
壁面位置、面积、所属单元并要求预测/修正壁面通量严格为零，再单独匹配双方
Cartesian 面；没有将额外壁面简单忽略。

完成八个 dt=1e-4 s 的物理时间步，末时刻 t=0.0008 s 的独立比较：

| 均匀分辨率 | 速度相对 L2 | 压力相对 L2 | 壁面剪切相对 L2 | 流量相对差 |
|---|---:|---:|---:|---:|
| n16 | 7.7374e-11 | 1.3955e-9 | 2.2284e-10 | 5.2342e-12 |
| n32 | 4.6155e-11 | 1.5027e-9 | 1.8101e-10 | 7.8095e-12 |

两组都通过原来的 1e-6 同网格门槛及独立质量守恒检查。它们仍是启动阶段
瞬态，不能用来宣称稳态近壁精度或网格收敛已经达标。

Cirrus 第一版程序为 `simple_channel_exp_momentum_v1.exe`，SHA256
`c61395d9b62659aea7150435ad65cafff81e4b806aeda156fabf98895d9c3fa9`。
后续 `simple_channel_exp_cached_v2.exe` 的 SHA256 为
`8d87de9e8612b6a0e1a559ea7a457bc38797fe752e1c08be4ec8f8564af03ab1`：
只在 exp 路径矩阵压缩索引和全部系数字节完全一致时复用线性分解，矩阵有
任何差异就重新计算。n16、n32 八步各 170 个数值文件逐字节一致；默认隐式路径的
28 个文件也与之前的已验证程序逐字节一致。n32 这两次实际运行耗时约 820/99 s，
但均与其他作业并行，不能当作隔离性能测试。此优化不改变方程或验收门槛。

`ours_ns16_explicit_steps8_v1/solution.pvd` 和 `walls.pvd` 已导出全部八个时刻。
`check_twisted_paraview.py` 使用实际 ParaView PVDReader 逐时刻读取两个数据集，
检查所有数值数组有限，并逐元素核对速度、压力和壁面剪切与求解 CSV 完全一致。
末步的 `cut_geometry.png` 已渲染检查。

稳态候选另设 dt=1e8 s、alpha_u=1e-12、alpha_p=0.3、两个物理步，使原 exp
更新的有效松弛时间尺度为 1e-4 s。Aphros `navier_stokes_n16_explicit_steady_halofix_v1`
与 Cirrus `ours_ns16_explicit_steady_aa5_v1` 独立从零开始；这两个计算在本检查点
尚未达到验收条件，不能用巨大时间步造成的小时间导数代替动量收敛。
原隐式路径的 64→128 近壁/剪切网格收敛失败仍然有效。下一步必须确认这个
完整残差路径的稳态行为，再做实际管壁加密和物理网格收敛检查。

独立诊断版本与八步基线的复现示例（输出目录必须是新的）：

```powershell
python validation/aphros/prepare_twisted_explicit_audit.py --aphros D:/Dropbox/Agent-simulation/twisted-baseline/aphros --output D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_exp_source
./validation/aphros/build_twisted_driver.ps1 -ExtraSources D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_exp_source/convdiffe.cpp -OutputDirectory D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_exp_build
python scripts/make_twisted_baseline.py --ny 16 --convection --momentum-mode exp --time-step 0.0001 --time-steps 8 --suffix _exp_reproduce
./scripts/run_twisted_baseline.ps1 -CaseDirectory D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_n16_exp_reproduce -Executable D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_exp_build/twisted_driver.exe -UseAmg -FixFluxHalo -FixExplicitResidualHalo
```

## 原隐式 NS64 八步完整基线完成（2026-09-07）

此前持续运行的完整 Hydro 驱动 `navier_stokes_n64_steady_v9_amg85_fixed`
于 13:59:48 UTC 正常退出：dt=1 s、八步、alpha_u=0.85、alpha_p=0.3，
共 2686 次内迭代，耗时 12717.035 s。末步的体积加权时间加速度为
5.2581e-12 m/s²。此处为原 `imp` 路径，与上面的 `exp` 候选分别记录。

对实际 136632 个八叉树流体单元、1536 个粗细交界面的
`ours_ns64_adaptive_aa5_v1/step_0008`，完整运行比较通过：速度相对 L2
0.0266903%、压力 0.268189%、壁面剪切 0.0212010%、流量 0.00917167%。
双方时间步一致，Cirrus 稳态动量残差 2.7605e-9，独立质量守恒及双方时间
稳态检查均通过。报告为 `compare_ns64_steps8_complete_aphros_v1.json`。
这补齐了先前只取得中途快照的完整运行证据，不改变 64→128 物理网格收敛
失败的结论，也不构成 `exp` 稳态路径的验收。

## 完整残差稳态失败与原 Proj 路径复现（2026-09-07）

前述 n16、dt=1e8、alpha_u=1e-12 的 `exp` 稳态候选已经终止：Cirrus
在第一步的 30000 次迭代后以 exit 3 退出，稳态动量残差约 0.001608，
最大速度 2.26549 m/s。最大速度出现在体积分数约 9.8784e-5 的薄切割单元。
Aphros 对应候选同样在 30000 次迭代后触发未收敛异常，退出码 -1073740791。
两者都失败，不能把巨大物理时间步导致的小时间导数当作稳态证据。

新增 `SIMPLE_EMBEDDED_OPERATOR_DUMP`（运行器 `--dump-operators`）仅输出
当前 C++ 已组装的稀疏插值、梯度、黏性和重分配算子及其实际网格。
`simple_channel_operator_audit_v3.exe` 的 SHA256 为
`05986add97513d5f1a386b382c1fd1b992ae96020bbdbb16bf7d3add8542cb0b`。
关闭该诊断时，n16 第一步的 27 个数值输出与已验证 exp v1 逐字节一致。

`audit_twisted_diffusion_spectrum.py --dense` 对实际 n16 的 2744 个单元求
完整特征谱。对 du/dt=-L u，L 的最小实部约为 0.0999441 /s；没有发现
负实部增长模态，不能将失败简单归因于线性黏性不稳定。但存在很弱的衰减模态。
稀疏 ARPACK 首次尝试未收敛的结果也保留，没有把它记作成功。

`audit_twisted_coupled_system.py` 独立组装同一均匀网格的 exp 固定点耦合系统。
其瞬态第一步与原生 SIMPLE 的速度相对 L2 约 1.81e-11、压力约 2.19e-9，
验证了诊断代数与实际实现的对应关系。稳态 Stokes 直接解的最大速度却达到
4.08689 m/s。控制实验从面/壁面梯度重建原始黏性通量，不做重分配；重新乘 R
后与原 dump 的相对无穷范数差为 1.39e-16。在保持其他方程不变时，这个控制
实验的稳态 Stokes 最大速度降至 0.0286007 m/s，带原 FOU 对流的稳态代数解
最大速度为 0.0285669 m/s。它们是原因诊断，不是另一套 Aphros 基线或精度验收。

继续检查原 Aphros 代码发现：`Proj::DiffusionImplicit` 对原梯度通量做隐式
求解和延迟修正，不调用 SIMPLE 路径的残差重分配。这提供了一条不改动 Aphros
数值算法的独立对照路线。`validation/aphros/twisted_projection_driver.cpp`
只负责同一曲管的几何、参数和只读导出，直接调用原静态库中的 Proj/Embed。
使用 `conv=imp, proj_bcg=1, proj_redistr_adv=0, proj_diffusion_iters=1,
proj_diffusion_consistent_guess=1`。原静态库未重建，SHA256 仍为
`b8f2bbf83c69678da30f99c79874de2989e9eef1b4a91a13f2b68a4382952732`；
驱动 SHA256 为 `0c6126d4d69e4d27708eef96f9a4c576ea81abc62e939e0114119c35b1fb07d1`。
原库中的既有 SIMPLE 专属修复在该 Proj 路径不参与更新。

`audit_twisted_projection.py` 是使用 Cirrus 实际几何算子的 Python/SciPy
小网格原型，按原 Proj 顺序进行初始压力、半步 BCG 面预测、预测通量投影、
BCG 对流、隐式扩散及最终压力投影。它目前要求均匀切割网格，并明确不构成
正式 C++ 投影求解器，也不覆盖粗细交界或近壁网格收敛。

初次 n16、dt=0.001 s、八步对比失败，速度相对差 3.99346e-6、压力
4.46599e-6、壁面剪切 1.21086e-6。独立原始函数诊断 `twisted_bcg_audit.cpp`
以非零给定场调用原 `UEmbed::InterpolateBcg`，面梯度一致到 2.89e-15，但
面预测值相差约 3.97e-4。原因是原 Aphros 初始化 is_boundary=false 后，
通过 eb.SuFaces() 标记 cut 面；该迭代器跳过 excluded 面，所以它们的标记
保持 false，梯度、通量和面积保持零。原型此前误把缺失面标为边界。
只修正原型后，给定场 BCG 差降至 1.39e-17，没有修改 Aphros 算法或放宽门槛。

修正后八步完整运行 `projection_ns16_implicit_audit_v3` 对独立 Aphros
`navier_stokes_proj_n16_implicit_probe_v1` 的比较通过：

| n16、t=0.008 s 比较项 | 相对差 |
|---|---:|
| 体积加权速度 L2 | 5.35497e-15 |
| 去压力规范后的压力 L2 | 1.11688e-14 |
| 切割单元速度 L2 | 1.55288e-14 |
| 壁面剪切向量 L2 | 2.05514e-14 |
| 面法向速度 L2 | 5.20846e-15 |
| 守恒流量 | 4.44089e-16 |

八步迭代次数均为 260、259、253、247、242、237、233、230，时间加速度
最大绝对差 5.55e-16。双方独立质量守恒通过原门槛。这里的“通过”严格限定
于同网格瞬态算法复现；该时刻仍在启动阶段，正式 C++ 实现、实际壁面加密、
时间步收敛和物理空间网格收敛仍需继续完成。原隐式 64→128 失败的证据仍有效。

`make_twisted_baseline.py --fluid-solver proj` 使用单独的案例名称和元数据；
原 SIMPLE 比较器拒绝混用 Proj。默认 SIMPLE 配置与 e6c4a98 生成器逐字节
一致，既有 n16 exp 八步比较仍通过。新的 `compare_twisted_projection_audit.py`
检查完整运行、物理时间、源文件/二进制哈希、几何对应、守恒及全部场误差，
并保留失败的原型报告。证据保存在 `results/projection_operator_checkpoint`。

复现入口（目录必须是新的）：

```powershell
./validation/aphros/build_twisted_driver.ps1 -Source validation/aphros/twisted_projection_driver.cpp -OutputDirectory D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_proj_build -ExecutableName twisted_projection_driver.exe
python scripts/make_twisted_baseline.py --ny 16 --convection --fluid-solver proj --time-step 0.001 --time-steps 8 --suffix _proj_reproduce
./scripts/run_twisted_baseline.ps1 -CaseDirectory D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_proj_n16_proj_reproduce -Executable D:/Dropbox/Agent-simulation/twisted-baseline/reproduce_proj_build/twisted_projection_driver.exe -UseAmg
python scripts/run_twisted_solver.py --config output/twisted/config_operators_ns16_exp_v3.json --exe output/twisted/simple_channel_operator_audit_v3.exe --dump-operators
python scripts/audit_twisted_projection.py --operators output/twisted/operators_ns16_exp_v3 --baseline D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_proj_n16_proj_reproduce --output output/twisted/reproduce_projection_audit
python scripts/compare_twisted_projection_audit.py --audit output/twisted/reproduce_projection_audit --aphros D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_proj_n16_proj_reproduce --output output/twisted/reproduce_projection_comparison.json
```

算子配置中的 output 也必须改成新的目录，并让下一条命令使用相应路径。
本检查点还启动了 n32 的八步独立投影基线，以及 n16、dt=0.005 s、128 步的
长时间推进；它们的最终结果尚未纳入本节验收。

其中 Python n16 长时间原型已经完成至 t=0.64 s：最大速度 0.0286081 m/s，
末步时间加速度 L2 为 1.12657e-12 m/s²。独立读取末步状态，用原投影路径的
BCG 对流、未重分配的黏性通量及压力/体力重新计算稳态动量，L2 为
1.23595e-11 m/s²，最大值为 1.71360e-8 m/s²。这证明了指定 dt 下原型的离散
稳态动量收敛，尚不能证明时间步独立性；对应 Aphros 完整长时间基线仍在运行。

## 原生 C++ projection 与粗细接口压力修正

本节更新上一检查点的完成状态。`simple/ProjectionSolver.cpp` 已实现独立 CPU
double 投影求解器，使用实际 `HADeviceGrid` 叶单元及同一份几何；求解过程不调用
Python 原型。配置 `fluid_solver=proj`、`convection_scheme=bcg`、`momentum_mode=imp`
启用该路径，默认仍为原来的 SIMPLE。原始 Aphros `Proj` 的 BCG 预测、预测通量
投影、隐式黏性迭代、最终压力投影及初始压力更新顺序均在均匀网格中复现。
黏性使用完整面梯度，不做 SIMPLE 的黏性延迟项重分配；对流仍使用原 Proj 的
重分配算子。材料目前为该单相常密度、常黏度算例。

每步分别检查速度内迭代变化、原隐式扩散方程残差、按真实切割体积计算的
质量守恒和各完整截面流量。完成时间序列不等于达到稳态；稳态还要求重新
计算的动量残差及时间加速度均小于 1e-8。`pressure_solver` 在此路径支持
`auto`、`ldlt`、`amg`，显式选择尚未实现的 `cg` 会报错。

完整独立基线的结果如下，所有数值为相对 L2 差：

| 检查 | 速度 | 压力 | 切割单元速度 | 壁面剪切 | 截面流量 |
|---|---:|---:|---:|---:|---:|
| n16，dt=.001，8 步瞬态 | 5.56e-15 | 1.38e-14 | 1.57e-14 | 2.06e-14 | 8.37e-16 |
| n32，dt=.001，8 步瞬态 | 1.00e-11 | 2.14e-11 | 7.67e-11 | 4.35e-11 | 2.52e-12 |
| n16，dt=.005，128 步稳态 | 5.70e-15 | 2.46e-14 | 2.03e-14 | 1.92e-14 | 2.91e-15 |

比较另外独立汇总全部共享面通量，检查零壁面穿透通量、逐单元散度、全局
绝对质量不平衡及截面流量差，并验证执行文件、基线构建记录、未改动的
链接库和配置哈希。n16 稳态最大速度为 0.0286080955832 m/s，流量为
5.45727064822e-5 m³/s，动量残差为 1.236e-11，时间加速度为 1.1265e-12。
`ours_proj16_steps8_v1` 的 8 个时刻与 `ours_proj16_steady_v1` 的 128 个时刻
均已导出 PVD，并用真正的 ParaView PVDReader 逐时刻读回，与求解 CSV 对比通过。

这些同网格结果不能替代时间步或空间收敛。n16 的 dt=.005 与 .0025 两个
已收敛稳态比较，速度差 0.07227%、压力差 0.47822%、切割单元速度差 0.24325%、
剪切差 0.04618%、流量差 0.01281%。压力未通过预设的 0.25% 时间步敏感性
门槛；失败报告保留，继续计算 dt=.00125。该检查只比较相同切割单元的采样值，
不引入空间插值，也不声明空间精度已经达标。

随后 dt=.00125 的 512 步也已完成。与 dt=.0025 相比，速度差为 0.05637%、
压力差为 0.52565%、切割单元速度差为 0.21541%、剪切差为 0.03076%、
流量差为 0.00722%。第二次减半仍未通过压力门槛，两个失败报告均保留。
这说明当前投影路径的有限网格压力对时间步仍敏感；不能仅凭速度/剪切差小，
宣称完整解已经达到时间步独立性。需要继续诊断压力模态与投影分裂的影响。

自适应扩展保留每个粗细子面的唯一通量。BCG 的子面值使用两侧二次 Taylor
重构，侧面梯度和横向输运速度按开口面积汇总；压力采用完整粗细面梯度。
实际 n64 墙面加密网格为 136,632 个流体单元、21,816 个壁面单元、1,536 个
流体粗细子面。二次面值多项式检查最大误差为 1.09e-15。

最初的绝对压力求解在小切割体积上遇到了舍入误差平台：残差从 0.0757 快速
降至数 e-9 后，重复压力求解/迭代改进不再降低残差。首轮实际相对散度为
4.0444e-7，未过 1e-7 门槛。`--trace-projection` 现在可记录每次压力校正的
残差和耗时；旧 v1、v2 诊断运行已明确终止并保留原因及原始数据。

v3 在粗细接口路径改为增量通量校正：以当前共享面通量的不平衡为右端，
求解紧凑压力矩阵，按完整面梯度修正通量，并累积压力增量。迭代仍求解
同一个完整压力算子，直接检查存储通量的散度，不放宽守恒容差。首轮的
速度/压力相对差为 1.17e-13/1.15e-13，实际相对散度降至 1.3145e-9。
`check_twisted_projection_iteration.py` 同时比较预测通量、扩散后速度等
中间状态；该检查不等于收敛流场验证。

默认 SIMPLE 的 27 个数值文件保持字节一致；均匀 projection 的全部 8 步、
117 个数值文件在增加诊断和修改粗细接口后也保持字节一致。当前实际运行文件
为 `output/twisted/simple_channel_projection_v3.exe`，SHA256 为
`05f1d9bb412c45d85b3fa6d4e950896bf5e065b22c935fbf2deb553e2bc79c95`。
各版源代码和构建哈希单独保存，运行中的旧文件没有被覆盖。

目前 n64 自适应 C++ 与对应 Aphros 完整一步比较、n32 长时间稳态以及更细网格
的固定物理近壁探针和剪切收敛仍未完成。尤其不能把上述 n16 稳态一致性，或
n64 的首轮代数一致性，表述为复杂曲管的近壁精度已经验收。

可复现入口示例（复制已归档配置，并改成新的 output）：

```powershell
xmake build simple_channel
python scripts/run_twisted_solver.py --config <新配置.json> --exe <保留的已构建程序.exe> --threads 2 --trace-projection
python scripts/compare_twisted.py --ours <运行目录/step_0008> --aphros <完整独立基线目录> --transient --output <新比较报告.json>
python scripts/check_twisted_projection_timestep.py --coarse <dt运行目录> --fine <dt减半运行目录> --output <新时间步报告.json>
python scripts/export_twisted_time_series.py --run <完整运行目录>
& 'C:/Program Files/ParaView 5.13.0/bin/pvpython.exe' validation/check_twisted_paraview.py --run <完整运行目录>
```

`results/native_projection_checkpoint` 保存原生 C++、独立 Aphros、失败诊断和
对比脚本的证据。大 CSV 使用确定性 gzip 压缩；`receipt.json` 同时保存
归档字节和解压后原始字节的 SHA256。标记为 diagnostic_snapshot 的运行片段
不参与完整流场通过的声明。

## 投影迭代加速、基线缓存与仍失败的精度检查（2026-09-07）

当前 `simple/` 是 CPU 显式稀疏矩阵验证实现。它使用原生八叉树叶单元，
但压力/扩散线性求解仍使用 Eigen LDLT 或 CPU AMGCL，并未接回
`src/AMGSolver.cu` 的原生 GPU matrix-free 后端。后续必须保留原生后端的
方向：把当前离散算子实现为 GPU 上的局部模板操作，以显式矩阵的 Ax、
真实切割体积残差和完整流场作参照。仅保留八叉树存储不等于已完成这项集成。

### 原生投影的可选 Anderson 加速

`anderson_depth=0` 仍为默认值。启用后，历史状态包括速度、压力、隐式扩散
猜测和全部共享面通量，按固定物理尺度归一化；每个物理时间步重置历史。
历史组合仅用于下一次内迭代的输入。最终验收和状态 dump 来自一次完整的
原始投影更新，不直接接受未经更新的混合场。

候选必须同时满足真实切割体积质量守恒，并降低完整内迭代固定点残差。
后者包含隐式扩散方程和速度/压力校正关系，保持旧物理时间的速度与通量
不变，并保留 Aphros 在第一个物理时间步每次内迭代重算初始压力的规则。
固定点残差还作为最终收敛的附加门槛，不替代原来的速度变化、扩散残差
和质量守恒要求。

v4 在 n32 上发现历史组合会放大极小切割体积的通量舍入误差；这些候选
被守恒检查拒绝。v5 对这样的候选额外做一次小幅压力投影，同时成对更新
压力和单元速度。只接受速度/压力/面法向速度相对改动均小于 1e-10 的修正，
随后仍检查质量守恒和完整固定点残差下降。n32 八步实际接受的最大修正为
1.36047e-12。`acceleration.csv` 记录拒绝、回溯和修正幅度。

完整八步（dt=0.001 s）的检查结果如下，数值为相对 L2 差：

| 网格 | 加速/普通总内迭代 | 对独立 Aphros 速度差 | 压力差 | 切割单元速度差 | 壁面剪切差 |
|---|---:|---:|---:|---:|---:|
| n16 | 400 / 1961 | 2.3064e-10 | 2.0715e-10 | 6.0694e-10 | 3.1497e-10 |
| n32 | 389 / 1950 | 2.2275e-10 | 3.4230e-10 | 1.4734e-9 | 7.8365e-10 |

每个物理时间步均与普通原生投影比较了速度、压力、切割单元速度、壁面
剪切和全部共享面通量。独立 Aphros 最终场的比较同时包括双方实际体积
质量守恒。v4 关闭加速时的 117 个数值文件与前版逐字节一致；v5 的新增
修正仅在启用加速并遇到不守恒候选时执行。实际 v5 程序 SHA256 为
`d7ca9420528e94f79dd5231472cb1729b105e4feaf32f9d11c3c2c762000de4b`。

n64 自适应包含 136,632 个流体单元、1,536 个真实流体粗细交界面。
v4 加速单步在 52 次内迭代完成，普通 v3 为 270 次；完整单步内部对比通过。
对独立 Aphros 的速度、压力、近壁速度、剪切、截面流量和全部面通量检查
通过，但整体比较仍失败：Aphros 的真实切割体积质量残差没有通过门槛。
这是 t=0.001 s 的瞬态启动状态，不能作为稳态或网格收敛证据。

### 只缓存完全相同的 Aphros 线性系统准备结果

原 Aphros Proj 用同一个线性求解对象处理半时间步压力、完整时间步压力和
隐式扩散。已有可选后端只保留一个矩阵，交替调用时反复构建 AMG 层次。
`APHROS_TWISTED_FACTOR_CACHE` 默认 1，允许设置 1..8；矩阵的压缩值、列/行
索引和后端类型必须全部相同才能命中。按最近使用顺序淘汰，不改变系数、
右端、零初值、线性容差或原矩阵残差检查。

这里的 AMG 情况是复用层次和预条件器，不能称为 LU 分解。只有选择直接
后端时才复用 LDLT/LU。`prepare_twisted_linear_cache.py` 复制原始
linear.cpp/linear.ipp/linear.h，并替换已有可选后端的头文件；通过额外
翻译单元链接，未重建或覆盖原始 Aphros 静态库。原 Proj 算法仍来自
SHA256 为 `b8f2bbf83c69678da30f99c79874de2989e9eef1b4a91a13f2b68a4382952732`
的同一静态库。

n16 完整八步中，容量 1 和容量 3 都与原始程序的全部保留 CSV、1961 条
迭代记录逐字节一致。10,065 次线性调用中，AMG 构建次数从 5,884 降到 3。
两个带详细 trace 的并行运行耗时约 743 / 450 秒；较早的无 trace 原始运行
为 304 秒。不同负载和日志开销使这三者不能直接作为硬件性能基准。

```powershell
python validation/aphros/prepare_twisted_linear_cache.py --aphros <Aphros目录> --output <新源代码目录>
./validation/aphros/build_twisted_driver.ps1 -Source validation/aphros/twisted_projection_driver.cpp -ExtraSources <新源代码目录/linear.cpp> -OutputDirectory <新构建目录> -ExecutableName twisted_projection_cache.exe -IncludeDirectories @('<Eigen头文件目录>','<AMGCL目录>') -Defines APHROS_TWISTED_HAVE_AMGCL
./scripts/run_twisted_baseline.ps1 -CaseDirectory <新算例目录> -Executable <新构建目录/twisted_projection_cache.exe> -Threads 2 -UseAmg -FactorCacheEntries 3
python scripts/check_twisted_projection_cache_pair.py --reference <完整原始运行> --candidate <完整缓存运行> --output <新报告.json>
```

### 不通过的物理精度检查

原始 Aphros n64 单步实际体积质量残差为 6.00334e-7，门槛仍为 1e-7。
总体积分散度很小不能豁免这个局部失败。把线性容差收紧至 1e-16 后，
原矩阵检查在 5.32907e-15 的残差下失败，失败日志保留；1e-15 的新复核
已启动，尚未用作通过证据。

此前面转移检查按每个周期面自己的通量归一化，质量检查则使用周期面的
无穷范数。一个约 -4.223e-16 m³/s 的面因 8.48e-27 的差异被前者误判。
现将转移检查统一为已有的 1e-12 相对无穷范数标准，仍报告原始差异。
全周期面最大差 1.1167e-23 m³/s，相对无穷范数为 2.8394e-15。分别使用
两份周期副本重算质量，最大实际体积残差均为 6.00334e-7；整体失败保留。

n32 原生投影 dt=0.005 s 的 128 步已完成，t=0.64 s 达到时间稳态。
其最终 solution.vtu / walls.vtp 已由实际 ParaView 读取，速度、压力、
壁面剪切与 CSV 逐项完全相同。但 n16→n32 的空间收敛检查不通过：
流量变化 9.2711%，最近壁面 1.953125 mm 探针的速度差 32.8915%，
壁面剪切差 5.0718%。压力探针差约 1.73%～2.20%，采样敏感性也未全部通过。
网格检查增加压力及压力采样敏感性，门槛分别为 1% 和 0.25%；原有门槛未放宽。

n16 的两次 dt 减半压力检查仍不通过。新增耦合线性诊断固定已完成流场的
动量源，仅改变压力/面通量关系中的 dt 系数；它重现了大部分压力变化的
空间方向。该诊断的源由当前解反推，不能作为独立 CFD 验证或正确物理解。
更小 dt=0.000625 s、1024 步的原生运行继续进行。

`results/projection_acceleration_checkpoint` 保存上述通过及失败证据、
不可变构建来源与脚本。整体目标仍未完成：原生 GPU 后端集成、细网格/时间步
近壁收敛和独立细网格守恒验收仍需推进。

## 原生 GPU 紧致算子检查（2026-09-07）

`Ax=b` 是线性方程的数学表示，不要求组装全局稀疏矩阵。
当前 CPU `ProjectionSolver` 有两个隐式系统：

- 压力：`A = -B diag(area) N`，未知量为压力冲量 `dt/rho * p`；
  `B` 对共享面通量求和，`N` 为紧致法向梯度。粗细交界的高阶修正
  通过完整面通量的增量投影继续迭代。
- 黏性：`A = rho/dt * diag(volume) + mu * K`，分别求三个速度分量；
  曲面壁面重构和高阶交界项还通过 deferred correction 处理。

原有 `src/AMGSolver.cu` 在 PCG 内调用 `AMGFullNegativeLaplacianOnLeafs`，
由 GPU tile 邻接及局部系数计算 Ax，没有全局 CSR 矩阵。新的 CPU 路径
目前使用 Eigen 显式矩阵及 LDLT/AMGCL；它并未自动使用原有 GPU AMG。
缓存 AMG 层级和预条件器也不等同于做 LU/LDLT 分解。

新增 `simple/NativeCompactGpu.cu` 开始接回原生拓扑：

- `nativeDeviceGrid` 返回现有 mesh 保有的 HADeviceGrid，未生成替代网格。
- 同层面使用原生 tile 的六向邻居指针及三个负向面系数场；x 周期面
  包装相应邻居指针。只对粗细交界保存共享子面的两端和系数。
- 每个原生流体叶单元直接累加 `area/distance * (x_i-x_j)`。
  粗细子面只计算一次通量，对两侧作相反号的原子累加。
- 压力壁面为零法向通量；紧致黏性壁面为 `area/distance * x_i`，
  隐式质量项为 `rho/dt * actual_fluid_volume * x_i`。
- 系数和工作向量存为 double 的独立字段，原有 float tile 提供拓扑。
  借用的 tile 类型、序号、邻接和通道由 RAII 恢复，包括初始化异常路径。
- 当前限定嵌入边界、x 周期及静止无滑移管壁；不支持的其他边界会拒绝。

`native_compact_gpu_audit` 单独用相同几何组装 CPU 显式压力/黏性矩阵，
把常数、周期光滑场、确定种子的随机场分别交给两条实现，保留所有 Ax
和差异 CSV。常数压力的 GPU 结果要求严格为零，避免用接近零的 CPU
舍入噪声作相对误差分母。其余检查门槛为
`max(abs(error)) / (max(abs(A) row_sum) * max(abs(x))) < 1e-12`。

复现命令（输出目录必须不存在）：

```powershell
xmake build native_compact_gpu_audit
./output/build_compact_rows/windows/x64/release/native_compact_gpu_audit.exe <已有嵌入管道config.json> <新的输出目录>
```

v2 在 n16、n32、adaptive64、adaptive128 上的全部 24 次算子检查通过。
最大缩放误差分别为 2.301e-16、3.078e-16、3.575e-16、3.965e-16；
adaptive64 覆盖 136632 个流体叶单元和 1536 个粗细共享子面，
adaptive128 覆盖 790216 个流体叶单元和 49152 个粗细共享子面。
128 此处也只是给定常系数的紧致算子检查，未解决原始 Aphros 在四个
极小面的物性插值异常或认证完整 128 投影流程。结果报告及构建来源保存在
`results/native_gpu_operator_checkpoint`，完整逐单元 CSV 留在
`output/twisted/native_compact_gpu_*_v2`，报告保存其 SHA256。

这个通过仅证明紧致 Ax 实现一致。极小切割体积会放大舍入差异，报告
另列 `actual_cut_volume_error_linf`，不能用上述通过替代流动质量守恒。
完整壁面拟合、非正交高阶修正、GPU 向量迭代及原生 AMG 预条件器尚未
接入。当前每次 apply 还拷贝主机/设备字段，不能据此报告求解器加速。
原生 AMG 的粗细插值与此紧致算子不同，不能只替换调用名称就认为等价。

同期两个物理验收失败也保留：n16 的第三次 dt 减半（0.00125 到
0.000625 s，最终均为 t=0.64 s）压力差为 0.5950%，超过 0.25% 门槛；
Aphros n64 的 1e-15 容差复核在 velocity0 的原矩阵残差
1.021405e-14 处退出，不能作为已通过的细网格对齐结果。

## GPU 线性求解接入完整投影（2026-09-07）

新增配置 `"linear_backend": "native_gpu"`，当前用于 `fluid_solver=proj`。
默认仍为 `cpu`。GPU 模式的压力与隐式黏性线性系统都调用
`NativeCompactGpu::solve`，不会调用 CPU LDLT 或 AMGCL。每次零初值求解
仅上传 RHS、下载最终解；PCG 的解、残差、搜索方向、预条件向量和 Ax
保存在 GPU。归约只传回标量。紧致面系数、真实切割体积和对角预条件器
所需的几何系数在同一原生八叉树上缓存。

这仍是混合实现：外层 BCG、壁面/非正交 deferred 项以及显式 CPU 算子
的独立残差检查尚未移到 GPU。当前使用对角预条件器，尚未接入原生 AMG。
`projection_method.json` 明确记录 `native_gpu_pcg_jacobi` 及这个范围，
`gpu_linear.csv` 逐次记录线性迭代数、重启次数、真实相对残差和耗时。
不能据此声称已经完成了整个 matrix-free GPU 流动求解器。

压力使用与 CPU 相同的零值对称 pin；黏性加入实际体积的质量对角项
和紧致无滑移壁面项。每次 PCG 在递推残差达到目标后重新计算 `b-Ax`，
真实残差仍需达到原配置的 `linear_tolerance`。若递推值与真实值不符，
从真实残差重启，未放宽容差或质量守恒门槛。

v1 每 100 次迭代重启的策略使 adaptive64 已知压力解在 4000 次后仍有
2.1897e-6 的相对残差；32 网格实际流动也出现线性求解失败。日志保留。
v2 取消固定周期重启，保留接受结果前的真实残差检查。相同 adaptive64
已知压力解在 621 次后通过，CPU 独立残差为 9.925e-14。四种网格的
24 项 Ax 检查与 8 项已知解线性求解检查全部通过；adaptive128 含
790216 个流体叶单元、49152 个粗细子面，压力/黏性求解分别为
1120/420 次，已知解相对 L2 误差分别为 2.355e-12/4.767e-13。
这些迭代数不是孤立负载下的性能基准。

`exportNativeFields` 增加可选的 `preserveOperatorMetadata`：GPU 求解
存活时，输出仅更新原生 tile 的四个物理字段，保留算子使用的邻接、
流体掩码和序号，避免下一个时间步使用被导出操作覆盖的拓扑。CPU 默认
仍采用原来的导出方式。新的 CPU 默认路径在 n16 的完整 8 个时间步中，
115 个 CSV 与 v5 逐字节一致。

完整流动结果：n16、dt=0.001 s、8 步、Anderson depth=5。v1 和 v2
都完成并通过每步 GPU/CPU 对齐及最终独立 Aphros 对比。v2 共执行
4512 次 GPU 线性求解，记录的最大真实相对残差为 9.9943e-14；全部
8 步的速度、压力、切割单元速度、壁面剪切、共享面通量相对 L2 差异
最大为 1.0873e-12，双方均为 400 次外层迭代。相对原始 Aphros 的最终
速度/压力/切割单元速度/WSS/截面流量误差分别为
2.305e-10 / 2.072e-10 / 6.065e-10 / 3.150e-10 / 1.238e-10。
每一步均重新从共享面通量及实际切割体积检查质量守恒。

`scripts/check_twisted_time_pair.py --linear-backend-pair --candidate <GPU目录>
--reference <CPU目录> --output <报告.json>` 检查相同外层参数的完整序列；
可用 `--steps` 检查已完成的前缀，但报告不会把前缀当作完成的全序列。
没有该选项时保留原来的 Anderson/普通迭代检查。

v2 最终 `solution.vtu` 和 `walls.vtp` 位于
`output/twisted/ours_proj16_native_gpu_v2/step_0008`，已由实际 ParaView
读取，与 CSV 速度、压力和壁面剪切逐值相同。这是 t=0.008 s 的瞬态
检查，不能当作定常或空间收敛结果。32 网格的完整流动验证仍在运行，
随后还需实际粗细交界流动验证、原生 AMG、物性插值修正及近壁网格/时间
收敛。完整目标尚未完成。

## 原生 AMG 预条件与压力零空间（2026-09-07，后续进展）

GPU 路径新增 `gpu_preconditioner=native_amg`。保持原生物理叶单元，
另建压力和黏性两个 HADeviceGrid 预条件层级，向上补充 NONLEAF 祖先，
最粗 y 分辨率为 8。层级缓存局部面系数、对角和掩码，不生成全局 CSR，
不做 LU/LDLT 分解。每次预条件调用原有 `AMGSolver::FASMuCycle`，
`src/AMGSolver.cu` 未改变。压力和黏性代理的边界及质量项分别处理。
粗细面系数考虑原生 ghost 插值的 1/2 因子，并向祖先聚合。

线性求解采用 double FGMRES，重启维数 20，两次正交化；原生 AMG
使用 float。预条件器仅近似目标算子，最终由 double 紧致算子及真实
`b-Ax` 验收。Krylov 向量和 AMG 循环在 GPU 上执行，标量归约/Givens
在 CPU 上。外层 BCG、壁面重构、deferred correction 仍用 CPU 显式算子，
不能称为整个流动流程已经无矩阵。

可选 `gpu_pressure_gauge=mean_zero` 要求 native AMG，在去常数子空间
求解不固定单点的压力算子；AMG 代理仍保留一个 pin。单点 pin 仍为默认。
最终压力按真实流体体积去均值。真实残差须满足原 `linear_tolerance`。

保留的失败及修正：

- v2 遍历了 tile 数组未使用的容量，覆盖物理叶类型；v3 只读取
  `hNumTiles` 指定的有效项。
- v3/v4 小压力修正的真实残差分别停在 1.336e-13 和 2.756e-13，未通过。
  GPU 投影改为对实际共享面通量增量修正。
- v5 对原始 RHS 复算为 1.042e-13；v6 把内部目标收紧至请求值的 1/4，
  为最后验收留出余量，外部 1e-13 要求不变。
- v6/v7 在仍需修正的通量上发现微小全局不兼容。v7 记录
  `sum(rhs)=4.31549e-22`、相对兼容修正 1.30526e-13、实际散度
  0.00694955，不能直接跳过。
- v8 用共享面通量的 Neumaier 补偿求和构造单元 RHS，也补偿全局求和，
  减少大通量相消得到小修正时的精度损失。MSVC 的 long double 本身
  不提供额外精度。仅当散度已满足原来的 1e-10 内部门槛时才跳过修正。

v8 的 n16/n32/adaptive64/adaptive128 全部 24 项 Ax 和 8 项已知解检查
通过，AMG 层数为 2/3/4/5。压力迭代数 28/44/108/341，黏性
5/8/13/38；最大已知解相对 L2 误差 4.332e-12，真实残差均小于 1e-13。
128 只验证常系数紧致系统，未解决原始 Aphros 的极小面物性插值，
也不代表完整 128 投影或物理网格收敛通过。迭代数不作为独立性能基准。

在原有管道配置中加入：

```json
{
  "fluid_solver": "proj",
  "linear_backend": "native_gpu",
  "gpu_preconditioner": "native_amg",
  "gpu_pressure_gauge": "mean_zero",
  "linear_tolerance": 1e-13
}
```

`gpu_linear.csv` 逐次记录真实残差和兼容修正大小。序列比较脚本还核对
实际后端、AMG 层级、压力规范及每个物理时间步的独立质量守恒。

此前 GPU Jacobi-PCG v2 已完成 n32 的 8 步和 adaptive64 的 1 步，
分别与 CPU 的 389/52 次外层迭代一致，完整序列对齐通过。n32 与原始
Aphros 的独立比较通过。adaptive64 的场值通过，但原始 Aphros 自身
的实际切割体积质量守恒失败，因此独立比较总体仍失败。

另一个独立证据是原始 Aphros SIMPLE/FOU 的 128 定常运行已完成。
与原生 adaptive128 SIMPLE 比较，速度、压力、切割单元速度、WSS、
截面流量相对 L2 差异分别为 0.01446%、0.19636%、0.01399%、
0.01409%、0.01988%，对齐门槛全部通过。Aphros 实际质量残差 9.315e-9。
这是此前的 SIMPLE 路径，不能归功于新 GPU 投影。64 到 128 的物理
网格收敛仍未通过，时间步压力敏感性也尚未解决。

v8 原生 AMG 的 n16 已完成 dt=0.001 s 的 8 个时间步，逐步 CPU 对齐
及最终独立 Aphros 对比全部通过，双方外层迭代均为 400 次。3011 次
GPU 线性求解的最大真实相对残差为 2.497e-14，最大兼容修正为
2.713e-18；最终实际质量残差为 6.856e-15。默认 CPU 的完整 8 步
115 个 CSV 与此前版本逐字节一致。

`output/twisted/ours_proj16_native_amg_v8/step_0008/solution.vtu` 及
`walls.vtp` 已由实际 ParaView 读取，速度、压力、壁面剪切与 CSV 逐值
一致；这是 t=0.008 s 的瞬态检查。此前 n32 GPU Jacobi 结果和 128
SIMPLE 定常结果也已完成同样的读取校验。原生 AMG n32/64 的完整流场
验收继续进行，不能从上述 n16 或已知解检查外推为已通过。

`results/native_amg_checkpoint` 保存不可变构建源码、成功/失败报告、
已完成的流场及新完成的独立 SIMPLE128 参考。`receipt.json` 记录原始
字节哈希及压缩文件哈希；未归档的大型逐单元检查 CSV 另列本地位置和
哈希。正在运行的案例仅保存启动参数，不把中间状态当作完成结果。

## 极小面物性插值与 128 投影接入（2026-09-07）

`validation/aphros/twisted_material_audit.cpp` 调用原始 Interpolate 和
InterpolateHarmonic，物性边界条件与 Proj 相同。原始库 SHA256 仍为
`b8f2bbf83c69678da30f99c79874de2989e9eef1b4a91a13f2b68a4382952732`；相关
`approx_eb.h/.ipp` 和 `proj.ipp` 相对上游没有改动。

n128 的 3190604 个开放面中，4 个极小面的 `w=I(1)` 约为
0.90939679468127。单元物性仍固定；原始函数产生的面系数为
`mu_face=mu*w`、`rho_face=rho/w`，约为 0.009094 和 1.09963。
原因是排除的 Cartesian 插值样本被置零。壁面物性的最大舍入偏差为
3.799e-13。n16、n32 和已用的偏移 n64 没有这类异常开放面。

ProjectionSolver 统一将权重用于压力紧致/交界修正、压力通量修正、
隐式黏性紧致项和完整梯度通量、面力项及压力加速度。deferred 黏性仍为
完整项与紧致项之差，单元加速度保持几何面积平均。几何面积、切割体积
和速度插值保持原值；仅在原先允许的 `abs(I(1)-1)<=1e-12` 舍入范围
将权重规范为 1。NativeCompactGpu 及原生 AMG 代理接收相同的正逐面
系数，继续通过 tile 邻接/粗细共享子面计算，不组装 GPU CSR。

独立检查采用预设标量 `.03*sin(2*pi*x/period)+y*y+.2*z`。原始 Aphros
逐面计算压力/黏性通量并累加到单元；原生端读取实际组装后的压力和完整
黏性矩阵作用于该标量的结果。4 个异常面和两侧 8 个单元全部匹配：
坐标、面积、权重和密度相同，黏度相对差 1.908e-16；压力算子相对 Linf
差 3.841e-14，黏性差 1.825e-13，加速度最大差 3.192e-14。
`scripts/check_twisted_material.py --assembled` 验收完整单元算子；不带
该选项时只检查面与加速度，报告明确标注范围。Projection 支持
`operator_only=true`：初始化真实算子及后端、导出诊断后退出，不做流动
迭代，也不生成已收敛流场报告。

GPU 变系数检查在 n16/adaptive64/adaptive128 使用
`0.8+0.15*sin(2*pi*x/extent_x)`，覆盖壁面和粗细面。18 项 Ax 与 6 项
已知解检查通过，最大真实残差 5.069e-14，最大已知解误差 5.074e-12。
默认 CPU 的完整 n16 八步中，原有 115 个字段/迭代 CSV 仍逐字节相同。

原生 AMG v8 的 n32 八步和 adaptive64 一步也已完成。GPU/CPU 字段的
最大相对 L2 差分别为 1.429e-12、1.319e-12，n32 独立 Aphros 对比通过。
adaptive64 场值通过，但原始 Aphros 的实际切割体积质量守恒仍失败，
该独立比较总体为失败。两个新结果均通过实际 ParaView 读取。

新的 `ours_proj128_material_gpu_v2` 已进入真实投影迭代；另启动原始
Aphros Proj 的 128、dt=0.001、一步参考，使用已有 AMGCL 线性后端及
三项精确矩阵缓存。双方完整流动、定常对齐、物理网格/时间步收敛仍待
验收。`results/material_interpolation_checkpoint` 保存算子证据、完成的
32/64 流场、构建源码及正在运行的 128 案例的不可变启动参数。

## 压力修正的浮点停滞诊断（2026-09-07）

完成的原始 Aphros Proj n32 定常参考与 `ours_proj32_steady_v1/step_0128`
在速度、压力、切割单元速度、WSS、截面流量上分别相差约
6.03e-13、1.16e-12、4.11e-12、1.58e-12、5.81e-13（相对 L2）。
但参考的实际切割体积质量残差为 9.470e-7，超过 1e-7；本组总体失败。
最差单元体积 5.266e-17，体积分数 8.834e-10，净通量约 1.091e-23。
独立 `math.fsum` 得到相同散度，不能把该问题归因于检查脚本的求和。

`SIMPLE_PROJECTION_TRACE` 现在还输出 `projection_updates.csv`，区分完整
压力修正的理论面通量增量和实际 double 字段的变化。首次出现相同散度时，
`projection_floor_faces.csv` 记录最差单元全部相邻面，供独立重算。
这两份记录不修改压力或面通量。n16 的完整八步验证中，113 个结果 CSV
与修改前逐字节一致。

adaptive128 诊断中的一个单元体积为 5.015e-20。理论通量增量能将其
散度从约 1e-9 降到 1e-20，但三个面分别写回 double 后，又出现约
1e-9 的散度。后续全场面速度修正约 8e-20，参考面速度约 1.18e-3；
压力冲量修正约 1.7e-22，冲量范围约 2.08e-5。仅用散度相同作判断
是不够的：早期相同散度对应的全场修正仍大于舍入量级。

GPU 路径因此新增受限的退出条件：至少第 3 个压力 pass、连续两个
散度 Linf 完全相同、散度处于原先接受范围 `[1e-10,1e-8]`，且完整面
速度修正和压力冲量修正分别不超过 double epsilon 乘以对应尺度。
面速度尺度取当前最大面速度；冲量尺度取当前冲量范围与
`dt*accelerationScale*extent_x` 的较大值，后者也适用于初始冲量为零的
舍入修复。所有尺度和修正必须有限。实际面通量及其非零残差保留，不做
截断、清零或重新分配。线性真实残差、完整外层固定点、最终相对质量
残差 1e-7 和独立 Aphros 比较门槛均不变。CPU 求解路径不使用该条件。

`projection_roundoff_exits.csv` 记录每次这类退出的全部标量依据；
`scripts/check_twisted_projection_floor.py` 保存完整行快照，以 `math.fsum`
重算局部通量平衡，检查实际加法结果及退出条件。报告只验证诊断算术，
不把退出、某个时间步或未完成运行称为定常流结果。

原始 Aphros Proj 共用一个线性求解器实例处理压力和速度。验证后端新增
可选 `APHROS_TWISTED_PRESSURE_TOLERANCE`，只收紧名为 pressure 的系统，
不允许比原先请求容差更松。harness 参数为 `-PressureSolveTolerance`，
默认清除该环境变量并记录实际设置。默认关闭时，n16 八步的 8 个 CSV
和 1961 条迭代记录均与旧可执行文件一致。原始 Proj/Embed 及静态库未变。
一起收紧所有方程至 1e-15、以及仅收紧压力至 1e-15 的 n32 尝试均触发
原始矩阵残差保护，失败结果保留；不能通过放宽最终物理检查将其改为成功。

压力专用容差 2e-15 的 n32 运行已保存第二个时间步（t=0.01）的完整
检查点，实际质量残差仍为 2.735e-6，总体失败。仅收紧线性容差尚未
解决此问题。`snapshot_twisted_reference.py` 支持 Proj 和 SIMPLE，
以“下一物理时间步已开始”及复制前后字段字节相同为屏障；浮点累计时间
采用相对 1e-12 的一致性检查，不伪造整个参考进程的完成记录。

128 的完整 GPU/参考流场、定常结果，以及物理网格和时间步收敛继续验收。

带停滞判据的 v2 已完成 n16 八步和 adaptive64 一步，分别与 CPU 的
400/52 次外层迭代一致，逐步场值及实际质量检查通过。adaptive64 的
最大字段相对 L2 差为 4.021e-13，真实线性残差最大 2.533e-14，实际
相对质量残差 1.460e-9。这两组未触发舍入退出，覆盖普通路径回归。
adaptive64 的独立 Aphros 比较中，所有场值和共享面通量通过，参考自身
质量检查仍失败。新 `step_0001/solution.vtu` 与 `walls.vtp` 均已由
ParaView 实际读取验证；物理时间为 0.001 s，并非定常结果。

adaptive128 v2 已实际触发舍入退出，前 3 次压力调用分别使用 16、13、
15 个 pass。独立算术检查验证了修正尺度、相邻面加法及实际残差。
第一轮完整原始外层更新与旧 material GPU v2 比较，速度、压力、共享
通量相对 L2 差分别为 1.452e-15、3.629e-16、4.191e-16；两边独立
相对质量残差均为 1.466e-8。检查还覆盖中间对流速度、扩散速度、
加速度及预测通量，全部通过。`check_twisted_projection_iteration.py`
按块读取数据，使用体积加权单元比较与实际共享面散度，不要求整个运行
已完成；下一轮 dump 或已刷盘的本轮加速记录提供写入完成屏障。
上述证据仅覆盖一个中间更新，128 的完整时间步及独立参考仍未完成。

`results/pressure_roundoff_checkpoint` 保存诊断、成功和失败记录、完成的
16/64 数据、构建源码及未完成任务的启动信息。大型 128 第一轮 CSV
保留在本地，receipt 单独记录路径和哈希，不冒充已归档的最终结果。

## 原始压力方程与实际切割体积（2026-09-07）

`-DumpPressureSystem` 在验证线性后端返回时保存原始七点压力方程；独立
driver 必须逐单元确认它的解就是最终 `GetPressure()`，才写出
`proj_final_b0_pressure_rows.csv`。其中包含原始对角、邻接系数、常数项、
压力与邻居 halo 压力、实际流体体积，以及被固定的压力自由度。
`check_aphros_pressure_snapshot.py` 用 Decimal 精确累加这些 double 系数的
乘积，并另从最终共享面通量重算质量守恒。n16 和 n32 的捕获开关分别
保留了全部 8 份字段/几何 CSV 和 1961/545 条迭代记录的字节一致性。

n32、dt=0.005、第二步的原始矩阵残差除以规则单元体积为 8.438e-15，
与后端报告一致；同一残差除以实际切割体积后达到 1.925e-7，实际面通量
给出 1.934e-7 /s。压力 halo 完全一致。70 个微小非对称非对角项均位于
周期 x 接缝，最大相对差为 2.602e-15；这是独立发现，尚未证明它是主因。

离线重放显示，原先按单元数平均去除兼容性残差，会向所有单元加入同样的
积分修正。它在极小切割体积上对应约 1.9e-7 /s。改成
`b_i -= sum(b) * V_i / sum(V)`，在相同矩阵、压力初值、固定自由度和 LU
因子下，该修正的散度约为 2.0e-16 /s；六次迭代精化后的实际体积方程
残差从约 2.1e-7 降到 1.8e-9 /s，压力最大变化 1.9e-16。
此离线重放仅诊断线性系统，不生成或替换完整流动结果。

可选 `-VolumePressureCompatibility` 将这个权重用于验证后端的压力
求解与迭代精化。实际体积由独立 driver 从原始 Embed 读取，经
`twisted_pressure_geometry.h` 传给线性后端。它同时要求原有的规则体积
兼容性检查和 `abs(sum(b))/sum(V) <= tolerance`，保留原始 RHS 和全部
原始矩阵行的最终残差检查。`pressure_compatibility.csv` 逐次记录总量、
体积和容差，供独立验证。选项默认关闭，Aphros Proj/Embed 算法和共享
静态库不变；也没有改变最终场值、近壁和质量验收门槛。

这里的 LU 仅用于离线诊断/CPU 对照。Cirrus 原生 GPU 紧致压力和隐式
黏性系统仍用 matrix-free 算子与原生 AMG；外层 CPU 稀疏算子的迁移、
完整定常/网格/时间步收敛继续属于未完成工作。

启用体积权重的 n32 两步（dt=0.005，t=0.01，压力专用容差 2e-15）
已从零初值完整重跑，并通过与 `ours_proj32_steady_v1/step_0002` 的
速度、压力、切割单元速度、壁面剪切和流量对比。上述相对 L2 差分别为
1.102e-14、1.879e-14、7.118e-14、6.775e-14、7.735e-15。参考实际
相对质量残差为 3.514e-8，低于未改变的 1e-7 门槛；6860 次压力兼容性
修正的最大散度为 2.394e-16 /s。该比较是瞬态两步验收，不是定常或网格
收敛证明。默认关闭新选项的 n16 八步也通过回归：9 份 CSV（含原始压力
方程）和全部 1961 条迭代记录保持逐字节相同。

旧的仅收紧压力容差运行保留了第 35 步、t=0.175 的完整检查点，其实际
相对质量残差仍为 1.044e-6。确认上述原因及修正后的完整对比通过后，于
2026-09-07 21:47:17 UTC 停止旧运行；中断原因、进程标识、检查点和失败
质量报告均保留。没有将中断运行记为完整定常结果。

后续 n32 原始压力容差 1e-13 的体积权重参考也完成了两步，并通过
全部场值和实际质量检查；不必为这个两步算例收紧到 2e-15。
相同配置的 t=0.64 定常过程参考仍在运行。
n64 原始容差 AMG 参考已完成一个 dt=0.001 时间步；实际相对质量
残差为 1.2748402e-7，仍高于 1e-7 门槛，因此完整 Aphros 对齐失败。
它的原始规则体积归一化残差为 6.21725e-15，但最小切割单元的实际
体积只有约 5.079e-18。原始压力方程、面通量和失败报告均保留；没有
以规则体积残差或场值一致替代实际质量验收。

## 将完整压力接口纳入 GPU 算子

可选 `gpu_pressure_operator="full"` 将粗细子面的既有高阶压力梯度直接
纳入 GPU 的 Ax；默认仍为 `compact`。新选项要求
`linear_backend="native_gpu"`、`gpu_preconditioner="native_amg"` 和
`gpu_pressure_gauge="mean_zero"`。物理模型、时间推进、面通量修正公式、
线性真实残差门槛和最终实际切割体积质量检查不变。

规则区域仍通过原生 HA tile 邻接计算紧致面通量。每个粗细子面保存其
局部高阶梯度采样索引和系数，替换原来的两点压力通量；同一个子面只
计算一次通量，再以相反符号加到两侧单元。梯度使用相对于 owner 压力
的差值计算，使常数压力零空间严格成立。GPU 不组装全局 CSR 矩阵。
原生 float AMG 仍近似求解紧致算子，外层 double FGMRES 的 Ax 和最终
真实残差则使用完整压力接口。隐式黏性路径及其预条件器保持原状。
CPU 仍构造参考/外层算子、壁面和黏性延迟修正以及输运项，因此不能把
这一阶段称为整个流体算法都已迁到 GPU。

独立算子检查在 n16、adaptive64 和 adaptive128 通过；后两者均采用
变化的面系数，覆盖常数、光滑周期和随机场，以及已知解线性求解。

| 检查 | adaptive64 | adaptive128 |
| --- | ---: | ---: |
| 实际流体单元 | 136632 | 790216 |
| 完整高阶粗细子面 | 1536 | 49152 |
| Ax 最大归一化差 | 4.000e-16 | 6.750e-16 |
| 压力真实相对残差 | 1.535e-14 | 2.171e-14 |
| 压力已知解相对 L2 差 | 9.125e-13 | 5.654e-12 |

GPU 常数压力零空间严格为零。归一化 Ax 差是算子比较，不能替代
除以实际切割体积后的物理散度检查；全部逐单元检查输出保留在
`output/twisted/full_pressure_operator{16,64,128}_v1`。

`ours_proj64_full_pressure_gpu_v1` 已完成一个 dt=0.001 时间步，与同一
CPU 对照 `ours_proj64_adaptive_aa5_v4` 都使用 52 次外迭代。全场速度、
去规范压力、切割单元速度、壁面剪切、共享面通量相对 L2 差分别为
3.840e-13、7.502e-14、1.803e-12、1.424e-12、3.361e-13；实际质量、
全部内迭代固定点和真实线性残差检查通过。默认 compact 路径的 n16
八步回归也通过。这些结果证明替换线性后端保留已验证的离散结果，
不构成独立 Aphros、定常或物理网格收敛证明。

同一 64 级问题的 compact GPU 压力求解共 2465 次、246024 次 Krylov
迭代；full 模式为 592 次、78305 次。黏性均为 156 次求解、2028 次
迭代。双方均通过对完全相同 CPU 参考文件的比较。由于运行时有其他
计算并发，这里只比较工作量，不将时间日志当作受控速度基准。

验证以上完整 64 级流动及 128 级算子后，在 2026-09-07 22:40:35 UTC
停止旧的 `ours_proj128_pressure_roundoff_v2`（最后完成第 22 次外迭代），
释放内存并启动 `ours_proj128_full_pressure_gpu_v1`。原始进程身份、
停止依据及中间 dump 均保留；旧运行没有完成物理时间步，也不记作
收敛结果。新运行第一轮的全部 790216 个单元和 2431944 个共享面已
与旧运行的不可变 dump 对比通过：速度、压力、面通量相对 L2 差分别
为 1.837e-15、3.185e-16、4.773e-16，实际质量残差双方均约 1.466e-8。
报告为 `results/full_pressure_n128_iteration1_pair_v1.json`。这仅验证
一次中间更新，新运行仍需完成流动及独立基线检查。

## GPU 完整隐式黏性

可选 `gpu_viscosity_operator="full"` 将黏性算子的壁面、切割开口面和
粗细子面全部纳入一次隐式求解，要求 `linear_backend="native_gpu"`
和 `gpu_preconditioner="native_amg"`。默认仍是 `compact`：保持原来的
紧致隐式求解和外层延迟修正。两个选项求解同一个收敛离散方程：
`(rho*V/dt + mu*K_full) * u_diffused = rho*V/dt * u_intermediate`。
full 模式不再从 RHS 减去旧迭代的黏性延迟项；压力和速度耦合的外层
迭代、BCG 输运、完整固定点及实际质量验收保留。

`NativeCompactGpu::configureViscosityFaces` 保存各特殊面的原始局部
梯度。规则面继续使用原生 tile 邻接；切割规则面的紧致系数被专用
掩码跳过，粗细紧致通量和紧致壁面对角项也被完整面梯度替代，避免
重复计入通量。内面的一个共享黏性通量以相反符号进入两侧单元；
无滑移壁面的通量只进入 owner。内面梯度使用 owner 相对速度，壁面
梯度直接使用速度值，以保留齐次 Dirichlet 条件而不是误设零梯度。
GPU 没有构建全局 CSR 或进行 LU/LDLT 分解，原生紧致 float AMG
仍作为完整 double FGMRES 算子的预条件器。

黏性使用 `EmbeddedOperators::faceGradient` 的切割面形心插值和
`wallGradient` 的原始壁面拟合。它与压力投影在同级切割开口面上的
两点法向梯度不同，不能简单地把完整压力算子当作完整黏性算子。
构建模板、CPU 外层输运和诊断仍然存在，尚未实现整个推进过程都在 GPU。

独立检查在 CPU 上从**全部面**组装完整黏性矩阵，再对比 GPU 中规则
tile 加局部特殊面模板的结果，以检查特殊面的遗漏或重复。n16/n64
的变化面系数检查通过，分别覆盖 4104/66984 个完整黏性特殊面；
已知黏性解的相对 L2 差为 8.043e-12/1.477e-12，真实残差为
5.874e-14/7.361e-14。此处仅为算子和线性求解证明。

完整流动比较使用 `check_twisted_time_pair.py --linear-backend-pair
--implicit-viscosity-pair`，明确区分 GPU 完整隐式求解和 CPU 延迟迭代。
它检查相同时间离散问题的每个收敛步骤、全部场值及共享面通量，并
核实实际 GPU 特殊面数量；不会要求不同求解方式具有相同中间迭代。

完整 n16 八步（dt=0.001）和 adaptive64 一步（dt=0.001）均通过 CPU
对照；外迭代数分别由 400 降为 66、由 52 降为 8。n64 的速度、压力、
切割单元速度、壁面剪切及面通量相对 L2 差分别为 2.104e-10、6.987e-11、
1.984e-9、1.643e-9、1.302e-10；完整固定点残差为 6.212e-11，实际
连续性相对残差为 1.849e-11。n16 全部八步的上述场值差也均低于
2.471e-9。场值差在现有停止准则内通过，不主张逐位相同。
默认关闭 full 黏性的 n16 八步回归也已完成并通过全部 CPU 后端检查，
双方均为 400 次外迭代；新选项没有替换默认求解路径。

n64 压力求解次数从 592 降到 86，压力 Krylov 迭代从 78305 降到
13538；黏性从 156 次/2028 次迭代降到 24 次/1624 次迭代。n16 黏性
Krylov 总数从 6000 增至 9590，但压力从 50219 降至 7848；完整算子
的单次求解更贵，比较时必须统计全部求解工作。这里没有把并发运行的
时间日志解释成受控性能测试。

新 n64 时间步的 `solution.vtu`/`walls.vtp` 已由实际 ParaView 读回，
所有速度、压力、壁面剪切值与 CSV 完全相同。n64 的 native 和原始
Aphros 两条 dt=0.005、128 步、t=0.64 的从零初值轨迹已经启动。
它们需要独立完成定常、实际质量和网格验收；不能以本节 CPU 后端对比
代替 Aphros 对齐，也不能以 t=0.001 的已完成瞬态时间步冒充定常结果。

## 输出间隔与已完成的独立定常对比

`output_stride` 默认 1，仅 Proj 支持大于 1 的值。所有物理时间步仍
执行相同推进、原生 GPU 物理通道同步和完整诊断；仅在第一步、最后
一步、间隔倍数步保存场 CSV、原生二进制及所选中间迭代。未收敛步
仍保留完整最终场。固定网格 CSV 与根目录文件建立硬链接，文件系统
不支持时复制。没有改变方程、停止准则或用于质量检查的面通量。

同一不可变程序的 n16、dt=0.001、八步测试比较默认输出和 stride=3：
全部步骤的迭代数和诊断一致，保留步骤 1、3、6、8 的最大场值相对
L2 差为 5.438e-16。默认输出还通过原 CPU 时间序列检查。实际
ParaView PVDReader 读回四个时刻 0.001、0.003、0.006、0.008，
速度、压力和壁面剪切与对应 CSV 完全相同。失败步强制输出路径本次
未通过人为制造失败来执行，不把正常八步测试当作该分支的验证。

原始 Aphros Proj/Embed 的 n32、dt=0.005、128 步运行已完成，使用
此前记录的体积加权压力兼容性适配器和原始 1e-13 线性容差。t=0.64
的 Cirrus 与独立参考通过全部定常和实际质量检查：速度、压力、切割
单元速度、壁面剪切及共享面法向速度相对 L2 差分别为 6.032e-13、
1.165e-12、4.106e-12、1.583e-12、5.872e-13。原生/参考实际质量
相对残差分别为 1.783e-8、1.139e-8。该网格为 19000 个流体单元、
零粗细交界；第一次错误附加 `--adaptive` 的报告因此失败并保留，
正确的均匀网格严格 1e-6 阈值对比通过。这仍不证明自适应网格收敛。

原始 Aphros 参数 `proj_diffusion_iters=8` 与默认 1 的 n32 两步运行
使用相同程序、物理设置、时间步及停止准则，只有该参数不同。两者
实际质量检查通过，最终五类场值差均低于 1.617e-9；外迭代总数从
545 降为 101，但推导的三个速度分量黏性求解总数从 1635 增为
2424。这是原程序的黏性内迭代设置，并非改变空间离散或壁面模型。
对比脚本新增显式 `--aphros-diffusion-iterations 8` 并核实实际配置，
不会自动放行不同参数；缺省调用仍拒绝该运行。该 N8 参考也通过
Cirrus n32 第 2 步对比；缺省 N1 的最终定常对比回归通过。

完整黏性 GPU 的 n128 算子检查完成：790216 个流体单元、49152 个
粗细子面和 310896 个特殊黏性面。压力/黏性已知解相对 L2 差分别
为 5.654e-12、1.224e-12，真实残差分别为 2.171e-14、8.692e-14。
据此和完整 n16/n64 流动对照，于 2026-09-07 23:49:07 UTC 停止
旧的 n128 紧致黏性运行并启动完整黏性 GPU 运行，原始证据全部保留。
旧运行未完成物理时间步，停止不是定时重启，也没有作为成功结果。

旧的无体积兼容性修正 Aphros n128 运行虽正常退出，实际质量相对
残差 1.968e-5，未通过 1e-7 阈值；不能用其很小的体积 L2 残差代替
最大局部守恒检查。新的 n128 参考使用已验证的兼容性适配器、N8
内迭代及压力方程 dump，原始线性容差仍为 1e-13。完整 n64/n128
定常自适应对齐及网格、时间步收敛仍未完成。以上运行与结果归档于
`results/output_schedule_checkpoint`，归档明确区分完成、失败和在运行输入。

## 及时检查 GPU 压力方程的原始右端

旧 n64 定常运行的不可变日志前缀中，614 次压力求解累计 554594 次
Krylov 迭代；其中 125 次已满足原始 1e-13 真实残差要求，却因为内部
追求 2.5e-14 的兼容右端残差而迭代到 4000 次。日志前缀只说明这个
耗时机制，不作为完整运行或定常结果。

`NativeCompactGpu` 现在额外保存 N 个 double 的原始缩放右端。在每次
FGMRES 重启的完整 stencil `Ax` 计算后，直接检查原始 `b-Ax`；达到
用户配置的原始容差即可返回。若尚未达到要求，兼容右端的 Krylov
残差不受该检查影响，继续迭代。已通过的原始右端检查复用同一次新算
出的 `Ax`，不重复计算；没有使用 Arnoldi 估计残差代替真实残差。
压力兼容性、实际面流量和外层固定点阈值均保留，黏性停止路径不变。
`gpu_linear.csv` 记录原始右端检查次数和是否据此接受。

n16/n64 的全部面算子及已知解检查通过；n64 压力的 CPU/GPU 真实
相对残差分别为 9.324e-14/9.321e-14，已知解相对误差 7.762e-12。
完整 n16 八步和 adaptive64 一步均通过原 CPU 后端对比，五类场值
最大相对 L2 差分别低于 2.471e-9、1.985e-9。n64 压力总迭代从
13538 降到 9601，单次最大值从 4000 降到 120；压力求解次数均为
86，黏性均为 24 次/1624 次迭代。n16 工作量保持相同。这里比较
实际工作量，不把并发运行时间解释成受控速度基准。

原 n64 Aphros N1 定常运行的第一步已保存为通过快照完整性验证的
不可变前缀；其实际质量相对残差 3.639e-7，仍未达标。原始压力方程
诊断可复现该误差，体积兼容性修正量的散度仅 3.779e-16，说明还需
处理压力方程精度及最终面通量的浮点误差，不能声称兼容性调整已解决
所有细网格守恒问题。2026-09-08 00:10:39 UTC 依据已完成的 N8/N1
对照停止旧 N1 长轨迹，并启动相同物理设置和容差的原始 N8 轨迹；
旧轨迹没有完成 128 步，不能计为定常成功。N8 只是原程序内迭代设置，
不预先声称它解决了实际质量问题。

新的 native n64 定常轨迹使用及时原始右端检查和 `output_stride=4`。
新旧 native 轨迹的完整首步已进一步对齐：速度、压力、切割单元速度、
壁面剪切与面法向速度的相对 L2 差均低于 9e-15，两者实际质量相对
残差均约 3.306e-11。依据该结果，旧轨迹于 2026-09-08 00:35:01 UTC
停止，保留其前三个完整物理步和未完成的第四步；它不能计为完整定常
轨迹。上述及时原始右端检查、不可变
编译源码和失败参考诊断保存在 `results/original_rhs_checkpoint`；
独立自适应定常对齐及网格、时间步收敛仍是未完成的验收项。

## 原始压力面表达式诊断与 GPU128 首步

`results/pressure_faces_checkpoint` 保存只读原 Proj 捕获版本及其验证。
新增捕获只读取原始面表达式 e0/e1/b、两侧压力与原始输出通量；最终
dump 前还逐面核对它们与求解器最终字段完全一致。n16 完整八步中，
开/关捕获以及重新编译的 Proj/原静态库两组均逐字节一致，包括全部
1961 次外迭代、速度、压力、壁面剪切、守恒通量与压力方程。捕获表达式
可逐位复算原面通量；这验证诊断不会改变该算例的参考结果。

n64 原始 N8 定常轨迹的首步质量相对残差仍为 2.793e-7，未达到
1e-7 阈值。另一个相同物理首步实验将压力容差收紧到 2e-15，因原矩阵
残差 3.730e-14 超过该后端检查要求而失败；没有产生完整物理步。
这些失败同样留档。n64 面表达式捕获已结束，原表达式逐位一致；压力差
形式和更高精度乘积的离线试算仍未通过质量门槛，详见
`PRESSURE_PRECISION.md`。诊断脚本的替代算术
只在内存中计算，不会覆盖参考 CSV，也不能作为实际守恒通过证据。

GPU128 的 full pressure/full viscosity 运行已完整结束：790216 个流体
单元、49152 个粗细网格子面、87248 个切割单元。dt=0.001 的首步在
8 次外迭代后通过，完整固定点残差 3.136e-11，实际质量相对残差
3.671e-9，截面流量相对极差 8.361e-15。但时间加速度/稳态动量残差
仍为 0.8127/0.8190，不能将此结果标记为定常或网格收敛。

结果位于 `output/twisted/ours_proj128_full_viscosity_gpu_v1/step_0001/`。
`solution.vtu` 的 Velocity/Pressure 和 `walls.vtp` 的 WallShear 均已
用实际 ParaView 5.13 pvpython reader 读回，与原求解器 CSV 逐值一致。
完整几何、原始场与二进制结果保留在运行目录，归档记录其哈希；编译
源码、运行记录、检查报告和不可变参考快照元数据进入 Git。独立的
adaptive64/128 定常流场对齐及物理网格/时间收敛验收仍未完成。
