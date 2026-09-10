# SIMPLE 窄通道验证结果

**11/11 组验证通过，三组细化检查通过。** 验证日期：2026-09-06。最终审计确认所有算例来自同一个当前二进制，配置和核心输出哈希一致。

分支为 `feature/simple-octree-pipe`，基于 `no-vtk` 的 `c8a0742`。
实现入口是独立目标 `simple_channel`。它使用原项目的 `HADeviceGrid<Tile>`
创建、加密和回写八叉树，以 CPU double 求解有限体积 SIMPLE；不使用 flow map。

## 实验与判据

域为 `Lx=1, Ly=Lz=0.125`，`rho=1, nu=0.01`。平行板通道的 z 方向周期，
四壁方管的 y/z 侧壁均无滑移。周期流使用轴向加速度 1，压力驱动流使用两端
压力 1 和 0、零体力。速度和压力欠松弛系数分别为 0.7 和 0.3。

平行板解析解为 `u(y)=g*y*(H-y)/(2*nu)`；方管使用独立的矩形管 Fourier
级数。速度误差是按控制体积加权的向量相对 L2。流量来自完整横截面上的
**实际守恒面通量**，不是用单元平均速度替代。剪切误差是壁面积加权的均值，
并不代表局部壁面剪切误差的上界。

解析解匹配与跨代码匹配分别检查。分辨率足够的参考案例要求速度 L2 和流量
相对误差小于 0.5%，平均壁面剪切误差小于 1%；真实未松弛动量、连续性、
非正交修正一致性和状态变化均小于 1e-8。粗网格使用事先固定的较宽误差
门槛，并单独展示；细化后误差还必须下降。全部阈值见 `scripts/validate_simple.py`。

## 最终结果

| 案例 | 单元数 | 速度 L2 误差 (%) | 流量误差 (%) | 平均剪切误差 (%) | 迭代数 |
|---|---:|---:|---:|---:|---:|
| `uniform8` | 4,096 | 1.09e-06 | 0.7812 | 8.87e-07 | 318 |
| `perturb8` | 4,096 | 1.09e-06 | 0.7812 | 8.87e-07 | 318 |
| `pressure8` | 4,096 | 1.11e-06 | 0.7812 | 9.09e-07 | 316 |
| `duct8` | 4,096 | 0.7229 | 1.3730 | 7.88e-07 | 163 |
| `uniform16` | 32,768 | 4.23e-08 | 0.1953 | 1.24e-08 | 34 |
| `adaptive8` | 18,432 | 0.2903 | 0.4514 | 0.0307 | 46 |
| `adaptive16` | 147,456 | 0.0741 | 0.1068 | 0.0148 | 115 |
| `duct16` | 32,768 | 0.2420 | 0.2474 | 4.05e-07 | 45 |
| `convective8` | 4,096 | 1.08e-06 | 0.7812 | 8.79e-07 | 331 |
| `accelerated8` | 4,096 | 8.22e-11 | 0.7812 | 3.50e-11 | 7 |
| `adaptive_convective8` | 18,432 | 0.2926 | 0.4354 | 0.0317 | 806 |

所有案例的归一化真实动量残差均低于 `1e-8`，连续性残差最大为 `5.107e-12`。压力驱动案例的物理压力绝对 L2 误差为 `1.709e-11`。

细八叉树的速度误差为 **0.0741%**、流量误差为 **0.1068%**；四壁细方管分别为 **0.2420%** 和 **0.2474%**。这两组满足事先设定的 0.5% 精度目标。粗方管的误差较大，已如实列入表中。

八叉树从基础 `ny=8` 加密到 `ny=16`，速度和流量误差分别降低约 **3.92 倍**和 **4.23 倍**。均匀通道的流量误差降低约 4 倍。平行板的极小单元中心误差来自二次解析解及相容壁面离散这一特殊情形；流量仍有中点积分误差，不能据此推断一般流动也达到相同精度。

`uniform8` 与 `accelerated8` 使用同一个未加速 Aphros 最终解作为参照，分别在 318 与 7 步收敛。`uniform16/adaptive8/adaptive16/duct16/accelerated8` 启用 Anderson；其他案例关闭。`adaptive16` 使用 LDLT，其余使用 CG。最终验收全部基于原始 SIMPLE 步的真实残差，而不是加速候选的变化量。

![网格细化误差](figures/accuracy_by_resolution.png)

![细方管截面](figures/duct16_section.png)

另见[八叉树轴向误差](figures/adaptive_axial_error.png)和[八叉树收敛曲线](figures/adaptive_convergence.png)。


## Aphros 对齐的内容

独立基线为 Aphros `b60ce3da52c19935fa24c778f62f02141eaf7f80`。原仓库
`C:/Code/aphros` 未修改，实验使用独立 worktree。补丁只添加中间状态 dump
和可控初始扰动模块，未修改其 SIMPLE 数值算法。

在同一个 `64x8x8` 网格上，从零初值和非零散度扰动分别比较第 1、2 步及
最终结果。检查完整坐标覆盖、动量对角、delta RHS、预测速度、压力修正、
校正速度及预测/校正面通量。压力去掉规范常数，周期重复面经过一致性检查。
中间步门槛是 `atol=1e-11, rtol=1e-8`；最终门槛为 `1e-8, 1e-6`。

调试中实际修正了两处重要差异：壁面黏性导数使用与基线一致的二次单边
闭合；单元压力梯度在壁面从内部外推，而物理墙面的面通量仍为零。
另对 Eigen BiCGSTAB 显式归一化 RHS，并检查原始矩阵残差，避免很小的
积分方程 RHS 导致误判线性收敛。

自适应网格与 Aphros 均匀网格未知量位置不同，因此不声称二者的矩阵逐项
相同；自适应精度由解析误差、守恒和细化趋势验证。Aphros 本版本 SIMPLE
对压力口的处理有限制，压力驱动案例采用解析验证，不伪装成该基线的压力口对比。

## 复现与证据

```powershell
xmake f -m release -y
xmake build -j 4 simple_channel
python scripts/validate_simple.py --full --baseline-root D:/Dropbox/Agent-simulation/simple-baseline --output output/simple_validation_full
python scripts/validate_simple.py --full --analyze-only --output output/simple_validation_full
python scripts/plot_simple_validation.py
```

完整机器记录见 [suite_results.json](results/suite_results.json)，构建与源文件哈希见 [build_manifest.json](results/build_manifest.json)，独立基线误差见 [aphros_summary.json](results/aphros_summary.json)。大型原始 cell/face/matrix dump 保留在本机 `output/simple_validation_full/`，可用上面的命令重新生成。

基线构建方法见 [Aphros README](aphros/README.md)，算法和所有参数见
[求解器 README](../simple/README.md)。验证脚本记录真实进程退出码、配置与
可执行文件 SHA256、输出文件哈希以及各项固定阈值。图表根据实际 CSV 生成，
不会用缺失数据补齐曲线。

## 适用范围与下一步

这是轴对齐直通道和方管层流的精度参考版本。八叉树保留实际粗细层级及共享
子面通量，矩阵求解仍在 CPU；原生 float tile 快照是补充输出，精度比较
以 double CSV 为准。Anderson 只用于 Stokes 的外迭代加速，最终必须通过
完整原始 SIMPLE 步的真实残差检查。LDLT 只是同一压力矩阵的可选线性后端。

当前尚未接入实际 CFX 工程或验证曲面 cut-cell、弯管、入口发展段、回流与
湍流模型。充分发展直管的连续对流导数为零；开启对流的测试检验当前离散
实现的该类流动，不能替代一般惯性流动的验证。实际工程对齐还需要同一几何、
物性、边界条件、流动模型和输出量。
