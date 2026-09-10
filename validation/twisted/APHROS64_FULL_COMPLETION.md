# 64 网格原始 Aphros 完整定常参考

2026-09-08 22:31:47 UTC，`aphros64_initial_native_v1` 正常完成全部
128 个物理时间步，dt=0.005，最终 t=0.64。实际进程退出码为 0；
执行文件、几何、配置和初始速度均保持不变，初始速度读回逐字节一致。
这是此前第 46 步定常快照的完整轨迹后续结果。

参考目录：`D:/CirrusExperiments/cirrus-amg/runs/aphros64_initial_native_v1`。
完整对比报告：`D:/CirrusExperiments/cirrus-amg/checks/aphros64_seeded_full128_pair_v1.json`。
该报告明确记录 `complete_reference_run_checked: true`，独立检查了
原始 Aphros 方程代码、构建来源、全部时间步和实际几何。

## 最终场对比

Cirrus 输入仍为
`C:/Code/Cirrus-amg/output/twisted/ours_proj64_steady_original_rhs_v1/step_0128`。
保留同一三维扭曲管道、壁面加密八叉树、曲面切割单元和原定验收门槛。
压力按体积均值对齐；136120 个细单元直接对应，512 个粗单元采用
原先验证过的张量三次插值。全部切割单元和壁面位置直接对应。

| 量 | 相对于 Aphros 的加权相对 L2 差 |
| --- | ---: |
| 速度 | 0.02364144% |
| 压力 | 0.12789821% |
| 切割单元速度 | 0.01941441% |
| 壁面剪切向量 | 0.01846165% |
| 截面流量 | 0.00757239% |

完整比较器的所有门槛通过，包括实际共享面通量、定常动量和完整
投影固定点检查。Aphros 最终时间变化诊断为
`7.727328323375253e-17`，低于原有 `1e-8` 定常门槛。
实际存储的扩展精度面通量以 100 位十进制复算，最大相对散度为
`1.1596704722873437e-10`，截面流量相对跨度为
`6.58898101604068e-18`，守恒检查通过。

上述数值是求解器之间的差异，不是真解误差。粗中心线性与三次插值
的全场速度差仍为约 0.052694%，因此不能解释为所有原始未知量逐点
一致。切割单元速度和壁面剪切的对比不含该粗中心插值。

本参考用 Cirrus 的已收敛速度作为初始猜测，再由原始 Aphros 方程
实际推进至定常；没有加载 Cirrus 的压力或面通量作为参考结果。
从零开始的另一条原始 Aphros 64 轨迹在第 24 步完成独立瞬态对齐后
已按计划结束，见 [冷启动对照记录](APHROS_COLD_CONTROL_RETIREMENT.md)；
它未被记为定常或完整 128 步运行。

## 可视化与剩余工作

完整最终参考的 ParaView 文件位于
`D:/CirrusExperiments/cirrus-amg/runs/aphros64_full128_comparison_viz_v1`：

- `solution.vtu`：原生切割体几何、Cirrus/Aphros 速度、压力及其差值。
- `walls.vtp`：原生壁面多边形、Cirrus/Aphros 剪切向量及其差值。
- `render_v1/comparison.pvsm`：在 ParaView 用 **File → Load State**
  打开，恢复速度差和剪切差两个视图；同目录有 `comparison.png`。

标题明确写出 `Aphros t=0.64 s; complete reference`。导出程序必须
显式使用 `--completed-reference`，并验证完整轨迹证明覆盖实际配置
的时间步数；把部分快照冒充完整参考、或未显式区分完整参考的输入
都会被拒绝，两个拒绝路径已实际检查。

ParaView 5.13 实际读回 136632 个体单元、21816 个壁面多边形和
14 个字段，逐字节对应原输出；从差值重算的加权指标与最终报告一致。
另一次独立的 File → Load State 检查也通过，两个视图的数据和标题
均正确。预览已经检查。验证记录为
`D:/CirrusExperiments/cirrus-amg/checks/aphros64_full128_visualization_v1/result.json`。

[第 46 步报告](APHROS64_STEADY_CHECKPOINT.md)中的旧 ParaView 差值文件
仍对应 t=0.23 的已验证快照，未被覆盖。

128 网格原生定常结果、128 原始 Aphros 定常参考和 64→128 近壁
空间收敛验收仍待完成。既有 32→64 空间检查失败仍有效，整体目标
没有因这项完整参考对齐而标记完成。
