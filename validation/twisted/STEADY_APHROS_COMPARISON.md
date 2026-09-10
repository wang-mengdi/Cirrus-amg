# 定常外迭代结果直接对比原始 Aphros

新增 `scripts/compare_twisted_steady_aphros.py`，接收真正完成的
`iterate_####` 定常输出，直接与完整原始 Aphros 轨迹的最终定常场
比较。它不为定常迭代添加虚构的物理步编号、物理时间或完成记录。
普通物理时间推进的现有比较入口保持不变。

原生输入先经过完整定常状态验证：实际编译源、原生八叉树拓扑、
GPU 原始残差、每次内迭代、保留场的共享通量守恒、最终原始方程
固定点和未加速原始 dump 均需通过，且输入必须是最终输出目录。
Aphros 输入另行验证原始方程代码、真实几何、完成状态、全部时间
记录、最终内迭代及最终时间变化；精确存储的面通量用既有高精度
累加程序核对。两边仍必须使用相同的动量时间步，因其进入有限网格
Rhie–Chow 离散。近壁、场量和守恒门槛没有变化。

## 已完成的直接 64 网格验证

原生输入：
`D:/CirrusExperiments/cirrus-amg/runs/steady_outer64_v1/iterate_0016`。
参考输入：
`D:/CirrusExperiments/cirrus-amg/runs/aphros64_initial_native_v1`。
结果：
`D:/CirrusExperiments/cirrus-amg/checks/steady_outer64_direct_aphros_v1.json`。

全部检查通过。两者独立满足定常条件，比较不要求伪时间与参考的
最终物理时间相等。压力去体积均值，细单元、切割单元和壁面直接
对应，粗中心保留现有三次插值及其敏感性记录。

| 量 | 相对 L2 差 |
| --- | ---: |
| 速度 | 0.02364144% |
| 压力 | 0.12789821% |
| 切割单元速度 | 0.01941441% |
| 壁面剪切 | 0.01846165% |
| 截面流量 | 0.00757246% |
| 共享面法向速度 | 0.02338979% |

这些是两个离散解之间的差异，不是真解误差。直接定常比较现在也有
对应的 ParaView 差值图，见下一节。[已有报告](APHROS64_FULL_COMPLETION.md)
中使用普通物理推进原生场的可视化仍保留，两个来源各自记录。

## 直接定常比较的 ParaView 文件

实际输出目录为
`D:/CirrusExperiments/cirrus-amg/runs/steady_outer64_direct_aphros_viz_v1`。
在 ParaView 的 File → Load State 中打开 `render_v1/comparison.pvsm`，
可同时看到速度差和壁面剪切差。也可以分别打开 `solution.vtu` 与
`walls.vtp`，选择 `VelocityDifferenceMagnitude` 和
`WallShearDifferenceMagnitude`。色标单位分别为 m/s 和 Pa，表示
绝对差值幅度，不是逐点百分比。

`export_twisted_aphros_difference.py` 新增 `--native-visualization`，
接受另存于 D 盘的、带有原始求解目录记录的几何导出；它仍核对
完整的已通过对比来源、原生字段和原有几何检查结果。图标题的
网格分辨率改从经过哈希核对的实际 `case.json` 读取，不再固定为 64。
旧的几何与字段同目录模式也用完整物理推进参考重新导出并通过。

本次真实定常外迭代输出通过 ParaView 5.13 读取：136632 个体单元、
21816 个壁面，14 个数组逐位相同，重算的速度、压力、切割单元速度
和剪切相对 L2 差与上述已通过报告一致。另一次 Load State 调用
核对了两个视图、实际单元数和参考时间标题。PNG 已人工式目视检查。
单独目录但缺少原始求解来源的几何输入被拒绝。

检查记录为 `checks/steady_aphros_visualization_v2/result.json`；
提交证据在 `results/steady_aphros_visualization`。第一版检查曾遇到
旧几何清单缺少 `source_step`，其失败日志保留；当前实现允许旧式
同目录输入，并要求单独几何目录明确记录来源。这项验证没有重新
运行流动求解，也不证明 128 网格或空间收敛已经通过。

## 比较器回归与拒绝范围

`check_twisted_steady_aphros_scope.py` 从实际原生定常场、普通推进
定常场和 Aphros 比较报告，检查五种加权误差满足三角不等式。
它直接读取原生和普通场的差异，没有把两份结果要求为逐位相同。
速度、压力、切割单元速度、剪切和截面流量的五项检查均通过。

实际 CLI 检查也确认以下三种输入被拒绝，且不生成通过报告：
普通物理时间步冒充定常外迭代、非最终定常外迭代、已捕获的部分
Aphros 轨迹冒充完整参考。记录为
`D:/CirrusExperiments/cirrus-amg/checks/steady_aphros_scope_v1/result.json`。

该入口可在完整 128 网格原生定常结果与独立 Aphros 参考都完成后
用于直接对比，不必等另一条普通物理推进对照也完成。这只补齐了
输入类型的验证路径，不能替代尚未完成的 128 对照或 64→128
空间收敛检查。现有 32→64 近壁空间验收失败仍有效。

当前的 [128 验收流程](APHROS128_STEADY_PIPELINE.md)已排队等待原生
定常结果，依次执行初值准备、空间检查、原始 Aphros 求解和本直接
比较；尚未把任何未完成的 128 结果列为通过。
