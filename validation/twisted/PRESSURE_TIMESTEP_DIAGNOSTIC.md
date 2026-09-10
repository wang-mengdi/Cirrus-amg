# 已完成定常场的压力时间步诊断

四个 16 网格运行均完成到 t=0.64 s，定常动量和时间加速度均小于
1e-8，实际保存的共享面通量通过原有质量守恒门槛。压力时间步敏感性
仍未通过 0.25% 门槛；本诊断没有改变求解器、基线或验收阈值。

## 从实际场量验证投影中的时间步项

记 I 为单元到面的速度插值，Gc 为压力到单元压力梯度的重构，Gf
为压力投影的双单元面梯度，Sf 为开口面积。在本次均匀切割网格、
常密度、常体力且开口面 I(1)=1 的条件下，最终场满足：

```
q_f = Sf (I u)_f + dt/rho * Sf [(I Gc p)_f - (Gf p)_f]
```

它来自 `simple/ProjectionSolver.cpp` 的 `velocityFlux`、`project`
以及最终单元速度修正。独立 Aphros 源码
`aphros/src/solver/proj.ipp` 的 `GetFlux`（343 行附近）、
`GetAcceleration`（408 行附近）和最终投影（546–574 行）也保留
相应的 dt 系数：先插值预测单元速度，再用面压力梯度修正通量，
最后用单元压力梯度修正速度。这里没有显式求逆或矩阵分解要求。

诊断使用已归档的原生算子 CSV 和每个已完成运行的最终 u、p、q。
网格与面连接逐项相同，捕获的 Gc 对实际最终压力源 dump 的最大
加速度差为 1.30e-12。源 dump 位于最后一次内迭代的输入侧，因此
这项检查允许已收敛迭代的差异，不声称逐位相同。

重构 q 与实际保存 q 的相对 L2 差最大为约 1.24e-16。上述 dt 项
因此确实存在于这些定常结果中。BCG 输运也依赖 dt；此恒等式并不
单独证明全部压力变化都由这一项造成。

## 已测得的变化及位置

| dt 减半（ms） | 压力相对 L2 差 | 单元压力梯度相对 L2 差 | 压力差能量位于切割单元的比例 |
|---|---:|---:|---:|
| 5 → 2.5 | 0.47822% | 1.20435% | 69.1849% |
| 2.5 → 1.25 | 0.52565% | 1.31539% | 72.1710% |
| 1.25 → 0.625 | 0.59502% | 1.42049% | 73.8592% |

压力差能量定义为 `sum(V * delta_p^2)`，压力均先减去体积加权均值。
切割单元只占流体体积的 31.0595%，却贡献上述大部分差异。
不是极小体积单元独自支配这个指标：例如最后一对中，体积分数
小于 0.001 的单元贡献压力差能量的约 0.0570%，而体积分数至少
0.1 的切割单元贡献约 69.6434%。详细分区保留在 JSON 中。

相邻压力差的绝对 RMS 比值为 1.0991 和 1.1318，尚未观察到随
连续减半而缩小的趋势。这些是不同离散解之间的差异，不是对精确
解误差的估计，也不支持据此做 Richardson 外推。

第一对使用同一个可执行文件和相同 Anderson 设置。后两对来自
此前不同构建，最后一个启用 Anderson；各自真实二进制哈希和设置
均记录在报告中。因此后两对不能被描述为严格只改变 dt 的受控
实验。四个场量均通过相同的算子/通量重构检查；第一对已经足以
证实该粗网格存在超过门槛的时间步敏感性。

直接插值单元速度得到的通量，其散度 RMS 为 0.00530–0.00918 /s；
经过压力修正后，实际保守通量的散度 RMS 为约 1e-15–1e-14 /s。
前者不是求解器使用的保守面通量，不能把前者当作质量守恒失败。

## 复现与下一步

```
python scripts/analyze_twisted_pressure_timestep.py --runs output/twisted/ours_proj16_steady_v1 output/twisted/ours_proj16_halfdt_steady_v1 output/twisted/ours_proj16_quarterdt_steady_v3 output/twisted/ours_proj16_eighthdt_steady_aa5_v5 --operators output/twisted/operators_ns16_exp_v3 --output <新诊断目录>
python scripts/plot_twisted_pressure_timestep.py --report <新诊断目录>/diagnostic.json --output <新图目录>
```

本次报告：`output/twisted/pressure_timestep_diagnostic16_v2/diagnostic.json`。
逐单元差异：同目录 `pressure_differences.csv`。
图：`output/twisted/pressure_timestep_plot16_v2/pressure_timestep.png`，
已实际查看，SVG 与输入/输出哈希也保留。

这项证据把后续诊断重点放在有限网格压力/速度耦合及近壁重构上，
不能以更小线性残差代替压力的时间步独立性，也不盲目继续增加
16 网格的时间步减半次数。正在运行的 64 网格时间步减半结果仍需
完成后独立检查；若也失败，再在同一构建和完整中间 dump 下区分
压力耦合与 BCG 的影响。此时没有修改 Aphros 离散方程。

128 原生定常运行、64 扩展精度 Aphros 完整物理步以及独立定常对齐
仍在推进。32→64 空间验收失败的结论不变，总体目标未完成。
