# 压力通量修正循环的只读诊断

本次没有更改求解器或运行中的配置。保留 twofold 压力迭代、原生
GPU AMG、1e-13 线性门槛及现有物理检查，仅分析已经写出的压力
修正日志。完整定常对齐及空间收敛目标仍未完成。

大型构建、运行与检查目录均为 `D:/CirrusExperiments/cirrus-amg`。
2026-09-08 16:07 UTC 左右，C 盘约剩 81 GB、D 盘约剩 365 GB；
本轮没有移动或删除运行中的文件。

## 观察到的循环

`checks/twofold128_roundoff_cycles_v2/result.json` 保存了
2026-09-08 16:13:52 UTC 的实际日志前缀。当前压力投影只在本次
散度与紧邻上一次完全相等时考虑舍入退出。因此，两个或多个
数值间的循环可能继续执行许多次，甚至达到 64 次上限。

| 已结束且日志配对完整的部分 | 数量或耗时 |
|---|---:|
| 有修正日志的压力调用 | 153 |
| 实际修正次数 | 947 |
| 实际修正耗时之和 | 1514.3906691 s |
| 核对通过的现有舍入退出 | 110 |
| 筛查到的非恒定重复循环调用 | 18 |
| 最早候选点之后仍执行的修正 | 308 |
| 这些后续修正的实际耗时 | 481.0746479 s |

例如调用 76 在第 7 次修正已有一个完整重复周期，原程序最终
执行到第 64 次；其后 57 次修正在这次运行中耗时 92.4582904 s。
这些数字是原轨迹中的已执行工作量，不能作为修改算法后的
整体节省时间或速度提升预测。

## 筛查条件与限制

新增 `scripts/analyze_twisted_roundoff_cycles.py` 只标记候选点：

- 长度为 2、3 或 4 的非恒定散度序列完整重复两遍。
- 窗口内每次原始散度都在 [1e-10, 1e-8]，补偿求和散度不超过 1e-8。
- 每次速度修正不超过 `0.5 * epsilon * 修正前速度尺度`。
- 每次压力冲量修正不超过 `0.5 * epsilon * 驱动冲量尺度`，且驱动
  尺度明确大于修正前冲量范围及本次修正的包络。

诊断日志记录的是修正前尺度，现有求解器检查修正后尺度；这里
特意采用有余量的筛查条件，仍不将其称为可执行退出规则的证明。
非恒定标量循环也不表示完整通量向量逐位重复。要采用提前退出，
仍需对修改后的实际流动重新检查守恒、完整固定点、速度、压力、
近壁速度和剪切，并与原轨迹及独立 Aphros 对照。

快照仅保留以换行符结束的完整 CSV 记录，按调用号和修正次数配对。
只有后续调用已出现时，前一调用才纳入统计；最后一个可能仍在
计算的调用被排除。无修正就直接返回的压力调用没有日志，不计入
表中的 153 次。两份文件读取之间新增的末尾记录也不当成完整调用。

可执行文件、构建清单、实际编译的 `ProjectionSolver.cpp`、配置、
几何元数据及采样日志都有哈希和字节快照。没有读取缓存的目录
大小来判断活跃 CSV 是否为空，而是实际读取至 EOF 后截取完整行。

## 检查结果

`checks/roundoff_cycle_screening_v2/result.json` 的 39 项检查通过，
包括 2/3/4 周期、缺少第二周期、恒定序列、非循环下降、大速度
修正、大冲量修正、超界散度、记录错位、未结束调用、重复/损坏
记录，以及真实快照的逐项重放。它仅验证筛查与记录处理。

原有独立诊断 `check_twisted_projection_floor.py` 也通过：
`checks/twofold128_projection_floor_v1.json` 核对了 990 次更新、
110 组最差单元的实际面通量求和以及 110 次现有舍入退出。
该快照采样稍晚，因此总行数与上表不同，二者不混作同一前缀。

复现命令（从仓库根目录执行，输出路径须使用新目录）：

```powershell
python scripts/analyze_twisted_roundoff_cycles.py --run D:/CirrusExperiments/cirrus-amg/runs/twofold128_prefix_v1 --build D:/CirrusExperiments/cirrus-amg/builds/pressure_twofold_v1 --output D:/CirrusExperiments/cirrus-amg/checks/twofold128_roundoff_cycles_NEW
python scripts/check_twisted_roundoff_cycle_screening.py --snapshot D:/CirrusExperiments/cirrus-amg/checks/twofold128_roundoff_cycles_NEW --output D:/CirrusExperiments/cirrus-amg/checks/roundoff_cycle_screening_NEW
```

原采样字节和当时的分析器位于本目录下
`results/pressure_cycle_profile_checkpoint`；检查脚本也可针对归档中的
`profile_v2` 目录重放。用 `--analyzer` 指向该目录的
`inputs/analyzer.py`，可直接加载当时保存的分析器，避免以后代码
或工作区换行符变化影响重放。v2 检查实际采用了此方式。

## 长计算状态

16:15 UTC，原生 128 运行已完成物理步 13，正在计算步 14。
步 13 日志报告完整固定点残差 1.45247e-9、连续性相对 Linf
3.09490e-10、黏性残差 4.76346e-11；时间变化仍约 0.0409131，
所以尚非定常。该步受输出步长控制，没有完整场文件，不能称为
已从导出场独立复核的结果。

独立 Aphros 128 首步、Aphros 64 冷启动及 64 速度初始化参考都
继续运行。下一步应验证循环退出的实际影响；本轮没有通过停止
长计算或额外并发 GPU 实验来代替该验证。
