# 64 网格扩展精度参考：完整首步通过

`navier_stokes_proj_n64_extended_cache_v7` 于
2026-09-08 05:10:50 UTC 正常完成一个 dt=0.005 s 的物理步。
本次实际独立比较全部通过，包括参考的实际通量守恒；它仍然是
t=0.005 s 的瞬态结果，不是定常或网格收敛结果。

## 参考及对齐范围

参考使用原 Aphros Proj 离散、原双精度计算几何的精确快照、八次
隐式扩散迭代和 64 位尾数的扩展精度标量。其几何、原方程主体、
完整构建及执行来源已由 `validate_aphros_extended_reference.py`
核对。可执行文件 SHA256：

```
962fb063332b407709384abc5023ac67691b40fc15a45cdf76ca18ca90e9b936
```

原生一侧为已完成 128 步运行
`ours_proj64_steady_original_rhs_v1/step_0001`。它在管壁保留细网格，
内部粗化后有 136632 个流体单元、1536 个粗细接口；参考均匀网格有
140216 个流体单元。近壁单元和壁面位置严格一致。内部 512 个粗
单元中心使用经过敏感性检查的三次传递，136120 个细单元直接取值。
全部原生面通量均比较，1344 个完整粗面聚合参考的四个细面。

| 比较量 | 相对 L2 差异 |
|---|---:|
| 速度 | 0.0134405% |
| 压力（统一规范） | 0.00767719% |
| 切割单元速度 | 0.0277774% |
| 壁面剪切向量 | 0.0116400% |
| 截面流量 | 0.00105730% |
| 全部共享面的法向速度 | 0.0114548% |

这些是两个离散结果之间的差异，不是对连续精确解的误差估计。
速度、压力、近壁、剪切、流量和全场通量均通过既有比较门槛。

## 实际守恒及精度边界

读取参考实际保存的十六进制面通量，以 100 位十进制精度累加，
而非重新计算压力表达式或先转换为 double，得到：

- 参考相对散度 Linf：5.27902668603e-10，小于原门槛 1e-7。
- 参考截面流量相对极差：2.04429986139e-17。
- 参考全域单元通量绝对失衡之和/通过流量：4.05115163334e-16。
- 原生相对散度 Linf：3.30586108765e-11。

周期接缝两份存储并非逐位相同，相对差为 1.6450e-15，通过原有
1e-12 接缝检查；使用另一侧接缝的独立累加也得到相同最坏相对
散度。此前双精度参考 64 网格实际守恒未通过的记录仍保留，
本次没有提高该门槛，也没有把双精度导入诊断当作扩展精度真值。

## 复现及定常推进

```
python scripts/check_aphros_exact_mass.py --aphros D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_proj_n64_extended_cache_v7 --output <新质量报告.json>
python scripts/compare_twisted.py --ours output/twisted/ours_proj64_steady_original_rhs_v1/step_0001 --aphros D:/Dropbox/Agent-simulation/twisted-baseline/navier_stokes_proj_n64_extended_cache_v7 --transient --adaptive --aphros-diffusion-iterations 8 --extended-reference --output <新比较报告.json>
```

实际报告为 `output/twisted/aphros_n64_extended_cache_exact_mass_v1.json`
及 `output/twisted/aphros_n64_extended_cache_native_pair_v1.json`。
完整原始字段保留在参考运行目录，归档保留实际十六进制场量、
报告、来源哈希及新运行的启动配置。

已于 05:13:00 UTC 启动
`navier_stokes_proj_n64_steady_extended_cache_v7`。仍为相同程序、
几何快照、求解环境、容差和 dt=0.005，只把终止时间改为 0.64 s
（128 步）；启动时实际进程 PID 为 30292。它从零速度开始完整推进，
没有把单步结果或其他求解器的定常结果替代参考轨迹。

原生 128 定常与原生 64 时间步减半运行继续保留。
独立定常对齐、128 网格结果、空间收敛和时间步检验仍未完成。
