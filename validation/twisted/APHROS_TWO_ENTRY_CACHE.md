# Aphros 两项 AMG 缓存的小算例核验

本实验只检查将 `APHROS_TWISTED_FACTOR_CACHE` 从 1 改为 2 的影响。
名称沿用早期实现，但当前 `APHROS_TWISTED_AMG=1`，缓存的是
压缩算子和 AMG 层次，并非 LDLT 分解。两个实验都使用同一份
冻结的原方程 Aphros 可执行文件、几何、初始速度和物理参数。

这是 16 级三维扭曲细管，采用 Proj、BCG、隐式黏性、`Ndiff=1`，
128 步、`dt=0.005`、终止时间 0.64。原 128 对照使用 `Ndiff=8`；
本实验不能说明在 128 级增加缓存的内存需求或性能收益。

2026-09-09 11:58:39 UTC 完成核验：

| 项目 | 单项缓存 | 双项缓存 |
| --- | --- | --- |
| 线性调用次数 | 15638 | 15638 |
| 命中次数 | 6400 | 6400 |
| 重建次数 | 9238 | 9238 |
| 完成物理步 | 128 | 128 |
| 观测耗时（秒） | 833.538896 | 974.554083 |
| 峰值工作集（B） | 46399488 | 58241024 |

速度、压力、壁面剪切、共享面通量、两份扩展精度单元/面输出
及时间记录共六份文件逐字节一致；完整外迭代误差序列也一致。
双方按存储的原始扩展精度通量重新计算质量守恒，均通过原门槛。
每次线性调用的系统名称、行数和非零数序列相同。其他任务同时
运行，因此表中时间仅为实际观测，不作独占性能结论。命中次数
未改善，没有据此修改正在运行的原 Aphros 128 对照。

实际编译的扩展精度头文件对矩阵值使用精确数值相等，并检查零
的符号；稀疏行列结构逐字节比较。这避开 `long double` 填充字节，
不放宽系数相等的要求。根目录中的 double 模板与实际编译的
`src/linear/twisted_direct.h` 均保留哈希和源快照。

第一次核验因配置文本哈希不同而拒绝：冻结运行器把 `tmax` 的
`0.64` 写成 `0.64000000000000001`。原始文件没有改动。独立比较
脚本只允许这一个确定的文本替换，要求其他配置全部相同，并
核对实际编译的 `Vars::Double` 为 `Map<double>`、两种文字解析为
相同 binary64 值 `0x1.47ae147ae147bp-1`、完整实际时间记录逐字节
一致。第二次核验在数值比较前因错误地要求扩展精度头文件也
使用 double 模板的 `memcmp` 写法而拒绝；修正源文件识别后通过。
这两次检查失败记录保留，未重跑流动，也未修改冻结的正式比较器。

复现位置均相对 `D:/CirrusExperiments/cirrus-amg`：

- 对照：`runs/aphros16_seeded_memory_v1`。
- 双缓存：`runs/aphros16_factor_cache2_v1`。
- 最终检查：`checks/aphros16_factor_cache2_v3/result.json` 和 `pair.json`。
- 失败记录：`checks/aphros16_factor_cache2_v1/failure.json`、
  `checks/aphros16_factor_cache2_v2/failure.json`。
- 检查脚本：`configs/check_aphros16_factor_cache2_v3.py`、
  `configs/compare_aphros_cache_duration_literal_v1.py`。

相关源快照、运行记录和报告并入本轮 `compact_quadratic_storage`
归档，但明确属于独立缓存核验。没有完成新的 128 级 Aphros 对齐、
256 级流动或空间收敛验收，整体 goal 仍为 active。
