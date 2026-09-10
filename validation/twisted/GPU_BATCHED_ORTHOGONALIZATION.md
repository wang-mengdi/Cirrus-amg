# GPU FGMRES 批量正交化验证

当前压力投影和隐式黏性的大规模线性系统继续使用原生八叉树模板、
双精度 FGMRES 和原生单精度 AMG 预条件器。新增可选
`gpu_orthogonalization: "cgs2"`，默认仍为 `"mgs2"`。
这次修改不改变流动离散、曲面壁面闭合、粗细交界、AMG 层级、
FGMRES 重启维数 20 或实际原方程残差门槛 `1e-13`。

原实现每次 MGS 内积分别把一个标量同步到 CPU。CGS2 每遍先同时
计算当前向量与全部已有基向量的内积，再一次减去它们的线性组合；
完整重复两遍。每遍只传回一次系数数组。这里仍有范数、压力归一化
等同步，不能把正交化传输次数说成整个求解器的全部同步次数。

CGS 配合再次正交化是已有的 GMRES 选择，参见
[PETSc 的经典 Gram–Schmidt 文档](https://petsc.org/release/manualpages/KSP/KSPGMRESClassicalGramSchmidtOrthogonalization/)
及其 [CGS refinement 选项](https://petsc.org/release/manualpages/KSP/KSPGMRESSetCGSRefinementType/)。
本实现是自写 CUDA kernel，没有移植 PETSc 源码。二次正交化不能
单凭算法名称保证稳定，因此保留实际原 RHS/新鲜 Ax 的停止检查，
并验证失败后的求解器复用。

## 已完成的线性系统检查

执行文件来自 `output/twisted/cgs2_build_v1`，原生求解器 SHA256 为
`a03842fe580726576f8b9a1a5a73693113684d7b50819e2e93bfa65f2b3bcc9d`，
审计程序 SHA256 为
`0c7be8f2188ce76a907b9ce5d7f22652fc8a68da619c2cd40464089af058dcd0`。

`cgs2_benchmark16_v1` 和 `cgs2_benchmark64_v1` 各按
MGS2、CGS2、CGS2、MGS2 顺序运行四个新进程，使用相同网格、
系数、RHS 和已知解。64 网格包含粗细交界及变化面系数。
所有算子、已知解、严格残差和强制失败后复用检查通过。
两组恢复解的最大相对差分别为 `3.71e-16`、`2.41e-16`。

| 64 网格线性系统 | 两种方法迭代数 | MGS2 正交化传输 | CGS2 正交化传输 |
|---|---:|---:|---:|
| 压力 | 100 | 2100 | 200 |
| 隐式黏性 | 65 | 1290 | 130 |

两次计时的中位数之比 MGS2/CGS2，压力约 2.04、黏性约 1.66。
机器当时同时运行其他算例；这些是观测值，不是独占 GPU 的性能
基准，也不能直接作为 128 网格整套流动求解器的加速比。

## 完整物理步检查

16 网格从零开始，`dt=0.001`，8 步。相同新可执行文件仅切换
正交化选项；两者都是 66 次外层迭代、482 次成功 GPU 线性求解。
逐步的速度、压力、近壁速度、壁面剪切和共享面通量相对 L2 差异
最大 `5.69e-16`。双方逐步实际通量守恒、完整内层固定点和实际
GPU 线性残差门槛均通过。CGS2 最大接受线性残差 `9.975e-14`。

CGS2 与既有 CPU 实现逐步比较的上述五类场最大相对 L2 差异
`2.48e-9`。默认 MGS2 的 CPU 对照也通过。

新 CGS2 最终物理步与刚完成的扩展精度 Aphros 原始 CG 路径
`navier_stokes_proj_n16_extended_cg_v3` 比较通过。该参考保留原始
投影、输运、壁面和 CG 求解算法，使用加载的原始 double 几何及
扩展精度标量；没有接入分解缓存后端。速度、压力、近壁速度、
壁面剪切相对 L2 差异分别约 `3.21e-10`、`3.90e-10`、
`8.99e-10`、`7.03e-10`，共享面通量和实际质量守恒也通过。
既有 MGS2 结果对这一参考的独立检查同样通过。

完整 16 网格轨迹观测耗时 MGS2 为 167.95 秒，CGS2 为 63.69 秒；
两次顺序执行期间存在其他运行负载，只作为运行记录保留。

64 自适应网格从零开始，`dt=0.005`，完整一步已正常结束。
185 次成功 GPU 求解的最大实际残差为 `8.869e-14`，完整内层
固定点残差 `5.0275e-11`，实际质量散度相对 Linf `3.3059e-11`。
与已完成的扩展精度 Aphros `navier_stokes_proj_n64_extended_cache_v7`
比较，全部 13 项检查通过，包括粗细交界、每个共享面通量、壁面
剪切和双方实际质量守恒。相对 L2 差异为：

| 量 | 相对 L2 差异 |
|---|---:|
| 速度 | 0.0134405% |
| 压力 | 0.0076772% |
| 切割单元速度 | 0.0277774% |
| 壁面剪切 | 0.0116400% |
| 截面流量 | 0.0010573% |
| 共享面法向速度 | 0.0114548% |

参考的实际存储面通量质量散度相对 Linf 为 `5.2790e-10`，
使用精确十六进制 dump 和 Decimal 高精度求和独立检查。
上述差异与既有 MGS2 结果相当；这是 `t=0.005` 的非定常比较。

另外已用相同新构建启动 `ours_proj128_cgs2_steps2_v1`，其物理
输入与既有 128 网格两步算例相同。这里只保存启动证据；在完整
结束、原残差及全部场比较通过之前，不视为 128 网格验证成功。

## 复现和限制

```text
python scripts/benchmark_twisted_orthogonalization.py --exe output/twisted/cgs2_build_v1/native_compact_gpu_audit.exe --config output/twisted/config_anderson_failure_audit64_v1.json --output <新目录>
python scripts/check_twisted_time_pair.py --candidate output/twisted/ours_proj16_cgs2_steps8_v1 --reference output/twisted/ours_proj16_mgs2_regression_v1 --orthogonalization-pair --output <新检查.json>
```

`--orthogonalization-pair` 要求两条完整轨迹、相同可执行文件、除
输出和正交化方法外完全一致的配置；逐步比较五类场，并对双方
独立检查实际面通量守恒、完整固定点和实际线性残差。

这些检查证明求解后端保持同一离散解，不证明网格或时间步已经
收敛。32→64 定常网格的近壁/剪切检查仍未通过，64 时间步减半、
128 定常和 64 独立 Aphros 定常运行仍在进行。
本次没有改变原来的物理验收门槛，整体目标未完成。
