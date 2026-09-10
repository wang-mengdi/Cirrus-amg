# 128 原生定常结果到独立 Aphros 验收

2026-09-09 更新：[128 原生完整状态](NATIVE128_STEADY_COMPLETION.md)
及速度初猜准备已通过。[64→128 空间检查](SPATIAL_CONVERGENCE_64_128.md)
已完成且未通过，原始 Aphros 128 参考已于 00:21:21 UTC 启动。
参考尚未完成；下面保留队列替换及执行流程的记录。

2026-09-08 23:28 UTC，将尚未启动求解的 128 Aphros 队列替换为
依赖完整原生 128 定常结果的流程。原队列准备使用从 64 网格插值的
速度初猜；新流程先验证最终 128 原生状态，再据其速度生成参考初猜。
这样做旨在减少初始调整，尚无本次 128 参考的耗时收益数据。

旧的 64→128 初猜仍保留在
`D:/CirrusExperiments/cirrus-amg/configs/initial_velocity128_from64_v1`，
未覆盖或删除。原始 Aphros 方程、几何、线性 AMG 后端、dt=0.005、
黏性内迭代次数 8 和请求的 128 个物理步不变。

## 已执行的队列替换

旧队列 PID 61464、创建时间 `1788907391.7961876`。替换前先核对
其执行文件和完整命令，确认没有子进程、没有 dispatch 记录、没有
参考输出目录；临时暂停该 Python 队列后再次检查，才结束队列。
实际退出码为 15。没有向任何流动求解器发出暂停、恢复或终止信号。
记录为
`D:/CirrusExperiments/cirrus-amg/checks/aphros128_queue_replacement_v1/intentional_stop.json`。

新的控制器为
`D:/CirrusExperiments/cirrus-amg/configs/queue_aphros128_native_steady_v1.py`。
其实际进程 PID 46076、创建时间 `1788910113.6022153`，已验证前置
回归的来源并进入 `waiting_for_native_steady_completion`。
启动时这只证明队列已经启动，后续步骤当时尚未执行；最新结果见上文。

## 后续真实执行顺序

1. 等待 `steady_outer128_v1` 的实际进程退出和正常完成记录；从
   `steady_summary.json` 选择真正的最终 `iterate_####`。
2. 用现有初值工具的 `--native-steady-iteration` 模式再次验证完整
   原生定常状态和守恒，再生成速度初猜。输出目录为
   `configs/initial_velocity128_native_steady_v1`。只初始化速度，
   压力和面通量不作为 Aphros 结果导入。
3. 在固定物理探针上执行原有 64→128 检查，两边均显式标注为
   定常外迭代。输出为 `checks/steady_refinement64_128_native_v1`。
   原门槛不变；数值验收失败会保存为失败，仍继续独立 Aphros 对照。
   若分析程序自身未能产生有效报告，则记录执行错误并停止后续派发。
4. 等待系统可用内存连续 30 秒不少于 11 GiB、D 盘可用空间不少于
   30 GiB，再启动原始 Aphros。新参考目录为
   `runs/aphros128_native_steady_seed_v1`。
5. 参考正常完成后，使用已经在 64 网格验证的
   `compare_twisted_steady_aphros.py` 直接比较两个定常结果。
   输出为 `checks/aphros128_native_steady_queue_v1/pair.json`。

本节相对路径均以 `D:/CirrusExperiments/cirrus-amg` 为根。状态、
每个实际子进程、命令、退出码、日志和检查结果保存在
`checks/aphros128_native_steady_queue_v1`。
辅助程序可用内存持续低于 2.5 GiB 时，仅停止本流程新启动的 CPU
辅助子进程；Aphros 运行器另行管理自己新启动的 CPU 求解进程。
队列不停止任何现有原生 GPU 进程，也不依赖暂停进程释放内存。

初猜准备、求解正常退出、Aphros 对齐和空间收敛是分别记录的结果。
即使流程最后全部通过，总体目标也需重新核查其证据和可视化交付，
控制器不会自行把目标标记完成。已有 32→64 近壁空间检查失败仍保留。
