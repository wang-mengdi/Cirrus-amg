# Aphros 速度方程后端试验：32 和 64 完整步对照

此项用于缩短独立参考解的推进时间，未改变 Cirrus 原生八叉树
或 GPU 算子。当前新增选项默认关闭；原 Aphros 定常运行继续。

## 实测瓶颈

`profile_aphros_live_backend.py` 核对实际 PID、创建时间、执行
文件、工作目录和配置，从正在运行的
`runs/aphros64_initial_native_v1` 捕获完整线性调用的日志前缀。
只分析已经出现 `snapshot-complete` 的调用，不把当前未完成调用
或物理步算成完成。其证据位于
`checks/aphros64_seed_backend_profile_v1`。

前缀含 17141 次完整调用，最近 200 次的后端累计时间为
342715 ms。其中首次求解占 91.2353%，组装等准备占 6.8617%，
迭代修正占 0.2661%，原始方程行检查占 1.6369%。速度分量的
首次求解中位时间约 1.5 秒。由于其他任务并发，不能将该记录
当作独占性能基准，也不能把后端比例直接外推成整体加速比例。

## 仅切换速度线性后端

`validation/aphros/build_velocity_direct_reference.py` 从已验证的
lazy-centers 扩展精度源树建立独立副本，仅修改实际使用的
`src/linear/twisted_direct.h` 的后端选择和诊断标签。

启用 `APHROS_TWISTED_DIRECT_VELOCITY` 时，压力继续使用既有
AMG，速度使用已经实现的直接求解器；依据原对称性判断选择
LDLT 或 LU。所有原始组装系数、浮点精度、求解后迭代修正、
全部原始方程行检查及门槛保持不变。此次没有改变 Aphros 的
Proj、BCG、黏性迭代、壁面拟合、几何、初值或物性。

编译器对全部 48 个翻译单元生成依赖表，只重新编译实际包含
该头文件的 `src/linear/linear.cpp`，复用 47 个经过哈希核对的
对象文件。原源树保持不变。冻结执行文件为
`D:/CirrusExperiments/cirrus-amg/builds/aphros_velocity_direct_v1/twisted_extended.exe`，
SHA256：
`cf4ce9b6423d3bd5734da1b266134ac5291f8d089f8316663e38bbeee9218f5b`。

运行器 `run_aphros_velocity_backend_reference.py` 显式记录
`--linear-backend amg-pressure-direct-velocity`，并保留原参考
配置、完整几何和资源保护。缓存日志对每个实际压力/速度调用
标明 `backend=amg` 或 `backend=direct`，便于核对真实路径。

## 已完成及中断的验证

关闭新选项的 16 网格完整第一步已经通过：
`checks/aphros_velocity_backend_probes_v1/pair16.json` 验证原始
方程来源与精确通量守恒，六份速度、压力、通量、壁面剪切及
时间输出均逐字节相同，全部 260 次外迭代误差记录完全一致。
这仅证明关闭选项时该算例保持原行为。

首次开启选项的 32 网格试验
`runs/aphros32_velocity_backend_hybrid_v1` 于
2026-09-08 18:22:53.986920 UTC 被内存保护停止，退出码 15。
保护只终止经过身份验证的新 CPU 参考进程。输入未变，观测峰值
工作集为 283127808 字节、最大 private bytes 为 316694528 字节；
系统可用内存最低为 1662263296 字节。未完成第一物理步，不能
据它宣称混合后端精度、定常或性能验证通过。

原严格比较器仍要求打印的全部外迭代误差一致。后续实验同时
保留这个严格结果、数值场差异和独立质量检查；数值场检查通过
不能覆盖或改写严格比较失败。该比较器及其门槛没有修改。

## 32 重跑结果和 64 队列

`configs/queue_aphros_velocity_backend_probes_v3.py` 等待已核对的
128 对照进程自然结束，以及可用内存连续十秒达到 5 GiB 后，
在 `runs/aphros32_velocity_backend_hybrid_v2` 重跑了完整第一步。
该运行于 2026-09-08 18:51:09.868651 UTC 正常退出，41 次外迭代。

`checks/aphros_velocity_backend_probes_v3/pair32.json` 的严格判定
仍为 false，因为外迭代误差不是逐位相同。数值场和双方精确质量
检查均通过：速度、压力、壁面剪切及面速度相对 L2 差分别为
1.29179e-17、5.09798e-17、2.15160e-17、1.38277e-17。
`native32.json` 的所有 Cirrus 对照门槛也通过。

实际后端日志确认 123 次压力调用使用 AMG，三个速度分量各
328 次调用使用直接后端。观测墙钟为 271.617 秒，峰值工作集
283115520 字节；旧 AMG 运行观测为 336.610 秒。两次并发负载
不同，不能据此宣称独占加速比。

32 场值、质量及 Cirrus 对照完成后，队列等待 7 GiB 余量，运行
`runs/aphros64_velocity_backend_hybrid_v1`。它于
2026-09-08 19:20:08.828853 UTC 正常退出，完成 39 次外迭代和
一个 dt=0.005 的完整物理步。

`pair64.json` 的严格外迭代日志比较仍为 false，但双方数值场
与精确质量检查通过；速度、压力、壁面剪切、面速度相对 L2 差
分别为 4.32959e-17、5.07175e-17、5.66558e-17、4.31504e-17。
`native64.json` 对 Cirrus 自适应网格的全部门槛也通过，其中
速度差 1.34405e-4、压力差 7.67719e-5、切割单元速度差
2.77774e-4、壁面剪切差 1.16400e-4、流量差 1.05730e-5。
实际 Aphros 面通量的相对散度 Linf 为 3.93603e-10。

这一对 64 运行还存在 lazy-centers 开关差异，因此原性能比较器
按原规则拒绝了“仅后端不同”的比较。差异和原始资源记录另存
`profile64_combined_controls.json`，未修改检查器以取得通过。
混合后端加 lazy-centers 的观测墙钟为 1620.667 秒、峰值工作集
2716999680 字节；对照为 1758.984 秒、1671479296 字节。
并发负载和多个控制项不同，不能据此隔离后端的加速比。

队列初版 v2 在启动检查中误从资源报告读取退出码而报 KeyError，
没有启动求解器。实际退出码保存在 `run_completion.json`；
v3 读取该文件，失败分发记录保留。

所有新增大文件和计算留在 `D:/CirrusExperiments/cirrus-amg`。
本次混合后端仍是候选方案；独立 Aphros 定常对齐及最终近壁
网格收敛没有因此完成，Goal 保持进行中。另一路原生 128
完整时间步对照结果见 `PRESSURE_CYCLE_EXIT.md`。

## 用相同初猜推进定常参考

`scripts/run_aphros_seeded_backend.py` 使用现有已记录初猜，复制
完整参考配置并核对种子文件、准备来源、形状及运行时哈希。
执行后检查实际初值回写逐字节一致。它明确从 t=0 重算相同初值
问题，不声称从参考运行的某个物理步恢复；步长、总步数和原始
流动方程均保留。

`runs/aphros16_seeded_hybrid_v1` 已完成 128 步，dt=0.005，最终
时间变化指标为 5.15138e-14，实际初值回写逐字节一致。与原来的
同初猜参考比较，数值场与双方精确质量检查通过，严格外迭代日志
比较仍为 false。对 Cirrus 的全部场值、守恒和定常门槛通过，报告
位于 `checks/aphros_seeded64_hybrid_queue_v1/pair16.json` 和
`native16.json`。观测墙钟 844.125 秒，峰值工作集 59904000 字节。

`configs/queue_aphros_seeded64_hybrid_v1.py` 在这些检查完成、128
原生续算实际恢复且内存余量达到要求后，于
2026-09-08 19:34:28.440221 UTC 启动
`runs/aphros64_seeded_hybrid_v1`，实际 PID 71688。
`checks/continued_solvers_observation_v1` 已核对该进程、原始完整
配置和实际初猜回写字节。原来的冷启动和同初猜 Aphros 参考继续
独立运行。资源保护只针对新启动的 CPU 参考进程。
