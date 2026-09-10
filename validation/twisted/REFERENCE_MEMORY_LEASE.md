# 为细网格构造暂时调度原参考进程内存

2026-09-09 的 256 级算子探针需要较多物理内存。原 CPU Aphros 128
已经运行很久；本次调度保留同一个参考进程，通过有时限的暂停
暂时让出物理内存，完成探针后恢复，未从头运行基线。

协调器先检查 PID、进程创建时间、实际可执行文件、输入哈希，
并确认没有其他原生 GPU 实验。原进程仅有 Windows 系统控制台
`conhost.exe` 子进程，保留其运行状态。首次协调器在暂停前因
严格的“无子进程”检查退出；该失败记录保留，没有启动暂停、
换页或探针。第二版在核实此子进程身份后执行调度。

正式调度于 12:31:15.489514 UTC 暂停 PID 42880（创建时间
1788913281.3414037），然后调用 Windows `EmptyWorkingSet`。
该 API 从指定进程的工作集中移除可移除的页；代码没有写入
求解器的用户内存，也没有更改输入或求解选项。
参见 [Microsoft API 文档](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset)。

暂停前后私有提交量均为 8757374976 B，CPU 时间保持
39435.671875 秒，已完成时间记录的 206 字节前缀 SHA-256 为
`443cbaaf56f0346f324a22a8fba47c941a77e4c3e5cc6eaa75ec477fb4b81b35`。
协调器每 0.5 秒核对原进程身份及 CPU 时间；原输入文件和时间
记录在调度结束前再次核对。这里的私有字节计数是资源观测，
不是对整个进程内存内容逐字节取哈希。

暂停前先启动独立 watchdog。它检查协调器身份和 900 秒截止时间，
协调器退出或达到期限时恢复原进程。正常路径也在 `finally` 中
恢复；本次走正常路径，watchdog 随后退出。256 探针沿用 9.625 GiB
启动预算和 3.5 GiB 运行期保护，原数值容差未改变。

12:37:55.750351 UTC 自动恢复同一原进程，实际暂停
400.280640 秒。原参考没有终止或重启，恢复后 CPU 时间继续增加。
256 探针及 watchdog 均已退出。参考运行的墙钟时间包含此次暂停，
应结合本记录分析耗时，不能用它作为独占性能基准。

复现记录相对 `D:/CirrusExperiments/cirrus-amg`：

- 实际调度：`checks/native_operator256_memory_lease_v2` 中的
  `lease.json`、`paused.json`、`resumed.json`、`result.json`。
- 恢复后活跃检查：同目录 `resume_progress.json`。
- 暂停前检查失败：`checks/native_operator256_memory_lease_v1/failure.json`。
- 未启动的 v11 队列取消：`checks/native_operator256_probe_v11/queue_cancellation.json`。
- 协调器：`configs/run_operator256_memory_lease_v2.py`。

此调度只解决运行资源安排，不是新的数值对齐或空间精度证据。
