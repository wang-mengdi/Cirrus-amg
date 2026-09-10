# 原生八叉树 GPU 元数据备份

`simple/NativeCompactGpu.cu` 在借用原生网格时，只修改 Tile 的邻居指针、
单元类型和序号。此前的两个 `HAHostTileHolder` 会同时复制每个 Tile
的全部 15 个数值通道。现在保存从 `mNeighbors` 开始的完整后缀，
编辑副本也只包含这个后缀。数值通道始终留在原来的 GPU Tile 中。

编译期检查要求 Tile 为标准布局、数值通道正好位于后缀之前，并且
需要修改的类型及序号位于后缀内。原网格的层次、遍历顺序、Tile
选择、单元槽位以及周期邻居处理保持一致。第一次上传前就标记为
已修改，构造失败时也由成员析构函数恢复原后缀。借用的网格由共享
所有权保持存活，恢复操作使用保存的原设备地址。

这次修改没有改变压力、黏性、粗细交界面或壁面离散，也没有改动
GPU 算子、FGMRES 和 AMG 求解函数。它减少的是 CPU 端临时及持久
备份，不代表整个求解器内存按同一比例下降。

## 执行检查

新增 `validation/native_tile_metadata_audit.cu`，由
`native_compact_gpu_audit` 的 `audit_metadata_restore: true` 开启。
它在 D 盘流式保存完整 Tile 字节，为所有 15 个数值通道设置不同的
非零哨兵值，并检查：

- 上传前初始化失败后，全部 Tile 字节恢复；
- 上传后初始化失败后，全部 Tile 字节恢复；
- 正常执行压力/黏性算子和 AMG 求解时，所有数值通道字节保持不变；
- 正常析构后，全部 Tile 字节恢复；最后恢复原算例的完整初始状态。

实际 16/64 网格分别覆盖 16 和 1049 个 Tile，两者均通过这四项检查，
随后原有的独立 CPU 显式算子与 GPU 算子/求解检查也通过。64 网格
包含全部 1536 个粗细交界面，并使用空间变化的面系数。实际布局为
每 Tile 44880 B，其中数值通道 44160 B、元数据后缀 720 B。

验证可使用同一几何配置，加上 `audit_metadata_restore: true` 后执行：

```powershell
& D:/CirrusExperiments/cirrus-amg/builds/tile_metadata_v1/native_compact_gpu_audit.exe <config.json> <全新的D盘输出目录>
```

完整字节快照保留在 D 盘，不复制到代码仓库。报告及源码哈希记录在
`D:/CirrusExperiments/cirrus-amg/checks/tile_metadata_audit_v1/result.json`。
这些检查验证实现一致性；独立 Aphros 对齐、实际稳态和网格收敛仍
分别验收，不能用上述通过结果替代。

## 实际流动回归

CPU 16、GPU 16、自适应 GPU 64 均完成两个实际时间步，与
`shared_quadratic_v1` 冻结构建对照。CPU 的中间和最终输出逐字节
一致；GPU 的速度、压力、切割单元速度、壁面剪切、共享面通量，
最大相对 L2 差分别为 `2.3227508930605824e-16` 和
`1.1184310739235221e-14`。原固定点、实际质量守恒及严格线性容差
检查均通过，初始化输出一致。GPU 64 的实测峰值工作集从
738689024 B 降为 691957760 B；这是一组观测，未作独占性能测试。

报告：`D:/CirrusExperiments/cirrus-amg/checks/tile_metadata_regression_v1/`。

## 128 级完整初始化

2026-09-09 13:15:45 UTC 的实际初始化通过，全部 50 个阶段完成；
除新增的两个元数据内存观测点外，阶段顺序与前版相同。六份完整
网格、材料和算子检查输出逐字节一致。峰值工作集从 2458808320 B
降为 2182803456 B，减少约 263.2 MiB；峰值提交量从 4136284160 B
降为 3848982528 B。此运行仅构造算子，没有计算新的流动结果。

报告：`D:/CirrusExperiments/cirrus-amg/checks/tile_metadata128_v1/result.json`。

## 256 级的新失败位置

`native_operator256_probe_v13` 完成全部几何导入、CPU 算子检查和
GPU 场量初始化，共到达 46 个观测阶段；压力与黏性离散检查均
通过，仍有 3995168 个流体单元、12263704 个面、280576 个粗细
交界面。`projection.gpu_host_operators_ready` 的工作集为
8327905280 B，元数据备份完成为 8350572544 B，GPU 场量完成为
8406446080 B。之前的 v12 在这一段因低内存保护停止，本次没有
触发内存保护。

随后原生 AMG 初始化报错：

```text
SIMPLE error: Native AMG missing a required face/ghost tile
```

`simple/NativeAmgPreconditioner.cu` 的面系数装配中，某个面要求的
Tile 在 AMG 主机层级查找表中不存在。该文件与前版完全相同；
本次记录定位了执行位置，尚未确定缺失 Tile 的坐标及原因。
不能省略这个面或把失败算例记为通过。下一步须记录对应面的
两侧层级、坐标和周期映射，修复层级覆盖并重新验证。

本次进程于 2026-09-09 13:21:51 UTC 以状态 1 退出，峰值工作集
10438307840 B，峰值提交量 12483133440 B。由于前后运行终止于
不同位置，不能把这两个未完成运行作为完整内存性能比较。256
尚未完成全部初始化，没有新的流动或 ParaView 速度结果。

这次资源调度暂停原 CPU Aphros 128 进程约 393.93 秒，独立看护
上限为 900 秒。13:22:35 UTC 恢复同一 PID 42880；13:24:15 UTC
确认 CPU 时间继续增加、输入和已完成时间步前缀保持不变。
运行协调报告因探测失败保留 `passed: false`；单独的恢复核验
通过，不能将恢复成功称为 256 求解成功。

完整记录位于 D 盘的 `checks/native_operator256_probe_v13/`、
`checks/native_operator256_memory_lease_v3/` 和
`checks/tile_metadata_summary_v1/`。归档位于
`validation/twisted/results/native_tile_metadata/`，只包含压缩的
小型证据和源码快照，大型字段及 Tile 字节快照保留在 D 盘。

原 64→128 空间收敛失败仍然有效，参见
[空间收敛报告](SPATIAL_CONVERGENCE_64_128.md)。本次修改与验证
没有完成整个 Aphros 对齐和近壁精度目标。
