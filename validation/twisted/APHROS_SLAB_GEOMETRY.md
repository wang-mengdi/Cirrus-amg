# 原始 Aphros 几何的分块导出与 256 输入

为继续处理 [64→128 空间收敛失败](SPATIAL_CONVERGENCE_64_128.md)，
新增一个仅生成几何的程序，在 z 方向逐块调用原始 Aphros
`Embed<MeshCartesian<double,3>>::Init`。已经生成了实际 256 几何；
尚未构造完整的 256 原生八叉树，也没有计算 256 流场。

## 原始算法与分块边界

`validation/aphros_geometry_slabs.cpp` 链接此前生成完整 128 参考
几何的同一份 Aphros 静态库，未重编或修改该库。库 SHA-256 为
`b8f2bbf83c69678da30f99c79874de2989e9eef1b4a91a13f2b68a4382952732`。
包含文件使用已冻结的 `geometry_state_double_sources_v1/src`。
构建沿用该参考程序的 MSVC C++17 `/O2` 选项，新程序 SHA-256 为
`e4e766bbf9f4ec1ce5b6b4ac3041127f3056e7605f7063ef17fb3ee28f2d114c`。

每块保留原来的全局整数坐标、全局网格尺寸、x 周期边界和两层
halo。节点值使用原参考驱动的 double 表达式计算，范围包括 halo。
本工具针对已有的未平移扭曲管：范围 `[.25,.125,.125]` m、半径
`.035` m、中心线幅度 `.015` m、周期 `.25` m。几何模板提供原来的
完整规格说明，工具检查固定几何参数。

网格节点、面开口、面积截断、切割法向、平面常数、体积和多边形
均通过原库生成。内部 z 块交界上的面由上方块输出一次，最上方
真实边界保留；x=0 与 x=L 的参考周期面记录仍分别输出。

原始 `Embed::Init` 会排队三个 halo 通信请求：邻域体积和、符号
距离和切割位移。本工具不进行流动求解，不使用这三个派生字段，
因此清除其通信请求。几何本身已经由包含 halo 的解析节点值计算。
程序检查原初始化的通信请求数和阶段数；这个几何对象不能复用于
流动求解。该处理不是对 Aphros 压力、速度或黏性算法的修改。

## 与完整区域导出的逐位验证

`scripts/check_aphros_geometry_slabs.py` 验证唯一几何编号，按编号
对齐后逐位比较所有列；不仅比较数量、法向或总流体体积。独立
参考来自此前完整区域 Aphros 导出，面/壁面多边形由原 CSV 流式
转为 double。原来的 17 位十进制记录可完整恢复 double。

| 实际检查 | 单元 | 面 | 壁面 | 多边形顶点 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 16，每块 3 层，含末尾不完整块 | 2744 | 7536 | 1368 | 35444 | 所有列逐位一致 |
| 16，每块 7 层，改变交界位置 | 2744 | 7536 | 1368 | 35444 | 所有列逐位一致 |
| 128，每块 8 层 | 1076936 | 3190604 | 87248 | 13109948 | 所有列逐位一致 |

数组记录顺序可以因分块而不同，因此上述结果是按唯一编号对齐后
一致。随后使用上一轮实际编译的 Cirrus 程序导入这份 128 几何，
生成 790216 个原生流体单元、2431944 个面；最终单元和面 CSV
与 `steady_outer128_v1` 的完整定常运行逐字节一致。端到端导入
检查于 2026-09-09 02:09:48 UTC 通过。

## 实际 256 几何

256 采用每块 4 层，原库导出于 2026-09-09 02:11:24 UTC 自然结束，
打包于 02:11:49 完成。独立拓扑检查于 02:14:10 通过：

| 项目 | 实际数值 |
| --- | ---: |
| 原始参考流体单元 | 8439328 |
| 正开口面（含两侧周期记录） | 25158384 |
| 合并重复周期记录后的面 | 25141960 |
| 切割壁面单元 | 349008 |
| 多边形顶点 | 102026716 |
| 需要加密的根 tile | 1704 |
| 最大面积向量闭合误差 / h² | 3.781e-14 |
| 最大周期开口面积差 / h² | 3.775e-14 |
| 最小流体体积分数 | 5.3403e-14 |

检查所有单元、面、壁面编号唯一；每个正开口都有两个流体邻居，
每个切割单元恰有一个壁面；体积为正且不超过完整立方体。所有
切割单元编号、流体单元总数和加密根标记均与解析节点分类器一致。
周期开口、法向和面积闭合沿用原导入器的相应阈值。极小切割单元
完整保留，不能据此推断其实际求解条件数或近壁流场已通过。

独立检查预估原生一级壁面加密会产生 20120 个物理叶 tile 和
3995168 个流体单元，与此前 [布局预测](WALL_REFINEMENT_LAYOUT.md)
一致。此处仍是预估，实际 256 原生拓扑尚未构造。

## 存储、内存与复现

输出采用已有 `CIRRCUT1` float64 单元/面/壁面格式，另有同格式的
八列多边形顶点表。`run_aphros_geometry_slabs.py` 记录实际编译输入、
可执行文件、原库、运行命令及产物 SHA-256，并根据真实壁面编号
生成相同的两格加密缓冲。大文件均保留在 D 盘。

Windows 实际导出进程累计峰值，MiB：

| 导出 | 工作集 | 提交计数 |
| --- | ---: | ---: |
| 16，每块 3 层 | 7.24 | 3.46 |
| 128，每块 8 层 | 190.59 | 190.11 |
| 256，每块 4 层 | 499.21 | 507.09 |

这些只涵盖 C++ 几何导出进程，不涵盖 Python 打包/校验、Cirrus
完整网格、流动求解或 Aphros 原始流动过程，也不是独占环境的
速度对比。256 仍需约 9 GB 的几何文件，分块不会消除输出规模。

```powershell
python scripts/run_aphros_geometry_slabs.py --build D:/CirrusExperiments/cirrus-amg/builds/aphros_geometry_slabs_v1 --ny 256 --slab-depth 4 --geometry-template output/twisted/packed_n128/geometry.json --output D:/CirrusExperiments/cirrus-amg/runs/<新目录>
python scripts/check_aphros_geometry_slabs.py --reference output/twisted/packed_n128/geometry.json --candidate D:/CirrusExperiments/cirrus-amg/runs/aphros_geometry_slabs128_d8_v1/geometry/geometry.json --output D:/CirrusExperiments/cirrus-amg/checks/<新目录>
python scripts/check_aphros_slab_topology.py --geometry D:/CirrusExperiments/cirrus-amg/runs/aphros_geometry_slabs256_d4_v1/geometry/geometry.json --output D:/CirrusExperiments/cirrus-amg/checks/<新目录>
```

实际 256 输入位于
`D:/CirrusExperiments/cirrus-amg/runs/aphros_geometry_slabs256_d4_v1/geometry/geometry.json`。
构建命令、源文件、验证和哈希归档于 `results/aphros_slab_geometry`。
现有 ParaView 导出器仍读取 CSV 壁面多边形；新二进制多边形输入
需要接入该导出路径后才能直接用于更细流场的可视化。

原生 128 和原始 Aphros 128 的运行未被替换、暂停或终止。这份
工作完成了更细参考几何的生成准备，没有改变当前空间验收失败，
没有证明 256 定常精度，也没有宣称独立 128 Aphros 对照已完成。
