# Aphros 笛卡尔坐标缓存与完整场回归

128 网格的独立 Aphros 参考此前触发了 RAM 资源限制。本项验证
使笛卡尔单元中心和面中心可以按原公式计算，省去坐标数组；
大型构建、测试和运行文件均放在 `D:/CirrusExperiments/cirrus-amg`。
该修改只作用于 Aphros 验证程序，Cirrus 原生八叉树和 GPU 算子
没有改动。

## 实现与数值范围

`validation/aphros/prepare_lazy_centers_reference.py` 从已验证的
`builds/aphros_cold_storage_v1` 复制独立源树，只修改 `driver.cpp`、
`src/geom/mesh.h` 和 `src/geom/mesh.ipp`。环境开关
`APHROS_TWISTED_LAZY_CENTERS` 默认关闭；开启后不构造两个
坐标数组，getter 使用原构造函数的同一表达式和求值顺序。
开关在数组和实现对象构造前初始化，运行期间不变。

原 `proj.ipp/proj.h/embed.ipp/approx_eb.ipp/convdiffi.ipp/convdiffe.ipp`
保持逐字节一致。没有改变计算区域、切割几何、边界条件、
物性、时间步或收敛门槛。全部 48 个翻译单元重新构建，关闭
fast-math 和浮点收缩。MinGW long double 使用 16 字节存储、
64 位有效尾数，并非 128 位数值精度。

实际执行文件为
`builds/aphros_lazy_centers_v1/twisted_extended.exe`，SHA256：
`58759aadd6f07bce4359fdb0ab54eb763dcaaba7cf6e1d3320c1cf69fa00e9a6`。
实际编译源保存在 `builds/aphros_lazy_centers_sources_v1`。

## 已完成验证

`check_cartesian_centers.py` 将同一个实际坐标测试分别链接到旧版
和新版对象，运行旧版、新版关闭开关、新版开启开关三种情况。
九组区域和 halo 宽度覆盖偏移及负索引、非二进制精确间距。
总计 51329 个单元、161409 个面；全部坐标输出为无损十六进制
long double。三份 CSV 逐字节相同，SHA256 均为
`5d5232c998274a1e7e4bb83a543f6d9376a3b503c2041c369193b70af2ed492f`。
报告：`checks/cartesian_centers_v1/result.json`。

完整流动回归使用 `runs/aphros16_cold_cache1_v1` 的相同几何和
配置，dt=0.001，一个物理步、一次黏性内迭代、缓存容量 1，
并开启已验证的构造缓存释放。

| 运行 | 坐标数组字节数 | 峰值工作集字节数 | 六份输出逐字节相同 |
|---|---:|---:|---|
| `aphros16_lazy_centers_off_v1` | 3132864 | 54419456 | 是 |
| `aphros16_lazy_centers_on_v1` | 0 | 45527040 | 是 |

六份输出覆盖速度/压力、面通量、壁面剪切、精确保存的单元/面
数据及时间变化记录。两次均保持原 260 次外迭代的全部误差序列，
各场相对 L2 差为零；真实存储通量的质量相对 Linf 均为
1.0592719191611579e-12。原独立比较器及门槛没有修改。
报告：`checks/aphros16_lazy_centers_off_pair_v1.json` 和
`checks/aphros16_lazy_centers_on_pair_v1.json`。

观测运行时间分别为 84.15 和 83.91 秒。其他任务同时运行，
不能据此宣称性能加速，也不能把整个峰值差都归因于坐标数组。

上一项 64 网格构造缓存回归现已完成：39 次外迭代及六份输出
与原版完全一致，实际质量相对 Linf 为 5.279026686027291e-10。
该运行尚未使用本项坐标优化。详见
[APHROS_COLD_STORAGE.md](APHROS_COLD_STORAGE.md)。

## 大规模实验与限制

新运行 `runs/aphros128_lazy_centers_cache1_v1` 使用完整
256×128×128 区域及原始 128 几何，dt=0.005，一个物理步、
8 次黏性内迭代。它组合坐标即时计算、构造缓存释放和容量 1
的原 AMG 预条件器缓存。容量下降会增加预条件器重建次数，
不改变离散算子或原方程残差检查。

原几何清点估计可省去 886431168 字节的坐标数组。这不是最终
128 运行峰值的实测降幅。运行器继续保留 2 GiB 可用 RAM、
持续 10 秒的资源保护，只能停止身份核实的新 CPU 参考进程。
最终状态以实际 `run_completion.json` 和验证报告为准；启动或
通过部分线性求解不构成完整 128 流场通过。

2026-09-08 13:30:24 UTC 的真实运行记录显示：3 次压力和每个
速度分量各 8 次线性求解已完成原始方程行残差检查，共 27 次；
仍未完成一个物理步。观测峰值工作集 7672287232 字节，最低
可用系统内存 2906779648 字节。构造缓存释放阶段实测工作集
减少 1255288832 字节，实际坐标数组占用为零。这已超过此前
128 版本只完成 3 次线性检查就触发资源保护的进度，但仍不能
保证后续阶段都能放入内存。

`configs/complete_lazy_centers128_v1.py` 已核实运行器和实际子进程，
仅在真实完成后调用原 `compare_twisted.py`，与已完成的原生
`ours_proj128_cgs2_steps2_v1/step_0001` 比较相同物理时刻。它
显式使用 transient、adaptive、原 8 次黏性内迭代和 extended
reference 验证；不会将第一步比较表述为定常验收。

为了腾出 RAM，旧 double 参考实验在保存第 15 个已完成物理步
后于 2026-09-08 13:24:17 UTC 被有意结束。保存前核实了 PID、
创建时间、执行文件哈希、目录、配置和下一个物理步已开始的
写入边界，完整输出复制前后哈希一致。快照位于
`runs/aphros64_double_retired_checkpoint_v1`；实际退出码 15。
它尚未定常，此保存是诊断证据，不是可重启状态或精度验收。
两组 extended-precision 64 网格参考和原生 GPU 128 计算继续运行。

本轮证据归档在 `results/aphros_lazy_centers_checkpoint`。
独立定常对齐、128 完整参考解、空间收敛和最终近壁精度仍待
完成，goal 保持进行中。
