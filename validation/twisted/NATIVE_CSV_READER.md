# 大网格轨迹比较的 CSV 读取

`check_twisted_time_pair.py` 改用 `numpy.loadtxt` 直接读取带字段名
的 float64 数组。保留 CSV 的全部列、行顺序、有限值检查，以及
原来的物理场、实际守恒和完整固定点验收条件。
旧的 `genfromtxt` 在读取大文件时需要较大的中间字符串列表。

在已完成的 128 网格两步算例中，逐项核对
`step_0002/iter_46/faces.csv` 的 2431944 行、7 列：新读取结果与
分 297 块调用原 `genfromtxt` 的结果，所有 float64 位均相同。
结果数组为 136188864 字节，包含参考分块核对的进程峰值工作集
为 273096704 字节，约 260.4 MiB。该数值是读取核对进程的峰值，
不代表整个两步流场比较的总内存。

NaN、无穷值、缺列、多列、空数据和重复列名的六项拒绝检查通过。
16 网格的新旧 GPU 构建完整 8 步比较通过，逐步场差异、守恒结果
及迭代数与修改前报告完全相同。

新运行数据位于 D 盘：

```text
D:\CirrusExperiments\cirrus-amg\checks\reader128_v1
D:\CirrusExperiments\cirrus-amg\checks\cgs2_n16_low_memory_pair_v1.json
```

128 网格第二步结束后，完整比较使用：

```text
python scripts/check_twisted_time_pair.py --candidate C:/Code/Cirrus-amg/output/twisted/ours_proj128_cgs2_steps2_v1 --reference C:/Code/Cirrus-amg/output/twisted/ours_proj128_anderson_failure_2steps_v1 --native-build-pair --output D:/CirrusExperiments/cirrus-amg/checks/cgs2_n128_full_pair_v1.json
```

这次检查验证解析器及比较程序保持一致，不是新的物理收敛结果。
定常 Aphros 对齐、时间步收敛及近壁网格收敛仍需完成。
