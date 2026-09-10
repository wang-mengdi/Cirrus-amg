# 实验存储位置

根据用户关于 C 盘空间的要求，后续新实验使用：

```text
D:\CirrusExperiments\cirrus-amg\runs
D:\CirrusExperiments\cirrus-amg\builds
D:\CirrusExperiments\cirrus-amg\checks
D:\CirrusExperiments\cirrus-amg\configs
```

新求解配置中的 `output` 使用上述 runs 目录下的新绝对路径。
构建脚本的 `--output` 使用 builds，检查报告使用 checks，启动
配置使用 configs。现有启动、构建和检查脚本支持这些绝对路径，
无需修改求解算法。

已启动的 C 盘实验继续使用其记录的路径。现有几何输入、编译
快照、历史结果及其哈希来源保持可访问；后续如需迁移已经结束的
实验，先核验完整性，并保留旧引用路径的访问方式。

Aphros 参考实验目前已经位于 D 盘的
`D:\Dropbox\Agent-simulation\twisted-baseline`。
提交到 Git 的小型检查记录和必要证据仍随代码库管理。
