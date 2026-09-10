# 扭曲细管：当前结论与大内存机器重现

更新时间：2026-09-10 UTC（本机时区 2026-09-09）。分支：`feature/simple-octree-pipe`。
用户决定等待新机器，本阶段停止新增流动计算；总体精度目标仍未完成。
这份文档是接续入口。旧实验文档中的“正在运行”是当时的记录，当前状态以本文和
`D:/CirrusExperiments/cirrus-amg/handoff/reproduction_metadata_20260910/status/summary.json` 为准。

## 1. 已完成什么，仍缺什么

| 检查 | 当前结果 | 解释 |
|---|---|---|
| 原生 Cirrus 64 定常解 vs 独立 Aphros 64 | 通过原有五项门槛 | 速度、压力、切割单元速度、壁面剪切、截面流量均已比较 |
| 原生 Cirrus 128 定常外迭代 vs 普通时间推进到定常 | 通过 | 两种推进方式在同一网格上的场量差约 1e-9 或更小 |
| 64→128 空间收敛 | **未通过** | 流量差 0.6063% > 0.5%；最近壁速度差 2.6509% > 1%；原采样 WSS 差 1.9673% > 1% |
| Aphros 128 独立定常参考 | **未完成** | 内存保护退出，只有 2 个完整物理步，不能用于定常验收 |
| 原生 Cirrus 256 | **未完成** | 已构造真实八叉树和切割几何；93 次完整 GPU 线性求解，没有完成定常外层 map |
| 128→256 空间收敛 | 尚无可比较的 256 定常解 | 尚未验收 |

64 对齐和网格收敛是不同问题：同一组粗网格结果彼此接近，并不能证明真实近壁流动已足够准确。
保留的原始报告为 `D:/CirrusExperiments/cirrus-amg/checks/steady_outer64_direct_aphros_v1.json`
和 `checks/steady_refinement64_128_native_v1/refinement.json`。依赖包包含这些报告及其校验输入。

壁面后处理另有一个**诊断候选**：局部二次拟合使用面面积权重。
它减小了极小切割面引起的采样波动，制造解的 128→256 WSS 探针误差由约 1.84% 降至 0.93%。
但该方法尚未替换正式验证器，64 网格候选采样敏感度仍为 0.2689%，超过 0.25% 门槛。
近壁速度的制造解插值误差远小于实际 64→128 流动差，仍需要真实的细网格求解。
见 [WALL_CLOSURE_ACCURACY.md](WALL_CLOSURE_ACCURACY.md)，不能通过改采样或放宽门槛宣称整体通过。

## 2. 此版本实际计算的物理问题和算法

- **三维**周期扭曲细管；域为 `[0.25, 0.125, 0.125] m`，横截面半径 `0.035 m`。
  `yc(x)=0.0625+0.015 sin(2πx/0.25)`，`zc(x)=0.0625+0.015 cos(2πx/0.25)`。
  截面在固定 x 平面内为圆，不能把它当成另一个沿中心线法平面扫掠的几何。
- x 周期，静止无滑移曲面壁；`rho=1`、`nu=0.01 m²/s`、体力加速度 `[1,0,0] m/s²`。
  一周期等效压降为 `0.25 Pa`。输出 Pressure 是周期扰动压力，比较时移除体积加权均值。
- 原生 HA 八叉树保留，管壁附近一级加密、两格细网格缓冲。256 表示最细横向分辨率，
  并非全域均匀 `256³`。实际有 3,995,168 个流体未知单元、12,263,704 个共享面。
- 曲面来自 Aphros 原始平面切割几何，保留体积、开口面积、壁面位置/法向和多边形，
  保留原有极小切割单元。粗细交界共享子面通量，使用二次重构。
- 当前对齐管线使用标准 **projection + BCG 对流 + 隐式黏性**，不使用 flow map。
  分支也有 SIMPLE 实现，但这里的已验证运行配置明确为 `fluid_solver=proj`。
- 压力与隐式黏性使用原生 GPU 算子；预条件器是 Cirrus float AMG，外层精度恢复和残差检查用 double/twofold。
  这些生产算子没有以 CPU 显式矩阵分解替代。Aphros 独立参考则有自己的 CPU 稀疏线性后端。
- 内层 Anderson 深度 5；定常外层 Anderson 深度 5。后者是伪时间定常加速，输出目录 `iterate_####`
  不表示真实物理时间。普通非定常推进使用 `step_####`，必须分别验证。
- 不改变：线性真残差 `1e-13`、投影内迭代 `1e-11`、定常方程门槛 `1e-8`，`dt=0.005`。
  精确守恒检查使用通量高、低两部分，不能丢弃 `flux_low`。

实现入口：`simple/ProjectionSolver.cpp`、`simple/ProjectionAssembly.h`、`simple/EmbeddedMesh.cpp`、
`simple/EmbeddedOperators.cpp`、`simple/NativeCompactGpu.cu`、`simple/NativeAmgPreconditioner.cu`、
`simple/AndersonAcceleration.cpp`。相关说明见 [IMPLEMENTATION.md](IMPLEMENTATION.md)、
[NATIVE_HOST_STORAGE.md](NATIVE_HOST_STORAGE.md) 和 [STEADY_OUTER_ANDERSON.md](STEADY_OUTER_ANDERSON.md)。

## 3. 最后一次运行的事实

原生 256 的 PID 为 71580，创建时间 `1788991861.8417583`，最后一次控制器于
`2026-09-10 00:07:01 UTC` 将它暂停。93 次已完成线性求解均通过原门槛；
`steady_history.csv` 为零字节，尚无可用的 256 定常场。进程保留的内存、Tile backing 和
Anderson 历史**不是可迁移的完整检查点**。重启电脑会失去进程状态，新机器需从头启动 256。
重现包不恢复 PID，不运行暂停/恢复控制器，不使用这些部分输出初始化新求解。

Aphros 128 的 PID 42880 已退出：`2026-09-09 23:53:11 UTC`、退出码 15。
其运行器检测到可用 RAM 连续 10 秒低于 2.5 GiB，触发已有保护；输入哈希未改变。
经过约 23.53 小时墙钟时间（包含被暂停的时间），只完成 `t=0.005` 和 `t=0.01` 两步。
第三步未完成。这不是已收敛参考，也不是已证实可重启的 projection 检查点。

本机 32 GiB 内存不足以可靠完成当前重任务。Windows working-set trim 只能改变驻留页，
不能释放求解器仍持有的全部分配。反复交替暂停两个大进程仍触发过内存保护。
短 trim 试验只验证了过程控制，不能当作内存问题已解决。

## 4. git pull 与大文件：必须一起准备

**Git 只保存源码、脚本、小型算例配置和说明文档。所有实验数据、输入清单、校验结果、日志、图和压缩快照均保存在本地，通过 U 盘等介质转移。**
只 `git pull` 不会下载这些忽略文件。原有 128 速度初猜的来源校验会读取完整历史输入，
因此不能只复制一个 `velocity.bin` 或 `solution.csv`。

已完成逐对象解压校验：压缩包 5,455,075,817 字节（约 5.46 GB），恢复文件合计约 27.09 GB。
本机输入包位置：

```text
D:\CirrusExperiments\cirrus-amg\handoff\twisted_reproduction_20260910.zip
```

文件列表、每个原文件的大小/SHA-256、去重后大小见
`D:/CirrusExperiments/cirrus-amg/handoff/inputs.json`。包本身大小和 SHA-256 见
`D:/CirrusExperiments/cirrus-amg/handoff/reproduction_metadata_20260910/bundle_receipt.json`。
它包括 256 精确几何、已验证 64/128 状态的必要材料、Aphros 128 几何和速度初猜、
冻结的 Aphros 实际编译源/可执行程序，以及旧构建的来源证明；不包含活进程内存。
传输方式可用移动硬盘或内网复制。此包尚未上传到任何外部存储。

当前交付的可执行路径是 **Windows x64**。建议先在至少 64 GiB、最好 128 GiB RAM 的机器上顺序运行；
这是资源规划建议，未测得 256 最终峰值，不能保证一个固定配置必然足够。
准备 NVIDIA CUDA GPU；旧机器为 RTX 3080 10 GiB，旧二进制面向 sm_86。
GPU 型号不同应重新编译原生求解器，再执行小算例检查。为构建和后续 dump 预留至少 100 GiB 实验空间。
Linux 机器需要移植 Windows 运行/资源计数脚本，并重新核查 long double 和编译器行为；本文没有把 Linux 声称为已验证路线。

为原样复现冻结的绝对路径校验，使用以下目录：

```text
C:\Code\Cirrus-amg
D:\CirrusExperiments\cirrus-amg
D:\Dropbox\Agent-simulation\twisted-baseline
```

后两个可以是普通目录，不要求安装 Dropbox。新机器没有 D 盘时应先安排实验卷路径；
不要对旧 manifest 全局替换路径后仍把它们称为原始校验记录。

```powershell
# 新机器已有仓库时：
Set-Location C:\Code\Cirrus-amg
git config core.autocrlf false
git fetch origin
git switch feature/simple-octree-pipe
git pull --ff-only origin feature/simple-octree-pipe

# 尚无仓库时，在 C:\Code 下：
Set-Location C:\Code
git -c core.autocrlf=false clone --branch feature/simple-octree-pipe git@github.com:wang-mengdi/Cirrus-amg.git Cirrus-amg
```

远端分支需要先推送成功。原来的 1.50 GB Git bundle 含有误提交的历史实验数据，已废弃为迁移入口。
本次将未推送的实验分支整理成基于原 `origin/main` 的一个干净开发提交，不改写共享 main 或已有标签。
当前版本不再跟踪场数据、实验报告、压缩档或原有场景模型。原 main 历史中的旧模型仍属于既有共享历史，
新分支提交和下面的增量包不增加这些数据。

离线转移代码时，可使用仅含本分支新增代码和文档的增量 Git bundle：

```text
D:\CirrusExperiments\cirrus-amg\handoff\cirrus_code_docs_20260910.bundle
```

它需要原仓库 main 作为基础，不能单独 clone 成完整仓库。在已有仓库中导入：

```powershell
git fetch origin main
git fetch D:/CirrusExperiments/cirrus-amg/handoff/cirrus_code_docs_20260910.bundle feature/simple-octree-pipe:feature/simple-octree-pipe
git switch feature/simple-octree-pipe
```

以上针对新机器尚无该本地分支的情况。若已导入旧实验分支，先保留本机未提交修改，
再用新分支名导入清理后的提交；历史已整理，不能把旧的实验提交重新合并回来。
在新分支已建立后，后续普通更新仍使用 `git pull --ff-only`。

U 盘重现所需的数据文件是 `twisted_reproduction_20260910.zip` 和同目录的 `inputs.json`；
`reproduction_metadata_20260910` 是可选的本地校验说明。
另有 `local_validation_artifacts_20260910.zip` 保存从 Git 移出的旧实验材料，按需转移以审阅历史诊断。
这个归档可解压到仓库根目录恢复原有文档里的本地 `validation/.../results` 链接；恢复后的文件由 Git 忽略。
备份目录 `D:/CirrusExperiments/cirrus-amg/local_git_cleanup_backups/20260910` 仅用于本机恢复旧历史，不是日常迁移包。

安装 Python 3.11 或更新版本，以及依赖：

```powershell
python -m pip install numpy scipy psutil matplotlib vtk
Set-Location C:\Code\Cirrus-amg
$bundle = 'D:\CirrusExperiments\cirrus-amg\handoff\twisted_reproduction_20260910.zip'
python scripts/twisted_reproduction_bundle.py verify --bundle $bundle
python scripts/twisted_reproduction_bundle.py restore --bundle $bundle
python scripts/twisted_reproduction_bundle.py verify-local
```

`verify` 检查包内对象 SHA-256 和当前 Git 文件；`restore` 只写清单允许的三个根目录，
已有文件相同则跳过，不同则报错，避免覆盖旧实验；`verify-local` 再核对所有恢复结果。
工具不会启动求解器。Git 拉取时保留原始换行字节；三个历史 CRLF 源文件已用 `.gitattributes` 固定，以满足原有哈希检查。应逐条检查退出码，任一步失败先处理失败，不继续计算。
需要重新打包时，可在原机器对同一清单运行 `pack --bundle <新的zip路径>`，不会覆盖已有包。

## 5. 原生程序构建与小算例

安装 Visual Studio 2022 C++ 工具链、CUDA Toolkit、xmake，并使 `xmake` 可在 PATH 中找到。
原实验使用 MSVC 14.44 和 CUDA 12.9，依赖通过仓库的 xmake 配置解析；新环境可能需要联网获取依赖。
构建器记录实际编译源快照和二进制哈希，不要求不同硬件构建出的 exe 哈希与旧 exe 相同。
下面都使用新目录，重复实验时换后缀，保留之前的结果。

```powershell
Set-Location C:\Code\Cirrus-amg
xmake f -m release -p windows -a x64 -o D:/CirrusExperiments/cirrus-amg/builds/reproduction_cache_v1
python scripts/build_twisted_reproduction.py --output D:/CirrusExperiments/cirrus-amg/builds/reproduction_native_v1 --build-cache D:/CirrusExperiments/cirrus-amg/builds/reproduction_cache_v1 --source-inventory D:/CirrusExperiments/cirrus-amg/builds/native_host_storage_v2/build_manifest.json

$native = 'D:/CirrusExperiments/cirrus-amg/builds/reproduction_native_v1/simple_channel.exe'
$env:SIMPLE_NATIVE_HOST_TILES_FILE = '1'
$env:SIMPLE_ANDERSON_FILE_HISTORY = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python scripts/run_twisted_solver.py --config validation/twisted/reproduction/native16.json --exe $native --threads 2 --measure-memory
python scripts/check_twisted_twofold_native_run.py --run D:/CirrusExperiments/cirrus-amg/runs/reproduction_native16_v1 --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_native16_v1.json
```

16 是两步三维小算例，只验证可执行程序、算子/迭代和守恒，不证明管壁精度。
`native16.json` 的几何元数据仅把表路径改成绝对路径，原始几何表字节不变。
新机器应在小算例通过后再运行 128/256；这里没有在旧机器启动另一组 GPU 实验。

## 6. 先补齐独立 Aphros 128

依赖包中的 Aphros 程序是已验证、静态链接的 Windows CPU extended-precision 构建：

```text
D:/CirrusExperiments/cirrus-amg/builds/aphros_lazy_centers_v1/twisted_extended.exe
SHA256 58759aadd6f07bce4359fdb0ab54eb763dcaaba7cf6e1d3320c1cf69fa00e9a6
```

冻结源码位于 `builds/aphros_lazy_centers_sources_v1`；该构建的 `build_manifest.json`
记录了实际编译命令和源文件哈希。它保留 Aphros projection、黏性和 BCG 方程，
有中间状态 dump、原方程残差验证和已回归的存储优化。MinGW long double 为扩展精度，
16 字节存储不代表 128 位有效精度。不要用 MSVC 的 double 等价 long double 替换它。
这一路线使用原 CPU AMG，CUDA AMG 候选尚未完成 128 全场验收，不在默认复现路径中。

以下命令沿用已验证的原生 128 速度初猜；**仅初始化速度**，Aphros 自己求解压力和通量，
并检查源输入与真实初始化回写值。它从 t=0 重新运行，不从失败参考的第二步续算。

```powershell
python scripts/run_aphros_seeded_reference.py --reference-case D:/CirrusExperiments/cirrus-amg/runs/aphros128_lazy_centers_cache1_v1 --seed D:/CirrusExperiments/cirrus-amg/configs/initial_velocity128_native_steady_v1/velocity.bin --executable D:/CirrusExperiments/cirrus-amg/builds/aphros_lazy_centers_v1/twisted_extended.exe --output D:/CirrusExperiments/cirrus-amg/runs/reproduction_aphros128_v1 --steps 128 --factor-cache-entries 1 --release-cold-storage --lazy-centers

python scripts/compare_twisted_steady_aphros.py --ours D:/CirrusExperiments/cirrus-amg/runs/steady_outer128_v1/iterate_0023 --aphros D:/CirrusExperiments/cirrus-amg/runs/reproduction_aphros128_v1 --adaptive --aphros-diffusion-iterations 8 --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_aphros128_pair_v1.json
```

先等运行真实退出且 `initial_run_completion.json` 通过，再执行比较。128 步是预定运行量，
不是预先保证已定常；比较器仍会检查最终时间变化和内部收敛。不满足门槛就保留失败报告。
运行器保留原有 2.5 GiB 可用内存持续 10 秒的保护，不通过降低这个限制强行完成。

若需要重新计算原生 64 或 128，使用同一原生运行命令，把配置分别换成
`validation/twisted/reproduction/native64.json` 或 `native128.json`；分别写到
`runs/reproduction_native64_v1`、`runs/reproduction_native128_v1`。
新结果用于比较前同样要通过下一节的定常状态检查。

## 7. 再完成原生 256 和 128→256 检查

确认没有其他重型求解同时运行，保留上一节的两个 `SIMPLE_*` 环境变量。
大机器使用普通运行器连续求解；不复用原机器的暂停控制器、PID 或部分 Anderson 文件。
普通运行器记录峰值计数，**本身没有循环检查 RAM 并暂停的保护**；开始前应确认资源，
运行时观察可用内存，若资源仍不足则保存失败记录并停止该新任务，不降低精度门槛。

```powershell
python scripts/run_twisted_solver.py --config validation/twisted/reproduction/native256.json --exe $native --threads 2 --measure-memory

python scripts/check_twisted_steady_iteration.py --run D:/CirrusExperiments/cirrus-amg/runs/reproduction_native256_v1 --state-only --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_native256_state_v1.json
```

只有 `run_completion.json` 正常退出、`steady_summary.json` 中 `steady_converged=true`、
且状态检查通过之后才进入空间比较。读取真实 `final_output`，不能假定一定是 `iterate_0064`。

```powershell
$summary = Get-Content D:/CirrusExperiments/cirrus-amg/runs/reproduction_native256_v1/steady_summary.json -Raw | ConvertFrom-Json
$fine = $summary.final_output
python scripts/analyze_twisted_refinement.py --coarse D:/CirrusExperiments/cirrus-amg/runs/steady_outer128_v1/iterate_0023 --fine $fine --coarse-steady-iteration --fine-steady-iteration --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_refinement128_256_v1
```

空间门槛保持：Q 0.5%、近壁 U 1%、WSS 1%、压力 1%、采样敏感度 0.25%。
正式比较继续使用原采样方法；面积加权候选单列诊断，不能替换失败结果。
如果 256 仍失败，应按几何/壁面离散、粗细界面、空间误差分别定位，不能凭残差小就验收。
独立 Aphros 128 对齐、128→256 收敛及必要的时间步检查全部成立后，才能重新评估总体目标。

## 8. ParaView 与接续记录

恢复包后，已有的 128 定常解可重新导出；256 则必须等到上一节真正完成：

```powershell
python scripts/export_twisted_paraview.py --run D:/CirrusExperiments/cirrus-amg/runs/steady_outer128_v1/iterate_0023 --output D:/CirrusExperiments/cirrus-amg/runs/reproduction_native128_viz_v1
# 256 完成后，使用上节取出的 $fine：
python scripts/export_twisted_paraview.py --run $fine --output D:/CirrusExperiments/cirrus-amg/runs/reproduction_native256_viz_v1
```

ParaView 打开 `solution.vtu`，Cell Data 中选择 `Speed`、`Velocity`、`Pressure`、`Level`；
壁面打开 `walls.vtp`，选择 `WallShearMagnitude` 或 `WallShear`。这些字段分别是流体速度/压力
与壁面剪切，不能把网格级别或 residual 当成物理速度。ParaView 的显示插值不能代替数值比较器。

后续每次实验使用新目录，保留配置、实际二进制/编译源、运行环境、开始/结束记录、
中间 dump、正式检查报告，全部保存到 D 盘并单独转移。Git 只提交代码和说明中的结论，不提交这些数据。
本次交付验证了依赖清点、包内容和准备入口，不宣称在尚不可用的新机器上完成了构建或计算。


## 9. 防止数据再次进入 Git

`AGENTS.md` 和 `.gitignore` 固定了代码/文档与本地数据的边界。提交前运行
`python scripts/check_git_content.py --staged`；交付前运行
`python scripts/check_git_content.py --base origin/main`，检查新增提交历史中没有再次带入数据。
不能通过压缩、改后缀或 `git add -f` 绕过该要求。已有 main 和标签的历史不在本次改写范围内。
