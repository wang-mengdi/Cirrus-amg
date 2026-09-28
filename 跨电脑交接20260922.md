# Cirrus-amg 新电脑交接说明

更新：2026-09-22。适用于继续“保留八叉树，用标准黏性流算法对齐 Aphros”的任务。
本文整理迁移步骤和接续目标，没有在旧电脑重新启动求解。旧文档中的 PID、正在运行状态和输出路径是历史记录，以本文的清理后状态为准。

## 1. 先确认要继续哪条路线

目标：保留 Cirrus 原生 HA 自适应八叉树，在管壁附近加密，不使用 flow map，准确求解三维扭曲细管中的定常不可压缩黏性流。可以通过非定常推进或伪时间迭代达到定常。关注速度、压力、流量和近壁剪切。

独立参考是 **Aphros**，目前没有 CFX 实测对齐结果。Aphros 同时有 SIMPLE 与 Proj：

| 阶段 | Cirrus 与 Aphros 的对应关系 | 接续时如何理解 |
|---|---|---|
| 早期 | SIMPLE 对 SIMPLE | CPU double 实现；验证过直管、方管，以及曲管的部分 64/128 算例，并非只做过直管。代码保留。 |
| 当前主线 | Proj 对 Proj | 投影法 + BCG 对流 + 隐式黏性；保留的成功 64 结果属于这条路线。 |

转向 Proj 的背景：研究切割单元近壁误差时，一条完整残差重分配实验路线出现异常；Aphros 原 Proj 的隐式黏性路径不采用该黏性残差重分配，因此复现它作为另一条对照路线，随后接入原生 GPU 后端。**这不证明 SIMPLE 不适用，也没有完成 SIMPLE 与 Proj 精度优劣的受控比较。** 不应因分支或可执行文件名包含 `simple`，把 Proj 结果记成 SIMPLE 结果。详见 [实现与历史诊断](validation/twisted/IMPLEMENTATION.md)。

当前压力和隐式黏性线性求解使用原生 GPU 算子与 Cirrus float AMG，配合 double/twofold 精度恢复、原方程残差和守恒检查。CPU 端仍有几何、重构及离散算子数据，整个程序尚不能称为全程无矩阵或已解决内存问题。Aphros 参考保留原流动离散，包含诊断输出、扩展精度线性后端和已回归的存储优化，不是未经修改的原版可执行文件。

## 2. 已完成的结果与明确缺口

| 检查 | 当前结论 |
|---|---|
| 早期 SIMPLE 标准通道 | 11 组验证和三组细化检查通过；细八叉树速度/流量解析误差为 0.0741%/0.1068%。 |
| 当前 Proj：Cirrus 64 vs 独立 Aphros 64 | 速度、压力、切割单元速度、WSS、截面流量五项既定门槛通过。 |
| 原生 128 定常加速 vs 同网格普通时间推进 | 通过，场量相对差约 `1e-9` 或更小。 |
| 历史 64 时间步减半检查 | 既定门槛通过；两次运行来自不同构建，需保留这一限制。见 [检查记录](validation/twisted/TIMESTEP64_COMPLETION.md)。 |
| 当前 Proj：64→128 空间收敛 | **未通过**：流量差 0.6063% > 0.5%；最近壁速度差 2.6509% > 1%；原采样 WSS 差 1.9673% > 1%。 |
| 当前 Proj：独立 Aphros 128 | **未完成**，旧机触发内存保护，仅完成两个物理步，不能作定常参考。早期 SIMPLE 128 完成过，不要混淆。 |
| 原生 Proj 256 | **没有定常结果**。旧运行完成几何构建及 93 次 GPU 线性求解，但未完成一个定常外层更新。 |
| 128→256 空间收敛 | 尚无可验收的 256 定常场。 |

同网格对齐、迭代收敛和网格收敛是三件不同的事。不能凭残差小或 64 对齐就宣称近壁精度已经达标。
面积加权壁面拟合仍是诊断候选，尚未替换正式验证器，不允许用改采样或放宽门槛消除失败。

## 3. 代码、数据及旧机清理后的状态

2026-09-22 已只读核实远端：

- 仓库：`https://github.com/wang-mengdi/Cirrus-amg`
- 分支：`feature/simple-octree-pipe`
- 已在远端的代码检查点：`08e78a58a21cad0bfe8da1f1525b578f336e34ce`（后续交接文档提交可在其后）。
- 当前代码工作区在核查时干净。原 `main` 为 `c6b0d01273a35a1f3ce664c0f4f8292ab045b498`。

Git 仅保存代码、脚本、小型算例配置和文档。**只 clone/pull 仓库不能重现实验，还要通过 U 盘拷贝本地数据包。**

旧机经用户授权清理后，旧 256 求解器及运行器均已结束，不能再恢复旧 PID。`runs` 中只保留以下两组关键结果，138 个文件曾逐项校验恢复：

```text
D:\CirrusExperiments\cirrus-amg\runs\steady_outer64_v1
  最终状态：iterate_0016
D:\CirrusExperiments\cirrus-amg\runs\aphros64_initial_native_v1
```

旧原生 128 最终场、256 几何、必要基线来源等仍在重现包中，按下节恢复。旧调试目录、旧编译缓存和旧 Git 历史备份已大量删除；并非所有历史中间 dump 都被打包。不要直接照历史日志访问已删除目录，也不要把旧暂停状态当成检查点。

### 必须从旧电脑拷走

源目录：`D:\CirrusExperiments\cirrus-amg\handoff`。

| 文件 | 大小 | 用途 |
|---|---:|---|
| `twisted_reproduction_20260910.zip` | 5,455,075,817 字节，约 5.46 GB | 必需；关键几何、64/128 状态、Aphros 初猜/程序/源码快照及校验依赖。 |
| `inputs.json` | 450,804 字节 | 必需；冻结的文件清单、路径、大小和哈希。 |
| 本交接文档 `HANDOFF_CN_20260922.md` | 小文件 | 建议随 U 盘携带；Git 上的旧提交不一定包含本次文档。 |
| `local_validation_artifacts_20260910.zip` | 2,081,982,387 字节，约 2.08 GB | 可选；审阅旧 `validation/.../results` 诊断材料。 |
| `reproduction_metadata_20260910`、`cleanup_*.json` | 小文件 | 可选；原打包、清理与校验记录。 |

主数据包与清单 SHA-256（2026-09-22 重新核对通过）：

```text
twisted_reproduction_20260910.zip
03f615d2c770f82bebe62d011c6f6bc8fe0060c59cb65e1cfa6c1d82290f2541

inputs.json
cd7807138ceb462f23bb9595958ce48d821f4e9d69f15e660dda598f8f1df880
```

包按内容哈希去重，**必须通过项目的 restore 脚本还原，不能直接解压后当作原始目录使用**。清单共 1,833 个文件，恢复后的逻辑大小约 27.09 GB，其中 27 个源码文件由 Git 提供。

## 4. 新机器准备和下载

当前实验迁移流程已验证的平台是 **Windows x64**。Linux 下基础项目可构建不等于本实验迁移脚本已验证；若用 Linux，应先移植绝对路径、Windows 进程/内存计数和构建流程，再核对扩展精度行为，不直接运行以下 PowerShell 命令。

资源建议基于旧机 32 GiB 内存失败的经验：至少准备 64 GiB，优先 128 GiB RAM，顺序运行重任务；256 完整峰值尚未测得，不能保证某个容量必然够用。预留至少 100 GiB 实验空间，观察新机实际峰值再决定是否继续。

工具：Git、Python 3.11+、Visual Studio 2022 C++ 工具链、CUDA Toolkit、xmake 3.0+；ParaView 可选。旧实验用 MSVC 14.44、CUDA 12.9、RTX 3080 10 GiB。旧原生二进制面向 `sm_86`；新 GPU 应重新构建，当前 xmake 目标使用 `add_cugencodes("native")`。

为保留原始绝对路径和来源校验，使用：

```text
C:\Code\Cirrus-amg
D:\CirrusExperiments\cirrus-amg
D:\Dropbox\Agent-simulation\twisted-baseline
```

最后一个只是目录名称，**无需安装 Dropbox**。当前重现构建器还要求输出和缓存位于 D 盘。没有 D 盘时，先安排合适的数据卷，或明确移植脚本并生成新的来源记录；不要直接替换旧清单中的路径、跳过校验后宣称原样重现。

使用有仓库访问权限的账号，在 PowerShell 中执行；每条外部命令均须正常退出再继续：

```powershell
New-Item -ItemType Directory -Force C:\Code | Out-Null
Set-Location C:\Code
git -c core.autocrlf=false clone --branch feature/simple-octree-pipe https://github.com/wang-mengdi/Cirrus-amg.git Cirrus-amg
if ($LASTEXITCODE -ne 0) { throw 'clone failed' }
Set-Location C:\Code\Cirrus-amg
git config core.autocrlf false
git status --short
git log -1 --oneline
git merge-base --is-ancestor 08e78a58a21cad0bfe8da1f1525b578f336e34ce HEAD
if ($LASTEXITCODE -ne 0) { throw 'Not the expected experiment branch' }
python -m pip install numpy scipy psutil matplotlib vtk
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies failed' }
```

已有仓库则先保留未提交修改，再 `git fetch origin`、`git switch feature/simple-octree-pipe`、`git pull --ff-only origin feature/simple-octree-pipe`。不要把原来含大量数据的旧实验历史重新合并回来。

本次可另外提供 `cirrus_code_docs_20260922.bundle`，用于携带新增交接文档和完整实验分支；它是基于原 main 的增量包，不是独立仓库。旧的 `cirrus_code_docs_20260910.bundle` 只到 `20529ff`，不含后来的 README 和本交接文档。
若远端尚未包含新文档，可在已 clone 的仓库中导入新包到临时本地分支，再快进当前分支：

```powershell
git bundle verify D:/CirrusExperiments/cirrus-amg/handoff/cirrus_code_docs_20260922.bundle
if ($LASTEXITCODE -ne 0) { throw 'Code bundle prerequisites missing' }
git fetch D:/CirrusExperiments/cirrus-amg/handoff/cirrus_code_docs_20260922.bundle feature/simple-octree-pipe:handoff-import-20260922
if ($LASTEXITCODE -ne 0) { throw 'Code bundle import failed' }
git merge --ff-only handoff-import-20260922
if ($LASTEXITCODE -ne 0) { throw 'Local history differs; inspect before merging' }
```

## 5. 恢复数据并验证，不先跑大算例

把主数据包和 `inputs.json` 复制到新电脑的 `D:\CirrusExperiments\cirrus-amg\handoff`。随后：

```powershell
Set-Location C:\Code\Cirrus-amg
$bundle = 'D:/CirrusExperiments/cirrus-amg/handoff/twisted_reproduction_20260910.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath $bundle
Get-FileHash -Algorithm SHA256 -LiteralPath D:/CirrusExperiments/cirrus-amg/handoff/inputs.json
# 先核对上节两个哈希，再执行以下命令。
python scripts/twisted_reproduction_bundle.py verify --bundle $bundle
if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed' }
python scripts/twisted_reproduction_bundle.py restore --bundle $bundle
if ($LASTEXITCODE -ne 0) { throw 'Input restore failed' }
python scripts/twisted_reproduction_bundle.py verify-local
if ($LASTEXITCODE -ne 0) { throw 'Restored inputs differ' }
```

`restore` 只恢复允许的三个根目录；同名文件相同则跳过，不同则拒绝覆盖。源文件哈希检查失败时先核对分支、Git 换行设置及 `.gitattributes`，不要删除校验逻辑。
旧机清理后直接执行 `verify-local` 会发现很多未恢复文件，这符合当前状态；新机必须先 restore。

恢复后可先复查已有 64 对齐，不需重新求流场：

```powershell
python scripts/compare_twisted_steady_aphros.py --ours D:/CirrusExperiments/cirrus-amg/runs/steady_outer64_v1/iterate_0016 --aphros D:/CirrusExperiments/cirrus-amg/runs/aphros64_initial_native_v1 --adaptive --aphros-diffusion-iterations 8 --output D:/CirrusExperiments/cirrus-amg/checks/handoff64_pair_v1.json
if ($LASTEXITCODE -ne 0) { throw 'Restored 64 comparison failed; investigate inputs first' }
```

这是计划在新机器执行的复查命令，本次文档整理没有在旧机重新恢复全部输入或重新运行该比较。

## 6. 构建并通过 16 小算例

先确认 `xmake --version`、`nvcc --version`、`nvidia-smi` 和 C++ 工具链可用。使用全新输出目录，不覆盖旧二进制、来源快照或结果。首次 xmake 配置可能需要联网下载依赖。

```powershell
Set-Location C:\Code\Cirrus-amg
xmake f -m release -p windows -a x64 -o D:/CirrusExperiments/cirrus-amg/builds/reproduction_cache_v1
if ($LASTEXITCODE -ne 0) { throw 'xmake configuration failed' }
python scripts/build_twisted_reproduction.py --xmake (Get-Command xmake).Source --output D:/CirrusExperiments/cirrus-amg/builds/reproduction_native_v1 --build-cache D:/CirrusExperiments/cirrus-amg/builds/reproduction_cache_v1 --source-inventory D:/CirrusExperiments/cirrus-amg/builds/native_host_storage_v2/build_manifest.json
if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }

$native = 'D:/CirrusExperiments/cirrus-amg/builds/reproduction_native_v1/simple_channel.exe'
$env:SIMPLE_NATIVE_HOST_TILES_FILE = '1'
$env:SIMPLE_ANDERSON_FILE_HISTORY = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python scripts/run_twisted_solver.py --config validation/twisted/reproduction/native16.json --exe $native --threads 2 --measure-memory
if ($LASTEXITCODE -ne 0) { throw '16 case failed' }
python scripts/check_twisted_twofold_native_run.py --run D:/CirrusExperiments/cirrus-amg/runs/reproduction_native16_v1 --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_native16_v1.json
if ($LASTEXITCODE -ne 0) { throw '16 case validation failed' }
```

16 是 `dt=0.001` 的两个物理步，只检查程序、算子、迭代及守恒，不证明近壁精度。两个 `SIMPLE_*` 环境变量是存储优化开关，并不表示当前算法为 SIMPLE。关闭重开终端后须重新设置 `$native` 和这些环境变量。

## 7. 接续计算顺序

### 第一步：完成独立 Aphros 128（Proj）

重现包提供已验证的 Windows CPU 扩展精度 Aphros 程序与实际源码快照。使用 MinGW 的扩展精度 `long double`；不能换成 MSVC 的 double 等价实现而仍称相同参考。CUDA AMG 候选尚未完成 128 全场验收，不是默认路线。

下面仅用原生 128 的速度作为初猜；Aphros 自己求解压力和通量，并检查初始化来源。128 个物理步是计划运行量，不保证一定达到定常。

```powershell
python scripts/run_aphros_seeded_reference.py --reference-case D:/CirrusExperiments/cirrus-amg/runs/aphros128_lazy_centers_cache1_v1 --seed D:/CirrusExperiments/cirrus-amg/configs/initial_velocity128_native_steady_v1/velocity.bin --executable D:/CirrusExperiments/cirrus-amg/builds/aphros_lazy_centers_v1/twisted_extended.exe --output D:/CirrusExperiments/cirrus-amg/runs/reproduction_aphros128_v1 --steps 128 --factor-cache-entries 1 --release-cold-storage --lazy-centers
if ($LASTEXITCODE -ne 0) { throw 'Aphros 128 did not finish successfully' }
python scripts/compare_twisted_steady_aphros.py --ours D:/CirrusExperiments/cirrus-amg/runs/steady_outer128_v1/iterate_0023 --aphros D:/CirrusExperiments/cirrus-amg/runs/reproduction_aphros128_v1 --adaptive --aphros-diffusion-iterations 8 --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_aphros128_pair_v1.json
if ($LASTEXITCODE -ne 0) { throw 'Aphros 128 alignment failed; preserve the report' }
```

只把正常完成并通过真实定常、内部收敛及守恒检查的参考用于验收。保留现有 2.5 GiB 可用内存持续 10 秒的保护条件，不降低它强行运行。

### 第二步：从头完成原生 256

先结束上一项重型计算，再启动这一项。旧 256 的部分输出、AMG Tile 文件和 Anderson 历史不是完整重启点。普通运行器记录峰值，但没有自动暂停保护；执行中需监测实际可用内存。

```powershell
python scripts/run_twisted_solver.py --config validation/twisted/reproduction/native256.json --exe $native --threads 2 --measure-memory
if ($LASTEXITCODE -ne 0) { throw 'Native 256 did not finish successfully' }
python scripts/check_twisted_steady_iteration.py --run D:/CirrusExperiments/cirrus-amg/runs/reproduction_native256_v1 --state-only --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_native256_state_v1.json
if ($LASTEXITCODE -ne 0) { throw 'Native 256 state validation failed' }
```

`run_completion.json` 须正常退出，`steady_summary.json` 须有 `steady_converged=true`，状态检查须通过。`iterate_####` 是定常伪迭代，不能冒充真实时间轨迹；`step_####` 才是普通物理时间推进输出。

### 第三步：128→256 近壁空间检查

```powershell
$summary = Get-Content D:/CirrusExperiments/cirrus-amg/runs/reproduction_native256_v1/steady_summary.json -Raw | ConvertFrom-Json
if (-not $summary.steady_converged) { throw '256 is not steady' }
$fine = $summary.final_output
python scripts/analyze_twisted_refinement.py --coarse D:/CirrusExperiments/cirrus-amg/runs/steady_outer128_v1/iterate_0023 --fine $fine --coarse-steady-iteration --fine-steady-iteration --output D:/CirrusExperiments/cirrus-amg/checks/reproduction_refinement128_256_v1
if ($LASTEXITCODE -ne 0) { throw 'Spatial checks failed; retain the diagnostics and investigate' }
```

正式空间门槛：流量 0.5%、固定物理探针处近壁速度 1%、WSS 1%、压力 1%、采样敏感性 0.25%。另保持线性真残差 `1e-13`、投影内迭代 `1e-11`、定常方程 `1e-8`。主线 64/128/256 的 `dt=0.005`；不能把减小残差当作空间误差已消失。

若仍失败，分别定位切割几何/壁面闭合、粗细界面、离散误差和采样敏感性，再做改动及同算法基线回归。必要的时间步减半应在同一构建、同一网格下进行。最终报告必须分别列出“独立 Aphros 对齐”“时间敏感性”“空间与采样敏感性”，不能混为一个通过标志。

## 8. 算例定义和可视化

三维周期扭曲细管：域 `[0.25, 0.125, 0.125] m`，半径 `0.035 m`，
`yc=0.0625+0.015 sin(2πx/0.25)`，`zc=0.0625+0.015 cos(2πx/0.25)`。
圆截面位于固定 x 平面，并非中心线法平面的扫掠圆；x 周期、静止无滑移壁，`rho=1`、`nu=0.01 m²/s`、体加速度 `[1,0,0] m/s²`。压力是周期扰动压力，比较时去除体积加权常数。

64/128/256 表示最细横向分辨率，不是全域均匀 `N³`。壁面一级加密、两格细网格缓冲；几何来自 Aphros 平面切割算法，共享开口面通量，保留极小切割单元。64 为 136,632 个流体单元、1,536 个真实粗细交界面；256 构造过 3,995,168 个流体未知单元和 12,263,704 个共享面。

数据恢复后，不重新求解即可导出已有 64 最终场：

```powershell
python scripts/export_twisted_paraview.py --run D:/CirrusExperiments/cirrus-amg/runs/steady_outer64_v1/iterate_0016 --output D:/CirrusExperiments/cirrus-amg/runs/handoff64_viz_v1
if ($LASTEXITCODE -ne 0) { throw 'ParaView export failed' }
```

打开输出目录的 `solution.vtu`，选择 Cell Data 的 `Speed`、`Velocity`、`Pressure`、`Level`；壁面打开 `walls.vtp`，选择 `WallShearMagnitude` 或 `WallShear`。VTU 不是直接求解输出，清理后需要重新导出；导出还依赖几何，不能仅拷贝 `solution.csv`。

## 9. 给接手同事或新电脑上的代理

可把下面这段连同本文交给新任务：

> 请在这台新电脑上继续 Cirrus-amg 的八叉树扭曲细管精度任务。先读取仓库根目录 HANDOFF_CN.md 和 AGENTS.md，建立一个目标并通过多轮迭代完成：独立 Aphros Proj 128 对齐、原生 Proj 256 定常计算、128→256 近壁空间收敛及必要时间步/采样检查。先检查本机资源和平台，恢复并校验本地输入包，复查已有 64 结果，构建并通过 16 小算例，再顺序启动重任务。保留 HA 八叉树、壁面加密和原生 GPU AMG，不使用 flow map；目前继续 Proj 对 Proj，不把结果记作 SIMPLE 验证。若要改变主算法，先说明具体动机和验证计划。所有场数据、网格、日志、报告和编译快照放到 D:/CirrusExperiments/cirrus-amg，Git 仅提交代码、脚本、小配置和文档。严守现有精度及内存保护条件，失败就保留证据并定位，不能放宽门槛宣称完成。

实现入口：`simple/ProjectionSolver.cpp`、`ProjectionAssembly.h`、`EmbeddedMesh.cpp`、`EmbeddedOperators.cpp`、`NativeCompactGpu.cu`、`NativeAmgPreconditioner.cu`、`AndersonAcceleration.cpp`（均在 `simple/`）。
更多细节见 [重现流程](validation/twisted/REPRODUCE_ON_LARGE_MACHINE.md)、[定常外层加速](validation/twisted/STEADY_OUTER_ANDERSON.md)、[壁面误差](validation/twisted/WALL_CLOSURE_ACCURACY.md)。

提交前：`python scripts/check_git_content.py --staged`；交付前：`python scripts/check_git_content.py --base origin/main`。不要使用 `git add -f` 提交数据，也不要把大文件压缩后放进 Git。
