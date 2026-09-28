# Aphros SIMPLE 窄管流独立基线

实际构建和运行日期：2026-09-06。基线为开源 Aphros，原仓库 `C:\Code\aphros`，提交 **b60ce3da52c19935fa24c778f62f02141eaf7f80**。原仓库干净且未修改；实验放在本目录的 `aphros` detached worktree。`aphros-simple-dumps.patch` 只新增 SIMPLE 调试输出和一个可控的初始扰动模块，未改变数值算法。

跨电脑的 Proj 对照使用本目录的 `sim_base.conf` 作为永久配置。它从只读历史存档复制而来；新运行目录的 `a.conf` 由运行脚本生成在 `D:\CirrusExperiments\cirrus-amg`，其中的 `sim_base.conf` 引用指向本目录。历史存档仅供读取，不作为运行输出目录。

## 复现

依赖为 Windows MSVC 2022 Community、Git for Windows 的 sh/awk、Python 3；不需要 WSL、MPI、HDF5 或 Hypre。MSVC 实测版本 19.44.35221。纯 Windows NMake build。

在新的实验目录中，先复制本目录的脚本和 patch，然后运行：

```powershell
$baselineTree = Join-Path (Get-Location).Path 'aphros'
git -C C:\Code\aphros worktree add --detach $baselineTree b60ce3da52c19935fa24c778f62f02141eaf7f80
git -C .\aphros apply ..\aphros-simple-dumps.patch
.\build_baseline.ps1
.\run_baseline.ps1
```

已有附带 worktree 时，跳过前两个命令。`build_baseline.ps1` 会执行上游 `make/bootstrap`，构建静态库并显式重新链接 `main.exe`。NMake 上游依赖规则不会在库改变后自动重新链接 `main.exe`，所以显式链接不可省略。首次构建必须先产生 `main.obj`：若干净构建尚无此文件，可在开发者命令行执行 `cl /c /O2 /nologo main.c`；构建脚本也会处理这一点。

只运行一个已有配置：

```powershell
Set-Location .\periodic_3d_n8
$env:APHROS_SIMPLE_DUMP='simple'
& ..\aphros\src\main.exe a.conf > run.log 2>&1
```

`make_cases.py` 会按其所在目录重新生成配置中的绝对 include 路径，不能直接拿旧绝对路径配置去另一台机器。

## 算例定义

无量纲一致单位：Lx=1，H=Ly=0.125，rho=1，mu=nu=0.01，单位体积力 fx=1（加速度 gx=1）。x 周期；上下 y 壁面无滑移。2D 网格分别 64×8、128×16、256×32。3D 对齐网格为 **64×8×8**，Lz=0.125，z 周期。

解析稳态解为 `u(y)=y*(H-y)/(2*nu)`，v=w=0，周期压力为任意常数；外加压降等效梯度 `-dp/dx=1`。精确 Umean=0.13020833333333334，Umax=0.1953125，单位深度 Q=0.016276041666666668。3D 实际体积流量还要乘 Lz=0.125。

使用 `fluid_solver=simple, stokes=1, explviscous=0`。这里 `stokes=1` 在 Aphros 同时去掉对流和时间导数，因此是直接稳态 SIMPLE；`dt0=1` 只是外层一次求解的标签。`explviscous=0` 保留隐式 mu*Laplacian，关闭中间非零散度时可能产生额外差异的显式 grad-div 项。不可用本组基线声称对流项已验证。速度松弛 0.7，压力松弛 0.3；压力和动量线性求解器均为上游 `conjugate`，max-norm 容差 1e-14（按体积归一化）；外层变化阈值 1e-12，另独立检查方程残差和通量散度。

`periodic_3d_n8` 从零速度、零压力起步。`perturb_3d_n8` 使用 A=0.01：

```text
u0 = A sin(2 pi x) sin(pi y/H)
v0 = A cos(2 pi x) sin(2 pi y/H)
w0 = 0, p0 = 0
```

该初值含非零散度，真实激活压力修正；压力不固定单点，全 Neumann 兼容系统，跨代码比较时去掉压力常数。

local backend 的 `loc_periodic_*` 必须保持默认 1，让全部 halo 有可读取的数据；物理边界由 `hypre_periodic_y=0` 和 wall BC 独立指定。最初把 loc_periodic_y 设为 0 导致 halo NaN，已通过实际调试确认并修正。配置开启 CHECKNAN，不能仅凭进程退出码判断数值有效。

## 输出和跨代码对齐

设置环境变量 `APHROS_SIMPLE_DUMP` 为文件名前缀。默认输出总迭代编号 0、1、2、9、99，编号从零开始；设置 `APHROS_SIMPLE_DUMP_ALL=1` 可输出所有迭代（数据量较大）。单块 id 为 b0。

- `simple_<iter>_b0_cells.csv`：cellcenter x/y/z，volume，校正后 u/v/w，更新后 p，p_previous，pcorr，pcorr_rhs，pcorr_diag，各方向 diag、delta_rhs、u_star/v_star/w_star。
- `simple_<iter>_b0_faces.csv`：facecenter x/y/z，axis，area，predicted_flux，corrected_flux。
- `simple_final_b0_cells.csv` 与 `simple_final_b0_faces.csv`：最终单元状态和面通量，17 位有效数字。
- `profile.csv`：固定 x/z 的最终 y/u/u_exact。
- `baseline_results.json`：实际误差、流量、残差、散度以及首步压力修正强度。

面通量方向统一为坐标轴正方向，量纲是体积/时间（2D 为每单位深度）。体积、面积是基线的实际维度量，2D dump 不能直接与 3D 的矩阵系数比较。压力修正使用 `A*pcorr=pcorr_rhs`，均为积分形式。动量求解用 delta form：`A_relaxed*delta_u=delta_rhs`。dump 的 diag 已除以 alpha_u、并乘回体积；delta_rhs 为上游表达式常数项取负，不是绝对速度方程的 RHS。若另一代码是 absolute form，应比较 `absolute_rhs - A_relaxed*u_old`。预测速度保存在 `u_star/v_star/w_star`，校正后的速度保存在 `u/v/w`。

3D 零初值算例首步的第一个 cell（x=y=z=0.0078125）：volume=3.814697265625e-6，diag_u=diag_v=diag_w=0.0015625，delta_rhs_u=3.814697265625e-6，u_star=u=0.005497872097883494。

**壁面离散需要一致。** 上游 `src/solver/approx_eb.ipp` 的规则网格 Dirichlet 梯度用二次单边式 `(8*u_wall - 9*u1 + u2)/(3*h)*sgn`；隐式矩阵保持两点式，差值作为 deferred correction 加到 RHS。因此求解后 cellcenter 值可以精确再现二次抛物线。用简单 ghost=-u 的壁面差分去检查这个基线会错误报告 0.25 的源项残差。

**压力的单元梯度和壁面通量是两件事。** 上游 `src/solver/simple.ipp` 的 `UpdateDerivedConditions` 对 wall 上的 pressure/pcorr 使用 `extrap`，不是零 Neumann；相应 `AverageGradient` 在靠壁第一排产生单边导数，例如 y 下壁的 `dp/dy=(p2-p1)/h`。如果另一实现先把壁面压力面梯度设为零再平均，第一排梯度会只有一半，导致横向 cellcenter 速度修正不一致。这个外推用于 cellcenter 压力/压力修正梯度；物理壁面的法向面通量仍严格为零，不能因此给墙面放开通量。该差异已由跨代码逐步 dump 定位：修复前动量矩阵、预测速度、pcorr 和面通量已对齐，但近壁横向速度修正存在差异。

## 实测结果和界限

2D ny=8/16/32，cellcenter 速度相对 L2 误差约 5.60e-11 / 2.19e-10 / 8.79e-10；流量相对误差仍为 0.78125% / 0.1953125% / 0.048828%，按二阶下降。前者是此二次解析解及壁面闭合的特殊性质，不代表一般流动达到这一精度；后者来自面上中点积分，不能用 cellcenter 误差替代流量误差。

3D 零初值和扰动两组均约 403 轮收敛，最终速度相对 L2 约 8.07e-11。扰动组第一步预测通量散度 L-infinity=0.3475434431758995，校正后 9.88e-15；pcorr L-infinity=0.3973441778276647。最终散度约 1.45e-15，横向速度约 1.90e-17。零初值组和扰动组最终达到相同解。

方程残差脚本按最终单向充分发展流及上述二次壁面梯度重新计算 streamwise 方程，平行板 3D约 1.0e-10；这不是任意三维流场的全动量残差检查。该组基线不覆盖入口发展段、弯管、曲面 cut-cell、八叉树粗细界面、惯性对流或湍流。上游此版本 SIMPLE 未将 inletpressure/outletpressure 的值写入其压力边界条件；相关处理可在 Proj 求解器找到，所以不把 SIMPLE 的压力入口配置当作有效基线。

## 四壁方形直管：实际 3D 管道截面验证

新增 `duct_3d_n8` 和 `duct_3d_n16`：Lx=1，Ly=Lz=H=W=0.125，x 周期；y/z 四个侧壁全部 no-slip。rho=1，nu=mu=0.01，gx=1，初值 u=v=w=p=0，SIMPLE 参数与前述相同。网格为 64×8×8 和 128×16×16，分别实际运行 209、749 轮。单独运行：

```powershell
.\run_baseline.ps1 -Cases duct_3d_n8,duct_3d_n16
```

参照矩形管充分发展解，独立在 `analyze.py` 中计算 Fourier 解析级数，未调用 Aphros 求解器或其 Poiseuille 实现。坐标为 y∈[0,H]、z∈[0,W]，加速度为 g：

```text
u(y,z) = g*y*(H-y)/(2*nu)
         - 4*g*H^2/(nu*pi^3) * sum(n odd)
           sin(n*pi*y/H)/n^3
           * cosh[n*pi*(z-W/2)/H]/cosh[n*pi*W/(2H)]

Q = g*W*H^3/(12*nu)
    * [1 - 192*H/(pi^5*W)*sum(n odd) tanh(n*pi*W/(2H))/n^5]
```

该式满足 `nu*(u_yy+u_zz)+g=0`，四壁速度为零。公式与 [MEK4300 讲义中的矩形管解析解](https://mikaem.github.io/MEK4300/content/chapter3/poiseuille.html#noncircular-ducts) 经坐标平移后等价。实现把双曲余弦比写成负指数的比值，避免大模态溢出；速度求和取奇数 n≤255，流量取 n≤4095。

`rectangular_reference_check.json` 记录解析实现自查：方形截面 y/z 交换误差 2.78e-17；三个内部点用四阶差分复核 PDE 的最大残差 1.20e-10；流量截断从 n=2047 提高到4095的变化 8.35e-18。两组基线的 cellcenter 解析值从 n=255 提高到511后，最大变化在 double 精度下为零。

本例精确 Umax=0.11511148950236531，Umean=0.054912896466856945，实际三维体积流量 Q=0.0008580140072946398。请勿沿用平行板的 Umean/Q。

| 网格 | 速度相对 L2 | 实测 Q | Q 相对误差 | streamwise 方程残差 L∞ | 面通量散度 L∞ |
|---|---:|---:|---:|---:|---:|
| 64×8×8 | 0.722868% | 0.000869794235141773 | +1.372965% | 1.05e-10 | 0 |
| 128×16×16 | 0.241974% | 0.0008601370500229116 | +0.247437% | 4.19e-10 | 0 |

方管速度在 y/z 两个方向都有曲率，因此不会像二次平行板解那样只有迭代误差。这两级加密显示误差下降，但不足以单独证明渐近收敛阶。流量误差还混合了截面中点积分误差和单元速度离散误差；解析速度在同一组单元中心做中点积分本身的 Q 误差分别为 +1.84773% 和 +0.462770%。

方管新增 `section.csv`，含固定 x 截面上所有 y/z/u/u_exact，便于检查近壁和角落误差。`baseline_results.json` 的 `section_flux` 是直接求和 x=0.5 截面上面通量；`section_flux_exact` 是独立级数积分 Q。`flow_per_depth` 仍表示 Q/W，以兼容平行板输出，其值不是三维实际 Q。方管残差检查包括 y/z 两方向黏性通量和体力；最终横向速度、压力变化、轴向速度的 x 变化均为零。
## 原始 Proj 面表达式的只读捕获

`prepare_twisted_linear_cache.py --capture-pressure-faces` 在新的输出目录中
复制原始 `proj_eb.cpp`、`proj.h`、`proj.ipp`，只在原压力通量的 `Eval`
循环后加入读取调用。`override_manifest.json` 保存原文件与实际编译文件
的 SHA256。共享的 `libaphros_static.lib` 不重编、不覆盖。

构建时将新目录的 `linear.cpp` 和 `proj_eb.cpp` 作为 `build_twisted_driver.ps1`
的 `ExtraSources`，使用同时复制到该目录的 driver 和诊断头文件。
运行时加 `run_twisted_baseline.ps1 -DumpPressureSystem -DumpPressureFaces`。
每次投影仅保留最后一组原始面表达式；每个完整物理步输出：

- `proj_final_b0_pressure_faces.csv`：面与相邻单元索引、几何、e0/e1/b、两侧压力、原始通量。
- `proj_final_b0_pressure_faces_snapshot.json`：投影次数、时间步和物理时间。

输出前逐面检查捕获的压力与通量是否和最终求解器字段完全一致；不一致
就失败。`check_aphros_capture_pair.py` 检查开关 dump 以及独立编译原
Proj 与原静态库的完整运行是否逐字节一致。16 网格 8 步的两项检查均
通过，完整 1961 次外迭代记录、几何、速度、压力、壁面剪切和通量一致。

`check_aphros_pressure_faces.py` 验证最终表达式可逐位复算，并在内存中
比较压力差形式、Decimal 乘积只舍入一次的结果。这两种重新计算仅用于
定位浮点抵消，不能替代实际参考场或作为独立守恒通过的证据。所有原始
CSV 保持不变。64 网格的完整诊断现已结束；上述替代算术均未通过
原质量门槛。完整独立物理步对比、扩展精度局部试算及原始模板编译
结果见 [压力精度记录](../twisted/PRESSURE_PRECISION.md)。
