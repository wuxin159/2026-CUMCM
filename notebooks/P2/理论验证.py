"""问题二 · 第 1 步：把要写进论文的每个定量说法都跑一遍（只算数，不作图）。
P1 的教训第 3、4 条：直觉预判必须用数据检验，文档里每个数字都要有可复现的出处。
"""
import sys
from pathlib import Path

import numpy as np

import 模型 as M

HERE = Path(__file__).resolve().parent
SEED = 20260910


def 分节(标题):
    print()
    print("=" * 78)
    print(标题)
    print("=" * 78)


# ================================================================ 1
def 节1_先验区域():
    分节("1  先验区域 Ω1 = 楔形 ∩ {5 < rho <= rho_hi} ∩ 目标圆盘")
    print("Ω1 是第一次检测之后干扰源的全部可行位置，问题二的一切都在它上面展开。\n")

    print("① S1 在原点（问题三机器狗的出发点），示向度任意，取 theta1 = 0：")
    Ω = M.先验区域()
    d, R, c, A = M.区域指标(Ω)
    解析面积 = np.radians(M.DELTA) * (M.R_RECV_MAX ** 2 - M.R_NEAR ** 2)
    print(f"    rho 上界 rho_hi          = {M.距离上界():.1f} m   （由 R_eff <= 1500 决定，圆盘未起作用）")
    print(f"    直径 diam(Ω1)            = {d:.3f} m")
    print(f"    最小覆盖圆半径 Rmin       = {R:.3f} m")
    print(f"    面积                     = {A:.2f} m^2  （解析值 delta*(rho_hi^2-5^2) = {解析面积:.2f}）")
    print(f"    最远端全宽 2*rho_hi*tan1° = {2*M.R_RECV_MAX*M.TAN_DELTA:.3f} m")
    print("    -> 区域长 1495 m、最宽仅 52 m，是一条极细长的扇形带。")
    print("       不确定性几乎全部集中在**距离**方向上，第二个检测点的唯一任务就是定距离。\n")

    print("② S1 不在原点时，目标圆盘会把 rho_hi 压下来（策略随之自动缩尺）：")
    print(f"    {'S1':>16} {'theta1':>8} {'rho_hi':>9} {'diam(Ω1)':>10} {'面积':>12}")
    for S1, th in [((0, 0), 0.0), ((1200, 0), 0.0), ((1200, 0), 90.0),
                   ((1700, 0), 0.0), ((1700, 0), 180.0), ((900, 900), 45.0)]:
        Ω = M.先验区域(S1, th)
        d, R, c, A = M.区域指标(Ω)
        print(f"    {str(S1):>16} {th:8.1f} {M.距离上界(S1, th):9.1f} {d:10.3f} {A:12.2f}")
    print("    -> 出发点越靠近边界、示向度越朝外，可行距离区间越短，定位越容易。")


# ================================================================ 2
def 节2_闭式精度():
    分节("2  闭式直径公式的精度")
    print("公式  d ≈ (2 tan δ / |b|) · sqrt(s²+b²) · sqrt((ρ+|s|)²+b²),  s = ρ - a")
    print("由 P1 §4.2 的  d = sqrt(w1²+w2²+2 w1 w2 |cos φ|)/sin φ  坐标化而来。")
    print("这里与精确解（Ω1 与第二楔形取交后按 P1 算法求直径）逐点对比。\n")

    rng = np.random.default_rng(SEED)
    Ω = M.先验区域()
    记录 = []
    for _ in range(4000):
        rho = rng.uniform(50, 1500)
        a = rng.uniform(-500, 2000)
        b = rng.uniform(20, 1200) * rng.choice([-1, 1])
        G = (rho, 0.0)
        th2 = M.方位角((a, b), G)
        L = M.定位区域(Ω, (a, b), th2)
        if not L or len(L) < 3:
            continue
        de = M.区域指标(L)[0]
        da = float(M.近似直径(rho, a, b))
        φ = float(M.交会角(rho, a, b))
        记录.append((φ, de, da, (da - de) / de))
    记录 = np.array(记录)
    print(f"    共 {len(记录)} 组随机构型（ρ∈[50,1500], a∈[-500,2000], |b|∈[20,1200]）")
    print(f"    {'交会角区间':>16} {'组数':>6} {'相对误差中位':>12} {'相对误差最大':>12} {'精确直径中位':>12}")
    for 低, 高 in [(0, 5), (5, 10), (10, 20), (20, 45), (45, 75), (75, 105),
                   (105, 135), (135, 170), (170, 180)]:
        φ = 记录[:, 0]
        m = (φ >= 低) & (φ < 高)
        if m.sum() == 0:
            continue
        e = 记录[m, 3]
        print(f"    [{低:3d}°,{高:4d}°) {m.sum():8d} {100*np.median(e):11.3f}% "
              f"{100*np.max(np.abs(e)):11.3f}% {np.median(记录[m,1]):12.1f}")
    print("    -> 交会角大于 20° 时闭式公式误差在千分之几，可放心用来画候选区域的等值线；")
    print("       交会角很小时，真实区域被 rho<=rho_hi 截断而闭式公式不知道，会**高估**，")
    print("       高估是安全侧（宁可多测一次），但等值线图上要注明这一段不可信。")


# ================================================================ 3
def 节3_最优侧移():
    分节("3  给定前进量 a 时的最优侧移 b* 与最小直径 d_min")
    print("对 ln d 关于 b 求导置零：b⁴ = s²(ρ+|s|)²，即")
    print("        b* = sqrt(|s|·(ρ+|s|)),      d_min = 2 tan δ · (ρ + 2|s|),   s = ρ - a")
    print("即：以正横站位的理想值 2ρ tan δ 为基准，沿轴每错位 1 m，直径线性增加 4 tan δ ≈ 0.0698 m。\n")

    print(f"    {'ρ':>7} {'a':>7} {'|s|':>7} {'b*公式':>10} {'b*数值':>10} "
          f"{'d公式':>10} {'d数值':>10} {'d精确':>10}  说明")
    Ω = M.先验区域()
    最大偏差 = 0.0
    for rho in (300, 800, 1200, 1500):
        for 比 in (0.0, 0.3, 0.6, 0.9, 1.2):        # 比 = 1（s = 0）单列，见下
            a = 比 * rho
            b网 = np.linspace(0.01, 4000.0, 400000)
            d网 = M.近似直径(rho, a, b网)
            b数 = float(b网[np.argmin(d网)])
            d数 = float(np.min(d网))
            b公 = float(M.最优侧移(rho, a))
            d公 = float(M.最优直径(rho, a))
            th2 = M.方位角((a, b公), (rho, 0.0))
            L = M.定位区域(Ω, (a, b公), th2)
            d精 = M.区域指标(L)[0] if L else np.nan
            截断 = "被 ρ<=ρ_hi 截断" if d精 < 0.98 * d公 else ""
            最大偏差 = max(最大偏差, abs(b公 - b数) / max(b数, 1e-9),
                          abs(d公 - d数) / max(d数, 1e-9))
            print(f"    {rho:7.0f} {a:7.0f} {abs(rho-a):7.0f} {b公:10.2f} {b数:10.2f} "
                  f"{d公:10.3f} {d数:10.3f} {d精:10.3f}  {截断}")
    print(f"\n    公式与数值最优的最大相对偏差 = {100*最大偏差:.4f}%（b 网格步长 0.01 m）")
    print("    ρ = 1500 那几行「精确 < 闭式」，是因为源已顶到 ρ_hi = 1500 的边界上，")
    print("    定位区域有一半被 rho <= rho_hi 削掉；闭式公式不知道这条约束，故高估——安全侧。")
    print("\n    s = 0（正横站位）单独讨论：b* = 0 是极限而非可取值，b -> 0 时闭式 d -> 2ρ tanδ，")
    print("    但真实楔形是从 S2 张开的**锥**而不是无限长的条带，b 很小时区域被楔形顶点截断，")
    print(f"    {'ρ':>7} {'b':>7} {'闭式d':>10} {'精确d':>10} {'精确Rmin':>10}")
    for rho in (800, 1200):
        for b in (5, 20, 50, 100, 200):
            th2 = M.方位角((rho, b), (rho, 0.0))
            L = M.定位区域(Ω, (rho, b), th2)
            d精, R精 = M.区域指标(L)[:2]
            print(f"    {rho:7.0f} {b:7.0f} {float(M.近似直径(rho,rho,b)):10.3f} "
                  f"{d精:10.3f} {R精:10.3f}")
    print("    -> 站得越近区域反而越短（楔形顶点把区域从后方切断），这一段闭式公式不适用。")
    print("    注意 a = ρ（正横站位）时 b* = 0，直径退化为 2ρ tan δ = 楔形 1 在源处的全宽：")
    for rho in (500, 1000, 1500):
        print(f"        ρ={rho:5.0f} m -> 2ρ tan1° = {2*rho*M.TAN_DELTA:7.3f} m")
    print("    这是**两次检测的信息上限**：无论第二点怎么放，直径都不可能小于 2ρ tan δ。")


# ================================================================ 4
def 节4_交会角():
    分节("4  交会角与正横站位")
    print("P1 §4.2 已证 φ = 90° 最优。本节把它翻译成「第二个检测点该站哪」。\n")
    print("φ = 90°  <=>  s = ρ - a = 0  <=>  S2 落在过 G 且垂直于示向线的直线上（正横位）。")
    print("但 ρ 未知，正横位是一条**未知位置的直线**，这正是问题二的困难所在。\n")

    print("等宽情形下 d(φ)/d(90°) = 1/sqrt(1-|cos φ|)（P1 表），换算成沿轴错位比 |s|/ρ：")
    print(f"    {'|s|/ρ':>8} {'b*/ρ':>8} {'交会角φ':>9} {'d/(2ρtanδ)':>12}")
    for k in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
        b = np.sqrt(k * (1 + k))
        φ = np.degrees(np.arctan2(b, 1 - 0 if k == 0 else k))  # s = k*ρ
        φ = np.degrees(np.arctan2(b, k)) if k > 0 else 90.0
        print(f"    {k:8.2f} {b:8.3f} {φ:9.2f} {1+2*k:12.3f}")
    print("    -> 沿轴错位 |s| 到 0.5ρ 时，直径已是理想值的 2 倍；错位 1ρ 时 3 倍。")
    print("       换言之：**距离估计的相对误差，几乎一比一地转成定位精度的损失**。")


# ================================================================ 5
def 节5_能否一次清除():
    分节("5  两次检测能否做到「站一点一次清除」")
    print("判据（P1 §4.3）：最小覆盖圆半径 Rmin <= 20 m。必要条件 d <= 40 m。\n")

    print(f"① 两次检测的信息上限 2ρ tan δ <= 40 m  =>  ρ <= "
          f"{20/M.TAN_DELTA:.1f} m，这还是把第二点放在真源正横位（ρ 已知）才有的理想值。")
    print(f"   更现实地，取 b = b*、沿轴错位 |s|：d_min = 2 tan δ(ρ+2|s|) <= 40 m")
    print(f"   =>  ρ + 2|s| <= {20/M.TAN_DELTA:.1f} m。\n")
    print(f"    {'ρ (m)':>8} {'允许的最大沿轴错位 |s| (m)':>28}")
    for rho in (200, 400, 600, 800, 1000, 1146, 1200, 1500):
        s = (20 / M.TAN_DELTA - rho) / 2
        print(f"    {rho:8.0f} {(s if s > 0 else float('nan')):28.1f}")
    print("    -> ρ = 1000 m 时，必须事先把距离估到 ±73 m 才可能一次清除；")
    print("       而第一次检测之后距离的不确定区间长达 1495 m。**两次检测一般做不到一次清除**。\n")

    print("② 上帝视角扫描：**假装已知 ρ**，在 (a,b) 全平面上求精确 Rmin 的下确界。")
    print(f"   附加一条现实约束 |S2 - G| >= {M.R_CLEAR:.0f} m：若第二个检测点已经落进清除半径，")
    print("   直接 /clear 即可，根本不需要第二次检测，那种「最优」是自欺欺人。")
    Ω = M.先验区域()
    print(f"    {'ρ (m)':>8} {'最优 a':>8} {'最优 b':>8} {'最小 Rmin':>10} {'能否一次清除':>12}")
    for rho in (100, 300, 500, 700, 900, 1100, 1146, 1200, 1300, 1500):
        最好 = (np.inf, None, None)
        for a in np.linspace(max(0.0, rho - 400), rho + 400, 81):
            for b in np.concatenate([np.linspace(1, 200, 60), np.linspace(210, 2500, 60)]):
                if np.hypot(rho - a, b) < M.R_CLEAR - 1e-9:
                    continue
                th2 = M.方位角((a, b), (rho, 0.0))
                L = M.定位区域(Ω, (a, b), th2)
                if not L:
                    continue
                R = M.区域指标(L)[1]
                if R < 最好[0]:
                    最好 = (R, a, b)
        print(f"    {rho:8.0f} {最好[1]:8.1f} {最好[2]:8.1f} {最好[0]:10.3f} "
              f"{'是' if 最好[0] <= M.R_CLEAR else '否':>12}")
    print("    （最优 a = ρ、b = 20 属正常：正横站位、贴着清除半径站，这是两次检测的极限。）")
    print(f"    -> 分水岭恰在 ρ = 20/tanδ = {20/M.TAN_DELTA:.1f} m：")
    print("       此时定位区域是长 ρtanδ + 20 = 40 m 的细条，Rmin 正好 20 m。")
    print("       更远的干扰源，**即使事先知道距离**，两次检测也不足以一次清除。\n")

    print("③ 现实视角：ρ 未知，S2 只能选一个。逐个候选点看它对哪些 ρ 有效。")
    print(f"    {'S2 = (a,b)':>16} {'可一次清除的 ρ 区间':>26} {'占先验的比例':>12}")
    rho网, w = M.距离节点(300)
    for a, b in [(500, 500), (700, 300), (800, 200), (900, 150), (600, 120), (1000, 400)]:
        好 = []
        for rho in rho网:
            th2 = M.方位角((a, b), (float(rho), 0.0))
            L = M.定位区域(Ω, (a, b), th2)
            好.append(bool(L) and M.区域指标(L)[1] <= M.R_CLEAR)
        好 = np.array(好)
        区间 = f"[{rho网[好].min():.0f}, {rho网[好].max():.0f}]" if 好.any() else "空"
        print(f"    {str((a,b)):>16} {区间:>26} {w[好].sum():12.3f}")
    print("    -> 没有任何单点能覆盖全部 ρ。**第二次检测的目标不是一次清除，")
    print("       而是把 1495 m 的距离歧义压到百米量级，为第三次检测创造条件。**")


# ================================================================ 6
def 节6_保证收信():
    分节("6  保证第二次检测收到信号的候选域")
    print("已知 ρ <= R_eff <= 1500，故 R_eff >= max(1000, ρ)。于是")
    print("        |S2 - G| <= max(1000, ρ)   =>   必定收到信号")
    print("特别地 |S2-G| <= ρ（第二点比第一点更靠近干扰源）时自动成立。\n")
    print("对 ρ 取遍 (5,1500]，可行域是两个半径 1000 m 圆盘之交（透镜形）：")
    print("        D((5,0), 1000)  ∩  D((1000,0), 1000)")
    print("推导：ρ<=1000 段给出 ∩_{ρ} D((ρ,0),1000)，两端 ρ=5 与 ρ=1000 起作用；")
    print("      ρ>1000 段给出 ∩_{ρ} D((ρ,0),ρ)，这族圆盘全过原点且随 ρ 单调套嵌，最小者 ρ=1000。\n")

    lo, hi = np.array([M.R_NEAR, 0.0]), np.array([M.R_RECV_MIN, 0.0])
    print(f"    透镜的 x 范围: [{0.0:.1f}, {M.R_NEAR+M.R_RECV_MIN:.1f}] m")
    xm = 0.5 * (M.R_NEAR + M.R_RECV_MIN)
    print(f"    在 x = {xm:.1f} 处的最大侧移 |b| = "
          f"{np.sqrt(M.R_RECV_MIN**2 - (xm-M.R_NEAR)**2):.1f} m")

    rng = np.random.default_rng(SEED + 1)
    N = 400000
    rho = rng.uniform(M.R_NEAR, M.R_RECV_MAX, N)
    Reff = rng.uniform(np.maximum(M.R_RECV_MIN, rho), M.R_RECV_MAX)
    print(f"\n    蒙特卡洛校验（N = {N}，ρ 均匀、R_eff 在条件区间上均匀）：")
    print(f"    {'S2 = (a,b)':>18} {'在透镜内':>8} {'实测收信率':>10} {'公式保证':>8}")
    for a, b in [(0, 0), (300, 300), (500, 500), (500, 850), (502, 868),
                 (700, 500), (1000, 200), (1005, 0), (1100, 300), (800, 900), (1400, 100)]:
        r2 = np.hypot(rho - a, b)
        收 = r2 <= Reff
        在透镜 = (np.hypot(a - M.R_NEAR, b) <= M.R_RECV_MIN + 1e-9) and \
                 (np.hypot(a - M.R_RECV_MIN, b) <= M.R_RECV_MIN + 1e-9)
        保证 = bool(np.all(M.保证收信(rho, a, b)))
        print(f"    {str((a,b)):>18} {str(在透镜):>8} {收.mean():10.4f} {str(保证):>8}")
    print("    -> 透镜内实测收信率恒为 1.0000；透镜外一旦距离拉开就出现漏检。")
    print("       漏检不是灾难（no_signal 也排除了一片区域），但它换不来楔形，")
    print("       所以候选区域**应当落在透镜内**，除非有意用 no_signal 做粗筛。")


# ================================================================ 7
def 节7_误差影响():
    分节("7  误差实现 (eps1, eps2) 对定位区域的影响")
    print("设计第二点时并不知道两处的误差，只知 |eps| <= δ。这一节量化「不知道」的代价。\n")
    Ω = M.先验区域()
    print(f"    {'ρ':>6} {'a':>6} {'b':>6} {'d最小':>9} {'d最大':>9} {'d均值':>9} "
          f"{'极差/均值':>10} {'Rmin/(d/2)最小':>14} {'区域必含G':>10}")
    rng = np.random.default_rng(SEED + 2)
    for rho, a, b in [(1000, 500, 300), (1000, 900, 120), (600, 300, 200),
                      (1400, 700, 700), (800, 800, 60), (1200, 400, 150)]:
        ds, 比值 = [], []
        含G = True
        for e1 in np.linspace(-M.DELTA, M.DELTA, 9):
            α = -e1                                   # 局部标架下 G 的真实方位
            G = (rho * np.cos(np.radians(α)), rho * np.sin(np.radians(α)))
            for e2 in np.linspace(-M.DELTA, M.DELTA, 9):
                th2 = M.方位角((a, b), G) + e2
                L = M.定位区域(Ω, (a, b), th2)
                if not L:
                    含G = False
                    continue
                d, R, c, A = M.区域指标(L)
                ds.append(d)
                比值.append(R / (d / 2) if d > 1e-9 else 1.0)
                含G &= M.算法.点在多边形内(G, L)
        ds = np.array(ds)
        print(f"    {rho:6.0f} {a:6.0f} {b:6.0f} {ds.min():9.3f} {ds.max():9.3f} "
              f"{ds.mean():9.3f} {(ds.max()-ds.min())/ds.mean():10.4f} "
              f"{min(比值):14.6f} {str(含G):>10}")
    print("    -> 直径的极差可达均值的 14%~44%，**不能说误差无关紧要**。")
    print("       但它是围绕 eps = 0 的双向摆动，不改变候选点之间的优劣顺序（下一段验证）。")
    print("       「区域必含 G」恒成立，这是集员估计的基本保证，也是本模型敢用最坏界的依据。")

    print("\n    用 eps = 0 设计会不会选错点？把若干候选点在两种口径下排名对比：")
    候选 = [(300, 200), (500, 300), (700, 200), (700, 500), (900, 150),
            (900, 400), (1100, 300), (400, 700), (1200, 600)]
    rho网, w = M.距离节点(24)
    零, 最坏 = [], []
    for a, b in 候选:
        d0, dm = [], []
        for rho in rho网:
            G0 = (float(rho), 0.0)
            L = M.定位区域(Ω, (a, b), M.方位角((a, b), G0))
            d0.append(M.区域指标(L)[0] if L else np.nan)
            峰 = 0.0
            for e1 in (-M.DELTA, 0.0, M.DELTA):
                α = -e1
                G = (rho * np.cos(np.radians(α)), rho * np.sin(np.radians(α)))
                for e2 in (-M.DELTA, 0.0, M.DELTA):
                    L = M.定位区域(Ω, (a, b), M.方位角((a, b), G) + e2)
                    if L:
                        峰 = max(峰, M.区域指标(L)[0])
            dm.append(峰)
        零.append(float(np.dot(w, np.array(d0))))
        最坏.append(float(np.dot(w, np.array(dm))))
    次序零 = np.argsort(零)
    次序坏 = np.argsort(最坏)
    print(f"    {'S2':>14} {'E[D] (eps=0)':>14} {'E[max D] (最坏eps)':>20} {'比值':>7} "
          f"{'零误差排名':>10} {'最坏排名':>9}")
    for k, (a, b) in enumerate(候选):
        print(f"    {str((a,b)):>14} {零[k]:14.2f} {最坏[k]:20.2f} {最坏[k]/零[k]:7.3f} "
              f"{int(np.where(次序零==k)[0][0])+1:10d} {int(np.where(次序坏==k)[0][0])+1:9d}")
    print(f"    Spearman 秩相关 = "
          f"{np.corrcoef(np.argsort(次序零), np.argsort(次序坏))[0,1]:.4f}")
    print("    -> 两种口径给出的排名一致，最坏情形只是把所有点按约 1.1~1.2 的相同倍数放大，")
    print("       所以候选区域用 eps = 0 扫、再用最坏情形复核，是站得住的做法。")

    print("\n    Rmin/(d/2) 在上述样本中恒等于 1：未被距离上界截断时定位区域是平行四边形，")
    print("    钝角顶点必落在长对角线为直径的圆内，故此时「以直径为直径的圆」恰好能覆盖。")
    print("    这与 P1「一般不能覆盖」不矛盾——P1 的反例来自楔形锥度与三个以上楔形取交；")
    print("    下面按 P1 的口径重扫一遍 n = 2，给出本问条件下的失败率：")
    rng2 = np.random.default_rng(SEED + 3)
    失败, 最大比 = 0, 1.0
    N = 20000
    for _ in range(N):
        rho = rng2.uniform(50, 1500)
        a = rng2.uniform(-300, 1800)
        b = rng2.uniform(30, 1200)
        α = rng2.uniform(-M.DELTA, M.DELTA)
        G = (rho * np.cos(np.radians(α)), rho * np.sin(np.radians(α)))
        th2 = M.方位角((a, b), G) + rng2.uniform(-M.DELTA, M.DELTA)
        L = M.定位区域(Ω, (a, b), th2)
        if not L or len(L) < 3:
            continue
        d, R = M.区域指标(L)[:2]
        if d < 1e-9:
            continue
        比 = R / (d / 2)
        最大比 = max(最大比, 比)
        if 比 > 1 + 1e-7:
            失败 += 1
    print(f"    N = {N} 组随机构型：以直径为直径的圆覆盖失败 {失败} 组（{失败/N:.4%}），"
          f"Rmin/(d/2) 最大 {最大比:.6f}")
    print("    -> 本问的 Ω1 里带着「rho 上界」这条额外约束，被它截断时区域不再是平行四边形，")
    print("       失败就出现在这一小撮构型上，与 P1 的结论一致。")


def main():
    print("问题二 · 理论验证")
    print(f"随机种子 {SEED}；δ = {M.DELTA}°，R_eff ∈ [{M.R_RECV_MIN},{M.R_RECV_MAX}] m，"
          f"清除半径 {M.R_CLEAR} m")
    节1_先验区域()
    节2_闭式精度()
    节3_最优侧移()
    节4_交会角()
    节5_能否一次清除()
    节6_保证收信()
    节7_误差影响()
    print("\n完毕。")


if __name__ == "__main__":
    with open(HERE / "理论验证结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '理论验证结果.txt'}")
