"""问题三 · 第 1 步：判空所需的圆覆盖问题（只算数，不作图）。

**这一节回答的是「什么时候可以停」。**

题目说干扰源总数在 10~16 之间但具体个数未知，所以「清完了」这件事不能靠计数得到，
必须对每个没清除的频道给出「它确实没有干扰源」的依据。唯一可用的依据是附录 2 第 2 条：

    有效接收半径 R_eff ∈ [1000, 1500]，因此
    在 p 处检测频道 c 返回 no_signal  =>  该频道的干扰源（若有）满足 |p - G| > R_eff >= 1000

即**一次 no_signal 排除一个半径 1000 m 的圆盘**。于是

    频道 c 判空  <=>  c 的各无信号点的 1000 m 圆盘之并 ⊇ 半径 1800 m 的目标圆域

这就把「何时可以退出」化成一个**圆覆盖问题**：用半径 r = 1000 的圆盘覆盖半径 R = 1800
的圆域，最少要几个？本脚本给出四件事：

    一、覆盖数 K* = 7，且 **6 个只差 0.6 米**——本题一处极窄的临界，必须算而不能猜
    二、K = 7 的显式构造（1 个圆心 + 6 个半径 1558.8 m 的环点，半径 900 m 即可）
    三、K = 7 时使巡回路程最短的锚点布局（供「先扫后清」基线与残余补测使用）
    四、由此得到的**判空代价下界**，是后面理论下界的一个组成部分

需要说清楚的一点：**策略的正确性并不依赖 K* = 7 这个常数**。算法在运行时直接检查
栅格残余掩码是否为空（`公共.栅格`，保守离散化），掩码空才允许退出。K* 只用于
代价核算与下界估计。即便 K* 的文献值有出入，策略仍然是完备的。

输出：覆盖锚点结果.txt、覆盖数_最优半径.csv、锚点布局.csv、覆盖残余_六点.csv、
      七点构造.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 公共 as C

HERE = Path(__file__).resolve().parent
SEED = 20260911

# 文献值：n 个等圆覆盖单位圆所需的最小半径比（Neville 1915；n=7 的 1/2 由正六边形构型给出）。
# 本脚本独立重算一遍作为交叉验证——不是拿来当结论用，是拿来检查自己的优化器。
文献最优半径比 = {1: 1.0, 2: 1.0, 3: np.sqrt(3) / 2, 4: np.sqrt(2) / 2,
                  5: 0.609383, 6: 0.555905, 7: 0.5}


# ================================================================ 采样与评价
def 圆域采样(R=C.R_AREA, 步长=8.0, 边界点数=2000):
    """圆域内的方格点 + 边界密采样。覆盖的最劣点总在边界或「深点」上，故边界要密。"""
    k = int(np.ceil(R / 步长))
    轴 = np.arange(-k, k + 1) * 步长
    X, Y = np.meshgrid(轴, 轴)
    内 = np.hypot(X, Y) <= R
    点 = np.stack([X[内], Y[内]], axis=1)
    t = np.linspace(0, 2 * np.pi, 边界点数, endpoint=False)
    边 = np.stack([R * np.cos(t), R * np.sin(t)], axis=1)
    return np.vstack([点, 边])


def 覆盖半径(中心, 采样):
    """给定中心集，覆盖采样集所需的最小公共半径 = max_点 min_中心 距离。"""
    c = np.asarray(中心, dtype=float).reshape(-1, 2)
    d = np.sqrt(((采样[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).max())


def _切比雪夫迭代(中心, 采样, 最大轮=80):
    """k-center 的 Lloyd 型迭代：分配 -> 每簇取最小覆盖圆圆心 -> 再分配。

    两步都不会让目标 max_点 min_i |点-中心_i| 变大：
      · 重新分配只会把点分给更近的中心；
      · 把中心移到该簇的最小覆盖圆圆心，按定义使该簇内的最大距离最小。
    所以迭代单调不增、必然收敛。它天生适配这个 min-max 目标；通用无导数优化器
    （Powell/Nelder-Mead）在这种非光滑目标上实测慢两个量级且更容易卡住。
    """
    c = np.asarray(中心, dtype=float).reshape(-1, 2).copy()
    上次 = np.inf
    for _ in range(最大轮):
        d = np.sqrt(((采样[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))
        归属 = d.argmin(axis=1)
        目标 = float(d.min(axis=1).max())
        for i in range(len(c)):
            簇 = 采样[归属 == i]
            if len(簇) == 0:                    # 空簇：挪到当前最劣点，别浪费一个中心
                c[i] = 采样[d.min(axis=1).argmax()]
                continue
            c[i] = C.最小覆盖圆_点集(簇)[0]
        if 上次 - 目标 < 1e-5:
            break
        上次 = 目标
    return 覆盖半径(c, 采样), c


def 求最优覆盖(n, 粗采样, 细采样, rng, 粗重启=60, 细候选=4, 扰动=12, 扰动幅=180.0):
    """两级搜索：粗采样上多起点 Lloyd 找盆地，细采样上对最好的几个做扰动再收敛。

    返回 (细采样上的覆盖半径, 中心集)。注意这是**上界**：采样是离散的，
    真实的最劣点可能落在采样点之间，故真实值不小于返回值（相差不超过步长/sqrt2）。
    """
    候选 = []
    起点集 = []
    for 环 in np.linspace(0.25, 0.95, 8) * C.R_AREA:
        for 相位 in (0.0, np.pi / max(n - 1, 1)):
            t = np.linspace(0, 2 * np.pi, n - 1, endpoint=False) + 相位
            起点集.append(np.vstack([[0.0, 0.0],
                                     np.stack([环 * np.cos(t), 环 * np.sin(t)], axis=1)]))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        起点集.append(np.stack([环 * np.cos(t), 环 * np.sin(t)], axis=1))
    while len(起点集) < 粗重启:
        起点集.append(rng.uniform(-C.R_AREA, C.R_AREA, size=(n, 2)))
    for x0 in 起点集:
        候选.append(_切比雪夫迭代(x0, 粗采样))
    候选.sort(key=lambda z: z[0])

    最好 = (np.inf, None)
    for _, c0 in 候选[:细候选]:
        r, c = _切比雪夫迭代(c0, 细采样)
        if r < 最好[0]:
            最好 = (r, c)
        for _ in range(扰动):                   # 盆地跳跃：小扰动后重收敛
            r2, c2 = _切比雪夫迭代(
                最好[1] + rng.normal(0, 扰动幅, size=(n, 2)), 细采样)
            if r2 < 最好[0]:
                最好 = (r2, c2)
    return 最好


# ================================================================ 显式七点构造
def 七点构造(R=C.R_AREA):
    """1 个圆心 + 6 个半径 sqrt(3)R/2 的环点，等圆半径 R/2 即可覆盖半径 R 的圆域。

    验证（初等）：设环点在半径 sqrt(3)R/2、相邻夹角 60°。
      · 边界点 R∠30°（相邻两环点的角平分方向，最劣）到环点 (sqrt(3)R/2)∠0° 的距离
        = sqrt(R² + 3R²/4 - 2·R·(sqrt(3)R/2)·cos30°) = sqrt(R² + 0.75R² - 1.5R²) = R/2。
      · 半径 R/2 的中心圆盖住 |x| <= R/2 的部分。
      · 其余部分逐点验证（见主流程的数值检查）。
    对本题 R = 1800：环半径 1558.85 m，等圆半径 900 m <= 可用的 1000 m，**有富余**。
    """
    t = np.arange(6) * np.pi / 3.0
    环 = np.sqrt(3.0) * R / 2.0
    return np.vstack([[0.0, 0.0], np.stack([环 * np.cos(t), 环 * np.sin(t)], axis=1)])


# ================================================================ 最短覆盖巡回
def 巡回长度(中心, 起点=(0.0, 0.0)):
    点 = [tuple(p) for p in np.asarray(中心, dtype=float).reshape(-1, 2)]
    顺序, L = C.开放TSP(起点, 点)
    return L, 顺序


def 结构族布局(K, 环半径, 相位, 含圆心):
    m = K - 1 if 含圆心 else K
    t = np.linspace(0, 2 * np.pi, m, endpoint=False) + 相位
    环 = np.stack([环半径 * np.cos(t), 环半径 * np.sin(t)], axis=1)
    return np.vstack([[0.0, 0.0], 环]) if 含圆心 else 环


def 求最短覆盖巡回(K, 采样, r=C.R_RECV_MIN):
    """在「半径 r 的圆盘之并覆盖目标圆域」的约束下，最小化从原点出发的开放巡回长度。

    先在「环 + 可选圆心」的结构族上扫参数（环半径 x 相位 x 是否含圆心），
    再对最好的做一次坐标下降式局部微调。之所以先用结构族而不是直接放开
    2K 个自由变量：目标（巡回长度）和约束（覆盖）都不光滑，自由优化极易停在
    既不覆盖也不短的地方；结构族保证每个候选都可行，微调只做改良。
    """
    最好 = (np.inf, None)
    for 含圆心 in (True, False):
        for 环半径 in np.arange(700.0, 1800.1, 12.5):
            for 相位 in np.linspace(0, 2 * np.pi / max(K - 1, 1), 9, endpoint=False):
                c = 结构族布局(K, 环半径, 相位, 含圆心)
                if 覆盖半径(c, 采样) > r:
                    continue
                L, _ = 巡回长度(c)
                if L < 最好[0]:
                    最好 = (L, c)
    if 最好[1] is None:
        return 最好
    # 坐标下降微调：逐点在小邻域内试探，只接受「仍覆盖且更短」的移动
    c = 最好[1].copy()
    L = 最好[0]
    步 = 120.0
    while 步 > 2.0:
        改进 = False
        for i in range(len(c)):
            for d in ((步, 0), (-步, 0), (0, 步), (0, -步),
                      (步, 步), (步, -步), (-步, 步), (-步, -步)):
                试 = c.copy()
                试[i] = 试[i] + d
                if 覆盖半径(试, 采样) > r:
                    continue
                L2, _ = 巡回长度(试)
                if L2 < L - 1e-6:
                    c, L, 改进 = 试, L2, True
        if not 改进:
            步 /= 2.0
    return L, c


# ================================================================ 六点残余
def 未覆盖点(中心, R=C.R_AREA, r=C.R_RECV_MIN, 步长=1.0):
    """列出目标圆域内到所有中心距离都大于 r 的方格中心。

    这些点是**严格的**未覆盖见证：不需要任何离散化裕度，一个点就足以否定该布局。
    """
    k = int(np.ceil(R / 步长))
    轴 = np.arange(-k, k + 1) * 步长
    X, Y = np.meshgrid(轴, 轴)
    内 = np.hypot(X, Y) <= R
    点 = np.stack([X[内], Y[内]], axis=1)
    c = np.asarray(中心, dtype=float).reshape(-1, 2)
    d = np.sqrt(((点[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    留 = d > r
    return 点[留], d[留], float(步长 *步长 * np.count_nonzero(留))


# ================================================================ 主流程
def main():
    rng = np.random.default_rng(SEED)
    粗 = 圆域采样(步长=25.0, 边界点数=1200)
    细 = 圆域采样(步长=8.0, 边界点数=3000)
    print("问题三 · 判空所需的圆覆盖问题")
    print(f"随机种子 {SEED}")
    print(f"粗采样 {len(粗)} 点（25 m 方格 + 边界 1200 点），"
          f"细采样 {len(细)} 点（8 m 方格 + 边界 3000 点）")
    print(f"目标圆域半径 R = {C.R_AREA:.0f} m；单次 no_signal 的排除半径 "
          f"r = R_eff 的下界 = {C.R_RECV_MIN:.0f} m\n")

    print("=" * 90)
    print("一、用半径 1000 m 的圆盘覆盖半径 1800 m 的圆域，最少要几个？")
    print("=" * 90)
    print("    先用文献值校准优化器：n 个等圆覆盖单位圆的最小半径比 r_n/R 是已知的")
    print("    （Neville 1915 求出 n<=6，n=7 的 1/2 由正六边形构型给出）。")
    print("    本脚本用 k-center 的切比雪夫迭代独立重算，两者吻合才说明结论可信。\n")
    print(f"    {'n':>3} {'本脚本 r_n/R':>14} {'文献 r_n/R':>12} {'相对偏差':>10} "
          f"{'折合 R=1800 (m)':>16} {'r=1000 够不够':>15}")
    行, 布局 = [], {}
    for n in range(3, 9):
        r_n, 中心 = 求最优覆盖(n, 粗, 细, rng)
        布局[n] = 中心
        比 = r_n / C.R_AREA
        文献 = 文献最优半径比.get(n, np.nan)
        够 = "够" if r_n <= C.R_RECV_MIN else "不够"
        print(f"    {n:>3} {比:>14.6f} {文献:>12.6f} "
              f"{((比 - 文献) / 文献 if np.isfinite(文献) else np.nan):>10.2e} "
              f"{r_n:>16.2f} {够:>15}")
        行.append({"锚点数n": n, "本脚本最优半径比": 比, "文献最优半径比": 文献,
                   "最优覆盖半径_米": r_n, "r1000是否够": 够})
    pd.DataFrame(行).to_csv(HERE / "覆盖数_最优半径.csv", index=False,
                            encoding="utf-8-sig")

    r6 = float([x["最优覆盖半径_米"] for x in 行 if x["锚点数n"] == 6][0])
    r6文 = 文献最优半径比[6] * C.R_AREA
    print(f"\n    ★ 结论：K* = 7。")
    print(f"      6 个锚点按文献最优构型需要半径 {r6文:.2f} m，本脚本数值搜索得 {r6:.2f} m，")
    print(f"      两者都**超过**可用的 1000 m——按文献值只超 {r6文 - C.R_RECV_MIN:.2f} 米。")
    print(f"      折算成半径比是 {文献最优半径比[6]:.6f} 对 "
          f"{C.R_RECV_MIN / C.R_AREA:.6f}，相对差仅 "
          f"{(r6文 / C.R_RECV_MIN - 1) * 100:.3f}%。")
    print("      这是本题一处极窄的临界：若题目把接收半径下界写成 1001 m，")
    print("      判空锚点数就从 7 掉到 6，全局代价随之下降一档。差得这么少，")
    print("      靠直觉是判不出来的，必须算。")
    print("\n    严谨性说明（论文里要照实写）：")
    print("      · 「6 个不够」依赖 r_6/R = 0.555905 这个**文献常数**——数值搜索只能给出界，")
    print("        证明不了下界。本脚本的作用是独立复现该常数，不是重新证明它。")
    print("      · 自洽的初等下界是 K* >= 5：目标圆周上按 72° 取 5 个点，相邻弦长")
    print(f"        2R sin36° = {2 * C.R_AREA * np.sin(np.radians(36)):.0f} m > "
          f"2r = 2000 m，故任一半径 1000 m 的圆盘至多含其中一个点。")
    print("      · **策略的完备性不依赖 K***：算法在运行时直接检查栅格残余掩码是否为空")
    print("        （保守离散化，见 公共.栅格），空才允许退出。K* 只进入代价核算。")

    print("\n" + "=" * 90)
    print("二、K = 7 的显式构造：1 个圆心 + 6 个环点")
    print("=" * 90)
    七 = 七点构造()
    r7 = 覆盖半径(七, 细)
    坏, _, _ = 未覆盖点(七, r=C.R_AREA / 2 + 1e-6, 步长=2.0)
    print(f"    环半径 sqrt(3)R/2 = {np.sqrt(3) * C.R_AREA / 2:.2f} m，理论等圆半径 R/2 = "
          f"{C.R_AREA / 2:.0f} m")
    print(f"    细采样实测最大最近距离 {r7:.2f} m（理论 900 m，"
          f"差 {abs(r7 - C.R_AREA / 2):.2f} m 来自采样离散）")
    print(f"    以 900 m 为半径逐点检查（2 m 方格）：未覆盖方格 {len(坏)} 个")
    print(f"    可用半径是 1000 m，比需要的 900 m 富余 {C.R_RECV_MIN - C.R_AREA / 2:.0f} m，")
    print("    这份富余正是后面「锚点可以往回缩、把巡回缩短」的空间来源。")
    pd.DataFrame({"序号": range(1, 8), "x": 七[:, 0], "y": 七[:, 1],
                  "距原点": np.hypot(七[:, 0], 七[:, 1])}).to_csv(
        HERE / "七点构造.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("三、六点布局究竟漏在哪里、漏了多少")
    print("=" * 90)
    残点, 残深, 残面积 = 未覆盖点(布局[6])
    print(f"    取本脚本找到的最优六点布局，用半径 1000 m 的圆盘去覆盖：")
    print(f"    未覆盖面积 {残面积:.0f} m²，占目标圆域 "
          f"{残面积 / (np.pi * C.R_AREA ** 2) * 100:.4f}%")
    if len(残点):
        print(f"    最深处距最近锚点 {残深.max():.2f} m，共 {len(残点)} 个 1 m² 方格")
        pd.DataFrame({"x": 残点[:, 0], "y": 残点[:, 1], "距最近锚点": 残深}).to_csv(
            HERE / "覆盖残余_六点.csv", index=False, encoding="utf-8-sig")
        print("    （已写入 覆盖残余_六点.csv，供作图）")
    print("    残余是**面积**小而不是**逻辑**小：干扰源恰好落在残余里的概率约等于面积比，")
    print("    工程上可忽略；但题目要求「确保所有干扰源被清除」，带 0.0x% 漏判概率的")
    print("    策略在论证上是不完整的。本文取 7。")

    print("\n" + "=" * 90)
    print("四、离散化裕度：栅格判空要再让一点")
    print("=" * 90)
    print("    栅格只有整格落进排除盘才划掉（中心距 <= 1000 - 步长/sqrt2），")
    print("    所以「掩码空 => 可行集空」严格成立，代价是有效排除半径变小。")
    print(f"    {'步长(m)':>8} {'裕度(m)':>9} {'有效排除半径(m)':>16} {'栅格点数':>9} "
          f"{'仍 >= 900 m?':>13}")
    for 步长 in (10.0, 20.0, 25.0, 40.0, 100.0):
        g = C.栅格(步长)
        有效 = C.R_RECV_MIN - g.裕度
        print(f"    {步长:>8.1f} {g.裕度:>9.2f} {有效:>16.2f} {g.n:>9d} "
              f"{'是' if 有效 >= C.R_AREA / 2 else '否':>13}")
    print("    本文取 20 m：有效排除半径 985.86 m，仍大于七点构造需要的 900 m，")
    print("    所以离散化不改变 K* = 7，只是让每个排除盘小了 14 m。")

    print("\n" + "=" * 90)
    print("五、K = 7 时使巡回路程最短的锚点布局")
    print("=" * 90)
    print("    锚点不是随便放的：既要覆盖，又要被走一遍。于是")
    print("        min  从原点出发的开放巡回长度")
    print("        s.t. 半径 1000 m 的圆盘之并 ⊇ 目标圆域")
    print(f"    {'K':>3} {'最短巡回(m)':>13} {'行进(s)':>10} "
          f"{'每空频道检测(s)':>17} {'判空总代价/空频道(s)':>22}")
    锚点表, 最优K = [], None
    for K in (7, 8, 9):
        L, 中心 = 求最短覆盖巡回(K, 粗)
        if 中心 is None:
            print(f"    {K:>3} {'未找到可行解':>13}")
            continue
        行进, 检测 = L / C.V_MOVE, K * (C.T_MEASURE + C.T_SWITCH)
        print(f"    {K:>3} {L:>13.1f} {行进:>10.1f} {检测:>17.1f} {行进 + 检测:>22.1f}")
        顺序 = C.开放TSP((0.0, 0.0), [tuple(p) for p in 中心])[0]
        for 序, i in enumerate(顺序):
            锚点表.append({"K": K, "次序": 序 + 1, "x": 中心[i][0], "y": 中心[i][1],
                           "距原点": float(np.hypot(*中心[i])), "巡回长度": L})
        if 最优K is None or 行进 + 检测 < 最优K[0]:
            最优K = (行进 + 检测, K, 中心, L)
    pd.DataFrame(锚点表).to_csv(HERE / "锚点布局.csv", index=False, encoding="utf-8-sig")

    _, K, 中心, L = 最优K
    顺序 = C.开放TSP((0.0, 0.0), [tuple(p) for p in 中心])[0]
    print(f"\n    ★ 取 K = {K}，巡回长度 {L:.1f} m（行进 {L / C.V_MOVE:.0f} s）。")
    print(f"    与「圆心 + 6 环点」的教科书构型相比："
          f"后者巡回 {巡回长度(七点构造())[0]:.0f} m，"
          f"优化后省 {巡回长度(七点构造())[0] - L:.0f} m。")
    print("    锚点（按访问次序）：")
    for 序, i in enumerate(顺序):
        print(f"        {序 + 1}. ({中心[i][0]:8.1f}, {中心[i][1]:8.1f})  "
              f"距原点 {np.hypot(*中心[i]):7.1f} m")
    print("    注意最优布局里没有「站在原点检测」这一项：原点当然要被某个排除盘盖住，")
    print("    但盖住它的不必是站在原点的那一次检测。")

    print("\n" + "=" * 90)
    print("六、判空代价的下界（后面理论下界的组成部分）")
    print("=" * 90)
    print("    每个空频道至少要 7 次检测（K* = 7），每次 5 s，切频道另计 1 s。")
    print(f"    {'源数':>6} {'空频道数':>10} {'最少检测次数':>14} {'检测+切频(s)':>15}")
    for n源 in (C.源数上界, 13, C.源数下界):
        e = C.频道数 - n源
        print(f"    {n源:>6} {e:>10} {7 * e:>14} "
              f"{7 * e * (C.T_MEASURE + C.T_SWITCH):>15.0f}")
    print("    这部分代价**任何策略都消不掉**，只取决于空频道个数。")
    print("    典型情形（20 频道、13 源、7 个空频道）为 294 s。")
    print("    注：7 次是「只用 no_signal => 距离 > 1000」这一条排除规则下的下界。")
    print("    收到过信号的频道另有更强的序约束可用（公共.序约束半平面），")
    print("    但空频道永远收不到信号，所以对空频道这个下界是紧的。")

    print("\n已写入 覆盖数_最优半径.csv、锚点布局.csv、覆盖残余_六点.csv、七点构造.csv")


if __name__ == "__main__":
    with open(HERE / "覆盖锚点结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '覆盖锚点结果.txt'}")
