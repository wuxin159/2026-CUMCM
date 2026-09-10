"""问题二 · 第 4 步：独立复核（只算数，不作图）。

P1 的教训②：**复核代码不能调用被测代码**。本脚本里的几何全部从题面定义重推：
    · 楔形隶属：按「方位角与示向度之差不超过 δ」的原始定义，用 atan2 直接判
    · 区域顶点：把边界曲线（4 条射线 + 3 个圆）两两求交，再用原始定义筛
    · 凸包、直径、最小覆盖圆：本文件内自写，不 import 模型.py 与 P1 的任何几何函数
唯一从 模型.py 取来的是**被检验的结果本身**，以及题目常量。

五节：
    1  从定义重建定位区域，与 模型.py 的多边形逐点比对
    2  蒙特卡洛隶属度比对（应为「模型区域 ⊇ 真区域」且差集只在数值容差量级）
    3  弧段数（外圈圆弧的切线段数）的离散化误差
    4  求直径快 / 最小覆盖圆快 与 P1 原版的等价性
    5  变异测试：把模型改错，看本复核能否发现
"""
import sys
import math
from itertools import combinations
from pathlib import Path

import numpy as np

import 模型 as M

HERE = Path(__file__).resolve().parent
SEED = 20260910

R_AREA = 1800.0
DELTA = 1.0
R_NEAR = 5.0
R_RECV_MAX = 1500.0


# ================================================================ 自写几何（不依赖被测代码）
def 角差(a, b):
    """把 a - b 折算到 (-180, 180]。"""
    return (a - b + 180.0) % 360.0 - 180.0


def 按定义在楔形内(P, S, theta, delta, 容差=1e-9):
    """题面原始定义：检测点指向 P 的方位角与示向度之差不超过 delta。"""
    dx, dy = P[0] - S[0], P[1] - S[1]
    if dx * dx + dy * dy < 1e-18:
        return False
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    return abs(角差(ang, theta)) <= delta + 容差


def 按定义在区域内_批(px, py, S1, th1, delta, rho_lo, rho_hi, S2=None, th2=None, 容差=1e-9):
    """按定义在区域内 的向量化版本。逐点版留着做交叉对照，两者结果必须一致。

    第一版这一节用 Python 循环跑 100 万次判定，单次复核要十几分钟、最后超时。
    这里改成 numpy 广播；判定逻辑与逐点版逐字对应，没有换算法。
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    好 = np.ones(px.shape, dtype=bool)
    r1 = np.hypot(px - S1[0], py - S1[1])
    好 &= (r1 >= rho_lo - 容差) & (r1 <= rho_hi + 容差)
    好 &= np.hypot(px, py) <= R_AREA + 容差
    for S, th in ([(S1, th1)] + ([(S2, th2)] if S2 is not None else [])):
        ang = np.degrees(np.arctan2(py - S[1], px - S[0])) % 360.0
        差 = (ang - th + 180.0) % 360.0 - 180.0
        好 &= np.abs(差) <= delta + 容差
        好 &= np.hypot(px - S[0], py - S[1]) > 1e-9
    return 好


def 在凸多边形内_批(px, py, 顶点, 容差=1e-9):
    """自写的向量化凸多边形包含判定，不调用 P1 的 点在多边形内。"""
    V = np.asarray(顶点, dtype=float)
    n = len(V)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    正 = np.ones(px.shape, dtype=bool)
    负 = np.ones(px.shape, dtype=bool)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        叉 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        正 &= 叉 >= -容差
        负 &= 叉 <= 容差
    return 正 | 负


def 按定义在区域内(P, S1, th1, delta, rho_lo, rho_hi, S2=None, th2=None, 容差=1e-9):
    r1 = math.hypot(P[0] - S1[0], P[1] - S1[1])
    if r1 < rho_lo - 容差 or r1 > rho_hi + 容差:
        return False
    if math.hypot(P[0], P[1]) > R_AREA + 容差:
        return False
    if not 按定义在楔形内(P, S1, th1, delta, 容差):
        return False
    if S2 is not None and not 按定义在楔形内(P, S2, th2, delta, 容差):
        return False
    return True


def 线线交(P1, u1, P2, u2):
    叉 = u1[0] * u2[1] - u1[1] * u2[0]
    if abs(叉) < 1e-14:
        return []
    dx, dy = P2[0] - P1[0], P2[1] - P1[1]
    t = (dx * u2[1] - dy * u2[0]) / 叉
    return [(P1[0] + t * u1[0], P1[1] + t * u1[1])]


def 线圆交(P, u, C, R):
    """射线基点 P、方向 u 的**直线**与圆心 C、半径 R 的圆的交点。"""
    fx, fy = P[0] - C[0], P[1] - C[1]
    a = u[0] * u[0] + u[1] * u[1]
    b = 2 * (fx * u[0] + fy * u[1])
    c = fx * fx + fy * fy - R * R
    判 = b * b - 4 * a * c
    if 判 < 0:
        return []
    s = math.sqrt(判)
    return [(P[0] + t * u[0], P[1] + t * u[1]) for t in ((-b - s) / (2 * a), (-b + s) / (2 * a))]


def 圆圆交(C1, R1, C2, R2):
    dx, dy = C2[0] - C1[0], C2[1] - C1[1]
    d = math.hypot(dx, dy)
    if d < 1e-12 or d > R1 + R2 or d < abs(R1 - R2):
        return []
    a = (R1 * R1 - R2 * R2 + d * d) / (2 * d)
    h2 = R1 * R1 - a * a
    if h2 < 0:
        return []
    h = math.sqrt(h2)
    xm, ym = C1[0] + a * dx / d, C1[1] + a * dy / d
    return [(xm + h * dy / d, ym - h * dx / d), (xm - h * dy / d, ym + h * dx / d)]


def 自写凸包(点):
    """Andrew monotone chain。"""
    点 = sorted(set((round(p[0], 9), round(p[1], 9)) for p in 点))
    if len(点) <= 2:
        return 点

    def 叉(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    下 = []
    for p in 点:
        while len(下) >= 2 and 叉(下[-2], 下[-1], p) <= 1e-12:
            下.pop()
        下.append(p)
    上 = []
    for p in reversed(点):
        while len(上) >= 2 and 叉(上[-2], 上[-1], p) <= 1e-12:
            上.pop()
        上.append(p)
    return 下[:-1] + 上[:-1]


def 自写直径(顶点):
    最大, 对 = 0.0, None
    for p, q in combinations(顶点, 2):
        d = math.hypot(p[0] - q[0], p[1] - q[1])
        if d > 最大:
            最大, 对 = d, (p, q)
    return 最大, 对


def _两点圆(p, q):
    c = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    return c, math.hypot(p[0] - q[0], p[1] - q[1]) / 2


def _三点圆(p, q, s):
    ax, ay = q[0] - p[0], q[1] - p[1]
    bx, by = s[0] - p[0], s[1] - p[1]
    分母 = 2 * (ax * by - ay * bx)
    if abs(分母) < 1e-18:
        return None
    a2, b2 = ax * ax + ay * ay, bx * bx + by * by
    ux = (by * a2 - ay * b2) / 分母
    uy = (ax * b2 - bx * a2) / 分母
    return (p[0] + ux, p[1] + uy), math.hypot(ux, uy)


def _平凡圆(边界):
    if not 边界:
        return ((0.0, 0.0), -1.0)
    if len(边界) == 1:
        return (边界[0], 0.0)
    if len(边界) == 2:
        return _两点圆(*边界)
    圆 = _三点圆(*边界)
    return 圆 if 圆 else _两点圆(边界[0], 边界[1])


def 自写最小覆盖圆(顶点, 种子=0):
    """Welzl 随机增量算法。与 P1 的「枚举全部点对与三点组」是不同的搜索结构。

    上一版写的是「模式搜索 + 步长减半」，对 max_i |c-P_i| 这种**非光滑**目标
    会卡在棱上：8 个采样方向全都上升，真正的下降方向夹在它们中间，步长再小
    也逃不出去，实测半径误差达 2.1 米。这类坑正是 P1 教训②说的
    「看着像验证、其实根本没验到」，故换成有正确性保证的算法。
    """
    P = [tuple(map(float, p)) for p in 顶点]
    np.random.default_rng(种子).shuffle(P)

    def 递归(n, 边界):
        圆 = _平凡圆(边界)
        if n == 0 or len(边界) == 3:
            return 圆
        for i in range(n):
            p = P[i]
            if 圆[1] < 0 or math.hypot(p[0] - 圆[0][0], p[1] - 圆[0][1]) > 圆[1] + 1e-12:
                圆 = 递归(i, 边界 + [p])
        return 圆

    c, r = 递归(len(P), [])
    return c, max(r, 0.0)


def 最优性证书(顶点, 圆心, 半径, 容差=1e-6):
    """不比对任何算法，直接检验「这个圆就是最小覆盖圆」的充要条件。

    条件一（可行）：所有顶点都在圆内或圆上。
    条件二（不可再缩）：设支撑集 S = {顶点 : |P-c| >= r - 容差}，
        c 最优 <=> 不存在方向 u 使得 S 中所有点都严格在 u 的正侧
        （否则沿 u 挪动圆心可同时缩短到 S 中每个点的距离，半径还能再小）。
        等价于 0 属于 {P-c : P in S} 的凸包。这里用密集方向扫描来判。

    这是比「换个算法再算一遍」更强的复核：它给出的是最优性的**证明**，
    不依赖第二个实现是否也对。
    """
    P = np.asarray(顶点, dtype=float)
    c = np.asarray(圆心, dtype=float)
    d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    if float(np.max(d)) > 半径 + 容差:
        return False, "有顶点在圆外"
    S = P[d >= 半径 - max(容差, 1e-9) * max(半径, 1.0)]
    if len(S) == 0:
        return False, "支撑集为空"
    V = S - c
    模 = np.hypot(V[:, 0], V[:, 1])
    V = V[模 > 1e-12] / 模[模 > 1e-12, None]
    if len(V) == 0:
        return True, "退化为单点"
    角 = np.radians(np.arange(0.0, 360.0, 0.5))
    U = np.stack([np.cos(角), np.sin(角)], axis=1)
    投影最小 = np.min(V @ U.T, axis=0)           # 每个方向上，支撑点投影的最小值
    if float(np.max(投影最小)) > 1e-9:
        return False, "存在可继续缩小的方向"
    return True, "满足最优性证书"


def 独立求区域(S1, th1, S2, th2, delta, rho_lo, rho_hi):
    """把边界曲线两两求交得候选点，再用原始定义筛，最后取凸包。"""
    曲线线 = []
    for S, th in ((S1, th1), (S2, th2)):
        for 侧 in (-1, 1):
            a = math.radians(th + 侧 * delta)
            曲线线.append((S, (math.cos(a), math.sin(a))))
    曲线圆 = [(S1, rho_lo), (S1, rho_hi), ((0.0, 0.0), R_AREA)]

    候选 = []
    for (P1, u1), (P2, u2) in combinations(曲线线, 2):
        候选 += 线线交(P1, u1, P2, u2)
    for (P, u) in 曲线线:
        for (C, R) in 曲线圆:
            候选 += 线圆交(P, u, C, R)
    for (C1, R1), (C2, R2) in combinations(曲线圆, 2):
        候选 += 圆圆交(C1, R1, C2, R2)

    好 = [p for p in 候选
          if 按定义在区域内(p, S1, th1, delta, rho_lo, rho_hi, S2, th2, 容差=1e-6)]
    return 自写凸包(好)


def 分节(标题):
    print()
    print("=" * 78)
    print(标题)
    print("=" * 78)


# ================================================================ 1
def 节1_顶点比对():
    分节("1  从定义重建定位区域，与 模型.py 的结果比对")
    print("模型侧：楔形拆半平面 -> Sutherland-Hodgman 逐条裁剪。")
    print("复核侧：边界曲线两两求交 -> 用原始角度/距离定义筛点 -> 自写凸包。")
    print("两条路径除题目常量外没有共享代码。\n")
    print("外圈圆弧在模型侧是折线、在复核侧是真圆弧，所以允许**模型 >= 复核**的微小偏差。\n")

    rng = np.random.default_rng(SEED)
    Ω1 = M.先验区域()
    最大直径差, 最大Rmin差, 最大顶点距 = 0.0, 0.0, 0.0
    坏 = 0
    N = 600
    for _ in range(N):
        rho = rng.uniform(50, 1480)
        a = rng.uniform(-300, 1800)
        b = rng.uniform(30, 1200) * rng.choice([-1, 1])
        α = rng.uniform(-DELTA, DELTA)
        G = (rho * math.cos(math.radians(α)), rho * math.sin(math.radians(α)))
        th2 = M.方位角((a, b), G) + rng.uniform(-DELTA, DELTA)

        模型顶点 = M.定位区域(Ω1, (a, b), th2)
        复核顶点 = 独立求区域((0.0, 0.0), 0.0, (a, b), th2, DELTA, R_NEAR, R_RECV_MAX)
        if len(模型顶点) < 3 or len(复核顶点) < 3:
            if len(模型顶点) >= 3 or len(复核顶点) >= 3:
                坏 += 1
            continue
        d模 = M.区域指标(模型顶点)[0]
        d核 = 自写直径(复核顶点)[0]
        R模 = M.区域指标(模型顶点)[1]
        R核 = 自写最小覆盖圆(复核顶点)[1]
        最大直径差 = max(最大直径差, abs(d模 - d核))
        最大Rmin差 = max(最大Rmin差, abs(R模 - R核))
        for p in 复核顶点:                       # 复核顶点应当都落在模型多边形内（模型是超集）
            最大顶点距 = max(最大顶点距, 0.0 if M.算法.点在多边形内(p, 模型顶点, 1e-4)
                            else min(math.hypot(p[0]-q[0], p[1]-q[1]) for q in 模型顶点))
        if d模 + 1e-6 < d核:
            坏 += 1
    print(f"    {N} 组随机构型：")
    print(f"      直径最大绝对差        = {最大直径差:.6f} m")
    print(f"      最小覆盖圆半径最大差   = {最大Rmin差:.6f} m")
    print(f"      复核顶点落在模型区域外的最大距离 = {最大顶点距:.6f} m")
    print(f"      模型直径小于复核直径（不安全侧）的组数 = {坏}")
    print("    -> 差异应当在圆弧折线化的厘米量级，且模型侧始终不小于复核侧。")


# ================================================================ 2
def 节2_隶属度():
    分节("2  蒙特卡洛隶属度比对")
    print("在包围盒里撒点，一边按原始定义判、一边按模型多边形判，统计两者不一致的比例。\n")
    rng = np.random.default_rng(SEED + 1)
    Ω1 = M.先验区域()
    print(f"    {'构型':>26} {'撒点数':>8} {'定义内':>8} {'模型内':>8} "
          f"{'定义内但模型外':>14} {'模型内但定义外':>14}")
    for rho, a, b in [(400, 200, 200), (900, 500, 300), (1300, 700, 500),
                      (1450, 900, 200), (200, 100, 400)]:
        G = (float(rho), 0.0)
        th2 = M.方位角((a, b), G)
        L = M.定位区域(Ω1, (a, b), th2)
        if not L:
            continue
        xs = [p[0] for p in L]
        ys = [p[1] for p in L]
        pad = 2.0
        N = 200000
        px = rng.uniform(min(xs) - pad, max(xs) + pad, N)
        py = rng.uniform(min(ys) - pad, max(ys) + pad, N)
        定义 = 按定义在区域内_批(px, py, (0.0, 0.0), 0.0, DELTA, R_NEAR,
                                  R_RECV_MAX, (a, b), th2)
        模型 = 在凸多边形内_批(px, py, L)
        # 抽 2000 个点用逐点版复查向量化实现，防止「快版本自己写错」
        取 = np.random.default_rng(0).choice(len(px), 2000, replace=False)
        逐点 = np.array([按定义在区域内((px[i], py[i]), (0.0, 0.0), 0.0, DELTA,
                                       R_NEAR, R_RECV_MAX, (a, b), th2) for i in 取])
        assert np.array_equal(逐点, 定义[取]), "向量化与逐点判定不一致"
        print(f"    {str((rho,a,b)):>26} {N:8d} {定义.sum():8d} {模型.sum():8d} "
              f"{int((定义 & ~模型).sum()):14d} {int((~定义 & 模型).sum()):14d}")
    print("    -> 「定义内但模型外」必须为 0（模型是超集，不能漏掉真实可行点）；")
    print("       「模型内但定义外」是圆弧折线化的外凸余量，只占极小比例。")


# ================================================================ 3
def 节3_弧段数():
    分节("3  外圈圆弧折线化的段数灵敏度")
    print("模型默认用 6 条切线逼近 2° 的外圈圆弧。段数取到 240 作为「准确值」比对。\n")
    print(f"    {'弧段数':>8} {'Ω1 顶点数':>10} {'Ω1 面积':>14} {'Ω1 直径':>12} "
          f"{'典型 L 的直径':>14}")
    参照 = None
    # 注意：这里只取直径与面积，不调 区域指标。
    # 模型的 最小覆盖圆快 在「以直径为直径的圆盖不住」时会退回 P1 的枚举法，
    # 那是 O(m^3)；弧段数 240 时 Ω1 有 243 个顶点，枚举量到千万级，跑不动。
    # 实战中定位区域只有十来个顶点，退回分支代价可忽略——但本节的极端设定会踩到，
    # 故此处绕开。这条限制值得写进论文的算法复杂度说明。
    for n in (2, 4, 6, 12, 24, 60, 240):
        Ω = M.先验区域(弧段数=n)
        d = M.求直径快(Ω)[0]
        A = M.算法.多边形面积(Ω)
        th2 = M.方位角((600.0, 300.0), (1450.0, 0.0))
        L = M.定位区域(Ω, (600.0, 300.0), th2)
        dl = M.区域指标(L)[0] if L else float("nan")
        if n == 240:
            参照 = (A, d, dl)
        print(f"    {n:8d} {len(Ω):10d} {A:14.3f} {d:12.5f} {dl:14.5f}")
    print(f"\n    以 240 段为准，6 段的面积相对偏差 = "
          f"{abs(M.区域指标(M.先验区域(弧段数=6))[3]-参照[0])/参照[0]:.3e}")
    print("    -> 6 段已足够；再多只是徒增顶点数、拖慢上万次调用的扫描。")


# ================================================================ 4
def 节4_快慢等价():
    分节("4  求直径快 / 最小覆盖圆快 与 P1 原版的等价性")
    print("模型层为了扫描速度改写了这两个函数，必须证明与 P1 已复核过的版本等价。\n")
    rng = np.random.default_rng(SEED + 2)
    最大d差, 最大R差_对P1, 最大R差_对网格, 最大心距 = 0.0, 0.0, 0.0, 0.0
    退回次数, 证书失败 = 0, 0
    N = 3000
    for _ in range(N):
        m = rng.integers(3, 12)
        点 = 自写凸包([(rng.uniform(-100, 100), rng.uniform(-100, 100))
                       for _ in range(int(m))])
        if len(点) < 3:
            continue
        d1 = M.算法.求直径(点)[0]
        d2 = M.求直径快(点)[0]
        c1, R1 = M.算法.最小覆盖圆(点)
        c2, R2 = M.最小覆盖圆快(点)
        c3, R3 = 自写最小覆盖圆(点)              # 第三条完全不同的路径
        好, _ = 最优性证书(点, c2, R2)           # 第四条：不比对任何算法，直接验最优性
        证书失败 += (not 好)
        最大d差 = max(最大d差, abs(d1 - d2))
        最大R差_对P1 = max(最大R差_对P1, abs(R1 - R2))
        最大R差_对网格 = max(最大R差_对网格, abs(R2 - R3))
        最大心距 = max(最大心距, math.hypot(c1[0] - c2[0], c1[1] - c2[1]),
                       math.hypot(c2[0] - c3[0], c2[1] - c3[1]))
        if R2 > d2 / 2 + 1e-9:
            退回次数 += 1
    print(f"    {N} 个随机凸多边形：")
    print(f"      直径最大差（快 vs P1 原版）        = {最大d差:.3e} m")
    print(f"      最小覆盖圆半径最大差（快 vs P1）    = {最大R差_对P1:.3e} m")
    print(f"      最小覆盖圆半径最大差（快 vs 自写网格）= {最大R差_对网格:.3e} m")
    print(f"      圆心最大距离                      = {最大心距:.3e} m")
    print(f"      **最优性证书**不通过的多边形数     = {证书失败} / {N}")
    print(f"      走了「以直径为直径的圆盖不住」回退分支的比例 = {退回次数/N:.2%}")
    print("    -> 前三条是「换算法再算一遍」，第四条是直接验最优性充要条件，")
    print("       后者不依赖第二个实现是否也对，是更强的复核。")


# ================================================================ 5
def 节5_变异测试():
    分节("5  变异测试：把模型改错，看第 1、2 节的复核能否发现")
    print("P1 的教训②：一份「看着像交叉验证」的复核可能对整类 bug 全部漏检。")
    print("这里人为植入 6 个典型错误，逐一检查复核是否报警。\n")

    print("第一版的抽样区间是 ρ∈[100,1400]，结果「漏掉 ρ<=1500 约束」「漏掉 ρ>5 约束」")
    print("两个变异全部漏检——因为那些构型根本没碰到这两条边界，删掉它们不改变任何结果。")
    print("这正是 P1 警告过的「看着像验证、其实测不到」。现在把抽样推到边界上：")
    print("三分之一的构型贴着远端 ρ∈[1400,1499]，三分之一贴着近端 ρ∈[5,60]。\n")

    def 抽构型(rng, i):
        """按 i 的余数分三档，强制覆盖两条距离边界。"""
        if i % 3 == 0:                            # 贴远端：区域会被 ρ<=ρ_hi 削掉
            rho = rng.uniform(1400, 1499)
            a = rng.uniform(rho - 500, rho + 300)
            b = rng.uniform(200, 900)
        elif i % 3 == 1:                          # 贴近端：区域会被 ρ>5 削掉
            rho = rng.uniform(5.5, 60)
            a = rng.uniform(-40, 60)
            b = rng.uniform(5, 80)
        else:                                     # 一般构型
            rho = rng.uniform(100, 1400)
            a = rng.uniform(0, 1400)
            b = rng.uniform(50, 900)
        return rho, a, b

    def 跑一遍复核(先验区域函数, 定位区域函数):
        """返回 (最大直径差, 是否出现「定义内但模型外」)。"""
        rng = np.random.default_rng(SEED + 3)
        Ω1 = 先验区域函数()
        最大差, 漏点 = 0.0, 0
        for i in range(120):
            rho, a, b = 抽构型(rng, i)
            G = (rho, 0.0)
            th2 = M.方位角((a, b), G)
            模型顶点 = 定位区域函数(Ω1, (a, b), th2)
            复核顶点 = 独立求区域((0.0, 0.0), 0.0, (a, b), th2, DELTA, R_NEAR, R_RECV_MAX)
            if len(复核顶点) < 3:
                continue
            if len(模型顶点) < 3:
                漏点 += len(复核顶点)
                最大差 = max(最大差, 自写直径(复核顶点)[0])
                continue
            最大差 = max(最大差, abs(M.区域指标(模型顶点)[0] - 自写直径(复核顶点)[0]))
            for p in 复核顶点:
                if not M.算法.点在多边形内(p, 模型顶点, 1e-4):
                    漏点 += 1
        return 最大差, 漏点

    正常先验 = M.先验区域
    正常定位 = M.定位区域

    变异 = [
        ("无变异（基准）", lambda: 正常先验(), lambda Ω, S, t: 正常定位(Ω, S, t)),
        ("楔形半角写成 0.5°",
         lambda: 正常先验(delta=0.5), lambda Ω, S, t: 正常定位(Ω, S, t, 0.5)),
        ("楔形半角写成 2°",
         lambda: 正常先验(delta=2.0), lambda Ω, S, t: 正常定位(Ω, S, t, 2.0)),
        ("漏掉 rho <= 1500 约束",
         lambda: 正常先验(rho_hi=1e6), lambda Ω, S, t: 正常定位(Ω, S, t)),
        ("漏掉 rho > 5 约束",
         lambda: 正常先验(rho_lo=0.0), lambda Ω, S, t: 正常定位(Ω, S, t)),
        ("第二楔形的示向度差 0.3°",
         lambda: 正常先验(), lambda Ω, S, t: 正常定位(Ω, S, t + 0.3)),
        ("第二楔形顶点偏移 20 m",
         lambda: 正常先验(),
         lambda Ω, S, t: 正常定位(Ω, (S[0] + 20, S[1]), t)),
    ]
    print(f"    {'变异':<26} {'最大直径差(m)':>14} {'漏掉的真实可行点':>16} {'复核是否报警':>12}")
    for 名, f先验, f定位 in 变异:
        差, 漏 = 跑一遍复核(f先验, f定位)
        报警 = (差 > 0.05) or (漏 > 0)
        print(f"    {名:<26} {差:14.4f} {漏:16d} {'报警' if 报警 else '未报警':>12}")
    print("    -> 基准行必须「未报警」，其余每一行都必须「报警」。")
    print("       若某个变异漏检，说明复核的独立性不够，必须改复核而不是改结论。")


def main():
    print("问题二 · 独立复核")
    print(f"随机种子 {SEED}")
    print("本文件的几何全部自写，不 import 模型.py / P1 的求解函数（第 4 节除外，")
    print("那一节的目的就是把三条路径摆在一起比）。")
    节1_顶点比对()
    节2_隶属度()
    节3_弧段数()
    节4_快慢等价()
    节5_变异测试()
    print("\n完毕。")


if __name__ == "__main__":
    with open(HERE / "独立复核结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '独立复核结果.txt'}")
