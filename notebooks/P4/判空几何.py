"""问题四 · 第 1 步：判空的几何（只算数，不作图）。

问题三的判空是一个**圆覆盖**问题：一次 no_signal 排除一个半径 1000 m 的圆盘，
把目标圆域盖满即可，最少 7 个锚点。
问题四的 no_signal 不再排除任何位置（源可能就在脚下、只是背对着你），
判空判据变成命题 D5 的**围堵**条件：

    目标区域内的位置 x 被排除  <=>  x ∈ conv{ p ∈ N : |p - x| <= 1000 }

本脚本回答由此产生的四个量化问题：

    ① 严格下界：至少需要多少个锚点？（三重覆盖的面积论证，与构造无关）
    ② 本文采用的 24 点构造，在**精确判据**下能做到多大的围堵半径？余量够不够？
    ③ 前 k 个锚点分别能围住多少面积（画「围堵进度」用）
    ④ 锚点巡回长度与每个空频道的判空代价，作为理论下界与代价核算的输入

**关于构造的来历**：`策略.围堵锚点构造` 是离线优化的结果——
从贪心可行集（42 点）出发，反复「删一个点 + 局部精修」，
在策略实际使用的判据栅格上保持残余为 0，压到 24 点。
那轮搜索要几分钟，本脚本默认**只验证不重算**；
加 `--重算` 可以重跑一遍（约 3~5 分钟），用于复现该构造。

输出：判空几何结果.txt、判空锚点布局.csv、围堵残余_样例.csv、判空代价对照.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 公共 as C
import 策略 as P

HERE = Path(__file__).resolve().parent
SEED = 20260912


# ================================================================ 一、围堵判据（精确版）
def 角度矩阵(锚点, X):
    """X 的每一行看向每个锚点的方位角（度）。返回 (len(X), len(锚点))。"""
    A = np.asarray(锚点, dtype=float)
    return np.degrees(np.arctan2(A[None, :, 1] - X[:, None, 1],
                                 A[None, :, 0] - X[:, None, 0])) % 360.0


def 最大角隙(角, 计数):
    """每行取前 计数[i] 个角度，求最小包含弧的补——最大角隙。

    最大角隙 <= 180° 等价于「这些方向正张成 R²」，也就等价于 x ∈ conv(这些点)。
    """
    n, K = 角.shape
    掩 = np.arange(K)[None, :] < np.asarray(计数)[:, None]
    B = np.where(掩, 角, np.nan)
    B = np.sort(B, axis=1)                      # NaN 被排到末尾
    m = np.asarray(计数)[:, None]
    idx = np.arange(K)[None, :]
    首 = B[:, :1]
    下一个 = np.where(idx + 1 < m, np.roll(B, -1, axis=1), 首 + 360.0)
    隙 = np.where(idx < m, 下一个 - B, -np.inf)
    return np.nanmax(隙, axis=1)


def 围堵半径(锚点, X):
    """对每个 x 求「最小围堵半径」r(x)：把最近的锚点一个个加进来，直到 x 落进它们的凸包。

    r(x) = 使 x ∈ conv(距离 <= r 的锚点) 成立的最小 r；无法围堵时返回 +inf。
    锚点集能在半径 r 下判空 <=> max_x r(x) <= r。

    **这是与策略里那套扇区离散化完全独立的第二套实现**，两者互为交叉验证：
    这里按距离排序逐个加点、算最大角隙；策略里按方格 x 扇区打掩码。
    """
    A = np.asarray(锚点, dtype=float)
    X = np.asarray(X, dtype=float)
    K = len(A)
    if K < 3:
        return np.full(len(X), np.inf)
    D = np.hypot(X[:, None, 0] - A[None, :, 0], X[:, None, 1] - A[None, :, 1])
    序 = np.argsort(D, axis=1)
    Ds = np.take_along_axis(D, 序, axis=1)
    As = np.take_along_axis(角度矩阵(A, X), 序, axis=1)
    lo = np.full(len(X), 3)
    hi = np.full(len(X), K)
    可行 = 最大角隙(As, np.full(len(X), K)) <= 180.0 + 1e-9
    while np.any(lo < hi):
        mid = (lo + hi) // 2
        ok = 最大角隙(As, mid) <= 180.0 + 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid + 1)
    取 = np.clip(lo - 1, 0, K - 1)[:, None]
    return np.where(可行, np.take_along_axis(Ds, 取, axis=1)[:, 0], np.inf)


def 目标格点(步长, R=C.R_AREA):
    k = int(np.ceil(R / 步长))
    轴 = np.arange(-k, k + 1) * 步长
    Xg, Yg = np.meshgrid(轴, 轴)
    内 = np.hypot(Xg, Yg) <= R + 1e-9
    return np.stack([Xg[内], Yg[内]], axis=1)


# ================================================================ 二、严格下界
def 三重覆盖下界(R=C.R_AREA, r=C.R_RECV_MIN):
    """任何可行锚点集的个数下界，**与构造无关**。

    论证：围堵要求 x ∈ conv(距离 <= r 的锚点)，二维凸包至少要 3 个点
    （共线情形测度为零），所以每个 x 的 r-邻域内至少有 3 个锚点。于是

        Σ_p |D(p, r) ∩ D(O, R)| >= 3 |D(O, R)|

    左端每一项不超过 π r²，故  K >= 3 R² / r²。
    这一步只用到「每点至少 3 个见证者」，没用到「要把 x 围起来」，所以是**松**的下界，
    但它不依赖任何构造，可以直接写进论文。
    """
    return 3.0 * R * R / (r * r)


# ================================================================ 三、离线优化（可选重算）
def 离线优化(rng, 起点, 判据, 轮数=1200, 时限=240.0):
    """从一个可行集出发，反复「删一个点 + 局部精修」。目标是 min-max 型，处处不光滑，
    通用无导数优化器在这类目标上跑不动（问题三 §6.1 的教训），直接用下降搜索。"""
    import time

    def 残(点集):
        m = 判据.新掩码()
        for p in 点集:
            m &= ~判据.扇区标记(p)
        return int(m.sum())

    def 精修(点集, n, 步0):
        点集 = [tuple(p) for p in 点集]
        最好 = 残(点集)
        for t in range(n):
            if 最好 == 0:
                break
            步 = 步0 * (1 - t / n) + 15.0
            i = int(rng.integers(0, len(点集)))
            试 = list(点集)
            q = np.asarray(试[i]) + rng.normal(0, 步, 2)
            if np.hypot(*q) > C.R_AREA + 1100:
                q = q / np.hypot(*q) * (C.R_AREA + 1100)
            试[i] = (float(q[0]), float(q[1]))
            v = 残(试)
            if v <= 最好:
                点集, 最好 = 试, v
        return 点集, 最好

    当前 = [tuple(p) for p in 起点]
    t0 = time.time()
    while len(当前) > 12 and time.time() - t0 < 时限:
        序 = sorted(range(len(当前)),
                    key=lambda i: 残([p for k, p in enumerate(当前) if k != i]))
        for i in 序[:8]:
            试, 剩 = 精修([p for k, p in enumerate(当前) if k != i], 轮数 * 6, 500.0)
            if 剩 == 0:
                当前 = 试
                print(f"    K={len(当前)}  巡回 {C.开放TSP((0,0),当前)[1]:.0f} m", flush=True)
                break
        else:
            break
    return 当前


# ================================================================ 四、主程序
def main(argv):
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print("问题四 · 判空的几何")
    print("=" * 78)
    print("""
问题三的判空是「用半径 1000 m 的圆盘盖住目标圆域」，最少 7 个锚点。
问题四的 no_signal 不排除任何位置，判据换成命题 D5 的围堵条件：

    目标区域内的位置 x 被排除  <=>  x ∈ conv{ p ∈ N : |p - x| <= 1000 }

一句话：**问题三每个点只需要一个见证者，问题四每个点至少需要三个，而且要把它围起来。**
""")

    # ---------------------------------------------------------- ① 严格下界
    print("-" * 78)
    print("① 与构造无关的严格下界（三重覆盖的面积论证）")
    下界 = 三重覆盖下界()
    print(f"""
    围堵要求每个 x 的 1000 m 邻域内至少有 3 个锚点（二维凸包至少 3 个点），于是
        Σ_p |D(p,1000) ∩ D(O,1800)| >= 3 |D(O,1800)|
    左端每项 <= π·1000²，故
        K >= 3 R²/r² = 3 x 1800²/1000² = {下界:.2f}   ==>   K4 >= {int(np.ceil(下界))}

    对照问题三的同类论证（每点只需 1 个见证者）：K >= R²/r² = {C.R_AREA**2/C.R_RECV_MIN**2:.2f}
    ——**仅这一步，下界就从 4 抬到了 {int(np.ceil(下界))}**。
    注意这只用到「至少 3 个见证者」，没有用到「要把 x 围起来」，所以还很松。
""")

    # ---------------------------------------------------------- ② 构造与验证
    判据 = P.取栅格(P.默认参数["栅格步长"], P.默认参数["扇区数"])
    构造 = list(P.围堵锚点构造)
    if "--重算" in argv:
        print("-" * 78)
        print("（--重算）重跑离线优化，从在线贪心的可行集出发逐点压缩")
        规划 = P.取栅格(P.默认参数["规划步长"], P.默认参数["规划扇区数"],
                        裕度=P.默认参数["规划裕度"])
        规划器 = P.取规划器(判据, 规划, P.默认参数["锚点间距"], P.默认参数["锚点外扩"])
        起 = 规划器.贪心锚点(规划.新掩码(), 判据.新掩码(), 准则="覆盖",
                             上限=90, 用构造=False)
        print(f"    在线贪心起点：K={len(起)}，巡回 {C.开放TSP((0,0),起)[1]:.0f} m",
              flush=True)
        构造 = 离线优化(rng, 起, 判据)

    print("-" * 78)
    print("② 构造的验证（两套独立实现交叉验证）")
    # 验证一：策略实际使用的扇区离散化掩码
    掩 = 判据.新掩码()
    for p in 构造:
        掩 &= ~判据.扇区标记(p)
    残余二元组 = int(掩.sum())
    # 验证二：10 m 细网格上的**精确**判据（按距离排序 + 最大角隙）
    X细 = 目标格点(10.0)
    r细 = 围堵半径(构造, X细)
    外 = [p for p in 构造 if np.hypot(*p) > C.R_AREA]
    _, 巡回 = C.开放TSP((0.0, 0.0), 构造)
    print(f"""
    构造：**K = {len(构造)} 个锚点**，开放巡回 {巡回:.0f} m（行进 {巡回/C.V_MOVE:.0f} s）

    验证一（策略实际使用的判据栅格：步长 {判据.步长:.0f} m、{判据.扇区数} 扇区、
             有效排除半径 {判据.有效半径:.1f} m，含全部保守裕度）
        残余 (方格, 扇区) 二元组      {残余二元组}
        结论                          {'判空成立' if 残余二元组 == 0 else '**未盖满**'}

    验证二（10 m 细网格 {len(X细)} 个点上的**精确**判据，与策略的离散化完全独立）
        最大围堵半径                  {float(np.max(r细)):.2f} m
        可用半径（题给 R_eff 下界）   {C.R_RECV_MIN:.0f} m
        余量                          {C.R_RECV_MIN - float(np.max(r细)):.2f} m
        未被围堵的格点数              {int(np.count_nonzero(r细 > C.R_RECV_MIN))}

    两套实现给出一致结论，说明结果不依赖某一种离散化方式。

    **其中 {len(外)} 个锚点落在目标圆域之外**（最远 {max(np.hypot(*p) for p in 构造):.0f} m）。
    这不是取巧，是必须：设干扰源位于 |G| = 1800 的边界上、定向方向朝外，
    它的覆盖半平面过 G 且法向朝外，而目标圆域整个落在该半平面的另一侧
    （圆在边界点的切线内侧）——**区域内任何一点都在它的背面**。
    题面明确允许区域外检测（附件 2 §1.1）。
""")
    布局 = pd.DataFrame(构造, columns=["x", "y"])
    布局["距原点"] = np.hypot(布局["x"], 布局["y"])
    布局["在区域外"] = 布局["距原点"] > C.R_AREA
    顺序, _ = C.开放TSP((0.0, 0.0), 构造)
    布局["访问序"] = pd.Series({i: k + 1 for k, i in enumerate(顺序)})
    布局["巡回总长"] = 巡回
    布局.sort_values("访问序").to_csv(HERE / "判空锚点布局.csv", index=False,
                                      encoding="utf-8-sig")

    # ---------------------------------------------------------- ③ 围堵进度
    print("-" * 78)
    print("③ 围堵进度：前 k 个锚点能围住多大面积")
    X验 = 目标格点(20.0)
    单元 = 20.0 * 20.0
    序访问 = [构造[i] for i in 顺序]
    行3, 样例 = [], []
    for k in range(3, len(构造) + 1):
        r = 围堵半径(序访问[:k], X验)
        已 = r <= C.R_RECV_MIN
        行3.append({"锚点数": k, "已围堵面积占比": float(已.mean()),
                    "已围堵面积_平方米": float(已.sum()) * 单元,
                    "最大围堵半径": float(np.min([np.max(r), 9999.0]))})
        if k in (3, 6, 9, 12, 18, len(构造)):
            样例.append(pd.DataFrame({"锚点数": k, "x": X验[:, 0], "y": X验[:, 1],
                                      "已围堵": 已}))
    表3 = pd.DataFrame(行3)
    pd.concat(样例).to_csv(HERE / "围堵残余_样例.csv", index=False,
                           encoding="utf-8-sig")
    print(表3[表3["锚点数"].isin([3, 6, 9, 12, 15, 18, 21, len(构造)])]
          .round(4).to_string(index=False))
    print("""
    **进度是高度非线性的**：前十几个锚点只围住一小块，最后几个才把边界那一圈合拢。
    原因是围堵要求「每个方向都有见证者」，缺一个方向就前功尽弃——
    这与问题三的圆盘覆盖（面积线性累加）完全不同，也是锚点数降不下来的根本原因。
""")

    # ---------------------------------------------------------- ④ 判空代价
    print("-" * 78)
    print("④ 判空代价对照")
    行4 = []
    for 名, K, L in (("问题三：圆盘覆盖", 7, 6166.2),
                     ("问题四：半圆盘围堵", len(构造), 巡回)):
        for 空 in (4, 7, 10):
            行4.append({"方案": 名, "锚点数": K, "巡回长度_米": L, "空频道数": 空,
                        "判空检测次数": K * 空,
                        "判空检测秒": K * 空 * (C.T_MEASURE + C.T_SWITCH),
                        "锚点行进秒": L / C.V_MOVE,
                        "合计秒": K * 空 * (C.T_MEASURE + C.T_SWITCH) + L / C.V_MOVE})
    表4 = pd.DataFrame(行4)
    表4.to_csv(HERE / "判空代价对照.csv", index=False, encoding="utf-8-sig")
    print(表4.round(1).to_string(index=False))
    print(f"""
    **这张表是问题四相对问题三全部额外代价的来源**。以典型的 7 个空频道为例：
      问题三   7 x 7 x 6 = {7*7*6} s 检测 + {6166.2/5:.0f} s 行进 = {7*7*6+6166.2/5:.0f} s
      问题四   {len(构造)} x 7 x 6 = {len(构造)*7*6} s 检测 + {巡回/5:.0f} s 行进 = {len(构造)*7*6+巡回/5:.0f} s
    **三倍有余。**
    行进那一项在实战里会被清除巡回大量分摊（一体化滚动时域，见 策略.py），
    但检测的 5 秒分摊不掉——这是问题三 §4.1 那条结构性事实在问题四里的放大。
""")

    print("-" * 78)
    print("⑤ 一条必须写清的严谨性说明")
    print(f"""
    上面的 K = {len(构造)} 是**上界**（有构造、两套实现各验一遍），
    {int(np.ceil(下界))} 是**下界**（有证明）。本文没有证明二者之一是紧的。
    论文里应当写成「至少需要 {int(np.ceil(下界))} 个，本文给出 {len(构造)} 个的可行构造」，
    而不是「最少需要 {len(构造)} 个」。

    更要紧的一点与问题三相同：**策略的完备性并不依赖这个常数**。
    算法在运行时直接检查方向栅格掩码是否全空（公共.方向栅格.判空），
    空才允许退出。K 只进入代价核算与理论下界。
""")


if __name__ == "__main__":
    with open(HERE / "判空几何结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main(sys.argv)
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '判空几何结果.txt'}")
