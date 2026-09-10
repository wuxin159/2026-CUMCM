"""问题一 · 独立复核：用与求解算法无关的方法交叉验证定位区域。

只重算直径是不够的——顶点本身是不是正确的半平面交，才是承重的那一步。
更要紧的是：复核代码**不能调用被测代码**，否则两边一起错还会对得严丝合缝。
本脚本据此分三层，每层都独立于 求解算法.py 的实现：

    层一  半平面构造的单元检验
          随机取点，比对「按 atan2 定义判定在不在楔形内」与
          「按 求解算法.楔形半平面 给出的两个不等式判定」。
          这一层直接测被测函数本身，是变异测试中唯一能抓住
          「半角写错」「顶点偏移」两类 bug 的关口。

    层二  求交的独立检验
          用**本脚本自己按法向式重新推导**的半平面（不调用被测代码）
          交给 scipy.spatial.HalfspaceIntersection（Qhull）求交，比对顶点与直径。

    层三  整体隶属度检验
          在足够大的框内均匀采样，比对「按楔形定义判定」与「按求得多边形判定」。

配套的 变异测试.py 会往求解算法里注入 bug，用以证明上述各层确实抓得住错误。

依赖：生成算例.py 与 求解算法.py 的输出
输出：独立复核结果.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import HalfspaceIntersection, ConvexHull

import 求解算法 as 算法

HERE = Path(__file__).resolve().parent
SEED = 20260910
单元检验点数 = 200_000
隶属检验点数 = 400_000
采样半宽倍数 = 3.0        # 采样框半宽 = 该倍数 × 直径，取大以免真实区域超出框外

报告 = []


def P(s=""):
    报告.append(s)
    print(s)


def 标题(s):
    P()
    P("=" * 78)
    P(s)
    P("=" * 78)


# ---------------------------------------------------------------- 独立实现
def 按定义在楔形内(点, S, theta_deg, delta_deg):
    """按题面定义判定：点相对 S 的方位角是否落在 [θ-δ, θ+δ] 内。

    只用 arctan2，不涉及任何半平面、叉积或被测代码。
    """
    v = np.asarray(点, dtype=float) - np.asarray(S, dtype=float)
    a = np.degrees(np.arctan2(v[1], v[0]))
    return abs((a - theta_deg + 180.0) % 360.0 - 180.0) <= delta_deg


def 独立楔形半平面(S, theta_deg, delta_deg):
    """本脚本自己推导的楔形半平面，采用**法向式**，与被测代码的叉积式不同路。

    取两条边界方向的内法向：
        n_lo = 方向(θ - δ + 90°)，  n_hi = 方向(θ + δ - 90°)
    则 P 在楔形内 <=> (P - S)·n_lo >= 0 且 (P - S)·n_hi >= 0。
    校验：P 在中心射线上时 (P-S)·n = sin δ > 0，方向正确。
    返回 scipy 要求的 [a, b, c] 形式，含义为 a·x + b·y + c <= 0。
    """
    S = np.asarray(S, dtype=float)
    面 = []
    for 角 in (theta_deg - delta_deg + 90.0, theta_deg + delta_deg - 90.0):
        n = np.array([np.cos(np.radians(角)), np.sin(np.radians(角))])
        面.append([-n[0], -n[1], float(np.dot(n, S))])
    return 面


def 顶点集最大配对偏差(甲, 乙):
    """两组顶点按最近邻两两配对后的最大距离，与顺序无关。"""
    剩 = list(map(tuple, 乙))
    最大 = 0.0
    for p in 甲:
        d = [np.hypot(p[0] - q[0], p[1] - q[1]) for q in 剩]
        i = int(np.argmin(d))
        最大 = max(最大, d[i])
        剩.pop(i)
    return 最大


# ---------------------------------------------------------------- 主流程
汇总 = pd.read_csv(HERE / "算例汇总.csv", encoding="utf-8-sig")
求解 = pd.read_csv(HERE / "求解汇总.csv", encoding="utf-8-sig")
rng = np.random.default_rng(SEED)


标题("层一  半平面构造的单元检验（直接测 求解算法.楔形半平面）")
P("  随机取点，比对「按 arctan2 定义」与「按被测函数给出的两个不等式」。")
P("  这一层不经过求交，故能单独暴露半角、顶点、角度约定一类的构造错误。")
P()
层一全通过 = True
for _, 行 in 汇总.iterrows():
    名 = str(行["算例"])
    delta = float(行["算法应用半角"])
    检测表 = pd.read_csv(HERE / f"算例{名}_检测点.csv", encoding="utf-8-sig")
    S全 = 检测表[["x", "y"]].to_numpy()
    TH全 = 检测表["示向度"].to_numpy()
    不符总数 = 0
    for S, th in zip(S全, TH全):
        # 在检测点周围一个大范围内采样，覆盖楔形内外
        采样 = np.asarray(S) + rng.uniform(-1500.0, 1500.0,
                                          size=(单元检验点数 // len(S全), 2))
        面 = 算法.楔形半平面(S, th, delta)
        for p in 采样:
            定义 = 按定义在楔形内(p, S, th, delta)
            不等式 = all(a * p[0] + b * p[1] <= c + 1e-9 for a, b, c in 面)
            if 定义 != 不等式:
                不符总数 += 1
    通过 = 不符总数 == 0
    层一全通过 &= 通过
    P(f"  算例 {名}：采样 {单元检验点数} 点，判定不一致 {不符总数} 个"
      f"  ->  {'通过' if 通过 else '未通过'}")


标题("层二  求交的独立检验（Qhull，半平面由本脚本按法向式独立推导）")
P("  注意：这里刻意不调用 求解算法.楔形半平面，否则构造错误会被两边同时继承。")
层二全通过 = True
for _, 行 in 汇总.iterrows():
    名 = str(行["算例"])
    delta = float(行["算法应用半角"])
    检测表 = pd.read_csv(HERE / f"算例{名}_检测点.csv", encoding="utf-8-sig")
    我的顶点 = pd.read_csv(HERE / f"算例{名}_定位区域顶点.csv",
                          encoding="utf-8-sig")[["x", "y"]].to_numpy()
    S全 = 检测表[["x", "y"]].to_numpy()
    TH全 = 检测表["示向度"].to_numpy()
    解 = 求解[求解["算例"] == 名].iloc[0]

    平面 = np.array([面 for S, th in zip(S全, TH全)
                    for 面 in 独立楔形半平面(S, th, delta)])
    内点 = 我的顶点.mean(axis=0)          # 只用作 Qhull 的可行内点，不影响求交结果
    hs = HalfspaceIntersection(平面, 内点)
    qh = hs.intersections[ConvexHull(hs.intersections).vertices]
    同数 = len(qh) == len(我的顶点)
    偏差 = 顶点集最大配对偏差(我的顶点, qh) if 同数 else float("nan")
    d_qh, _ = 算法.求直径([tuple(p) for p in qh])
    d差 = abs(d_qh - float(解["直径"]))
    通过 = 同数 and 偏差 < 1e-6 and d差 < 1e-6
    层二全通过 &= 通过
    P()
    P(f"  算例 {名}（n = {len(S全)}）")
    P(f"    顶点数 {len(qh)} vs 求解算法 {len(我的顶点)}")
    if 同数:
        P(f"    顶点逐一配对最大偏差 {偏差:.3e} m")
    P(f"    直径 {d_qh:.9f} vs {float(解['直径']):.9f}，差 {d差:.3e} m  ->  "
      f"{'通过' if 通过 else '未通过'}")


标题("层三  整体隶属度检验（按楔形定义 vs 按求得多边形）")
P(f"  采样框半宽取 {采样半宽倍数:g} × 直径，取大以免真实区域超出框外而漏检。")
P()
层三全通过 = True
for _, 行 in 汇总.iterrows():
    名 = str(行["算例"])
    delta = float(行["算法应用半角"])
    检测表 = pd.read_csv(HERE / f"算例{名}_检测点.csv", encoding="utf-8-sig")
    源表 = pd.read_csv(HERE / f"算例{名}_干扰源.csv", encoding="utf-8-sig")
    我的顶点 = pd.read_csv(HERE / f"算例{名}_定位区域顶点.csv",
                          encoding="utf-8-sig")[["x", "y"]].to_numpy()
    S全 = 检测表[["x", "y"]].to_numpy()
    TH全 = 检测表["示向度"].to_numpy()
    G = (float(源表["x"].iloc[0]), float(源表["y"].iloc[0]))
    解 = 求解[求解["算例"] == 名].iloc[0]

    中心 = 我的顶点.mean(axis=0)
    幅 = float(解["直径"]) * 采样半宽倍数
    采样 = 中心 + rng.uniform(-幅, 幅, size=(隶属检验点数, 2))
    多边形 = [tuple(v) for v in 我的顶点]
    按定义 = np.array([all(按定义在楔形内(p, S, th, delta)
                          for S, th in zip(S全, TH全)) for p in 采样])
    按多边形 = np.array([算法.点在多边形内(tuple(p), 多边形, 容差=1e-9) for p in 采样])
    不符 = int((按定义 != 按多边形).sum())
    区内 = int(按定义.sum())
    通过 = 不符 == 0
    层三全通过 &= 通过
    P(f"  算例 {名}：采样 {隶属检验点数} 点（落在区域内 {区内} 个），"
      f"两种判定不一致 {不符} 个  ->  {'通过' if 通过 else '未通过'}")

    # 顺带复核覆盖判定与含真值
    A = np.array([解["Ax"], 解["Ay"]])
    B = np.array([解["Bx"], 解["By"]])
    圆心 = (A + B) / 2
    r = float(解["直径"]) / 2
    超出 = max(float(np.hypot(*(np.array(v) - 圆心))) - r for v in 我的顶点)
    点积 = max(float(np.dot(A - np.array(v), B - np.array(v))) for v in 我的顶点)
    P(f"            覆盖：最大超出 {超出:.6f} m（表中 {float(解['最大超出量']):.6f}），"
      f"最大点积 {点积:+.4f} m² -> {'不能覆盖' if 点积 > 0 else '能覆盖'}"
      f"（表中 {'不能覆盖' if not 解['能否覆盖'] else '能覆盖'}）")
    P(f"            含真值：G 在区域内 {算法.点在多边形内(tuple(G), 多边形)}"
      f"（表中 {bool(解['区域含真值G'])}）")


标题("层四  退化分支与有界性判据自测")
for 名, 顶点 in [("单点", [(1.0, 2.0)]), ("线段", [(0.0, 0.0), (3.0, 4.0)])]:
    d, 点对 = 算法.求直径(顶点)
    外接 = 算法.最小覆盖圆(顶点)
    能, _, 超出 = 算法.覆盖判定(顶点, *点对)
    P(f"  {名}：直径 {d}，最小覆盖圆 {外接}，覆盖判定 {能}（超出 {超出}）")
P("  两种退化形态的直径都有定义（点为 0、线段为其长），覆盖判定平凡成立，")
P("  故求解算法照常计算，不作特例跳过。")
P()
用例 = [
    ("n = 1", np.array([[0.0, 0.0]]), np.array([30.0]), "无界"),
    ("两示向度相差 0.5°（区间重叠）", np.array([[0.0, 0.0], [10.0, 0.0]]),
     np.array([30.0, 30.5]), "无界"),
    ("两示向度相差 2.5°（区间分离）", np.array([[0.0, 0.0], [500.0, 0.0]]),
     np.array([30.0, 32.5]), "多边形"),
    ("两示向度相差 90°", np.array([[0.0, 0.0], [1000.0, 0.0]]),
     np.array([45.0, 135.0]), "多边形"),
]
判据全通过 = True
for 说明, S, TH, 期望 in 用例:
    _, 状态 = 算法.求定位区域(S, TH, 1.0)
    符 = 状态 == 期望
    判据全通过 &= 符
    P(f"  {说明:30s} -> 判定「{状态}」，期望「{期望}」{'  通过' if 符 else '  未通过'}")


标题("总结")
for 名, 通过 in [("层一 半平面构造", 层一全通过), ("层二 独立求交", 层二全通过),
                ("层三 整体隶属度", 层三全通过), ("层四 退化与有界性", 判据全通过)]:
    P(f"  {名:16s} {'全部通过' if 通过 else '存在未通过项'}")
P()
P("  各层的有效性由 变异测试.py 给出证据：层一与层三能抓住注入的构造类 bug，")
P("  而若层二共用被测代码构造半平面，则会漏检——本脚本已改为独立推导。")


(HERE / "独立复核结果.txt").write_text("\n".join(报告), encoding="utf-8")
print()
print(f"报告已写入 {HERE / '独立复核结果.txt'}")
