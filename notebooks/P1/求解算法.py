"""问题一 · 第 3 步：求解算法（只算数，不作图）。

流程：楔形拆成两个半平面 -> 半平面交得凸多边形 -> 求直径 -> 覆盖判定。

一次检测给出的不是一条射线，而是顶点在 S_i、中心方向 θ_i、半角 δ 的楔形：
    W_i = {P : |∠(P - S_i) - θ_i| <= δ}
由于 2δ = 2° 远小于 180°，该楔形恰为两个半平面之交，于是
    n 次检测 = 2n 个半平面，定位区域 L = ∩ W_i 必为凸多边形，边数至多 2n。

依赖：生成算例.py 的输出
输出：
    求解汇总.csv                 直径、覆盖判定、最小覆盖圆等，供求解绘图.py 使用
    算例{名}_定位区域顶点.csv     定位区域顶点表
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
输出目录 = HERE

R_AREA = 1800.0          # 目标区域半径，米
DELTA = 1.0              # 楔形半角，度。问题一取精确 1°；对接模拟器时应取 1.005°
BOX = 10.0 * R_AREA      # 初始有界化方框的半边长
EPS = 1e-9


# ============================================================ 几何基础
def 裁剪(多边形, a, b, c):
    """保留满足 a*x + b*y <= c 的部分（Sutherland-Hodgman 半平面裁剪）。"""
    结果 = []
    n = len(多边形)
    for i in range(n):
        p, q = 多边形[i], 多边形[(i + 1) % n]
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        if dp <= EPS:
            结果.append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            t = dp / (dp - dq)
            结果.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return 结果


def 楔形半平面(S, theta_deg, delta_deg=DELTA):
    """顶点 S、中心方向 theta、半角 delta 的楔形，返回其两个半平面 (a, b, c)。

    半角远小于 90°，故楔形恰为两半平面之交：
        u_lo x (P - S) >= 0   且   u_hi x (P - S) <= 0
    化为 a*x + b*y <= c 的形式。
    """
    lo = np.radians(theta_deg - delta_deg)
    hi = np.radians(theta_deg + delta_deg)
    u_lo = (np.cos(lo), np.sin(lo))
    u_hi = (np.cos(hi), np.sin(hi))
    h1 = (u_lo[1], -u_lo[0], u_lo[1] * S[0] - u_lo[0] * S[1])
    h2 = (-u_hi[1], u_hi[0], -u_hi[1] * S[0] + u_hi[0] * S[1])
    return [h1, h2]


def 去冗余(多边形, 长度容差=1e-7, 共线容差=1e-9):
    """去掉重合点与共线中间点，否则求直径与覆盖判定会出现零长度边。"""
    点 = []
    for p in 多边形:
        if not 点 or np.hypot(p[0] - 点[-1][0], p[1] - 点[-1][1]) > 长度容差:
            点.append(p)
    if len(点) > 1 and np.hypot(点[0][0] - 点[-1][0], 点[0][1] - 点[-1][1]) <= 长度容差:
        点.pop()
    if len(点) < 3:
        return 点
    保留 = []
    m = len(点)
    for i in range(m):
        前, 当, 后 = 点[(i - 1) % m], 点[i], 点[(i + 1) % m]
        叉 = (当[0] - 前[0]) * (后[1] - 前[1]) - (当[1] - 前[1]) * (后[0] - 前[0])
        基 = max(np.hypot(当[0] - 前[0], 当[1] - 前[1]),
                 np.hypot(后[0] - 当[0], 后[1] - 当[1]), 1.0)
        if abs(叉) / 基 > 共线容差:
            保留.append(当)
    return 保留 if len(保留) >= 3 else 点


def 求定位区域(检测点, 示向度, delta_deg=DELTA):
    """n 个楔形取交，返回 (顶点表, 状态)。状态为 正常 / 空 / 无界。

    空  ：读数互相矛盾（读数异常，或检测期间干扰源已被清除），必须报错。
    无界：示向度近乎平行或 n=1，需用目标区域截断后才有有限直径。
    """
    多边形 = [(-BOX, -BOX), (BOX, -BOX), (BOX, BOX), (-BOX, BOX)]
    for S, th in zip(检测点, 示向度):
        for a, b, c in 楔形半平面(S, th, delta_deg):
            多边形 = 裁剪(多边形, a, b, c)
            if not 多边形:
                return [], "空"
    顶点 = 去冗余(多边形)
    if not 顶点:
        return [], "空"
    if max(max(abs(p[0]), abs(p[1])) for p in 顶点) > BOX * (1 - 1e-6):
        return 顶点, "无界"
    return 顶点, "正常"


def 求直径(顶点):
    """凸多边形的直径必在顶点对取到。

    依据：固定 P 时 Q -> |PQ| 是凸函数，凸函数在凸多边形上的最大值必在顶点取到；
    对 P 再作一次同样论证即可。复杂度 O(m^2)；m 很小，无需旋转卡壳。
    """
    最大, 点对 = 0.0, None
    for p, q in combinations(顶点, 2):
        d = float(np.hypot(p[0] - q[0], p[1] - q[1]))
        if d > 最大:
            最大, 点对 = d, (p, q)
    return 最大, 点对


def 覆盖判定(顶点, A, B):
    """以 AB 为直径的圆能否覆盖多边形。

    泰勒斯定理：V 在圆内或圆上 <=> ∠AVB >= 90° <=> (A-V)·(B-V) <= 0。
    写成点积可避免反三角函数。返回 (能否覆盖, 最坏顶点, 最大超出量)。
    """
    A, B = np.asarray(A), np.asarray(B)
    圆心 = (A + B) / 2.0
    半径 = float(np.hypot(*(A - B))) / 2.0
    最坏, 超出 = None, -np.inf
    能覆盖 = True
    for V in 顶点:
        V = np.asarray(V)
        if float(np.dot(A - V, B - V)) > 1e-7:
            能覆盖 = False
        e = float(np.hypot(*(V - 圆心))) - 半径
        if e > 超出:
            超出, 最坏 = e, tuple(V)
    return 能覆盖, 最坏, 超出


def 最小覆盖圆(顶点):
    """暴力枚举点对（作直径）与三点（作外接圆），顶点数很少，代价可忽略。"""
    最优 = None
    for p, q in combinations(顶点, 2):
        c = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        r = np.hypot(p[0] - q[0], p[1] - q[1]) / 2
        if all(np.hypot(v[0] - c[0], v[1] - c[1]) <= r + 1e-7 for v in 顶点):
            if 最优 is None or r < 最优[1]:
                最优 = (c, r)
    for p, q, s in combinations(顶点, 3):
        ax, ay = q[0] - p[0], q[1] - p[1]
        bx, by = s[0] - p[0], s[1] - p[1]
        分母 = 2 * (ax * by - ay * bx)
        if abs(分母) < 1e-12:
            continue
        a2, b2 = ax * ax + ay * ay, bx * bx + by * by
        ux = (by * a2 - ay * b2) / 分母
        uy = (ax * b2 - bx * a2) / 分母
        c = (p[0] + ux, p[1] + uy)
        r = float(np.hypot(ux, uy))
        if all(np.hypot(v[0] - c[0], v[1] - c[1]) <= r + 1e-7 for v in 顶点):
            if 最优 is None or r < 最优[1]:
                最优 = (c, r)
    return 最优


def 点在多边形内(点, 顶点, 容差=1e-7):
    """凸多边形（顶点顺逆时针均可）的包含判定，用于校验区域是否含真值 G。"""
    符号 = []
    m = len(顶点)
    for i in range(m):
        p, q = 顶点[i], 顶点[(i + 1) % m]
        叉 = (q[0] - p[0]) * (点[1] - p[1]) - (q[1] - p[1]) * (点[0] - p[0])
        if abs(叉) > 容差:
            符号.append(叉 > 0)
    return len(set(符号)) <= 1


def 多边形面积(顶点):
    s = 0.0
    m = len(顶点)
    for i in range(m):
        x1, y1 = 顶点[i]
        x2, y2 = 顶点[(i + 1) % m]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


# ============================================================ 主流程
def main():
    汇总表 = pd.read_csv(输出目录 / "算例汇总.csv", encoding="utf-8-sig")
    结果 = []

    for _, 行 in 汇总表.iterrows():
        名 = str(行["算例"])
        检测表 = pd.read_csv(输出目录 / f"算例{名}_检测点.csv", encoding="utf-8-sig")
        源表 = pd.read_csv(输出目录 / f"算例{名}_干扰源.csv", encoding="utf-8-sig")
        检测点 = 检测表[["x", "y"]].to_numpy()
        示向度 = 检测表["示向度"].to_numpy()
        G = (float(源表["x"].iloc[0]), float(源表["y"].iloc[0]))
        delta = float(行["算法应用半角"])

        顶点, 状态 = 求定位区域(检测点, 示向度, delta)
        if 状态 != "正常":
            print(f"算例 {名}：定位区域{状态}，跳过")
            结果.append({"算例": 名, "n": len(检测点), "状态": 状态})
            continue

        直径, 点对 = 求直径(顶点)
        能覆盖, 最坏, 超出 = 覆盖判定(顶点, *点对)
        外接 = 最小覆盖圆(顶点)
        含G = 点在多边形内(G, 顶点)

        pd.DataFrame(顶点, columns=["x", "y"]).to_csv(
            输出目录 / f"算例{名}_定位区域顶点.csv", index=False, encoding="utf-8-sig")

        结果.append({
            "算例": 名,
            "n": len(检测点),
            "状态": 状态,
            "顶点数": len(顶点),
            "直径": 直径,
            "面积": 多边形面积(顶点),
            "Ax": 点对[0][0], "Ay": 点对[0][1],
            "Bx": 点对[1][0], "By": 点对[1][1],
            "以直径为直径的圆半径": 直径 / 2,
            "能否覆盖": 能覆盖,
            "最大超出量": 超出,
            "圆外顶点x": 最坏[0] if not 能覆盖 else np.nan,
            "圆外顶点y": 最坏[1] if not 能覆盖 else np.nan,
            "最小覆盖圆x": 外接[0][0],
            "最小覆盖圆y": 外接[0][1],
            "最小覆盖圆半径": 外接[1],
            "最小覆盖圆比值": 外接[1] / (直径 / 2),
            "Jung上界": 直径 / np.sqrt(3),
            "Jung比值上界": 2 / np.sqrt(3),
            "区域含真值G": 含G,
        })

    汇总 = pd.DataFrame(结果)
    汇总.to_csv(输出目录 / "求解汇总.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    print("求解汇总")
    print(汇总[["算例", "n", "顶点数", "直径", "面积", "能否覆盖",
                "最大超出量", "最小覆盖圆半径", "最小覆盖圆比值",
                "Jung上界", "区域含真值G"]].round(6).to_string(index=False))
    print()
    print(f"结果已写入 {输出目录}")


if __name__ == "__main__":
    main()
