"""问题一 · 变异测试：证明独立复核确实抓得住错误。

一个复核方法值不值钱，不看它在正确代码上是否通过，而看它在**错误代码**上是否失败。
本脚本往 求解算法.楔形半平面 注入若干典型 bug，逐一考察三层复核的反应。

历史教训：独立复核.py 的层二最初用 求解算法.楔形半平面 构造平面再交给 Qhull，
看似「独立实现」，实则与被测代码共用了构造步骤——本脚本证明它会漏检
「半角写错」「顶点偏移」这两类 bug，故层二已改为在复核脚本内独立按法向式推导。

依赖：生成算例.py 与 求解算法.py 的输出
输出：变异测试结果.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import HalfspaceIntersection, ConvexHull

import 求解算法 as 算法
from 独立复核 import 按定义在楔形内, 独立楔形半平面, 顶点集最大配对偏差

HERE = Path(__file__).resolve().parent
SEED = 20260910
样本数 = 60_000

原始楔形半平面 = 算法.楔形半平面
报告 = []


def P(s=""):
    报告.append(s)
    print(s)


汇总 = pd.read_csv(HERE / "算例汇总.csv", encoding="utf-8-sig")
行 = 汇总[汇总["算例"] == "B"].iloc[0]
检测表 = pd.read_csv(HERE / "算例B_检测点.csv", encoding="utf-8-sig")
S全 = 检测表[["x", "y"]].to_numpy()
TH全 = 检测表["示向度"].to_numpy()
DELTA = float(行["算法应用半角"])
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- 变异体
def 变异_半角缩小(S, theta_deg, delta_deg=None):
    """楔形半角误用 0.9° 而非 1°（把误差范围记错）。"""
    return 原始楔形半平面(S, theta_deg, 0.9)


def 变异_符号翻转(S, theta_deg, delta_deg=None):
    """第二个半平面的不等号方向写反。"""
    h1, h2 = 原始楔形半平面(S, theta_deg, DELTA)
    return [h1, (-h2[0], -h2[1], -h2[2])]


def 变异_顶点偏移(S, theta_deg, delta_deg=None):
    """楔形顶点误用 S + (5, 5)，例如坐标列读错或单位换算漏乘。"""
    return 原始楔形半平面((S[0] + 5.0, S[1] + 5.0), theta_deg, DELTA)


def 变异_角度反向(S, theta_deg, delta_deg=None):
    """把示向度理解成「干扰源指向检测点」，方向差 180°。"""
    return 原始楔形半平面(S, theta_deg + 180.0, DELTA)


def 变异_半角放大(S, theta_deg, delta_deg=None):
    """楔形半角误用 1.1°（例如误加了舍入裕度）。"""
    return 原始楔形半平面(S, theta_deg, 1.1)


变异体 = [
    ("无变异（对照）", 原始楔形半平面),
    ("半角 1°→0.9°", 变异_半角缩小),
    ("半角 1°→1.1°", 变异_半角放大),
    ("楔形顶点偏移 (5,5)", 变异_顶点偏移),
    ("半平面符号翻转", 变异_符号翻转),
    ("示向度反向 180°", 变异_角度反向),
]


# ---------------------------------------------------------------- 三层复核
def 层一_半平面单元检验(n=40_000):
    """随机取点，比对「按 arctan2 定义」与「按被测函数的两个不等式」。"""
    不符 = 0
    每点 = max(n // len(S全), 1)
    for S, th in zip(S全, TH全):
        采样 = np.asarray(S) + rng.uniform(-1500.0, 1500.0, size=(每点, 2))
        面 = 算法.楔形半平面(S, th, DELTA)
        for p in 采样:
            定义 = 按定义在楔形内(p, S, th, DELTA)
            不等式 = all(a * p[0] + b * p[1] <= c + 1e-9 for a, b, c in 面)
            不符 += (定义 != 不等式)
    return 不符 == 0, f"不一致 {不符}"


def 层二旧_共用被测代码(顶点):
    """反面教材：用被测的 楔形半平面 构造平面再交给 Qhull。"""
    try:
        平面 = np.array([[a, b, -c] for S, th in zip(S全, TH全)
                        for a, b, c in 算法.楔形半平面(S, th, DELTA)])
        hs = HalfspaceIntersection(平面, np.asarray(顶点, float).mean(axis=0))
        qh = hs.intersections[ConvexHull(hs.intersections).vertices]
        if len(qh) != len(顶点):
            return False, f"顶点数 {len(qh)}≠{len(顶点)}"
        return 顶点集最大配对偏差(顶点, qh) < 1e-6, "顶点吻合"
    except Exception as e:
        return False, type(e).__name__


def 层二新_独立推导(顶点):
    """现行做法：半平面由复核脚本按法向式独立推导。"""
    try:
        平面 = np.array([面 for S, th in zip(S全, TH全)
                        for 面 in 独立楔形半平面(S, th, DELTA)])
        hs = HalfspaceIntersection(平面, np.asarray(顶点, float).mean(axis=0))
        qh = hs.intersections[ConvexHull(hs.intersections).vertices]
        if len(qh) != len(顶点):
            return False, f"顶点数 {len(qh)}≠{len(顶点)}"
        return 顶点集最大配对偏差(顶点, qh) < 1e-6, "顶点吻合"
    except Exception as e:
        return False, type(e).__name__


def 层三_整体隶属度(顶点, 半宽倍数=3.0):
    """按楔形定义 vs 按求得多边形。"""
    顶 = np.asarray(顶点, dtype=float)
    d, _ = 算法.求直径([tuple(v) for v in 顶])
    采样 = 顶.mean(axis=0) + rng.uniform(-max(d, 1.0) * 半宽倍数,
                                        max(d, 1.0) * 半宽倍数, size=(样本数, 2))
    多 = [tuple(v) for v in 顶]
    按定义 = np.array([all(按定义在楔形内(p, S, th, DELTA)
                          for S, th in zip(S全, TH全)) for p in 采样])
    按多边形 = np.array([算法.点在多边形内(tuple(p), 多, 容差=1e-9) for p in 采样])
    不符 = int((按定义 != 按多边形).sum())
    return 不符 == 0, f"不一致 {不符}/{样本数}"


# ---------------------------------------------------------------- 执行
P("=" * 96)
P("变异测试：注入 bug 后各层复核的反应（算例 B）")
P("=" * 96)
P("  「通过」= 复核没报错。对变异体而言，通过即为**漏检**。")
P()
P(f"  {'变异体':<20} {'求解结果':<10} {'层一 构造':<12} "
  f"{'层二(旧) 共用代码':<18} {'层二(新) 独立推导':<18} {'层三 隶属度':<12}")
P("  " + "-" * 92)

for 名, fn in 变异体:
    算法.楔形半平面 = fn
    顶点, 状态 = 算法.求定位区域(S全, TH全, DELTA)
    if 状态 not in ("多边形", "线段", "单点"):
        P(f"  {名:<20} {'区域=' + 状态:<10} {'（求解阶段即暴露，无需复核）':<40}")
        continue
    一, 一说 = 层一_半平面单元检验()
    二旧, _ = 层二旧_共用被测代码(顶点)
    二新, _ = 层二新_独立推导(顶点)
    三, 三说 = 层三_整体隶属度(顶点)

    def 标(通过):
        if 名 == "无变异（对照）":
            return "通过" if 通过 else "★未通过★"
        return "★漏检★" if 通过 else "抓到"

    P(f"  {名:<20} {'多边形':<10} {标(一):<12} {标(二旧):<18} {标(二新):<18} {标(三):<12}")

算法.楔形半平面 = 原始楔形半平面

P()
P("=" * 96)
P("结论")
P("=" * 96)
P("  1. 层二若共用被测代码构造半平面，会漏检『半角写错』『顶点偏移』这两类构造 bug——")
P("     因为构造错了两边一起错，求交步骤再正确也发现不了。独立复核.py 已改为独立推导。")
P("  2. 层一直接比对被测函数与 arctan2 定义，是构造类 bug 的主要关口。")
P("  3. 层三从定义出发做整体隶属度检验，对区域形状的任何偏差都敏感。")
P("  4. 符号翻转、示向度反向这两类 bug 会让区域直接变空，在求解阶段即暴露，无需复核介入。")

P()
P("=" * 96)
P("层三的灵敏度：把正确多边形人为缩放，看多小的偏差能被抓到")
P("=" * 96)
顶点, _ = 算法.求定位区域(S全, TH全, DELTA)
顶 = np.asarray(顶点, dtype=float)
中心 = 顶.mean(axis=0)
for 缩放 in [0.999, 0.99, 0.95, 1.01, 1.05, 1.2]:
    变形 = [tuple(中心 + 缩放 * (v - 中心)) for v in 顶]
    通过, 说明 = 层三_整体隶属度(变形)
    P(f"  缩放至 {缩放:5.3f} 倍  ->  {'★漏检★' if 通过 else '抓到'}   {说明}")
P()
P(f"  采样 {样本数} 点即可分辨 1% 量级的形状偏差；独立复核.py 用 400000 点，余量充足。")

(HERE / "变异测试结果.txt").write_text("\n".join(报告), encoding="utf-8")
print()
print(f"报告已写入 {HERE / '变异测试结果.txt'}")
