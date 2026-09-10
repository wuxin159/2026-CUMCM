"""问题二 · 公共模型层（只算数，不作图，不写文件）。

被 理论验证.py / 候选区域.py / 策略仿真.py / 绘图.py 共同 import。

坐标一律用「局部标架」：把第一个检测点 S1 平移到原点、把它测得的示向度 theta1
旋转到 x 轴正向。于是
    真实干扰源 G = rho * (cos alpha, sin alpha)，其中 alpha = -eps1，|alpha| <= delta
    候选第二检测点 S2 = (a, b)，a 为沿示向线的前进量，b 为侧移量（可正可负）
所有几何量都在这个标架里算，最后再转回全局坐标（转回函数见 局部转全局）。

P1 已经把「楔形取交 -> 凸多边形 -> 直径 / 最小覆盖圆」做完并交叉验证过，
本文件直接 import，不重写。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P1"))
import 求解算法 as 算法          # noqa: E402  P1 的几何主力

# ---------------------------------------------------------------- 题目常量
R_AREA = 1800.0          # 目标区域半径，米（题目正文）
DELTA = 1.0              # 示向度误差半角，度（附录 2：误差在 [-1°,1°] 内）
R_RECV_MIN = 1000.0      # 有效接收半径下界，米（附录 2 第 2 条）
R_RECV_MAX = 1500.0      # 有效接收半径上界，米
R_NEAR = 5.0             # 近距离阈值，米，返回 near 不给示向度（附录 2 第 9 条）
R_CLEAR = 20.0           # 清除半径，米（附录 2 第 8 条）

V_MOVE = 5.0             # 机器狗速度，米/秒（附录 2 第 6 条）
T_MEASURE = 5.0          # 一次检测耗时，秒（附录 2 第 5 条）
T_SWITCH = 1.0           # 切换频道耗时，秒（附录 2 第 4 条）
T_CLEAR_OK = 5.0         # 清除成功耗时，秒（附录 2 第 8 条：定位 3 + 清除 2）
T_CLEAR_FAIL = 3.0       # 未发现耗时，秒

TAN_DELTA = float(np.tan(np.radians(DELTA)))


# ================================================================ 一、先验区域 Ω1
def 距离上界(S1=(0.0, 0.0), theta1=0.0, delta=DELTA, R=R_AREA):
    """一次检测之后，干扰源到 S1 的距离 rho 的确定上界。

    两条来源：
      ① 收得到信号 => rho <= R_eff <= R_RECV_MAX = 1500
      ② 干扰源必在目标区域内 => 沿楔形任一方向的射线不能超出圆 R
    取两者较小值。楔形内 alpha 取 [-delta, delta]，取其中最远的一条射线。
    """
    S1 = np.asarray(S1, dtype=float)
    最远 = 0.0
    for a in np.linspace(theta1 - delta, theta1 + delta, 21):
        u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        # 解 |S1 + t u| = R 的正根
        b = float(S1 @ u)
        c = float(S1 @ S1) - R * R
        判别式 = b * b - c
        if 判别式 <= 0:
            continue
        t = -b + np.sqrt(判别式)
        最远 = max(最远, max(t, 0.0))
    return float(min(R_RECV_MAX, 最远))


def 圆的外切半平面(圆心, 半径, 段数=180, 角范围=None):
    """用外切多边形逼近圆盘，返回若干 (a, b, c) 表示 a x + b y <= c。

    取「外切」而不是「内接」：得到的是圆盘的外逼近，
    所以裁出来的区域是真实区域的**超集**，宁可高估定位区域也不低估。
    """
    圆心 = np.asarray(圆心, dtype=float)
    低, 高 = (0.0, 360.0) if 角范围 is None else 角范围
    角 = np.radians(np.linspace(低, 高, 段数, endpoint=(角范围 is not None)))
    出 = []
    for t in 角:
        n = (np.cos(t), np.sin(t))
        出.append((n[0], n[1], 半径 + n[0] * 圆心[0] + n[1] * 圆心[1]))
    return 出


def 先验区域(S1=(0.0, 0.0), theta1=0.0, delta=DELTA,
             rho_lo=R_NEAR, rho_hi=None, R=R_AREA, 弧段数=6):
    """第一次检测之后干扰源的可行集 Ω1（凸多边形顶点表）。

        Ω1 = 楔形 W1  ∩  {rho_lo <= |P-S1| <= rho_hi}  ∩  目标圆盘

    三项约束的出处：楔形来自示向度读数；rho <= rho_hi 来自「收得到信号」与
    「干扰源在目标区域内」；rho > rho_lo = 5 来自「返回的是 direction 而不是 near」。

    r >= rho_lo 本身非凸，但在 2delta = 2° 的窄扇形上，弦与弧的矢高只有
    rho_lo (1 - cos delta) = 0.8 毫米，用弦所在半平面代替，且方向上取的是
    **更宽松**的一侧（保留了 [rho_lo cos delta, rho_lo) 这层薄壳），仍是超集。

    弧段数 = 6 时外圈用 6 条切线逼近 2° 的圆弧，外凸误差
    rho_hi (1/cos(delta/6) - 1) ≈ 2.5 厘米，且同样偏向超集一侧；
    独立复核.py 中与 弧段数 = 240 的结果逐点对比过。
    """
    S1 = np.asarray(S1, dtype=float)
    if rho_hi is None:
        rho_hi = 距离上界(S1, theta1, delta, R)

    BOX = 10.0 * R
    多边形 = [(-BOX, -BOX), (BOX, -BOX), (BOX, BOX), (-BOX, BOX)]

    半平面 = list(算法.楔形半平面(S1, theta1, delta))
    # 外圈 rho <= rho_hi：以 S1 为心的圆，只需覆盖楔形张角那一小段
    半平面 += 圆的外切半平面(S1, rho_hi, 段数=弧段数,
                             角范围=(theta1 - delta - 1e-9, theta1 + delta + 1e-9))
    # 内圈 rho >= rho_lo：弦所在半平面，法向指向 -u(theta1)
    u = np.array([np.cos(np.radians(theta1)), np.sin(np.radians(theta1))])
    c = rho_lo * np.cos(np.radians(delta))
    半平面.append((-u[0], -u[1], -(c + float(S1 @ u))))
    # 目标区域圆盘
    半平面 += 圆的外切半平面((0.0, 0.0), R, 段数=360)

    for a, b, cc in 半平面:
        多边形 = 算法.裁剪(多边形, a, b, cc)
        if not 多边形:
            return []
    return 算法.去冗余(多边形)


def 定位区域(Ω1, S2, theta2, delta=DELTA):
    """第二次检测之后的定位区域 L = Ω1 ∩ W2（凸多边形顶点表）。"""
    多边形 = list(Ω1)
    for a, b, c in 算法.楔形半平面(S2, theta2, delta):
        多边形 = 算法.裁剪(多边形, a, b, c)
        if not 多边形:
            return []
    return 算法.去冗余(多边形)


def 求直径快(顶点):
    """与 P1 的 求直径 等价，只是把 O(m^2) 的两重 Python 循环换成一次 numpy 广播。

    本问要在上万个候选点 x 几十个距离节点上重复调用，纯 Python 版会慢两个量级。
    等价性在 独立复核.py 中对随机多边形逐点比对。
    """
    P = np.asarray(顶点, dtype=float)
    if len(P) == 1:
        return 0.0, (tuple(P[0]), tuple(P[0]))
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    i, j = np.unravel_index(int(np.argmax(D)), D.shape)
    return float(D[i, j]), (tuple(P[i]), tuple(P[j]))


def 最小覆盖圆快(顶点):
    """先试「以直径点对为直径的圆」，盖不住再退回 P1 的暴力枚举。

    合法性：恒有 Rmin >= d/2（直径点对本身就要被盖住），所以只要以 AB 为直径的
    圆能盖住全部顶点，它就**必然**是最小覆盖圆，无需再枚举。P1 §4.1 的
    覆盖判定说明这一分支并非总能成立，故保留了完整的回退路径。
    """
    P = np.asarray(顶点, dtype=float)
    if len(P) == 1:
        return (tuple(P[0]), 0.0)
    d, (A, B) = 求直径快(P)
    A, B = np.asarray(A), np.asarray(B)
    c = 0.5 * (A + B)
    if np.all(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) <= d / 2 + 1e-9):
        return (tuple(c), float(d / 2))
    return 算法.最小覆盖圆([tuple(p) for p in P])


def 区域指标(顶点):
    """返回 (直径, 最小覆盖圆半径, 最小覆盖圆心, 面积)。空区域返回 nan。"""
    if not 顶点:
        return np.nan, np.nan, (np.nan, np.nan), np.nan
    if len(顶点) == 1:
        return 0.0, 0.0, tuple(顶点[0]), 0.0
    直径, _ = 求直径快(顶点)
    圆心, 半径 = 最小覆盖圆快(顶点)
    面积 = 算法.多边形面积(顶点) if len(顶点) >= 3 else 0.0
    return float(直径), float(半径), tuple(圆心), float(面积)


# ================================================================ 二、闭式公式
def 方位角(起点, 终点):
    d = np.asarray(终点, dtype=float) - np.asarray(起点, dtype=float)
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def 近似直径(rho, a, b, delta=DELTA):
    """两楔形交（平行四边形）长对角线的闭式长度。P1 §4.2 公式的坐标化形式。

        d = (2 tan delta / |b|) * sqrt(s^2+b^2) * sqrt((rho+|s|)^2 + b^2),  s = rho - a

    推导：源附近两楔形各退化为宽 w_i = 2 r_i tan delta 的条带，交会角 phi
    满足 sin phi = |b|/r2、cos phi = |s|/r2，代入
    d = sqrt(w1^2+w2^2+2 w1 w2 |cos phi|)/sin phi 即得。b -> 0 时发散（共线，无交会）。
    """
    rho = np.asarray(rho, dtype=float)
    b = np.asarray(b, dtype=float)
    s = rho - np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.hypot(s, b)
        d = 2.0 * np.tan(np.radians(delta)) / np.abs(b) * r2 * np.hypot(rho + np.abs(s), b)
    return d


def 最优侧移(rho, a):
    """给定前进量 a（即给定沿轴错位 |s| = |rho - a|），使近似直径最小的侧移量。

        b* = sqrt(|s| (rho + |s|))
    对 ln d 关于 b 求导置零解得，是 |s| 与 rho+|s| 的几何平均。
    """
    s = np.abs(np.asarray(rho, dtype=float) - np.asarray(a, dtype=float))
    return np.sqrt(s * (np.asarray(rho, dtype=float) + s))


def 最优直径(rho, a, delta=DELTA):
    """取 b = b* 时的最小可能直径：d_min = 2 tan delta (rho + 2|rho - a|)。

    含义：以 2 rho tan delta（正横站位的理想值）为基准，沿轴每错位 1 米，
    直径线性增加 4 tan delta ≈ 0.0698 米。
    """
    s = np.abs(np.asarray(rho, dtype=float) - np.asarray(a, dtype=float))
    return 2.0 * np.tan(np.radians(delta)) * (np.asarray(rho, dtype=float) + 2.0 * s)


def 交会角(rho, a, b):
    """干扰源处两条示向线的夹角 phi，单位度，取 [0,90] 的锐角化值前的原始值。"""
    s = np.asarray(rho, dtype=float) - np.asarray(a, dtype=float)
    return np.degrees(np.arctan2(np.abs(b), s)) % 180.0


# ================================================================ 三、接收概率
def 保证收信(rho, a, b):
    """第二次检测是否**必定**收到信号（不依赖未知的有效接收半径 R_eff）。

    已知 rho <= R_eff <= 1500，故 R_eff >= max(1000, rho)。
    于是 |S2 - G| <= max(1000, rho) 就保证收得到。
    特例：|S2-G| <= rho（比第一个检测点更靠近干扰源）时自动满足。
    """
    rho = np.asarray(rho, dtype=float)
    r2 = np.hypot(rho - np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return r2 <= np.maximum(R_RECV_MIN, rho)


def 收信概率(rho, a, b):
    """在 R_eff ~ U[1000,1500] 且已知 R_eff >= rho 的条件下收到信号的概率。"""
    rho = np.asarray(rho, dtype=float)
    r2 = np.hypot(rho - np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    下 = np.maximum(R_RECV_MIN, rho)                 # R_eff 的条件下界
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(r2 <= 下, 1.0,
                     np.where(r2 >= R_RECV_MAX, 0.0,
                              (R_RECV_MAX - r2) / np.maximum(R_RECV_MAX - 下, 1e-12)))
    return p


# ================================================================ 四、rho 的先验
def 距离先验(rho, rho_lo=R_NEAR, rho_hi=R_RECV_MAX):
    """已在 S1 收到信号的条件下，rho 的后验密度（未归一化）。

        p(rho) ∝ rho * P(R_eff >= rho),   R_eff ~ U[1000, 1500]

    第一项 rho 来自「干扰源在目标区域内均匀分布」的面积元；
    第二项是「这一距离上还能收得到」的概率。
    题目未给 R_eff 的分布，均匀是我们的假设，须在论文假设一节声明，
    并在灵敏度分析里换成 rho 上的均匀分布重跑。
    """
    rho = np.asarray(rho, dtype=float)
    可收 = np.clip((R_RECV_MAX - np.maximum(rho, R_RECV_MIN)) /
                   (R_RECV_MAX - R_RECV_MIN), 0.0, 1.0)
    密度 = rho * 可收
    return np.where((rho >= rho_lo) & (rho <= rho_hi), 密度, 0.0)


def 距离节点(n=40, rho_lo=R_NEAR, rho_hi=R_RECV_MAX, 先验="题设"):
    """rho 的离散节点与归一化权重，供期望值积分使用（中点法）。"""
    边 = np.linspace(rho_lo, rho_hi, n + 1)
    节点 = 0.5 * (边[:-1] + 边[1:])
    if 先验 == "题设":
        w = 距离先验(节点, rho_lo, rho_hi)
    elif 先验 == "均匀":
        w = np.ones_like(节点)
    else:
        raise ValueError(f"未知先验：{先验}")
    return 节点, w / w.sum()


# ================================================================ 五、代价模型
def 行进时间(距离):
    return np.asarray(距离, dtype=float) / V_MOVE


def 绕行(rho, a, b):
    """相对于「沿示向线直奔干扰源」多走的路程（米）。

    直奔需要 rho；经 S2 需要 |S1 S2| + |S2 G|。差额就是侧移换来的代价。
    """
    rho = np.asarray(rho, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.hypot(a, b) + np.hypot(rho - a, b) - rho


def 收尾代价(D2, r2, delta=DELTA):
    """第二次检测之后，把区域从直径 D2 收到 20 米以内并完成清除的补充路程。

    收尾动作：走向定位区域，在其中心的**正横**方向、侧移 Lambda = D2/2 处补测一次，
    再走到区域中心清除。相对于「直奔区域中心」的额外路程约为

        Lambda + Lambda^2 / (2 r2)

    第一项是补测点回到区域中心的一段，第二项是绕到正横位置的勾股余量。
    补测后区域尺度约 2 tan delta * (r3^2 + Lambda^2)/Lambda，在 Lambda = r3 时
    取到 4 Lambda tan delta，只要 Lambda <= 286 米就能压到 20 米以内。
    """
    Λ = 0.5 * np.asarray(D2, dtype=float)
    r2 = np.maximum(np.asarray(r2, dtype=float), 1.0)
    return Λ + Λ * Λ / (2.0 * r2)


# ================================================================ 六、坐标变换
def 局部转全局(点, S1, theta1):
    """把局部标架（S1 在原点、theta1 指向 x 轴正向）中的点转回全局坐标。"""
    点 = np.atleast_2d(np.asarray(点, dtype=float))
    t = np.radians(theta1)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    出 = 点 @ R.T + np.asarray(S1, dtype=float)
    return 出 if 出.shape[0] > 1 else 出[0]


def 全局转局部(点, S1, theta1):
    点 = np.atleast_2d(np.asarray(点, dtype=float))
    t = np.radians(theta1)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    出 = (点 - np.asarray(S1, dtype=float)) @ R.T
    return 出 if 出.shape[0] > 1 else 出[0]
