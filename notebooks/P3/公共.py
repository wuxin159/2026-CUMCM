"""问题三 · 公共模型层（只算数，不作图，不写文件）。

被 覆盖锚点.py / 模拟器.py / 策略.py / 参数搜索.py / 策略对照.py /
灵敏度.py / 独立复核.py / 作图数据.py 共同 import。

问题三与前两问的差别，一句话：**从「一个源、检测点外生或只选一个」变成
「10~16 个源、频道未知、路径与检测序列全部内生」**。于是要维护的不再是
一个定位区域，而是 20 个频道各自的**集员估计状态**（本文件的 `频道信念`），
以及一张判断「某频道确实没有源」用的**公共栅格**（本文件的 `栅格`）。

几何主力仍然复用 P1，闭式与代价换算复用 P2，本文件只补问题三特有的三件事：

    ① 三类**新的**约束（P1、P2 都没有，因为单源单频道时用不上）
       - 无信号排除盘        no_signal  =>  |x - p| > R_eff >= 1000
       - 接收半径序约束      在 q 收到、在 p 没收到  =>  |x - q| < |x - p|（半平面，精确）
       - 清除失败排除盘      no_target_in_range  =>  |x - p| > 20
    ② 判定「频道为空」的**可证伪据**：无信号排除盘之并覆盖整个目标圆域
    ③ 把非凸的排除盘装进凸多边形框架的**保守化**处理（宁可高估区域，绝不漏掉真源）

「保守」在本文件里始终指同一个方向：**我们维护的集合必须是真实可行集的超集**。
一旦某一步偏到子集侧，真源就可能被排除，「确保清除」就无从谈起。
每个近似的地方都标注了它偏向哪一侧。
"""
import sys
from pathlib import Path

import numpy as np

_根 = Path(__file__).resolve().parents[1]
# 借 P1、P2 的代码，但**用完就把它们的目录从 sys.path 摘掉**。
# 不摘的话，P1 也有 生成算例.py、P1/P2 都有 独立复核.py，会把 P3 的同名文件遮住
# （P2/模型.py 内部还会再 insert 一次 P1，所以摘的时候要连它插的那条一起摘）。
_旧路径 = list(sys.path)
sys.path.insert(0, str(_根 / "P1"))
sys.path.insert(0, str(_根 / "P2"))
import 求解算法 as 几何          # noqa: E402  P1：楔形半平面、裁剪、去冗余、最小覆盖圆
import 模型 as P2模型            # noqa: E402  P2：求直径快、最小覆盖圆快、圆的外切半平面
_借用目录 = {(_根 / "P1").resolve(), (_根 / "P2").resolve()}
sys.path[:] = [p for p in sys.path
               if p in _旧路径 or Path(p).resolve() not in _借用目录]

# ---------------------------------------------------------------- 题目常量
R_AREA = 1800.0          # 目标区域半径，米（题目正文）
DELTA = 1.0              # 示向度误差半角，度（附录 2 第 1 条：误差在 [-1°,1°] 内）
DELTA_算法 = 1.005       # 算法侧楔形半角：+0.005° 是 svd_deg 两位小数舍入的裕度（P1 §5.4）
R_RECV_MIN = 1000.0      # 有效接收半径下界，米（附录 2 第 2 条）
R_RECV_MAX = 1500.0      # 有效接收半径上界，米
R_NEAR = 5.0             # 近距离阈值，米（附录 2 第 9 条）
R_CLEAR = 20.0           # 清除半径，米（附录 2 第 8 条）

V_MOVE = 5.0             # 机器狗速度，米/秒
T_MEASURE = 5.0          # 检测耗时，秒
T_SWITCH = 1.0           # 切换频道耗时，秒
T_CLEAR_OK = 5.0         # 清除成功耗时，秒（光学定位 3 + 激光清除 2）
T_CLEAR_FAIL = 3.0       # 未发现耗时，秒

频道数 = 20              # 频道编号 1..20（附件 2 §1.3）
源数下界, 源数上界 = 10, 16

# 一次检测折合的路程：5 s × 5 m/s = 25 m（P2 §4.1 的换算，问题三沿用）
米每检测 = T_MEASURE * V_MOVE
米每切频 = T_SWITCH * V_MOVE

EPS = 1e-9
局部细化上限 = 400.0     # 区域直径超过它就不去扣 20 m 清除失败盘（忽略偏保守）

# ---------------------------------------------------------------- 消融开关
# 只给 策略对照.py 的消融实验用。正常运行时都保持 True，改动它们等于换一个模型。
启用序约束 = True        # 「在 q 收到、在 p 没收到 => |x-q| < |x-p|」这一族半平面
启用2opt = True          # 巡回排序是否做 2-opt 改良（关掉就退化成纯最近邻）


# ================================================================ 一、角度与向量
def 方位角(起点, 终点):
    d = np.asarray(终点, dtype=float) - np.asarray(起点, dtype=float)
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def 单位向量(角度_度):
    t = np.radians(角度_度)
    return np.array([np.cos(t), np.sin(t)])


def 距(p, q):
    return float(np.hypot(q[0] - p[0], q[1] - p[1]))


# ================================================================ 二、半平面工具
def 圆内半平面(圆心, 半径, 段数=180, 角范围=None):
    """|x - 圆心| <= 半径 的**外切**多边形约束，是圆盘的超集（偏保守侧）。"""
    return P2模型.圆的外切半平面(圆心, 半径, 段数=段数, 角范围=角范围)


def 序约束半平面(近点, 远点):
    """|x - 近点| <= |x - 远点|，精确的半平面，不含任何近似。

    出处：有效接收半径 R_eff 是该干扰源的固有常数。在 近点 收到信号 =>
    |近点 - G| <= R_eff；在 远点 没收到 => |远点 - G| > R_eff。两式相减即得。
    化成 a x + b y <= c：2 x·(远-近) <= |远|^2 - |近|^2。
    """
    q = np.asarray(近点, dtype=float)
    p = np.asarray(远点, dtype=float)
    a, b = 2.0 * (p - q)
    c = float(p @ p - q @ q)
    return (float(a), float(b), c)


目标圆约束 = 圆内半平面((0.0, 0.0), R_AREA, 段数=180)


def _初始多边形():
    """目标圆域的外切 180 边形，作为一切裁剪的起点。

    用它而不是 P1 的大方框：问题三里「干扰源必在目标区域内」是要**主动用上**的
    先验（判定频道为空全靠它），不像问题一那样只当边角情形讨论。
    外切多边形的外凸余量 R(1/cos1° - 1) = 0.27 m，偏超集侧。
    """
    多边形 = [(-10 * R_AREA, -10 * R_AREA), (10 * R_AREA, -10 * R_AREA),
              (10 * R_AREA, 10 * R_AREA), (-10 * R_AREA, 10 * R_AREA)]
    for a, b, c in 目标圆约束:
        多边形 = 几何.裁剪(多边形, a, b, c)
    return 几何.去冗余(多边形)


目标圆多边形 = _初始多边形()


def 半平面交(半平面列表, 起始=None):
    """从目标圆多边形出发依次裁剪，返回凸多边形顶点表（空则返回 []）。"""
    多边形 = list(起始 if 起始 is not None else 目标圆多边形)
    for a, b, c in 半平面列表:
        多边形 = 几何.裁剪(多边形, a, b, c)
        if not 多边形:
            return []
    return 几何.去冗余(多边形)


def 圆外保守半平面(多边形, 圆心, 半径):
    """把非凸约束 |x - 圆心| > 半径 在给定多边形上**保守地**线性化。

    返回 (半平面, 情形)：
        情形 = "无交"   多边形整体已在圆外，无需裁剪，半平面为 None
        情形 = "矛盾"   多边形整体在圆内，与约束冲突（读数异常）
        情形 = "裁剪"   给出一个半平面，其与多边形之交 **包含** 真实的
                        {多边形} ∩ {|x-圆心|>半径}

    做法：取 u = 圆心指向多边形形心的单位向量，把多边形分解为轴向分量 t 与横向
    分量 s。由 |x-圆心| = sqrt(t^2+s^2) <= t + s^2/(2 t_min)，故
        |x-圆心| > 半径  =>  t > 半径 - s_max^2 / (2 t_min)
    右端即保守的轴向阈值：真实要剔除的点一定被剔除得**更少**，集合仍是超集。
    """
    P = np.asarray(多边形, dtype=float)
    c = np.asarray(圆心, dtype=float)
    d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    if float(d.min()) >= 半径:
        return None, "无交"
    if float(d.max()) <= 半径:
        return None, "矛盾"
    形心 = P.mean(axis=0)
    v = 形心 - c
    n = float(np.hypot(*v))
    if n < EPS:
        return None, "矛盾"
    u = v / n
    t = (P - c) @ u
    s = np.abs((P - c) @ np.array([-u[1], u[0]]))
    t_min = max(float(t.min()), 1.0)
    阈值 = 半径 - float(s.max()) ** 2 / (2.0 * t_min)
    # {u·(x-c) >= 阈值} 写成 a x + b y <= c 的形式
    return (-u[0], -u[1], -(阈值 + float(u @ c))), "裁剪"


# ================================================================ 三、公共栅格
class 栅格:
    """覆盖整个目标圆域的方格点阵，用于「某频道确实没有源」的可证伪据。

    判据（附录 2 第 2 条 + 附件 2 §2.1）：接收不到信号 => 距离 > R_eff >= 1000，
    所以一次 no_signal 就排除了以检测点为心、半径 1000 m 的整个圆盘。
    于是「频道 c 为空」等价于：**c 的各无信号点的 1000 m 圆盘之并覆盖目标圆域**。

    离散化必须偏保守，否则「判空」就成了没有依据的猜测：
      · 只有当整个方格都落进排除盘时才划掉它，即要求
        中心距 <= 半径 - 步长/sqrt2；
      · 方格中心取到 R_AREA + 步长/sqrt2，保证目标圆域被格子完全盖住。
    这样一来「掩码全空 => 可行集为空」是严格成立的（反向不成立，偏保守）。
    """

    def __init__(self, 步长=20.0, R=R_AREA):
        self.步长 = float(步长)
        self.裕度 = float(步长) / np.sqrt(2.0)
        k = int(np.ceil((R + self.裕度) / 步长))
        轴 = np.arange(-k, k + 1) * 步长
        X, Y = np.meshgrid(轴, 轴)
        内 = np.hypot(X, Y) <= R + self.裕度
        self.点 = np.stack([X[内], Y[内]], axis=1)
        self.n = len(self.点)
        self.单元面积 = 步长 * 步长

    def 新掩码(self):
        return np.ones(self.n, dtype=bool)

    def 排除圆盘(self, 掩码, 圆心, 半径):
        """保守地划掉整格落入圆盘的方格；返回新掩码。"""
        d = np.hypot(self.点[:, 0] - 圆心[0], self.点[:, 1] - 圆心[1])
        return 掩码 & (d > 半径 - self.裕度)

    def 圆盘新增(self, 掩码, 圆心, 半径):
        """若在此处检测一次，能新排除多少个方格（用于选锚点）。"""
        d = np.hypot(self.点[掩码][:, 0] - 圆心[0], self.点[掩码][:, 1] - 圆心[1])
        return int(np.count_nonzero(d <= 半径 - self.裕度))

    def 面积(self, 掩码):
        return float(np.count_nonzero(掩码)) * self.单元面积

    def 形心(self, 掩码):
        if not np.any(掩码):
            return None
        return self.点[掩码].mean(axis=0)


# ================================================================ 四、频道信念
class 频道信念:
    """单个频道的集员估计状态。

    三种宏观状态：
        未探测  从未收到过该频道的信号。可行集 = 目标圆域 减去各无信号排除盘，
                非凸，用公共栅格掩码表示。掩码为空即**判定该频道没有干扰源**。
        已探测  至少收到过一次信号。可行集 = 若干凸约束之交（楔形、|x-p|<=1500、
                序约束、目标圆域）再减去若干排除盘，用凸多边形 + 保守线性化表示。
        已清除  收到 success，退出一切后续考虑。
    """

    def __init__(self, 频道, delta=DELTA_算法):
        self.频道 = int(频道)
        self.delta = float(delta)
        self.已清除 = False
        self.收到信号 = False
        self.判空 = False
        self.半平面 = []                 # 凸约束（不含目标圆域，那是裁剪起点）
        self.信号点 = []                 # [(位置, 示向度)]，near 时示向度为 None
        self.无信号点 = []
        self.失败清除点 = []
        self.检测次数 = 0
        self.异常 = None                 # 记录出现过的矛盾，供复核统计
        self._缓存 = None

    # ------------------------------------------------------------ 记录观测
    def 记录检测(self, p, 结果, svd=None):
        p = (float(p[0]), float(p[1]))
        self.检测次数 += 1
        self._缓存 = None
        if 结果 == "direction":
            self.收到信号 = True
            self.信号点.append((p, float(svd)))
            self.半平面 += 几何.楔形半平面(p, float(svd), self.delta)
            self.半平面 += 圆内半平面(p, R_RECV_MAX, 段数=72)
            if 启用序约束:
                for q in self.无信号点:
                    self.半平面.append(序约束半平面(p, q))
        elif 结果 == "near":
            self.收到信号 = True
            self.信号点.append((p, None))
            self.半平面 += 圆内半平面(p, R_NEAR, 段数=36)
            if 启用序约束:
                for q in self.无信号点:
                    self.半平面.append(序约束半平面(p, q))
        elif 结果 == "no_signal":
            self.无信号点.append(p)
            if 启用序约束:
                for q, _ in self.信号点:
                    self.半平面.append(序约束半平面(q, p))
        else:
            raise ValueError(f"未知检测结果：{结果}")

    def 记录清除失败(self, p):
        self.失败清除点.append((float(p[0]), float(p[1])))
        self._缓存 = None

    def 记录清除成功(self):
        self.已清除 = True
        self._缓存 = None

    # ------------------------------------------------------------ 可行集
    def 多边形(self):
        """已探测频道的凸可行集（含排除盘的保守线性化）。未探测频道返回 None。"""
        if not self.收到信号:
            return None
        if self._缓存 is not None:
            return self._缓存
        P = 半平面交(self.半平面)
        # 非凸排除逐个保守线性化。迭代两轮：区域缩小后阈值会更紧。
        for _ in range(2):
            if not P:
                break
            for q in self.无信号点:
                h, 情形 = 圆外保守半平面(P, q, R_RECV_MIN)
                if 情形 == "矛盾":
                    self.异常 = "无信号盘吞没可行集"
                    P = []
                    break
                if h is not None:
                    P = 几何.去冗余(几何.裁剪(P, *h))
                    if not P:
                        break
            if not P:
                break
        self._缓存 = P
        return P

    def 指标(self, 细网格步长=1.0):
        """返回 dict：直径、最小覆盖圆半径 Rmin、圆心、顶点数、是否被排除盘切过。

        有失败清除点压在区域上时，凸多边形不足以描述残余集，改用 1 m 细网格；
        细网格的 Rmin 再加半格对角线 (0.71 m) 作为保守修正。
        """
        P = self.多边形()
        if P is None:
            return None
        if not P:
            return {"空": True}
        直径, _ = P2模型.求直径快(P)
        圆心, Rmin = P2模型.最小覆盖圆快(P)
        圆心 = np.asarray(圆心, dtype=float)
        切过 = False
        # 清除失败盘只有半径 20 m，只有当区域已经缩到同一量级时才值得费劲扣掉它。
        # 区域还很大时（例如只有一条示向线、区域是 1500 m 长的细条），
        # 忽略这个 20 m 的洞不会影响任何决策，而且**忽略是偏保守的**（区域更大）。
        压住 = [c for c in self.失败清除点
                if 距(c, 圆心) <= R_CLEAR + Rmin] if 直径 <= 局部细化上限 else []
        if 压住:
            V = np.asarray(P, dtype=float)
            # 步长必须**随区域尺度缩放**。早先写成 max(1 m, 直径/150)，
            # 区域缩到几米宽时格子比区域还大，一个格点都落不进多边形，
            # 于是残余被判成空、频道被误重置——独立复核抓到过一次（见 分析记录 §6.3）。
            残 = None
            for 因子 in (1.0, 0.25, 0.0625):
                步 = float(np.clip(直径 / 120.0 * 因子, 2e-2, 5.0))
                lo, hi = V.min(axis=0) - 步, V.max(axis=0) + 步
                xs = np.arange(lo[0], hi[0] + 步, 步)
                ys = np.arange(lo[1], hi[1] + 步, 步)
                X, Y = np.meshgrid(xs, ys)
                点 = np.stack([X.ravel(), Y.ravel()], axis=1)
                留 = (np.array([几何.点在多边形内(tuple(t), P) for t in 点])
                      if len(P) >= 3 else np.ones(len(点), dtype=bool))
                # 只划掉**整格**都落进 20 m 清除盘的方格：留下的仍是真实残余的超集。
                # 若按格心判（阈值恰取 20），Rmin 略大于 20 的区域在盲清失手后
                # 可能被整片划空，把「还没打中的源」误判成不存在。
                半格 = 步 / np.sqrt(2.0)
                for c in 压住:
                    留 &= np.hypot(点[:, 0] - c[0], 点[:, 1] - c[1]) > R_CLEAR - 半格
                if np.any(留):
                    残 = 点[留]
                    break
            if 残 is None:
                # 加密三轮仍分辨不出残余。**不能据此判空**——判空会触发重置、
                # 把真源丢出可行集。按凸区域保守处理（集合更大，绝不漏源）。
                self.异常 = "细网格未能分辨清除失败盘的残余，按凸区域保守处理"
                return {"空": False, "多边形": P, "直径": float(直径),
                        "Rmin": float(Rmin),
                        "圆心": (float(圆心[0]), float(圆心[1])),
                        "顶点数": len(P), "切过": False}
            if len(残) == 1:
                圆心, Rmin, 直径, 切过 = 残[0], 0.0, 0.0, True
            else:
                壳 = 残
                if len(残) > 3:
                    try:
                        from scipy.spatial import ConvexHull
                        壳 = 残[ConvexHull(残).vertices]
                    except Exception:
                        pass
                d, _ = P2模型.求直径快([tuple(t) for t in 壳])
                cc, rr = 最小覆盖圆_Welzl(壳)
                圆心 = np.asarray(cc, dtype=float)
                Rmin = rr + 半格                            # 保守修正
                直径, 切过 = d, True
        return {"空": False, "多边形": P, "直径": float(直径), "Rmin": float(Rmin),
                "圆心": (float(圆心[0]), float(圆心[1])), "顶点数": len(P), "切过": 切过}


# ================================================================ 四点五、最小覆盖圆（点集版）
def _三点外接圆(a, b, c):
    ax, ay = b[0] - a[0], b[1] - a[1]
    bx, by = c[0] - a[0], c[1] - a[1]
    分母 = 2.0 * (ax * by - ay * bx)
    if abs(分母) < 1e-12:
        return None
    a2, b2 = ax * ax + ay * ay, bx * bx + by * by
    ux = (by * a2 - ay * b2) / 分母
    uy = (ax * b2 - bx * a2) / 分母
    return (a[0] + ux, a[1] + uy), float(np.hypot(ux, uy))


def _圆含(圆, p, 容差=1e-7):
    (cx, cy), r = 圆
    return np.hypot(p[0] - cx, p[1] - cy) <= r + 容差


def _平凡圆(边界):
    if not 边界:
        return ((0.0, 0.0), -1.0)
    if len(边界) == 1:
        return (tuple(边界[0]), 0.0)
    if len(边界) == 2:
        a, b = 边界
        return (((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                float(np.hypot(a[0] - b[0], a[1] - b[1]) / 2))
    圆 = _三点外接圆(*边界)
    if 圆 is None:                     # 三点共线，退化成最远两点的直径圆
        最优 = ((0.0, 0.0), np.inf)
        for i in range(3):
            for j in range(i + 1, 3):
                c = _平凡圆([边界[i], 边界[j]])
                if c[1] < 最优[1] and all(_圆含(c, p) for p in 边界):
                    最优 = c
        return 最优
    return 圆


def 最小覆盖圆_Welzl(点集, rng=None):
    """Welzl 随机增量算法，期望 O(n)。点数上千时请先取凸包再调用。

    P2 §6 的教训：最小覆盖圆的复核实现若用模式搜索会卡在非光滑棱上（当时误差 3.84 m）。
    这里直接用精确算法，不留近似的余地。
    """
    P = [tuple(map(float, p)) for p in 点集]
    if not P:
        return ((0.0, 0.0), -1.0)
    (rng or np.random.default_rng(0)).shuffle(P)
    圆 = _平凡圆([P[0]])
    for i in range(1, len(P)):
        if _圆含(圆, P[i]):
            continue
        圆 = _平凡圆([P[i]])
        for j in range(i):
            if _圆含(圆, P[j]):
                continue
            圆 = _平凡圆([P[i], P[j]])
            for k in range(j):
                if not _圆含(圆, P[k]):
                    圆 = _平凡圆([P[i], P[j], P[k]])
    return 圆


def 最小覆盖圆_点集(点集):
    """先取凸包压缩再 Welzl。最小覆盖圆只由极点决定，凸包不改变结果。"""
    P = np.asarray(点集, dtype=float).reshape(-1, 2)
    if len(P) <= 3:
        return 最小覆盖圆_Welzl(P)
    try:
        from scipy.spatial import ConvexHull
        P = P[ConvexHull(P).vertices]
    except Exception:                  # 共线等退化情形，Qhull 会抛异常
        pass
    return 最小覆盖圆_Welzl(P)


# ================================================================ 五、代价模型
def 移动时间(距离):
    return float(距离) / V_MOVE


def 停靠代价(当前位置, 目标位置, 频道表, 当前频道):
    """走到 目标位置 并依次检测 频道表 的虚拟耗时（秒）。

    切频道耗时只看相邻两次 /measure 的频道是否相同（附件 2 §4.3），
    所以第一条与当前频道相同则省 1 s，其余每条各 1 s。
    """
    t = 移动时间(距(当前位置, 目标位置))
    上一个 = 当前频道
    for c in 频道表:
        if c != 上一个:
            t += T_SWITCH
        t += T_MEASURE
        上一个 = c
    return t


def 排序检测频道(频道表, 当前频道):
    """把与当前频道相同的那一个排到最前，省 1 秒切频。其余顺序无差别。"""
    频道表 = list(频道表)
    if 当前频道 in 频道表:
        频道表.remove(当前频道)
        频道表.insert(0, 当前频道)
    return 频道表


# ================================================================ 六、巡回路线
def 最近邻路线(起点, 点表):
    """最近邻构造开放巡回（不回起点）。返回访问顺序的下标列表。"""
    剩 = list(range(len(点表)))
    路 = []
    当前 = np.asarray(起点, dtype=float)
    while 剩:
        j = min(剩, key=lambda i: 距(当前, 点表[i]))
        路.append(j)
        当前 = np.asarray(点表[j], dtype=float)
        剩.remove(j)
    return 路


def 路线长度(起点, 点表, 顺序):
    当前 = np.asarray(起点, dtype=float)
    s = 0.0
    for i in 顺序:
        s += 距(当前, 点表[i])
        当前 = np.asarray(点表[i], dtype=float)
    return s


def 二opt(起点, 点表, 顺序, 最大轮=40):
    """开放路径的 2-opt 改良（反转一段子路径）。点数很少，直接双重循环。"""
    顺序 = list(顺序)
    n = len(顺序)
    if n < 3:
        return 顺序
    最好 = 路线长度(起点, 点表, 顺序)
    for _ in range(最大轮):
        改进 = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                新 = 顺序[:i] + 顺序[i:j + 1][::-1] + 顺序[j + 1:]
                L = 路线长度(起点, 点表, 新)
                if L < 最好 - 1e-9:
                    顺序, 最好, 改进 = 新, L, True
        if not 改进:
            break
    return 顺序


def 最小生成树长度(点表, 收缩=0.0, 起点收缩=0.0, 起点=(0.0, 0.0)):
    """{起点} ∪ 点表 的最小生成树总长（Prim）。用作巡回长度的**严格下界**。

    任何访问全部点的路径都是一棵生成树，故路径长度 >= MST 长度。
    收缩：本题的机器狗只要走到距干扰源 20 m 以内就能清除，所以两点之间实际
    需要移动的距离至少是 d - 2x20；与起点之间至少是 d - 20。把每条边按这个
    量削短再求 MST，得到的仍是严格下界（削短只会让下界更小）。
    """
    P = [tuple(map(float, 起点))] + [tuple(map(float, p)) for p in 点表]
    n = len(P)
    if n <= 1:
        return 0.0
    A = np.asarray(P, dtype=float)
    D = np.hypot(A[:, None, 0] - A[None, :, 0], A[:, None, 1] - A[None, :, 1])
    D = np.maximum(D - 收缩, 0.0)
    D[0, :] = np.maximum(np.hypot(A[:, 0] - A[0, 0], A[:, 1] - A[0, 1])
                         - 起点收缩, 0.0)
    D[:, 0] = D[0, :]
    np.fill_diagonal(D, np.inf)
    在树 = np.zeros(n, dtype=bool)
    在树[0] = True
    最近 = D[0].copy()
    总 = 0.0
    for _ in range(n - 1):
        最近[在树] = np.inf
        j = int(np.argmin(最近))
        总 += float(最近[j])
        在树[j] = True
        最近 = np.minimum(最近, D[j])
    return 总


def 开放TSP(起点, 点表):
    """最近邻 + 2-opt。返回 (顺序, 长度)。"""
    if not len(点表):
        return [], 0.0
    顺序 = 最近邻路线(起点, 点表)
    if 启用2opt:
        顺序 = 二opt(起点, 点表, 顺序)
    return 顺序, 路线长度(起点, 点表, 顺序)
