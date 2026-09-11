"""问题四 · 公共模型层（只算数，不作图，不写文件）。

被 判空几何.py / 模拟器.py / 生成算例.py / 策略.py / 参数搜索.py / 策略对照.py /
灵敏度.py / 命题与引理.py / 独立复核.py / 作图数据.py 共同 import。

问题四与问题三的差别只有一句话：**目标区域里除了全向干扰源，还有定向干扰源**
（有效覆盖角度范围为定向方向两侧各 90°，含边界；个数与方向都未知）。
这一句话把问题三的信息模型从根上改了一处、只改了一处，但改得很彻底：

    问题三   no_signal 在 p  ==>  |p - G| > R_eff >= 1000     （排除一个 1000 m 圆盘）
    问题四   no_signal 在 p  ==>  |p - G| > R_eff  **或**  (p - G)·u < 0

后者对**位置**没有任何约束——源可以就在脚下，只是背对着你。于是问题三赖以
「判空」的排除盘整个消失，判空的论证必须重做。本文件的核心就是重做后的那一套：

    ① 覆盖角是一个**半平面**：定向源的有效覆盖范围 = 过 G、法向为定向方向 u 的闭半平面
       （两侧各 90° 恰好是 180°，这是全题最关键的一条几何事实，见 §定向几何）
    ② 判空判据从「圆盘覆盖」变成「**半圆盘围堵**」：
          频道 c 判空  <=>  对目标区域内每一点 x 与每个方向 u，
                            半圆盘 {p : |p-x| <= 1000, (p-x)·u >= 0} 内至少有一个
                            c 的无信号检测点
       等价地：x 落在「距 x 不超过 1000 m 的无信号点」的凸包内。
    ③ 序约束（问题三 §3.2 的②）在问题四**失效**，只能在「已证明该频道是全向源」
       之后才允许使用。

「保守」在本文件里始终指同一个方向：**我们维护的集合必须是真实可行集的超集**。
一旦某一步偏到子集侧，真源就可能被排除，「确保清除」就无从谈起。
每个近似的地方都标注了它偏向哪一侧。

几何主力仍复用 P1（楔形半平面、裁剪、最小覆盖圆）与 P2（求直径快、圆的外切半平面）。
通用的巡回 / MST / Welzl 工具与 P3 同源，为使本目录可独立运行而就地重写，
并在各函数的文档串里标明出处。
"""
import sys
from pathlib import Path

import numpy as np

_根 = Path(__file__).resolve().parents[1]
# 借 P1、P2 的几何代码，但**用完就把它们的目录从 sys.path 摘掉**，
# 否则 P1/P2/P3 里的同名文件（生成算例.py、独立复核.py、模型.py…）会遮住本目录的。
_旧路径 = list(sys.path)
sys.path.insert(0, str(_根 / "P1"))
sys.path.insert(0, str(_根 / "P2"))
import 求解算法 as 几何          # noqa: E402  P1：楔形半平面、裁剪、去冗余、点在多边形内
import 模型 as P2模型            # noqa: E402  P2：求直径快、最小覆盖圆快、圆的外切半平面
_借用目录 = {(_根 / "P1").resolve(), (_根 / "P2").resolve()}
sys.path[:] = [p for p in sys.path
               if p in _旧路径 or Path(p).resolve() not in _借用目录]

# ---------------------------------------------------------------- 题目常量
R_AREA = 1800.0          # 目标区域半径，米（题目正文）
DELTA = 1.0              # 示向度误差半角，度（附录 2 第 1 条）
DELTA_算法 = 1.005       # 算法侧楔形半角：+0.005° 是 svd_deg 两位小数舍入的裕度（P1 §5.4）
R_RECV_MIN = 1000.0      # 有效接收半径下界，米（附录 2 第 2 条）
R_RECV_MAX = 1500.0      # 有效接收半径上界，米
R_NEAR = 5.0             # 近距离阈值，米（附录 2 第 9 条）
R_CLEAR = 20.0           # 清除半径，米（附录 2 第 8 条）
半角_定向 = 90.0         # 定向源覆盖角度范围的半角，度（附录 1 第 3 条：两侧各 90°，含）

V_MOVE = 5.0             # 机器狗速度，米/秒
T_MEASURE = 5.0          # 检测耗时，秒
T_SWITCH = 1.0           # 切换频道耗时，秒
T_CLEAR_OK = 5.0         # 清除成功耗时，秒（光学定位 3 + 激光清除 2）
T_CLEAR_FAIL = 3.0       # 未发现耗时，秒

频道数 = 20              # 频道编号 1..20（附件 2 §1.3）
源数下界, 源数上界 = 10, 16

米每检测 = T_MEASURE * V_MOVE      # 一次检测折合 25 米路程（P2 §4.1 的换算）
米每切频 = T_SWITCH * V_MOVE

EPS = 1e-9
局部细化上限 = 400.0     # 区域直径超过它就不去扣 20 m 清除失败盘（忽略偏保守）

# ---------------------------------------------------------------- 消融开关
# 只给 策略对照.py 的消融实验用。正常运行时保持默认值，改动它们等于换一个模型。
启用2opt = True          # 巡回排序是否做 2-opt 改良（关掉就退化成纯最近邻）
启用分类器 = True        # 是否做「全向 / 定向」判定（判定为全向后可启用序约束与排除盘）
启用镜像补测 = True      # 补测失败后是否走「镜像必达」点（见 §定向几何 引理 D3）


# ================================================================ 〇、定向几何
# 这一节是问题四全部新结论的出处，先把话说死，后面的每个判据都回指这里。
#
# 【事实 D0（覆盖角 = 半平面）】
#   定向源位于 G、定向方向为单位向量 u 时，其有效覆盖角度范围是「u 两侧各 90°（含）」，
#   即检测点 p 能收到信号的角度条件为
#
#         angle(p - G, u) <= 90°   <==>   (p - G)·u >= 0
#
#   这是一个**过 G 的闭半平面**。全向源相当于「对一切 u 都成立」，
#   在本文件里用 u = None 表示。
#   半平面是凸集，且含 G——问题四几乎所有可用的结论都由这两条性质推出。
#
# 【引理 D1（收信锥）】设某频道在 p_1,…,p_k 处都收到过信号，则对真实源位置 G，
#   有 (p_i - G)·u >= 0（i = 1..k）。由半平面的凸性，
#
#         G + cone{p_1 - G, …, p_k - G}  整体落在覆盖角内。
#
#   即：**由已成功的检测点张成的锥内，任何位置都保证在覆盖角范围内**。
#   k = 1 时锥退化为射线 G->p_1：只有「沿着已测方位线」才是保证收得到的。
#
# 【引理 D2（三点包围 ⇒ 全向）】若 G 落在信号点凸包的内部，则 {p_i - G} 正张成 R²，
#   不存在满足全部 (p_i - G)·u >= 0 的 u，故该频道**必为全向源**。
#   这是本文唯一能**证明**某频道是全向源的判据。
#
# 【引理 D3（镜像必达）】设某频道在 p 处收到信号、在 q 处未收到，且已知 q 在有效
#   接收距离内（|q - G| <= R_eff），则对任意 λ > 0，
#
#         r = p + λ(p - q)   满足   (r - G)·u > 0，
#
#   即 **r 一定落在覆盖角范围内**。
#   证明：(r-G)·u = (1+λ)(p-G)·u - λ(q-G)·u >= 0 + λ·(正数) > 0。∎
#   注意结论**不含 G**——不需要知道源在哪就能用。λ = 1 即「把 q 关于 p 作中心对称」。
#   它把「换个角度再测一次」的最坏代价钉死在 2 次检测：第一次赌一个大侧移角，
#   万一收不到，第二次走镜像点必定收到。
#
# 【引理 D4（Farkas 排除）】设某频道在 p_1..p_k 收到信号、在 q 未收到，
#   且 q 在有效接收距离内。则真源位置 G 必满足
#
#         q - G  ∉  cone{p_1 - G, …, p_k - G}。
#
#   证明：由 Farkas 引理，∃u 使 u·(p_i-G) >= 0 (∀i) 且 u·(q-G) < 0
#   当且仅当 q-G 不属于该锥。∎  这是问题四里 no_signal 唯一还能提供的位置信息。
#
# 【命题 D5（判空判据）】频道 c 的无信号点集为 N_c。目标区域内的位置 x 可以被排除，
#   当且仅当
#
#         对每个方向 u，  ∃ p ∈ N_c :  |p - x| <= 1000  且  (p - x)·u >= 0，
#
#   等价地   x ∈ conv{ p ∈ N_c : |p - x| <= 1000 }。
#
#   证明：x 不能被排除 <=> 存在 u 使所有「在接收距离内」的无信号点都落在
#   开半平面 (p-x)·u < 0 里 <=> {p - x} 落在某个开半平面内 <=> x ∉ conv(·)。
#   （距离用下界 1000 是因为 R_eff >= 1000 是题给的唯一保证。）∎
#
#   **一句话**：问题三的判空是「用 1000 m 圆盘盖住目标区域」，
#   问题四的判空是「用无信号点把目标区域里的每一点**围起来**」。
#   前者每点只需 1 个见证者，后者每点至少需要 3 个，代价因此成倍上升（见 判空几何.py）。


def 覆盖角内(源位置, 定向方向_度, 检测点):
    """检测点是否落在该干扰源的有效覆盖角度范围内。定向方向为 None 表示全向源。"""
    if 定向方向_度 is None:
        return True
    d = np.asarray(检测点, dtype=float) - np.asarray(源位置, dtype=float)
    if float(np.hypot(*d)) < EPS:          # 检测点与源重合：必然收得到
        return True
    u = 单位向量(定向方向_度)
    return float(d @ u) >= 0.0             # 边界含在内（附录 1 第 3 条的「含」）


# ================================================================ 一、角度与向量
def 方位角(起点, 终点):
    d = np.asarray(终点, dtype=float) - np.asarray(起点, dtype=float)
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def 单位向量(角度_度):
    t = np.radians(角度_度)
    return np.array([np.cos(t), np.sin(t)])


def 距(p, q):
    return float(np.hypot(q[0] - p[0], q[1] - p[1]))


def 角差(a, b):
    """把 a - b 折算到 (-180, 180]，度。"""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def 张角跨度(角度表):
    """一组方位角的「最小包含弧」跨度（度）。跨度 > 180° 说明它们不可能同处一个半平面。"""
    a = np.sort(np.mod(np.asarray(角度表, dtype=float), 360.0))
    if len(a) == 0:
        return 0.0
    if len(a) == 1:
        return 0.0
    间隙 = np.diff(np.concatenate([a, a[:1] + 360.0]))
    return float(360.0 - 间隙.max())


# ================================================================ 二、半平面工具
def 圆内半平面(圆心, 半径, 段数=180, 角范围=None):
    """|x - 圆心| <= 半径 的**外切**多边形约束，是圆盘的超集（偏保守侧）。"""
    return P2模型.圆的外切半平面(圆心, 半径, 段数=段数, 角范围=角范围)


def 序约束半平面(近点, 远点):
    """|x - 近点| <= |x - 远点|，精确半平面。**问题四里只有已证明为全向源的频道能用。**

    出处（问题三 §3.2 的②）：R_eff 是干扰源的固有常数。在近点收到信号 =>
    |近点 - G| <= R_eff；在远点没收到 => |远点 - G| > R_eff。两式相减即得。
    问题四里「没收到」还可能是角度原因，推不出距离关系，所以默认禁用。
    """
    q = np.asarray(近点, dtype=float)
    p = np.asarray(远点, dtype=float)
    a, b = 2.0 * (p - q)
    c = float(p @ p - q @ q)
    return (float(a), float(b), c)


目标圆约束 = 圆内半平面((0.0, 0.0), R_AREA, 段数=180)


def _初始多边形():
    """目标圆域的外切 180 边形，作为一切裁剪的起点（外凸余量 0.27 m，偏超集侧）。"""
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
    """把非凸约束 |x - 圆心| > 半径 在给定多边形上**保守地**线性化（同 P3）。

    返回 (半平面, 情形)，情形 ∈ {"无交", "矛盾", "裁剪"}。
    由 |x-c| = sqrt(t²+s²) <= t + s²/(2 t_min) 得
        |x-c| > r  ==>  t > r - s_max²/(2 t_min)
    右端是保守的轴向阈值：真实要剔除的点一定被剔除得**更少**，集合仍是超集。
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
    return (-u[0], -u[1], -(阈值 + float(u @ c))), "裁剪"


# ================================================================ 三、方向栅格（判空的新机器）
class 方向栅格:
    """覆盖目标圆域的方格点阵，**每个方格再按方位角分成若干扇区**。

    这是问题四对问题三「判空栅格」的推广，也是本问最核心的一件数据结构。

    问题三：一个方格只有「已排除 / 未排除」两种状态，一次 no_signal 就把
            半径 1000 m 内的方格全部排除。
    问题四：一次 no_signal 只能排除「方格 x + 方向 u」这样的**二元组**——
            它说明「若源在 x 且定向方向为 u，则应当在 p 收到信号，与观测矛盾」，
            但只对那些满足 (p-x)·u >= 0 的 u 成立。
            方格 x 被真正排除，当且仅当它的**全部扇区**都被排除
            （命题 D5：每个方向都被某个无信号点见证）。

    保守离散化（两侧都必须偏保守，否则「判空」就成了没有依据的猜测）：
      · 距离：只有 |p - 方格中心| <= 1000 - 步长/sqrt2 才承认「必在接收距离内」，
        这样方格里任何一点 y 都满足 |p - y| <= 1000 <= R_eff；
      · 角度：只有当扇区**整体**落在 [θ - β, θ + β] 内才标记，其中
            θ = 方格中心看向 p 的方位角，
            β = arccos( (步长/sqrt2) / |p - 中心| )  = 90° - arcsin(ε/d)。
        这样对方格内任何一点 y 与该扇区内任何方向 u，都有 (p - y)·u >= 0。
      · 方格中心取到 R_AREA + 步长/sqrt2，保证目标圆域被格子完全盖住。

    于是「全部扇区都被标记 ==> 该方格里没有任何 (位置, 方向) 组合是可行的」
    严格成立（反向不成立，偏保守）。
    """

    def __init__(self, 步长=40.0, 扇区数=36, R=R_AREA, 额外裕度=0.0, 裕度=None):
        self.步长 = float(步长)
        self.扇区数 = int(扇区数)
        # 裕度默认由方格自身的半对角线决定；规划用的粗栅格另行指定（它的方格只是
        # 「采样位置」，裕度该由「判据格点到最近规划格点的距离」定，与自身步长无关）。
        self.裕度 = (float(步长) / np.sqrt(2.0) + float(额外裕度) if 裕度 is None
                     else float(裕度))
        self.扇宽 = 360.0 / self.扇区数
        k = int(np.ceil((R + float(步长) / np.sqrt(2.0)) / 步长))
        轴 = np.arange(-k, k + 1) * 步长
        X, Y = np.meshgrid(轴, 轴)
        内 = np.hypot(X, Y) <= R + float(步长) / np.sqrt(2.0)
        self.点 = np.stack([X[内], Y[内]], axis=1)
        self.n = len(self.点)
        self.单元面积 = 步长 * 步长
        self.有效半径 = R_RECV_MIN - self.裕度
        # 扇区边界的方位角（扇区 j 占 [边界[j], 边界[j+1]]）
        self.边界 = np.arange(self.扇区数 + 1) * self.扇宽
        # 同一个停靠点对整池频道的扇区标记完全相同，算一次就够（整池扫描时命中率很高）
        self._缓存 = {}

    # ------------------------------------------------------------ 掩码
    def 新掩码(self):
        """True = 该 (方格, 扇区) 仍未被排除。全 True 即「一无所知」。"""
        return np.ones((self.n, self.扇区数), dtype=bool)

    def 扇区标记(self, p, 行索引=None):
        """一次 no_signal 在 p 处能排除哪些 (方格, 扇区)。返回 (n, 扇区数) 布尔阵。

        行索引 给定时只算这些方格（收尾锚点只关心最后残下的那几格，不必全算）。
        """
        if 行索引 is None:
            键 = (round(float(p[0]), 3), round(float(p[1]), 3))
            命中 = self._缓存.get(键)
            if 命中 is not None:
                return 命中
        点 = self.点 if 行索引 is None else self.点[行索引]
        m = len(点)
        d = np.hypot(点[:, 0] - p[0], 点[:, 1] - p[1])
        在范围 = d <= self.有效半径
        标记 = np.zeros((m, self.扇区数), dtype=bool)
        if not np.any(在范围):
            if 行索引 is None:
                self._存(键, 标记)
            return 标记
        idx = np.where(在范围)[0]
        dd = d[idx]
        θ = np.degrees(np.arctan2(p[1] - 点[idx, 1], p[0] - 点[idx, 0]))
        # β = 90° - arcsin(ε/d)：方格里任何一点看向 p 的方位角与 θ 差不超过 arcsin(ε/d)
        比 = np.clip(self.裕度 / np.maximum(dd, EPS), 0.0, 1.0)
        β = 90.0 - np.degrees(np.arcsin(比))
        # 扇区整体落在 [θ-β, θ+β] 内 <=> 它的两条边界都落在里面（β < 90° < 180°，无歧义）
        边内 = np.abs(角差(self.边界[None, :], θ[:, None])) <= β[:, None]
        标记[idx] = 边内[:, :-1] & 边内[:, 1:]
        if 行索引 is None:
            self._存(键, 标记)
        return 标记

    def _存(self, 键, 值):
        if len(self._缓存) > 400:            # 一局用到的停靠点远少于这个数，防长跑膨胀
            self._缓存.clear()
        self._缓存[键] = 值

    def 并入无信号(self, 掩码, p):
        """把一次 no_signal 并进掩码（原地修改并返回）。"""
        掩码 &= ~self.扇区标记(p)
        return 掩码

    def 判空(self, 掩码):
        """掩码全部被排除 ==> 该频道在目标区域内不存在干扰源（命题 D5）。"""
        return not bool(掩码.any())

    def 残余数(self, 掩码):
        return int(掩码.sum())

    def 残余方格数(self, 掩码):
        """还有至少一个方向没被排除的方格个数（面积口径的残余）。"""
        return int(掩码.any(axis=1).sum())

    def 残余面积(self, 掩码):
        return float(self.残余方格数(掩码)) * self.单元面积

    def 新增覆盖(self, 掩码, p):
        """若在 p 处检测一次，能新排除多少个 (方格, 扇区) 二元组。"""
        return int((掩码 & self.扇区标记(p)).sum())

    def 残余方格点(self, 掩码):
        return self.点[掩码.any(axis=1)]


# ================================================================ 四、频道信念
class 频道信念:
    """单个频道的集员估计状态（问题四版）。

    与问题三的**唯一**区别在于「哪些约束是合法的」：

        约束                     问题三   问题四
        楔形（示向度）             ✔        ✔     覆盖角内场强只随距离衰减，语义完全相同
        |x - p| <= 1500           ✔        ✔     收到信号 => 在接收半径内
        |x - p| <= 5   (near)     ✔        ✔
        |x - p| > 5    (direction)✔        ✔
        清除失败盘 |x - p| > 20   ✔        ✔     清除与覆盖角无关（附件 2 §8.2 明文）
        无信号排除盘 |x-p| > 1000 ✔        ✘     可能只是背对着你
        接收半径序约束             ✔        ✘*    * 证明为全向源之后才可用

    三种宏观状态：未探测 / 已探测 / 已清除，与问题三相同。
    另外维护一个**定向性判定**：未知 / 已证全向（引理 D2）/ 已证定向（全向假设自相矛盾）。
    """

    def __init__(self, 频道, delta=DELTA_算法):
        self.频道 = int(频道)
        self.delta = float(delta)
        self.已清除 = False
        self.收到信号 = False
        self.半平面 = []                 # 问题四下**始终**合法的凸约束
        self.全向附加 = []               # 只在「已证全向」后才合法的凸约束（序约束）
        self.信号点 = []                 # [(位置, 示向度)]，near 时示向度为 None
        self.无信号点 = []
        self.失败清除点 = []
        self.检测次数 = 0
        self.定向性 = "未知"             # 未知 / 全向 / 定向
        self.最近无信号 = None           # 最近一次检测若是 no_signal，记下它的位置
        self.补测次数 = 0                # 为定位该频道而做的检测次数（不含整池扫描）
        self.异常 = None
        self._缓存 = None

    # ------------------------------------------------------------ 记录观测
    def 记录检测(self, p, 结果, svd=None):
        p = (float(p[0]), float(p[1]))
        self.检测次数 += 1
        self._缓存 = None
        if 结果 == "direction":
            self.收到信号 = True
            self.最近无信号 = None
            self.信号点.append((p, float(svd)))
            self.半平面 += 几何.楔形半平面(p, float(svd), self.delta)
            self.半平面 += 圆内半平面(p, R_RECV_MAX, 段数=72)
            for q in self.无信号点:
                self.全向附加.append(序约束半平面(p, q))
        elif 结果 == "near":
            self.收到信号 = True
            self.最近无信号 = None
            self.信号点.append((p, None))
            self.半平面 += 圆内半平面(p, R_NEAR, 段数=36)
            for q in self.无信号点:
                self.全向附加.append(序约束半平面(p, q))
        elif 结果 == "no_signal":
            self.最近无信号 = p
            self.无信号点.append(p)
            for q, _ in self.信号点:
                self.全向附加.append(序约束半平面(q, p))
        else:
            raise ValueError(f"未知检测结果：{结果}")

    def 记录清除失败(self, p):
        self.失败清除点.append((float(p[0]), float(p[1])))
        self._缓存 = None

    def 记录清除成功(self):
        self.已清除 = True
        self._缓存 = None

    # ------------------------------------------------------------ 定向性判定
    def 更新定向性(self):
        """尝试判定该频道是全向源还是定向源。两条判据都是**充分**的，不做猜测。

        判为定向：全向假设下的可行集为空（序约束或 1000 m 排除盘自相矛盾）。
                  既然真源一定存在于基础可行集里，全向分支空就说明它只能是定向源。
        判为全向：基础可行集整体落在信号点凸包的内部（引理 D2）。
                  此时不存在能同时覆盖全部信号点的半平面，定向分支被排除。
        """
        if not 启用分类器 or self.定向性 != "未知" or not self.收到信号:
            return self.定向性
        P = self.多边形()
        if not P:
            return self.定向性
        # —— 判为全向：可行集 ⊆ int conv(信号点) ——
        S = [s for s, _ in self.信号点]
        if len(S) >= 3:
            try:
                from scipy.spatial import ConvexHull
                壳 = np.asarray(S, dtype=float)[ConvexHull(np.asarray(S, dtype=float)).vertices]
                if all(几何.点在多边形内(tuple(v), [tuple(t) for t in 壳]) for v in P):
                    self.定向性 = "全向"
                    self._缓存 = None
                    return self.定向性
            except Exception:
                pass                       # 共线等退化情形，Qhull 抛异常，放弃这一判据
        # —— 判为定向：全向假设自相矛盾 ——
        if self.无信号点 and self.全向附加:
            if not self._全向多边形():
                self.定向性 = "定向"
        return self.定向性

    def _全向多边形(self):
        """全向假设下的可行集（基础约束 + 序约束 + 1000 m 排除盘）。只用于判定与收紧。"""
        P = 半平面交(self.半平面 + self.全向附加)
        for _ in range(2):
            if not P:
                break
            for q in self.无信号点:
                h, 情形 = 圆外保守半平面(P, q, R_RECV_MIN)
                if 情形 == "矛盾":
                    return []
                if h is not None:
                    P = 几何.去冗余(几何.裁剪(P, *h))
                    if not P:
                        return []
        return P

    # ------------------------------------------------------------ 可行集
    def 多边形(self):
        """可行集的凸表示。未探测频道返回 None（它由方向栅格掩码表示）。

        **与问题三的关键差别：默认不并入任何依赖「无信号」的约束。**
        只有 定向性 == "全向" 时才启用序约束与 1000 m 排除盘。
        """
        if not self.收到信号:
            return None
        if self._缓存 is not None:
            return self._缓存
        if self.定向性 == "全向":
            P = self._全向多边形()
            if not P:                       # 理论上不该发生（已判全向说明全向分支非空）
                self.异常 = "全向假设下可行集为空"
                self.定向性 = "定向"
                P = 半平面交(self.半平面)
        else:
            P = 半平面交(self.半平面)
        self._缓存 = P
        return P

    def 指标(self):
        """返回 dict：直径、最小覆盖圆半径 Rmin、圆心、顶点数、是否被清除失败盘切过。

        实现与问题三同源：有失败清除点压在区域上时改用局部细网格，
        细网格的 Rmin 再加半格对角线作为保守修正。清除与覆盖角无关，所以这一段照搬。
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
        压住 = [c for c in self.失败清除点
                if 距(c, 圆心) <= R_CLEAR + Rmin] if 直径 <= 局部细化上限 else []
        if 压住:
            V = np.asarray(P, dtype=float)
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
                半格 = 步 / np.sqrt(2.0)
                for c in 压住:
                    留 &= np.hypot(点[:, 0] - c[0], 点[:, 1] - c[1]) > R_CLEAR - 半格
                if np.any(留):
                    残 = 点[留]
                    break
            if 残 is None:
                # 加密三轮仍分辨不出残余：**不能据此判空**，按凸区域保守处理。
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
                Rmin = rr + 半格
                直径, 切过 = d, True
        return {"空": False, "多边形": P, "直径": float(直径), "Rmin": float(Rmin),
                "圆心": (float(圆心[0]), float(圆心[1])), "顶点数": len(P), "切过": 切过}

    # ------------------------------------------------------------ 定向专用查询
    def 收信锥(self, x):
        """在候选源位置 x 处，由已成功检测点张成的**保证覆盖锥**的方位角区间（引理 D1）。

        返回 (中心方位角, 半张角) 或 None（只有一个信号点时半张角为 0，即一条射线）。
        """
        if not self.信号点:
            return None
        角 = [方位角(x, s) for s, _ in self.信号点]
        跨 = 张角跨度(角)
        if 跨 >= 180.0 - 1e-9:
            return None                      # 信号点跨过半平面，锥不存在（该位置不可行）
        基 = 角[0]
        偏 = [float(角差(a, 基)) for a in 角]
        中心 = (基 + (min(偏) + max(偏)) / 2.0) % 360.0
        return 中心, 跨 / 2.0

    def 保证距离界(self, x):
        """在候选源位置 x 处，可以**保证**能收到信号的最大距离（R_eff 的已知下界）。

        R_eff >= 1000（题给），且每个曾收到信号的点 s 都给出 R_eff >= |s - x|。
        """
        界 = R_RECV_MIN
        for s, _ in self.信号点:
            界 = max(界, 距(s, x))
        return 界


# ================================================================ 五、最小覆盖圆（点集版）
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
    if 圆 is None:
        最优 = ((0.0, 0.0), np.inf)
        for i in range(3):
            for j in range(i + 1, 3):
                c = _平凡圆([边界[i], 边界[j]])
                if c[1] < 最优[1] and all(_圆含(c, p) for p in 边界):
                    最优 = c
        return 最优
    return 圆


def 最小覆盖圆_Welzl(点集, rng=None):
    """Welzl 随机增量算法，期望 O(n)。点数上千时请先取凸包再调用。"""
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
    P = np.asarray(点集, dtype=float).reshape(-1, 2)
    if len(P) <= 3:
        return 最小覆盖圆_Welzl(P)
    try:
        from scipy.spatial import ConvexHull
        P = P[ConvexHull(P).vertices]
    except Exception:
        pass
    return 最小覆盖圆_Welzl(P)


# ================================================================ 六、代价模型
def 移动时间(距离):
    return float(距离) / V_MOVE


def 停靠代价(当前位置, 目标位置, 频道表, 当前频道):
    """走到目标位置并依次检测频道表的虚拟耗时（秒）。切频只看相邻两次 /measure。"""
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


# ================================================================ 七、巡回路线
def 最近邻路线(起点, 点表):
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
    """开放路径的 2-opt 改良（反转一段子路径）。

    **实现上与问题三的差别只在常数因子，但这个常数因子是必须的。**
    问题三每步要排的点不过十来个，直接「重排一次、整条路重算长度」($O(n)$ 每次试探)
    也能跑；问题四的任务表里挤着二三十个围堵锚点，$O(n^3)$ 的写法在剖析里
    占掉了单局一半以上的算力。这里改成**预先算好距离矩阵、只算反转带来的增量**，
    每次试探 $O(1)$：

        反转 [i..j] 的增量 = d(前, v_j) + d(v_i, 后) - d(前, v_i) - d(v_j, 后)

    其中「前」是 i 的前驱（i = 0 时为起点），「后」是 j 的后继（j = n-1 时不存在，
    开放路径的末端没有回程边，这一项为 0）。
    """
    顺序 = list(顺序)
    n = len(顺序)
    if n < 3:
        return 顺序
    P = np.vstack([np.asarray(起点, dtype=float).reshape(1, 2),
                   np.asarray(点表, dtype=float).reshape(-1, 2)])
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    路 = [0] + [i + 1 for i in 顺序]            # 0 号是起点
    for _ in range(最大轮):
        改进 = False
        m = len(路)
        for i in range(1, m - 1):
            前 = 路[i - 1]
            for j in range(i + 1, m):
                后 = 路[j + 1] if j + 1 < m else None
                增 = D[前, 路[j]] - D[前, 路[i]]
                if 后 is not None:
                    增 += D[路[i], 后] - D[路[j], 后]
                if 增 < -1e-9:
                    路[i:j + 1] = 路[i:j + 1][::-1]
                    前 = 路[i - 1]
                    改进 = True
        if not 改进:
            break
    return [k - 1 for k in 路[1:]]


def 开放TSP(起点, 点表):
    """最近邻 + 2-opt。返回 (顺序, 长度)。"""
    if not len(点表):
        return [], 0.0
    顺序 = 最近邻路线(起点, 点表)
    if 启用2opt:
        顺序 = 二opt(起点, 点表, 顺序)
    return 顺序, 路线长度(起点, 点表, 顺序)


def 最小生成树长度(点表, 收缩=0.0, 起点收缩=0.0, 起点=(0.0, 0.0)):
    """{起点} ∪ 点表 的最小生成树总长（Prim）。用作巡回长度的**严格下界**。"""
    P = [tuple(map(float, 起点))] + [tuple(map(float, p)) for p in 点表]
    n = len(P)
    if n <= 1:
        return 0.0
    A = np.asarray(P, dtype=float)
    D = np.hypot(A[:, None, 0] - A[None, :, 0], A[:, None, 1] - A[None, :, 1])
    D = np.maximum(D - 收缩, 0.0)
    D[0, :] = np.maximum(np.hypot(A[:, 0] - A[0, 0], A[:, 1] - A[0, 1]) - 起点收缩, 0.0)
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
