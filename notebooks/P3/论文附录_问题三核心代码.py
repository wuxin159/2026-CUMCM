"""问题三核心代码（由 论文附录代码.py 自动抽取，勿手改）。

完整可运行的全部脚本见支撑材料 notebooks/P3/。
本文件只收「计算求解」部分；绘图脚本 绘图.py 不进附录。
"""
import numpy as np


# ========================================================================
# 集员估计：三类新约束的构造与凸可行集
# 出处：notebooks/P3/公共.py
# ========================================================================
class 频道信念:
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

# ========================================================================
# 非凸排除盘的保守线性化（定理 4 的引理）
# 出处：notebooks/P3/公共.py
# ========================================================================
def 圆外保守半平面(多边形, 圆心, 半径):
    """把非凸约束 |x - 圆心| > 半径 在给定多边形上**保守地**线性化。"""
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

# ========================================================================
# 判空栅格：只有整格落入排除盘才划掉
# 出处：notebooks/P3/公共.py
# ========================================================================
class 栅格:
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

    def 排除圆盘(self, 掩码, 圆心, 半径):
        """保守地划掉整格落入圆盘的方格；返回新掩码。"""
        d = np.hypot(self.点[:, 0] - 圆心[0], self.点[:, 1] - 圆心[1])
        return 掩码 & (d > 半径 - self.裕度)

# ========================================================================
# 判空锚点的「覆盖每米」贪心
# 出处：notebooks/P3/策略.py
# ========================================================================
class 锚点规划器:
    def 插入代价(self, 路线):
        """把每个候选点插进给定开放路线的最小边际长度。路线首元素是当前位置。"""
        V = np.asarray(路线, dtype=float).reshape(-1, 2)
        P = self.候选
        最小 = np.hypot(P[:, 0] - V[-1, 0], P[:, 1] - V[-1, 1])      # 接在末尾
        for i in range(len(V) - 1):
            a, b = V[i], V[i + 1]
            ab = float(np.hypot(*(b - a)))
            d = (np.hypot(P[:, 0] - a[0], P[:, 1] - a[1]) +
                 np.hypot(P[:, 0] - b[0], P[:, 1] - b[1]) - ab)
            最小 = np.minimum(最小, d)
        return 最小

    def 贪心锚点(self, 掩码, 上限=14, 预扣点=(), 当前位置=None, 近邻优先=0.0,
                 准则="覆盖", 每米尺度=400.0, 路线=None):
        """贪心最大覆盖。预扣点是「反正要去的停靠点」，先记它们的覆盖再选锚点。"""
        剩 = self.子掩码(掩码)
        子点 = self.栅格.点[self.子索引]
        for p in 预扣点:
            d = np.hypot(子点[:, 0] - p[0], 子点[:, 1] - p[1])
            剩 = 剩 * (d > self.规划半径)
        出 = []
        代价 = (self.插入代价(路线) if (准则 == "覆盖每米" and 路线 is not None)
                else None)
        while 剩.sum() > 0 and len(出) < 上限:
            计数 = self.关联 @ 剩
            if 代价 is not None:
                # 覆盖得多不等于划算：除以「把它插进现有路线要多走多少米」
                得分 = 计数 / (每米尺度 + 代价)
                i = int(np.argmax(得分))
                if 计数[i] <= 0:
                    合格 = np.where(计数 > 0)[0]
                    if not len(合格):
                        break
                    i = int(合格[int(np.argmax(得分[合格]))])
                出.append(tuple(self.候选[i]))
                剩 = 剩 * (1.0 - self.关联[i])
                continue
            i = int(np.argmax(计数))
            if 计数[i] <= 0:
                break                                  # 残余落在所有候选点的射程外
            if 近邻优先 > 0.0 and 当前位置 is not None:
                合格 = np.where(计数 >= (1.0 - 近邻优先) * 计数[i])[0]
                if len(合格):
                    d = np.hypot(self.候选[合格, 0] - 当前位置[0],
                                 self.候选[合格, 1] - 当前位置[1])
                    i = int(合格[int(np.argmin(d))])
            出.append(tuple(self.候选[i]))
            剩 = 剩 * (1.0 - self.关联[i])
        # 去冗余：倒序尝试删除，仍能覆盖就删
        改 = True
        while 改 and len(出) > 1:
            改 = False
            for k in range(len(出) - 1, -1, -1):
                试 = 出[:k] + 出[k + 1:]
                剩 = self.子掩码(掩码)
                for p in list(预扣点) + 试:
                    d = np.hypot(子点[:, 0] - p[0], 子点[:, 1] - p[1])
                    剩 = 剩 * (d > self.规划半径)
                if 剩.sum() == 0:
                    出, 改 = 试, True
                    break
        # **收尾兜底**：子点阵上已经没东西可覆盖，但完整栅格上还剩几格。
        if len(self.残余点(掩码, tuple(预扣点) + tuple(出))):
            出 = list(出) + self.收尾锚点(掩码, tuple(预扣点) + tuple(出))
        return 出

# ========================================================================
# 不依赖分布的收信把握下界
# 出处：notebooks/P3/策略.py
# ========================================================================
def 保证收信比例(信念, m, q):
    """在 q 处**保证**能收到该频道信号的把握，用可行集里被保证覆盖的比例度量。"""
    P = np.asarray(m["多边形"], dtype=float)
    if len(P) == 0:
        return 0.0
    采样 = [P.mean(axis=0)] + [tuple(p) for p in P]
    if len(P) >= 2:                                  # 边中点，让长条形也有代表点
        采样 += [tuple((P[i] + P[(i + 1) % len(P)]) / 2) for i in range(len(P))]
    信号点 = [np.asarray(s, dtype=float) for s, _ in 信念.信号点]
    好 = 0
    for x in 采样:
        x = np.asarray(x, dtype=float)
        界 = C.R_RECV_MIN
        for s in 信号点:
            界 = max(界, float(np.hypot(*(s - x))))
        if float(np.hypot(q[0] - x[0], q[1] - x[1])) <= 界:
            好 += 1
    return 好 / len(采样)

# ========================================================================
# 一体化滚动时域主循环（算法 1 的实现）
# 出处：notebooks/P3/策略.py
# ========================================================================
def 推荐策略(狗, 参数=None, 采集轨迹=False):
    """一体化滚动时域策略。返回统计字典。"""
    参数 = dict(默认参数, **(参数 or {}))
    st = 状态(参数)
    轨迹 = []
    狗.进入()
    动作 = 0
    while 动作 < 参数["最多动作"]:
        if st.全部解决():
            break
        if 狗.现实剩余 < 参数["现实时间留白"]:
            st.异常.append("现实时间不足，提前收尾")
            break
        位置 = (float(狗.位置[0]), float(狗.位置[1]))

        # ---------- A. 就地整池扫描（未探测频道） ----------
        if 值得扫描(st, 位置, 参数):
            扫描(狗, st, 位置)
            动作 += 1
            if 采集轨迹:
                轨迹.append(("扫描", *位置, 狗.虚拟时刻))
            continue

        # ---------- B. 就地补测（已探测未定位频道） ----------
        做了 = False
        for c in st.已探测未清除():
            m = st.指标(c)
            if m is None or m.get("空"):
                continue
            if m["Rmin"] <= 参数["盲清阈值"]:
                continue
            if st._键(位置, c) in st.已测:
                continue
            p收 = (保证收信比例(st.信念[c], m, 位置)
                   if 参数["用收信把握"] else 1.0)
            if p收 <= 0.0:
                continue
            d, R = 预演(st.信念[c], 位置, 参数["delta"])
            # 期望相对收缩：收不到信号的那部分把握不计入收益。
            if p收 * (1.0 - d / max(m["直径"], 1e-9)) >= 1.0 - 参数["精测增益γ"]:
                回 = 狗.检测(位置, c)
                st.记录检测(位置, c, 回)
                st.计数["补测检测"] += 1
                st.计数["就地补测"] += 1
                动作 += 1
                做了 = True
                break
        if 做了:
            continue

        # ---------- C. 就地清除 ----------
        目标 = 就地清除目标(st, 位置, 参数)
        if 目标 is not None:
            c, 点 = 目标
            回 = 狗.清除(点, c)
            st.记录清除(点, c, 回)
            动作 += 1
            if 采集轨迹:
                轨迹.append(("清除", *点, 狗.虚拟时刻))
            continue

        # ---------- D. 规划下一站并执行第一段 ----------
        任务 = 构造任务(st, 位置, 参数)
        if not 任务:
            # 走到这里说明规划器没给出任何任务。若此时还有频道既没清除、
            if not st.全部解决():
                st.异常.append("任务表为空但仍有频道未决，被迫退出")
            break
        点表 = [t["点"] for t in 任务]
        顺序, _ = C.开放TSP(位置, 点表)
        第一 = 任务[顺序[0]]
        if 第一["类型"] == "清除":
            回 = 狗.清除(第一["点"], 第一["频道"])
            st.记录清除(第一["点"], 第一["频道"], 回)
            if 采集轨迹:
                轨迹.append(("清除", *第一["点"], 狗.虚拟时刻))
        elif 第一["类型"] == "锚点":
            st.计数["锚点停靠"] += 1
            扫描(狗, st, 第一["点"])
            if 采集轨迹:
                轨迹.append(("锚点", *第一["点"], 狗.虚拟时刻))
        else:
            回 = 狗.检测(第一["点"], 第一["频道"])
            st.记录检测(第一["点"], 第一["频道"], 回)
            st.计数["补测检测"] += 1
            st.计数["专程补测"] += 1
            if 采集轨迹:
                轨迹.append(("补测", *第一["点"], 狗.虚拟时刻))
        动作 += 1
    狗.退出()
    return 收尾统计(狗, st, 轨迹 if 采集轨迹 else None)

# ========================================================================
# 严格下界与全知参照
# 出处：notebooks/P3/策略.py
# ========================================================================
def 理论下界(案例, 空频道判空次数=7, 参数=None):
    """两条参照线。**它们的性质不同，论文里不能混为一谈。**"""
    参数 = dict(默认参数, **(参数 or {}))
    位置 = 案例.位置表()
    空 = C.频道数 - 案例.源数
    清除 = 案例.源数 * C.T_CLEAR_OK

    # ① 严格下界
    MST = C.最小生成树长度(位置, 收缩=2 * C.R_CLEAR, 起点收缩=C.R_CLEAR)
    严格判空检测 = 空频道判空次数 * 空 * C.T_MEASURE        # 不算切频，保证是下界
    严格下界 = MST / C.V_MOVE + 清除 + 严格判空检测

    # ② 全知参照
    _, L1 = C.开放TSP((0.0, 0.0), 位置)
    st = 状态(参数)
    补锚 = st.规划器.贪心锚点(st.残余联合(), 预扣点=tuple(位置))
    _, L2 = C.开放TSP((0.0, 0.0), list(位置) + list(补锚))
    判空检测 = 空频道判空次数 * 空 * (C.T_MEASURE + C.T_SWITCH)
    参照 = L2 / C.V_MOVE + 清除 + 判空检测
    return {"MST路程": MST, "严格行进时间": MST / C.V_MOVE,
            "严格判空检测": 严格判空检测,
            "严格下界": 严格下界, "严格下界平均": 严格下界 / 案例.源数,
            "TSP路程": L1, "行进时间": L1 / C.V_MOVE,
            "清除时间": 清除, "判空检测时间": 判空检测,
            "下界总时间": 参照 - (L2 - L1) / C.V_MOVE,   # 只含源点巡回的参照
            "下界平均": (参照 - (L2 - L1) / C.V_MOVE) / 案例.源数,
            "补锚点数": len(补锚), "含判空TSP路程": L2,
            "下界2总时间": 参照, "下界2平均": 参照 / 案例.源数}
