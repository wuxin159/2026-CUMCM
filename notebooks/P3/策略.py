"""问题三 · 第 4 步：搜索定位清除策略（只算数，不作图）。"""
import numpy as np

import 公共 as C

# ---------------------------------------------------------------- 默认参数
默认参数 = {
    "delta": C.DELTA_算法,      # 算法侧楔形半角（含两位小数舍入裕度）
    "栅格步长": 20.0,           # 判空栅格步长
    "锚点间距": 150.0,          # 锚点候选点阵间距
    # ζ 与 γ 由 参数搜索.py 的配对统计定下（独立样本确认 -126.7 ± 24.1 s，t = -5.27）。
    "扫描增益ζ": 1.40,          # 在此处整池扫描的门槛：新增覆盖 >= ζ x 残余/还需锚点数
    "精测增益γ": 0.50,          # 补测门槛：期望相对收缩 >= 1 - γ
    "盲清阈值": 28.0,           # R_min <= 该值时直接去清（必要时多试几次），否则先补测
    "盲清上限": 4,              # 同一频道连续盲清的次数上限，超过则强制补测
    "补测λ": (0.30, 0.50, 0.70),   # 补测点沿长轴的相对位置
    "补测κ": (0.20, 0.35, 0.55),   # 补测点的侧移量，单位「区域半长」
    "补测最小偏移": 12.0,       # 侧移的下限，米
    "空测罚": 90.0,             # 「这一测可能收不到信号」的等效秒数罚，见 选补测点
    "用收信把握": True,         # 决策时是否计入「在这里到底收不收得到」，见 §5.4
    "预扣覆盖": True,           # 选锚点前先扣掉「反正要去的停靠点」自带的覆盖
    "补测任务上限": float("inf"),  # 区域大于它就不专门跑一趟补测，等锚点扫描顺带解决
    # 锚点选择准则。"覆盖每米" 明显更好（配对差 -291 ± 60 s，t = -4.85，24 局里 20 局改善），
    "锚点准则": "覆盖每米",     # "覆盖" 只看新增覆盖；"覆盖每米" 除以插入路线的边际代价
    "锚点每米尺度": 400.0,      # "覆盖每米" 的分母常数，单位米（200~800 无差别）
    "锚点近邻优先": 0.15,       # 仅当 锚点准则="覆盖" 时生效：覆盖相近则取近的
    "最多动作": 1500,
    "现实时间留白": 20.0,       # 剩余现实时间少于该值就直接收尾
}


# ================================================================ 锚点规划器
class 锚点规划器:
    """判空锚点的候选点阵与贪心覆盖。"""

    def __init__(self, 栅格, 间距=150.0, 半径=C.R_RECV_MIN):
        self.栅格 = 栅格
        self.半径有效 = 半径 - 栅格.裕度
        # 规划只看 40 m 子点阵（为了快），而判空判据看完整 20 m 栅格。
        self.子点最大偏差 = 栅格.步长 * np.sqrt(2.0)
        self.规划半径 = self.半径有效 - self.子点最大偏差
        # 候选点阵：六角排布，覆盖到目标圆外 100 m（题面允许在区域外检测）
        点 = []
        行距 = 间距 * np.sqrt(3) / 2
        j = 0
        y = -(C.R_AREA + 100.0)
        while y <= C.R_AREA + 100.0:
            x = -(C.R_AREA + 100.0) + (间距 / 2 if j % 2 else 0.0)
            while x <= C.R_AREA + 100.0:
                if np.hypot(x, y) <= C.R_AREA + 100.0:
                    点.append((x, y))
                x += 间距
            y += 行距
            j += 1
        self.候选 = np.asarray(点, dtype=float)
        # 子采样：20 m 栅格上每隔一格取一个，得到 40 m 子点阵
        步 = 栅格.步长
        idx = np.where((np.round(栅格.点[:, 0] / 步).astype(int) % 2 == 0) &
                       (np.round(栅格.点[:, 1] / 步).astype(int) % 2 == 0))[0]
        self.子索引 = idx
        子点 = 栅格.点[idx]
        d = np.sqrt(((子点[None, :, :] - self.候选[:, None, :]) ** 2).sum(axis=2))
        self.关联 = (d <= self.规划半径).astype(np.float32)

    def 子掩码(self, 掩码):
        return 掩码[self.子索引].astype(np.float32)

    def 残余数(self, 掩码):
        return int(掩码[self.子索引].sum())

    def 新增覆盖(self, 掩码, p):
        d = np.hypot(self.栅格.点[self.子索引, 0] - p[0],
                     self.栅格.点[self.子索引, 1] - p[1])
        return int(np.count_nonzero(掩码[self.子索引] & (d <= self.规划半径)))

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

    def 残余点(self, 掩码, 预扣点=()):
        """完整栅格上还没被覆盖的方格中心（扣掉预扣点自带的覆盖）。"""
        点 = self.栅格.点[掩码]
        for p in 预扣点:
            if not len(点):
                break
            点 = 点[np.hypot(点[:, 0] - p[0], 点[:, 1] - p[1]) > self.半径有效]
        return 点

    def 收尾锚点(self, 掩码, 预扣点=(), 上限=6):
        """**只在完整栅格上**为最后残下的那几格挑锚点。"""
        剩 = self.残余点(掩码, 预扣点)
        出 = []
        while len(剩) and len(出) < 上限:
            d = np.sqrt(((剩[None, :, :] - self.候选[:, None, :]) ** 2).sum(axis=2))
            计数 = (d <= self.半径有效).sum(axis=1)
            i = int(np.argmax(计数))
            if 计数[i] == 0:                       # 候选点阵够密，正常到不了这里
                出.append((float(剩[0][0]), float(剩[0][1])))
                剩 = 剩[1:]
                continue
            出.append((float(self.候选[i][0]), float(self.候选[i][1])))
            剩 = 剩[d[i] > self.半径有效]
        return 出

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


_规划器缓存 = {}


def 取规划器(栅格, 间距):
    键 = (栅格.步长, 间距)
    if 键 not in _规划器缓存:
        _规划器缓存[键] = 锚点规划器(栅格, 间距)
    return _规划器缓存[键]


# ================================================================ 信念状态
class 状态:
    """策略侧维护的全部信息。**不含任何真值**。"""

    def __init__(self, 参数):
        self.参数 = 参数
        self.栅格 = C.栅格(参数["栅格步长"])
        self.规划器 = 取规划器(self.栅格, 参数["锚点间距"])
        self.信念 = {c: C.频道信念(c, 参数["delta"]) for c in range(1, C.频道数 + 1)}
        # 每个频道**各有一张**判空栅格掩码。原先只用一张公共掩码，前提是
        self.掩码 = {c: self.栅格.新掩码() for c in range(1, C.频道数 + 1)}
        self.已清除 = set()
        self.已测 = set()                 # (x, y, 频道) 去重：同点同频道重测零信息
        self.已扫点 = set()
        self.盲清计数 = {c: 0 for c in range(1, C.频道数 + 1)}
        self.异常 = []
        self.计数 = {"扫描停靠": 0, "扫描检测": 0, "补测检测": 0,
                     "就地补测": 0, "专程补测": 0, "锚点停靠": 0, "清除停靠": 0}
        self._指标缓存 = {}
        self._锚点缓存 = None
        self._掩码版本 = 0
        self._联合缓存 = None

    # ---------------------------------------------------------- 查询
    @property
    def 未探测(self):
        """还没收到过信号、也没被清除的频道。它们要靠「覆盖判空」才能收尾。"""
        return {c for c in range(1, C.频道数 + 1)
                if c not in self.已清除 and not self.信念[c].收到信号}

    def 判空集(self):
        return {c for c in self.未探测 if not np.any(self.掩码[c])}

    def 残余联合(self):
        """还需要覆盖的方格：各未探测频道残余掩码的并。锚点规划以它为准。"""
        if self._联合缓存 is not None and self._联合缓存[0] == self._掩码版本:
            return self._联合缓存[1]
        待 = [self.掩码[c] for c in self.未探测]
        m = np.zeros(self.栅格.n, dtype=bool) if not 待 else np.logical_or.reduce(待)
        self._联合缓存 = (self._掩码版本, m)
        return m

    def 全部解决(self):
        for c in range(1, C.频道数 + 1):
            if c in self.已清除:
                continue
            if self.信念[c].收到信号:
                return False
            if np.any(self.掩码[c]):
                return False
        return True

    def 指标(self, c):
        b = self.信念[c]
        键 = (c, b.检测次数, len(b.失败清除点), id(b))
        if self._指标缓存.get("键" + str(c)) != 键:
            self._指标缓存["键" + str(c)] = 键
            self._指标缓存[c] = b.指标()
        return self._指标缓存[c]

    def 已探测未清除(self):
        return [c for c in range(1, C.频道数 + 1)
                if c not in self.已清除 and self.信念[c].收到信号]

    def 锚点(self, 预扣点=(), 当前位置=None, 路线=None):
        近 = self.参数.get("锚点近邻优先", 0.0)
        准则 = self.参数.get("锚点准则", "覆盖")
        键 = (self._掩码版本, tuple(sorted(预扣点)), 准则,
              None if (近 <= 0 and 准则 == "覆盖") or 当前位置 is None
              else (round(当前位置[0], -1), round(当前位置[1], -1)))
        if self._锚点缓存 is None or self._锚点缓存[0] != 键:
            self._锚点缓存 = (键, self.规划器.贪心锚点(
                self.残余联合(), 预扣点=预扣点, 当前位置=当前位置,
                近邻优先=近, 准则=准则,
                每米尺度=self.参数.get("锚点每米尺度", 400.0),
                路线=路线))
        return self._锚点缓存[1]

    # ---------------------------------------------------------- 更新
    def _键(self, p, c):
        return (round(float(p[0]), 3), round(float(p[1]), 3), int(c))

    def 记录检测(self, p, c, 回):
        self.已测.add(self._键(p, c))
        r = 回["measure_result"]
        self.信念[c].记录检测(p, r, 回.get("svd_deg"))
        if r == "no_signal":
            # 无信号 => 距离 > R_eff >= 1000，排除以 p 为心、半径 1000 m 的圆盘
            self.掩码[c] = self.栅格.排除圆盘(self.掩码[c], p, C.R_RECV_MIN)
            self._掩码版本 += 1

    def 记录扫描停靠(self, p):
        self.计数["扫描停靠"] += 1
        self.已扫点.add((round(float(p[0]), 1), round(float(p[1]), 1)))

    def 重建掩码(self, c):
        """按该频道**自己的**无信号点重建残余掩码（重置信念时用）。"""
        m = self.栅格.新掩码()
        for p in self.信念[c].无信号点:
            m = self.栅格.排除圆盘(m, p, C.R_RECV_MIN)
        self.掩码[c] = m
        self._掩码版本 += 1

    def 记录清除(self, p, c, 回):
        if 回["clear_result"] == "success":
            self.信念[c].记录清除成功()
            self.已清除.add(c)
        else:
            self.信念[c].记录清除失败(p)
            self.盲清计数[c] += 1


# ================================================================ 几何：预演与补测点
def 预演(信念, q, delta):
    """假设在 q 处收到该频道信号、且示向度恰好指向当前区域形心，问区域会缩成什么样。"""
    m = 信念.指标()
    if m is None or m.get("空"):
        return np.inf, np.inf
    P = m["多边形"]
    形心 = np.asarray(P, dtype=float).mean(axis=0)
    if C.距(q, 形心) < 1e-6:
        return m["直径"], m["Rmin"]
    θ = C.方位角(q, 形心)
    新 = list(P)
    for a, b, c in C.几何.楔形半平面(q, θ, delta):
        新 = C.几何.裁剪(新, a, b, c)
        if not 新:
            return np.inf, np.inf
    新 = C.几何.去冗余(新)
    if len(新) < 2:
        return 0.0, 0.0
    d, _ = C.P2模型.求直径快(新)
    _, R = C.P2模型.最小覆盖圆快(新)
    return float(d), float(R)


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


def 补测点候选(m, 当前位置, 参数, 信念=None):
    """给出若干候选补测点：沿区域长轴取基点、再朝靠近机器狗的一侧作正横侧移。"""
    P = np.asarray(m["多边形"], dtype=float)
    D = m["直径"]
    圆心 = np.asarray(m["圆心"], dtype=float)
    _, (A, B) = C.P2模型.求直径快([tuple(p) for p in P])
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    轴 = B - A
    L = float(np.hypot(*轴))
    if L < 1e-9:
        return [tuple(圆心 + np.array([参数["补测最小偏移"], 0.0]))]
    轴 = 轴 / L
    n = np.array([-轴[1], 轴[0]])
    if float(n @ (np.asarray(当前位置, dtype=float) - 圆心)) < 0:
        n = -n
    Λ = D / 2.0
    出 = []
    for λ in 参数["补测λ"]:
        基 = A + λ * (B - A) if D > 120.0 else 圆心
        for κ in 参数["补测κ"]:
            偏 = max(κ * Λ, 参数["补测最小偏移"])
            for s in (1.0, -1.0):
                q = 基 + s * 偏 * n
                出.append((float(q[0]), float(q[1])))
    # 区域很长时，正横侧移点可能离区域另一端太远而收不到信号。
    for λ in (0.15, 0.35, 0.55, 0.75):
        基 = A + λ * (B - A)
        for 偏 in (max(0.12 * Λ, 参数["补测最小偏移"]), 60.0, 150.0):
            q = 基 + 偏 * n
            出.append((float(q[0]), float(q[1])))
    return 出


def 选补测点(信念, m, 当前位置, 参数):
    """在候选里挑一个：既要一次把区域压进清除半径，又不能为此多绕路，还得收得到。"""
    最好 = (np.inf, None, None)
    for q in 补测点候选(m, 当前位置, 参数, 信念):
        d, R = 预演(信念, q, 参数["delta"])
        p收 = 保证收信比例(信念, m, q) if 参数["用收信把握"] else 1.0
        if p收 <= 0.0:
            continue
        行程 = C.距(当前位置, q) + C.距(q, m["圆心"])
        罚 = 0.0 if R <= C.R_CLEAR else (
            30.0 if R <= 参数["盲清阈值"] else 60.0 + min(R, 400.0))
        罚 += (1.0 - p收) * 参数["空测罚"]
        分 = 行程 / C.V_MOVE + C.T_MEASURE + 罚
        if 分 < 最好[0]:
            最好 = (分, q, R)
    if 最好[1] is None:                          # 全都可能收不到：退回贴近区域中心
        方向 = np.asarray(m["圆心"], dtype=float) - np.asarray(当前位置, dtype=float)
        n = float(np.hypot(*方向))
        q = (np.asarray(当前位置, dtype=float) + 方向 * min(1.0, 800.0 / max(n, 1e-9))
             ) if n > 1e-9 else np.asarray(m["圆心"], dtype=float)
        return (float(q[0]), float(q[1])), np.inf
    return 最好[1], 最好[2]


# ================================================================ 推荐策略
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


def 值得扫描(st, p, 参数):
    """在 p 处把未探测池整池测一遍值不值。"""
    if not st.未探测:
        return False
    if (round(p[0], 1), round(p[1], 1)) in st.已扫点:
        return False
    联合 = st.残余联合()
    残 = st.规划器.残余数(联合)
    if 残 == 0:
        return False
    新增 = st.规划器.新增覆盖(联合, p)
    if 新增 <= 0:
        return False
    还需 = len(st.锚点())
    if 还需 <= 1:
        return True
    return 新增 >= 参数["扫描增益ζ"] * 残 / 还需


def 扫描(狗, st, p):
    """在 p 处把当前未探测池整池测一遍。"""
    池 = sorted(st.未探测)
    for c in C.排序检测频道(池, 狗.频道):
        if st._键(p, c) in st.已测:
            continue
        回 = 狗.检测(p, c)
        st.记录检测(p, c, 回)
        st.计数["扫描检测"] += 1
    st.记录扫描停靠(p)


def 就地清除目标(st, 位置, 参数):
    """脚下就能打的目标：已定位且清除点离当前位置不超过半米。"""
    for c in st.已探测未清除():
        m = st.指标(c)
        if m is None or m.get("空"):
            continue
        if m["Rmin"] <= C.R_CLEAR and C.距(位置, m["圆心"]) <= 0.5:
            return c, m["圆心"]
    return None


def 构造任务(st, 位置, 参数):
    """把当前所有「必须去的地方」列出来，交给 TSP 排序。"""
    任务, 缓 = [], []
    for c in st.已探测未清除():
        m = st.指标(c)
        if m is None or m.get("空"):
            st.异常.append(f"频道{c}可行集为空")
            继续 = 重置信念(st, c)
            if not 继续:
                continue
            m = st.指标(c)
            if m is None or m.get("空"):
                continue
        if m["Rmin"] <= C.R_CLEAR or (
                m["Rmin"] <= 参数["盲清阈值"] and
                st.盲清计数[c] < 参数["盲清上限"]):
            任务.append({"类型": "清除", "点": m["圆心"], "频道": c})
        else:
            q, _ = 选补测点(st.信念[c], m, 位置, 参数)
            if q is not None:
                项 = {"类型": "补测", "点": q, "频道": c}
                (缓 if m["直径"] > 参数["补测任务上限"] else 任务).append(项)
    if np.any(st.残余联合()):
        预扣 = tuple(t["点"] for t in 任务) if 参数["预扣覆盖"] else ()
        路线 = [位置]
        if 任务:
            顺 , _ = C.开放TSP(位置, [t["点"] for t in 任务])
            路线 += [任务[i]["点"] for i in 顺]
        for a in st.锚点(预扣点=预扣, 当前位置=位置, 路线=路线):
            任务.append({"类型": "锚点", "点": a, "频道": None})
    if not 任务:
        任务 = 缓                                  # 别把自己饿死：没别的可做就去补测
    return 任务


def 重置信念(st, c):
    """可行集变空说明模型与读数矛盾（理论上不该发生，但实测程序必须能兜住）。"""
    旧 = st.信念[c]
    新 = C.频道信念(c, st.参数["delta"])
    新.无信号点 = list(旧.无信号点)
    新.失败清除点 = list(旧.失败清除点)
    新.异常 = 旧.异常
    st.信念[c] = 新
    st.重建掩码(c)                       # 退回未探测状态，但用**它自己的**无信号历史
    st._指标缓存.pop(c, None)
    st._指标缓存.pop("键" + str(c), None)
    return False


def 收尾统计(狗, st, 轨迹):
    return {
        "虚拟总时间": 狗.虚拟时刻,
        "检测次数": 狗.检测次数,
        "清除次数": 狗.清除次数,
        "清除成功": 狗.清除成功,
        "判空频道数": len(st.判空集()),
        "计数": dict(st.计数),
        "异常": list(st.异常),
        "状态": st,
        "轨迹": 轨迹,
    }


# ================================================================ 基线一：先扫后清
def 基线_先扫后清(狗, 参数=None, 锚点表=None):
    """先把判空锚点走一遍（每站测遍所有存活频道），再对已定位的源做 TSP 清除。"""
    参数 = dict(默认参数, **(参数 or {}))
    st = 状态(参数)
    狗.进入()
    if 锚点表 is None:
        锚点表 = st.锚点()
        顺序, _ = C.开放TSP((0.0, 0.0), 锚点表)
        锚点表 = [锚点表[i] for i in 顺序]
    # 阶段一：锚点扫描（原点先扫一遍，反正不用走路）
    for p in [(0.0, 0.0)] + list(锚点表):
        池 = sorted(set(range(1, C.频道数 + 1)) - st.已清除)
        for c in C.排序检测频道(池, 狗.频道):
            if st._键(p, c) in st.已测:
                continue
            回 = 狗.检测(p, c)
            st.记录检测(p, c, 回)
        st.记录扫描停靠(p)
    # 阶段二：对已探测频道做 TSP，逐个走到位、必要时补测、再清除
    while True:
        待 = [c for c in st.已探测未清除() if not (st.指标(c) or {}).get("空", True)]
        if not 待:
            break
        点表 = [st.指标(c)["圆心"] for c in 待]
        顺序, _ = C.开放TSP(tuple(狗.位置), 点表)
        c = 待[顺序[0]]
        护 = 0
        while 护 < 8:
            m = st.指标(c)
            if m is None or m.get("空"):
                break
            if m["Rmin"] <= C.R_CLEAR or st.盲清计数[c] >= 参数["盲清上限"]:
                回 = 狗.清除(m["圆心"], c)
                st.记录清除(m["圆心"], c, 回)
                if 回["clear_result"] == "success":
                    break
            else:
                q, _ = 选补测点(st.信念[c], m, tuple(狗.位置), 参数)
                if q is None or st._键(q, c) in st.已测:
                    回 = 狗.清除(m["圆心"], c)
                    st.记录清除(m["圆心"], c, 回)
                    if 回["clear_result"] == "success":
                        break
                else:
                    回 = 狗.检测(q, c)
                    st.记录检测(q, c, 回)
            护 += 1
        if 护 >= 8:
            st.异常.append(f"频道{c}收尾未收敛")
            st.已清除.add(c)                    # 避免死循环，记为放弃
    # 阶段三：残余判空
    护 = 0
    while np.any(st.残余联合()) and 护 < 12:
        锚 = st.锚点()
        if not 锚:
            break
        顺序, _ = C.开放TSP(tuple(狗.位置), 锚)
        扫描(狗, st, 锚[顺序[0]])
        护 += 1
    狗.退出()
    return 收尾统计(狗, st, None)


# ================================================================ 基线二：逐源处理
def 基线_逐源(狗, 参数=None):
    """原点测遍 20 个频道，然后一次只盯一个源，按 P2 的单源办法走完再换下一个。"""
    参数 = dict(默认参数, **(参数 or {}))
    st = 状态(参数)
    狗.进入()
    扫描(狗, st, (0.0, 0.0))
    护总 = 0
    while 护总 < 60:
        护总 += 1
        待 = [c for c in st.已探测未清除() if not (st.指标(c) or {}).get("空", True)]
        if 待:
            点表 = [st.指标(c)["圆心"] for c in 待]
            顺序, _ = C.开放TSP(tuple(狗.位置), 点表)
            c = 待[顺序[0]]
            护 = 0
            while 护 < 8:
                m = st.指标(c)
                if m is None or m.get("空"):
                    break
                if m["Rmin"] <= C.R_CLEAR or st.盲清计数[c] >= 参数["盲清上限"]:
                    回 = 狗.清除(m["圆心"], c)
                    st.记录清除(m["圆心"], c, 回)
                    if 回["clear_result"] == "success":
                        break
                else:
                    q, _ = 选补测点(st.信念[c], m, tuple(狗.位置), 参数)
                    if q is None or st._键(q, c) in st.已测:
                        回 = 狗.清除(m["圆心"], c)
                        st.记录清除(m["圆心"], c, 回)
                        if 回["clear_result"] == "success":
                            break
                    else:
                        回 = 狗.检测(q, c)
                        st.记录检测(q, c, 回)
                护 += 1
            if 护 >= 8:
                st.异常.append(f"频道{c}收尾未收敛")
                st.已清除.add(c)
            continue
        if np.any(st.残余联合()):
            锚 = st.锚点()
            if not 锚:
                break
            顺序, _ = C.开放TSP(tuple(狗.位置), 锚)
            扫描(狗, st, 锚[顺序[0]])
            continue
        break
    狗.退出()
    return 收尾统计(狗, st, None)


# ================================================================ 基线三：盲清搜索（解析估计）
def 盲清搜索代价(R=C.R_AREA, r=C.R_CLEAR):
    """完全不用示向度，只靠 20 m 清除盘铺满目标圆域的代价（解析估计）。"""
    单点面积 = 1.5 * np.sqrt(3.0) * r * r
    点数 = int(np.ceil(np.pi * R * R / 单点面积))
    列距 = np.sqrt(3.0) * r
    路程 = 点数 * 列距
    return {"清除点数": 点数, "路程": 路程,
            "行进时间": 路程 / C.V_MOVE,
            "清除时间": 点数 * C.T_CLEAR_FAIL,
            "总时间": 路程 / C.V_MOVE + 点数 * C.T_CLEAR_FAIL}


# ================================================================ 理论下界
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
