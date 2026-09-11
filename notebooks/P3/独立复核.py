"""问题三 · 模型检验：独立复核与变异测试（只算数，不作图）。

P1、P2 的教训是同一条：**复核代码不能调用被测代码**，否则构造错了两边一起错，
还会对得严丝合缝。本脚本的判据全部从题面定义直接重写，一行都不借用
`公共.频道信念`、`公共.栅格` 的实现。

六层检验：

    一  计时复刻：拿附件 2 §10 的完整算例逐步对时刻，再对随机指令序列独立重算虚拟时钟
    二  可行集含真源（**承重的一条**）：整局逐步检查真源是否始终在维护的可行集内
    三  判空不出错：策略每宣布一个频道为空，就核对该频道确实没有源
    四  超集性与保守度：过程中取快照，蒙特卡洛比对「按定义可行」与「模型可行」
    五  策略不接触真值：用只暴露四个方法的代理后端跑完整局
    六  变异测试：往关键常数与约束里植入 7 个错误，检查上面各层是否真的抓得住

输出：独立复核结果.txt、复核_保守度.csv、复核_变异测试.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 公共 as C
import 模拟器 as S
import 生成算例 as G
import 策略 as ST

HERE = Path(__file__).resolve().parent
SEED = 20260911


# ================================================================ 独立的可行性定义
def 按定义可行(点, 观测表, 失败清除点=(), delta=C.DELTA_算法, R=C.R_AREA):
    """一个位置能否解释全部观测。完全按题面写，不引用任何被测代码。

    观测表元素 = (检测位置, 结果, 示向度)，结果 ∈ {direction, near, no_signal}。

    题面给出的全部约束：
      · 干扰源在目标区域内                              |x| <= R
      · direction：|方位角(p,x) - svd| <= delta 且 d > 5 且 d <= R_eff
      · near：d <= 5
      · no_signal：d > R_eff
      · 存在性：R_eff 是该源的固有常数，必须存在一个 R_eff ∈ [1000,1500]
        同时满足上面所有的 d <= R_eff 与 d > R_eff
      · 清除未发现：d > 20
    最后一条「存在性」把序约束、1000 m 排除盘、1500 m 外界三件事一次性包含了，
    比逐条写更不容易漏。
    """
    x = np.asarray(点, dtype=float)
    if float(np.hypot(*x)) > R:
        return False
    有信号 = []
    无信号 = []
    for p, r, svd in 观测表:
        d = float(np.hypot(x[0] - p[0], x[1] - p[1]))
        if r == "direction":
            if d <= C.R_NEAR:
                return False
            真 = float(np.degrees(np.arctan2(x[1] - p[1], x[0] - p[0])) % 360.0)
            差 = (真 - float(svd) + 180.0) % 360.0 - 180.0
            if abs(差) > delta:
                return False
            有信号.append(d)
        elif r == "near":
            if d > C.R_NEAR:
                return False
            有信号.append(d)
        elif r == "no_signal":
            无信号.append(d)
        else:
            raise ValueError(r)
    for q in 失败清除点:
        if float(np.hypot(x[0] - q[0], x[1] - q[1])) <= C.R_CLEAR:
            return False
    低 = max([C.R_RECV_MIN] + 有信号)                 # R_eff 至少要这么大
    高 = min([C.R_RECV_MAX] + 无信号)                 # R_eff 必须严格小于每个无信号距离
    if 低 > C.R_RECV_MAX:
        return False
    if 无信号 and 低 >= min(无信号):
        return False
    return 低 <= 高


def 在模型可行集内(信念, 掩码, 栅格, 点, 已探测):
    """被测代码给出的可行集是否含该点。已探测用多边形，未探测用栅格掩码。"""
    if 已探测:
        m = 信念.指标()
        if m is None or m.get("空"):
            return False
        P = m["多边形"]
        if len(P) < 3:
            return bool(P) and C.距(P[0], 点) < 1.0
        return bool(C.几何.点在多边形内(tuple(点), P, 容差=1e-6))
    d = np.hypot(栅格.点[:, 0] - 点[0], 栅格.点[:, 1] - 点[1])
    i = int(np.argmin(d))
    return bool(掩码[i]) or d[i] > 栅格.步长          # 落在格心附近就看该格


# ================================================================ 一、计时复刻
def 检验计时():
    """先对附件 2 §10 的算例，再对随机指令序列独立重算虚拟时钟。"""
    print("=" * 88)
    print("一、本地模拟器的计时是否复刻正确")
    print("=" * 88)
    源 = [{"频道": 3, "x": 5000.0, "y": 0.0, "R_eff": 1000.0}]   # 放远处，保证不命中
    环境 = S.电磁环境(np.random.default_rng(0), 模式="零")
    后端 = S.本地后端(S.案例(源, 环境))
    后端.enter()
    步 = [("measure", 300, 400, 1, 105.0), ("measure", 300, 400, 2, 111.0),
          ("clear", 300, 0, 3, 194.0), ("measure", 300, 0, 2, 199.0)]
    print("    附件 2 §10 的完整计时算例（教材给的虚拟时刻写在最后一列）：")
    print(f"    {'指令':>10} {'位置':>14} {'频道':>5} {'本地模拟器':>12} "
          f"{'附件 2':>10} {'差':>8}")
    最大差 = 0.0
    for 名, x, y, ch, 期望 in 步:
        回 = 后端.measure(x, y, ch) if 名 == "measure" else 后端.clear(x, y, ch)
        差 = abs(回["virtual_time_s"] - 期望)
        最大差 = max(最大差, 差)
        print(f"    {'/' + 名:>10} {f'({x},{y})':>14} {ch:>5} "
              f"{回['virtual_time_s']:>12.1f} {期望:>10.1f} {差:>8.2e}")
    print(f"    -> 与附件 2 逐步吻合，最大差 {最大差:.2e} s")

    rng = np.random.default_rng(SEED)
    案 = G.造案例(rng, 源数=12)
    狗 = S.机器狗(S.本地后端(案), 现实时限=1e9)
    ST.推荐策略(狗)
    日志 = pd.DataFrame(狗.日志)
    # 独立重算：位置差 / 5 + 5 + 切频 1 + 清除 5 或 3
    t, 位置, 频道 = 0.0, np.zeros(2), 1
    最大差 = 0.0
    for _, 行 in 日志.iterrows():
        if 行["动作"] in ("enter", "exit"):
            continue
        p = np.array([行["x"], 行["y"]], dtype=float)
        t += float(np.hypot(*(p - 位置))) / 5.0
        位置 = p
        if 行["动作"] == "measure":
            if int(行["频道"]) != 频道:
                t += 1.0
                频道 = int(行["频道"])
            t += 5.0
        else:
            t += 5.0 if 行["结果"] == "success" else 3.0
        最大差 = max(最大差, abs(t - float(行["虚拟时刻"])))
    print(f"    整局 {len(日志)} 条指令独立重算虚拟时钟，与模拟器最大差 {最大差:.2e} s")
    return 最大差


# ================================================================ 二、可行集含真源
def 逐步含真源(案, 参数=None, 每步全查=True):
    """把策略跑一遍，每次观测之后立刻检查每个源是否还在自己频道的可行集里。

    实现方式：在 状态.记录检测 / 记录清除 外面套一层钩子。钩子只**读**，不改行为。
    """
    参数 = dict(ST.默认参数, **(参数 or {}))
    真值 = {int(s["频道"]): (s["x"], s["y"]) for s in 案.源表}
    违规 = []
    观测表 = {c: [] for c in range(1, C.频道数 + 1)}
    失败点 = {c: [] for c in range(1, C.频道数 + 1)}

    原记录检测 = ST.状态.记录检测
    原记录清除 = ST.状态.记录清除

    def 记录检测(self, p, c, 回):
        原记录检测(self, p, c, 回)
        观测表[c].append(((float(p[0]), float(p[1])), 回["measure_result"],
                          回.get("svd_deg")))
        查(self, c)

    def 记录清除(self, p, c, 回):
        原记录清除(self, p, c, 回)
        if 回["clear_result"] != "success":
            失败点[c].append((float(p[0]), float(p[1])))
        查(self, c)

    def 查(self, c):
        if c not in 真值 or c in self.已清除:
            return
        G0 = 真值[c]
        在模型 = 在模型可行集内(self.信念[c], self.掩码[c], self.栅格, G0,
                              self.信念[c].收到信号)
        按定义 = 按定义可行(G0, 观测表[c], 失败点[c], 参数["delta"])
        if not 按定义:
            违规.append(("真源与观测不相容", c, G0))       # 定义层就矛盾 => 误差超界
        if not 在模型:
            违规.append(("真源掉出模型可行集", c, G0))

    ST.状态.记录检测 = 记录检测
    ST.状态.记录清除 = 记录清除
    try:
        狗 = S.机器狗(S.本地后端(案), 现实时限=1e9)
        r = ST.推荐策略(狗, 参数)
    finally:
        ST.状态.记录检测 = 原记录检测
        ST.状态.记录清除 = 原记录清除
    return r, 违规


def 检验含真源(N=40, 参数=None, 静默=False):
    rng = np.random.default_rng(SEED + 100)
    坏局, 总违规, 清除率, 判空错 = 0, 0, [], 0
    for i in range(N):
        案 = G.造案例(rng)
        r, 违规 = 逐步含真源(案, 参数)
        if 违规:
            坏局 += 1
            总违规 += len(违规)
            if not 静默 and 坏局 <= 3:
                print(f"      算例 {i}：{违规[0][0]}，频道 {违规[0][1]}，"
                      f"共 {len(违规)} 次")
        清除率.append(r["清除成功"] / 案.源数)
        判空 = r["状态"].判空集()
        判空错 += len(判空 & 案.频道集())
    return {"局数": N, "出错局数": 坏局, "违规次数": 总违规,
            "平均清除率": float(np.mean(清除率)), "判空误判": 判空错}


# ================================================================ 四、超集性与保守度
def 检验超集(N=6, 采样=6000, 每隔=17):
    """**过程中**随机取快照，对该频道撒点，比对「按定义可行」与「模型可行」。

    必须在过程中取，不能等局末：局末时有源频道都已清除、空频道的可行集本来就空，
    比对是平凡的。这里每隔若干次检测取一次快照，覆盖从「一条示向线的细长条」
    到「几十米的小多边形」的各个阶段。

    要的是**单向**结论：按定义可行的点必须都在模型集合里（模型是超集）。
    反向的分歧是保守化留下的余量，允许存在，但要量化。
    """
    rng = np.random.default_rng(SEED + 200)
    行 = []
    for i in range(N):
        案 = G.造案例(rng)
        观测表 = {c: [] for c in range(1, C.频道数 + 1)}
        失败点 = {c: [] for c in range(1, C.频道数 + 1)}
        计 = {"n": 0}
        原记录检测 = ST.状态.记录检测
        原记录清除 = ST.状态.记录清除

        def 快照(self, c):
            已探测 = self.信念[c].收到信号
            m = self.信念[c].指标() if 已探测 else None
            if 已探测 and (m is None or m.get("空")):
                return
            if 已探测:
                V = np.asarray(m["多边形"], dtype=float)
                跨 = (V.max(axis=0) - V.min(axis=0)).max()
                垫 = max(2.0, 0.15 * 跨)
                lo, hi = V.min(axis=0) - 垫, V.max(axis=0) + 垫
                点 = rng.uniform(lo, hi, size=(采样, 2))
            else:
                t = rng.uniform(0, 2 * np.pi, 采样)
                rr = C.R_AREA * np.sqrt(rng.uniform(0, 1, 采样))
                点 = np.stack([rr * np.cos(t), rr * np.sin(t)], axis=1)
            定义内 = np.array([按定义可行(p, 观测表[c], 失败点[c]) for p in 点])
            模型内 = np.array([在模型可行集内(self.信念[c], self.掩码[c],
                                             self.栅格, p, 已探测) for p in 点])
            行.append({"算例": i, "频道": c, "已探测": 已探测, "采样": 采样,
                       "区域直径": (m or {}).get("直径", np.nan),
                       "定义内": int(定义内.sum()), "模型内": int(模型内.sum()),
                       "定义内但模型外": int((定义内 & ~模型内).sum()),
                       "模型内但定义外": int((模型内 & ~定义内).sum())})

        def 记录检测(self, p, c, 回):
            原记录检测(self, p, c, 回)
            观测表[c].append(((float(p[0]), float(p[1])), 回["measure_result"],
                              回.get("svd_deg")))
            计["n"] += 1
            if 计["n"] % 每隔 == 0:
                快照(self, c)

        def 记录清除(self, p, c, 回):
            原记录清除(self, p, c, 回)
            if 回["clear_result"] != "success":
                失败点[c].append((float(p[0]), float(p[1])))
                快照(self, c)

        ST.状态.记录检测 = 记录检测
        ST.状态.记录清除 = 记录清除
        try:
            狗 = S.机器狗(S.本地后端(案), 现实时限=1e9)
            ST.推荐策略(狗)
        finally:
            ST.状态.记录检测 = 原记录检测
            ST.状态.记录清除 = 原记录清除
    return pd.DataFrame(行)


# ================================================================ 四点五、判空蕴含
def 检验判空蕴含(裕度倍数=1.0, 采样=400000, 种子=SEED + 400):
    """直接验判空那条蕴含式，不绕道整局仿真。

        掩码为空  ==>  目标圆域内不存在同时满足所有 |x - p| > 1000 的点

    做法：取一组能覆盖全圆域的无信号点，按被测代码的规则消掉栅格；掩码空之后，
    在圆域内密集撒点，数一数还有几个点满足全部「距离 > 1000」。数学上必须是 0。

    为什么要单独做这一层：栅格的保守裕度只在「真源恰好落在排除圆边缘 14 m 带内、
    且该源的 R_eff 恰好贴着 1000」时才起作用，这两件事同时发生的概率极低，
    靠整局仿真的变异测试**打不中**（第一版就漏检了）。直接验蕴含式则一击即中。
    """
    栅 = C.栅格(20.0)
    栅.裕度 = 栅.裕度 * 裕度倍数
    掩码 = 栅.新掩码()
    候选 = ST.取规划器(C.栅格(20.0), 150.0).候选
    点集 = []
    # 贪心挑无信号点，直接对**完整栅格**算覆盖（不能用规划用的子采样，
    # 否则子格清空之后就选不出有用的点了——第一版正是这么写的，掩码清不空）
    for _ in range(40):
        if not np.any(掩码):
            break
        剩 = 栅.点[掩码]
        d = np.sqrt(((剩[None, :, :] - 候选[:, None, :]) ** 2).sum(axis=2))
        计数 = (d <= C.R_RECV_MIN - 栅.裕度).sum(axis=1)
        i = int(np.argmax(计数))
        if 计数[i] == 0:
            break
        点集.append(tuple(候选[i]))
        掩码 = 栅.排除圆盘(掩码, 点集[-1], C.R_RECV_MIN)
    if np.any(掩码):
        return {"裕度倍数": 裕度倍数, "无信号点数": len(点集), "掩码空": False,
                "反例格数": -1, "最大最近距离": np.nan}
    # 掩码空了。用 2 m 细网格逐点算「到最近无信号点的距离」，
    # 只要有点 > 1000 m，就说明真实可行集其实非空 —— 判空是错的。
    步 = 2.0
    k = int(np.ceil(C.R_AREA / 步))
    轴 = np.arange(-k, k + 1) * 步
    X, Y = np.meshgrid(轴, 轴)
    内 = np.hypot(X, Y) <= C.R_AREA
    x = np.stack([X[内], Y[内]], axis=1)
    最近 = np.full(len(x), np.inf)
    for p in 点集:
        最近 = np.minimum(最近, np.hypot(x[:, 0] - p[0], x[:, 1] - p[1]))
    坏 = 最近 > C.R_RECV_MIN
    return {"裕度倍数": 裕度倍数, "无信号点数": len(点集), "掩码空": True,
            "反例格数": int(坏.sum()), "最大最近距离": float(最近.max())}


# ================================================================ 五、策略不接触真值
class 封闭后端:
    """只转发四个方法，别的属性一律拒绝。策略若偷看真值会立刻抛异常。"""

    def __init__(self, 内层):
        object.__setattr__(self, "_内层", 内层)

    def enter(self):
        return object.__getattribute__(self, "_内层").enter()

    def measure(self, x, y, c):
        return object.__getattribute__(self, "_内层").measure(x, y, c)

    def clear(self, x, y, c):
        return object.__getattribute__(self, "_内层").clear(x, y, c)

    def exit(self):
        return object.__getattribute__(self, "_内层").exit()

    def __getattr__(self, 名):
        raise AttributeError(f"策略试图访问后端的 `{名}`——真值不允许被读取")


# ================================================================ 六、变异测试
def 变异测试(N=25):
    """往关键约束里植入错误，检查检验层是否报警。

    「复核方法值不值钱，不看它在正确代码上是否通过，看它在错误代码上是否失败。」
    """
    记录 = []
    基准 = 检验含真源(N=N, 静默=True)
    记录.append({"变异": "（基准，无变异）", "描述": "正确实现",
                 "出错局数": 基准["出错局数"], "违规次数": 基准["违规次数"],
                 "判空误判": 基准["判空误判"], "清除率": 基准["平均清除率"],
                 "是否报警": "否（应当如此）"})

    def 跑变异(名, 描述, 装, 卸, 参数=None, 局数=N):
        装()
        try:
            r = 检验含真源(N=局数, 参数=参数, 静默=True)
        except Exception as e:
            r = {"出错局数": -1, "违规次数": -1, "判空误判": -1,
                 "平均清除率": float("nan")}
            描述 += f"（抛异常：{type(e).__name__}）"
        finally:
            卸()
        报 = "是" if (r["出错局数"] > 0 or r["判空误判"] > 0 or
                      r["出错局数"] == -1) else "**漏检**"
        记录.append({"变异": 名, "描述": 描述, "出错局数": r["出错局数"],
                     "违规次数": r["违规次数"], "判空误判": r["判空误判"],
                     "清除率": r["平均清除率"], "是否报警": 报})

    # ① 楔形半角丢掉两位小数舍入裕度（P1 §5.4 的那 0.005°）
    跑变异("楔形半角 1.005° -> 1.000°",
           "忽略 svd_deg 两位小数舍入，真值会被楔形排除在外",
           lambda: None, lambda: None, 参数={"delta": 1.000}, 局数=max(N, 40))
    # ② 楔形半角取小
    跑变异("楔形半角 -> 0.90°", "楔形比误差界还窄，系统性排除真源",
           lambda: None, lambda: None, 参数={"delta": 0.90})
    # ③ 无信号排除半径取大。**只改策略侧**：若连同 按定义可行 里的常数一起改，
    #    裁判与被测对象一起错，检验就失效了（第一版正是这么写的，结果漏检）。
    原排除 = C.栅格.排除圆盘

    def 装3():
        C.栅格.排除圆盘 = (lambda self, 掩码, 圆心, 半径:
                           原排除(self, 掩码, 圆心, 半径 * 1.2))

    def 卸3():
        C.栅格.排除圆盘 = 原排除
    跑变异("判空排除半径 x1.2（只改策略侧）",
           "把 R_eff 下界当成 1200，会把 1000~1200 m 处的真源从判空栅格里划掉",
           装3, 卸3)
    # ⑤ 序约束方向反了
    原序 = C.序约束半平面

    def 装5():
        C.序约束半平面 = lambda 近, 远: 原序(远, 近)

    def 卸5():
        C.序约束半平面 = 原序
    跑变异("序约束方向反转", "把「收到信号的点更近」写反", 装5, 卸5)
    # ⑥ 圆外保守半平面改成不保守（直接用半径当阈值）
    原圆外 = C.圆外保守半平面

    def 装6():
        def 新(多边形, 圆心, 半径):
            h, 情形 = 原圆外(多边形, 圆心, 半径)
            if 情形 != "裁剪":
                return h, 情形
            P = np.asarray(多边形, dtype=float)
            c = np.asarray(圆心, dtype=float)
            u = (P.mean(axis=0) - c)
            u = u / max(float(np.hypot(*u)), 1e-12)
            return (-u[0], -u[1], -(半径 + float(u @ c))), "裁剪"
        C.圆外保守半平面 = 新

    def 卸6():
        C.圆外保守半平面 = 原圆外
    跑变异("排除盘线性化取消保守修正",
           "用切线直接当阈值，会把圆外的真源一起切掉", 装6, 卸6)
    # ⑦ 忘记「收到信号 => 距离 <= 1500」
    原信念 = C.频道信念.记录检测

    def 装7():
        def 新(self, p, 结果, svd=None):
            if 结果 == "direction":
                p = (float(p[0]), float(p[1]))
                self.检测次数 += 1
                self._缓存 = None
                self.收到信号 = True
                self.信号点.append((p, float(svd)))
                self.半平面 += C.几何.楔形半平面(p, float(svd), self.delta)
                # 故意漏掉 圆内半平面(p, 1500) 与序约束
                return
            原信念(self, p, 结果, svd)
        C.频道信念.记录检测 = 新

    def 卸7():
        C.频道信念.记录检测 = 原信念
    跑变异("漏掉「收信 => 距离<=1500」与序约束",
           "少加约束不会排除真源，只会让区域偏大——应当**不**报警", 装7, 卸7)
    return pd.DataFrame(记录)


# ================================================================ 主流程
def main():
    print("问题三 · 独立复核与变异测试")
    print(f"随机种子 {SEED}\n")
    检验计时()

    print("\n" + "=" * 88)
    print("二、可行集是否始终包含真实干扰源位置（承重的一条）")
    print("=" * 88)
    print("    每次检测、每次清除之后立刻检查：该频道的真源是否仍在维护的可行集内。")
    print("    这条不变量一旦破了，「确保所有干扰源被清除」就没有依据了。")
    r2 = 检验含真源(N=40)
    print(f"    局数 {r2['局数']}，出错局数 {r2['出错局数']}，"
          f"违规次数 {r2['违规次数']}")
    print(f"    平均清除率 {r2['平均清除率']:.4f}（必须是 1.0000）")

    print("\n" + "=" * 88)
    print("三、判空是否可靠")
    print("=" * 88)
    print(f"    策略宣布为空的频道里，实际有源的个数：{r2['判空误判']}")
    print("    判空的依据是「无信号排除盘之并覆盖目标圆域」，且栅格离散化偏保守，")
    print("    所以这一项在数学上必须是 0；实测为 0 才说明实现与推导一致。")

    print("\n" + "=" * 88)
    print("四、超集性与保守度")
    print("=" * 88)
    表 = 检验超集()
    表.to_csv(HERE / "复核_保守度.csv", index=False, encoding="utf-8-sig")
    print("    在**过程中**每隔若干次检测取一次快照（局末取样是平凡的：")
    print("    有源频道都已清除、空频道的可行集本来就空）。")
    print(f"    共 {len(表)} 个快照，每个撒 {int(表['采样'].iloc[0])} 点：")
    print(f"    定义内但模型外（**必须为 0**）：{int(表['定义内但模型外'].sum())}")
    print(f"    模型内但定义外（保守余量，允许）：{int(表['模型内但定义外'].sum())}")
    有 = 表[表["定义内"] > 0]
    if len(有):
        比 = (有["模型内"] / 有["定义内"].clip(lower=1)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        print(f"    模型可行集 / 定义可行集 的点数比：中位 {比.median():.3f}，"
              f"最大 {比.max():.3f}")
    print("    保守余量来自三处：楔形半角多给 0.005°、圆用外切多边形逼近、")
    print("    非凸排除盘的线性化取的是「少剔除」一侧。三处都只会让区域偏大。")

    print("\n" + "=" * 88)
    print("四点五、直接验判空的蕴含式：掩码空 ⟹ 可行集空")
    print("=" * 88)
    print("    整局仿真打不中这条——保守裕度只在极窄的一圈里起作用。直接验更干脆。")
    print(f"    {'栅格裕度':>10} {'无信号点数':>12} {'掩码是否空':>12} "
          f"{'2 m 细网格反例数':>18} {'最大最近距离(m)':>17}")
    蕴含 = []
    for 倍 in (1.0, 0.0):
        r = 检验判空蕴含(倍)
        蕴含.append(r)
        print(f"    {('标准 14.1 m' if 倍 == 1 else '置零'):>10} "
              f"{r['无信号点数']:>12} {str(r['掩码空']):>12} "
              f"{r['反例格数']:>18} {r['最大最近距离']:>17.2f}")
    print("    标准裕度下反例必须为 0（最大最近距离 <= 1000 m）；")
    print("    裕度置零后出现反例，说明这 14 米不是装饰，而是「掩码空 ⟹ 可行集空」")
    print("    这条蕴含式成立的前提。")
    pd.DataFrame(蕴含).to_csv(HERE / "复核_判空蕴含.csv", index=False,
                              encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("五、策略是否偷看了真值")
    print("=" * 88)
    rng = np.random.default_rng(SEED + 300)
    案 = G.造案例(rng)
    狗 = S.机器狗(封闭后端(S.本地后端(案)), 现实时限=1e9)
    try:
        r = ST.推荐策略(狗)
        print(f"    用只暴露 enter/measure/clear/exit 四个方法的代理后端跑完整局：通过")
        print(f"    清除 {r['清除成功']}/{案.源数}，虚拟总时间 {r['虚拟总时间']:.1f} s")
    except AttributeError as e:
        print(f"    **失败**：{e}")

    print("\n" + "=" * 88)
    print("六、变异测试：上面几层复核真的抓得住错误吗")
    print("=" * 88)
    变 = 变异测试(N=25)
    变.to_csv(HERE / "复核_变异测试.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(变[["变异", "出错局数", "违规次数", "判空误判", "清除率",
              "是否报警"]].round(4).to_string(index=False))
    print()
    for _, r in 变.iterrows():
        print(f"    · {r['变异']}：{r['描述']}")
    print("\n    第 ⑦ 项是**阴性对照**：漏加约束只会让可行集偏大，不会排除真源，")
    print("    所以它**不应该**触发含真源检验。若它也报警，说明检验层过于敏感、")
    print("    把保守和错误混为一谈了。")

    print("\n已写入 复核_保守度.csv、复核_变异测试.csv")


if __name__ == "__main__":
    with open(HERE / "独立复核结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '独立复核结果.txt'}")
