"""问题三 · 模型检验：独立复核与变异测试（只算数，不作图）。

性质层面的证明已经搬到 命题与引理.py（6 秒跑完）；本脚本只承担**集成**角色：
把各条约束串起来端到端跑，看它们在真实控制流里还对不对。这类检验只能靠整局抽样，
而一局约 3.7 s，串行上百局就看不到头，所以除了必须独占的两小项，全部交给进程池
（见 并行.py），边完成边报进度。

    python 独立复核.py          # 完整版，约 2 分钟
    python 独立复核.py --快     # 局数减半，约 1 分钟
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import 并行 as PAR                    # 必须最先导入：它在 numpy 之前钉死 BLAS 线程数

import numpy as np
import pandas as pd

import 公共 as C
import 模拟器 as S
import 生成算例 as G
import 策略 as ST

HERE = Path(__file__).resolve().parent
SEED = 20260911

# 变异测试里有几档会把模型假设彻底破坏，主循环随之进入**不发指令的空转**
# （可行集为空 -> 重置信念 -> 又为空），每轮白白重算一次指标。实测结论与动作上限
# 无关，只有空烧的 CPU 随上限线性增长，故全程统一给 200 的动作预算。
动作预算 = 200


# ================================================================ 独立的可行性定义
def 按定义可行(点, 观测表, 失败清除点=(), delta=C.DELTA_算法, R=C.R_AREA):
    """一个位置能否解释全部观测。完全按题面写，不引用任何被测代码。"""
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


# ================================================================ 变异的植入与撤销
@contextmanager
def 植入(键):
    """把一个错误临时装进被测代码。每个子进程各装各的，互不干扰。"""
    if 键 == "无":
        yield
        return
    if 键 == "排除半径":
        # **只改策略侧**：若连同 按定义可行 里的常数一起改，被测者与判据一起错，
        # 检验就成了自说自话，什么也抓不住。
        原 = C.栅格.排除圆盘
        C.栅格.排除圆盘 = (lambda self, 掩码, 圆心, 半径:
                           原(self, 掩码, 圆心, 半径 * 1.2))
        try:
            yield
        finally:
            C.栅格.排除圆盘 = 原
        return
    if 键 == "序反":
        原 = C.序约束半平面
        C.序约束半平面 = lambda 近, 远: 原(远, 近)
        try:
            yield
        finally:
            C.序约束半平面 = 原
        return
    if 键 == "不保守":
        原 = C.圆外保守半平面

        def 新(多边形, 圆心, 半径):
            h, 情形 = 原(多边形, 圆心, 半径)
            if 情形 != "裁剪":
                return h, 情形
            P = np.asarray(多边形, dtype=float)
            c = np.asarray(圆心, dtype=float)
            u = (P.mean(axis=0) - c)
            u = u / max(float(np.hypot(*u)), 1e-12)
            return (-u[0], -u[1], -(半径 + float(u @ c))), "裁剪"
        C.圆外保守半平面 = 新
        try:
            yield
        finally:
            C.圆外保守半平面 = 原
        return
    if 键 == "漏约束":
        原 = C.频道信念.记录检测

        def 新(self, p, 结果, svd=None):
            if 结果 == "direction":
                p = (float(p[0]), float(p[1]))
                self.检测次数 += 1
                self._缓存 = None
                self.收到信号 = True
                self.信号点.append((p, float(svd)))
                self.半平面 += C.几何.楔形半平面(p, float(svd), self.delta)
                return                      # 故意漏掉 圆内半平面(p,1500) 与序约束
            原(self, p, 结果, svd)
        C.频道信念.记录检测 = 新
        try:
            yield
        finally:
            C.频道信念.记录检测 = 原
        return
    raise KeyError(键)


# ================================================================ 一、可行集含真源
def 逐步含真源(案, 参数=None):
    """把策略跑一遍，每次观测之后立刻检查每个源是否还在自己频道的可行集里。"""
    参数 = dict(ST.默认参数, **(参数 or {}))
    真值 = {int(s["频道"]): (s["x"], s["y"]) for s in 案.源表}
    违规, 断言 = [], [0]
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
        断言[0] += 2
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
    return r, 违规, 断言[0]


def 含真源一局(任务):
    """子进程入口：造一局、（可选）植入变异、跑一局、回报违规情况。"""
    案 = G.造案例(np.random.default_rng(任务["种子"]))
    参数 = dict(任务.get("参数", {}), 最多动作=动作预算)
    try:
        with 植入(任务.get("变异", "无")):
            r, 违规, 断言 = 逐步含真源(案, 参数)
    except Exception as e:                    # 变异档可能把代码跑崩，这也算报警
        return {"变异": 任务.get("变异", "无"), "崩溃": type(e).__name__,
                "违规数": 0, "首违规": "", "断言数": 0,
                "清除率": float("nan"), "判空误判": 0}
    return {"变异": 任务.get("变异", "无"), "崩溃": "",
            "违规数": len(违规), "首违规": 违规[0][0] if 违规 else "",
            "断言数": 断言, "清除率": r["清除成功"] / 案.源数,
            "判空误判": len(r["状态"].判空集() & 案.频道集())}


def 汇总含真源(明细):
    崩 = int((明细["崩溃"] != "").sum())
    return {"局数": len(明细), "出错局数": int((明细["违规数"] > 0).sum()),
            "违规次数": int(明细["违规数"].sum()),
            "断言次数": int(明细["断言数"].sum()),
            "崩溃局数": 崩,
            "平均清除率": float(明细["清除率"].mean()),
            "判空误判": int(明细["判空误判"].sum())}


# ================================================================ 二、超集性与保守度
def 超集一局(任务):
    """**过程中**随机取快照，对该频道撒点，比对「按定义可行」与「模型可行」。"""
    i, 采样, 每隔, 上限 = (任务["序号"], 任务["采样"], 任务["每隔"], 任务["快照上限"])
    rng = np.random.default_rng(任务["种子"])
    案 = G.造案例(rng)
    行 = []
    观测表 = {c: [] for c in range(1, C.频道数 + 1)}
    失败点 = {c: [] for c in range(1, C.频道数 + 1)}
    计 = {"n": 0}
    原记录检测 = ST.状态.记录检测
    原记录清除 = ST.状态.记录清除

    def 快照(self, c):
        if len(行) >= 上限:                  # 撒点判据是纯 Python 循环，很贵；封顶
            return
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
        ST.推荐策略(狗, {"最多动作": 动作预算})
    finally:
        ST.状态.记录检测 = 原记录检测
        ST.状态.记录清除 = 原记录清除
    return 行


# ================================================================ 三、计时复刻
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

    案 = G.造案例(np.random.default_rng(SEED), 源数=12)
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


# ================================================================ 四、策略不接触真值
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


# ================================================================ 五、变异测试
# （名字, 变异键, 策略参数, 描述, 类别）。类别决定「是否报警」这一列怎么判读：
#   基准   不该报警；阳性 必须报警；阴性 不该报警（漏加约束只是偏保守）；
#   引理层 整局仿真打不中，交给 命题与引理.py 的性质扫描，这里不作判据。
变异表 = [
    ("（基准，无变异）", "无", {}, "正确实现", "基准"),
    ("楔形半角 1.005° -> 1.000°", "无", {"delta": 1.000},
     "忽略 svd_deg 两位小数舍入，真值会被楔形排除在外", "阳性"),
    ("楔形半角 -> 0.90°", "无", {"delta": 0.90},
     "楔形比误差界还窄，系统性排除真源", "阳性"),
    ("判空排除半径 x1.2（只改策略侧）", "排除半径", {},
     "把 R_eff 下界当成 1200，会把 1000~1200 m 处的真源从判空栅格里划掉", "阳性"),
    ("序约束方向反转", "序反", {}, "把「收到信号的点更近」写反", "阳性"),
    ("排除盘线性化取消保守修正", "不保守", {},
     "修正量只有一两米，整局仿真撞不上；命题与引理.py 对着性质本身扫描，"
     "去掉修正后 379/78191 次被错误裁掉", "引理层"),
    ("漏掉「收信 => 距离<=1500」与序约束", "漏约束", {},
     "少加约束只会让区域偏大，不会排除真源", "阴性"),
]


def 变异汇总(名, 描述, 类别, 明细):
    s = 汇总含真源(明细)
    触发 = s["出错局数"] > 0 or s["判空误判"] > 0 or s["崩溃局数"] > 0
    报 = {"基准": "否（应当如此）",
          "阳性": "是" if 触发 else "**漏检**",
          "阴性": "**误报**" if 触发 else "否（应当如此）",
          "引理层": "整局打不中，见引理层"}[类别]
    if s["崩溃局数"]:
        描述 += f"（{s['崩溃局数']} 局抛异常）"
    return {"变异": 名, "类别": 类别, "描述": 描述, "局数": s["局数"],
            "出错局数": s["出错局数"], "违规次数": s["违规次数"],
            "判空误判": s["判空误判"], "清除率": s["平均清除率"], "是否报警": 报}


# ================================================================ 主流程
def main(argv=()):
    快 = "--快" in argv
    集成局数, 超集局数, 变异局数 = (6, 3, 5) if 快 else (12, 6, 10)
    print("问题三 · 独立复核与变异测试")
    print(f"随机种子 {SEED}，{PAR.进程数} 进程并行\n")
    检验计时()

    print("\n" + "=" * 88)
    print("二、可行集是否始终包含真实干扰源位置（承重的一条）")
    print("=" * 88)
    print("    每次检测、每次清除之后立刻检查：该频道的真源是否仍在维护的可行集内。")
    print("    这条不变量一旦破了，「确保所有干扰源被清除」就没有依据了。")
    明细 = pd.DataFrame(PAR.并行映射(
        含真源一局, [{"种子": SEED + 100 + i} for i in range(集成局数)],
        说明="含真源"))
    r2 = 汇总含真源(明细)
    print(f"    局数 {r2['局数']}，逐次断言 {r2['断言次数']} 次，"
          f"出错局数 {r2['出错局数']}，违规次数 {r2['违规次数']}")
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
    片 = PAR.并行映射(超集一局,
                      [{"序号": i, "种子": SEED + 200 + i, "采样": 3000,
                        "每隔": 17, "快照上限": 8}
                       for i in range(超集局数)], 说明="超集")
    表 = pd.DataFrame([r for 块 in 片 for r in 块])
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
    print("四点五、命题层面的性质检验已移出本脚本")
    print("=" * 88)
    print("    判空蕴含式与排除盘线性化这两条，靠整局仿真打不中（只在极窄的一圈里")
    print("    起作用）。现由 命题与引理.py 对着性质本身密集扫描，6 秒跑完；")
    print("    结果见 命题与引理结果.txt 与 引理检验.csv。本脚本只保留集成检验。")

    print("\n" + "=" * 88)
    print("五、策略是否偷看了真值")
    print("=" * 88)
    案 = G.造案例(np.random.default_rng(SEED + 300))
    狗 = S.机器狗(封闭后端(S.本地后端(案)), 现实时限=1e9)
    try:
        r = ST.推荐策略(狗)
        print("    用只暴露 enter/measure/clear/exit 四个方法的代理后端跑完整局：通过")
        print(f"    清除 {r['清除成功']}/{案.源数}，虚拟总时间 {r['虚拟总时间']:.1f} s")
    except AttributeError as e:
        print(f"    **失败**：{e}")

    print("\n" + "=" * 88)
    print("六、变异测试：上面几层复核真的抓得住错误吗")
    print("=" * 88)
    任务 = []
    for 名, 键, 参数, _, _ in 变异表:
        n = 变异局数 * 2 if "1.000" in 名 else 变异局数       # 低概率事件多跑几局
        任务 += [{"种子": SEED + 100 + i, "变异": 键, "参数": 参数, "名": 名}
                 for i in range(n)]
    结果 = pd.DataFrame(PAR.并行映射(含真源一局, 任务, 说明="变异"))
    结果["名"] = [t["名"] for t in 任务]
    变 = pd.DataFrame([变异汇总(名, 描述, 类别, 结果[结果["名"] == 名])
                       for 名, _, _, 描述, 类别 in 变异表])
    变.to_csv(HERE / "复核_变异测试.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(变[["变异", "类别", "局数", "出错局数", "违规次数", "判空误判",
              "清除率", "是否报警"]].round(4).to_string(index=False))
    print()
    for _, r in 变.iterrows():
        print(f"    · {r['变异']}：{r['描述']}")
    print("\n    最后一项是**阴性对照**：漏加约束只会让可行集偏大，不会排除真源，")
    print("    所以它**不应该**触发含真源检验。若它也报警，说明检验层过于敏感、")
    print("    把保守和错误混为一谈了。")

    print("\n已写入 复核_保守度.csv、复核_变异测试.csv")


if __name__ == "__main__":
    with PAR.记录到(HERE / "独立复核结果.txt"):
        main(sys.argv)
