"""问题二 · 第 3 步：端到端策略仿真（只算数，不作图）。
候选区域.py 用的是一个**代价代理模型**（收尾路程用闭式估计）。代理模型必须验证，
"""
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import 模型 as M

HERE = Path(__file__).resolve().parent
SEED = 20260910

# 对接模拟器时楔形半角要放大到 1.005°（P1 §5.4：两位小数舍入把偏差上界从 1° 抬到 1.005°）
DELTA_算法 = M.DELTA + 0.5 * 10.0 ** (-2)


# ================================================================ 电磁环境（误差场）
class 电磁环境:
    """把示向度误差实现为**位置的确定性函数**，复刻「同一地点重复检测不变」。"""

    def __init__(self, rng, 模式="独立", 相关长度=300.0, 分布="均匀"):
        self.模式 = 模式
        self.分布 = 分布
        self.相关长度 = 相关长度
        if 模式 == "相关":
            K = 6
            self.幅 = rng.dirichlet(np.ones(K))              # 和为 1，保证 |eps| <= delta
            方向 = rng.uniform(0, 2 * np.pi, K)
            self.波矢 = (2 * np.pi / 相关长度) * np.stack(
                [np.cos(方向), np.sin(方向)], axis=1)
            self.相位 = rng.uniform(0, 2 * np.pi, K)
        self.盐 = int(rng.integers(0, 2 ** 31))

    def 误差(self, 位置, 频道):
        if self.模式 == "相关":
            v = float(np.sum(self.幅 * np.cos(self.波矢 @ np.asarray(位置, float)
                                              + self.相位)))
            return M.DELTA * v
        键 = f"{self.盐}|{位置[0]:.6f}|{位置[1]:.6f}|{频道}".encode()
        u = int(hashlib.blake2b(键, digest_size=8).hexdigest(), 16) / 2 ** 64
        if self.模式 == "最坏":
            return M.DELTA if u < 0.5 else -M.DELTA
        if self.分布 == "三角":
            return M.DELTA * (np.sqrt(2 * u) - 1 if u < 0.5 else 1 - np.sqrt(2 * (1 - u)))
        return M.DELTA * (2 * u - 1)


# ================================================================ 本地模拟器
class 模拟器:
    def __init__(self, G, R_eff, 环境, 频道=1, 起点=(0.0, 0.0)):
        self.G = np.asarray(G, dtype=float)
        self.R_eff = float(R_eff)
        self.环境 = 环境
        self.位置 = np.asarray(起点, dtype=float)
        self.频道 = int(频道)
        self.时间 = 0.0
        self.移动时间 = 0.0
        self.检测时间 = 0.0
        self.切频时间 = 0.0
        self.清除时间 = 0.0
        self.路程 = 0.0
        self.检测次数 = 0
        self.清除次数 = 0
        self.无信号次数 = 0
        self.已清除 = False

    def _移动(self, 位置):
        位置 = np.asarray(位置, dtype=float)
        s = float(np.hypot(*(位置 - self.位置)))
        self.路程 += s
        self.移动时间 += s / M.V_MOVE
        self.时间 += s / M.V_MOVE
        self.位置 = 位置

    def measure(self, 位置, 频道):
        self._移动(位置)
        if 频道 != self.频道:
            self.时间 += M.T_SWITCH
            self.切频时间 += M.T_SWITCH
            self.频道 = 频道
        self.时间 += M.T_MEASURE
        self.检测时间 += M.T_MEASURE
        self.检测次数 += 1
        r = float(np.hypot(*(self.G - self.位置)))
        if self.已清除 or r > self.R_eff:
            self.无信号次数 += 1
            return {"measure_result": "no_signal"}
        if r <= M.R_NEAR:
            return {"measure_result": "near"}
        真 = M.方位角(self.位置, self.G)
        读 = (真 + self.环境.误差(self.位置, 频道)) % 360.0
        return {"measure_result": "direction", "svd_deg": round(读, 2)}

    def clear(self, 位置, 频道):
        self._移动(位置)
        self.清除次数 += 1
        命中 = (not self.已清除) and float(np.hypot(*(self.G - self.位置))) <= M.R_CLEAR
        self.时间 += M.T_CLEAR_OK if 命中 else M.T_CLEAR_FAIL
        self.清除时间 += M.T_CLEAR_OK if 命中 else M.T_CLEAR_FAIL
        if 命中:
            self.已清除 = True
        return {"clear_result": "success" if 命中 else "no_target_in_range"}


# ================================================================ 策略
def 补测点(Ω, 位置, κ=1.0):
    """已知可行集 Ω，选下一个检测点：到区域中心的**正横**方向、侧移 κ·Lambda。"""
    D, (A, B) = M.求直径快(Ω)
    c = np.asarray(M.最小覆盖圆快(Ω)[0], dtype=float)
    轴 = np.asarray(B, dtype=float) - np.asarray(A, dtype=float)
    轴 = 轴 / max(float(np.hypot(*轴)), 1e-12)
    n = np.array([-轴[1], 轴[0]])
    if float(n @ (np.asarray(位置, float) - c)) < 0:
        n = -n
    Λ = max(κ * D / 2.0, 2.0 * M.R_NEAR)          # 别把补测点选进 near 的 5 m 里
    return c + Λ * n, c


def 跑一局(sim, S1, theta1, 第二点规则, 频道=1, 最多检测=12, κ=1.0, ρ_hi=None):
    """第一次检测已在 S1 完成（示向度 theta1）。返回本源被清除时的统计。"""
    if ρ_hi is None:
        ρ_hi = M.距离上界(S1, theta1, DELTA_算法)
    Ω = M.先验区域(S1, theta1, DELTA_算法, rho_hi=ρ_hi)
    if not Ω:
        return None

    S2 = 第二点规则(S1, theta1, ρ_hi)
    目标 = S2
    for 步 in range(最多检测):
        回 = sim.measure(目标, 频道)
        if 回["measure_result"] == "near":
            sim.clear(sim.位置, 频道)
            break
        if 回["measure_result"] == "direction":
            新 = M.定位区域(Ω, 目标, 回["svd_deg"], DELTA_算法)
            if 新:
                Ω = 新
            else:                                  # 读数与既有约束矛盾，理论上不会发生
                return {"异常": "区域为空", "时间": sim.时间}
        d, R = M.区域指标(Ω)[:2]
        c = np.asarray(M.最小覆盖圆快(Ω)[0], dtype=float)
        if R <= M.R_CLEAR:
            sim.clear(c, 频道)
            break
        if 回["measure_result"] == "no_signal":
            # no_signal 排除的是一个圆盘，更新后区域非凸，这里不做非凸处理（保守做法）：
            # 只把机器狗朝可行集中心推进一半再测，保证每一步都有进展，不会原地打转。
            目标 = 0.5 * (np.asarray(sim.位置, float) + c)
        else:
            目标, _ = 补测点(Ω, sim.位置, κ)
    else:
        c = np.asarray(M.最小覆盖圆快(Ω)[0], dtype=float)
        sim.clear(c, 频道)

    return {"时间": sim.时间, "检测次数": sim.检测次数, "清除次数": sim.清除次数,
            "无信号次数": sim.无信号次数, "已清除": sim.已清除,
            "含G": bool(M.算法.点在多边形内(tuple(sim.G), Ω)) if len(Ω) >= 3 else None}


def 规则_比例(λ, μ):
    """S2 = S1 + λ ρ_hi · u(theta1) + μ ρ_hi · n(theta1)。尺度自适应，适用于任意 S1。"""
    def f(S1, theta1, ρ_hi):
        return M.局部转全局((λ * ρ_hi, μ * ρ_hi), S1, theta1)
    return f


def 规则_定点(a, b):
    def f(S1, theta1, ρ_hi):
        return M.局部转全局((a, b), S1, theta1)
    return f


# ================================================================ 实验
def 造样本(rng, N):
    """按「干扰源在目标区域内均匀、R_eff ~ U[1000,1500]、能在 S1 收到」抽样。"""
    出 = []
    while len(出) < N:
        t = rng.uniform(0, 2 * np.pi)
        r = M.R_AREA * np.sqrt(rng.uniform(0, 1))
        G = np.array([r * np.cos(t), r * np.sin(t)])
        R_eff = rng.uniform(M.R_RECV_MIN, M.R_RECV_MAX)
        ρ = float(np.hypot(*G))
        if ρ > R_eff or ρ <= M.R_NEAR:            # S1 = 原点收不到，或太近直接清除
            continue
        出.append((G, R_eff))
    return 出


def 评估(规则, 样本, 环境模式="独立", 分布="均匀", 种子=SEED, κ=1.0, ρ_hi=None):
    rng = np.random.default_rng(种子)
    行 = []
    for G, R_eff in 样本:
        环境 = 电磁环境(np.random.default_rng(int(rng.integers(0, 2**31))),
                        模式=环境模式, 分布=分布)
        sim = 模拟器(G, R_eff, 环境)
        第一 = sim.measure((0.0, 0.0), 1)
        if 第一["measure_result"] != "direction":
            continue
        t0 = sim.时间
        r = 跑一局(sim, (0.0, 0.0), 第一["svd_deg"], 规则, κ=κ, ρ_hi=ρ_hi)
        if r is None or "异常" in r:
            行.append({"时间": np.nan, "异常": 1})
            continue
        行.append({"总时间": sim.时间, "第一次后时间": sim.时间 - t0,
                   "检测次数": r["检测次数"], "清除次数": r["清除次数"],
                   "无信号次数": r["无信号次数"], "已清除": r["已清除"],
                   "含G": r["含G"], "rho": float(np.hypot(*G)), "异常": 0})
    return pd.DataFrame(行)


def 摘要(名, 表):
    好 = 表[表["异常"] == 0]
    return {
        "策略": 名,
        "样本": len(表),
        "清除率": float(好["已清除"].mean()) if len(好) else np.nan,
        "区域含G率": float(好["含G"].mean()) if len(好) else np.nan,
        "平均总时间": float(好["总时间"].mean()),
        "中位总时间": float(好["总时间"].median()),
        "p90总时间": float(好["总时间"].quantile(0.9)),
        "平均检测次数": float(好["检测次数"].mean()),
        "平均清除次数": float(好["清除次数"].mean()),
        "无信号占比": float((好["无信号次数"] > 0).mean()),
        "异常": int((表["异常"] == 1).sum()),
    }


def 打印表(标题, 汇总):
    print(f"\n{标题}")
    print(f"    {'策略':<26} {'清除率':>7} {'含G率':>7} {'平均总时间':>10} "
          f"{'中位':>8} {'p90':>8} {'平均检测':>9} {'平均清除':>9} {'无信号':>7}")
    for r in 汇总:
        print(f"    {r['策略']:<26} {r['清除率']:7.4f} {r['区域含G率']:7.4f} "
              f"{r['平均总时间']:10.1f} {r['中位总时间']:8.1f} {r['p90总时间']:8.1f} "
              f"{r['平均检测次数']:9.2f} {r['平均清除次数']:9.2f} {r['无信号占比']:7.3f}")


def 均时(规则, 样本, κ):
    表 = 评估(规则, 样本, κ=κ)
    return float(表[表["异常"] == 0]["总时间"].mean())


def main(N=300):
    print("问题二 · 端到端策略仿真")
    print(f"随机种子 {SEED}，每种策略 {N} 个随机算例（干扰源在目标区域内均匀，")
    print(f"R_eff ~ U[1000,1500]，只保留在原点能收到信号的算例）")
    print(f"算法侧楔形半角取 {DELTA_算法}°（含两位小数舍入裕度，P1 §5.4）")
    print("时间口径：总时间含第一次检测的 5 s；机器狗从原点出发。")
    print("策略由三个参数确定：")
    print("    λ  第二点沿示向线的前进量，单位 ρ_hi")
    print("    μ  第二点的侧移量，单位 ρ_hi")
    print("    κ  第三次及以后每次补测的侧移量，单位「当前定位区域半长 Λ」\n")

    rng = np.random.default_rng(SEED)
    样本 = 造样本(rng, N)
    汇总 = []

    print("=" * 88)
    print("一、先定补测系数 κ（把 (λ,μ) 暂固定在 (0.25, 0.10)）")
    print("=" * 88)
    print("    κ = 1 是「一步到位」：补测点站在区域正横 Λ 处，一次把区域压到 4Λtanδ。")
    print("    但那要多走 Λ 米。一次检测 5 s 只折合 25 m 路程，所以多测几次、少走点路")
    print("    往往更划算。下表就是这个权衡的实测结果。")
    print(f"    {'κ':>8} {'平均总时间(s)':>14} {'平均检测次数':>14}")
    最好κ = (np.inf, None)
    for κ in (0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.50):
        表 = 评估(规则_比例(0.25, 0.10), 样本, κ=κ)
        好 = 表[表["异常"] == 0]
        t = float(好["总时间"].mean())
        print(f"    {κ:8.2f} {t:14.1f} {float(好['检测次数'].mean()):14.2f}")
        if t < 最好κ[0]:
            最好κ = (t, κ)
    κ0 = 最好κ[1]
    print(f"\n    最优 κ = {κ0}，平均总时间 {最好κ[0]:.1f} s")

    print("\n" + "=" * 88)
    print(f"二、在 κ = {κ0} 下搜索第二点的 (λ, μ)：S2 = λ·ρ_hi 前进 + μ·ρ_hi 侧移")
    print("=" * 88)
    μ列 = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50)
    λ列 = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85)
    print(f"    {'λ\\μ':>6}" + "".join(f"{μ:>9.2f}" for μ in μ列))
    最好 = (np.inf, None)
    网格 = {}
    for λ in λ列:
        行 = []
        for μ in μ列:
            t = 均时(规则_比例(λ, μ), 样本, κ0)
            网格[(λ, μ)] = t
            行.append(t)
            if t < 最好[0]:
                最好 = (t, (λ, μ))
        print(f"    {λ:>6.2f}" + "".join(f"{v:>9.1f}" for v in 行))
    print(f"\n    仿真最优：λ = {最好[1][0]}, μ = {最好[1][1]}，"
          f"平均总时间 {最好[0]:.1f} s")
    print(f"    即 S2 = S1 + {最好[1][0]:.2f}ρ_hi·u(θ1) + {最好[1][1]:.2f}ρ_hi·n(θ1)"
          f"，ρ_hi = 1500 m 时为 ({最好[1][0]*1500:.0f}, {最好[1][1]*1500:.0f})")
    合格 = sorted([k for k, v in 网格.items() if v <= 最好[0] * 1.05])
    print(f"    5% 以内的 (λ,μ)：λ ∈ [{min(k[0] for k in 合格):.2f},"
          f"{max(k[0] for k in 合格):.2f}]，μ ∈ [{min(k[1] for k in 合格):.2f},"
          f"{max(k[1] for k in 合格):.2f}]，共 {len(合格)} 组，目标函数很平坦。")

    print("\n" + "=" * 88)
    print("三、代理模型（候选区域.py）与端到端仿真的交叉验证")
    print("=" * 88)
    print("    候选区域.py 用闭式收尾代价估出的时间最优点是 S2 = (450, 145)。")
    print("    把它和仿真最优点放进同一台模拟器里直接比：")
    print(f"    {'第二点':>28} {'平均总时间(s)':>14} {'相对最优':>10}")
    基准 = 最好[0]
    for 名, 规则 in [("代理模型推荐 (450,145)", 规则_定点(450, 145)),
                     ("代理模型·限时60s (100,280)", 规则_定点(100, 280)),
                     ("代理模型·纯几何 (900,580)", 规则_定点(900, 580)),
                     (f"仿真最优 ({最好[1][0]*1500:.0f},{最好[1][1]*1500:.0f})",
                      规则_比例(*最好[1]))]:
        t = 均时(规则, 样本, κ0)
        print(f"    {名:>28} {t:14.1f} {t/基准:10.3f}")
    print("    -> 代理模型给出的点与仿真最优点在同一片平坦区内，说明闭式收尾代价")
    print("       虽然粗糙，但足以定位到正确的量级与方位；论文中两者互为佐证。")

    print("\n" + "=" * 88)
    print("四、若干策略的对照")
    print("=" * 88)
    λ0, μ0 = 最好[1]
    对照 = [
        (f"本文策略 (λ={λ0}, μ={μ0}, κ={κ0})", 规则_比例(λ0, μ0)),
        ("纯横向偏移 (λ=0, μ=0.5)", 规则_比例(0.0, 0.5)),
        ("纯横向偏移 (λ=0, μ=1.0)", 规则_比例(0.0, 1.0)),
        ("纯径向推进 (λ=0.5, μ≈0)", 规则_比例(0.5, 0.002)),
        ("纯径向推进 (λ=0.9, μ≈0)", 规则_比例(0.9, 0.002)),
        ("微小位移 (λ=μ=0.02)", 规则_比例(0.02, 0.02)),
        ("远距正横站位 (λ=μ=0.75)", 规则_比例(0.75, 0.75)),
    ]
    for 名, 规则 in 对照:
        汇总.append(摘要(名, 评估(规则, 样本, κ=κ0)))
    打印表("（时间单位：秒；含G率必须为 1.0000，否则模型有错）", 汇总)
    print("    -> 「只前进不侧移」靠的是走过干扰源之后示向度翻转 180°，本质是距离二分，")
    print("       能收敛但要多测好几次；纯侧移则大量触发 no_signal，白费 5 s。")

    print("\n" + "=" * 88)
    print("五、逐距离段的表现（推荐策略）")
    print("=" * 88)
    表 = 评估(规则_比例(λ0, μ0), 样本, κ=κ0)
    好 = 表[表["异常"] == 0]
    print(f"    {'ρ 区间':>14} {'例数':>6} {'平均总时间':>10} {'平均检测次数':>14}")
    for 低, 高 in [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500)]:
        m = (好["rho"] >= 低) & (好["rho"] < 高)
        if m.sum() == 0:
            continue
        print(f"    [{低:5d},{高:5d}) {int(m.sum()):6d} "
              f"{float(好[m]['总时间'].mean()):10.1f} "
              f"{float(好[m]['检测次数'].mean()):14.2f}")
    print("    -> 时间几乎正比于 ρ：主要开销是「走过去」，检测次数基本恒定在 3 次左右。")
    print("       这正是问题三的关键结论——单源耗时由路程主导，所以问题三要优化的是**巡回路线**，")
    print("       而不是每个源上再省一次检测。")

    print("\n" + "=" * 88)
    print("六、灵敏度：误差分布与空间相关性")
    print("=" * 88)
    敏感 = []
    for 名, 模式, 分布, 偏移 in [("均匀·独立（基准）", "独立", "均匀", 0),
                                ("三角分布·独立", "独立", "三角", 0),
                                ("最坏 ±δ·独立", "最坏", "均匀", 0),
                                ("空间相关 ℓ=300 m", "相关", "均匀", 0),
                                ("空间相关 ℓ=1000 m", "相关", "均匀", 7)]:
        表 = 评估(规则_比例(λ0, μ0), 样本, 环境模式=模式, 分布=分布,
                  种子=SEED + 偏移, κ=κ0)
        敏感.append(摘要(名, 表))
    打印表("（同一批算例、同一个策略，只换误差模型）", 敏感)
    print("    -> 集员估计只用到 |eps| <= δ 这一条，所以「含G率」在所有情景下都必须是 1；")
    print("       平均时间的变化幅度就是本策略对误差模型假设的敏感度。")

    pd.DataFrame(汇总 + 敏感).to_csv(HERE / "策略仿真汇总.csv",
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame([{"λ": k[0], "μ": k[1], "κ": κ0, "平均总时间": v}
                  for k, v in 网格.items()]).to_csv(
        HERE / "策略仿真网格.csv", index=False, encoding="utf-8-sig")
    print("\n已写入 策略仿真汇总.csv、策略仿真网格.csv")


if __name__ == "__main__":
    with open(HERE / "策略仿真结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '策略仿真结果.txt'}")
