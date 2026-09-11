"""问题二 · 第 3b 步：把「为什么这个策略最优」变成可核对的数字（只算数，不作图）。
策略仿真.py 只在**我们自己的策略族内**搜了最优，那只能说明「族内最好」。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 模型 as M
import 策略仿真 as SIM

HERE = Path(__file__).resolve().parent
SEED = SIM.SEED
DELTA_算法 = SIM.DELTA_算法

# 策略仿真.py 搜出的最优参数，本脚本一律以它为基准
LAM, MU, KAP = 0.30, 0.08, 0.25

# 候选区域（局部标架，单位米，ρ_hi = 1500 时）。
框 = dict(a低=150.0, a高=450.0, b低=75.0, b高=180.0)


def 分节(标题):
    print()
    print("=" * 86)
    print(标题)
    print("=" * 86)


# ================================================================ 一、上帝视角
def 节1_上帝视角(样本):
    分节("一、上帝视角下界：任何策略都不可能比它更快")
    print("假设机器狗一开始就知道干扰源的精确位置：直奔过去，清除，结束。")
    print("耗时 = 距离/5 + 5（清除）。这是**绝对地板**，与用什么算法无关。\n")

    d = np.array([float(np.hypot(*G)) for G, _ in 样本])
    地板 = d / M.V_MOVE + M.T_CLEAR_OK
    含首测 = 地板 + M.T_MEASURE
    print(f"    样本量 {len(样本)}，干扰源到原点的平均距离 {d.mean():.1f} m"
          f"（中位 {np.median(d):.1f}，p90 {np.percentile(d,90):.1f}）")
    print(f"    地板（不含任何检测）        平均 {地板.mean():7.1f} s")
    print(f"    地板 + 题设的第一次检测 5 s  平均 {含首测.mean():7.1f} s")
    return float(地板.mean()), float(含首测.mean())


# ================================================================ 二、覆盖式清除搜索
def 覆盖点列(S1, theta1, delta, rho_lo, rho_hi, R清除=M.R_CLEAR, 列距=28.0):
    """构造一串 /clear 点，保证「以每点为心、半径 20 m 的圆盘之并」覆盖整个楔形。"""
    半 = np.sqrt(max(R清除 ** 2 - (列距 / 2) ** 2, 1e-9))     # 14.28 m
    t = np.tan(np.radians(delta))
    点 = []
    x = rho_lo
    向上 = True
    while x <= rho_hi + 列距:
        w = x * t
        n = max(1, int(np.ceil(w / 半)))
        ys = [w * (2 * i + 1 - n) / n for i in range(n)] if n > 1 else [0.0]
        点 += [(x, y) for y in (ys if 向上 else ys[::-1])]
        向上 = not 向上
        x += 列距
    return [tuple(M.局部转全局(p, S1, theta1)) for p in 点]


def 跑覆盖搜索(sim, S1, theta1, 频道=1, 上限=400):
    """完全不做交会定位：按覆盖点列逐点 /clear，命中即止。"""
    rho_hi = M.距离上界(S1, theta1, DELTA_算法)
    点列 = 覆盖点列(S1, theta1, DELTA_算法, M.R_NEAR, rho_hi)
    for k, P in enumerate(点列[:上限]):
        if sim.clear(P, 频道)["clear_result"] == "success":
            return True
    return False


def 跑混合(sim, S1, theta1, 频道=1):
    """折中方案：只做**一次**交会检测，之后不再检测，直接把定位区域覆盖式清除搜索掉。"""
    Ω = M.先验区域(S1, theta1, DELTA_算法)
    rho_hi = M.距离上界(S1, theta1, DELTA_算法)
    S2 = M.局部转全局((LAM * rho_hi, MU * rho_hi), S1, theta1)
    回 = sim.measure(S2, 频道)
    if 回["measure_result"] == "near":
        return sim.clear(sim.位置, 频道)["clear_result"] == "success"
    if 回["measure_result"] == "direction":
        新 = M.定位区域(Ω, S2, 回["svd_deg"], DELTA_算法)
        if 新:
            Ω = 新
    # 沿定位区域的长轴，按 2*sqrt(R^2 - (半宽)^2) 的间隔布清除点
    D, (A, B) = M.求直径快(Ω)
    A, B = np.asarray(A, float), np.asarray(B, float)
    轴 = (B - A) / max(float(np.hypot(*(B - A))), 1e-9)
    法 = np.array([-轴[1], 轴[0]])
    半宽 = max(float(np.max([abs(np.dot(np.asarray(p, float) - A, 法)) for p in Ω])), 0.0)
    步 = 2.0 * np.sqrt(max(M.R_CLEAR ** 2 - min(半宽, M.R_CLEAR - 1e-6) ** 2, 1.0))
    n = max(1, int(np.ceil(D / 步)))
    for i in range(n):
        P = A + 轴 * (D * (2 * i + 1) / (2 * n))
        if sim.clear(P, 频道)["clear_result"] == "success":
            return True
    return False


def 评估自定义(跑法, 样本, 种子=SEED):
    rng = np.random.default_rng(种子)
    行 = []
    for G, R_eff in 样本:
        环境 = SIM.电磁环境(np.random.default_rng(int(rng.integers(0, 2 ** 31))))
        sim = SIM.模拟器(G, R_eff, 环境)
        第一 = sim.measure((0.0, 0.0), 1)
        if 第一["measure_result"] != "direction":
            continue
        成 = 跑法(sim, (0.0, 0.0), 第一["svd_deg"])
        行.append({"总时间": sim.时间, "检测次数": sim.检测次数,
                   "清除次数": sim.清除次数, "已清除": bool(成),
                   "rho": float(np.hypot(*G)), "异常": 0})
    return pd.DataFrame(行)


def 节2_覆盖搜索(样本, 地板):
    分节("二、覆盖式清除搜索基线：完全不定位，直接用 /clear 覆盖式扫楔形")
    print("为什么必须认真算这一条：清除「未发现」只要 3 秒，比检测的 5 秒还便宜。")
    print("如果覆盖式清除搜索也差不多快，那整套交会定位就白做了。\n")
    print("覆盖构造：列距 28 m，列内点距不超过 2*sqrt(20²-14²) = 28.56 m，")
    print("保证「每点半径 20 m 的圆盘之并」覆盖整个楔形，命中率必为 100%。")
    S1, th1 = (0.0, 0.0), 0.0
    点列 = 覆盖点列(S1, th1, DELTA_算法, M.R_NEAR, M.R_RECV_MAX)
    print(f"    覆盖整条楔形需要 {len(点列)} 个清除点"
          f"（楔形面积 {np.radians(DELTA_算法)*(1500**2-25):.0f} m²，"
          f"单个圆盘 {np.pi*400:.0f} m²）\n")

    表 = 评估自定义(跑覆盖搜索, 样本)
    好 = 表[表["异常"] == 0]
    print(f"    清除率            {好['已清除'].mean():.4f}")
    print(f"    平均总时间        {好['总时间'].mean():7.1f} s"
          f"   （地板的 {好['总时间'].mean()/地板:.2f} 倍）")
    print(f"    中位 / p90        {好['总时间'].median():7.1f} / "
          f"{好['总时间'].quantile(0.9):7.1f} s")
    print(f"    平均清除次数      {好['清除次数'].mean():7.2f} 次")
    print(f"\n    {'ρ 区间':>14} {'例数':>6} {'平均总时间':>11} {'平均清除次数':>14}")
    for 低, 高 in [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500)]:
        m = (好["rho"] >= 低) & (好["rho"] < 高)
        if m.sum() == 0:
            continue
        print(f"    [{低:5d},{高:5d}) {int(m.sum()):6d} "
              f"{float(好[m]['总时间'].mean()):11.1f} "
              f"{float(好[m]['清除次数'].mean()):14.2f}")
    print("    -> 近处的源覆盖式清除搜索尚可，远处急剧恶化：楔形越远越宽，要多列扫，路程翻几倍。")
    return 表


def 节2b_单次交会后覆盖清除(样本, 地板):
    print("\n  折中方案：只测一次交会，之后不再检测，直接把定位区域覆盖清除掉")
    表 = 评估自定义(跑混合, 样本)
    好 = 表[表["异常"] == 0]
    print(f"    清除率 {好['已清除'].mean():.4f}   平均总时间 {好['总时间'].mean():7.1f} s"
          f"   平均清除次数 {好['清除次数'].mean():.2f}")
    return 表


# ================================================================ 三、候选区域边界
def 节3_边界验证(样本, 最优时间):
    分节("三、候选区域的边界验证")
    print(f"把红框的定义改成可直接验证的一句话：**框内每一点的实测平均耗时"
          f" <= 最优值的 105%**。")
    print(f"红框 = 前进 a ∈ [{框['a低']:.0f},{框['a高']:.0f}] m，"
          f"侧移 b ∈ [{框['b低']:.0f},{框['b高']:.0f}] m（ρ_hi = 1500 时）")
    print(f"基准最优时间 {最优时间:.1f} s，阈值 {最优时间*1.05:.1f} s\n")

    角 = [("框·左下角", 框["a低"], 框["b低"]), ("框·右下角", 框["a高"], 框["b低"]),
          ("框·左上角", 框["a低"], 框["b高"]), ("框·右上角", 框["a高"], 框["b高"]),
          ("框·中心", (框["a低"]+框["a高"])/2, (框["b低"]+框["b高"])/2),
          ("框·上边中点", (框["a低"]+框["a高"])/2, 框["b高"]),
          ("框·右边中点", 框["a高"], (框["b低"]+框["b高"])/2),
          ("推荐点", 450, 120), ("建议点(留P3余量)", 450, 150)]
    外 = [("框外·太靠后", 800, 120), ("框外·侧移太大", 390, 320),
          ("框外·旧框右上角", 600, 180), ("框外·纯几何最优", 900, 580),
          ("框外·原点附近", 60, 40), ("框外·大侧移", 300, 600)]
    边缘 = [("边缘·更靠前 a=100", 100, 120), ("边缘·侧移更小 b=35", 390, 35),
            ("边缘·a=600", 600, 120), ("边缘·b=250", 390, 250)]

    print(f"    {'位置':>20} {'a':>6} {'b':>6} {'平均总时间':>11} {'相对最优':>9} {'判定':>8}")
    行 = []
    for 组名, 组 in [("框内", 角), ("框外（应更差）", 外), ("框外邻域（仍可用）", 边缘)]:
        print(f"    ---- {组名} " + "-" * 46)
        for 名, a, b in 组:
            t = SIM.均时(SIM.规则_定点(a, b), 样本, KAP)
            比 = t / 最优时间
            判 = "达标" if 比 <= 1.05 else ("超 5%" if 比 <= 1.20 else "明显更差")
            print(f"    {名:>20} {a:6.0f} {b:6.0f} {t:11.1f} {比:9.3f} {判:>8}")
            行.append({"分组": 组名, "位置": 名, "a": a, "b": b,
                       "平均总时间": t, "相对最优": 比})
    print("\n    -> 框内每一点都必须 <= 1.05，这是候选区域的**定义**，不是事后描述。")
    print("       「框外邻域」说明 5% 域本身比这个内接矩形更大、且不是矩形；")
    print("       我们给矩形是为了好写好用，代价是保守——框外未必不好，框内一定好。")
    return pd.DataFrame(行)


# ================================================================ 三b、覆盖式清除搜索与交会的分界
def 节3b_覆盖与交会(rng):
    分节("三b、什么时候该覆盖式清除搜索、什么时候该交会：按可行距离上界 ρ_hi 分界")
    print("上一节发现：ρ < 300 m 的近源，覆盖式清除搜索（66.7 s）比我们的交会策略（167.5 s）快得多。")
    print("原因很直白——交会要先走到 (450,120)，对一个 150 m 外的源纯属绕远。")
    print("但那是**事后知道 ρ** 才能说的话。可事前知道的是 ρ_hi（可行距离上界）。")
    print("所以正确的问法是：ρ_hi 小到什么程度时，应该放弃交会、直接覆盖式清除搜索？\n")
    print(f"    {'ρ_hi':>7} {'例数':>6} {'交会策略':>10} {'覆盖式清除搜索':>10} {'地板':>10} {'谁更快':>8}")
    出 = []
    for ρ_hi in (200, 300, 400, 600, 800, 1000, 1500):
        样本 = []
        while len(样本) < 150:
            # 在「已知 ρ <= ρ_hi」的条件下按题设先验抽样
            ρ = rng.uniform(M.R_NEAR, ρ_hi)
            if rng.uniform(0, ρ_hi) > ρ:          # 拒绝采样实现密度 ∝ ρ
                continue
            θ = rng.uniform(0, 2 * np.pi)
            G = np.array([ρ * np.cos(θ), ρ * np.sin(θ)])
            样本.append((G, M.R_RECV_MAX))
        地板 = float(np.mean([np.hypot(*G) / M.V_MOVE + M.T_CLEAR_OK for G, _ in 样本]))

        def 跑交会(sim, S1, th1, 频道=1, _h=ρ_hi):
            # 关键：交会一侧必须**同样知道** ρ <= ρ_hi，否则它按 1500 缩尺、
            # 白跑到 (450,120) 去，而覆盖式清除搜索一侧却只扫到 ρ_hi，比较就不公平了。
            return SIM.跑一局(sim, S1, th1, SIM.规则_比例(LAM, MU),
                              κ=KAP, ρ_hi=_h) is not None

        def 跑盲(sim, S1, th1, 频道=1):
            点列 = 覆盖点列(S1, th1, DELTA_算法, M.R_NEAR, ρ_hi)
            for P in 点列[:400]:
                if sim.clear(P, 频道)["clear_result"] == "success":
                    return True
            return False

        t交 = float(评估自定义(跑交会, 样本)["总时间"].mean())
        t盲 = float(评估自定义(跑盲, 样本)["总时间"].mean())
        print(f"    {ρ_hi:7.0f} {len(样本):6d} {t交:10.1f} {t盲:10.1f} {地板:10.1f} "
              f"{('覆盖式清除搜索' if t盲 < t交 else '交会'):>8}")
        出.append({"ρ_hi": ρ_hi, "交会": t交, "覆盖式清除搜索": t盲, "地板": 地板})
    print("\n    -> 分界点就是两列相等的那个 ρ_hi。问题三里每个源都能算出自己的 ρ_hi，")
    print("       据此二选一，比一律交会或一律覆盖式清除搜索都好。")
    return pd.DataFrame(出)


# ================================================================ 四、非原点起测
def 节4_非原点(rng):
    分节("四、起测点不在原点：尺度自适应规则是否真的成立")
    print("推荐规则写成「前进 λ·ρ_hi、侧移 μ·ρ_hi」，其中 ρ_hi 由可行集自动算出。")
    print("若规则成立，则换个起测点、换个示向度，同一组 (λ,μ) 仍应接近最优。\n")
    print(f"    {'起测点 S1':>16} {'ρ_hi(平均)':>11} {'固定(λ,μ)':>11} "
          f"{'重搜最优(λ,μ)':>15} {'重搜后耗时':>11} {'固定规则耗时':>13} {'损失':>7}")
    出 = []
    for S1 in [(0.0, 0.0), (900.0, 0.0), (1400.0, 0.0), (1000.0, 1000.0)]:
        样本 = []
        while len(样本) < 150:
            t = rng.uniform(0, 2 * np.pi)
            r = M.R_AREA * np.sqrt(rng.uniform(0, 1))
            G = np.array([r * np.cos(t), r * np.sin(t)])
            R_eff = rng.uniform(M.R_RECV_MIN, M.R_RECV_MAX)
            d = float(np.hypot(*(G - np.asarray(S1))))
            if d > R_eff or d <= M.R_NEAR:
                continue
            样本.append((G, R_eff))

        def 评(λ, μ):
            r2 = np.random.default_rng(SEED)
            ts = []
            for G, R_eff in 样本:
                环境 = SIM.电磁环境(np.random.default_rng(int(r2.integers(0, 2 ** 31))))
                sim = SIM.模拟器(G, R_eff, 环境, 起点=S1)
                第一 = sim.measure(S1, 1)
                if 第一["measure_result"] != "direction":
                    continue
                SIM.跑一局(sim, S1, 第一["svd_deg"], SIM.规则_比例(λ, μ), κ=KAP)
                ts.append(sim.时间)
            return float(np.mean(ts))

        ρhi = np.mean([M.距离上界(S1, M.方位角(S1, G)) for G, _ in 样本])
        固定 = 评(LAM, MU)
        最好 = (np.inf, None)
        for λ in (0.1, 0.2, 0.3, 0.4, 0.55):
            for μ in (0.04, 0.08, 0.12, 0.20):
                t = 评(λ, μ)
                if t < 最好[0]:
                    最好 = (t, (λ, μ))
        print(f"    {str(tuple(int(v) for v in S1)):>16} {ρhi:11.0f} "
              f"{f'({LAM},{MU})':>11} {str(最好[1]):>15} {最好[0]:11.1f} "
              f"{固定:13.1f} {固定/最好[0]-1:6.1%}")
        出.append({"S1": str(S1), "ρ_hi均值": ρhi, "固定规则": 固定,
                   "重搜最优": 最好[0], "λ*": 最好[1][0], "μ*": 最好[1][1]})
    print("\n    -> 「损失」是照搬固定 (λ,μ) 相对于就地重搜的代价。若都在几个百分点内，")
    print("       说明尺度自适应规则站得住，问题三可以直接照用，不必每次重优化。")
    return pd.DataFrame(出)


# ================================================================ 五、有距离先验
def 节5_有距离先验():
    分节("五、已有部分距离先验时的规则（问题三反复要用）")
    print("问题三里，一个源往往已经被别的检测点约束过，距离已知在 ρ̂ ± Δ 内。")
    print("此时不必再用 ρ_hi 缩尺，直接用闭式：\n")
    print("        前进 a = ρ̂        （站到正横位）")
    print("        侧移 b = sqrt(Δ·(ρ̂+Δ))")
    print("        可得 直径 <= 2 tanδ (ρ̂ + 2Δ)\n")
    print(f"    {'ρ̂':>7} {'Δ':>7} {'侧移 b*':>9} {'保证直径':>9} {'能否一次清除':>12}")
    for ρ in (400, 800, 1200):
        for Δ in (400, 200, 100, 50, 20):
            b = float(M.最优侧移(ρ, ρ - Δ))
            d = float(M.最优直径(ρ, ρ - Δ))
            print(f"    {ρ:7.0f} {Δ:7.0f} {b:9.1f} {d:9.2f} "
                  f"{'是' if d/2 <= M.R_CLEAR else '否':>12}")
    print("\n    -> 距离一旦收到 ±100 m 以内，一次交会就能压进清除半径，不必再补测。")
    print("       这正是问题三「越测越省」的机制：先验越紧，后续每次检测越便宜。")


# ================================================================ 主流程
def main(N=300):
    print("问题二 · 策略对照（为什么这个策略最优）")
    print(f"随机种子 {SEED}，{N} 个随机算例，与 策略仿真.py 用的是同一批样本。")
    print(f"基准策略：λ={LAM}, μ={MU}, κ={KAP}\n")

    rng = np.random.default_rng(SEED)
    样本 = SIM.造样本(rng, N)

    地板, 含首测 = 节1_上帝视角(样本)
    基准 = SIM.均时(SIM.规则_比例(LAM, MU), 样本, KAP)
    print(f"\n    我们的策略实测 {基准:.1f} s，是地板的 {基准/地板:.2f} 倍。")
    print(f"    **全部可改进空间的上界 = {(基准-地板)/基准:.1%}**，"
          f"这是「有没有更好的策略」的确定答案。")

    盲 = 节2_覆盖搜索(样本, 地板)
    混 = 节2b_单次交会后覆盖清除(样本, 地板)
    边界 = 节3_边界验证(样本, 基准)
    对比表 = 节3b_覆盖与交会(np.random.default_rng(SEED + 23))

    分节("六、汇总：所有方案放在同一把尺子上")
    行 = [("理论下界（已知干扰源位置）", 地板, None, None)]
    for 名, 规则 in [("本文策略 (λ=0.30, μ=0.08, κ=0.25)", SIM.规则_比例(LAM, MU)),
                     ("纯径向推进（示向度反转二分）", SIM.规则_比例(0.5, 0.002)),
                     ("微小位移 (λ=μ=0.02)", SIM.规则_比例(0.02, 0.02)),
                     ("纯横向偏移 (λ=0, μ=0.5)", SIM.规则_比例(0.0, 0.5)),
                     ("几何准则最优点 (900,580)", SIM.规则_定点(900, 580)),
                     ("极小极大准则最优点 (1125,925)", SIM.规则_定点(1125, 925))]:
        表 = SIM.评估(规则, 样本, κ=KAP)
        好 = 表[表["异常"] == 0]
        行.append((名, float(好["总时间"].mean()),
                   float(好["检测次数"].mean()), float(好["清除次数"].mean())))
    好盲 = 盲[盲["异常"] == 0]
    行.append(("覆盖式清除搜索（不作交会）", float(好盲["总时间"].mean()),
               1.0, float(好盲["清除次数"].mean())))
    好混 = 混[混["异常"] == 0]
    行.append(("单次交会后覆盖清除", float(好混["总时间"].mean()),
               2.0, float(好混["清除次数"].mean())))
    行.sort(key=lambda r: r[1])
    print(f"    {'方案':>32} {'平均总时间(s)':>13} {'/地板':>7} "
          f"{'检测次数':>9} {'清除次数':>9}")
    for 名, t, m, c in 行:
        print(f"    {名:>32} {t:13.1f} {t/地板:7.2f} "
              f"{('%9.2f' % m) if m else '        —':>9} "
              f"{('%9.2f' % c) if c else '        —':>9}")

    非原点 = 节4_非原点(np.random.default_rng(SEED + 11))
    节5_有距离先验()

    pd.DataFrame([{"方案": n, "平均总时间": t, "相对下界": t / 地板,
                   "平均检测次数": m, "平均清除次数": c} for n, t, m, c in 行]).to_csv(
        HERE / "策略对照汇总.csv", index=False, encoding="utf-8-sig")
    边界.to_csv(HERE / "候选区域边界验证.csv", index=False, encoding="utf-8-sig")
    非原点.to_csv(HERE / "非原点起测.csv", index=False, encoding="utf-8-sig")
    对比表.to_csv(HERE / "覆盖搜索与交会对比.csv", index=False, encoding="utf-8-sig")
    print("\n已写入 策略对照汇总.csv、候选区域边界验证.csv、"
          "非原点起测.csv、覆盖搜索与交会对比.csv")


if __name__ == "__main__":
    with open(HERE / "策略对照结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '策略对照结果.txt'}")
