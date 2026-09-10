"""问题二 · 阶段 2、3 的图表数据（只算数，不作图）。

分析分五个阶段，本脚本补齐第 2、3 阶段作图要用而别处没有的数据：

    阶段 2（时间加权）：效率前沿——给定行进距离，纯几何最好能做到多少
    阶段 3（25 米换算）：边际账——侧移与前进各自的代价/收益曲线
                          途中加测——一点式与两点式按距离分段的对照

输出：
    效率前沿.csv       图 4
    边际账_侧移.csv     图 5 左
    边际账_前进.csv     图 5 右
    途中加测对比.csv    图 6
    代价分解.csv        表：总时间折成米之后的去向
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 模型 as M
import 策略仿真 as SIM

HERE = Path(__file__).resolve().parent
SEED = SIM.SEED
DA = SIM.DELTA_算法
KAP = 0.25
一米 = 1.0 / M.V_MOVE                      # 1 米 = 0.2 秒
一次检测折合米 = M.T_MEASURE * M.V_MOVE      # 5 秒 × 5 m/s = 25 米


def 写(名, 表):
    表.to_csv(HERE / 名, index=False, encoding="utf-8-sig")
    print(f"  {名:<22} {len(表):>6} 行")
    return 表


# ================================================================ 阶段 2：效率前沿
def 效率前沿():
    """给定「走多远」这个预算，纯几何目标最好能把 E[D] 压到多少。

    两个口径的答案都是这条曲线上的点，只是取了不同的预算。
    曲线在 E[D] ≈ 50 m 处饱和，而一次清除需要 D <= 40 m——**够不着**，
    这是「不该为几何精度多走路」的根据。
    """
    网 = pd.read_csv(HERE / "候选区域网格.csv", encoding="utf-8-sig")
    网 = 网[网["a"] >= -50].copy()
    网["d"] = np.hypot(网["a"], 网["b"])
    网["alpha"] = np.degrees(np.arctan2(网["b"], 网["a"]))
    行 = []
    for 中 in np.arange(100, 1750, 50.0):
        s = 网[(网["d"] >= 中 - 30) & (网["d"] < 中 + 30)]
        if len(s) < 3:
            continue
        g = s.loc[s["D均值"].idxmin()]              # 该预算下几何最优
        t = s.loc[s["时间均值"].idxmin()]           # 该预算下时间最优
        行.append({
            "行进距离": 中, "行进时间": 中 / M.V_MOVE,
            "几何最优_alpha": g["alpha"], "几何最优_ED": g["D均值"],
            "几何最优_ET": g["时间均值"], "几何最优_一次清除": g["一次清除概率"],
            "几何最优_收信率": g["收信概率"],
            "时间最优_alpha": t["alpha"], "时间最优_ED": t["D均值"],
            "时间最优_ET": t["时间均值"],
            "一次清除门槛": 2 * M.R_CLEAR,
        })
    return 写("效率前沿.csv", pd.DataFrame(行))


# ================================================================ 阶段 3：边际账
def 边际账(rho=855.0, a固=450.0, b固=120.0):
    """把「侧移」「前进」两个方向的代价与收益都折成米，画出交点。

    代价 = 相对直奔干扰源多走的路程（米）
    收益 = 定位区域变小之后，收尾补测少走的路程（米），按 收尾绕路 ≈ κ·D/2 折算
    两者同单位，可以直接相减，交点就是最优。
    """
    行 = []
    for b in np.concatenate([np.arange(10, 200, 5.0), np.arange(200, 900, 20.0)]):
        D = float(M.近似直径(rho, a固, b))
        行.append({"侧移b": b,
                   "绕路米": float(M.绕行(rho, a固, b)),
                   "区域直径": D,
                   "收尾绕路米": KAP * D / 2.0,
                   "合计米": float(M.绕行(rho, a固, b)) + KAP * D / 2.0})
    侧 = 写("边际账_侧移.csv", pd.DataFrame(行))

    行 = []
    for a in np.arange(-100, 1300, 20.0):
        D = float(M.近似直径(rho, a, b固))
        # 超前惩罚：走过头的部分要走两遍，按题设先验取期望
        节, w = M.距离节点(400)
        超前 = 2.0 * float(np.dot(w, np.maximum(a - 节, 0.0)))
        行.append({"前进a": a,
                   "绕路米": float(M.绕行(rho, a, b固)),
                   "区域直径": D,
                   "收尾绕路米": KAP * D / 2.0,
                   "期望超前惩罚米": 超前,
                   "合计米": float(M.绕行(rho, a, b固)) + KAP * D / 2.0})
    前 = 写("边际账_前进.csv", pd.DataFrame(行))
    return 侧, 前


# ================================================================ 阶段 3：途中加测
def 跑序列(sim, S1, th1, 点表, 频道=1, 最多=12, κ=KAP):
    """按给定点表逐点检测，点表用完后交给标准的自适应补测规则。"""
    Ω = M.先验区域(S1, th1, DA)
    序 = [M.局部转全局(p, S1, th1) for p in 点表]
    目标, k = 序[0], 0
    for _ in range(最多):
        回 = sim.measure(目标, 频道)
        if 回["measure_result"] == "near":
            sim.clear(sim.位置, 频道)
            return
        if 回["measure_result"] == "direction":
            新 = M.定位区域(Ω, 目标, 回["svd_deg"], DA)
            if 新:
                Ω = 新
        d, R = M.区域指标(Ω)[:2]
        c = np.asarray(M.最小覆盖圆快(Ω)[0], dtype=float)
        if R <= M.R_CLEAR:
            sim.clear(c, 频道)
            return
        k += 1
        目标 = 序[k] if k < len(序) else SIM.补测点(Ω, sim.位置, κ)[0]
    sim.clear(np.asarray(M.最小覆盖圆快(Ω)[0], dtype=float), 频道)


def 评估序列(工厂, 样本, 种子=SEED):
    rng = np.random.default_rng(种子)
    行 = []
    for G, R_eff in 样本:
        环境 = SIM.电磁环境(np.random.default_rng(int(rng.integers(0, 2 ** 31))))
        sim = SIM.模拟器(G, R_eff, 环境)
        第一 = sim.measure((0.0, 0.0), 1)
        if 第一["measure_result"] != "direction":
            continue
        ρ_hi = M.距离上界((0.0, 0.0), 第一["svd_deg"], DA)
        跑序列(sim, (0.0, 0.0), 第一["svd_deg"], 工厂(ρ_hi))
        行.append({"总时间": sim.时间, "路程": sim.路程, "移动时间": sim.移动时间,
                   "检测时间": sim.检测时间, "清除时间": sim.清除时间,
                   "检测次数": sim.检测次数, "rho": float(np.hypot(*G))})
    return pd.DataFrame(行)


def 途中加测(样本):
    """一点式 vs 两点式：25 米换算下「顺路免费加测一次」值不值。"""
    一点 = 评估序列(lambda h: [(0.30 * h, 0.08 * h)], 样本)
    两点 = 评估序列(lambda h: [(0.15 * h, 0.04 * h), (0.35 * h, 0.08 * h)], 样本)
    行 = []
    for 低, 高 in [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500)]:
        m1 = (一点["rho"] >= 低) & (一点["rho"] < 高)
        m2 = (两点["rho"] >= 低) & (两点["rho"] < 高)
        if m1.sum() == 0:
            continue
        行.append({"区间下": 低, "区间上": 高, "例数": int(m1.sum()),
                   "一点式": float(一点[m1]["总时间"].mean()),
                   "两点式": float(两点[m2]["总时间"].mean()),
                   "差": float(两点[m2]["总时间"].mean() - 一点[m1]["总时间"].mean()),
                   "一点式检测次数": float(一点[m1]["检测次数"].mean()),
                   "两点式检测次数": float(两点[m2]["检测次数"].mean())})
    行.append({"区间下": 0, "区间上": 1500, "例数": len(一点),
               "一点式": float(一点["总时间"].mean()),
               "两点式": float(两点["总时间"].mean()),
               "差": float(两点["总时间"].mean() - 一点["总时间"].mean()),
               "一点式检测次数": float(一点["检测次数"].mean()),
               "两点式检测次数": float(两点["检测次数"].mean())})
    return 写("途中加测对比.csv", pd.DataFrame(行)), 一点, 两点


# ================================================================ 阶段 3：代价分解
def 代价分解(一点, 两点, 样本):
    直奔 = float(np.mean([np.hypot(*G) / M.V_MOVE + M.T_CLEAR_OK for G, _ in 样本]))
    行 = []
    for 名, 表 in [("一点式", 一点), ("两点式", 两点)]:
        路 = float(表["路程"].mean())
        直路 = float(np.mean([np.hypot(*G) for G, _ in 样本]))
        额外检测 = (float(表["检测次数"].mean()) - 1) * M.T_MEASURE
        行 += [
            {"策略": 名, "项目": "移动", "秒": float(表["移动时间"].mean()),
             "折合米": float(表["移动时间"].mean()) * M.V_MOVE},
            {"策略": 名, "项目": "检测", "秒": float(表["检测时间"].mean()),
             "折合米": float(表["检测时间"].mean()) * M.V_MOVE},
            {"策略": 名, "项目": "清除", "秒": float(表["清除时间"].mean()),
             "折合米": float(表["清除时间"].mean()) * M.V_MOVE},
            {"策略": 名, "项目": "总计", "秒": float(表["总时间"].mean()),
             "折合米": float(表["总时间"].mean()) * M.V_MOVE},
            {"策略": 名, "项目": "—上帝视角地板", "秒": 直奔, "折合米": 直奔 * M.V_MOVE},
            {"策略": 名, "项目": "—超出：绕路", "秒": (路 - 直路) / M.V_MOVE,
             "折合米": 路 - 直路},
            {"策略": 名, "项目": "—超出：额外检测", "秒": 额外检测,
             "折合米": 额外检测 * M.V_MOVE},
            {"策略": 名, "项目": "—超出：首次检测(题设)", "秒": M.T_MEASURE,
             "折合米": 一次检测折合米},
        ]
    return 写("代价分解.csv", pd.DataFrame(行))


def main():
    print("阶段 2、3 数据导出：")
    print(f"  换算：1 次检测 = {M.T_MEASURE:.0f} s = {一次检测折合米:.0f} m；"
          f"1 次清除 = {M.T_CLEAR_OK:.0f} s = {M.T_CLEAR_OK*M.V_MOVE:.0f} m")
    效率前沿()
    边际账()
    rng = np.random.default_rng(SEED)
    样本 = SIM.造样本(rng, 300)
    对比, 一点, 两点 = 途中加测(样本)
    代价分解(一点, 两点, 样本)
    print(f"\n  途中加测：一点式 {对比.iloc[-1]['一点式']:.1f} s -> "
          f"两点式 {对比.iloc[-1]['两点式']:.1f} s"
          f"（{对比.iloc[-1]['差']:+.1f} s，检测次数 "
          f"{对比.iloc[-1]['一点式检测次数']:.2f} -> {对比.iloc[-1]['两点式检测次数']:.2f}）")


if __name__ == "__main__":
    main()
