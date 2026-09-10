"""问题二 · 第 5 步：把作图要用的几何量导成 CSV（只算数，不作图）。

仓库规矩：算与画分离——数据脚本不 import matplotlib，绘图脚本只读 CSV。
定稿后这些 CSV 直接挪进 results/ 交 PGFPlots，论文成图不必再跑 Python。

输出（前缀统一为「图_」，便于日后整体搬运）：
    图_先验区域.csv      Ω1 的顶点（局部标架）
    图_交会示意.csv      一组典型 (S1,S2,G) 的楔形边界与定位区域顶点
    图_定位区域顶点.csv  上述定位区域的顶点
    图_直径随距离.csv    若干候选第二点的 D(rho) 与 Rmin(rho) 曲线
    图_透镜边界.csv      「保证收到信号」透镜域的边界
    图_轨迹示例.csv      一局完整策略的机器狗轨迹与逐步收缩的区域尺度
"""
from pathlib import Path

import numpy as np
import pandas as pd

import 模型 as M
import 策略仿真 as SIM

HERE = Path(__file__).resolve().parent
SEED = 20260910

# 与 候选区域.py / 策略仿真.py 的输出保持一致：
#   仿真最优  λ=0.30, μ=0.08  ->  (450, 120)      策略仿真结果.txt §二
#   代理最优  (450, 145)                          候选区域结果.txt §一
#   纯几何最优 (900, 580)                          候选区域结果.txt §一
候选点 = [("推荐点(仿真最优)", 450.0, 120.0),
          ("代理模型最优", 450.0, 145.0),
          ("纯几何最优", 900.0, 580.0),
          ("纯侧移", 0.0, 750.0),
          ("几乎不侧移", 450.0, 20.0)]


def 写(名, 表):
    表.to_csv(HERE / 名, index=False, encoding="utf-8-sig")
    print(f"  {名:<24} {len(表):>6} 行")


def main():
    print("导出作图数据：")

    # ---------------------------------------------------------- 1 先验区域
    Ω1 = M.先验区域()
    写("图_先验区域.csv", pd.DataFrame(Ω1, columns=["x", "y"]))

    # ---------------------------------------------------------- 2 交会示意
    rho, a, b = 900.0, 300.0, 150.0
    G = (rho, 0.0)
    S1, S2 = (0.0, 0.0), (a, b)
    th1, th2 = 0.0, M.方位角(S2, G)
    段 = []
    for 名, S, th in [("S1", S1, th1), ("S2", S2, th2)]:
        for 侧, 标 in ((-1, "下边"), (0, "中线"), (1, "上边")):
            ang = np.radians(th + 侧 * M.DELTA)
            L = 1600.0
            段.append({"楔形": 名, "线": 标,
                       "x1": S[0], "y1": S[1],
                       "x2": S[0] + L * np.cos(ang), "y2": S[1] + L * np.sin(ang)})
    写("图_交会示意.csv", pd.DataFrame(段))
    L = M.定位区域(Ω1, S2, th2)
    d, R, c, A = M.区域指标(L)
    顶点表 = pd.DataFrame(L, columns=["x", "y"])
    顶点表["直径"] = d
    顶点表["Rmin"] = R
    顶点表["圆心x"], 顶点表["圆心y"] = c[0], c[1]
    顶点表["Gx"], 顶点表["Gy"] = G[0], G[1]
    顶点表["S2x"], 顶点表["S2y"] = a, b
    写("图_定位区域顶点.csv", 顶点表)

    # ---------------------------------------------------------- 3 直径随距离
    行 = []
    rho网 = np.arange(20.0, 1501.0, 10.0)
    for 名, a, b in 候选点:
        for rho in rho网:
            Lp = M.定位区域(Ω1, (a, b), M.方位角((a, b), (float(rho), 0.0)))
            if not Lp:
                continue
            d, R = M.区域指标(Lp)[:2]
            行.append({"候选点": 名, "a": a, "b": b, "rho": rho,
                       "直径": d, "Rmin": R,
                       "交会角": float(M.交会角(rho, a, b)),
                       "闭式直径": float(M.近似直径(rho, a, b)),
                       "保证收信": int(bool(M.保证收信(rho, a, b)))})
    写("图_直径随距离.csv", pd.DataFrame(行))

    # ---------------------------------------------------------- 4 保证收信透镜
    # 边界 = D((5,0),1000) 与 D((1000,0),1000) 两段圆弧
    弧 = []
    for 心, 记 in [((M.R_NEAR, 0.0), "近端圆"), ((M.R_RECV_MIN, 0.0), "远端圆")]:
        for t in np.linspace(0, 2 * np.pi, 721):
            p = (心[0] + M.R_RECV_MIN * np.cos(t), 心[1] + M.R_RECV_MIN * np.sin(t))
            在另一圆内 = (np.hypot(p[0] - (M.R_RECV_MIN if 记 == "近端圆" else M.R_NEAR),
                                   p[1]) <= M.R_RECV_MIN + 1e-9)
            if 在另一圆内:
                弧.append({"来源": 记, "x": p[0], "y": p[1]})
    写("图_透镜边界.csv", pd.DataFrame(弧))

    # ---------------------------------------------------------- 5 一局轨迹示例
    rng = np.random.default_rng(SEED)
    G = np.array([760.0, 430.0])                 # 固定一个典型算例，便于论文引用
    环境 = SIM.电磁环境(np.random.default_rng(SEED), 模式="独立")
    sim = SIM.模拟器(G, 1250.0, 环境)
    第一 = sim.measure((0.0, 0.0), 1)
    Ω = M.先验区域((0.0, 0.0), 第一["svd_deg"], SIM.DELTA_算法)
    ρ_hi = M.距离上界((0.0, 0.0), 第一["svd_deg"], SIM.DELTA_算法)
    轨迹 = [{"步": 0, "动作": "检测1", "x": 0.0, "y": 0.0,
             "虚拟时刻": sim.时间, "区域直径": M.区域指标(Ω)[0],
             "区域Rmin": M.区域指标(Ω)[1]}]
    目标 = M.局部转全局((0.20 * ρ_hi, 0.08 * ρ_hi), (0.0, 0.0), 第一["svd_deg"])
    区域历史 = [("检测1", list(Ω))]
    for 步 in range(1, 8):
        回 = sim.measure(目标, 1)
        if 回["measure_result"] == "direction":
            新 = M.定位区域(Ω, 目标, 回["svd_deg"], SIM.DELTA_算法)
            if 新:
                Ω = 新
        d, R, c, _ = M.区域指标(Ω)
        轨迹.append({"步": 步, "动作": f"检测{步+1}", "x": float(目标[0]), "y": float(目标[1]),
                     "虚拟时刻": sim.时间, "区域直径": d, "区域Rmin": R})
        区域历史.append((f"检测{步+1}", list(Ω)))
        if R <= M.R_CLEAR:
            sim.clear(np.asarray(c, float), 1)
            轨迹.append({"步": 步 + 1, "动作": "清除", "x": float(c[0]), "y": float(c[1]),
                         "虚拟时刻": sim.时间, "区域直径": d, "区域Rmin": R})
            break
        目标, _ = SIM.补测点(Ω, sim.位置, 0.25)
    表 = pd.DataFrame(轨迹)
    表["真源x"], 表["真源y"] = G[0], G[1]
    写("图_轨迹示例.csv", 表)
    区域行 = []
    for 名, 顶点 in 区域历史:
        for p in 顶点:
            区域行.append({"阶段": 名, "x": p[0], "y": p[1]})
    写("图_轨迹区域.csv", pd.DataFrame(区域行))
    print(f"  示例算例：真源 ({G[0]:.0f},{G[1]:.0f})，"
          f"共 {sim.检测次数} 次检测，虚拟总时间 {sim.时间:.1f} s")


if __name__ == "__main__":
    main()
