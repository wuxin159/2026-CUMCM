"""问题二 · 第 2 步：在 (a,b) 平面上扫出候选区域（只算数，不作图）。

局部标架：S1 在原点，示向度 theta1 指向 x 轴正向；候选第二检测点 S2 = (a,b)。
b 的正负对称，只扫 b > 0，论文中说明另一侧镜像即可。

三个目标函数（论文里按这个顺序讲）：
    甲 纯几何   J_D    = E_rho[ 定位区域直径 ]           —— 不计代价，答案「走得越远越好」
    乙 限时几何 J_D 在 |S1S2| <= 预算 下的最优           —— 给定时间预算时怎么放
    丙 时间     J_T    = E_rho[ 从现在到该源被清除的总虚拟时间 ]  —— 真正要最小化的量

输出：
    候选区域网格.csv      整张网格的各项指标，供 绘图.py 与 PGFPlots 使用
    推荐点.csv            各目标下的最优点及其性能
    候选区域结果.txt      文字摘要
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 模型 as M

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 扫描参数（写死，保证可复现）
A网 = np.arange(-400.0, 1801.0, 25.0)
B网 = np.arange(10.0, 1201.0, 15.0)
RHO节点数 = 24


# ================================================================ 代价模型
def 收尾(D, r):
    """从「已知区域直径为 D、机器狗距区域中心 r」到清除完成，还要多走多远、多测几次。

    单步规则：走到区域中心的正横方向、侧移 Lambda = D/2 处补测一次，再走回中心。
    额外路程 = Lambda + Lambda^2/(2r)（后项是绕到正横位的勾股余量）；
    补测后新的区域尺度 D' = 4 Lambda tan delta（正横 45° 交会的结果）。
    D' 收缩极快（比值 2 tan delta ≈ 0.035），循环至多两三步。
    """
    路程, 次数 = 0.0, 0
    while D / 2.0 > M.R_CLEAR and 次数 < 8:
        Λ = D / 2.0
        路程 += Λ + Λ * Λ / (2.0 * max(r, 1.0))
        次数 += 1
        D = 4.0 * Λ * M.TAN_DELTA
        r = Λ
    return 路程, 次数


def 单点评估(Ω1, a, b, rho节点, 权重):
    """给定候选点 (a,b)，在 rho 的先验下评估各项指标。误差取 eps1 = eps2 = 0。"""
    D = np.empty_like(rho节点)
    R = np.empty_like(rho节点)
    T = np.empty_like(rho节点)
    走近 = np.hypot(rho节点 - a, b)                 # |S2 G|
    到S2 = float(np.hypot(a, b))

    for k, rho in enumerate(rho节点):
        G = (float(rho), 0.0)
        th2 = M.方位角((a, b), G)
        L = M.定位区域(Ω1, (a, b), th2)
        if not L:
            D[k] = R[k] = T[k] = np.nan
            continue
        d, r, c, _ = M.区域指标(L)
        D[k], R[k] = d, r
        到中心 = float(np.hypot(c[0] - a, c[1] - b))
        if r <= M.R_CLEAR:
            路程, 次数 = 0.0, 0
        else:
            路程, 次数 = 收尾(d, max(到中心, 1.0))
        T[k] = ((到S2 + 到中心 + 路程) / M.V_MOVE
                + M.T_MEASURE * (1 + 次数) + M.T_CLEAR_OK)

    好 = ~np.isnan(D)
    w = 权重[好] / 权重[好].sum()
    return {
        "a": a, "b": b,
        "到S2": 到S2,
        "行进时间": 到S2 / M.V_MOVE,
        "D均值": float(np.dot(w, D[好])),
        "D最大": float(np.nanmax(D)),
        "D中位": float(np.percentile(D[好], 50)),
        "Rmin均值": float(np.dot(w, R[好])),
        "一次清除概率": float(np.dot(w, (R[好] <= M.R_CLEAR).astype(float))),
        "时间均值": float(np.dot(w, T[好])),
        "时间最大": float(np.nanmax(T)),
        "保证收信": bool(np.all(M.保证收信(rho节点, a, b))),
        "收信概率": float(np.dot(权重 / 权重.sum(), M.收信概率(rho节点, a, b))),
        "绕行均值": float(np.dot(权重 / 权重.sum(), M.绕行(rho节点, a, b))),
    }


# ================================================================ 主流程
def 扫描(先验="题设"):
    Ω1 = M.先验区域()
    rho节点, 权重 = M.距离节点(RHO节点数, 先验=先验)
    行 = []
    for a in A网:
        for b in B网:
            行.append(单点评估(Ω1, float(a), float(b), rho节点, 权重))
    return pd.DataFrame(行)


def 最优(表, 列, 约束=None):
    子 = 表 if 约束 is None else 表[约束]
    if len(子) == 0:
        return None
    return 子.loc[子[列].idxmin()]


def 打印点(名, 行):
    if 行 is None:
        print(f"    {名:<26} 无可行点")
        return
    print(f"    {名:<26} a={行['a']:7.0f}  b={行['b']:7.0f}  "
          f"|S1S2|={行['到S2']:7.0f} m ({行['行进时间']:5.1f} s)  "
          f"E[D]={行['D均值']:7.1f} m  maxD={行['D最大']:7.1f} m  "
          f"E[T]={行['时间均值']:6.1f} s  P(一次清除)={行['一次清除概率']:.3f}  "
          f"保证收信={'是' if 行['保证收信'] else '否'}")


def main():
    print("问题二 · 候选区域扫描")
    print(f"网格：a ∈ [{A网[0]:.0f},{A网[-1]:.0f}] 步长 {A网[1]-A网[0]:.0f} m（{len(A网)} 点），"
          f"b ∈ [{B网[0]:.0f},{B网[-1]:.0f}] 步长 {B网[1]-B网[0]:.0f} m（{len(B网)} 点）")
    print(f"rho 节点 {RHO节点数} 个；误差取 eps1 = eps2 = 0（§7 已验证误差对直径的影响在百分之几）")
    print("S1 在原点、theta1 = 0 的局部标架；b < 0 与 b > 0 完全对称。\n")

    表 = 扫描("题设")
    表.to_csv(HERE / "候选区域网格.csv", index=False, encoding="utf-8-sig")

    print("=" * 78)
    print("一、三个目标各自的最优点")
    print("=" * 78)
    推荐 = []
    项 = [
        ("甲 纯几何 min E[D]", 最优(表, "D均值"), None),
        ("甲' 纯几何 min max D", 最优(表, "D最大"), None),
        ("丙 时间 min E[T]", 最优(表, "时间均值"), None),
        ("丙' 时间 min max T", 最优(表, "时间最大"), None),
    ]
    for 名, 行, _ in 项:
        打印点(名, 行)
        if 行 is not None:
            推荐.append({"目标": 名, **行.to_dict()})

    print("\n    只允许落在「保证收信」透镜内：")
    透镜 = 表["保证收信"]
    for 名, 列 in [("甲 min E[D] | 透镜内", "D均值"), ("丙 min E[T] | 透镜内", "时间均值")]:
        行 = 最优(表, 列, 透镜)
        打印点(名, 行)
        if 行 is not None:
            推荐.append({"目标": 名, **行.to_dict()})

    print("\n" + "=" * 78)
    print("二、乙 限时几何：给定行进时间预算，能把 E[D] 压到多少")
    print("=" * 78)
    print(f"    {'预算(s)':>8} {'预算(m)':>8} {'最优a':>7} {'最优b':>7} "
          f"{'E[D](m)':>9} {'maxD(m)':>9} {'E[T](s)':>9} {'保证收信':>8}")
    for 预算 in (20, 40, 60, 80, 100, 140, 200, 300, 400):
        行 = 最优(表, "D均值", 表["行进时间"] <= 预算 + 1e-9)
        if 行 is None:
            continue
        print(f"    {预算:8.0f} {预算*M.V_MOVE:8.0f} {行['a']:7.0f} {行['b']:7.0f} "
              f"{行['D均值']:9.1f} {行['D最大']:9.1f} {行['时间均值']:9.1f} "
              f"{'是' if 行['保证收信'] else '否':>8}")
        推荐.append({"目标": f"乙 预算{预算}s", **行.to_dict()})
    print("    -> E[D] 随预算单调下降但收益递减；而 E[T] 有内部极小，这就是为什么")
    print("       「定位效果」不能脱离代价单独定义。")

    print("\n" + "=" * 78)
    print("三、候选区域：目标丙的 alpha-最优域")
    print("=" * 78)
    最好T = 表["时间均值"].min()
    print(f"    最优期望总时间 T* = {最好T:.1f} s")
    print(f"    {'alpha':>6} {'阈值(s)':>9} {'点数':>6} {'占比':>7} "
          f"{'a 范围':>18} {'b 范围':>18} {'|S1S2| 范围':>16}")
    for α in (0.02, 0.05, 0.10, 0.20, 0.35):
        m = 表["时间均值"] <= 最好T * (1 + α)
        子 = 表[m]
        print(f"    {α:6.2f} {最好T*(1+α):9.1f} {m.sum():6d} {m.mean():7.2%} "
              f"[{子['a'].min():7.0f},{子['a'].max():7.0f}] "
              f"[{子['b'].min():7.0f},{子['b'].max():7.0f}] "
              f"[{子['到S2'].min():6.0f},{子['到S2'].max():6.0f}]")

    print(f"\n    同一族阈值下的纯几何 alpha-最优域（目标甲，E[D]* = {表['D均值'].min():.1f} m）：")
    最好D = 表["D均值"].min()
    print(f"    {'alpha':>6} {'阈值(m)':>9} {'点数':>6} {'占比':>7} "
          f"{'a 范围':>18} {'b 范围':>18} {'|S1S2| 范围':>16}")
    for α in (0.02, 0.05, 0.10, 0.20, 0.35):
        m = 表["D均值"] <= 最好D * (1 + α)
        子 = 表[m]
        print(f"    {α:6.2f} {最好D*(1+α):9.1f} {m.sum():6d} {m.mean():7.2%} "
              f"[{子['a'].min():7.0f},{子['a'].max():7.0f}] "
              f"[{子['b'].min():7.0f},{子['b'].max():7.0f}] "
              f"[{子['到S2'].min():6.0f},{子['到S2'].max():6.0f}]")

    print("\n" + "=" * 78)
    print("四、推荐点的逐 rho 性能")
    print("=" * 78)
    Ω1 = M.先验区域()
    rho节点, 权重 = M.距离节点(RHO节点数)
    推荐点 = [("时间最优", 最优(表, "时间均值")),
              ("时间最优|透镜内", 最优(表, "时间均值", 透镜)),
              ("几何最优|预算60s", 最优(表, "D均值", 表["行进时间"] <= 60))]
    for 名, 行 in 推荐点:
        if 行 is None:
            continue
        a, b = float(行["a"]), float(行["b"])
        print(f"\n    {名}：S2 = ({a:.0f}, {b:.0f})，"
              f"|S1S2| = {np.hypot(a,b):.0f} m，行进 {np.hypot(a,b)/M.V_MOVE:.0f} s")
        print(f"      {'rho':>6} {'交会角':>8} {'|S2G|':>8} {'D(m)':>9} {'Rmin(m)':>9} "
              f"{'一次清除':>8} {'T(s)':>8} {'保证收信':>8}")
        for rho in (100, 300, 500, 700, 900, 1100, 1300, 1500):
            G = (float(rho), 0.0)
            th2 = M.方位角((a, b), G)
            L = M.定位区域(Ω1, (a, b), th2)
            d, r, c, _ = M.区域指标(L)
            到中心 = float(np.hypot(c[0] - a, c[1] - b))
            路程, 次数 = (0.0, 0) if r <= M.R_CLEAR else 收尾(d, max(到中心, 1.0))
            T = ((np.hypot(a, b) + 到中心 + 路程) / M.V_MOVE
                 + M.T_MEASURE * (1 + 次数) + M.T_CLEAR_OK)
            print(f"      {rho:6.0f} {float(M.交会角(rho,a,b)):8.2f} "
                  f"{np.hypot(rho-a,b):8.0f} {d:9.2f} {r:9.2f} "
                  f"{'是' if r <= M.R_CLEAR else '否':>8} {T:8.1f} "
                  f"{'是' if bool(M.保证收信(rho,a,b)) else '否':>8}")

    print("\n" + "=" * 78)
    print("五、灵敏度：换成 rho 上的均匀先验")
    print("=" * 78)
    表2 = 扫描("均匀")
    for 名, 列 in [("甲 min E[D]", "D均值"), ("丙 min E[T]", "时间均值")]:
        打印点(名 + " | 均匀先验", 最优(表2, 列))
        推荐.append({"目标": 名 + "|均匀先验", **最优(表2, 列).to_dict()})
    print("    -> 先验换掉后最优点位置的变化幅度，就是这个策略对分布假设的敏感度。")

    pd.DataFrame(推荐).to_csv(HERE / "推荐点.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写入 候选区域网格.csv（{len(表)} 行）与 推荐点.csv")


if __name__ == "__main__":
    with open(HERE / "候选区域结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '候选区域结果.txt'}")
