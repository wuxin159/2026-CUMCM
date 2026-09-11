"""问题三 · 第 3 步：算例生成（只算数，不作图）。

正式测试的案例由官方模拟器随机生成，其分布不可见。本文按题面给出的**全部**已知
条件构造算例，凡题面未规定的一律写成假设并在灵敏度分析里换掉：

    题面规定（附件 2 §1.3、§2.1，题目正文）        本文取值
    ------------------------------------------    ----------------------------
    干扰源总数 10~16                               10..16 等概率
    每个频道至多一个干扰源，频道 1..20             从 20 个频道中不重复随机取
    所有干扰源位于半径 1800 m 的目标区域内         位置在圆域内均匀（假设）
    有效接收半径 1000~1500 m，各源不同             U[1000,1500] 独立（假设）
    示向度误差在 [-1°,1°] 内，同地点不变           均匀分布、各点独立（假设）

「均匀」是只知道支撑区间时的最大熵选择，与 P1 §5.1、P2 §8 的口径一致。
灵敏度分析（灵敏度.py）会换成聚集、贴边、内圈三种非均匀布局，以及三角分布、
最坏 ±δ、空间相关三种误差模型。

三个具名算例 A/B/C 固定种子存成 CSV，供作图与逐例讲解：
    A  13 个源、均匀布局——典型局面
    B  16 个源（上界）、其中 3 个贴近目标圆边界——最不利于「判空」的局面
    C  10 个源（下界）、明显聚集成两团——最不利于「巡回」的局面

输出：算例汇总.csv、算例{A,B,C}_干扰源.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 公共 as C
import 模拟器 as S

HERE = Path(__file__).resolve().parent
SEED = 20260911


# ================================================================ 位置抽样
def 抽位置(rng, n, 布局="均匀"):
    """在目标圆域内抽 n 个干扰源位置。"""
    if 布局 == "均匀":
        t = rng.uniform(0, 2 * np.pi, n)
        r = C.R_AREA * np.sqrt(rng.uniform(0, 1, n))
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    if 布局 == "贴边":                       # 2/3 落在最外圈的环带里
        t = rng.uniform(0, 2 * np.pi, n)
        u = rng.uniform(0, 1, n)
        r = np.where(u < 2 / 3,
                     C.R_AREA * np.sqrt(rng.uniform(0.85, 1.0, n)),
                     C.R_AREA * np.sqrt(rng.uniform(0, 1, n)))
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    if 布局 == "内圈":                       # 全部挤在半径 900 m 以内
        t = rng.uniform(0, 2 * np.pi, n)
        r = 0.5 * C.R_AREA * np.sqrt(rng.uniform(0, 1, n))
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    if 布局 == "聚集":                       # 2 个团，团内标准差 250 m
        中心 = []
        for _ in range(2):
            t = rng.uniform(0, 2 * np.pi)
            r = 0.75 * C.R_AREA * np.sqrt(rng.uniform(0.4, 1.0))
            中心.append(np.array([r * np.cos(t), r * np.sin(t)]))
        出 = []
        while len(出) < n:
            p = 中心[len(出) % 2] + rng.normal(0, 250.0, 2)
            if np.hypot(*p) <= C.R_AREA:
                出.append(p)
        return np.stack(出)
    raise ValueError(f"未知布局：{布局}")


def 造案例(rng, 源数=None, 布局="均匀", 误差模式="独立", 误差分布="均匀",
           相关长度=300.0, 最小间距=0.0):
    """生成一局测试的真值。源数为 None 时在 10~16 之间等概率抽取。"""
    n = int(rng.integers(C.源数下界, C.源数上界 + 1)) if 源数 is None else int(源数)
    while True:
        位置 = 抽位置(rng, n, 布局)
        if 最小间距 <= 0 or n == 1:
            break
        d = np.hypot(位置[:, None, 0] - 位置[None, :, 0],
                     位置[:, None, 1] - 位置[None, :, 1])
        np.fill_diagonal(d, np.inf)
        if d.min() >= 最小间距:
            break
    频道 = rng.choice(np.arange(1, C.频道数 + 1), size=n, replace=False)
    R_eff = rng.uniform(C.R_RECV_MIN, C.R_RECV_MAX, n)
    源表 = [{"频道": int(频道[i]), "x": float(位置[i, 0]), "y": float(位置[i, 1]),
             "R_eff": float(R_eff[i])} for i in range(n)]
    环境 = S.电磁环境(rng, 模式=误差模式, 相关长度=相关长度, 分布=误差分布)
    return S.案例(源表, 环境)


def 造一批(rng, N, **kw):
    return [造案例(rng, **kw) for _ in range(N)]


# ================================================================ 具名算例
def 具名算例():
    """A/B/C 三个固定算例。种子写死，换机器可完全复现。"""
    出 = {}
    出["A"] = 造案例(np.random.default_rng(SEED + 1), 源数=13, 布局="均匀")
    出["B"] = 造案例(np.random.default_rng(SEED + 2), 源数=16, 布局="贴边")
    出["C"] = 造案例(np.random.default_rng(SEED + 3), 源数=10, 布局="聚集")
    return 出


def main():
    print("问题三 · 算例生成")
    print(f"随机种子 {SEED}\n")
    算例 = 具名算例()
    汇总 = []
    for 名, 案 in 算例.items():
        表 = pd.DataFrame(案.源表)
        表["距原点"] = np.hypot(表["x"], 表["y"])
        表["原点可收"] = 表["距原点"] <= 表["R_eff"]
        表["距边界"] = C.R_AREA - 表["距原点"]
        表 = 表.sort_values("频道").reset_index(drop=True)
        表.to_csv(HERE / f"算例{名}_干扰源.csv", index=False, encoding="utf-8-sig")
        位置 = 表[["x", "y"]].to_numpy()
        d = np.hypot(位置[:, None, 0] - 位置[None, :, 0],
                     位置[:, None, 1] - 位置[None, :, 1])
        np.fill_diagonal(d, np.inf)
        _, 巡回 = C.开放TSP((0.0, 0.0), [tuple(p) for p in 位置])
        汇总.append({
            "算例": 名, "源数": len(表),
            "空频道数": C.频道数 - len(表),
            "频道": " ".join(str(c) for c in 表["频道"]),
            "最近两源间距": float(d.min()),
            "平均距原点": float(表["距原点"].mean()),
            "最远距原点": float(表["距原点"].max()),
            "原点可收个数": int(表["原点可收"].sum()),
            "最短巡回长度": 巡回,
            "巡回行进时间": 巡回 / C.V_MOVE,
        })
    df = pd.DataFrame(汇总)
    df.to_csv(HERE / "算例汇总.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.drop(columns=["频道"]).round(2).to_string(index=False))
    print()
    for _, r in df.iterrows():
        print(f"算例 {r['算例']}：频道 {r['频道']}")

    print("\n各算例中「站在原点就能收到」的源数：")
    for _, r in df.iterrows():
        print(f"    {r['算例']}：{int(r['原点可收个数'])}/{int(r['源数'])} = "
              f"{r['原点可收个数'] / r['源数']:.1%}")
    print("\n理论值：源在圆域内均匀、R_eff ~ U[1000,1500] 时，")
    print("    P(原点可收) = E[R_eff²]/R² = "
          f"{(1500 ** 3 - 1000 ** 3) / (3 * 500) / C.R_AREA ** 2:.4f}")
    print("    **只有不到一半的源能在出发点被发现**——这一条直接否定了")
    print("    「站在原点把 20 个频道测一遍就能拿到全部方位」的想法，")
    print("    也就决定了策略必须包含一个把机器狗撒出去的扫描环节。")

    print("\n已写入 算例汇总.csv、算例A/B/C_干扰源.csv")


if __name__ == "__main__":
    with open(HERE / "算例结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '算例结果.txt'}")
