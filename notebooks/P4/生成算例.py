"""问题四 · 第 3 步：算例生成（只算数，不作图）。

正式测试的案例由官方模拟器随机生成，其分布不可见。本文按题面给出的**全部**已知
条件构造算例，凡题面未规定的一律写成假设并在灵敏度分析里换掉：

    题面规定（正文问题四、附录 1、附件 2 §1.3、§2.1）      本文取值
    ------------------------------------------------      ----------------------------
    干扰源总数 10~16                                       10..16 等概率
    每个频道至多一个干扰源，频道 1..20                     从 20 个频道中不重复随机取
    所有干扰源位于半径 1800 m 的目标区域内                 位置在圆域内均匀（假设）
    有效接收半径 1000~1500 m，各源不同                     U[1000,1500] 独立（假设）
    示向度误差在 [-1°,1°] 内，同地点不变                   均匀分布、各点独立（假设）
    **既有全向干扰源，又有定向干扰源**，个数与方向均未知   每源独立以 q=0.5 为定向，
                                                           并约束两类各至少一个（假设）
    定向源覆盖角度范围为定向方向两侧各 90°                 定向方向 U[0°,360°)（假设）

「均匀」是只知道支撑区间时的最大熵选择，与 P1 §5.1、P2 §8、P3 的口径一致。
**定向比例 q 是问题四新增的一个假设，也是最要紧的一个**——它直接决定策略的难度，
灵敏度分析（灵敏度.py）会把它从 0 扫到 1。

四个具名算例固定种子存成 CSV，供作图与逐例讲解：
    A  13 源、均匀布局、定向占一半——典型局面
    B  16 源（上界）、3 个贴近目标圆边界、定向占多数——最不利于「判空」的局面
    C  10 源（下界）、明显聚集成两团——最不利于「巡回」的局面
    D  13 源、**12 个定向**（只留 1 个全向）——定向比例的极端，检验策略不依赖全向源

输出：算例汇总.csv、算例{A,B,C,D}_干扰源.csv、算例结果.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 公共 as C
import 模拟器 as S

HERE = Path(__file__).resolve().parent
SEED = 20260912


# ================================================================ 位置抽样
def 抽位置(rng, n, 布局="均匀"):
    """在目标圆域内抽 n 个干扰源位置（四种布局与问题三一致，便于横向对照）。"""
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


def 抽定向(rng, n, 定向比例=0.5, 定向个数=None, 两类齐全=True):
    """给 n 个源分配「全向 / 定向」。返回长度 n 的数组，全向为 np.nan。

    题面说「既有全向干扰源，又有定向干扰源」，所以默认约束两类各至少一个。
    定向个数给定时按精确个数抽（灵敏度分析用）。
    """
    if 定向个数 is not None:
        k = int(np.clip(定向个数, 0, n))
        是定向 = np.zeros(n, dtype=bool)
        是定向[rng.choice(n, size=k, replace=False)] = True
    else:
        for _ in range(200):
            是定向 = rng.uniform(0, 1, n) < 定向比例
            if not 两类齐全 or (0 < int(是定向.sum()) < n):
                break
        else:                                   # 极端比例下兜底，强制凑齐两类
            是定向 = rng.uniform(0, 1, n) < 定向比例
            是定向[0], 是定向[1] = True, False
    朝向 = np.full(n, np.nan)
    朝向[是定向] = rng.uniform(0, 360, int(是定向.sum()))
    return 朝向


def 造案例(rng, 源数=None, 布局="均匀", 误差模式="独立", 误差分布="均匀",
           相关长度=300.0, 最小间距=0.0, 定向比例=0.5, 定向个数=None,
           两类齐全=True, R_eff范围=None):
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
    lo, hi = R_eff范围 or (C.R_RECV_MIN, C.R_RECV_MAX)
    R_eff = rng.uniform(lo, hi, n)
    朝向 = 抽定向(rng, n, 定向比例=定向比例, 定向个数=定向个数, 两类齐全=两类齐全)
    源表 = [{"频道": int(频道[i]), "x": float(位置[i, 0]), "y": float(位置[i, 1]),
             "R_eff": float(R_eff[i]),
             "定向方向": (None if np.isnan(朝向[i]) else float(朝向[i]))}
            for i in range(n)]
    环境 = S.电磁环境(rng, 模式=误差模式, 相关长度=相关长度, 分布=误差分布)
    return S.案例(源表, 环境)


def 造一批(rng, N, **kw):
    return [造案例(rng, **kw) for _ in range(N)]


# ================================================================ 具名算例
def 具名算例():
    """A/B/C/D 四个固定算例。种子写死，换机器可完全复现。"""
    出 = {}
    出["A"] = 造案例(np.random.default_rng(SEED + 1), 源数=13, 布局="均匀",
                     定向个数=6)
    出["B"] = 造案例(np.random.default_rng(SEED + 2), 源数=16, 布局="贴边",
                     定向个数=10)
    出["C"] = 造案例(np.random.default_rng(SEED + 3), 源数=10, 布局="聚集",
                     定向个数=5)
    出["D"] = 造案例(np.random.default_rng(SEED + 4), 源数=13, 布局="均匀",
                     定向个数=12)
    return 出


# ================================================================ 原点可收概率
def 原点可收概率(定向比例=0.5):
    """站在原点能收到某个源的概率（理论值）。

    全向：P = E[R_eff²]/R²（源在圆域内均匀、R_eff ~ U[1000,1500]）
    定向：再乘以「原点落在覆盖角内」的概率。定向方向均匀 => 恰好 1/2。
    """
    p距 = (C.R_RECV_MAX ** 3 - C.R_RECV_MIN ** 3) / (3 * (C.R_RECV_MAX - C.R_RECV_MIN)) \
        / C.R_AREA ** 2
    return p距 * (1 - 定向比例) + p距 * 0.5 * 定向比例, p距


def main():
    print("问题四 · 算例生成")
    print(f"随机种子 {SEED}\n")
    算例 = 具名算例()
    汇总 = []
    for 名, 案 in 算例.items():
        表 = pd.DataFrame([{**s, "定向方向": (np.nan if s["定向方向"] is None
                                              else s["定向方向"])} for s in 案.源表])
        表["是定向"] = 表["定向方向"].notna()
        表["距原点"] = np.hypot(表["x"], 表["y"])
        表["原点在距离内"] = 表["距原点"] <= 表["R_eff"]
        表["原点在覆盖角内"] = [C.覆盖角内((r.x, r.y),
                                          None if np.isnan(r.定向方向) else r.定向方向,
                                          (0.0, 0.0)) for r in 表.itertuples()]
        表["原点可收"] = 表["原点在距离内"] & 表["原点在覆盖角内"]
        表["距边界"] = C.R_AREA - 表["距原点"]
        表 = 表.sort_values("频道").reset_index(drop=True)
        表.to_csv(HERE / f"算例{名}_干扰源.csv", index=False, encoding="utf-8-sig")
        位置 = 表[["x", "y"]].to_numpy()
        d = np.hypot(位置[:, None, 0] - 位置[None, :, 0],
                     位置[:, None, 1] - 位置[None, :, 1])
        np.fill_diagonal(d, np.inf)
        _, 巡回 = C.开放TSP((0.0, 0.0), [tuple(p) for p in 位置])
        汇总.append({
            "算例": 名, "源数": len(表), "定向数": int(表["是定向"].sum()),
            "全向数": int((~表["是定向"]).sum()),
            "空频道数": C.频道数 - len(表),
            "频道": " ".join(str(c) for c in 表["频道"]),
            "最近两源间距": float(d.min()),
            "平均距原点": float(表["距原点"].mean()),
            "最远距原点": float(表["距原点"].max()),
            "原点在距离内个数": int(表["原点在距离内"].sum()),
            "原点可收个数": int(表["原点可收"].sum()),
            "最短巡回长度": 巡回,
            "巡回行进时间": 巡回 / C.V_MOVE,
        })
    df = pd.DataFrame(汇总)
    df.to_csv(HERE / "算例汇总.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.drop(columns=["频道"]).round(2).to_string(index=False))
    print()
    for _, r in df.iterrows():
        print(f"算例 {r['算例']}：频道 {r['频道']}")

    p混, p距 = 原点可收概率(0.5)
    print(f"""
各算例中「站在原点就能收到」的源数：""")
    for _, r in df.iterrows():
        print(f"    {r['算例']}：{int(r['原点可收个数'])}/{int(r['源数'])}"
              f"（其中 {int(r['原点在距离内个数'])} 个在距离内，"
              f"差额 {int(r['原点在距离内个数'] - r['原点可收个数'])} 个是被覆盖角挡掉的）")

    print(f"""
理论值（源在圆域内均匀、R_eff ~ U[1000,1500]、定向方向均匀）：

    P(距离够)            = E[R_eff²]/R² = {p距:.4f}          （问题三就是这个数）
    P(全向源原点可收)    = {p距:.4f}
    P(定向源原点可收)    = {p距:.4f} x 1/2 = {p距/2:.4f}
    P(混合、定向占一半)  = {p混:.4f}

**问题三里「不到一半的源能在出发点被发现」，问题四降到不足四分之一。**
这一条把「把机器狗撒出去」的必要性又推进了一步：
问题三撒出去是为了**够得着**，问题四撒出去还要为了**绕到正面**。

另一条更要紧的差别：问题三里一次 no_signal 至少还排除一个 1000 m 圆盘，
问题四里它**什么位置都不排除**（源可能就在脚下背对着你）。
于是「没收到」这件事在问题四里几乎不再是信息，判空必须靠围堵（见 判空几何.py）。
""")
    print("已写入 算例汇总.csv、算例A/B/C/D_干扰源.csv")


if __name__ == "__main__":
    with open(HERE / "算例结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '算例结果.txt'}")
