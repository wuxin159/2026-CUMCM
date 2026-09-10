"""问题一 · 第 1 步：生成算例（只造数据，不作图）。

题目「已知若干个检测点坐标及某干扰源关于这些检测点的示向度」——
「某干扰源」为单数，故每组算例只含 1 个干扰源、n 个检测点，
每个检测点对该干扰源给出且仅给出 1 条示向度。

示向度一律保留全精度浮点：问题一的示向度是题目给的理论量，不经过模拟器，
故楔形半角取精确的 1°。附件 2（§2.3、§7.2、§7.3）规定模拟器的 svd_deg
只保留两位小数，舍入会使真值与报数的偏差达到 1.005°，单个检测点约有
0.13% 的概率被 1° 楔形排除；该结论写入论文以示严谨，并在问题三、四对接
模拟器时由 生成算例(..., 模拟器舍入=True) 复现、算法侧改用 1.005° 半角。

输出（探索阶段一律留在本目录，待结论确定后再挪进 results/ 交 PGFPlots 作图）：
    算例汇总.csv
    算例{名}_干扰源.csv
    算例{名}_检测点.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
输出目录 = HERE

# ---------------------------------------------------------------- 题目常量
R_AREA = 1800.0          # 目标区域半径，米
DELTA = 1.0              # 示向度误差半角，度（附录 2：误差在 [-1°, 1°] 内）
R_RECV_MIN = 1000.0      # 有效接收半径下界，米（保证能收到）
R_RECV_MAX = 1500.0      # 有效接收半径上界，米
R_NEAR = 5.0             # 近距离阈值，米。距离不超过此值时信号过强，测不出示向度（附录 2 第 9 条）

# 模拟器把 svd_deg 截成两位小数（附件 2），舍入本身会再引入至多半格的偏差。
SVD_DECIMALS = 2
舍入裕度 = 0.5 * 10.0 ** (-SVD_DECIMALS)

SEED = 20260910


def 方位角(起点, 终点):
    """x 轴正向逆时针转到「起点->终点」向量的角度，单位度，范围 [0, 360)。"""
    d = np.asarray(终点, dtype=float) - np.asarray(起点, dtype=float)
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def 圆周布点(n, 半径, 起始角=0.0, 圆心=(0.0, 0.0)):
    """在以圆心为中心的圆周上均布 n 个点。"""
    角 = np.radians(起始角 + np.arange(n) * 360.0 / n)
    return np.stack([圆心[0] + 半径 * np.cos(角),
                     圆心[1] + 半径 * np.sin(角)], axis=1)


def 生成误差(n, 模式, rng):
    """误差绑定在检测点上，一经生成对该点固定不变（附录 2：重复检测不改变误差）。"""
    if 模式 == "随机":
        return rng.uniform(-DELTA, DELTA, size=n)
    if 模式 == "极端交替":
        return np.array([DELTA if i % 2 == 0 else -DELTA for i in range(n)])
    if 模式 == "无误差":
        return np.zeros(n)
    raise ValueError(f"未知的误差模式：{模式}")


def 生成算例(名称, 干扰源, 检测点, 误差模式, rng, 说明="", 模拟器舍入=False):
    """构造一组算例，返回 (检测点表, 元信息)。

    模拟器舍入=False：示向度保留全精度，算法侧楔形半角取 DELTA=1°（问题一）。
    模拟器舍入=True ：示向度截成两位小数，复刻模拟器返回格式，
                      算法侧楔形半角须取 DELTA+舍入裕度=1.005°（问题三、四）。
    """
    干扰源 = np.asarray(干扰源, dtype=float)
    检测点 = np.asarray(检测点, dtype=float)
    n = len(检测点)

    距离 = np.linalg.norm(检测点 - 干扰源, axis=1)

    # 题面对输入的两条硬性要求，不满足则该算例根本不合法：
    # ① 附录 2 第 9 条：距离不超过 5 米时信号过强，测不出示向度，此时无输入可言。
    if (距离 <= R_NEAR + 1e-9).any():
        raise ValueError(f"算例 {名称}：存在检测点距干扰源不超过 {R_NEAR} m，"
                         f"按附录 2 第 9 条无法获得示向度")
    # ② 附录 2：误差由地点决定，同一地点重复检测误差不变。
    #    故两个检测点若重合，其误差必须相同，独立抽样即与题面矛盾。
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(检测点[i] - 检测点[j]) <= 1e-6:
                raise ValueError(f"算例 {名称}：检测点 {i+1} 与 {j+1} 重合。"
                                 f"同一地点误差相同，重合点不能独立抽样误差")

    真实方位角 = np.array([方位角(S, 干扰源) for S in 检测点])
    误差 = 生成误差(n, 误差模式, rng)
    示向度 = (真实方位角 + 误差) % 360.0
    if 模拟器舍入:
        示向度 = np.round(示向度, SVD_DECIMALS)
    应用半角 = DELTA + (舍入裕度 if 模拟器舍入 else 0.0)

    # 校验：真实方位角必须落在示向度 ±应用半角 的楔形内，否则楔形交不含真值。
    偏差 = (真实方位角 - 示向度 + 180.0) % 360.0 - 180.0
    楔形含真值 = np.abs(偏差) <= 应用半角 + 1e-9

    表 = pd.DataFrame({
        "序号": np.arange(1, n + 1),
        "x": 检测点[:, 0],
        "y": 检测点[:, 1],
        "距离": 距离,
        "真实方位角": 真实方位角,
        "误差": 误差,
        "示向度": 示向度,
        "楔形含真值": 楔形含真值,
        "保证可收到": 距离 <= R_RECV_MIN + 1e-6,
        "可能收不到": 距离 > R_RECV_MAX + 1e-6,
        # 附件 2 §1.1：机器狗可以提交目标区域外的位置，故本列仅供参考，不是合法性要求
        "检测点在区域内": np.linalg.norm(检测点, axis=1) <= R_AREA + 1e-6,
    })

    元 = {
        "算例": 名称,
        "n": n,
        "干扰源x": 干扰源[0],
        "干扰源y": 干扰源[1],
        "干扰源距圆心": float(np.linalg.norm(干扰源)),
        "误差模式": 误差模式,
        "模拟器舍入": 模拟器舍入,
        "算法应用半角": 应用半角,
        "最近检测点距离": float(距离.min()),
        "最远检测点距离": float(距离.max()),
        "全部保证可收到": bool((距离 <= R_RECV_MIN + 1e-6).all()),
        "全部楔形含真值": bool(楔形含真值.all()),
        "说明": 说明,
    }
    return 表, 元


def 构造全部算例():
    rng = np.random.default_rng(SEED)
    算例 = []

    # 算例 A：附录 2 图 2 的标准两点交会，交会角 90°，各距干扰源 1000 m。
    #         其随机误差恰好使定位区域成为 n=2 的覆盖圆反例。
    G = np.array([600.0, 300.0])
    S = np.array([G + 1000.0 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
                  for a in (150.0, 240.0)])
    算例.append(生成算例("A", G, S, "随机", rng,
                        "两点交会，交会角 90°，对应附录 2 图 2 的四边形"))

    # 算例 B：三点在圆周上互成 120°、干扰源位于圆心、误差取零，
    #         定位区域为关于 G 对称的正六边形。这是「以直径为直径的圆盖不住
    #         定位区域」最干净的反例，用作问题一第二小问的论文插图。
    算例.append(生成算例("B", (0.0, 0.0), 圆周布点(3, 900.0, 起始角=90.0), "无误差", rng,
                        "三点均布 120° 且零误差，定位区域为正六边形，覆盖圆反例"))

    # 算例 C：五个检测点随机布置，检验算法对一般 n 的适用性。
    G = np.array([-500.0, 800.0])
    半径 = rng.uniform(400.0, 950.0, size=5)
    角 = rng.uniform(0.0, 360.0, size=5)
    S = G + np.stack([半径 * np.cos(np.radians(角)),
                      半径 * np.sin(np.radians(角))], axis=1)
    算例.append(生成算例("C", G, S, "随机", rng,
                        "五个检测点随机布置，一般 n 的情形"))

    return 算例


def main():
    输出目录.mkdir(parents=True, exist_ok=True)
    全部 = 构造全部算例()
    汇总 = []
    for 表, 元 in 全部:
        名 = 元["算例"]
        表.to_csv(输出目录 / f"算例{名}_检测点.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"x": 元["干扰源x"], "y": 元["干扰源y"]}]).to_csv(
            输出目录 / f"算例{名}_干扰源.csv", index=False, encoding="utf-8-sig")
        汇总.append(元)

    汇总表 = pd.DataFrame(汇总)
    汇总表.to_csv(输出目录 / "算例汇总.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print("算例汇总")
    print(汇总表[["算例", "n", "干扰源x", "干扰源y", "误差模式",
                  "模拟器舍入", "算法应用半角",
                  "全部保证可收到", "全部楔形含真值"]].to_string(index=False))
    print()
    for 表, 元 in 全部:
        print(f"--- 算例 {元['算例']}：{元['说明']}")
        print(表[["序号", "x", "y", "距离", "真实方位角", "误差", "示向度",
                  "楔形含真值"]].round(3).to_string(index=False))
        print()
    print(f"CSV 已写入 {输出目录}")


if __name__ == "__main__":
    main()
