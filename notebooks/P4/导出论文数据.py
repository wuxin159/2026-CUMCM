"""问题四 · 把定稿数据导出到 results/，供论文的 PGFPlots 直接读取。

沿用 P2、P3 的做法：**论文只读 `results/问题四_*.csv`，不读 notebooks 下的中间文件**。
这样一来，notebooks 里的探索产物可以随便改名、重算，而论文引用的数据是一份
显式挑选并定名的快照；换一台机器 clone 下来，`paper/main.tex` 也能直接编译。

命名一律 `问题四_xxx.csv`，与 README「命名与作图」一节的口径一致。

输出（写到仓库根的 results/）：
    问题四_判空代价.csv       圆盘覆盖 vs 半圆盘围堵的判空代价随空频道数的变化
    问题四_围堵进度.csv       前 k 个锚点能围住的面积占比（高度非线性）
    问题四_锚点布局.csv       24 点围堵构造与巡回访问序
    问题四_策略对照.csv       主表
    问题四_消融.csv           消融实验
    问题四_时间去向.csv       四分项
    问题四_算例A_轨迹.csv     单局全景：机器狗轨迹
    问题四_算例A_干扰源.csv   单局全景：干扰源（含定向朝向）
    问题四_算例A_停靠.csv     单局全景：各类停靠点
    问题四_进度曲线.csv       清除进度：推荐策略 vs 两个基线
    问题四_把握校准.csv       「收信把握」的校准曲线
    问题四_引理检验.csv       11 条引理的扫描结果
    问题四_正式测试.csv       三次正式测试的复盘汇总
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results"

# 直接改名搬运的（源文件 -> 目标名）
搬运 = {
    "图_判据对照.csv": "问题四_判空代价.csv",
    "图_锚点布局.csv": "问题四_锚点布局.csv",
    "图_算例A_轨迹.csv": "问题四_算例A_轨迹.csv",
    "图_算例A_干扰源.csv": "问题四_算例A_干扰源.csv",
    "图_算例A_停靠.csv": "问题四_算例A_停靠.csv",
    "图_进度曲线.csv": "问题四_进度曲线.csv",
    "图_时间去向.csv": "问题四_时间去向.csv",
    "图_把握校准.csv": "问题四_把握校准.csv",
    "策略对照汇总.csv": "问题四_策略对照.csv",
    "消融汇总.csv": "问题四_消融.csv",
    "引理检验.csv": "问题四_引理检验.csv",
    "实战复盘汇总.csv": "问题四_正式测试.csv",
}


def main():
    OUT.mkdir(exist_ok=True)
    print("问题四 · 导出论文数据")
    缺 = []
    for 源, 目标 in 搬运.items():
        p = HERE / 源
        if not p.exists():
            缺.append(源)
            continue
        shutil.copyfile(p, OUT / 目标)
        n = len(pd.read_csv(p, encoding="utf-8-sig"))
        print(f"    {目标:<28} {n:>5} 行   <- {源}")

    # 围堵进度：由锚点布局现算（论文图要用，判空几何.py 里只打印不落盘）
    sys.path.insert(0, str(HERE))
    import 公共 as C                      # noqa: E402
    import 策略 as P                      # noqa: E402
    import 判空几何 as G                  # noqa: E402
    构造 = list(P.围堵锚点构造)
    顺序, 巡回 = C.开放TSP((0.0, 0.0), 构造)
    序访问 = [构造[i] for i in 顺序]
    X = G.目标格点(20.0)
    单元 = 20.0 * 20.0
    行 = []
    for k in range(3, len(构造) + 1):
        r = G.围堵半径(序访问[:k], X)
        已 = r <= C.R_RECV_MIN
        行.append({"锚点数": k, "已围堵面积占比": float(已.mean()),
                   "已围堵面积_平方米": float(已.sum()) * 单元})
    df = pd.DataFrame(行)
    df.to_csv(OUT / "问题四_围堵进度.csv", index=False, encoding="utf-8-sig")
    print(f"    {'问题四_围堵进度.csv':<28} {len(df):>5} 行   <- 现算")

    # —— 按类别拆表：PGFPlots 不便按字符串列筛行，直接拆成几张小表 ——
    停 = OUT / "问题四_算例A_停靠.csv"
    if 停.exists():
        d = pd.read_csv(停, encoding="utf-8-sig")
        for 类, 名 in (("锚点", "锚点停靠"), ("扫描", "顺路扫描"),
                       ("补测", "补测停靠"), ("清除", "清除点")):
            子 = d[d["类型"] == 类][["x", "y", "虚拟时刻"]]
            子.to_csv(OUT / f"问题四_算例A_{名}.csv", index=False,
                      encoding="utf-8-sig")
            print(f"    {'问题四_算例A_' + 名 + '.csv':<28} {len(子):>5} 行   <- 拆表")

    代 = OUT / "问题四_判空代价.csv"
    if 代.exists():
        d = pd.read_csv(代, encoding="utf-8-sig")
        for 方, 名 in (("问题三：圆盘覆盖", "圆盘"), ("问题四：半圆盘围堵", "围堵")):
            子 = d[d["方案"] == 方][["空频道数", "判空检测秒", "锚点行进秒", "合计秒"]]
            子.to_csv(OUT / f"问题四_判空代价_{名}.csv", index=False,
                      encoding="utf-8-sig")
            print(f"    {'问题四_判空代价_' + 名 + '.csv':<28} {len(子):>5} 行   <- 拆表")

    源 = OUT / "问题四_算例A_干扰源.csv"
    if 源.exists():
        d = pd.read_csv(源, encoding="utf-8-sig")
        全 = d[~d["是定向"]][["频道", "x", "y"]]
        全.to_csv(OUT / "问题四_算例A_源全向.csv", index=False, encoding="utf-8-sig")
        定 = d[d["是定向"]].copy()
        # 给 quiver 用：箭头终点 = 源点 + 260 m x 定向方向
        定["dx"] = 260.0 * np.cos(np.radians(定["定向方向"]))
        定["dy"] = 260.0 * np.sin(np.radians(定["定向方向"]))
        定[["频道", "x", "y", "定向方向", "dx", "dy"]].to_csv(
            OUT / "问题四_算例A_源定向.csv", index=False, encoding="utf-8-sig")
        print(f"    {'问题四_算例A_源全向.csv':<28} {len(全):>5} 行   <- 拆表")
        print(f"    {'问题四_算例A_源定向.csv':<28} {len(定):>5} 行   <- 拆表")

    进 = OUT / "问题四_进度曲线.csv"
    if 进.exists():
        d = pd.read_csv(进, encoding="utf-8-sig")
        for 策, 名 in (("推荐策略", "推荐"), ("基线一：先扫后清", "先扫后清"),
                       ("基线二：逐源处理", "逐源")):
            子 = d[d["策略"] == 策][["虚拟时刻", "已清除数"]]
            子.to_csv(OUT / f"问题四_进度_{名}.csv", index=False,
                      encoding="utf-8-sig")
            print(f"    {'问题四_进度_' + 名 + '.csv':<28} {len(子):>5} 行   <- 拆表")

    if 缺:
        print("\n[警告] 下列源文件不存在，对应的论文数据未导出：")
        for s in 缺:
            print(f"    {s}")
        print("    请先运行 作图数据.py / 策略对照.py / 命题与引理.py / 复盘实战日志.py")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
