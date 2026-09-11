"""问题三 · 把论文要用的数据导出到 results/（只复制与筛列，不做任何计算）。"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
根 = HERE.parents[1]
结果目录 = 根 / "results"

# (本目录文件, results/ 新名, 要保留的列或 None 表示全留)
清单 = [
    # ---- 图 A：判空所需的等圆覆盖半径 ----
    ("覆盖数_最优半径.csv", "问题三_覆盖数.csv",
     ["锚点数n", "本脚本最优半径比", "文献最优半径比", "最优覆盖半径_米"]),
    # ---- 图 B：单局全景（算例 A） ----
    ("图_算例A_轨迹.csv", "问题三_算例A_轨迹.csv",
     ["序号", "动作", "x", "y", "频道", "结果", "虚拟时刻"]),
    ("图_算例A_停靠.csv", "问题三_算例A_停靠.csv",
     ["x", "y", "检测数", "清除数", "到达时刻", "类型"]),
    ("图_算例A_干扰源.csv", "问题三_算例A_干扰源.csv", ["频道", "x", "y"]),
    # ---- 图 C：定位区域的收缩 ----
    ("图_收缩曲线.csv", "问题三_收缩曲线.csv",
     ["频道", "第几条有效读数", "结果", "直径", "Rmin"]),
    # ---- 图 D：清除进度曲线 ----
    ("图_进度曲线.csv", "问题三_进度曲线.csv", ["算例", "策略", "虚拟时刻", "已清除"]),
    # ---- 表：策略对照、消融、分组、引理 ----
    ("策略对照汇总.csv", "问题三_策略对照.csv", None),
    ("消融汇总.csv", "问题三_消融.csv", None),
    ("分组统计.csv", "问题三_分组统计.csv", None),
    ("引理检验.csv", "问题三_引理检验.csv", None),
    # ---- 表：正式测试与赛后复盘 ----
    ("实战复盘汇总.csv", "问题三_正式测试复盘.csv", None),
    # ---- 备选图：锚点布局、时间去向 ----
    ("锚点布局.csv", "问题三_锚点布局.csv", None),
    ("图_时间去向.csv", "问题三_时间去向.csv", None),
]


def 拆分(预览):
    """PGFPlots 按列值筛行不方便，这里按「策略」「动作」预先拆成独立文件。

    仍然只是筛行与改名，不做任何计算。
    """
    出 = []
    进度 = pd.read_csv(HERE / "图_进度曲线.csv", encoding="utf-8-sig")
    for 名策, 文件 in [("推荐策略", "问题三_进度_推荐.csv"),
                       ("先扫后清", "问题三_进度_先扫后清.csv"),
                       ("逐源处理", "问题三_进度_逐源.csv")]:
        子 = 进度[(进度["算例"] == "A") & (进度["策略"] == 名策)]
        出.append((文件, 子[["虚拟时刻", "已清除"]].sort_values("虚拟时刻")))
    轨 = pd.read_csv(HERE / "图_算例A_轨迹.csv", encoding="utf-8-sig")
    清 = 轨[(轨["动作"] == "clear") & (轨["结果"] == "success")]
    出.append(("问题三_算例A_清除点.csv", 清[["x", "y", "频道", "虚拟时刻"]]))
    收 = pd.read_csv(HERE / "图_收缩曲线.csv", encoding="utf-8-sig")
    收 = 收[(收["第几条有效读数"] >= 1) & 收["Rmin"].notna()]
    出.append(("问题三_收缩散点.csv", 收[["频道", "第几条有效读数", "Rmin"]]))
    for 文件, df in 出:
        print(f"  {'（拆分）':<26} -> results/{文件:<28} {len(df):>4} 行")
        if not 预览:
            结果目录.mkdir(exist_ok=True)
            df.to_csv(结果目录 / 文件, index=False, encoding="utf-8-sig")


def main(argv):
    预览 = "--预览" in argv
    print(f"源目录  {HERE}")
    print(f"目标目录 {结果目录}{'（预览，不写盘）' if 预览 else ''}\n")
    缺, 写 = [], 0
    for 源, 新, 列 in 清单:
        p = HERE / 源
        if not p.exists():
            缺.append(源)
            print(f"  [缺] {源}")
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        if 列:
            少 = [c for c in 列 if c not in df.columns]
            if 少:
                print(f"  [列缺] {源}：{少}，改为全列导出")
            else:
                df = df[列]
        print(f"  {源:<26} -> results/{新:<28} {len(df):>4} 行 x {df.shape[1]} 列")
        if not 预览:
            结果目录.mkdir(exist_ok=True)
            df.to_csv(结果目录 / 新, index=False, encoding="utf-8-sig")
            写 += 1
    拆分(预览)
    print(f"\n{'预览完毕' if 预览 else f'已写入 {写} 个文件与 5 个拆分文件'}；"
          f"缺失 {len(缺)} 个：{缺}")
    if 缺:
        print("缺失的多半是某个分析脚本还没跑，按 重跑全部.py 的顺序补跑即可。")
    return 1 if 缺 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
