"""问题一 · 第 4 步：求解绘图（只读 CSV 作图，不算数）。

画的是「解长什么样」：在算例绘图的楔形线之上，叠加求解得到的定位区域、
直径 AB、以直径为直径的圆、最小覆盖圆，并标出落在圆外的顶点（若有）。

依赖：生成算例.py 与 求解算法.py 的输出
输出：求解图.pdf   仅供自己看，论文用图后续交给 PGFPlots
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from 算例绘图 import 配置中文字体, 楔形色, 干扰源色, R_AREA

HERE = Path(__file__).resolve().parent
输出目录 = HERE

定位区域色 = "#1f77b4"    # 蓝：定位区域
直径圆色 = "#d62728"      # 红：以直径为直径的圆、直径 AB
最小圆色 = "#2ca02c"      # 绿：最小覆盖圆
圆外顶点色 = "#ff7f0e"    # 橙：落在圆外的顶点


def 画一组(pdf, 元, 检测表, G, 顶点, 解):
    S = 检测表[["x", "y"]].to_numpy()
    示向度 = 检测表["示向度"].to_numpy()
    delta = float(元["算法应用半角"])

    顶 = np.asarray(顶点, dtype=float)
    中心 = 顶.mean(axis=0)
    A = np.array([解["Ax"],解["Ay"]])
    B = np.array([解["Bx"],解["By"]])
    直径 = float(解["直径"])
    圆心_d = (A + B) / 2.0
    半径_d = 直径 / 2.0
    最小圆心 = np.array([解["最小覆盖圆x"], 解["最小覆盖圆y"]])
    最小半径 = float(解["最小覆盖圆半径"])
    能覆盖 = bool(解["能否覆盖"])
    超出 = float(解["最大超出量"])

    半宽 = max(直径 * 0.95, 最小半径 * 1.9,
               float(np.hypot(*(np.asarray(G) - 中心))) * 1.5) * 1.15

    # 楔形线会横贯整幅右图，任何角落都不空，故把图例放到画布之外
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.9),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- 左：全局布置
    ax1.add_patch(plt.Circle((0, 0), R_AREA, fill=False, lw=1.2, ls="--", color="0.5"))
    for i, (s, th) in enumerate(zip(S, 示向度), start=1):
        色 = 楔形色[(i - 1) % len(楔形色)]
        r = float(np.hypot(*(中心 - s)))
        终 = s + r * 1.15 * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
        ax1.plot([s[0], 终[0]], [s[1], 终[1]], lw=1.1, color=色, alpha=0.9)
        ax1.plot(s[0], s[1], "o", color=色, ms=7, zorder=4)
        ax1.annotate(f"$S_{{{i}}}$", s, textcoords="offset points", xytext=(8, 8), color=色)
    ax1.plot(G[0], G[1], "o", color=干扰源色, ms=9, zorder=5)
    ax1.annotate("$G$", G, textcoords="offset points", xytext=(9, 9), color=干扰源色)
    ax1.set_aspect("equal")
    ax1.set_xlim(-R_AREA * 1.12, R_AREA * 1.12)
    ax1.set_ylim(-R_AREA * 1.12, R_AREA * 1.12)
    ax1.set_xlabel("$x$ / 米")
    ax1.set_ylabel("$y$ / 米")
    ax1.set_title(f"算例 {元['算例']}：全局布置（$n={元['n']}$）")
    ax1.grid(alpha=0.25)

    # ---- 右：先铺楔形边界线，它们正是切出这个多边形的那些线
    for i, (s, th) in enumerate(zip(S, 示向度), start=1):
        色 = 楔形色[(i - 1) % len(楔形色)]
        for 偏, 样式, 宽, 透 in ((-delta, "--", 0.9, 0.75),
                                (0.0, "-", 1.1, 0.9),
                                (delta, "--", 0.9, 0.75)):
            u = np.array([np.cos(np.radians(th + 偏)), np.sin(np.radians(th + 偏))])
            基 = s + np.dot(中心 - s, u) * u
            端 = 基 + np.outer(np.array([-半宽 * 3, 半宽 * 3]), u)
            标 = (f"$S_{{{i}}}$ 的示向线与 $\\pm{delta:g}^\\circ$ 边界") if 偏 == 0.0 else None
            ax2.plot(端[:, 0], 端[:, 1], 样式, lw=宽, color=色, alpha=透, zorder=1, label=标)

    环 = np.vstack([顶, 顶[:1]])
    ax2.fill(环[:, 0], 环[:, 1], color=定位区域色, alpha=0.18, zorder=2)
    ax2.plot(环[:, 0], 环[:, 1], "-", color=定位区域色, lw=2.2, zorder=4,
             label=f"定位区域（{len(顶)} 个顶点）")
    ax2.plot(顶[:, 0], 顶[:, 1], "o", color=定位区域色, ms=5, zorder=5)

    ax2.add_patch(plt.Circle(圆心_d, 半径_d, fill=False, lw=1.8, ls="-", color=直径圆色,
                             zorder=3, label=f"以直径为直径的圆（$r={半径_d:.2f}$ m）"))
    ax2.add_patch(plt.Circle(最小圆心, 最小半径, fill=False, lw=1.6, ls=":", color=最小圆色,
                             zorder=3, label=f"最小覆盖圆（$r={最小半径:.2f}$ m）"))
    ax2.plot([A[0], B[0]], [A[1], B[1]], "-", color=直径圆色, lw=1.8, zorder=6,
             label=f"直径 $AB={直径:.3f}$ m")
    ax2.plot([A[0], B[0]], [A[1], B[1]], "o", color=直径圆色, ms=7, zorder=7)
    ax2.annotate("$A$", A, textcoords="offset points", xytext=(7, 7), color=直径圆色)
    ax2.annotate("$B$", B, textcoords="offset points", xytext=(7, 7), color=直径圆色)

    if not 能覆盖:
        圆外 = (float(解["圆外顶点x"]), float(解["圆外顶点y"]))
        ax2.plot(圆外[0], 圆外[1], "o", color=圆外顶点色, ms=11, zorder=8,
                 markeredgecolor="#7f3f00", markeredgewidth=1.2,
                 label=f"落在圆外的顶点（超出 {超出:.3f} m）")

    ax2.plot(G[0], G[1], "o", color=干扰源色, ms=9, zorder=9,
             markeredgecolor="white", markeredgewidth=1.0, label="干扰源真实位置 $G$")

    if float(np.hypot(*中心)) + 半宽 > R_AREA:
        角 = np.linspace(0, 2 * np.pi, 4000)
        ax2.plot(R_AREA * np.cos(角), R_AREA * np.sin(角), ls="--", lw=1.2,
                 color="0.45", zorder=1, label="目标区域边界")

    ax2.set_aspect("equal")
    ax2.set_xlim(中心[0] - 半宽, 中心[0] + 半宽)
    ax2.set_ylim(中心[1] - 半宽, 中心[1] + 半宽)
    ax2.set_xlabel("$x$ / 米")
    ax2.set_ylabel("$y$ / 米")
    结论 = "圆能覆盖" if 能覆盖 else "圆不能覆盖"
    ax2.set_title(f"算例 {元['算例']}：定位区域，直径 $d={直径:.3f}$ m —— {结论}")
    ax2.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=8,
               framealpha=0.95, borderaxespad=0.0)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    配置中文字体()
    算例汇总 = pd.read_csv(输出目录 / "算例汇总.csv", encoding="utf-8-sig")
    求解汇总 = pd.read_csv(输出目录 / "求解汇总.csv", encoding="utf-8-sig")

    with PdfPages(输出目录 / "求解图.pdf") as pdf:
        for _, 元 in 算例汇总.iterrows():
            名 = str(元["算例"])
            匹配 = 求解汇总[求解汇总["算例"] == 名]
            if 匹配.empty or 匹配.iloc[0]["状态"] not in ("多边形", "线段", "单点"):
                print(f"算例 {名}：定位区域为「{匹配.iloc[0]['状态'] if not 匹配.empty else '缺失'}」，无有限直径，跳过")
                continue
            解 = 匹配.iloc[0]
            检测表 = pd.read_csv(输出目录 / f"算例{名}_检测点.csv", encoding="utf-8-sig")
            源表 = pd.read_csv(输出目录 / f"算例{名}_干扰源.csv", encoding="utf-8-sig")
            顶点表 = pd.read_csv(输出目录 / f"算例{名}_定位区域顶点.csv", encoding="utf-8-sig")
            G = (float(源表["x"].iloc[0]), float(源表["y"].iloc[0]))
            画一组(pdf, 元, 检测表, G, 顶点表[["x", "y"]].to_numpy(), 解)
            print(f"算例 {名} 已绘制")
    print(f"求解图已写入 {输出目录 / '求解图.pdf'}")


if __name__ == "__main__":
    main()
