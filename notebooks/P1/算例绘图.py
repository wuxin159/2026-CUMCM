"""问题一 · 第 2 步：算例绘图（只读 CSV 作图，不算数）。

画的是「输入长什么样」：检测点的全局布置，以及干扰源附近各条示向线与
±δ 楔形边界的交会情况。此时尚未求解定位区域，故右图只有线没有多边形。

依赖：生成算例.py 的输出
输出：算例预览.pdf   仅供自己看，论文用图后续交给 PGFPlots
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
输出目录 = HERE

R_AREA = 1800.0          # 目标区域半径，米

# 各检测点楔形线的配色。避开求解绘图中正文元素占用的
# 红（以直径为直径的圆）、绿（最小覆盖圆）、蓝（定位区域）、橙（圆外顶点）。
楔形色 = ["#7f7f7f", "#8c564b", "#9467bd", "#17becf", "#bcbd22",
          "#4b4b4b", "#b5651d", "#5d3fd3", "#008b8b", "#808000"]

干扰源色 = "#d62728"


def 配置中文字体():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
    })


def 两线交点(S1, th1, S2, th2):
    """两条示向线（中心线）的交点，近乎平行时返回 None。"""
    u1 = np.array([np.cos(np.radians(th1)), np.sin(np.radians(th1))])
    u2 = np.array([np.cos(np.radians(th2)), np.sin(np.radians(th2))])
    叉 = u1[0] * u2[1] - u1[1] * u2[0]
    if abs(叉) < 1e-6:
        return None
    d = np.asarray(S2, dtype=float) - np.asarray(S1, dtype=float)
    t1 = (d[0] * u2[1] - d[1] * u2[0]) / 叉
    return np.asarray(S1, dtype=float) + t1 * u1


def 交会中心(S, 示向度):
    """所有示向线两两交点的中位数，作为定位区域的粗略中心。

    取中位数是为了抗近平行线对产生的野值；不能直接用干扰源真实位置 G 取景，
    因为误差会把交会区域推离 G，那样会把区域切掉一角。
    """
    交点 = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            p = 两线交点(S[i], 示向度[i], S[j], 示向度[j])
            if p is not None:
                交点.append(p)
    if not 交点:
        return None
    return np.median(np.array(交点), axis=0)


def 画一组(pdf, 表, 元):
    G = np.array([元["干扰源x"], 元["干扰源y"]])
    S = 表[["x", "y"]].to_numpy()
    示向度 = 表["示向度"].to_numpy()
    距离 = 表["距离"].to_numpy()
    delta = float(元["算法应用半角"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.9),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- 左：全局布置
    ax1.add_patch(plt.Circle((0, 0), R_AREA, fill=False, lw=1.2, ls="--", color="0.5"))
    for i, (s, th, r) in enumerate(zip(S, 示向度, 距离), start=1):
        色 = 楔形色[(i - 1) % len(楔形色)]
        终 = s + (r * 1.2) * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
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
    ax1.set_title(f"算例 {元['算例']}：全局布置（$n={元['n']}$，误差{元['误差模式']}）")
    ax1.grid(alpha=0.25)

    # ---- 右：交会区域附近放大
    中心 = 交会中心(S, 示向度)
    if 中心 is None:
        中心 = G
    半宽 = max(距离.max() * np.tan(np.radians(delta)) * 2.5,
               float(np.linalg.norm(中心 - G)) * 1.8,
               25.0)
    for i, (s, th) in enumerate(zip(S, 示向度), start=1):
        色 = 楔形色[(i - 1) % len(楔形色)]
        for 偏, 样式, 宽, 透 in ((-delta, "--", 0.9, 0.75),
                                (0.0, "-", 1.2, 0.95),
                                (delta, "--", 0.9, 0.75)):
            u = np.array([np.cos(np.radians(th + 偏)), np.sin(np.radians(th + 偏))])
            基 = s + np.dot(中心 - s, u) * u
            端 = 基 + np.outer(np.array([-半宽 * 3, 半宽 * 3]), u)
            标 = (f"$S_{{{i}}}$ 的示向线与 $\\pm{delta:g}^\\circ$ 边界") if 偏 == 0.0 else None
            ax2.plot(端[:, 0], 端[:, 1], 样式, lw=宽, color=色, alpha=透, zorder=2, label=标)

    ax2.plot(G[0], G[1], "o", color=干扰源色, ms=9, zorder=5,
             markeredgecolor="white", markeredgewidth=1.0, label="干扰源真实位置 $G$")

    if float(np.linalg.norm(中心)) + 半宽 * 1.2 > R_AREA:
        角 = np.linspace(0, 2 * np.pi, 4000)
        ax2.plot(R_AREA * np.cos(角), R_AREA * np.sin(角), ls="--", lw=1.2,
                 color="0.45", zorder=1, label="目标区域边界")

    ax2.set_aspect("equal")
    ax2.set_xlim(中心[0] - 半宽, 中心[0] + 半宽)
    ax2.set_ylim(中心[1] - 半宽, 中心[1] + 半宽)
    ax2.set_xlabel("$x$ / 米")
    ax2.set_ylabel("$y$ / 米")
    ax2.set_title(f"算例 {元['算例']}：干扰源附近放大（实线为示向度，虚线为 $\\pm{delta:g}^\\circ$ 边界）")
    ax2.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=8,
               framealpha=0.95, borderaxespad=0.0)
    ax2.grid(alpha=0.25)

    fig.suptitle(元["说明"], y=0.02, fontsize=10, color="0.3")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    配置中文字体()
    汇总表 = pd.read_csv(输出目录 / "算例汇总.csv", encoding="utf-8-sig")
    with PdfPages(输出目录 / "算例预览.pdf") as pdf:
        for _, 元 in 汇总表.iterrows():
            表 = pd.read_csv(输出目录 / f"算例{元['算例']}_检测点.csv", encoding="utf-8-sig")
            画一组(pdf, 表, 元)
            print(f"算例 {元['算例']} 已绘制")
    print(f"预览图已写入 {输出目录 / '算例预览.pdf'}")


if __name__ == "__main__":
    main()
