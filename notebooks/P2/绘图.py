"""问题二 · 图（只读 CSV 作图，不算数）。

每张图单独输出一个矢量 PDF，便于 LaTeX 直接 \\includegraphics。
不输出位图。字体按 TrueType 嵌入（pdf.fonttype=42），避免 Type 3 在期刊/打印环节出问题。

按分析的四个阶段组织：

    阶段 1  数学分析   图 1 可行集      图 2 交会几何      图 3 直径随源距的变化
    阶段 2  时间加权   图 4 两个准则的等值面（主图）  图 5 效率前沿  图 6 候选区域
    阶段 3  代价换算   图 7 边际分析    图 8 途中加测
    阶段 4  策略仿真   图 9 参数面与单例过程         图 10 各策略对比

论文成图仍推荐交给 PGFPlots 读同一批 CSV；本脚本的 PDF 作为矢量备选与校对用图。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as 多边形块, Circle as 圆块, Rectangle as 矩形块

HERE = Path(__file__).resolve().parent
图目录 = HERE / "图"
sys.path.insert(0, str(HERE.parents[0] / "P1"))
import 算例绘图 as P1绘图                        # noqa: E402  借配色与中文字体设置

楔形色 = P1绘图.楔形色
干扰源色 = P1绘图.干扰源色
区域色 = "#1f77b4"
主色 = "#d62728"          # 时间准则
次色 = "#7f7f7f"          # 几何准则
约束色 = "#2ca02c"        # 接收条件

R_AREA, R_RECV_MIN, R_RECV_MAX, R_CLEAR = 1800.0, 1000.0, 1500.0, 20.0
时间框 = dict(a低=150.0, a高=450.0, b低=75.0, b高=180.0)     # 策略对照结果.txt §三
几何框 = dict(a低=750.0, a高=1050.0, b低=400.0, b高=835.0)   # 候选区域结果.txt §三


def 读(名):
    return pd.read_csv(HERE / 名, encoding="utf-8-sig")


def 存(fig, 名):
    fig.savefig(图目录 / f"{名}.pdf", bbox_inches="tight")
    plt.close(fig)
    return 名


# ================================================================ 阶段 1
def 图1_可行集():
    Ω = 读("图_先验区域.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].add_patch(圆块((0, 0), R_AREA, fill=False, ls="--", color="0.5",
                         label="目标区域边界（$R=1800$ m）"))
    ax[0].add_patch(多边形块(Ω[["x", "y"]].to_numpy(), closed=True,
                             facecolor=区域色, alpha=0.5, edgecolor=区域色,
                             label="可行集 $\\Omega_1$"))
    ax[0].plot(0, 0, "k^", ms=9, label="检测点 $S_1$")
    ax[0].set_xlim(-2000, 2000); ax[0].set_ylim(-2000, 2000)
    ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3)
    ax[0].set_title("(a) 可行集在目标区域中的位置")
    ax[0].set_xlabel("沿示向线方向 (m)"); ax[0].set_ylabel("横向 (m)")
    ax[0].legend(fontsize=8, loc="upper left")

    ax[1].add_patch(多边形块(Ω[["x", "y"]].to_numpy(), closed=True,
                             facecolor=区域色, alpha=0.5, edgecolor=区域色))
    ax[1].set_xlim(-50, 1560); ax[1].set_ylim(-58, 58); ax[1].grid(alpha=0.3)
    ax[1].set_title("(b) 横向放大 14 倍：纵长 1495 m，最大横宽 52 m")
    ax[1].set_xlabel("距离方向 (m)"); ax[1].set_ylabel("横向 (m)")
    for x in (500, 1000, 1500):
        ax[1].annotate(f"$\\pm{x*np.tan(np.radians(1)):.1f}$ m",
                       xy=(x, x*np.tan(np.radians(1))), xytext=(x-190, 44),
                       fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    fig.suptitle("图 1　单次检测后的可行集 $\\Omega_1$：方位不确定度 $\\pm1°$，距离不确定度 1495 m")
    fig.tight_layout()
    return 存(fig, "图1_可行集")


def 图2_交会几何():
    段 = 读("图_交会示意.csv"); 顶 = 读("图_定位区域顶点.csv")
    G = (顶["Gx"].iloc[0], 顶["Gy"].iloc[0]); S2 = (顶["S2x"].iloc[0], 顶["S2y"].iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, (名, 组) in enumerate(段.groupby("楔形")):
        for _, r in 组.iterrows():
            ax[0].plot([r["x1"], r["x2"]], [r["y1"], r["y2"]], color=楔形色[k],
                       lw=1.6 if r["线"] == "中线" else 0.8,
                       ls="-" if r["线"] == "中线" else "--",
                       label=f"${名[0]}_{名[1]}$ 示向线及 $\\pm\\delta$ 边界"
                             if r["线"] == "中线" else None)
    ax[0].plot(*G, "*", color=干扰源色, ms=14, label="干扰源 $G$")
    ax[0].plot(0, 0, "k^", ms=8); ax[0].plot(*S2, "ks", ms=7)
    for xy, xytext in [((S2[0], 0), (0, 0)), (S2, (S2[0], 0)), ((G[0], 0), (S2[0], 0))]:
        ax[0].annotate("", xy=xy, xytext=xytext,
                       arrowprops=dict(arrowstyle="<->", color="0.4"))
    ax[0].text(S2[0]/2, -62, "$a$", ha="center", fontsize=11, color="0.3")
    ax[0].text(S2[0]+20, S2[1]/2, "$b$", fontsize=11, color="0.3")
    ax[0].text((G[0]+S2[0])/2, -62, "$s=\\rho-a$", ha="center", fontsize=10, color="0.3")
    t = np.linspace(0, np.arctan2(S2[1], S2[0]), 40)
    ax[0].plot(180*np.cos(t), 180*np.sin(t), color="0.4", lw=0.8)
    ax[0].text(152, 52, "$\\alpha$", fontsize=11, color="0.3")
    ax[0].text(60, 300, "直角坐标 $(a,b)$：前进量与侧移量\n"
                        "极坐标 $(d,\\alpha)$：行进距离与偏角", fontsize=9,
               bbox=dict(fc="w", ec="0.7", alpha=0.9))
    ax[0].set_xlim(-80, 1150); ax[0].set_ylim(-110, 400); ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("(a) 交会几何与参数定义")

    P = 顶[["x", "y"]].to_numpy(); c = (顶["圆心x"].iloc[0], 顶["圆心y"].iloc[0])
    ax[1].add_patch(多边形块(P, closed=True, facecolor=区域色, alpha=0.5,
                             edgecolor=区域色, label="定位区域 $L$"))
    ax[1].add_patch(圆块(c, 顶["Rmin"].iloc[0], fill=False, color=约束色,
                         label=f"最小覆盖圆 $R_{{\\min}}={顶['Rmin'].iloc[0]:.1f}$ m"))
    ax[1].add_patch(圆块(c, R_CLEAR, fill=False, color=主色, ls="--",
                         label="清除半径 $R_c=20$ m"))
    ax[1].plot(*G, "*", color=干扰源色, ms=13, label="干扰源 $G$")
    半 = max(顶["直径"].iloc[0], 60)*0.75
    ax[1].set_xlim(c[0]-半, c[0]+半); ax[1].set_ylim(c[1]-半, c[1]+半)
    ax[1].set_aspect("equal"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_title(f"(b) 定位区域局部：直径 $D={顶['直径'].iloc[0]:.1f}$ m $>2R_c$")
    fig.suptitle("图 2　两次检测的交会定位：定位区域为两楔形之交")
    fig.tight_layout()
    return 存(fig, "图2_交会几何")


def 图3_直径随源距():
    t = 读("图_直径随距离.csv")
    名映 = {"推荐点(仿真最优)": "本文策略 $(450,120)$", "代理模型最优": "代理模型最优 $(450,145)$",
            "纯几何最优": "几何准则最优 $(900,580)$", "纯侧移": "纯横向偏移 $(0,750)$",
            "几乎不侧移": "近似径向 $(450,20)$"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, (名, 组) in enumerate(t.groupby("候选点", sort=False)):
        ax[0].plot(组["rho"], 组["直径"], color=楔形色[k % len(楔形色)],
                   label=名映.get(名, 名))
        ax[1].plot(组["rho"], 组["交会角"], color=楔形色[k % len(楔形色)])
    ax[0].axhline(2*R_CLEAR, color=主色, ls="--", lw=1.2,
                  label="一次清除的必要条件 $D\\leq 2R_c=40$ m")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("干扰源真实距离 $\\rho$ (m)")
    ax[0].set_ylabel("定位区域直径 $D$ (m)")
    ax[0].set_title("(a) 定位区域直径随源距的变化")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3)
    ax[1].axhline(90, color="0.4", ls=":", lw=1)
    ax[1].set_xlabel("干扰源真实距离 $\\rho$ (m)")
    ax[1].set_ylabel("交会角 $\\varphi$ (°)")
    ax[1].set_title("(b) 交会角 $\\varphi=\\arctan(b/s)$ 随源距的变化")
    ax[1].grid(alpha=0.3)
    fig.suptitle("图 3　任一固定第二检测点仅对部分源距区间有效，"
                 "故不存在对全部 $\\rho$ 一致最优的点")
    fig.tight_layout()
    return 存(fig, "图3_直径随源距")


# ================================================================ 阶段 2
def 图4_两个准则():
    网 = 读("候选区域网格.csv"); 透镜 = 读("图_透镜边界.csv")
    A = np.sort(网["a"].unique()); B = np.sort(网["b"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for k, (列, 题, 单位, 框, 色) in enumerate([
            ("D均值", "(a) 几何准则：期望定位区域直径 $E[D]$", "m", 几何框, 次色),
            ("时间均值", "(b) 时间准则：期望总清除时间 $E[T]$", "s", 时间框, 主色)]):
        Z = 网.pivot(index="b", columns="a", values=列).reindex(index=B, columns=A).to_numpy()
        最优 = 网.loc[网[列].idxmin()]
        cs = ax[k].contourf(A, B, Z, levels=28, cmap="viridis_r")
        ax[k].contour(A, B, Z, levels=[最优[列]*f for f in (1.05, 1.10, 1.20)],
                      colors="w", linewidths=[2.0, 1.1, 1.0])
        fig.colorbar(cs, ax=ax[k], label=f"{'$E[D]$ (m)' if k==0 else '$E[T]$ (s)'}",
                     pad=0.02)
        ax[k].add_patch(矩形块((框["a低"], 框["b低"]), 框["a高"]-框["a低"],
                               框["b高"]-框["b低"], fill=False, edgecolor=色,
                               lw=2.4, label="候选区域"))
        ax[k].plot(最优["a"], 最优["b"], "*", color=色, ms=17, mec="k", mew=0.7,
                   label=f"最优点 $({最优['a']:.0f},\\,{最优['b']:.0f})$")
        for 记, 组 in 透镜.groupby("来源"):
            心 = 5.0 if 记 == "近端圆" else 1000.0
            ang = np.degrees(np.arctan2(组["y"].to_numpy(), 组["x"].to_numpy()-心))
            序 = np.argsort(ang if 记 == "近端圆" else ang % 360.0)
            ax[k].plot(组["x"].to_numpy()[序], 组["y"].to_numpy()[序],
                       color=约束色, lw=1.6,
                       label="接收可靠域边界" if 记 == "近端圆" else None)
        ax[k].plot([0, 1500], [0, 0], "k--", lw=0.8); ax[k].plot(0, 0, "k^", ms=8)
        ax[k].set_xlim(A.min(), 1500); ax[k].set_ylim(0, 1100); ax[k].set_aspect("equal")
        ax[k].set_xlabel("前进量 $a$ (m)"); ax[k].set_ylabel("侧移量 $b$ (m)")
        ax[k].set_title(题 + "\n白色等值线：相对最优 $+5\\%/+10\\%/+20\\%$", fontsize=10.5)
        ax[k].legend(fontsize=8, loc="upper right")
    fig.suptitle("图 4　两种准则下第二检测点的目标函数等值面：最优点相距约 700 m", y=1.0)
    fig.tight_layout()
    return 存(fig, "图4_两个准则")


def 图5_效率前沿():
    f = 读("效率前沿.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(f["行进距离"], f["几何最优_ED"], "-o", ms=3, color=次色,
               label="给定行进距离下的 $\\min E[D]$")
    ax[0].axhline(2*R_CLEAR, color=主色, ls="--",
                  label="一次清除的必要条件 $D\\leq 40$ m")
    ax[0].axhline(f["几何最优_ED"].min(), color="0.3", ls=":",
                  label=f"全平面下确界 {f['几何最优_ED'].min():.1f} m")
    ax[0].annotate("下确界高于门限，\n增加行进距离无法跨越",
                   xy=(1050, 51), xytext=(430, 150), fontsize=9,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].set_yscale("log")
    ax[0].set_xlabel("至第二检测点的行进距离 $d$ (m)")
    ax[0].set_ylabel("期望定位区域直径 $E[D]$ (m)")
    ax[0].set_title("(a) 定位精度的效率前沿")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(f["行进距离"], f["几何最优_ET"], "-o", ms=3, color=次色,
               label="按几何准则选点")
    ax[1].plot(f["行进距离"], f["时间最优_ET"], "-s", ms=3, color=主色,
               label="按时间准则选点")
    k = f["时间最优_ET"].idxmin()
    ax[1].plot(f.loc[k, "行进距离"], f.loc[k, "时间最优_ET"], "*",
               color=主色, ms=18, mec="k", mew=0.6,
               label=f"极小值点 $d\\approx{f.loc[k,'行进距离']:.0f}$ m")
    ax[1].set_xlabel("至第二检测点的行进距离 $d$ (m)")
    ax[1].set_ylabel("期望总清除时间 $E[T]$ (s)")
    ax[1].set_title("(b) 总清除时间存在内部极小值")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("图 5　定位精度的边际收益递减，且其下确界 50.2 m 高于一次清除门限 40 m")
    fig.tight_layout()
    return 存(fig, "图5_效率前沿")


def 图6_候选区域():
    网 = 读("候选区域网格.csv"); 透镜 = 读("图_透镜边界.csv"); Ω1 = 读("图_先验区域.csv")
    A = np.sort(网["a"].unique()); B = np.sort(网["b"].unique())
    Z = 网.pivot(index="b", columns="a", values="时间均值").reindex(
        index=B, columns=A).to_numpy()
    B全 = np.concatenate([-B[::-1], B]); Z全 = np.vstack([Z[::-1, :], Z])
    最优值 = float(np.nanmin(Z))
    框 = 时间框

    fig = plt.figure(figsize=(13.2, 5.9))
    左 = fig.add_axes([0.055, 0.08, 0.27, 0.80])
    右 = fig.add_axes([0.40, 0.08, 0.50, 0.80])

    左.add_patch(圆块((0, 0), R_AREA, fill=False, ls="--", color="0.55", lw=1.2))
    左.add_patch(多边形块(Ω1[["x", "y"]].to_numpy(), closed=True, facecolor=区域色,
                          alpha=0.55, edgecolor=区域色, lw=0.8, label="可行集 $\\Omega_1$"))
    for 号 in (1, -1):
        左.add_patch(矩形块((框["a低"], 框["b低"] if 号 > 0 else -框["b高"]),
                            框["a高"]-框["a低"], 框["b高"]-框["b低"],
                            facecolor=主色, alpha=0.75, edgecolor="none",
                            label="第二检测点候选区域" if 号 > 0 else None))
    左.plot(450, 120, "*", color=主色, ms=14, mec="k", mew=0.5)
    左.plot(0, 0, "k^", ms=10); 左.text(-95, -270, "$S_1$", fontsize=12)
    左.set_xlim(-1950, 1950); 左.set_ylim(-1950, 1950); 左.set_aspect("equal")
    左.set_title("(a) 在目标区域中的位置", fontsize=11)
    左.set_xlabel("沿示向线方向 (m)"); 左.set_ylabel("横向 (m)")
    左.legend(fontsize=7.5, loc="upper left"); 左.grid(alpha=0.25)

    cs = 右.contourf(A, B全, Z全, levels=np.linspace(最优值, 最优值*2.6, 40),
                     cmap="viridis_r", extend="max")
    fig.colorbar(cs, ax=右, pad=0.02).set_label("期望总清除时间 $E[T]$ (s)", fontsize=9)
    for 系数, 宽 in [(1.05, 2.4), (1.10, 1.2), (1.20, 1.0)]:
        右.contour(A, B全, Z全, levels=[最优值*系数], colors="w", linewidths=宽)
    for 记, 组 in 透镜.groupby("来源"):
        心 = 5.0 if 记 == "近端圆" else 1000.0
        ang = np.degrees(np.arctan2(组["y"].to_numpy(), 组["x"].to_numpy()-心))
        序 = np.argsort(ang if 记 == "近端圆" else ang % 360.0)
        x, y = 组["x"].to_numpy()[序], 组["y"].to_numpy()[序]
        右.plot(x, y, color=约束色, lw=1.8,
                label="接收可靠域边界" if 记 == "近端圆" else None)
    右.annotate("", xy=(1500, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.4, ls="--"))
    右.text(1250, -60, "示向度方向", fontsize=9)
    右.plot(0, 0, "k^", ms=11, zorder=5); 右.text(-165, -62, "$S_1$", fontsize=11)
    for 号 in (1, -1):
        右.add_patch(矩形块((框["a低"], 框["b低"] if 号 > 0 else -框["b高"]),
                            框["a高"]-框["a低"], 框["b高"]-框["b低"], fill=False,
                            edgecolor=主色, lw=2.2, zorder=6,
                            label="候选区域（域内各点 $E[T]\\leq1.05\\,E[T]^*$）"
                                  if 号 > 0 else None))
        右.plot(450, 号*120, "*", color=主色, ms=19, mec="k", mew=0.7, zorder=7,
                label="推荐点 $(450,\\,120)$，$E[T]=216.5$ s" if 号 > 0 else None)
    右.plot(900, 580, "X", color=次色, ms=12, mec="w", mew=0.8, zorder=7,
            label="几何准则最优点 $(900,\\,580)$，$E[T]=366.1$ s")
    右.text(-330, -840, "白色等值线：相对最优 $+5\\%/+10\\%/+20\\%$",
            fontsize=9, color="w")
    右.set_xlim(-350, 1750); 右.set_ylim(-900, 900); 右.set_aspect("equal")
    右.set_xlabel("前进量 $a$ (m)"); 右.set_ylabel("侧移量 $b$ (m)")
    右.set_title("(b) 时间准则下的目标函数等值面与候选区域", fontsize=11)
    右.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    fig.suptitle("图 6　第二检测点的候选区域："
                 "$a\\in[0.10,0.30]\\rho_{hi}$，$|b|\\in[0.05,0.12]\\rho_{hi}$，左右对称",
                 fontsize=13)
    return 存(fig, "图6_候选区域")


# ================================================================ 阶段 3
def 图7_边际分析():
    侧 = 读("边际账_侧移.csv"); 前 = 读("边际账_前进.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(侧["侧移b"], 侧["绕路米"], color="#8c564b",
               label="附加行程（$\\propto b^{2}$）")
    ax[0].plot(侧["侧移b"], 侧["收尾绕路米"], color="#17becf",
               label="后续补测行程（$\\propto 1/b$）")
    ax[0].plot(侧["侧移b"], 侧["合计米"], color=主色, lw=2, label="合计")
    k = 侧["合计米"].idxmin()
    ax[0].plot(侧.loc[k, "侧移b"], 侧.loc[k, "合计米"], "*", color=主色, ms=16, mec="k")
    ax[0].axhline(25, color="0.4", ls=":", label="一次检测的等效行程 25 m")
    ax[0].set_xlim(0, 500); ax[0].set_ylim(0, 260)
    ax[0].set_xlabel("侧移量 $b$ (m)"); ax[0].set_ylabel("等效行程 (m)")
    ax[0].set_title(f"(a) 侧移量：两项相消，最优 $b^*\\approx{侧.loc[k,'侧移b']:.0f}$ m")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    合 = 前["绕路米"] + 前["收尾绕路米"] + 前["期望超前惩罚米"]
    ax[1].plot(前["前进a"], 前["绕路米"], color="#8c564b", label="侧移的附加行程")
    ax[1].plot(前["前进a"], 前["收尾绕路米"], color="#17becf", label="后续补测行程")
    ax[1].plot(前["前进a"], 前["期望超前惩罚米"], color="#9467bd",
               label="超越真实源距的往返行程")
    ax[1].plot(前["前进a"], 合, color=主色, lw=2, label="合计")
    k = 合.idxmin()
    ax[1].plot(前.loc[k, "前进a"], 合.iloc[k], "*", color=主色, ms=16, mec="k")
    ax[1].axhline(25, color="0.4", ls=":")
    ax[1].set_ylim(0, 300)
    ax[1].set_xlabel("前进量 $a$ (m)"); ax[1].set_ylabel("等效行程 (m)")
    ax[1].set_title(f"(b) 前进量：U 形代价 + 非对称惩罚，最优 $a^*\\approx{前.loc[k,'前进a']:.0f}$ m")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("图 7　以等效行程为统一量纲的边际分析（一次检测 $=5$ s $=25$ m）")
    fig.tight_layout()
    return 存(fig, "图7_边际分析")


def 图8_途中加测():
    t = 读("途中加测对比.csv"); 段 = t[t["区间上"] < 1500]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(段)); w = 0.38
    ax[0].bar(x-w/2, 段["一点式"], w, color=次色, label="单点方案（仅在 $0.30\\rho_{hi}$ 处检测）")
    ax[0].bar(x+w/2, 段["两点式"], w, color=主色,
              label="两点方案（$0.15\\rho_{hi}$ 处沿途增测一次）")
    for i, r in enumerate(段.itertuples()):
        ax[0].text(i, max(r.一点式, r.两点式)+6, f"{r.差:+.0f} s", ha="center", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{int(r.区间下)}–{int(r.区间上)}" for r in 段.itertuples()])
    ax[0].set_xlabel("干扰源真实距离 $\\rho$ (m)"); ax[0].set_ylabel("平均总清除时间 (s)")
    ax[0].set_title("(a) 增益集中于近距离源：沿途点近似其正横位置")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")

    全 = t[t["区间上"] == 1500].iloc[0]
    ax[1].bar([0, 1], [全["一点式"], 全["两点式"]], 0.5, color=[次色, 主色])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["单点方案", "两点方案"])
    ax[1].set_ylim(150, 240); ax[1].set_ylabel("平均总清除时间 (s)")
    for i, (v, n) in enumerate([(全["一点式"], 全["一点式检测次数"]),
                                (全["两点式"], 全["两点式检测次数"])]):
        ax[1].text(i, v+2, f"{v:.1f} s\n检测 {n:.2f} 次", ha="center", fontsize=9)
    ax[1].set_title(f"(b) 总体平均：{全['差']:+.1f} s，"
                    f"检测次数增加 {全['两点式检测次数']-全['一点式检测次数']:.2f}")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("图 8　沿途增测的效果：检测的等效行程仅 25 m，且沿途检测不产生附加行程")
    fig.tight_layout()
    return 存(fig, "图8_途中加测")


# ================================================================ 阶段 4
def 图9_仿真():
    网 = 读("策略仿真网格.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    Λ = np.sort(网["λ"].unique()); Μ = np.sort(网["μ"].unique())
    Z = 网.pivot(index="μ", columns="λ", values="平均总时间").reindex(
        index=Μ, columns=Λ).to_numpy()
    cs = ax[0].contourf(Λ, Μ, Z, levels=24, cmap="viridis_r")
    fig.colorbar(cs, ax=ax[0], label="平均总清除时间 (s)", pad=0.02)
    最优 = 网.loc[网["平均总时间"].idxmin()]
    ax[0].contour(Λ, Μ, Z, levels=[最优["平均总时间"]*1.05], colors="w", linewidths=2)
    ax[0].plot(最优["λ"], 最优["μ"], "*", color=主色, ms=17, mec="k",
               label=f"仿真最优 $({最优['λ']:.2f},\\,{最优['μ']:.2f})$")
    ax[0].plot(0.30, 0.097, "s", color=次色, ms=8, mec="k",
               label="代理模型最优 $(0.30,\\,0.10)$")
    ax[0].set_xlabel("$\\lambda$：前进量 / $\\rho_{hi}$")
    ax[0].set_ylabel("$\\mu$：侧移量 / $\\rho_{hi}$")
    ax[0].set_title("(a) 代理模型与端到端仿真的一致性\n白色等值线：相对最优 $+5\\%$",
                    fontsize=10.5)
    ax[0].legend(fontsize=8)

    轨 = 读("图_轨迹示例.csv"); 区 = 读("图_轨迹区域.csv")
    for k, (阶段, 组) in enumerate(区.groupby("阶段", sort=False)):
        ax[1].add_patch(多边形块(组[["x", "y"]].to_numpy(), closed=True, alpha=0.4,
                                 facecolor=楔形色[k % len(楔形色)],
                                 edgecolor=楔形色[k % len(楔形色)],
                                 label=f"第 {k+1} 次检测后的定位区域"))
    ax[1].plot(轨["x"], 轨["y"], "k.-", lw=1.2, ms=8, label="行进轨迹")
    ax[1].plot(轨["真源x"].iloc[0], 轨["真源y"].iloc[0], "*", color=干扰源色,
               ms=15, label="干扰源真实位置")
    for _, r in 轨.iterrows():
        ax[1].annotate(f"{r['虚拟时刻']:.0f} s", (r["x"], r["y"]),
                       fontsize=7.5, xytext=(6, 6), textcoords="offset points")
    ax[1].set_aspect("equal"); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, loc="upper left")
    ax[1].set_xlabel("$x$ (m)"); ax[1].set_ylabel("$y$ (m)")
    ax[1].set_title("(b) 单个算例的完整过程：定位区域逐次收缩", fontsize=10.5)
    fig.suptitle("图 9　端到端仿真：参数空间与典型算例")
    fig.tight_layout()
    return 存(fig, "图9_仿真")


def 图10_策略对比():
    t = 读("策略对照汇总.csv").sort_values("平均总时间")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    色 = [主色 if "本文" in n else (约束色 if "下界" in n else 次色) for n in t["方案"]]
    y = np.arange(len(t))
    ax.barh(y, t["平均总时间"], color=色)
    ax.set_yticks(y); ax.set_yticklabels(t["方案"], fontsize=9); ax.invert_yaxis()
    下界 = float(t["平均总时间"].min())
    ax.axvline(下界, color=约束色, ls="--", lw=1.2)
    for i, r in enumerate(t.itertuples()):
        ax.text(r.平均总时间+6, i, f"{r.平均总时间:.1f} s（{r.相对下界:.2f}$\\times$）",
                va="center", fontsize=8.5)
    ax.set_xlim(0, float(t["平均总时间"].max())*1.24)
    ax.set_xlabel("平均总清除时间 (s)")
    ax.set_title("图 10　各策略平均总清除时间对比（虚线为理论下界 173.9 s）")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return 存(fig, "图10_策略对比")


def main():
    P1绘图.配置中文字体()
    matplotlib.rcParams["pdf.fonttype"] = 42      # TrueType 嵌入，避免 Type 3
    matplotlib.rcParams["pdf.compression"] = 6
    图目录.mkdir(exist_ok=True)
    for f in (图1_可行集, 图2_交会几何, 图3_直径随源距, 图4_两个准则, 图5_效率前沿,
              图6_候选区域, 图7_边际分析, 图8_途中加测, 图9_仿真, 图10_策略对比):
        print(f"  {f()}.pdf")
    print(f"已写入 {图目录}/*.pdf（矢量，可直接 \\includegraphics）")


if __name__ == "__main__":
    main()
