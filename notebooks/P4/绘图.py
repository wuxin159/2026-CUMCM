"""问题四 · 图（只读 CSV 作图，不算数）。

每张图单独输出一个矢量 PDF，便于 LaTeX 直接 \\includegraphics。不输出位图。
字体按 TrueType 嵌入（pdf.fonttype=42），避免 Type 3 在打印环节出问题。
配色沿用 P1~P3 的一套，免得四问的图风格不一致。

    图 1  判空判据的对比：圆盘覆盖 vs 半圆盘围堵，以及两者的代价
    图 2  围堵判据的几何：一个候选位置要被「围起来」才能排除
    图 3  定向几何三件事：半平面覆盖、收信锥、镜像必达
    图 4  围堵锚点构造与巡回（24 点，13 个在目标区域外）
    图 5  单局全景：算例 A 的轨迹、停靠点与干扰源（定向源画出朝向）
    图 6  定位区域随检测次数的收缩
    图 7  清除进度曲线：推荐策略 vs 两个基线
    图 8  时间去向的四分项
    图 9  策略对照与两条参照线（含「套用问题三判空」的清除率）
    图 10 消融实验
    图 11 分组统计：源数与定向比例
    图 12 「收信把握」的校准曲线

论文成图仍推荐交给 PGFPlots 读同一批 CSV；本脚本的 PDF 是矢量备选与校对用图。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as 圆块, Wedge as 扇块, Polygon as 多边形块

HERE = Path(__file__).resolve().parent
图目录 = HERE / "图"
图目录.mkdir(exist_ok=True)

R_AREA, R_RECV_MIN, R_CLEAR = 1800.0, 1000.0, 20.0

主色 = "#d62728"       # 推荐策略 / 关键量
次色 = "#1f77b4"       # 对照 / 区域
三色 = "#2ca02c"       # 约束 / 下界
四色 = "#ff7f0e"       # 警示 / 超出
灰 = "#7f7f7f"


def 配置():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                            "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def 读(名):
    p = HERE / 名
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def 存(fig, 名):
    fig.savefig(图目录 / f"{名}.pdf")
    plt.close(fig)
    print(f"    {名}.pdf")


def 目标圆(ax, 半径=R_AREA, **kw):
    ax.add_patch(圆块((0, 0), 半径, fill=False, ls="--", lw=1.0,
                      color=kw.pop("color", 灰), **kw))


# ================================================================ 图 1
def 图1_判据对照():
    表 = 读("图_判据对照.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) 两种判据的示意
    ax[0].add_patch(圆块((-1.1, 0), 0.95, fc=次色, alpha=0.18, ec=次色))
    ax[0].plot([-1.1], [0], "o", color=次色, ms=6)
    ax[0].plot([-1.1], [0.35], "*", color=主色, ms=13)
    ax[0].text(-1.1, -1.25, "问题三：一个见证者就够\n无信号 ⇒ 排除整个 1000 m 圆盘",
               ha="center", fontsize=9)
    ang = [90, 210, 330]
    for a in ang:
        p = 0.85 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) + [1.1, 0]
        ax[0].plot(*p, "o", color=次色, ms=6)
        ax[0].plot([1.1, p[0]], [0, p[1]], "-", color=灰, lw=0.8)
    ax[0].plot([1.1], [0], "*", color=主色, ms=13)
    ax[0].add_patch(多边形块(
        [0.85 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) + [1.1, 0]
         for a in ang], fc=三色, alpha=0.16, ec=三色))
    ax[0].text(1.1, -1.25, "问题四：要把它围起来\n无信号点的凸包必须含住候选位置",
               ha="center", fontsize=9)
    ax[0].set_xlim(-2.4, 2.4)
    ax[0].set_ylim(-1.7, 1.4)
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    ax[0].set_title("(a) 判空判据：从「覆盖」变成「围堵」")
    # (b) 代价对照
    for 名, c, mk in (("问题三：圆盘覆盖", 次色, "s-"),
                      ("问题四：半圆盘围堵", 主色, "o-")):
        子 = 表[表["方案"] == 名]
        ax[1].plot(子["空频道数"], 子["合计秒"], mk, color=c, lw=1.8, ms=5,
                   label=f"{名}（{int(子['锚点数'].iloc[0])} 个锚点）")
    ax[1].set_xlabel("空频道个数")
    ax[1].set_ylabel("判空总代价（秒）")
    ax[1].set_title("(b) 判空代价：锚点行进 + 逐频道检测")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    存(fig, "图1_判据对照")


# ================================================================ 图 2
def 图2_围堵示意():
    表 = 读("图_围堵示意.csv")
    if 表 is None:
        return
    情形 = list(dict.fromkeys(表["情形"]))
    fig, ax = plt.subplots(1, len(情形), figsize=(4.0 * len(情形), 4.0))
    for k, (a, 名) in enumerate(zip(np.atleast_1d(ax), 情形)):
        子 = 表[表["情形"] == 名]
        a.plot([0], [0], "*", color=主色, ms=16, label="候选干扰源位置 $x$")
        for _, r in 子.iterrows():
            a.plot([r["px"]], [r["py"]], "o", color=次色, ms=7)
            a.annotate("", xy=(r["px"] * 0.35, r["py"] * 0.35), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color=次色, lw=1.3))
            a.add_patch(扇块((0, 0), 1150, r["方位角"] - 90, r["方位角"] + 90,
                             fc=次色, alpha=0.10, ec="none"))
        排 = bool(子["已排除"].iloc[0])
        a.add_patch(圆块((0, 0), R_RECV_MIN, fill=False, ls=":", color=灰, lw=1.0))
        a.set_title(f"{名}：{'可以排除' if 排 else '不能排除'}",
                    color=(三色 if 排 else 四色))
        a.set_xlim(-1300, 1300)
        a.set_ylim(-1300, 1300)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("阴影 = 已被见证的定向方向；留白 = 源仍可能朝这个方向（于是排除不掉）",
                 fontsize=10, y=0.02)
    fig.tight_layout()
    存(fig, "图2_围堵示意")


# ================================================================ 图 3
def 图3_定向几何():
    表 = 读("图_定向几何.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    源 = 表[表["元素"] == "源"].iloc[0]
    φ = 源["角"]
    u = np.array([np.cos(np.radians(φ)), np.sin(np.radians(φ))])
    ax.add_patch(扇块((0, 0), 1400, φ - 90, φ + 90, fc=三色, alpha=0.13, ec=三色,
                      lw=1.2, label="有效覆盖范围（半平面）"))
    ax.annotate("", xy=700 * u, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=三色, lw=2.0))
    ax.text(*(760 * u), "定向方向 $u$", color=三色, fontsize=10)
    ax.plot([0], [0], "*", color=主色, ms=18, label="干扰源 $G$")
    p = {r["元素"]: (r["x"], r["y"]) for _, r in 表.iterrows()}
    for 名, mk, c in (("检测点1", "o", 次色), ("检测点2", "o", 次色),
                      ("落空点", "x", 四色), ("镜像点", "^", 主色)):
        if 名 in p:
            ax.plot(*p[名], mk, color=c, ms=10, mew=2)
            ax.annotate(名, p[名], textcoords="offset points", xytext=(8, 6),
                        fontsize=9, color=c)
    if "检测点1" in p and "检测点2" in p:
        ax.add_patch(多边形块([(0, 0), p["检测点1"],
                               np.asarray(p["检测点1"]) * 1.6,
                               np.asarray(p["检测点2"]) * 1.9, p["检测点2"]],
                              fc=次色, alpha=0.22, ec=次色, ls="--",
                              label="收信锥（引理 D1：锥内必定收得到）"))
    if "落空点" in p and "镜像点" in p:
        ax.plot([p["落空点"][0], p["检测点1"][0], p["镜像点"][0]],
                [p["落空点"][1], p["检测点1"][1], p["镜像点"][1]],
                "--", color=主色, lw=1.2)
        ax.text(*(np.asarray(p["镜像点"]) + [30, -90]),
                "引理 D3 镜像必达\n$r=p+\\lambda(p-q)$", color=主色, fontsize=9)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1200, 1600)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("定向源的覆盖范围是一个过 $G$ 的半平面")
    fig.tight_layout()
    存(fig, "图3_定向几何")


# ================================================================ 图 4
def 图4_锚点布局():
    表 = 读("图_锚点布局.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    目标圆(ax)
    ax.add_patch(圆块((0, 0), R_AREA, fc=次色, alpha=0.05))
    内 = 表[~表["在区域外"]]
    外 = 表[表["在区域外"]]
    ax.plot(内["x"], 内["y"], "o", color=次色, ms=8, label="区域内锚点")
    ax.plot(外["x"], 外["y"], "s", color=四色, ms=8, label="区域外锚点")
    路 = 表.sort_values("访问序")
    ax.plot([0] + list(路["x"]), [0] + list(路["y"]), "-", color=灰, lw=1.0,
            label=f"开放巡回 {表['巡回总长'].iloc[0]:.0f} m")
    ax.plot([0], [0], "*", color=主色, ms=16, label="出发点")
    for _, r in 表.iterrows():
        ax.annotate(int(r["访问序"]), (r["x"], r["y"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.5)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"围堵锚点构造：{len(表)} 个点，其中 {len(外)} 个在目标区域之外")
    fig.tight_layout()
    存(fig, "图4_锚点布局")


# ================================================================ 图 5
def 图5_单局全景():
    轨 = 读("图_算例A_轨迹.csv")
    源 = 读("图_算例A_干扰源.csv")
    停 = 读("图_算例A_停靠.csv")
    if 轨 is None or 源 is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    目标圆(ax)
    ax.plot(轨["x"], 轨["y"], "-", color=灰, lw=0.9, alpha=0.85, label="机器狗轨迹")
    if 停 is not None and len(停):
        for 类型, c, mk, lab in (("锚点", 次色, "s", "判空锚点停靠"),
                                 ("扫描", 三色, "o", "顺路整池扫描"),
                                 ("补测", 四色, "^", "补测停靠"),
                                 ("清除", 主色, "x", "清除动作")):
            子 = 停[停["类型"] == 类型]
            if len(子):
                ax.plot(子["x"], 子["y"], mk, color=c, ms=6, ls="none", mew=1.6,
                        alpha=0.9, label=lab)
    全 = 源[~源["是定向"]]
    定 = 源[源["是定向"]]
    ax.plot(全["x"], 全["y"], "*", color="k", ms=14, ls="none", label="全向干扰源")
    ax.plot(定["x"], 定["y"], "P", color="#8c564b", ms=11, ls="none", label="定向干扰源")
    for _, r in 定.iterrows():
        u = np.array([np.cos(np.radians(r["定向方向"])),
                      np.sin(np.radians(r["定向方向"]))])
        ax.annotate("", xy=(r["x"] + 260 * u[0], r["y"] + 260 * u[1]),
                    xytext=(r["x"], r["y"]),
                    arrowprops=dict(arrowstyle="-|>", color="#8c564b", lw=1.4))
    ax.plot([0], [0], "D", color=主色, ms=8, label="出发点 $(0,0)$")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title("算例 A：信息获取与执行清除走的是同一条路")
    fig.tight_layout()
    存(fig, "图5_单局全景")


# ================================================================ 图 6
def 图6_收缩曲线():
    表 = 读("图_收缩曲线.csv")
    if 表 is None or 表.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for c, 子 in 表.groupby("频道"):
        ax.plot(子["检测序"], np.maximum(子["Rmin"], 1e-1), "-", lw=1.2, alpha=0.8)
    ax.axhline(R_CLEAR, color=主色, lw=1.8, ls="--",
               label=f"清除半径 {R_CLEAR:.0f} m（区域压到这条线以下即可一枪必中）")
    ax.set_yscale("log")
    ax.set_xlabel("检测次数（全局计数）")
    ax.set_ylabel("定位区域最小覆盖圆半径 $R_{\\min}$ (m)")
    ax.set_title("每条曲线是一个频道：从上千米压到 20 米以内")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    存(fig, "图6_收缩曲线")


# ================================================================ 图 7
def 图7_进度曲线():
    表 = 读("图_进度曲线.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    色 = {"推荐策略": 主色, "基线一：先扫后清": 次色, "基线二：逐源处理": 三色}
    for 名, 子 in 表.groupby("策略"):
        ax.step(子["虚拟时刻"], 子["已清除数"], where="post", lw=1.9,
                color=色.get(名, 灰), label=名)
    ax.set_xlabel("虚拟时刻 (s)")
    ax.set_ylabel("累计已清除干扰源个数")
    ax.set_title("清除进度：基线的长平台段就是「信息与执行分离」的代价")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    存(fig, "图7_进度曲线")


# ================================================================ 图 8
def 图8_时间去向():
    表 = 读("图_时间去向.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    色 = [次色, 主色, 四色, 三色]
    ax[0].bar(表["项目"], 表["秒"], color=色[:len(表)])
    for i, r in 表.reset_index().iterrows():
        ax[0].text(i, r["秒"], f"{r['秒']:.0f} s\n{r['占比']:.1%}", ha="center",
                   va="bottom", fontsize=9)
    ax[0].set_ylabel("虚拟耗时 (s)")
    ax[0].set_title("(a) 时间去向")
    ax[0].grid(alpha=0.3, axis="y")
    ax[0].set_ylim(0, 表["秒"].max() * 1.25)
    对 = pd.DataFrame({"项目": ["移动", "检测", "切频", "清除"],
                       "问题三": [0.720, 0.217, 0.042, 0.021]})
    宽 = 0.38
    idx = np.arange(len(对))
    ax[1].bar(idx - 宽 / 2, 对["问题三"], 宽, color=灰, label="问题三")
    实 = [float(表.loc[表["项目"] == k, "占比"].iloc[0]) if (表["项目"] == k).any()
          else 0.0 for k in 对["项目"]]
    ax[1].bar(idx + 宽 / 2, 实, 宽, color=主色, label="问题四")
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels(对["项目"])
    ax[1].set_ylabel("占总时间比例")
    ax[1].set_title("(b) 与问题三对比：检测占比大幅上升")
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    存(fig, "图8_时间去向")


# ================================================================ 图 9
def 图9_策略对照():
    表 = 读("策略对照汇总.csv")
    if 表 is None:
        return
    表 = 表[表["平均总时间"] < 5e4]                 # 盲清搜索差两个量级，单列不画
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6))
    y = np.arange(len(表))
    色 = [主色 if "推荐" in s else (四色 if "问题三" in s else 次色)
          for s in 表["策略"]]
    ax[0].barh(y, 表["平均总时间"], color=色)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(表["策略"], fontsize=9)
    ax[0].invert_yaxis()
    for i, v in enumerate(表["平均总时间"]):
        ax[0].text(v, i, f" {v:.0f}", va="center", fontsize=9)
    ax[0].set_xlabel("平均虚拟总时间 (s)")
    ax[0].set_title("(a) 时间")
    ax[0].grid(alpha=0.3, axis="x")
    ax[1].barh(y, 表["清除率"], color=色)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].axvline(1.0, color="k", lw=1.0, ls="--")
    for i, v in enumerate(表["清除率"]):
        ax[1].text(v, i, f" {v:.4f}", va="center", fontsize=9)
    ax[1].set_xlim(min(0.9, 表["清除率"].min() - 0.02), 1.02)
    ax[1].set_xlabel("被清除干扰源个数的比例")
    ax[1].set_title("(b) 完成度：套用问题三判空的那一行掉下来了")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    存(fig, "图9_策略对照")


# ================================================================ 图 10
def 图10_消融():
    表 = 读("消融汇总.csv")
    if 表 is None:
        return
    表 = 表.iloc[1:].sort_values("配对差")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(表))
    色 = [主色 if abs(t) >= 2 else 灰 for t in 表["t"]]
    ax.barh(y, 表["配对差"], xerr=表["标准误"], color=色, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(表["配置"], fontsize=8.5)
    ax.axvline(0, color="k", lw=1.0)
    for i, (d, t) in enumerate(zip(表["配对差"], 表["t"])):
        ax.text(d, i, f"  {d:+.0f} s (t={t:+.1f})", va="center", fontsize=8)
    ax.set_xlabel("关掉该设计后的逐局配对时间差 (s)，正值表示该设计有用")
    ax.set_title("消融实验：深色为 |t| 不小于 2（显著），浅色在噪声量级内")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    存(fig, "图10_消融")


# ================================================================ 图 11
def 图11_分组():
    表 = 读("分组统计.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.0))
    子 = 表[表["分组"] == "源数"].dropna(subset=["源数"])
    if len(子):
        ax[0].plot(子["源数"], 子["平均总时间"], "o-", color=次色, lw=1.8,
                   label="平均总时间")
        a2 = ax[0].twinx()
        a2.plot(子["源数"], 子["平均定位清除时间"], "s--", color=主色, lw=1.6,
                label="平均定位清除时间")
        a2.set_ylabel("平均定位清除时间 (s/源)", color=主色)
        ax[0].set_xlabel("干扰源个数")
        ax[0].set_ylabel("平均虚拟总时间 (s)", color=次色)
        ax[0].set_title("(a) 按源数")
        ax[0].grid(alpha=0.3)
    子 = 表[表["分组"] == "定向比例"].dropna(subset=["定向比例档"])
    if len(子):
        ax[1].bar(子["定向比例档"].astype(str), 子["平均总时间"], color=次色)
        ax[1].set_xlabel("定向源占比")
        ax[1].set_ylabel("平均虚拟总时间 (s)")
        ax[1].set_title("(b) 按定向比例")
        ax[1].grid(alpha=0.3, axis="y")
    子 = 表[表["分组"] == "布局"].dropna(subset=["布局"])
    if len(子):
        ax[2].bar(子["布局"].astype(str), 子["平均总时间"], color=三色)
        ax[2].set_xlabel("干扰源布局")
        ax[2].set_ylabel("平均虚拟总时间 (s)")
        ax[2].set_title("(c) 按布局")
        ax[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    存(fig, "图11_分组")


# ================================================================ 图 12
def 图12_把握校准():
    表 = 读("图_把握校准.csv")
    if 表 is None or 表.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], "--", color=灰, lw=1.2, label="完美校准")
    ax.plot(表["预测均值"], 表["实际比例"], "o-", color=主色, lw=1.8, ms=7,
            label="本文的「收信把握」")
    for _, r in 表.iterrows():
        ax.annotate(f"n={int(r['次数'])}", (r["预测均值"], r["实际比例"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("预测的收信把握")
    ax.set_ylabel("实际收到信号的比例")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("「收信把握」的校准：点落在对角线附近才说明模型没说谎")
    fig.tight_layout()
    存(fig, "图12_把握校准")


# ================================================================ 主程序
def main():
    配置()
    print("问题四 · 出图")
    for f in (图1_判据对照, 图2_围堵示意, 图3_定向几何, 图4_锚点布局,
              图5_单局全景, 图6_收缩曲线, 图7_进度曲线, 图8_时间去向,
              图9_策略对照, 图10_消融, 图11_分组, 图12_把握校准):
        try:
            f()
        except Exception as e:                     # 某张图缺数据不该拖垮其余的
            print(f"    [跳过] {f.__name__}：{e!r}")
    print(f"图已写入 {图目录}")


if __name__ == "__main__":
    main()
