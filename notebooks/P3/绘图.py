"""问题三 · 图（只读 CSV 作图，不算数）。

每张图单独输出一个矢量 PDF，便于 LaTeX 直接 \\includegraphics。不输出位图。
字体按 TrueType 嵌入（pdf.fonttype=42），避免 Type 3 在打印环节出问题。
配色沿用 P1 的一套，免得三问的图风格不一致。

    图 1  判空所需的圆覆盖数（六点差 0.63 m 的临界）
    图 2  锚点布局：七点构造与最短巡回布局
    图 3  三类约束的几何：楔形、无信号排除盘、接收半径序约束
    图 4  单局全景：算例 A 的轨迹、停靠点与干扰源
    图 5  定位区域随检测次数的收缩
    图 6  清除进度曲线：三种策略对比
    图 7  时间去向的四分项
    图 8  策略对照与理论下界
    图 9  参数灵敏度
    图 10 消融实验
    图 11 分组统计：源数与布局
    图 12 稳健性：误差模型、R_eff、楔形半角

论文成图仍推荐交给 PGFPlots 读同一批 CSV；本脚本的 PDF 是矢量备选与校对用图。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as 圆块, Polygon as 多边形块

HERE = Path(__file__).resolve().parent
图目录 = HERE / "图"
图目录.mkdir(exist_ok=True)

R_AREA, R_RECV_MIN, R_CLEAR = 1800.0, 1000.0, 20.0

楔形色 = ["#7f7f7f", "#8c564b", "#9467bd", "#17becf", "#bcbd22"]
主色 = "#d62728"       # 推荐策略 / 关键量
次色 = "#1f77b4"       # 对照 / 区域
三色 = "#2ca02c"       # 约束 / 下界
四色 = "#ff7f0e"       # 警示 / 超出


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
    return 名


# ================================================================ 图 1
def 图1_覆盖数():
    表 = 读("覆盖数_最优半径.csv")
    残 = 读("覆盖残余_六点.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    n = 表["锚点数n"]
    ax[0].plot(n, 表["最优覆盖半径_米"], "o-", color=次色, lw=1.8,
               label="本文数值搜索")
    有文献 = 表["文献最优半径比"].notna()
    ax[0].plot(n[有文献], 表.loc[有文献, "文献最优半径比"] * R_AREA, "s--",
               color=楔形色[0], ms=5, lw=1.0, label="文献最优值")
    ax[0].axhline(R_RECV_MIN, color=主色, lw=1.6,
                  label="可用排除半径 $r=1000$ m")
    ax[0].annotate(f"$n=6$：需要 1000.63 m\n只差 0.63 m",
                   xy=(6, 表.loc[表["锚点数n"] == 6, "最优覆盖半径_米"].iloc[0]),
                   xytext=(6.3, 1350), fontsize=9, color=主色,
                   arrowprops=dict(arrowstyle="->", color=主色, lw=1.0))
    ax[0].set_xlabel("锚点个数 $n$")
    ax[0].set_ylabel("覆盖半径 1800 m 圆域所需的等圆半径 (m)")
    ax[0].set_title("(a) 覆盖数 $K^*=7$：六点差 0.63 米")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=9)

    ax[1].add_patch(圆块((0, 0), R_AREA, fill=False, ls="--", color="0.5",
                         label="目标区域边界"))
    if 残 is not None and len(残):
        ax[1].scatter(残["x"], 残["y"], s=6, color=四色, zorder=3,
                      label=f"六点布局的未覆盖残余（{len(残)} m²）")
    ax[1].set_xlim(-2000, 2000)
    ax[1].set_ylim(-2000, 2000)
    ax[1].set_aspect("equal")
    ax[1].grid(alpha=0.3)
    ax[1].set_xlabel("$x$ (m)")
    ax[1].set_ylabel("$y$ (m)")
    ax[1].set_title("(b) 最优六点布局漏掉的地方")
    ax[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("图 1　判空所需的圆覆盖问题：为什么是 7 个锚点")
    fig.tight_layout()
    return 存(fig, "图1_覆盖数")


# ================================================================ 图 2
def 图2_锚点布局():
    七 = 读("七点构造.csv")
    布局 = 读("锚点布局.csv")
    if 七 is None or 布局 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.0))
    for k, (标题, 点表) in enumerate([
            ("(a) 教科书构造：圆心 + 6 个环点，等圆半径 900 m", 七),
            ("(b) 巡回最短布局（$K=9$，路程 5770 m）",
             布局[布局["K"] == 9] if 布局 is not None else None)]):
        a = ax[k]
        a.add_patch(圆块((0, 0), R_AREA, fill=False, ls="--", color="0.5"))
        if 点表 is None or not len(点表):
            continue
        P = 点表[["x", "y"]].to_numpy()
        for p in P:
            a.add_patch(圆块(p, R_RECV_MIN, fill=True, alpha=0.10, color=次色))
            a.add_patch(圆块(p, R_RECV_MIN, fill=False, lw=0.7, color=次色))
        路 = np.vstack([[0, 0], P])
        a.plot(路[:, 0], 路[:, 1], "-", color=主色, lw=1.4, zorder=4,
               label="巡回路线")
        a.plot(P[:, 0], P[:, 1], "o", color=主色, ms=6, zorder=5)
        a.plot(0, 0, "k^", ms=9, zorder=6, label="出发点 (0,0)")
        for i, p in enumerate(P):
            a.annotate(str(i + 1), xy=p, xytext=(6, 6),
                       textcoords="offset points", fontsize=8)
        a.set_xlim(-2900, 2900)
        a.set_ylim(-2900, 2900)
        a.set_aspect("equal")
        a.grid(alpha=0.3)
        a.set_title(标题, fontsize=10)
        a.set_xlabel("$x$ (m)")
        a.set_ylabel("$y$ (m)")
        a.legend(fontsize=8, loc="upper left")
    fig.suptitle("图 2　判空锚点：淡蓝圆是单次 no_signal 排除的 1000 m 圆盘")
    fig.tight_layout()
    return 存(fig, "图2_锚点布局")


# ================================================================ 图 3
def 图3_三类约束():
    段 = 读("图_约束示意.csv")
    点 = 读("图_约束示意点.csv")
    顶 = 读("图_定位区域.csv")
    if 段 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for k in (0, 1):
        a = ax[k]
        a.add_patch(圆块((0, 0), R_AREA, fill=False, ls="--", color="0.5"))
        for i, (名, 组) in enumerate(段[段["元素"].str.startswith("楔形")]
                                     .groupby("元素")):
            for _, r in 组.iterrows():
                a.plot([r["x1"], r["x2"]], [r["y1"], r["y2"]],
                       color=楔形色[i], lw=1.5 if r["线"] == "中线" else 0.7,
                       ls="-" if r["线"] == "中线" else "--",
                       label=f"{名} 示向线" if r["线"] == "中线" else None)
        序 = 段[段["元素"] == "序约束中垂线"]
        for _, r in 序.iterrows():
            a.plot([r["x1"], r["x2"]], [r["y1"], r["y2"]], color=三色,
                   lw=1.6, ls="-.", label="序约束边界（中垂线）")
        无 = 点[点["元素"] == "无信号点D"]
        for _, r in 无.iterrows():
            a.add_patch(圆块((r["x"], r["y"]), r["参数"], fill=True, alpha=0.12,
                             color=四色))
            a.add_patch(圆块((r["x"], r["y"]), r["参数"], fill=False, lw=1.0,
                             color=四色, ls=":"))
        for _, r in 点.iterrows():
            记 = {"检测点S1": ("k^", 9), "检测点S2": ("k^", 9),
                  "无信号点D": ("kv", 9), "真源G": ("o", 7)}[r["元素"]]
            if r["元素"] == "真源G":
                a.plot(r["x"], r["y"], "o", color=主色, ms=7, mec="w", zorder=6,
                       label="真实干扰源 $G$" if k == 0 else None)
            else:
                a.plot(r["x"], r["y"], 记[0], ms=记[1], zorder=6,
                       label=r["元素"] if k == 0 else None)
            a.annotate(r["元素"], xy=(r["x"], r["y"]), xytext=(8, -12),
                       textcoords="offset points", fontsize=8)
        if 顶 is not None and len(顶) >= 3:
            a.add_patch(多边形块(顶[["x", "y"]].to_numpy(), closed=True,
                                 facecolor=次色, alpha=0.55, edgecolor=次色,
                                 zorder=5))
    ax[0].set_xlim(-2200, 2200)
    ax[0].set_ylim(-2200, 2200)
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3)
    ax[0].set_title("(a) 全局：两条示向线、一个无信号排除盘、一条序约束边界",
                    fontsize=10)
    ax[0].legend(fontsize=7, loc="lower left")
    if 顶 is not None and len(顶):
        c = 顶[["x", "y"]].to_numpy().mean(axis=0)
        跨 = max(顶["x"].max() - 顶["x"].min(), 顶["y"].max() - 顶["y"].min())
        w = max(跨 * 1.6, 30.0)
        ax[1].set_xlim(c[0] - w, c[0] + w)
        ax[1].set_ylim(c[1] - w, c[1] + w)
    ax[1].set_aspect("equal")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("(b) 放大：两条示向线交出的定位区域", fontsize=10)
    for a in ax:
        a.set_xlabel("$x$ (m)")
        a.set_ylabel("$y$ (m)")
    fig.suptitle("图 3　问题三新增的三类约束（全部化成半平面或排除盘）")
    fig.tight_layout()
    return 存(fig, "图3_三类约束")


# ================================================================ 图 4
def 图4_单局全景():
    轨 = 读("图_算例A_轨迹.csv")
    源 = 读("图_算例A_干扰源.csv")
    停 = 读("图_算例A_停靠.csv")
    if 轨 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4),
                           gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    a.add_patch(圆块((0, 0), R_AREA, fill=False, ls="--", color="0.5",
                     label="目标区域边界"))
    a.plot(轨["x"], 轨["y"], "-", color="0.72", lw=0.9, zorder=2,
           label=f"机器狗轨迹（{轨['移动距离'].sum():.0f} m）")
    if 停 is not None:
        for 名, 色, 记 in [("扫描停靠", 次色, "s"), ("补测停靠", 楔形色[2], "^")]:
            子 = 停[停["类型"] == 名]
            if len(子):
                a.plot(子["x"], 子["y"], 记, color=色, ms=5, ls="none",
                       zorder=4, label=f"{名}（{len(子)} 次）")
    清 = 轨[(轨["动作"] == "clear") & (轨["结果"] == "success")]
    if len(清):
        a.plot(清["x"], 清["y"], "o", color=三色, ms=9, mfc="none", mew=1.6,
               ls="none", zorder=5, label=f"成功清除（{len(清)} 次）")
    a.plot(源["x"], 源["y"], "*", color=主色, ms=13, mec="w", mew=0.6, zorder=6,
           label=f"干扰源真实位置（{len(源)} 个）")
    a.plot(0, 0, "k^", ms=11, zorder=7, label="出发点 (0,0)")
    a.set_xlim(-2100, 2100)
    a.set_ylim(-2100, 2100)
    a.set_aspect("equal")
    a.grid(alpha=0.3)
    a.set_xlabel("$x$ (m)")
    a.set_ylabel("$y$ (m)")
    a.set_title("(a) 全程轨迹（算例 A）", fontsize=10)
    a.legend(fontsize=7.5, loc="upper left")

    a = ax[1]
    清 = 轨[(轨["动作"] == "clear") & (轨["结果"] == "success")]
    a.step(np.concatenate([[0], 清["虚拟时刻"].to_numpy(),
                           [轨["虚拟时刻"].max()]]),
           np.concatenate([[0], np.arange(1, len(清) + 1), [len(清)]]),
           where="post", color=主色, lw=2.0, label="已清除干扰源数")
    a2 = a.twinx()
    测 = 轨[轨["动作"] == "measure"]
    a2.plot(测["虚拟时刻"], np.arange(1, len(测) + 1), color=次色, lw=1.2,
            ls="--", label="累计检测次数")
    a.set_xlabel("虚拟时刻 (s)")
    a.set_ylabel("已清除干扰源数", color=主色)
    a2.set_ylabel("累计检测次数", color=次色)
    a.grid(alpha=0.3)
    a.set_title("(b) 清除进度与检测投入", fontsize=10)
    线1, 标1 = a.get_legend_handles_labels()
    线2, 标2 = a2.get_legend_handles_labels()
    a.legend(线1 + 线2, 标1 + 标2, fontsize=8, loc="upper left")
    fig.suptitle("图 4　一局完整过程：信息获取与清除执行走的是同一条路线")
    fig.tight_layout()
    return 存(fig, "图4_单局全景")


# ================================================================ 图 5
def 图5_区域收缩():
    表 = 读("图_收缩曲线.csv")
    if 表 is None:
        return
    列 = "第几条有效读数" if "第几条有效读数" in 表.columns else "第几次检测"
    有 = 表[表["Rmin"].notna() & (表[列] >= 1)]
    有 = 有.sort_values([列]).groupby(["频道", 列], as_index=False).last()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, 组 in 有.groupby("频道"):
        ax[0].plot(组[列], 组["Rmin"], "o-", ms=3, lw=0.9, alpha=0.75)
    ax[0].axhline(R_CLEAR, color=主色, lw=1.6, label="清除半径 20 m")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("该频道的第几条有效示向度读数")
    ax[0].set_ylabel("定位区域最小覆盖圆半径 $R_{\\min}$ (m)")
    ax[0].set_title("(a) 每个频道的区域收缩过程", fontsize=10)
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=9)

    统 = 有.groupby(列)["Rmin"].agg(["median", "count",
                                     lambda s: s.quantile(0.9)])
    统.columns = ["中位", "样本", "p90"]
    统 = 统[统["样本"] >= 2]
    ax[1].plot(统.index, 统["中位"], "o-", color=次色, label="中位数")
    ax[1].plot(统.index, 统["p90"], "s--", color=楔形色[1], label="90 分位")
    ax[1].axhline(R_CLEAR, color=主色, lw=1.6, label="清除半径 20 m")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("该频道的第几条有效示向度读数")
    ax[1].set_ylabel("$R_{\\min}$ (m)")
    ax[1].set_title("(b) 汇总：第几条读数能压进清除半径", fontsize=10)
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=9)
    fig.suptitle("图 5　集员估计的收缩：从千米量级压进 20 m 清除半径只需三四条读数")
    fig.tight_layout()
    return 存(fig, "图5_区域收缩")


# ================================================================ 图 6
def 图6_进度曲线():
    表 = 读("图_进度曲线.csv")
    if 表 is None:
        return
    算例 = list(dict.fromkeys(表["算例"]))
    fig, ax = plt.subplots(1, len(算例), figsize=(4.2 * len(算例), 4.0),
                           sharey=False)
    ax = np.atleast_1d(ax)
    色 = {"推荐策略": 主色, "先扫后清": 次色, "逐源处理": 楔形色[1]}
    for k, 名算 in enumerate(算例):
        子 = 表[表["算例"] == 名算]
        for 名策, 组 in 子.groupby("策略"):
            组 = 组.sort_values("虚拟时刻")
            ax[k].step(组["虚拟时刻"], 组["已清除"], where="post",
                       color=色.get(名策, "0.4"), lw=1.8, label=名策)
        ax[k].set_title(f"算例 {名算}", fontsize=10)
        ax[k].set_xlabel("虚拟时刻 (s)")
        ax[k].grid(alpha=0.3)
        if k == 0:
            ax[k].set_ylabel("已清除干扰源数")
            ax[k].legend(fontsize=9)
    fig.suptitle("图 6　清除进度：推荐策略的曲线整体靠左，且没有长时间的平台段")
    fig.tight_layout()
    return 存(fig, "图6_进度曲线")


# ================================================================ 图 7
def 图7_时间去向():
    表 = 读("图_时间去向.csv")
    if 表 is None:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    项 = ["移动", "检测", "切频", "清除"]
    色 = [次色, 主色, 四色, 三色]
    底 = np.zeros(len(表))
    y = np.arange(len(表))
    for 名, c in zip(项, 色):
        ax.barh(y, 表[名], left=底, color=c, label=名, height=0.6)
        for i, (v, b) in enumerate(zip(表[名], 底)):
            if v > 120:
                ax.text(b + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="w")
        底 = 底 + 表[名].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels(表["策略"])
    ax.invert_yaxis()
    ax.set_xlabel("虚拟时间 (s)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=9, ncol=4, loc="lower right")
    for i, v in enumerate(底):
        ax.text(v + 40, i, f"{v:.0f} s", va="center", fontsize=9)
    ax.set_title("图 7　时间去向：移动始终是大头，但检测占比比单源问题高得多")
    fig.tight_layout()
    return 存(fig, "图7_时间去向")


# ================================================================ 图 8
def 图8_策略对照():
    表 = 读("策略对照汇总.csv")
    if 表 is None:
        return
    表 = 表[表["平均总时间"] < 20000].sort_values("平均总时间")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4),
                           gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(表))
    色 = [主色 if "推荐" in s else (三色 if "下界" in s else 次色)
          for s in 表["策略"]]
    ax[0].barh(y, 表["平均总时间"], color=色, height=0.6)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(表["策略"], fontsize=9)
    ax[0].invert_yaxis()
    for i, (v, r) in enumerate(zip(表["平均总时间"], 表["相对下界2"])):
        ax[0].text(v + 30, i, f"{v:.0f} s（{r:.2f}×）", va="center", fontsize=8.5)
    ax[0].set_xlabel("平均虚拟总时间 (s)")
    ax[0].grid(alpha=0.3, axis="x")
    ax[0].set_xlim(0, 表["平均总时间"].max() * 1.32)      # 给右侧数字留地方
    ax[0].set_title("(a) 平均总时间与相对下界的倍数", fontsize=10)

    子 = 表[表["检测次数"].notna()]
    ax[1].scatter(子["路程"], 子["检测次数"], s=70, c=[
        主色 if "推荐" in s else (三色 if "下界" in s else 次色)
        for s in 子["策略"]])
    for _, r in 子.iterrows():
        ax[1].annotate(r["策略"], xy=(r["路程"], r["检测次数"]),
                       xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax[1].set_xlabel("平均路程 (m)")
    ax[1].set_ylabel("平均检测次数")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("(b) 路程与检测次数的取舍", fontsize=10)
    fig.suptitle("图 8　策略对照")
    fig.tight_layout()
    return 存(fig, "图8_策略对照")


# ================================================================ 图 9
def 图9_参数灵敏():
    表 = 读("参数搜索_单扫.csv")
    if 表 is None:
        return
    参数表 = [p for p in dict.fromkeys(表["参数"]) if isinstance(p, str)]
    n = min(len(参数表), 6)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(4.0 * ((n + 1) // 2), 7.0))
    ax = np.atleast_1d(ax).ravel()
    for k, 名 in enumerate(参数表[:n]):
        子 = 表[表["参数"] == 名]
        x = np.arange(len(子))
        ax[k].plot(x, 子["平均总时间"], "o-", color=主色)
        最 = 子["平均总时间"].min()
        ax[k].axhline(最 * 1.05, color="0.6", ls=":", lw=1.0,
                      label="最优 +5%")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([str(v) for v in 子["取值"]], rotation=30,
                              fontsize=7.5, ha="right")
        ax[k].set_title(名, fontsize=10)
        ax[k].set_ylabel("平均总时间 (s)")
        ax[k].grid(alpha=0.3)
        ax[k].legend(fontsize=8)
    for k in range(n, len(ax)):
        ax[k].axis("off")
    fig.suptitle("图 9　参数单扫：虚线以下为「最优 +5%」的平坦区")
    fig.tight_layout()
    return 存(fig, "图9_参数灵敏")


# ================================================================ 图 10
def 图10_消融():
    表 = 读("消融汇总.csv")
    if 表 is None:
        return
    基 = float(表.iloc[0]["平均总时间"])
    子 = 表.iloc[1:].copy()
    子["增量"] = 子["平均总时间"] - 基
    子 = 子.sort_values("增量")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    y = np.arange(len(子))
    ax.barh(y, 子["增量"], color=[主色 if v > 0 else 三色 for v in 子["增量"]],
            height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(子["策略"], fontsize=9)
    ax.invert_yaxis()
    左, 右 = 子["增量"].min(), 子["增量"].max()
    跨 = max(右 - 左, 1.0)
    for i, v in enumerate(子["增量"]):
        # 负值的标注放在柱子右侧（靠近 0 的一边），免得压到 y 轴的中文标签上
        ax.text(v + 0.02 * 跨 if v >= 0 else v + 0.02 * 跨, i,
                f"{v:+.0f} s（{v / 基:+.1%}）",
                va="center", ha="left", fontsize=8.5)
    ax.set_xlim(左 - 0.10 * 跨, 右 + 0.34 * 跨)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel(f"关掉该设计后平均总时间的变化（完整策略 = {基:.0f} s）")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("图 10　消融实验：每个设计各自值多少秒")
    fig.tight_layout()
    return 存(fig, "图10_消融")


# ================================================================ 图 11
def 图11_分组():
    表 = 读("分组统计.csv")
    if 表 is None:
        return
    源 = 表[表["分组"] == "源数"]
    布 = 表[表["分组"] == "布局"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if len(源):
        x = 源["取值"].astype(float)
        ax[0].plot(x, 源["平均总时间"], "o-", color=次色, label="平均总时间")
        a2 = ax[0].twinx()
        a2.plot(x, 源["平均定位清除时间"], "s--", color=主色,
                label="平均定位清除时间")
        ax[0].set_xlabel("干扰源个数")
        ax[0].set_ylabel("平均总时间 (s)", color=次色)
        a2.set_ylabel("平均定位清除时间 (s/源)", color=主色)
        ax[0].grid(alpha=0.3)
        线1, 标1 = ax[0].get_legend_handles_labels()
        线2, 标2 = a2.get_legend_handles_labels()
        ax[0].legend(线1 + 线2, 标1 + 标2, fontsize=8, loc="center left")
        ax[0].set_title("(a) 源越多，单源分摊到的时间越少", fontsize=10)
    if len(布):
        x = np.arange(len(布))
        ax[1].bar(x - 0.2, 布["平均总时间"], width=0.4, color=次色,
                  label="平均总时间")
        a2 = ax[1].twinx()
        a2.bar(x + 0.2, 布["平均定位清除时间"], width=0.4, color=主色,
               label="平均定位清除时间")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(布["取值"])
        ax[1].set_ylabel("平均总时间 (s)", color=次色)
        a2.set_ylabel("平均定位清除时间 (s/源)", color=主色)
        ax[1].grid(alpha=0.3, axis="y")
        ax[1].set_title("(b) 干扰源布局的影响", fontsize=10)
        线1, 标1 = ax[1].get_legend_handles_labels()
        线2, 标2 = a2.get_legend_handles_labels()
        ax[1].legend(线1 + 线2, 标1 + 标2, fontsize=8, loc="upper left")
    fig.suptitle("图 11　分组统计")
    fig.tight_layout()
    return 存(fig, "图11_分组")


# ================================================================ 图 12
def 图12_稳健性():
    表 = 读("灵敏度汇总.csv")
    if 表 is None:
        return
    分块 = [("误差模型", ["均匀·独立（基准）", "三角分布·独立", "最坏 ±δ",
                        "空间相关 ℓ=300 m", "空间相关 ℓ=1000 m"]),
            ("有效接收半径", ["U[1000,1500]（基准）", "全部取下界 1000",
                            "全部取上界 1500", "U[1000,1200] 偏小"]),
            ("楔形半角 delta", [s for s in 表["情景"] if str(s).startswith("delta")]),
            ("R_eff 真实下界", [s for s in 表["情景"] if str(s).startswith("真实下界")])]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for k, (标题, 名单) in enumerate(分块):
        子 = 表[表["情景"].isin(名单)]
        if not len(子):
            ax[k].axis("off")
            continue
        x = np.arange(len(子))
        色 = [主色 if v < 0.9999 else 次色 for v in 子["清除率"]]
        ax[k].bar(x, 子["平均总时间"], color=色, width=0.6)
        for i, (t, cr) in enumerate(zip(子["平均总时间"], 子["清除率"])):
            ax[k].text(i, t + 40, f"{cr:.3f}", ha="center", fontsize=8,
                       color=主色 if cr < 0.9999 else "0.3")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(子["情景"], rotation=32, ha="right", fontsize=7.5)
        ax[k].set_ylabel("平均总时间 (s)")
        ax[k].set_title(标题, fontsize=10)
        ax[k].grid(alpha=0.3, axis="y")
    fig.suptitle("图 12　稳健性：柱上数字是清除率，红柱表示清除率跌破 1")
    fig.tight_layout()
    return 存(fig, "图12_稳健性")


# ================================================================ 主流程
def main():
    配置()
    print("问题三 · 绘图")
    for f in (图1_覆盖数, 图2_锚点布局, 图3_三类约束, 图4_单局全景,
              图5_区域收缩, 图6_进度曲线, 图7_时间去向, 图8_策略对照,
              图9_参数灵敏, 图10_消融, 图11_分组, 图12_稳健性):
        try:
            f()
        except Exception as e:                    # 某张图缺数据不应挡住其余的
            print(f"    [跳过] {f.__name__}：{type(e).__name__}: {e}")
    print(f"图已写入 {图目录}")


if __name__ == "__main__":
    main()
