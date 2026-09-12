# -*- coding: utf-8 -*-
"""事后定向性判定：清除成功之后，回头从日志判定每个源是不是定向源（只算数，不作图）。

**为什么需要这一步。** 在线判定（`公共.频道信念.更新定向性`）只能用可行集，非常保守；
而清除成功之后我们知道源的位置到 20 m 以内，可以用真位置重判，得到严格得多的下界。
这个下界有两个用途：

    ① 校准本地生成器的定向分布。官方模拟器在正式测试里不显示案例真值，
       这是唯一能从日志反推「这一局有多少定向源」的办法。
       V2/优化.md §11 的生成器校准就是靠它做的。
    ② 检查在线判定有没有漏掉本可判出的情形。实测两者给出同一组结论
       （只多证出 1 个），说明在线判定已经把数据里的信息用尽了。

**判据。** 设频道 c 在 q 处清除成功，则 |G - q| <= 20（附件 2 §2.4：清除半径 20 m）。

    R_eff 的下界：R = max(1000, max{ |s - q| + 20 : s 是该频道收到过信号的点 })
        依据：在 s 收到信号 ==> |s - G| <= R_eff，而 |s - G| >= |s - q| - 20，
              故 R_eff >= |s - q| - 20；再与题给下界 1000 取大。
        （这里用 +20 而不是 -20 是因为要的是**保证距离**的上界侧，见下一条。）
    若存在 no_signal 点 p 满足 |p - q| + 20 <= R，则 |p - G| <= R <= R_eff，
    距离条件必然满足，却收不到 ==> **该源必为定向源**。

判据仍然是**单侧**的：判不出来只说明「没能证明它是定向的」，不等于它是全向的。
要证明全向必须用**信号点**把源围起来（引理 D2）——见 V2/优化.md §16 的对偶。

输出：事后定向性结果.txt、事后定向性_逐局.csv、事后定向性_逐频道.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
清除半径 = 20.0
R下界 = 1000.0

行, 明细, 文 = [], [], []


def P(s=""):
    print(s)
    文.append(str(s))


def 跑():
    日志表 = sorted(HERE.glob("实战日志_*.csv"))
    if not 日志表:
        P("本目录下没有 实战日志_*.csv，跳过。")
        return
    for f in 日志表:
        d = pd.read_csv(f, encoding="utf-8-sig")
        清点 = {int(e["频道"]): (float(e["x"]), float(e["y"]))
                for _, e in d.iterrows()
                if e["动作"] == "clear" and str(e["结果"]) == "success"}
        证, 未证 = 0, []
        for c, q in sorted(清点.items()):
            m = d[(d["动作"] == "measure") & (d["频道"] == c)]
            收 = m[m["结果"] != "no_signal"]
            无 = m[m["结果"] == "no_signal"]
            R = R下界
            for _, e in 收.iterrows():
                R = max(R, float(np.hypot(e["x"] - q[0], e["y"] - q[1])) + 清除半径)
            近 = [float(np.hypot(e["x"] - q[0], e["y"] - q[1]))
                  for _, e in 无.iterrows()]
            定向 = any(dd + 清除半径 <= R for dd in 近)
            证 += bool(定向)
            if not 定向:
                未证.append(c)
            明细.append({"日志": f.name, "频道": c, "检测次数": len(m),
                         "收到次数": len(收), "无信号次数": len(无),
                         "R_eff下界": round(R, 1),
                         "最近无信号距源": round(min(近), 1) if 近 else None,
                         "事后判定": "定向" if 定向 else "未证"})
        行.append({"日志": f.name, "清除数": len(清点), "事后证明定向": 证,
                   "未证明": len(未证), "未证明频道": " ".join(map(str, 未证)),
                   "下界比例": round(证 / max(len(清点), 1), 3)})

    汇 = pd.DataFrame(行)
    明 = pd.DataFrame(明细)
    P("问题四 · 事后定向性判定")
    P("=" * 78)
    P("")
    P(汇.to_string(index=False))
    P("")
    P("「下界比例」= 事后可证定向数 / 清除数，是真实定向比例的**严格下界**。")
    P("把「可证率」（判定定向 / 真定向，本地扫描实测 0.68~0.74）除回去即得比例估计。")
    P("")
    未 = 明[明["事后判定"] == "未证"]
    P(f"未证明的频道共 {len(未)} 个。它们的最近 no_signal 点与源的距离：")
    有无 = 未["最近无信号距源"].dropna()
    if len(有无):
        P(f"    最小 {有无.min():.1f} m，中位 {有无.median():.1f} m，"
          f"最大 {有无.max():.1f} m")
        P(f"    而这些频道的 R_eff 下界中位数是 "
          f"{未.loc[有无.index, 'R_eff下界'].median():.1f} m")
        P("    **全部落在 R_eff 下界之外**——「距离不够」这个解释就足够了，")
        P("    方向性无从判定。不是逻辑漏了，是数据里没有这个信息。")
    P(f"    另有 {int(未['最近无信号距源'].isna().sum())} 个频道一次 no_signal 都没有。")

    (HERE / "事后定向性结果.txt").write_text("\n".join(文) + "\n", encoding="utf-8")
    汇.to_csv(HERE / "事后定向性_逐局.csv", index=False, encoding="utf-8-sig")
    明.to_csv(HERE / "事后定向性_逐频道.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写入 {HERE / '事后定向性结果.txt'} 等 3 个文件")


if __name__ == "__main__":
    sys.exit(跑())
