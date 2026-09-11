"""本地模拟评测台：同一批算例上对照多个策略，给出配对统计（只算数，不作图）。

问题三与问题四共用。它不是「论文对照表」的生成器（那是各题自己的 策略对照.py），
而是**优化过程中的快速判据**：改一版策略，跑一次，二十几秒知道有没有变好。

    python 评测.py                          # 默认：P3 的 V1 与 V2，40 局
    python 评测.py 推荐策略 推荐策略V2       # 同上，写全名
    python 评测.py V1 V2 --局数 400          # 跑 400 局（约 4 分钟）
    python 评测.py V1 V2 --快                # 16 局，约 10 秒，改完先看一眼
    python 评测.py V1 --目录 P4              # 只评一个策略
    python 评测.py P4:策略.推荐策略 P4:策略V2.推荐策略 --输出 P4/V2

判据分两层，**硬判据不过就直接判负，不看时间**：

    硬判据   每局清除率 = 1.0000；退出时未决频道 = 0；异常数 = 0
             「未决」指既没清除、也没判空的频道。它比清除率严格：
             正式测试那次缺陷就是清除率 1.0000 而 41.7% 的局没拿到完备性证书。
    软判据   同一批算例（公共随机数）+ 逐局配对 t 检验，|t| < 2 一律按「没区别」。
             单局总时间标准差约 240 s，不配对连 100 s 的差异都分辨不出来。

默认**不跑基线、不算全知参照与严格下界**（要多花一倍时间，且优化过程中用不上）。
定版时加 --带参照 才算，或直接跑各题自己的 策略对照.py。

--------------------------------------------------------------------------------
策略怎么被找到
--------------------------------------------------------------------------------
策略标识写成 `目录:模块.函数`，目录相对本文件所在的 notebooks/：

    P3:策略.推荐策略            notebooks/P3/策略.py 里的 推荐策略
    P3/V2:策略V2.推荐策略       notebooks/P3/V2/策略V2.py 里的 推荐策略

省略目录时用 --目录（默认 P3）。另有四个简称：

    V1 / 推荐策略     ->  <目录>:策略.推荐策略
    V2 / 推荐策略V2   ->  <目录>/V2:策略V2.推荐策略

每个策略单开一个进程池跑完再换下一个。这不是为了省事——P3 与 P4 各有自己的
`公共.py`、`策略.py`，同名模块在一个进程里会互相遮住，只有进程隔离才能同场对照。

--------------------------------------------------------------------------------
策略模块要满足的接口（P4 照此实现即可复用本脚本）
--------------------------------------------------------------------------------
算例目录（默认取第一个策略的目录）下需要有：

    模拟器.本地后端(案例)            -> 后端
    模拟器.机器狗(后端, 现实时限=x)  -> 狗；.日志 是逐条指令记录（含 x, y）
    生成算例.造案例(rng, **kw)       -> 案例；.源数、.频道集()
    公共.频道数                      -> 20

策略函数 `f(狗, 参数=None)` 返回 dict，至少含：

    虚拟总时间, 检测次数, 清除次数, 清除成功, 异常(list), 状态
    状态.已清除（可迭代的频道集合）、状态.判空集() -> set

--------------------------------------------------------------------------------
输出
--------------------------------------------------------------------------------
屏幕打两张表（各策略汇总、相对第一个策略的配对差），同时写三个 CSV 到 --输出：

    评测_逐局.csv   每局一行，可用来复算任何统计量
    评测_汇总.csv   每策略一行
    评测_配对.csv   每个对照一行：配对差、标准误、t、改善局数

这三个文件就是「我们确实做了优化、跑了多少局、拿到什么结果」的凭据，
可以直接进支撑材料。
"""
import argparse
import importlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # notebooks/
默认种子 = 20260911
进程数 = int(os.environ.get("EVAL_PROCS") or max(1, (os.cpu_count() or 4) // 3))

简称 = {"V1": "{d}:策略.推荐策略", "推荐策略": "{d}:策略.推荐策略",
        "V2": "{d}/V2:策略V2.推荐策略", "推荐策略V2": "{d}/V2:策略V2.推荐策略"}


def 报(*a):
    print(*a, file=sys.stderr, flush=True)


def 解析(标识, 默认目录):
    """'P3/V2:策略V2.推荐策略' -> (目录, 模块, 函数)。"""
    if 标识 in 简称:
        标识 = 简称[标识].format(d=默认目录)
    目录, _, 尾 = 标识.rpartition(":")
    目录 = 目录 or 默认目录
    模块, _, 函数 = 尾.rpartition(".")
    if not 模块:
        模块, 函数 = 尾, "推荐策略"
    return 目录, 模块, 函数


def _装载(目录, 模块):
    """把目录（及其父目录，便于 V2 复用上一级的 公共.py）放进 sys.path 再导入。"""
    d = (HERE / 目录).resolve()
    for p in (str(d.parent), str(d)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(模块)


def 跑一局(任务):
    """子进程入口。所有随机性只来自 任务['种子']，任一局都能独立重建。"""
    目录, 模块名, 函数名 = 任务["目录"], 任务["模块"], 任务["函数"]
    算例目录 = 任务["算例目录"]
    C = _装载(算例目录, "公共")
    S = _装载(算例目录, "模拟器")
    G = _装载(算例目录, "生成算例")
    f = getattr(_装载(目录, 模块名), 函数名)

    案 = G.造案例(np.random.default_rng(任务["种子"]), **任务.get("造案", {}))
    狗 = S.机器狗(S.本地后端(案), 现实时限=1e9)
    t0 = time.perf_counter()
    r = f(狗, 任务.get("参数") or None)
    耗时 = time.perf_counter() - t0

    st = r["状态"]
    已清 = set(st.已清除)
    判空 = set(st.判空集())
    xy = np.array([[e["x"], e["y"]] for e in 狗.日志
                   if e["动作"] in ("measure", "clear")], dtype=float)
    路程 = float(np.hypot(*np.diff(np.vstack([[0.0, 0.0], xy]), axis=0).T).sum()) \
        if len(xy) else 0.0
    行 = {"策略": 任务["名"], "局号": 任务["局号"], "种子": 任务["种子"],
          "源数": 案.源数, "空频道": C.频道数 - 案.源数,
          "清除成功": r["清除成功"], "清除率": r["清除成功"] / max(案.源数, 1),
          "未决频道": C.频道数 - len(已清) - len(判空),
          "漏清除": len(set(案.频道集()) - 已清),
          "判空误判": len(判空 & set(案.频道集())),
          "虚拟总时间": r["虚拟总时间"], "检测次数": r["检测次数"],
          "清除次数": r["清除次数"], "路程": 路程,
          "异常数": len(r["异常"]), "程序耗时": 耗时}
    if 任务.get("带参照"):
        ST = _装载(目录, 模块名)
        下 = (getattr(ST, "理论下界", None) or
              getattr(_装载(算例目录, "策略"), "理论下界"))(案)
        行["严格下界"] = 下.get("严格下界", np.nan)
        行["全知参照"] = 下.get("下界2总时间", np.nan)
    return 行


def 并行跑(任务表, 说明, 进程):
    t0 = time.time()
    出, n, 完成 = [None] * len(任务表), len(任务表), 0
    with ProcessPoolExecutor(max_workers=进程) as 池:
        期 = {池.submit(跑一局, t): i for i, t in enumerate(任务表)}
        for f in as_completed(期):
            出[期[f]] = f.result()
            完成 += 1
            if 完成 % max(1, n // 10) == 0 or 完成 == n:
                用 = time.time() - t0
                报(f"    [{说明}] {完成}/{n}  已用 {用:.0f}s  "
                   f"预计还需 {用 / 完成 * (n - 完成):.0f}s")
    return 出


def 汇总(d):
    g = d.groupby("策略", sort=False)
    出 = pd.DataFrame({
        "局数": g.size(),
        "清除率": g["清除率"].mean(),
        "全清局占比": g["清除率"].apply(lambda s: float((s >= 1.0).mean())),
        "未决频道": g["未决频道"].sum(),
        "漏清除": g["漏清除"].sum(),
        "判空误判": g["判空误判"].sum(),
        "异常": g["异常数"].sum(),
        "平均总时间": g["虚拟总时间"].mean(),
        "中位": g["虚拟总时间"].median(),
        "p90": g["虚拟总时间"].quantile(0.9),
        "检测次数": g["检测次数"].mean(),
        "路程": g["路程"].mean(),
        "程序耗时": g["程序耗时"].mean(),
    }).reset_index()
    出["合格"] = ((出["清除率"] >= 1.0) & (出["未决频道"] == 0) &
                  (出["异常"] == 0))
    return 出


def 配对(d, 名表):
    """逐局配对差：每个策略减去第一个策略。局号对齐，缺局不参与。"""
    基名 = 名表[0]
    宽 = d.pivot(index="局号", columns="策略", values="虚拟总时间")
    行 = []
    for 名 in 名表[1:]:
        两 = 宽[[基名, 名]].dropna()
        差 = (两[名] - 两[基名]).to_numpy()
        n = len(差)
        se = float(差.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        行.append({"对照": f"{名} - {基名}", "局数": n,
                   "配对差": float(差.mean()), "标准误": se,
                   "t": float(差.mean() / se) if se and se == se else float("nan"),
                   "改善局数": int((差 < 0).sum()),
                   "相对": float(差.mean() / 两[基名].mean())})
    return pd.DataFrame(行)


def 打印汇总(s):
    print(f"\n{'策略':<14}{'局数':>5}{'清除率':>9}{'未决':>6}{'漏清':>6}"
          f"{'误判':>6}{'异常':>6}{'平均时间':>10}{'中位':>9}{'p90':>9}"
          f"{'检测':>7}{'路程':>9}{'判定':>7}")
    print("-" * 110)
    for _, r in s.iterrows():
        print(f"{r['策略']:<14}{r['局数']:>5d}{r['清除率']:>9.4f}"
              f"{r['未决频道']:>6d}{r['漏清除']:>6d}{r['判空误判']:>6d}"
              f"{r['异常']:>6d}{r['平均总时间']:>10.1f}{r['中位']:>9.1f}"
              f"{r['p90']:>9.1f}{r['检测次数']:>7.1f}{r['路程']:>9.0f}"
              f"{'合格' if r['合格'] else '**判负**':>7}")


def 打印配对(p):
    if not len(p):
        return
    print(f"\n{'对照':<26}{'局数':>5}{'配对差(s)':>12}{'标准误':>9}{'t':>8}"
          f"{'改善局数':>9}{'相对':>9}  判读")
    print("-" * 100)
    for _, r in p.iterrows():
        判 = ("**显著变好**" if r["t"] <= -2 else
              "**显著变差**" if r["t"] >= 2 else "无区别(|t|<2)")
        print(f"{r['对照']:<26}{r['局数']:>5d}{r['配对差']:>12.1f}"
              f"{r['标准误']:>9.1f}{r['t']:>8.2f}{r['改善局数']:>9d}"
              f"{r['相对']:>8.1%}  {判}")


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True, description="本地模拟评测台")
    ap.add_argument("策略", nargs="*", default=["V1", "V2"])
    ap.add_argument("--目录", default="P3", help="策略标识省略目录时的默认目录")
    ap.add_argument("--算例目录", default=None, help="造算例用哪套模块，默认同第一个策略")
    ap.add_argument("--局数", type=int, default=None)
    ap.add_argument("--快", action="store_true", help="16 局，约 10 秒")
    ap.add_argument("--种子", type=int, default=默认种子)
    ap.add_argument("--输出", default=None, help="CSV 落盘目录，默认第一个策略的目录")
    ap.add_argument("--进程", type=int, default=进程数)
    ap.add_argument("--带参照", action="store_true", help="另算严格下界与全知参照（慢一倍）")
    ap.add_argument("--标签", default=None,
                    help="给本次实验起名，CSV 落成 评测_逐局_<标签>.csv，不覆盖上一次")
    ap.add_argument("--布局", default=None, help="全部算例固定用某种布局")
    ap.add_argument("--源数", type=int, default=None, help="全部算例固定源数")
    a = ap.parse_args(argv)

    局数 = a.局数 if a.局数 else (16 if a.快 else 40)
    规格 = [(名, *解析(名, a.目录)) for 名 in a.策略]
    算例目录 = a.算例目录 or 规格[0][1]
    造案 = {}
    if a.布局:
        造案["布局"] = a.布局
    if a.源数:
        造案["源数"] = a.源数
    出目录 = Path(a.输出) if a.输出 else (HERE / 规格[-1][1])
    if not 出目录.is_absolute():
        出目录 = HERE / 出目录
    出目录.mkdir(parents=True, exist_ok=True)

    print("本地模拟评测台")
    print(f"算例：{算例目录}，种子 {a.种子}，{局数} 局"
          f"{'，' + str(造案) if 造案 else ''}；{a.进程} 进程")
    for 名, d, m, f in 规格:
        print(f"策略：{名:<12} -> {d}/{m}.py :: {f}")

    行 = []
    for 名, d, m, f in 规格:
        任务 = [{"名": 名, "目录": d, "模块": m, "函数": f, "算例目录": 算例目录,
                 "局号": i, "种子": a.种子 + i, "造案": 造案,
                 "带参照": a.带参照} for i in range(局数)]
        行 += 并行跑(任务, 名, a.进程)
    d = pd.DataFrame(行)
    s = 汇总(d)
    p = 配对(d, [g[0] for g in 规格])
    打印汇总(s)
    打印配对(p)

    坏 = s[~s["合格"]]["策略"].tolist()
    if 坏:
        print(f"\n**硬判据不过：{坏}**。清除率必须 1.0000、未决频道与异常必须为 0，"
              "不满足就不必比时间。")
    else:
        print("\n全部策略通过硬判据（清除率 1.0000、未决 0、异常 0）。")

    尾 = f"_{a.标签}" if a.标签 else ""
    for 名, 表 in [("逐局", d), ("汇总", s), ("配对", p)]:
        表.to_csv(出目录 / f"评测_{名}{尾}.csv", index=False, encoding="utf-8-sig")
    print(f"已写入 {出目录 / f'评测_逐局{尾}.csv'} 等 3 个文件")
    return 1 if 坏 else 0


if __name__ == "__main__":
    sys.exit(main())
