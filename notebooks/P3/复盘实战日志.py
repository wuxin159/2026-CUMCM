"""问题三 · 把实战日志喂回信念模型，复盘每一局到底停在了什么状态（只算数，不作图）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 公共 as C
import 策略 as ST

HERE = Path(__file__).resolve().parent


def 核对计时(日志路径):
    """用附件 1 表 1、附件 2 §4 的规则独立重算虚拟时钟，与官方模拟器逐条比对。"""
    日志 = pd.read_csv(日志路径, encoding="utf-8-sig")
    t, 位置, 频道 = 0.0, np.zeros(2), 1
    最大差, 分项 = 0.0, {"移动": 0.0, "检测": 0.0, "切频": 0.0, "清除": 0.0}
    for _, 行 in 日志.iterrows():
        if 行["动作"] in ("enter", "exit"):
            continue
        p = np.array([float(行["x"]), float(行["y"])])
        走 = float(np.hypot(*(p - 位置))) / C.V_MOVE
        t += 走
        分项["移动"] += 走
        位置 = p
        if 行["动作"] == "measure":
            if int(行["频道"]) != 频道:
                t += C.T_SWITCH
                分项["切频"] += C.T_SWITCH
                频道 = int(行["频道"])
            t += C.T_MEASURE
            分项["检测"] += C.T_MEASURE
        else:
            dt = C.T_CLEAR_OK if 行["结果"] == "success" else C.T_CLEAR_FAIL
            t += dt
            分项["清除"] += dt
        最大差 = max(最大差, abs(t - float(行["虚拟时刻"])))
    return 最大差, 分项, float(日志["虚拟时刻"].max())


def 复盘(日志路径, 参数=None):
    参数 = dict(ST.默认参数, **(参数 or {}))
    日志 = pd.read_csv(日志路径, encoding="utf-8-sig")
    st = ST.状态(参数)
    虚拟时刻 = 0.0
    检测 = 清除 = 成功 = 0
    for _, 行 in 日志.iterrows():
        动作 = 行["动作"]
        if 动作 in ("enter", "exit"):
            虚拟时刻 = float(行["虚拟时刻"])
            continue
        p = (float(行["x"]), float(行["y"]))
        c = int(行["频道"])
        虚拟时刻 = float(行["虚拟时刻"])
        if 动作 == "measure":
            检测 += 1
            回 = {"measure_result": 行["结果"]}
            if 行["结果"] == "direction":
                回["svd_deg"] = float(行["示向度"])
            st.记录检测(p, c, 回)
        else:
            清除 += 1
            回 = {"clear_result": 行["结果"]}
            成功 += int(行["结果"] == "success")
            st.记录清除(p, c, 回)

    判空 = st.判空集()
    未决 = [c for c in range(1, C.频道数 + 1)
            if c not in st.已清除 and c not in 判空]
    行表 = []
    for c in range(1, C.频道数 + 1):
        b = st.信念[c]
        m = b.指标() if b.收到信号 else None
        剩格 = int(np.count_nonzero(st.掩码[c]))
        行表.append({
            "频道": c,
            "状态": ("已清除" if c in st.已清除 else
                     "判空" if c in 判空 else
                     "未决·已探测" if b.收到信号 else "未决·未探测"),
            "检测次数": b.检测次数,
            "收到信号": b.收到信号,
            "判空残余格数": 剩格,
            "判空残余面积": 剩格 * st.栅格.单元面积,
            "区域直径": (m or {}).get("直径", np.nan),
            "区域Rmin": (m or {}).get("Rmin", np.nan),
        })
    明细 = pd.DataFrame(行表)

    # 规划器只看 40 m 子点阵，判空判据看完整 20 m 栅格——这两者不一致时会卡死
    联合 = st.残余联合()
    子残 = st.规划器.残余数(联合)
    全残 = int(np.count_nonzero(联合))
    return {
        "日志": Path(日志路径).name,
        "指令数": len(日志), "检测次数": 检测, "清除指令": 清除, "清除成功": 成功,
        "虚拟总时间": 虚拟时刻,
        "平均定位清除时间": 虚拟时刻 / 成功 if 成功 else np.nan,
        "已清除频道数": len(st.已清除), "判空频道数": len(判空),
        "未决频道数": len(未决), "未决频道": " ".join(str(c) for c in 未决),
        "退出时已全部解决": st.全部解决(),
        "残余格数_完整栅格": 全残, "残余格数_规划子点阵": 子残,
        "残余面积": 全残 * st.栅格.单元面积,
    }, 明细


def main(argv):
    文件 = [a for a in argv[1:] if not a.startswith("--")]
    if not 文件:
        文件 = sorted(str(p) for p in HERE.glob("实战日志_*.csv"))
    if not 文件:
        print("本目录下没有 实战日志_*.csv")
        return
    print("问题三 · 实战日志复盘")
    print("把日志里的指令与响应按原顺序重放进信念模型，重建退出时刻的状态。\n")
    汇总, 明细表 = [], []
    for f in 文件:
        s, 明细 = 复盘(f)
        汇总.append(s)
        明细["日志"] = s["日志"]
        明细表.append(明细)
    df = pd.DataFrame(汇总)
    pd.set_option("display.width", 240)
    pd.set_option("display.unicode.east_asian_width", True)
    print("=" * 120)
    print("一、每局的收尾状态")
    print("=" * 120)
    print(df[["日志", "指令数", "检测次数", "清除成功", "虚拟总时间",
              "平均定位清除时间", "已清除频道数", "判空频道数", "未决频道数",
              "退出时已全部解决"]].round(1).to_string(index=False))
    print()
    print(f"    {'日志':<28} {'未决频道':<22} {'残余面积(m²)':>14} "
          f"{'完整栅格残余格':>15} {'规划子点阵残余':>15}")
    for s in 汇总:
        print(f"    {s['日志']:<28} {s['未决频道'] or '（无）':<22} "
              f"{s['残余面积']:>14.0f} {s['残余格数_完整栅格']:>15d} "
              f"{s['残余格数_规划子点阵']:>15d}")
    print()
    print("    「判空频道数 + 已清除频道数 = 20」才说明退出时每个频道都有依据。")
    print("    若出现未决频道，说明策略是**被迫**退出的，不是解完退出的。")

    print("\n" + "=" * 120)
    print("二、用官方模拟器反验本地复刻件的计时规则")
    print("=" * 120)
    print("    按附件 1 表 1、附件 2 §4 独立重算虚拟时钟，与日志里模拟器给的逐条比对。")
    print(f"    {'日志':<28} {'最大差(s)':>10} {'移动':>9} {'检测':>9} "
          f"{'切频':>8} {'清除':>8} {'合计':>9} {'模拟器':>9}")
    for f in 文件:
        差, 分项, 末 = 核对计时(f)
        合 = sum(分项.values())
        print(f"    {Path(f).name:<28} {差:>10.2e} {分项['移动']:>9.1f} "
              f"{分项['检测']:>9.1f} {分项['切频']:>8.1f} {分项['清除']:>8.1f} "
              f"{合:>9.1f} {末:>9.1f}")
    print("    最大差为 0 说明：本地复刻件的代价模型与官方模拟器**完全一致**，")
    print("    离线那几百局的时间统计因此可以直接与正式测试对照。")

    print("\n" + "=" * 120)
    print("三、逐频道明细")
    print("=" * 120)
    全 = pd.concat(明细表, ignore_index=True)
    for f, 组 in 全.groupby("日志", sort=False):
        print(f"\n{f}")
        print(组[["频道", "状态", "检测次数", "判空残余格数", "区域直径",
                  "区域Rmin"]].round(2).to_string(index=False))

    df.to_csv(HERE / "实战复盘汇总.csv", index=False, encoding="utf-8-sig")
    全.to_csv(HERE / "实战复盘_频道明细.csv", index=False, encoding="utf-8-sig")
    print("\n已写入 实战复盘汇总.csv、实战复盘_频道明细.csv")


if __name__ == "__main__":
    with open(HERE / "实战复盘结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main(sys.argv)
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '实战复盘结果.txt'}")
