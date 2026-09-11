"""问题三 · 作图数据（只算数，不作图）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import 公共 as C
import 模拟器 as S
import 生成算例 as G
import 策略 as ST

HERE = Path(__file__).resolve().parent
SEED = 20260911


# ================================================================ 单局过程
def 跑一局带记录(案, 策略函数=ST.推荐策略, 参数=None):
    """跑一局，同时记录每个频道每次检测之后的 Rmin，以及完整轨迹。"""
    收缩 = []
    有效计数 = {}
    原记录检测 = ST.状态.记录检测

    def 记录检测(self, p, c, 回):
        原记录检测(self, p, c, 回)
        m = self.信念[c].指标()
        # 横坐标用「第几条**有效**读数」而不是「第几次检测」：no_signal 不给方位，
        if 回["measure_result"] in ("direction", "near"):
            有效计数[c] = 有效计数.get(c, 0) + 1
        收缩.append({"频道": c, "第几次检测": self.信念[c].检测次数,
                     "第几条有效读数": 有效计数.get(c, 0),
                     "结果": 回["measure_result"],
                     "直径": (m or {}).get("直径", np.nan),
                     "Rmin": (m or {}).get("Rmin", np.nan),
                     "残余面积": self.栅格.面积(self.掩码[c])})

    ST.状态.记录检测 = 记录检测
    try:
        后端 = S.本地后端(案)
        狗 = S.机器狗(后端, 现实时限=1e9)
        r = 策略函数(狗, 参数)
    finally:
        ST.状态.记录检测 = 原记录检测
    return 狗, 后端, r, pd.DataFrame(收缩)


def 轨迹表(狗):
    行 = []
    位置 = np.zeros(2)
    for i, 项 in enumerate(狗.日志):
        if 项["动作"] in ("enter", "exit"):
            continue
        p = np.array([项["x"], 项["y"]], dtype=float)
        行.append({"序号": i, "动作": 项["动作"], "x": p[0], "y": p[1],
                   "频道": 项["频道"], "结果": 项["结果"],
                   "虚拟时刻": 项["虚拟时刻"],
                   "移动距离": float(np.hypot(*(p - 位置)))})
        位置 = p
    return pd.DataFrame(行)


def 停靠表(轨):
    """把连续的同位置指令合并成一次停靠。"""
    行 = []
    当前 = None
    for _, r in 轨.iterrows():
        键 = (round(r["x"], 3), round(r["y"], 3))
        if 当前 is None or 当前["键"] != 键:
            if 当前 is not None:
                行.append(当前)
            当前 = {"键": 键, "x": r["x"], "y": r["y"], "检测数": 0,
                    "清除数": 0, "到达时刻": r["虚拟时刻"], "移动距离": r["移动距离"]}
        当前["检测数"] += int(r["动作"] == "measure")
        当前["清除数"] += int(r["动作"] == "clear")
    if 当前 is not None:
        行.append(当前)
    表 = pd.DataFrame(行).drop(columns=["键"])
    表["类型"] = np.where(表["检测数"] >= 4, "扫描停靠", "补测停靠")
    表["含清除"] = 表["清除数"] > 0
    return 表


def 进度曲线(狗, 名):
    行, n = [], 0
    for 项 in 狗.日志:
        if 项["动作"] == "clear" and 项["结果"] == "success":
            n += 1
            行.append({"策略": 名, "虚拟时刻": 项["虚拟时刻"], "已清除": n})
    if 行:
        行.insert(0, {"策略": 名, "虚拟时刻": 0.0, "已清除": 0})
        行.append({"策略": 名, "虚拟时刻": 狗.虚拟时刻, "已清除": n})
    return 行


# ================================================================ 约束示意
def 约束示意():
    """一个人造小场景，把三类约束画在一张图上。"""
    G0 = np.array([120.0, 60.0])
    A, B, D = np.array([-600.0, -400.0]), np.array([500.0, -500.0]), \
              np.array([-200.0, 1200.0])
    段, 半平面 = [], []
    信念 = C.频道信念(1)
    for 名, p in [("S1", A), ("S2", B)]:
        θ = C.方位角(p, G0)
        信念.记录检测(p, "direction", round(θ, 2))
        for 记号, dθ in [("中线", 0.0), ("下界", -C.DELTA_算法),
                         ("上界", +C.DELTA_算法)]:
            u = C.单位向量(θ + dθ)
            段.append({"元素": f"楔形{名}", "线": 记号,
                       "x1": p[0], "y1": p[1],
                       "x2": p[0] + 2600 * u[0], "y2": p[1] + 2600 * u[1]})
    信念.记录检测(D, "no_signal")
    m = 信念.指标()
    顶点 = pd.DataFrame(m["多边形"], columns=["x", "y"])
    # 序约束：|x-A| <= |x-D| 的边界（AD 的中垂线）
    中 = 0.5 * (A + D)
    方向 = D - A
    方向 = np.array([-方向[1], 方向[0]]) / float(np.hypot(*方向))
    段.append({"元素": "序约束中垂线", "线": "边界",
               "x1": 中[0] - 2600 * 方向[0], "y1": 中[1] - 2600 * 方向[1],
               "x2": 中[0] + 2600 * 方向[0], "y2": 中[1] + 2600 * 方向[1]})
    点 = pd.DataFrame([
        {"元素": "检测点S1", "x": A[0], "y": A[1], "参数": 0.0},
        {"元素": "检测点S2", "x": B[0], "y": B[1], "参数": 0.0},
        {"元素": "无信号点D", "x": D[0], "y": D[1], "参数": C.R_RECV_MIN},
        {"元素": "真源G", "x": G0[0], "y": G0[1], "参数": 0.0},
    ])
    return pd.DataFrame(段), 点, 顶点, m


# ================================================================ 主流程
def main():
    print("问题三 · 作图数据")
    算例 = G.具名算例()
    案A = 算例["A"]

    # ---------- 单局过程（算例 A） ----------
    狗, 后端, r, 收缩 = 跑一局带记录(案A)
    pd.DataFrame(案A.源表).to_csv(HERE / "图_算例A_干扰源.csv", index=False,
                                  encoding="utf-8-sig")
    轨 = 轨迹表(狗)
    轨.to_csv(HERE / "图_算例A_轨迹.csv", index=False, encoding="utf-8-sig")
    停 = 停靠表(轨)
    停.to_csv(HERE / "图_算例A_停靠.csv", index=False, encoding="utf-8-sig")
    收缩.to_csv(HERE / "图_收缩曲线.csv", index=False, encoding="utf-8-sig")
    print(f"    算例 A：清除 {r['清除成功']}/{案A.源数}，"
          f"虚拟总时间 {r['虚拟总时间']:.1f} s，"
          f"停靠 {len(停)} 次，路程 {后端.路程:.0f} m")

    # ---------- 进度曲线（三策略 x 三算例） ----------
    行 = []
    for 名算, 案 in 算例.items():
        for 名策, fn in [("推荐策略", ST.推荐策略),
                         ("先扫后清", ST.基线_先扫后清),
                         ("逐源处理", ST.基线_逐源)]:
            狗2 = S.机器狗(S.本地后端(案), 现实时限=1e9)
            fn(狗2)
            for x in 进度曲线(狗2, 名策):
                x["算例"] = 名算
                行.append(x)
    pd.DataFrame(行).to_csv(HERE / "图_进度曲线.csv", index=False,
                            encoding="utf-8-sig")
    print(f"    进度曲线 {len(行)} 行")

    # ---------- 约束示意 ----------
    段, 点, 顶点, m = 约束示意()
    段.to_csv(HERE / "图_约束示意.csv", index=False, encoding="utf-8-sig")
    点.to_csv(HERE / "图_约束示意点.csv", index=False, encoding="utf-8-sig")
    顶点.to_csv(HERE / "图_定位区域.csv", index=False, encoding="utf-8-sig")
    print(f"    约束示意：定位区域 {len(顶点)} 个顶点，"
          f"直径 {m['直径']:.2f} m，Rmin {m['Rmin']:.2f} m")

    # ---------- 时间去向 ----------
    样本 = [G.造案例(np.random.default_rng(SEED + 900 + i)) for i in range(12)]
    行 = []
    for 名, fn in [("推荐策略", ST.推荐策略), ("先扫后清", ST.基线_先扫后清),
                   ("逐源处理", ST.基线_逐源)]:
        计 = {"移动": 0.0, "检测": 0.0, "切频": 0.0, "清除": 0.0}
        for 案 in 样本:
            后端2 = S.本地后端(案)
            狗2 = S.机器狗(后端2, 现实时限=1e9)
            fn(狗2)
            for k in 计:
                计[k] += 后端2.计时[k] / len(样本)
        行.append({"策略": 名, **计})
    下 = pd.DataFrame([ST.理论下界(案) for 案 in 样本]).mean()
    行.append({"策略": "理论下界", "移动": 下["含判空TSP路程"] / C.V_MOVE,
               "检测": 下["判空检测时间"] * 5 / 6, "切频": 下["判空检测时间"] / 6,
               "清除": 下["清除时间"]})
    pd.DataFrame(行).to_csv(HERE / "图_时间去向.csv", index=False,
                            encoding="utf-8-sig")
    print("    时间去向：")
    for x in 行:
        print(f"        {x['策略']:<10} 移动 {x['移动']:7.1f}  检测 {x['检测']:7.1f}"
              f"  切频 {x['切频']:6.1f}  清除 {x['清除']:6.1f}")

    print("\n已写入 图_*.csv")


if __name__ == "__main__":
    with open(HERE / "作图数据结果.txt", "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = 原
    print(f"已写入 {HERE / '作图数据结果.txt'}")
