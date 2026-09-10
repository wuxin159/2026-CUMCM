"""问题二 · 按依赖顺序重跑全部脚本，验证可复现性。

    python 重跑全部.py

所有脚本都固定随机种子、写死扫描参数，换台机器重跑应得到完全相同的结果文件。
顺序不能乱：候选区域.py 的网格是 阶段数据.py 与 绘图.py 的输入。
整轮约 20 分钟。
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

顺序 = [
    ("理论验证.py", "阶段 1：闭式公式与几何结论的逐条验证"),
    ("候选区域.py", "阶段 1-2：全平面扫描，两个口径的等值面"),
    ("阶段数据.py", "阶段 2-3：效率前沿、边际账、途中加测"),
    ("策略仿真.py", "阶段 4：端到端仿真与参数搜索"),
    ("策略对照.py", "阶段 4：上帝视角下界、盲扫基线、边界验证"),
    ("独立复核.py", "阶段 5：从题面定义重推 + 变异测试"),
    ("作图数据.py", "导出作图用的几何量"),
    ("绘图.py", "十张矢量图，每图一个 PDF"),
]


def main():
    print(f"依次重跑 {len(顺序)} 个脚本，工作目录 {HERE}\n")
    起 = time.time()
    for 名, 说明 in 顺序:
        t0 = time.time()
        print(f"[{名:<16}] {说明} ...", flush=True)
        r = subprocess.run([sys.executable, 名], cwd=HERE,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if r.returncode != 0:
            print(f"  失败（退出码 {r.returncode}）")
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            return 1
        print(f"  完成，用时 {time.time()-t0:.0f} s")
    print(f"\n全部完成，总用时 {time.time()-起:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
