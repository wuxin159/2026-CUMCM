"""问题四 · 按依赖顺序重跑全部脚本，验证可复现性。

    python 重跑全部.py            # 全跑一遍（约 10 分钟）
    python 重跑全部.py --快        # 跳过最耗时的两步（判空几何、策略对照），约 3 分钟

顺序即依赖顺序：几何常数 -> 算例 -> 命题检验 -> 对照 -> 复核 -> 日志复盘 -> 作图数据 -> 绘图。
`实战程序.py` 需要官方模拟器在线，不在此列。

**Windows 上的一条注意**：策略对照.py 用了 multiprocessing。
后台分离（nohup / detached）运行时 Windows 会在子进程复制句柄时报
PermissionError: [WinError 5]，必须在前台运行，或把 _进程数 改成 1。
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

流程 = [
    ("判空几何.py", "围堵判据的几何：下界、构造、代价", True),
    ("生成算例.py", "算例生成", False),
    ("命题与引理.py", "模型检验的主干：命题 + 11 条引理", False),
    ("策略对照.py", "策略对照、消融、分组（最耗时）", True),
    ("独立复核.py", "独立复核与变异测试", False),
    ("复盘实战日志.py", "正式测试日志复盘（无日志时自动跳过）", False),
    ("作图数据.py", "作图数据", False),
    ("绘图.py", "出图", False),
]


def main(argv):
    快 = "--快" in argv
    print(f"问题四 · 重跑全部（{'快速模式' if 快 else '完整模式'}）")
    总 = time.time()
    for 名, 说明, 慢 in 流程:
        if 快 and 慢:
            print(f"[跳过] {名:<14} {说明}")
            continue
        t = time.time()
        print(f"[运行] {名:<14} {说明} …", flush=True)
        r = subprocess.run([sys.executable, str(HERE / 名)], cwd=str(HERE))
        if r.returncode != 0:
            print(f"[失败] {名} 返回码 {r.returncode}，停止")
            return 1
        print(f"[完成] {名:<14} 用时 {time.time() - t:.1f} s", flush=True)
    print(f"\n全部完成，总用时 {(time.time() - 总) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
