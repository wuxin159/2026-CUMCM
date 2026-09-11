"""问题三 · 按依赖顺序重跑全部脚本，验证可复现性。

    python 重跑全部.py            # 全跑一遍（约 1.5~2 小时，多数时间花在参数搜索上）
    python 重跑全部.py --快        # 跳过最耗时的两步（参数搜索、独立复核），约 15 分钟

顺序即依赖顺序：几何常数 -> 算例 -> 参数 -> 对照 -> 灵敏度 -> 复核 -> 作图数据 -> 绘图。
`实战程序.py` 需要官方模拟器在线，不在此列。
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

流程 = [
    ("覆盖锚点.py", "判空所需的圆覆盖问题", False),
    ("生成算例.py", "算例生成", False),
    ("参数搜索.py", "策略参数调优（最耗时）", True),
    ("策略对照.py", "策略对照、消融、分组", False),
    ("灵敏度.py", "灵敏度与稳健性", False),
    ("独立复核.py", "独立复核与变异测试（耗时）", True),
    ("作图数据.py", "作图数据", False),
    ("绘图.py", "出图", False),
]


def main(argv):
    快 = "--快" in argv
    print(f"问题三 · 重跑全部（{'快速模式' if 快 else '完整模式'}）")
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
