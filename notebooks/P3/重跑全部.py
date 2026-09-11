"""问题三 · 按依赖顺序重跑全部脚本，验证可复现性。"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (文件, 说明, 快速模式下跳过, 自己支持 --快)
流程 = [
    ("覆盖锚点.py", "判空所需的圆覆盖问题", False, False),
    ("生成算例.py", "算例生成", False, False),
    ("命题与引理.py", "模型检验主干：三命题八引理密集扫描（6 秒）", False, False),
    ("参数搜索.py", "策略参数调优（最耗时）", True, False),
    ("策略对照.py", "策略对照、消融、分组", False, False),
    ("灵敏度.py", "灵敏度与稳健性（多进程，约 15 分钟）", False, True),
    ("独立复核.py", "集成检验与变异测试（多进程，约 4 分钟）", False, True),
    ("作图数据.py", "作图数据", False, False),
    ("绘图.py", "出图", False, False),
    ("复盘实战日志.py", "实战日志复盘（需先有 实战日志_*.csv）", False, False),
    ("导出论文数据.py", "把论文用数据导出到 results/", False, False),
    ("论文附录代码.py", "抽取论文附录用的核心代码", False, False),
]


def main(argv):
    快 = "--快" in argv
    print(f"问题三 · 重跑全部（{'快速模式' if 快 else '完整模式'}）")
    总 = time.time()
    for 名, 说明, 慢, 支持快 in 流程:
        if 快 and 慢:
            print(f"[跳过] {名:<14} {说明}")
            continue
        额外 = ["--快"] if (快 and 支持快) else []
        t = time.time()
        print(f"[运行] {名:<14} {说明} {' '.join(额外)}…", flush=True)
        r = subprocess.run([sys.executable, str(HERE / 名)] + 额外, cwd=str(HERE))
        if r.returncode != 0:
            print(f"[失败] {名} 返回码 {r.returncode}，停止")
            return 1
        print(f"[完成] {名:<14} 用时 {time.time() - t:.1f} s", flush=True)
    print(f"\n全部完成，总用时 {(time.time() - 总) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
