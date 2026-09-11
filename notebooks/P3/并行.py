"""多进程跑独立算例，边完成边报进度（只算数，不作图）。

灵敏度与复核都是「几百个互不相干的单局」，一局约 3.7 s，串行几百局就是几十分钟；
原先又把 print 重定向到文件、结束才刷盘，中途看不出任何进度。这里统一解决三件事：

    单线程   先把 BLAS 的线程数钉成 1。本问题的矩阵都很小，多线程只会让 12 个
             进程互相抢核；必须在 numpy 被导入之前设好，故本模块要最先 import。
    并行     用进程池。变异测试尤其需要进程隔离——它靠 monkey-patch 改模块全局量，
             同一进程里并行会互相污染，独立进程则天然互不干扰。
    进度     每完成一批就往 stderr 打一行并 flush；正文用 记录到() 同时写屏与写文件。

算例改为**按序号取种子**（`default_rng(基种子 + 序号)`）而不是从一个 rng 顺序抽，
这样任一子进程都能独立重建第 i 局；同一情景组内各档共用同一批种子，
共同随机数的比较口径与串行版一致。
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# 这台机器是三个人共用的，队友也会开进程池；默认只吃三分之一的核，
# 需要独占时用环境变量 P3_进程数 覆盖。
进程数 = int(os.environ.get("P3_进程数") or max(1, (os.cpu_count() or 4) // 3))


def 报(*a):
    """进度只写 stderr：正文重定向到文件时，进度仍然留在屏幕上。"""
    print(*a, file=sys.stderr, flush=True)


class _双写:
    def __init__(self, 屏, 文件):
        self.屏, self.文件 = 屏, 文件

    def write(self, s):
        try:
            self.屏.write(s)
            self.屏.flush()
        except Exception:                      # 控制台编码不认某个字符也不能中断
            pass
        self.文件.write(s)
        self.文件.flush()

    def flush(self):
        self.文件.flush()


@contextmanager
def 记录到(路径):
    """正文同时写屏与写文件，且每行都 flush——中途 tail 得到东西。"""
    with open(路径, "w", encoding="utf-8") as f:
        原 = sys.stdout
        sys.stdout = _双写(原, f)
        try:
            yield
        finally:
            sys.stdout = 原
    报(f"已写入 {路径}")


def 并行映射(函数, 任务表, 说明="", 进程=None):
    """对任务表并行求值，按输入顺序返回结果列表。

    进度用 as_completed 而不是 map：`map` 按提交顺序交付，**一个慢任务会把计数器
    卡住**，后面早就算完的也看不见——这正是「跑了十分钟没动静」的另一半原因。
    """
    任务表 = list(任务表)
    n = len(任务表)
    if n == 0:
        return []
    t0 = time.time()
    出 = [None] * n
    完成 = 0
    with ProcessPoolExecutor(max_workers=进程 or 进程数) as 池:
        期物 = {池.submit(函数, t): i for i, t in enumerate(任务表)}
        for f in as_completed(期物):
            出[期物[f]] = f.result()
            完成 += 1
            if 完成 % max(1, n // 20) == 0 or 完成 == n:
                用 = time.time() - t0
                报(f"    [{说明}] {完成}/{n}  已用 {用:.0f}s  "
                   f"预计还需 {用 / 完成 * (n - 完成):.0f}s")
    return 出
