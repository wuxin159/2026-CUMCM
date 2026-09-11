"""问题三 · 对接官方模拟器的机器狗程序（演练测试与正式测试共用）。"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import 公共 as C
import 模拟器 as S
import 策略 as ST

HERE = Path(__file__).resolve().parent


def 取队号(argv):
    for a in argv[1:]:
        if not a.startswith("--"):
            return a.strip()
    if os.environ.get("CUMCM_ROBOT_ID"):
        return os.environ["CUMCM_ROBOT_ID"].strip()
    f = HERE / "参赛队号.txt"
    if f.exists():
        for 行 in f.read_text(encoding="utf-8").splitlines():
            if 行.strip() and not 行.strip().startswith("#"):
                return 行.strip()
    raise SystemExit(
        "未找到参赛队号。请用以下任一方式提供：\n"
        "  python 实战程序.py <队号>\n"
        "  设置环境变量 CUMCM_ROBOT_ID\n"
        "  在本目录建 参赛队号.txt，第一行写队号（该文件不要提交）")


def 取端口(argv, 默认=2026):
    for i, a in enumerate(argv):
        if a == "--端口" and i + 1 < len(argv):
            return int(argv[i + 1])
    return 默认


def main(argv):
    队号 = 取队号(argv)
    端口 = 取端口(argv)
    基址 = f"http://127.0.0.1:{端口}"
    标签 = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"队号 {队号}，接口 {基址}")
    print("正在连接模拟器…（若报连接失败，请确认测试已开始且 5 秒倒计时已结束）")

    后端 = S.HTTP后端(队号, 基址=基址)
    狗 = S.机器狗(后端, 记录日志=True)
    结果, 错 = None, None
    现实起 = time.time()
    try:
        # 推荐策略内部会调用 /enter 与 /exit；/enter 的响应里带本局实际可用的现实时间，
        结果 = ST.推荐策略(狗)
    except Exception as e:                       # 任何异常都要先退出再抛
        错 = e
        try:
            狗.退出()
        except Exception:
            pass
    现实用时 = time.time() - 现实起

    # ---------------------------------------------------------- 落盘
    日志 = pd.DataFrame(狗.日志 or [])
    日志文件 = HERE / f"实战日志_{标签}.csv"
    日志.to_csv(日志文件, index=False, encoding="utf-8-sig")

    清除成功 = 狗.清除成功
    虚拟 = 狗.虚拟时刻
    平均 = 虚拟 / 清除成功 if 清除成功 else float("nan")
    摘要 = {
        "时间标签": 标签,
        "清除干扰源个数": 清除成功,
        "虚拟总时间_秒": round(虚拟, 3),
        "平均定位清除时间_秒": round(平均, 3) if 清除成功 else "",
        "程序运行时间_秒": round(现实用时, 3),
        "检测次数": 狗.检测次数,
        "清除指令次数": 狗.清除次数,
        "判空频道数": 结果["判空频道数"] if 结果 else "",
        "异常": ";".join(结果["异常"]) if 结果 else repr(错),
    }
    pd.DataFrame([摘要]).to_csv(HERE / f"实战摘要_{标签}.csv", index=False,
                                encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("本次运行结果（论文表 1 的三列取下面前三项）")
    print("=" * 60)
    for k, v in 摘要.items():
        print(f"    {k:<22} {v}")
    print(f"\n    指令日志已写入 {日志文件.name}")
    print("    测试案例编码请从模拟器界面抄录，加密行为日志请从模拟器导出。")
    if 错 is not None:
        print("\n" + "!" * 60)
        print(f"本次运行中断：{type(错).__name__}: {错}")
        if "ConnectionRefused" in repr(错) or "URLError" in repr(错):
            print("看起来是连不上机器狗接口。逐条核对：")
            print("  1) 模拟器已联网登录，并已**启动**对应测试模块")
            print("  2) 界面显示「机器狗接口已就绪」——5 秒倒计时期间连不上属正常")
            print(f"  3) 端口与模拟器设置一致（当前用的是 {基址}）")
            print("  4) 本机时间与服务器时间相差不超过 60 秒")
        elif "accepted" in repr(错):
            print("请求被模拟器拒收。检查队号是否与当前登录的参赛队号完全一致。")
        print("!" * 60)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
