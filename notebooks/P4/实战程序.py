"""问题四 · 对接官方模拟器的机器狗程序（演练测试与正式测试共用）。

    python 实战程序.py                       # 队号从 参赛队号.txt 或环境变量读
    python 实战程序.py 12345678              # 直接给队号
    python 实战程序.py 12345678 --端口 2027  # 模拟器改过端口时
    python 实战程序.py --演练                # 只是在结束时多打印一段演练自查清单

队号来源优先级：命令行参数 > 环境变量 CUMCM_ROBOT_ID > 本目录 参赛队号.txt 第一行。
**参赛队号.txt 不要提交**（个人环境信息，同 README「路径铁律」的口径），
仓库里只放 参赛队号.txt.example。

策略代码与后端解耦：`模拟器.本地后端` 与 `模拟器.HTTP后端` 实现同一组
enter/measure/clear/exit，`策略.推荐策略` 只看见 `模拟器.机器狗`。
**离线跑了几百局的那份策略代码，一行不改就能对接官方模拟器。**

协议上必须守的几条（附件 2 §5、§12），以及本程序里的落实位置：

    每个新动作用新的 request_id，只有网络异常重试同一动作才复用
        -> 模拟器.HTTP后端._新id 递增；重试在 _发 内部，复用同一载荷
    必须**同时**检查 HTTP 状态与 accepted
        -> 模拟器.机器狗.检测/清除 对 accepted is not True 直接抛异常
    accepted=false 时 virtual_time_s 是 0，不能当作当前时刻
        -> 只在 accepted=true 时更新 虚拟时刻
    用 /enter 返回的 remaining_real_duration_s，不要假定 1200 s
        -> 机器狗.进入 读该字段设 现实时限，策略按 现实剩余 收尾
    逐次等待响应，不得并发发送不同动作
        -> 全程单线程串行
    选手须自行记录指令序列与响应
        -> 机器狗.日志 全程记录，落盘为 实战日志_*.csv
    加密日志 <= 2 MB，避免异常高频循环
        -> 策略.默认参数["最多动作"] = 2500；实测一局约 350~450 条指令
    测试结束后不要再调 /exit 查询原因
        -> 异常路径只尝试一次 /exit，失败即放弃
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import 公共 as C
import 模拟器 as S
import 策略 as P

HERE = Path(__file__).resolve().parent


def 取队号(argv):
    for a in argv[1:]:
        if not a.startswith("--") and a.strip():
            return a.strip()
    v = os.environ.get("CUMCM_ROBOT_ID", "").strip()
    if v:
        return v
    f = HERE / "参赛队号.txt"
    if f.exists():
        s = f.read_text(encoding="utf-8").strip().splitlines()
        if s and s[0].strip():
            return s[0].strip()
    raise SystemExit(
        "没有找到参赛队号。三选一：命令行传入、设环境变量 CUMCM_ROBOT_ID、"
        "或在本目录建 参赛队号.txt（第一行写队号，该文件不要提交）。")


def 取端口(argv):
    for i, a in enumerate(argv):
        if a == "--端口" and i + 1 < len(argv):
            return int(argv[i + 1])
    return 2026


def main(argv):
    队号 = 取队号(argv)
    端口 = 取端口(argv)
    戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"问题四 · 机器狗程序")
    print(f"队号 {队号}    接口 http://127.0.0.1:{端口}")
    print("请确认模拟器已联网登录、已启动「问题四演练测试」或「问题四正式测试」，")
    print("且界面显示机器狗接口已就绪（5 秒倒计时期间连不上属正常现象）。")
    后端 = S.HTTP后端(队号, 基址=f"http://127.0.0.1:{端口}")
    狗 = S.机器狗(后端, 记录日志=True)
    现实起 = time.time()
    try:
        r = P.推荐策略(狗)
    except Exception as e:
        print(f"\n[异常] {e!r}")
        try:
            狗.退出()
        except Exception:
            pass
        r = None
    现实用时 = time.time() - 现实起

    日志 = pd.DataFrame(狗.日志)
    日志路径 = HERE / f"实战日志_{戳}.csv"
    日志.to_csv(日志路径, index=False, encoding="utf-8-sig")

    清除个数 = 狗.清除成功
    虚拟总时间 = 狗.虚拟时刻
    平均 = 虚拟总时间 / 清除个数 if 清除个数 else float("nan")
    print("\n" + "=" * 70)
    print("本次测试结果（论文表 1 的三列直接抄下面三行）")
    print("=" * 70)
    print(f"  清除干扰源个数    {清除个数}")
    print(f"  平均定位清除时间  {平均:.2f} s"
          f"   （= 虚拟总时间 {虚拟总时间:.2f} s / 清除个数）")
    print(f"  程序运行时间      {现实用时:.1f} s   （/enter 成功到 /exit 之间的现实秒数）")
    print(f"  指令条数          {len(日志)}（检测 {狗.检测次数}，清除 {狗.清除次数}）")
    if r is not None:
        st = r["状态"]
        未决 = [c for c in range(1, C.频道数 + 1)
                if c not in st.已清除 and (st.信念[c].收到信号 or st.掩码[c].any())]
        print(f"  已清除 + 判空     {len(st.已清除)} + {r['判空频道数']} = "
              f"{len(st.已清除) + r['判空频道数']} / 20"
              f"{'   ✅ 每个频道都有依据' if not 未决 else f'   ⚠ 未决频道 {未决}'}")
        print(f"  判定为定向的频道  {r['计数'].get('判定定向', 0)} 个；"
              f"判定为全向 {r['计数'].get('判定全向', 0)} 个")
        if r["异常"]:
            print(f"  异常记录          {r['异常']}")
    print(f"\n  测试案例编码请从模拟器界面抄下，与本文件对应：{日志路径.name}")

    摘要 = pd.DataFrame([{
        "时间戳": 戳, "队号": 队号, "清除干扰源个数": 清除个数,
        "虚拟总时间": 虚拟总时间, "平均定位清除时间": 平均,
        "程序运行时间": 现实用时, "指令条数": len(日志),
        "检测次数": 狗.检测次数, "清除指令次数": 狗.清除次数,
        "判空频道数": (r["判空频道数"] if r else None),
        "未决频道数": (r["未决频道数"] if r else None),
        "测试案例编码": "（从模拟器界面抄写）",
    }])
    摘要.to_csv(HERE / f"实战摘要_{戳}.csv", index=False, encoding="utf-8-sig")
    print(f"  已写入 {日志路径.name} 与 实战摘要_{戳}.csv")

    if "--演练" in argv:
        print("""
演练测试自查清单（每局都要走一遍）：
  [ ] 界面显示的干扰源总数 == 上面的「清除干扰源个数」
      这是**唯一**能直接验证「判空没有误判」的机会，务必每次都核
  [ ] 界面显示的定向干扰源数量，与上面「判定为定向的频道」数量对得上
      （对不上不一定是错——分类器只在证据充分时才下结论，漏判是保守的）
  [ ] 「已清除 + 判空 = 20」成立
  [ ] 日志里没有 accepted=false、没有 HTTP 4xx
  [ ] 单局指令数在 400 上下，没有异常高频循环（日志要 <= 2 MB）
  [ ] 程序运行时间远小于 20 分钟
""")


if __name__ == "__main__":
    main(sys.argv)
