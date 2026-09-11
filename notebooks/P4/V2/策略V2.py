# -*- coding: utf-8 -*-
"""问题四策略 V2 的对照变体入口。

**V2 的两条改动已经直接落在 `P4/公共.py` 与 `P4/策略.py` 里**（改动极小：
一行格点范围 + 一个默认参数），所以 `P4/策略.推荐策略` 本身就是 V2，
`实战程序.py` 不需要任何改动。本文件只提供**对照用的变体**，
让 `notebooks/评测.py` 能把「改动前」「各条被否的改法」重新跑出来。

用法（在 notebooks/ 下）：

    python 评测.py P4:策略.推荐策略 P4/V2:策略V2.旧版 --局数 100 --种子 20300912
    python 评测.py P4:策略.推荐策略 P4/V2:策略V2.ζ1 --局数 40
    python 评测.py P4:策略.推荐策略 P4/V2:策略V2.内圈优先 --局数 40

每个变体都是一个模块级函数，共用同一份 `策略.推荐策略`，只是参数覆盖不同，
**不存在第二份主流程**。逐条结论见同目录 `优化.md`。
"""
import sys
from pathlib import Path

import numpy as np

_P4 = Path(__file__).resolve().parents[1]
if str(_P4) not in sys.path:
    sys.path.insert(0, str(_P4))

import 公共 as C          # noqa: E402
import 策略 as P          # noqa: E402

_原格点 = C.方向栅格.__init__


def _退回修正前(开):
    """把 方向栅格 的格点范围换回「按自身步长截断」的老写法，用于复现改动前的结果。

    老写法把格点一直铺到 R + 步长/sqrt2：规划栅格（步长 120、裕度 28.3）于是铺到
    半径 1878，而判据栅格只到 1828，多出来的 28 个格点代表的邻域整个在目标区域外，
    却逼着贪心把锚点从 24 个买到 31 个。详见 优化.md §1。
    """
    if not 开:
        C.方向栅格.__init__ = _原格点
    else:
        def 老init(self, 步长=40.0, 扇区数=36, R=C.R_AREA, 额外裕度=0.0, 裕度=None):
            self.步长 = float(步长)
            self.扇区数 = int(扇区数)
            self.裕度 = (float(步长) / np.sqrt(2.0) + float(额外裕度)
                         if 裕度 is None else float(裕度))
            self.扇宽 = 360.0 / self.扇区数
            k = int(np.ceil((R + float(步长) / np.sqrt(2.0)) / 步长))
            轴 = np.arange(-k, k + 1) * 步长
            X, Y = np.meshgrid(轴, 轴)
            内 = np.hypot(X, Y) <= R + float(步长) / np.sqrt(2.0)
            self.点 = np.stack([X[内], Y[内]], axis=1)
            self.n = len(self.点)
            self.单元面积 = 步长 * 步长
            self.有效半径 = C.R_RECV_MIN - self.裕度
            self.边界 = np.arange(self.扇区数 + 1) * self.扇宽
            self._缓存 = {}
        C.方向栅格.__init__ = 老init
    P._栅格缓存.clear()
    P._规划器缓存.clear()


def _跑(狗, 参数, 覆盖, 老格点=False):
    _退回修正前(老格点)
    try:
        return P.推荐策略(狗, dict(P.默认参数, **覆盖, **(参数 or {})))
    finally:
        _退回修正前(False)


# ================================================================ 变体
def 推荐策略(狗, 参数=None):
    """= P4/策略.推荐策略，放在这里只为让评测台能用同一套寻址。"""
    return _跑(狗, 参数, {})


def 旧版(狗, 参数=None):
    """改动前的版本：老格点范围 + ζ = 1.00。用于复现 优化.md §0 的总账。"""
    return _跑(狗, 参数, {"扫描增益ζ": 1.00}, 老格点=True)


def 只旧格点(狗, 参数=None):
    """只退回格点范围，ζ 保持 4.00——用来把两条改动拆开看。"""
    return _跑(狗, 参数, {}, 老格点=True)


def ζ1(狗, 参数=None):
    """只退回 ζ = 1.00（顺手整池扫描打开），格点已修正。"""
    return _跑(狗, 参数, {"扫描增益ζ": 1.00})


def ζ25(狗, 参数=None):
    return _跑(狗, 参数, {"扫描增益ζ": 2.50})


def 盲清60(狗, 参数=None):
    """被否：盲清阈值抬到 60 m。见 优化.md §3。"""
    return _跑(狗, 参数, {"盲清阈值": 60.0, "盲清上限": 10})


def 不盲清(狗, 参数=None):
    """被否：区域必须 <= 20 m 才开枪。"""
    return _跑(狗, 参数, {"盲清阈值": 20.0})


def 按路线(狗, 参数=None):
    """被否：补测点行程按「到最近后续停靠点」算。见 优化.md §4。"""
    return _跑(狗, 参数, {"补测按路线": True})


def 内圈优先(狗, 参数=None):
    """被否：先走内圈锚点、外环留到最后。见 优化.md §3。

    实现方式是包一层 状态.锚点，只在仍有内圈锚点时返回内圈的那些。
    """
    原 = P.状态.锚点

    def 锚点(self, *a, **k):
        出 = 原(self, *a, **k)
        内 = [q for q in 出 if float(np.hypot(q[0], q[1])) <= 1400.0]
        return 内 or 出
    P.状态.锚点 = 锚点
    try:
        return _跑(狗, 参数, {})
    finally:
        P.状态.锚点 = 原


def 清够即停(狗, 参数=None):
    """被否（收益 -5 s）：附件 2 §1.3 给出源数 <= 16，清够 16 个即可收工。

    可证安全，但 24 局里只有 1 局受益，而且会让日志出现「未决频道数 > 0」。
    """
    原 = P.状态.全部解决

    def 全部解决(self):
        if len(self.已清除) >= C.源数上界:
            return True
        return 原(self)
    P.状态.全部解决 = 全部解决
    try:
        return _跑(狗, 参数, {})
    finally:
        P.状态.全部解决 = 原


def 零信息跳过(狗, 参数=None):
    """被否（收益恰好 0，一次都没触发）：残余离本站 1500 m 以外就不测该频道。

    判据：若频道 c 的判空残余里没有任何方格落在 q 的 1500 m（有效接收半径上界）
    之内，则在 q 测 c **必定** no_signal 且排除不掉任何 (方格, 扇区)——可证零信息。
    实测锚点总落在残余的 1500 m 内，与问题三同结论。
    """
    原 = P.扫描

    def 扫描(狗_, st, q):
        池 = []
        界 = 1500.0 + st.栅格.步长 / np.sqrt(2.0)
        for c in sorted(st.未探测):
            行 = st.掩码[c].any(axis=1)
            if not 行.any():
                continue
            pt = st.栅格.点[行]
            if np.hypot(pt[:, 0] - q[0], pt[:, 1] - q[1]).min() > 界:
                st.计数["零信息跳过"] = st.计数.get("零信息跳过", 0) + 1
                continue
            池.append(c)
        for c in C.排序检测频道(池, 狗_.频道):
            if st._键(q, c) in st.已测:
                continue
            回 = 狗_.检测(q, c)
            st.记录检测(q, c, 回)
            st.计数["扫描检测"] += 1
        st.记录扫描停靠(q)
    P.扫描 = 扫描
    try:
        return _跑(狗, 参数, {})
    finally:
        P.扫描 = 原
