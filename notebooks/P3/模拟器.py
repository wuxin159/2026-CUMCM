"""问题三 · 第 2 步：本地模拟器与机器狗接口（只算数，不作图，不写文件）。

策略代码不应该知道自己面对的是本地复刻件还是官方模拟器，所以这里定一个统一的
「后端」接口，两种实现共用：

    本地后端    按附件 1 表 1、附件 2 §4 复刻规则，用于离线的成千上万次实验
    HTTP后端    按附件 2 §5~§9 的 HTTP+JSON 协议对接官方模拟器，用于正式测试

`机器狗` 类包在后端外面，负责三件事：记录完整指令序列（附件 2 §12 明确要求
「机器狗程序应该自行记录测试过程中的指令序列、响应信息等内容，模拟器没有这一功能」）、
维护本地的位置/频道/虚拟时刻镜像、以及分项计时（移动/检测/切频/清除），
后者是分析时间去向的依据。

复刻的规则与出处：
    移动耗时 = 直线距离 / 5                      附件 2 §4.2
    检测 5 s；频道与上次 /measure 不同再加 1 s   附件 2 §4.3
    清除成功 5 s，未发现 3 s；不切频道           附件 2 §4.4、§8.2
    /enter、/exit 不推进虚拟时钟                 附件 2 §4.1
    距离 > R_eff 返回 no_signal                  附件 2 §2.1
    距离 <= 5 m 返回 near，不给示向度            附件 2 §2.4
    示向度含 [-1°,1°] 误差、保留两位小数         附件 2 §2.3
    误差绑定在地点上，重复检测不变               题目正文附录 2 第 1 条
"""
import hashlib
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

import 公共 as C


# ================================================================ 电磁环境
class 电磁环境:
    """把示向度误差实现为**位置（与频道）的确定性函数**，复刻「同一地点重复检测不变」。

    沿用 P2 的做法。四种模式：
        独立    不同地点相互独立（基准假设，只知误差确界时的最大熵选择）
        相关    误差是平滑随机场，相关长度可调（P1 §5.3 提出的灵敏度情景）
        最坏    恒取 ±delta，检验集员估计在边界上的表现
        零      无误差，用于复核几何
    """

    def __init__(self, rng, 模式="独立", 相关长度=300.0, 分布="均匀"):
        self.模式, self.分布, self.相关长度 = 模式, 分布, 相关长度
        if 模式 == "相关":
            K = 6
            self.幅 = rng.dirichlet(np.ones(K))          # 和为 1，保证 |eps| <= delta
            方向 = rng.uniform(0, 2 * np.pi, K)
            self.波矢 = (2 * np.pi / 相关长度) * np.stack(
                [np.cos(方向), np.sin(方向)], axis=1)
            self.相位 = rng.uniform(0, 2 * np.pi, K)
        self.盐 = int(rng.integers(0, 2 ** 31))

    def 误差(self, 位置, 频道):
        if self.模式 == "零":
            return 0.0
        if self.模式 == "相关":
            v = float(np.sum(self.幅 * np.cos(
                self.波矢 @ np.asarray(位置, float) + self.相位)))
            return C.DELTA * v
        键 = f"{self.盐}|{位置[0]:.6f}|{位置[1]:.6f}|{频道}".encode()
        u = int(hashlib.blake2b(键, digest_size=8).hexdigest(), 16) / 2 ** 64
        if self.模式 == "最坏":
            return C.DELTA if u < 0.5 else -C.DELTA
        if self.分布 == "三角":
            return C.DELTA * (np.sqrt(2 * u) - 1 if u < 0.5
                              else 1 - np.sqrt(2 * (1 - u)))
        return C.DELTA * (2 * u - 1)


# ================================================================ 案例
class 案例:
    """一局测试的真值：若干干扰源的频道、位置、有效接收半径。

    这些数据在正式测试中完全不可读（附录 3），本地复刻件里保留是为了
    （1）计算理论下界，（2）复核「可行集始终含真源」这条不变量。
    策略代码**不得**访问本类的任何字段，`独立复核.py` 会检查这一点。
    """

    def __init__(self, 源表, 环境):
        self.源表 = [dict(s) for s in 源表]     # [{频道, x, y, R_eff}]
        self.环境 = 环境

    @property
    def 源数(self):
        return len(self.源表)

    def 位置表(self):
        return [(s["x"], s["y"]) for s in self.源表]

    def 频道集(self):
        return {int(s["频道"]) for s in self.源表}


# ================================================================ 本地后端
class 本地后端:
    def __init__(self, 案例对象):
        self.案例 = 案例对象
        self.已清除 = {int(s["频道"]): False for s in 案例对象.源表}
        self.源 = {int(s["频道"]): (np.array([s["x"], s["y"]], dtype=float),
                                    float(s["R_eff"])) for s in 案例对象.源表}
        self.位置 = np.zeros(2)
        self.频道 = 1
        self.虚拟时刻 = 0.0
        self.已进入 = False
        self.已退出 = False
        # 分项计时
        self.计时 = {"移动": 0.0, "检测": 0.0, "切频": 0.0, "清除": 0.0}
        self.路程 = 0.0

    # ---------------------------------------------------------- 内部
    def _移动(self, 位置):
        位置 = np.asarray(位置, dtype=float)
        s = float(np.hypot(*(位置 - self.位置)))
        self.路程 += s
        t = s / C.V_MOVE
        self.计时["移动"] += t
        self.虚拟时刻 += t
        self.位置 = 位置

    # ---------------------------------------------------------- 接口
    def enter(self):
        self.已进入 = True
        return {"accepted": True, "virtual_time_s": 0.0,
                "max_virtual_duration_s": 360000.0, "max_real_duration_s": 1200.0,
                "remaining_real_duration_s": 1200.0}

    def measure(self, x, y, 频道):
        if not self.已进入 or self.已退出:
            return {"accepted": False, "virtual_time_s": 0.0}
        频道 = int(频道)
        self._移动((x, y))
        if 频道 != self.频道:
            self.虚拟时刻 += C.T_SWITCH
            self.计时["切频"] += C.T_SWITCH
            self.频道 = 频道
        self.虚拟时刻 += C.T_MEASURE
        self.计时["检测"] += C.T_MEASURE
        回 = {"accepted": True, "virtual_time_s": self.虚拟时刻}
        if 频道 not in self.源 or self.已清除[频道]:
            回["measure_result"] = "no_signal"
            return 回
        G, R_eff = self.源[频道]
        d = float(np.hypot(*(G - self.位置)))
        if d > R_eff:
            回["measure_result"] = "no_signal"
        elif d <= C.R_NEAR:
            回["measure_result"] = "near"
        else:
            真 = C.方位角(self.位置, G)
            读 = (真 + self.案例.环境.误差(tuple(self.位置), 频道)) % 360.0
            回["measure_result"] = "direction"
            回["svd_deg"] = round(读, 2)
        return 回

    def clear(self, x, y, 频道):
        if not self.已进入 or self.已退出:
            return {"accepted": False, "virtual_time_s": 0.0}
        频道 = int(频道)
        self._移动((x, y))
        命中 = (频道 in self.源) and (not self.已清除[频道]) and \
               float(np.hypot(*(self.源[频道][0] - self.位置))) <= C.R_CLEAR
        dt = C.T_CLEAR_OK if 命中 else C.T_CLEAR_FAIL
        self.虚拟时刻 += dt
        self.计时["清除"] += dt
        if 命中:
            self.已清除[频道] = True
        return {"accepted": True, "virtual_time_s": self.虚拟时刻,
                "clear_result": "success" if 命中 else "no_target_in_range"}

    def exit(self):
        self.已退出 = True
        return {"accepted": True, "virtual_time_s": self.虚拟时刻,
                "exit_reason": "user_exit"}


# ================================================================ HTTP 后端
class HTTP后端:
    """对接官方模拟器。字段与校验按附件 2 §5~§9。

    三条容易踩的协议细节，都在这里落实：
      · 每个新动作用新的 request_id；只有重试**完全相同**的动作才复用原 ID 与原请求体
      · 必须同时检查 HTTP 状态与 accepted，两者缺一不可（附件 2 §5.3）
      · accepted=false 时 virtual_time_s 是 0，不能当作当前虚拟时刻（附件 2 §4.1）
    """

    def __init__(self, robot_id, 基址="http://127.0.0.1:2026", 超时=5.0, 重试=3):
        self.robot_id = str(robot_id)
        self.基址 = 基址.rstrip("/")
        self.超时 = float(超时)
        self.重试 = int(重试)
        self._序号 = 0
        self.虚拟时刻 = 0.0

    def _新id(self, 前缀):
        self._序号 += 1
        return f"{前缀}-{self._序号}"

    def _发(self, 路径, 载荷):
        """发送并解析。只有网络层异常才重试（且复用同一 request_id 与请求体）。

        HTTP 4xx 是我们自己的请求写错了（附件 2 §5.3），重试没有意义，
        而且 400 不占用 request_id、可以修好再发；所以直接把响应体抛出来看错在哪。
        429 与 5xx 属于服务端侧的临时状况，照常重试。
        """
        最后错 = None
        体 = json.dumps(载荷).encode("utf-8")       # 无 BOM 的 UTF-8（附件 2 §5.1）
        for 次 in range(self.重试):
            try:
                请求 = Request(self.基址 + 路径, data=体,
                               headers={"Content-Type": "application/json"},
                               method="POST")
                with urlopen(请求, timeout=self.超时) as 响应:
                    回 = json.loads(响应.read().decode("utf-8"))
                if 回.get("accepted") is True:
                    self.虚拟时刻 = float(回.get("virtual_time_s", self.虚拟时刻))
                return 回
            except HTTPError as e:
                文本 = ""
                try:
                    文本 = e.read().decode("utf-8", "replace")
                except Exception:
                    pass
                if 400 <= e.code < 500 and e.code != 429:
                    raise RuntimeError(
                        f"{路径} HTTP {e.code}：{文本}（请求体：{体.decode()}）")
                最后错 = f"HTTP {e.code}：{文本}"
            except Exception as e:                 # 连接被直接关闭时没有 JSON 体
                最后错 = repr(e)
            time.sleep(0.2 * (次 + 1))
        raise RuntimeError(f"{路径} 通信失败：{最后错}")

    def _基(self, 前缀):
        return {"arena_id": "default", "robot_id": self.robot_id,
                "request_id": self._新id(前缀)}

    def enter(self):
        return self._发("/enter", self._基("enter"))

    def measure(self, x, y, 频道):
        载荷 = self._基("measure")
        载荷["position"] = {"x": float(x), "y": float(y)}
        载荷["channel"] = int(频道)
        return self._发("/measure", 载荷)

    def clear(self, x, y, 频道):
        载荷 = self._基("clear")
        载荷["position"] = {"x": float(x), "y": float(y)}
        载荷["channel"] = int(频道)
        return self._发("/clear", 载荷)

    def exit(self):
        return self._发("/exit", self._基("exit"))


# ================================================================ 机器狗
class 机器狗:
    """策略唯一可见的对象。除本类暴露的方法外，策略不得访问任何真值。"""

    def __init__(self, 后端, 记录日志=True, 现实时限=None):
        self.后端 = 后端
        self.位置 = np.zeros(2)
        self.频道 = 1
        self.虚拟时刻 = 0.0
        self.日志 = [] if 记录日志 else None
        self.检测次数 = 0
        self.清除次数 = 0
        self.清除成功 = 0
        self.现实起点 = None
        self.现实时限 = 现实时限
        self.已退出 = False

    # ---------------------------------------------------------- 生命周期
    def 进入(self):
        回 = self.后端.enter()
        if 回.get("accepted") is not True:
            raise RuntimeError(f"/enter 未被接受：{回}")
        self.现实起点 = time.time()
        if self.现实时限 is None:
            self.现实时限 = float(回.get("remaining_real_duration_s", 1200.0))
        self._记(("enter", None, None, 回))
        return 回

    def 退出(self):
        if self.已退出:
            return None
        回 = self.后端.exit()
        self.已退出 = True
        self._记(("exit", None, None, 回))
        return 回

    @property
    def 现实剩余(self):
        if self.现实起点 is None:
            return np.inf
        return self.现实时限 - (time.time() - self.现实起点)

    # ---------------------------------------------------------- 动作
    def 检测(self, 位置, 频道):
        位置 = (float(位置[0]), float(位置[1]))
        回 = self.后端.measure(位置[0], 位置[1], int(频道))
        if 回.get("accepted") is not True:
            raise RuntimeError(f"/measure 未被接受：{回}")
        self.位置 = np.asarray(位置, dtype=float)
        self.频道 = int(频道)
        self.虚拟时刻 = float(回["virtual_time_s"])
        self.检测次数 += 1
        self._记(("measure", 位置, int(频道), 回))
        return 回

    def 清除(self, 位置, 频道):
        位置 = (float(位置[0]), float(位置[1]))
        回 = self.后端.clear(位置[0], 位置[1], int(频道))
        if 回.get("accepted") is not True:
            raise RuntimeError(f"/clear 未被接受：{回}")
        self.位置 = np.asarray(位置, dtype=float)   # /clear 不改频道（附件 2 §8.2）
        self.虚拟时刻 = float(回["virtual_time_s"])
        self.清除次数 += 1
        if 回["clear_result"] == "success":
            self.清除成功 += 1
        self._记(("clear", 位置, int(频道), 回))
        return 回

    def _记(self, 项):
        if self.日志 is not None:
            动作, 位置, 频道, 回 = 项
            self.日志.append({"动作": 动作, "x": 位置[0] if 位置 else None,
                              "y": 位置[1] if 位置 else None, "频道": 频道,
                              "虚拟时刻": self.虚拟时刻,
                              "结果": 回.get("measure_result") or
                                      回.get("clear_result") or
                                      回.get("exit_reason") or "",
                              "示向度": 回.get("svd_deg")})
