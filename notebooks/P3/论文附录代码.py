"""按名字把问题三的核心函数抽出来，拼成论文附录用的单文件（只搬运，不计算）。

论文附录不适合贴全部 4600 行脚本，但手抄摘录又会和代码脱节。
这里用 ast 按**函数名**定位源码行区间原样抽取，`\\lstinputlisting` 直接引用产物，
代码一改、重跑一次即同步，不会出现「论文里的代码和仓库里的不一样」。

    python 论文附录代码.py        # 生成 论文附录_问题三核心代码.py
"""
import ast
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
输出 = HERE / "论文附录_问题三核心代码.py"

# (源文件, 顶层名字, 若是类则取其中哪些方法, 小标题)
清单 = [
    ("公共.py", "频道信念", ["记录检测", "多边形"],
     "集员估计：三类新约束的构造与凸可行集"),
    ("公共.py", "圆外保守半平面", None,
     "非凸排除盘的保守线性化（定理 4 的引理）"),
    ("公共.py", "栅格", ["__init__", "排除圆盘"],
     "判空栅格：只有整格落入排除盘才划掉"),
    ("策略.py", "锚点规划器", ["插入代价", "贪心锚点"],
     "判空锚点的「覆盖每米」贪心"),
    ("策略.py", "保证收信比例", None,
     "不依赖分布的收信把握下界"),
    ("策略.py", "推荐策略", None,
     "一体化滚动时域主循环（算法 1 的实现）"),
    ("策略.py", "理论下界", None,
     "严格下界与全知参照"),
]


def 取源码(文件, 名, 方法=None):
    源 = io.open(HERE / 文件, encoding="utf-8").read()
    行 = 源.split("\n")
    for n in ast.parse(源).body:
        if getattr(n, "name", None) != 名:
            continue
        if 方法 is None:
            return "\n".join(行[n.lineno - 1:n.end_lineno])
        块 = [f"class {名}:"]
        for m in n.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and m.name in 方法:
                块.append("\n".join(行[m.lineno - 1:m.end_lineno]))
                块.append("")
        return "\n".join(块).rstrip()
    raise KeyError(f"{文件} 里没有 {名}")


def main():
    段 = ['"""问题三核心代码（由 论文附录代码.py 自动抽取，勿手改）。',
          "",
          "完整可运行的全部脚本见支撑材料 notebooks/P3/。",
          "本文件只收「计算求解」部分；绘图脚本 绘图.py 不进附录。",
          '"""',
          "import numpy as np", ""]
    for 文件, 名, 方法, 标题 in 清单:
        段.append("")
        段.append("# " + "=" * 72)
        段.append(f"# {标题}")
        段.append(f"# 出处：notebooks/P3/{文件}")
        段.append("# " + "=" * 72)
        段.append(取源码(文件, 名, 方法))
    文 = "\n".join(段) + "\n"
    io.open(输出, "w", encoding="utf-8", newline="\n").write(文)
    print(f"已写入 {输出.name}：{len(文.split(chr(10)))} 行，{len(清单)} 段")


if __name__ == "__main__":
    main()
