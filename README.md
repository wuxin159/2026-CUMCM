# 2026 全国大学生数学建模竞赛（CUMCM）

> 2026 年高教社杯全国大学生数学建模竞赛工作仓库，用于统一管理题目资料、数据、代码、图表与论文。
> 技术栈：**Python（建模与求解）+ LaTeX（论文排版）**。

## 仓库用途

- 集中存放赛题原文、附件数据与官方说明
- 管理建模、求解、可视化的全部代码，保证结果可复现
- 版本化论文 LaTeX 源文件，避免多人协作时互相覆盖
- 记录建模思路、假设、参数与阶段性结论

## 多机协作说明（重要）

本仓库会在**多台电脑上同时协作**：操作系统、用户名、Python 与 LaTeX 的安装位置在每台机器上都不相同。
因此仓库里的任何文件都必须做到**换一台机器 clone 下来就能直接跑**。

### 路径铁律

> **除本仓库内部的相对路径外，禁止在任何被提交的文件中硬编码路径。**

允许的只有一种：**以仓库根目录为基准的相对路径**，例如 `data/raw/attach1.xlsx`、`figures/fig1.pdf`。

禁止出现在提交内容里的东西：

- 绝对路径：`C:\Users\WuXin\Desktop\2026-CUMCM\data\...`、`/home/xxx/...`、`D:\数模\...`
- 工具的安装路径：Python 解释器、`pdflatex`/`xelatex`、Gurobi、Node 等可执行文件的完整路径
- 个人环境专属配置：本机虚拟环境目录、IDE 的绝对路径设置、字体文件的绝对路径
- 依赖当前工作目录的写法：直接 `open("data/raw/x.csv")` 而不解析仓库根，换个目录执行就会失败

### 正确做法

Python 中用 `pathlib` 从文件自身位置推出仓库根，再拼相对路径：

```python
from pathlib import Path

# 假设本文件在 src/ 下，parents[1] 即仓库根目录
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"

df = pd.read_csv(DATA_RAW / "attach1.csv")
plt.savefig(FIGURES / "fig1.pdf")
```

LaTeX 中用相对路径引图，不写盘符：

```latex
\graphicspath{{../figures/}}
\includegraphics[width=0.8\linewidth]{fig1.pdf}
```

调用外部工具时用命令名而非全路径，让系统 `PATH` 去解析：

```python
subprocess.run(["python", "-c", "..."])   # 而不是 C:/Python311/python.exe
```

确实需要机器相关配置时，放在**本地不提交**的文件里（如 `.env`、`config.local.py`），并写进 `.gitignore`，同时在仓库中提交一份 `*.example` 模板供他人参考。

### 其他多机注意事项

- **换行符与编码**：源文件统一 UTF-8；git 在 Windows 上会提示 LF/CRLF 转换，属正常现象
- **文件名用中文**：脚本、数据、图表一律用中文命名，见「命名与作图」一节。
  已实测中文名可正常运行与编译。唯一禁止的是空格，用它分隔会给命令行操作带来麻烦
- **依赖同步**：新装的包及时写入 `requirements.txt`，其他机器 `pip install -r requirements.txt` 即可对齐
- **大小写敏感**：Windows 不区分文件名大小写而 Linux/macOS 区分，引用文件时大小写务必与实际一致
- **开工前先 `git pull`，收工后及时 `git push`**，减少多机之间的冲突

## 目录结构（规划）

```text
2026-CUMCM/
├── README.md              # 本文件
├── requirements.txt       # Python 依赖清单
├── problem/               # 赛题原文、附件、官方说明
├── data/
│   ├── raw/               # 原始数据（只读，不做修改）
│   └── processed/         # 清洗、整理后的数据
├── src/                   # 建模与求解代码（Python）
├── notebooks/             # 探索性分析 Jupyter Notebook
├── figures/               # 论文用图表（矢量图优先）
├── paper/                 # 论文 LaTeX 源文件及模板
├── results/               # 模型输出结果、中间产物
└── docs/                  # 思路记录、会议纪要、参考文献
```

## 快速开始

克隆后先确认两个命令可用，安装位置不重要，能被 `PATH` 找到即可：

```powershell
python --version
xelatex --version
```

论文编译，在 `paper/` 目录下执行，必须跑两遍，第一遍生成交叉引用，第二遍才把图表编号填对：

```powershell
xelatex -synctex=1 -interaction=nonstopmode -shell-escape main.tex
xelatex -synctex=1 -interaction=nonstopmode -shell-escape main.tex
```

用 VS Code 则更省事。仓库自带 `.vscode/settings.json`，装好 LaTeX Workshop 扩展后：

| 操作 | 快捷键 |
| ---- | ------ |
| 编译 | `Ctrl+Alt+B`，或直接保存文件自动触发 |
| 侧边预览 PDF | `Ctrl+Alt+V` |
| 源码跳到 PDF 对应位置 | `Ctrl+Alt+J` |
| PDF 跳回源码 | 在 PDF 上 `Ctrl+左键` |

## 作图

### 命名

| 位置 | 示例 |
| ---- | ---- |
| `src/` | `数据预处理.py`、`问题一遗传算法.py` |
| `results/` | `问题一收敛曲线.csv`、`灵敏度分析结果.csv` |
| `figures/` | `技术路线图.pdf`、`问题二热力图.pdf` |

中文命名已实测可用，脚本、CSV、PDF 均无问题。两条限制：文件名不要带空格；
必须用 xelatex 编译，pdflatex 认不了中文文件名。

图中的坐标轴、图例、注释一律中文，不要中英混排。

### 三种画法

| 图的类型 | 画法 |
| -------- | ---- |
| 有数据的常规图：折线、散点、柱状、误差棒、热力图 | PGFPlots 读 CSV |
| PGFPlots 吃力的图：地理分布、复杂网络、上万点散点 | Python 导出矢量 PDF |
| 无数据的示意图：技术路线、模型结构、机理图 | 手写 TikZ |

一律输出矢量，禁止 PNG，禁止截图。

### PGFPlots 读数据（默认方式）

Python 只负责算，结果写进 `results/`：

```python
df[["迭代次数", "目标函数值"]].to_csv(
    ROOT / "results" / "问题一收敛曲线.csv", index=False
)
```

LaTeX 侧读同一份数据，改完数据重编译即可，不必碰任何图形文件：

```latex
\begin{tikzpicture}
  \begin{axis}[xlabel={迭代次数}, ylabel={目标函数值 $f(\bm{x})$}, grid=major]
    \addplot table [x=迭代次数, y=目标函数值, col sep=comma]
      {../results/问题一收敛曲线.csv};
  \end{axis}
\end{tikzpicture}
```

### Python 出图

只在 PGFPlots 吃力时用。必须配中文字体回退列表，不要写死单个字体名：

```python
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"],
    "axes.unicode_minus": False,
})
fig.savefig(ROOT / "figures" / "问题三地理分布.pdf")
```

### 工具关系与性能

PGFPlots 基于 TikZ，TikZ 基于 PGF，三者都是 LaTeX 宏包，随 TeX Live 一起装好，
不需要额外软件。

数据点上千后 PGFPlots 编译会明显变慢，上万点可能超时。先在 Python 侧抽稀，
散点保留几百到一两千个肉眼看不出差别。仍然慢就在 `main.tex` 里启用图形外部化，
把每张图缓存成独立 PDF，编译参数需带 `-shell-escape`。

## 论文模板

`paper/` 使用 [latexstudio/CUMCMThesis](https://github.com/latexstudio/CUMCMThesis) 模板，
由 LaTeX 工作室社区维护，**并非竞赛官方出品**。官方只发布《论文格式规范》文字规定，
不提供任何 LaTeX 模板。该模板 2026 年 8 月更新，已适配当年格式要求。

`cumcmthesis.cls` 与 `cumcm2026.sty` 已随仓库提交，clone 下来即可编译，无需另行安装。
模板自带的使用手册放在 `paper/reference/`，比赛断网时可离线查阅。

文档类选项按提交方式切换：

| 用途 | 写法 |
| ---- | ---- |
| 电子版提交 | `\documentclass[withoutpreface, bwprint]{cumcmthesis}` |
| 打印纸质版 | `\documentclass{cumcmthesis}`，会生成承诺书与编号专用页 |

## AI 工具使用声明（2026 年硬性要求）

官方《人工智能工具使用规定（2026 年试行）》要求所有参赛队在**参考文献之前**
设置「AI 工具使用声明」，两者必选其一，不可省略。模板已封装成命令：

| 情况 | 命令 |
| ---- | ---- |
| 未使用 AI | `\CumcmAINotUsed` |
| 使用了 AI | `\CumcmAIUsed{简要用途}` |

使用了 AI 的队伍还必须在支撑材料中提交一份 PDF，文件名规定为
「AI工具使用详情.pdf」，内容包含四项：

1. 所用 AI 工具名称、版本或型号
2. 具体使用目的和环节
3. 主要提示方式与使用过程说明，可附典型交互示例
4. 对 AI 输出的采纳、人工修改和核验的主要情况，语言润色除外

规定同时要求核心建模与分析由参赛队主导，并对 AI 参与的内容逐项人工审查核实。
故意隐瞒或作虚假声明按违反竞赛规则处理。

## 分工

| 角色 | 成员 | 主要职责 |
| ---- | ---- | -------- |
| 建模 | 待定 | 问题分析、模型建立、假设与符号说明 |
| 编程 | 待定 | 数据处理、模型求解、灵敏度分析、绘图 |
| 写作 | 待定 | 论文撰写、排版、摘要打磨、格式检查 |

## 环境

- **Python 3.x**：`numpy`、`pandas`、`scipy`、`scikit-learn` 等，依赖统一记录在 `requirements.txt`。
  `matplotlib` 仅用于 notebook 里快速查看数据，论文成图一律交给 PGFPlots，见下节
- **LaTeX**：推荐 TeX Live，中文文档用 `xelatex` 编译。论文模板见「论文模板」一节，
  注意官方并不提供 LaTeX 模板

各机器的安装位置可以不同，但需保证 `python` 与 `xelatex` 均已加入系统 `PATH`。

## 约定

- `data/raw/` 中的原始数据不做任何修改，所有处理结果写入 `data/processed/`
- 代码提交前确保可以从头运行，随机过程固定随机种子
- 图表统一风格，字号与线宽在论文中清晰可读，优先导出 PDF / SVG
- 提交信息用简洁的中文或英文说明改动内容，例如 `feat: 完成问题一模型求解`
- 不提交任何自动生成物：`__pycache__/`、LaTeX 中间文件、虚拟环境目录

## 参考资料

- 全国大学生数学建模竞赛官网：<http://www.mcm.edu.cn/>
- 历年赛题与优秀论文：可在官网及各高校数模社团整理资料中查阅
