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
- **文件名**：只用英文、数字、`-`、`_`，避免中文和空格，防止跨系统与 LaTeX 编译出错
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

## 分工

| 角色 | 成员 | 主要职责 |
| ---- | ---- | -------- |
| 建模 | 待定 | 问题分析、模型建立、假设与符号说明 |
| 编程 | 待定 | 数据处理、模型求解、灵敏度分析、绘图 |
| 写作 | 待定 | 论文撰写、排版、摘要打磨、格式检查 |

## 环境

- **Python 3.x**：`numpy`、`pandas`、`scipy`、`matplotlib`、`scikit-learn` 等，依赖统一记录在 `requirements.txt`
- **LaTeX**：推荐 TeX Live，使用 CUMCM 官方格式模板（如 `cumcmthesis`），中文文档用 `xelatex` 编译

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
