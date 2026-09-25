"""全局配置：路径、常量、绘图风格。"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 路径
ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> dict:
    """读取项目根目录的 .env，把配置写进环境变量。

    .env 已被 .gitignore 忽略，API Key 不会进入代码仓库。
    已经存在的环境变量优先级更高，不会被覆盖。
    """
    path = path or (ROOT / ".env")
    loaded = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        loaded[k] = v
        os.environ.setdefault(k, v)
    return loaded


load_env()

DATA_RAW = ROOT / "data" / "raw"
OUTPUT_ROOT = Path(os.environ.get("EXPERIMENT_OUTPUT", str(ROOT))).resolve()
DATA_PROC = OUTPUT_ROOT / "data" / "processed"
IMG_DIR = OUTPUT_ROOT / "img"
RESULT_DIR = OUTPUT_ROOT / "results"
MODEL_DIR = OUTPUT_ROOT / "models"

for _d in (DATA_RAW, DATA_PROC, IMG_DIR, RESULT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "four_mutations_full_data.csv"
RAW_FASTA = DATA_RAW / "5LDE_1.fasta"

# 任务常量
# GB1 (免疫球蛋白结合蛋白 G 的 B1 结构域) 四位点组合突变库
# Wu et al. 2016, eLife —— 经 FLIP benchmark 整理
DATASET_NAME = "GB1 four-site combinatorial library (Wu et al., 2016)"
PROTEIN_NAME = "Protein G domain B1 (GB1)"
TASK_DESC = "提高 GB1 对人 IgG Fc 段的结合适应度 (binding fitness)"

# GB1 结构域中的四个位点（1-based），与 FLIP 序列表示的 N 端编号一致。
MUT_POSITIONS = [39, 40, 41, 54]
GB1_LENGTH = 56
WT_COMBO = "VDGV"                      # 野生型在四个位点上的氨基酸
WT_FITNESS = 1.0                       # 数据集中野生型适应度被归一化为 1.0

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"   # 20 种标准氨基酸
N_POS = len(MUT_POSITIONS)

# fitness -> log 空间的偏移量(数据中存在大量 零适应度记录)
LOG_EPS = 1e-2

# 实验设置
RANDOM_SEED = 42
TRAIN_MAX_HD = 2        # "已完成实验"：野生型 + 单点 + 双点突变
VAL_RATIO = 0.2         # 从已完成实验中划分验证集
N_ROUNDS = 3            # 虚拟定向进化轮数
BATCH_SIZE = 12         # 每轮"送去实验"的候选数量
TOP_K = 5               # 每轮最终推荐的 Top-k

# 绘图风格
# 浅色系 / 白底 / 微软雅黑 / 不使用紫色
PALETTE = {
    "teal":   "#0E9F9B",
    "sky":    "#3B93E0",
    "amber":  "#E8A33D",
    "coral":  "#E4694E",
    "green":  "#4CA97B",
    "rose":   "#D96A84",
    "slate":  "#5B6B7C",
    "ink":    "#22313F",
    "muted":  "#8C9BAA",
    "grid":   "#E6ECF2",
    "bg":     "#FFFFFF",
}
SERIES_COLORS = [PALETTE["teal"], PALETTE["sky"], PALETTE["amber"],
                 PALETTE["coral"], PALETTE["green"], PALETTE["rose"]]

# 策略配色(全流程统一)
STRATEGY_COLORS = {
    "Random": PALETTE["muted"],
    "Model-Greedy": PALETTE["sky"],
    "LLM-Agent": PALETTE["amber"],
    "LLM-Agent+KB": PALETTE["teal"],
    "Rule-Agent": PALETTE["amber"],
    "Rule-Agent+KB": PALETTE["teal"],
    "Rule-Agent+KB-static": PALETTE["coral"],
}

# 浅色连续色带(青 -> 琥珀)，用于热图，避开紫色
LIGHT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gb1_light", ["#F5FAFA", "#CFEAE7", "#8FD3CC", "#E8CFA0", "#E4915E"], N=256
)


def rel(path) -> str:
    """把绝对路径转成相对项目根目录的写法，用于打印。"""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def setup_plot_style() -> None:
    """统一的浅色系绘图风格。"""
    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 1.1,
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.9,
        "grid.alpha": 0.9,
        "xtick.color": PALETTE["slate"],
        "ytick.color": PALETTE["slate"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
    })


def polish(ax, title=None, xlabel=None, ylabel=None, hide=("top", "right")):
    """去掉多余边框、统一标题，返回 ax。"""
    for s in hide:
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def save_fig(fig, name: str) -> str:
    """保存到 img/ 并返回路径字符串。"""
    path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"[图片已保存] {rel(path)}")
    return str(path)
