"""数据加载、清洗、划分，以及"虚拟湿实验"预言机(Oracle)。

数据集: GB1 four-site combinatorial library (Wu et al. 2016)
        149,361 个变体覆盖 V39 / D40 / G41 / V54 四个位点的 20^4 组合空间。
        经 FLIP benchmark 整理后提供 Fitness(野生型 = 1.0)。

定向进化场景模拟:
    * "已完成实验" = 汉明距离 <= 2 的变体(野生型 + 单点 + 双点)，共 2168 条
      -> 训练集 / 验证集
    * "未知候选空间" = 汉明距离 >= 3 的变体(三点 + 四点)，共 147,193 条
      -> 测试集，同时充当后续每轮虚拟实验的真值来源
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (AA_ALPHABET, DATA_PROC, GB1_LENGTH, LOG_EPS, MUT_POSITIONS, RANDOM_SEED,
                     RAW_CSV, RAW_FASTA, TRAIN_MAX_HD, VAL_RATIO, WT_COMBO, rel)


# 基础工具
def validate_combo(combo: str) -> str:
    """验证四位点组合的长度和标准氨基酸字母表。"""
    if not isinstance(combo, str) or len(combo) != len(MUT_POSITIONS):
        raise ValueError("组合必须包含 4 个氨基酸，依次对应位点 39/40/41/54")
    if any(aa not in AA_ALPHABET for aa in combo):
        raise ValueError("组合只允许 20 种标准氨基酸，不允许终止符或非标准残基")
    return combo


def combo_to_mutations(combo: str, wt: str = WT_COMBO,
                       positions: Sequence[int] = MUT_POSITIONS) -> list[str]:
    """'FWAA' -> ['V39F', 'D40W', 'G41A', 'V54A']（只返回真正发生改变的位点）。"""
    validate_combo(combo)
    return [f"{wt[i]}{positions[i]}{combo[i]}" for i in range(len(wt)) if combo[i] != wt[i]]


def mutations_to_combo(mutations: Iterable[str], wt: str = WT_COMBO,
                       positions: Sequence[int] = MUT_POSITIONS) -> str:
    """['D40W', 'V54A'] -> 'VWGA'。"""
    chars = list(wt)
    pos_index = {p: i for i, p in enumerate(positions)}
    seen = set()
    for m in mutations:
        m = m.strip()
        pos = int(m[1:-1])
        if pos not in pos_index:
            raise ValueError(f"位点 {pos} 不在可突变位点 {list(positions)} 中")
        if m[0].upper() != wt[pos_index[pos]] or m[-1].upper() not in AA_ALPHABET:
            raise ValueError(f"突变记号不符合参考序列或标准氨基酸字母表：{m}")
        if pos in seen:
            raise ValueError(f"位点 {pos} 重复指定")
        seen.add(pos)
        chars[pos_index[pos]] = m[-1].upper()
    return "".join(chars)


def hamming(combo: str, wt: str = WT_COMBO) -> int:
    if len(combo) != len(wt):
        raise ValueError("计算汉明距离的两条序列必须等长")
    return sum(1 for a, b in zip(combo, wt) if a != b)


def mutation_label(combo: str) -> str:
    """用于展示：野生型显示为 'WT'，否则显示 'D40W + V54A'。"""
    muts = combo_to_mutations(combo)
    return "WT" if not muts else " + ".join(muts)


def to_log_fitness(fitness: np.ndarray | pd.Series | float) -> np.ndarray:
    """log10(fitness + eps)。单调变换，不影响 Spearman，但显著改善回归。"""
    return np.log10(np.asarray(fitness, dtype=float) + LOG_EPS)


def from_log_fitness(log_fitness: np.ndarray | pd.Series | float) -> np.ndarray:
    """log 空间还原为原始 fitness 尺度，便于解释。"""
    return np.clip(np.power(10.0, np.asarray(log_fitness, dtype=float)) - LOG_EPS, 0.0, None)


def read_wt_sequence(full_construct: bool = False) -> str:
    """读取 GB1 的 56 aa 参考序列，也可返回 FLIP 的完整融合序列表示。"""
    lines = RAW_FASTA.read_text(encoding="utf-8").strip().splitlines()
    sequence = "".join(l.strip() for l in lines if not l.startswith(">"))
    return sequence if full_construct else sequence[:GB1_LENGTH]


def build_full_sequence(combo: str, wt_seq: str | None = None) -> str:
    """把四位点组合写回完整蛋白序列。"""
    wt_seq = wt_seq or read_wt_sequence()
    validate_combo(combo)
    chars = list(wt_seq)
    for i, pos in enumerate(MUT_POSITIONS):
        chars[pos - 1] = combo[i]
    return "".join(chars)


# 加载与清洗
def load_dataset(verbose: bool = True) -> pd.DataFrame:
    """读取原始 csv，整理成统一的分析表。"""
    source = RAW_CSV if RAW_CSV.exists() else RAW_CSV.with_suffix(".csv.zip")
    if not source.exists():
        raise FileNotFoundError(f"缺少 GB1 数据文件：{source}")
    raw = pd.read_csv(source, compression="infer", low_memory=False)
    df = pd.DataFrame({
        "variant": raw["Variants"].astype(str).str.upper(),
        "fitness": raw["Fitness"].astype(float),
        "count_input": raw["Count input"].astype(float),
        "count_selected": raw["Count selected"].astype(float),
    })

    # 只保留由 20 种标准氨基酸组成的 4 位点组合
    valid_alpha = df["variant"].str.fullmatch(f"[{AA_ALPHABET}]{{4}}")
    dropped_alpha = int((~valid_alpha).sum())
    df = df[valid_alpha].copy()

    # 去重、去缺失、去负值
    before = len(df)
    df = df.dropna(subset=["fitness"])
    df = df[np.isfinite(df["fitness"]) & (df["fitness"] >= 0)]
    df = df.drop_duplicates(subset=["variant"], keep="first").reset_index(drop=True)
    dropped_quality = before - len(df)

    df["n_mut"] = df["variant"].map(hamming)
    df["mutations"] = df["variant"].map(lambda v: combo_to_mutations(v))
    df["mut_label"] = df["variant"].map(mutation_label)
    df["log_fitness"] = to_log_fitness(df["fitness"].values)
    # 相对野生型的倍数变化
    df["fold_wt"] = df["fitness"] / 1.0
    df = df.sort_values("variant").reset_index(drop=True)

    if verbose:
        print(f"[数据] 原始记录        : {len(raw):,}")
        print(f"[数据] 剔除非标准氨基酸: {dropped_alpha:,}")
        print(f"[数据] 剔除缺失/重复   : {dropped_quality:,}")
        print(f"[数据] 清洗后变体数    : {len(df):,}")
        print(f"[数据] 组合空间理论上限: {20 ** 4:,}  覆盖率 {len(df) / 20 ** 4:.1%}")
        print(f"[数据] fitness 范围    : {df.fitness.min():.4f} ~ {df.fitness.max():.4f}"
              f"  (野生型 = 1.0)")
        print(f"[数据] 零适应度变体(fit=0) : {(df.fitness == 0).sum():,}"
              f"  ({(df.fitness == 0).mean():.1%})")
    return df


# 数据划分
def make_splits(df: pd.DataFrame, max_hd: int = TRAIN_MAX_HD,
                val_ratio: float = VAL_RATIO, seed: int = RANDOM_SEED,
                verbose: bool = True) -> dict[str, pd.DataFrame]:
    """按突变阶数划分，模拟真实定向进化的信息可得性。

    训练/验证 : n_mut <= max_hd  —— 实验室"第一轮已经测过"的低阶突变
    测试/候选 : n_mut >  max_hd  —— 尚未测量的高阶组合空间
    """
    rng = np.random.default_rng(seed)
    known = df[df.n_mut <= max_hd].copy()
    unknown = df[df.n_mut > max_hd].copy()

    # 按突变阶数分层抽验证集，保证单点/双点比例一致
    val_idx = []
    for hd, grp in known.groupby("n_mut"):
        if hd == 0:          # 野生型始终留在训练集
            continue
        k = max(1, int(round(len(grp) * val_ratio)))
        val_idx.extend(rng.choice(grp.index.values, size=k, replace=False).tolist())
    val = known.loc[sorted(val_idx)].copy()
    train = known.drop(index=val_idx).copy()

    splits = {"train": train.reset_index(drop=True),
              "val": val.reset_index(drop=True),
              "test": unknown.reset_index(drop=True),
              "known": known.reset_index(drop=True)}

    if verbose:
        print("\n[划分] 模拟场景：低阶突变 = 已完成实验，高阶组合 = 未知候选空间")
        for name in ("train", "val", "test"):
            s = splits[name]
            dist = s.n_mut.value_counts().sort_index().to_dict()
            print(f"  {name:<6}: n={len(s):>7,}  阶数分布={dist}  "
                  f"最高 fitness={s.fitness.max():.3f}")
        print(f"  训练可见的最优变体 : {train.loc[train.fitness.idxmax(), 'variant']}"
              f" ({train.fitness.max():.3f})")
        print(f"  候选空间真实最优   : {unknown.loc[unknown.fitness.idxmax(), 'variant']}"
              f" ({unknown.fitness.max():.3f})  <-- Agent 需要找到它")
    return splits


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    cols = ["variant", "mut_label", "n_mut", "fitness", "log_fitness"]
    for name, s in splits.items():
        if name == "known":
            continue
        path = DATA_PROC / f"gb1_{name}.csv"
        s[cols].to_csv(path, index=False)
        print(f"[保存] {rel(path)}  ({len(s):,} 行)")


# 虚拟实验预言机
@dataclass
class VirtualLabOracle:
    """虚拟湿实验平台。

    文库覆盖理论组合空间的约 93.4%。离线实验仅在有实测记录的候选集合内开展，
    每轮选择完成后才查询这些候选的真实适应度。
    """
    table: dict[str, float]
    measured: set[str]
    n_assays: int = 0

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, already_measured: Iterable[str] = ()):
        return cls(table=dict(zip(df.variant, df.fitness)),
                   measured=set(already_measured))

    def is_known(self, variant: str) -> bool:
        return variant in self.table

    def was_measured(self, variant: str) -> bool:
        return variant in self.measured

    def assay(self, variants: Sequence[str]) -> pd.DataFrame:
        """对一批候选做"虚拟实验"，返回真实 fitness。"""
        variants = list(variants)
        if len(set(variants)) != len(variants):
            raise ValueError("送检批次包含重复候选")
        if any(v not in self.table or v in self.measured for v in variants):
            raise ValueError("送检候选必须有实测记录且尚未测量")
        rows = []
        for v in variants:
            self.n_assays += 1
            self.measured.add(v)
            rows.append({"variant": v,
                         "mut_label": mutation_label(v),
                         "n_mut": hamming(v),
                         "fitness": self.table[v],
                         "log_fitness": float(to_log_fitness(self.table[v]))})
        return pd.DataFrame(rows)

    def unmeasured_pool(self, candidate_space: Sequence[str]) -> list[str]:
        return [v for v in candidate_space if v not in self.measured]

    def true_fitness(self, variant: str) -> float | None:
        return self.table.get(variant)
