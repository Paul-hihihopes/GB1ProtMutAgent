"""特征工程：把 4 位点组合编码成机器学习可用的数值向量。

三种编码可叠加：
1. one-hot       : 4 位点 × 20 氨基酸 = 80 维，刻画"哪个位点是哪个残基"
2. physicochem   : 4 位点 × 7 个理化描述符 = 28 维，提供泛化到未见残基的能力
3. pair-interact : 6 个位点对 × (BLOSUM 和/体积差/疏水差...) = 30 维，刻画上位效应
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import AA_ALPHABET, MUT_POSITIONS, N_POS, WT_COMBO
from .knowledge import AA_PROPERTIES, NUMERIC_PROPS, blosum_score

AA_INDEX = {a: i for i, a in enumerate(AA_ALPHABET)}

# 标准化后的理化性质矩阵 (20 × 7)
_PROP_MATRIX = np.array([[AA_PROPERTIES[a][p] for p in NUMERIC_PROPS] for a in AA_ALPHABET],
                        dtype=float)
_PROP_MEAN = _PROP_MATRIX.mean(axis=0)
_PROP_STD = _PROP_MATRIX.std(axis=0) + 1e-9
_PROP_Z = (_PROP_MATRIX - _PROP_MEAN) / _PROP_STD

_PAIRS = list(combinations(range(N_POS), 2))


def _onehot(combo: str) -> np.ndarray:
    v = np.zeros(N_POS * len(AA_ALPHABET), dtype=np.float32)
    for i, aa in enumerate(combo):
        v[i * len(AA_ALPHABET) + AA_INDEX[aa]] = 1.0
    return v


def _physchem(combo: str) -> np.ndarray:
    return np.concatenate([_PROP_Z[AA_INDEX[aa]] for aa in combo]).astype(np.float32)


def _pairwise(combo: str) -> np.ndarray:
    """位点对之间的相互作用描述符，帮助线性模型捕捉部分上位效应。"""
    feats = []
    for i, j in _PAIRS:
        a, b = combo[i], combo[j]
        pa, pb = AA_PROPERTIES[a], AA_PROPERTIES[b]
        feats.extend([
            blosum_score(a, b) / 4.0,
            (pa["volume"] + pb["volume"]) / 200.0,
            abs(pa["volume"] - pb["volume"]) / 100.0,
            (pa["hydropathy"] + pb["hydropathy"]) / 5.0,
            pa["charge"] * pb["charge"],
        ])
    return np.asarray(feats, dtype=np.float32)


def feature_names(use_physchem: bool = True, use_pairwise: bool = True) -> list[str]:
    names = [f"onehot_P{MUT_POSITIONS[i]}_{aa}"
             for i in range(N_POS) for aa in AA_ALPHABET]
    if use_physchem:
        names += [f"prop_P{MUT_POSITIONS[i]}_{p}" for i in range(N_POS) for p in NUMERIC_PROPS]
    if use_pairwise:
        for i, j in _PAIRS:
            pi, pj = MUT_POSITIONS[i], MUT_POSITIONS[j]
            names += [f"pair_{pi}x{pj}_blosum", f"pair_{pi}x{pj}_volsum",
                      f"pair_{pi}x{pj}_voldiff", f"pair_{pi}x{pj}_hydrosum",
                      f"pair_{pi}x{pj}_chargeprod"]
    return names


def encode(variants, use_physchem: bool = True, use_pairwise: bool = True) -> np.ndarray:
    """把一批组合字符串编码成特征矩阵。"""
    if isinstance(variants, (str, bytes)):
        variants = [variants]
    rows = []
    for combo in variants:
        parts = [_onehot(combo)]
        if use_physchem:
            parts.append(_physchem(combo))
        if use_pairwise:
            parts.append(_pairwise(combo))
        rows.append(np.concatenate(parts))
    return np.vstack(rows).astype(np.float32)


def encode_dataframe(df: pd.DataFrame, **kw) -> np.ndarray:
    return encode(df["variant"].tolist(), **kw)


def describe_encoding(use_physchem: bool = True, use_pairwise: bool = True) -> None:
    names = feature_names(use_physchem, use_pairwise)
    n_oh = N_POS * len(AA_ALPHABET)
    n_pc = N_POS * len(NUMERIC_PROPS) if use_physchem else 0
    n_pr = len(_PAIRS) * 5 if use_pairwise else 0
    print(f"[特征] one-hot        : {n_oh} 维  ({N_POS} 位点 × {len(AA_ALPHABET)} 氨基酸)")
    print(f"[特征] 理化描述符     : {n_pc} 维  ({N_POS} 位点 × {len(NUMERIC_PROPS)} 性质: {NUMERIC_PROPS})")
    print(f"[特征] 位点对交互     : {n_pr} 维  ({len(_PAIRS)} 个位点对 × 5 个描述符)")
    print(f"[特征] 合计           : {len(names)} 维")
