"""对照基线策略，与 LLM Agent 使用完全相同的预测模型和候选空间。

1. Random        —— 随机突变（传统随机诱变 / 无模型指导）
2. Model-Greedy  —— 适应度模型直接推荐：对整个未测空间打分后取 Top-K
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BATCH_SIZE, RANDOM_SEED, TOP_K
from .data import from_log_fitness, hamming, mutation_label


@dataclass
class SimpleProposal:
    round_id: int
    strategy: str
    batch: pd.DataFrame
    top_k: pd.DataFrame
    n_candidates: int
    best_variant_before: str
    best_fitness_before: float
    critic_comment: str = ""
    hypotheses: list = None
    analysis: dict = None


def _mark_topk(batch: pd.DataFrame, k: int) -> pd.DataFrame:
    """显式标注批次内的 Top-k 及其名次。"""
    topk = batch.nlargest(k, "pred_log_fitness").reset_index(drop=True)
    batch["is_topk"] = batch.variant.isin(set(topk.variant))
    batch["topk_rank"] = batch.variant.map(
        {v: i + 1 for i, v in enumerate(topk.variant)}).astype("Int64")
    return topk


class RandomStrategy:
    """随机突变基线：在未测空间中均匀随机抽取。"""

    name = "Random"

    def __init__(self, batch_size: int = BATCH_SIZE, top_k: int = TOP_K, seed: int = RANDOM_SEED):
        self.batch_size, self.top_k = batch_size, top_k
        self.rng = np.random.default_rng(seed)

    def propose(self, measured: pd.DataFrame, model, allowed: set[str],
                round_id: int = 1, verbose: bool = True) -> SimpleProposal:
        pool = sorted(allowed - set(measured.variant))
        pick = self.rng.choice(pool, size=min(self.batch_size, len(pool)), replace=False)
        mu, sigma = model.predict_with_uncertainty(list(pick))
        batch = pd.DataFrame({
            "variant": pick,
            "mut_label": [mutation_label(v) for v in pick],
            "n_mut": [hamming(v) for v in pick],
            "pred_log_fitness": mu,
            "pred_fitness": from_log_fitness(mu),
            "uncertainty": sigma,
            "hypothesis_id": "RANDOM",
            "selection_reason": "随机抽取（无模型指导）",
        })
        bi = measured.fitness.idxmax()
        topk = _mark_topk(batch, self.top_k)
        return SimpleProposal(round_id, self.name, batch, topk,
                              len(pool), str(measured.loc[bi, "variant"]),
                              float(measured.loc[bi, "fitness"]),
                              critic_comment="随机基线，不做任何筛选。")


class ModelGreedyStrategy:
    """适应度模型直接推荐：对整个未测候选空间打分，取预测值最高的 B 条。"""

    name = "Model-Greedy"

    def __init__(self, batch_size: int = BATCH_SIZE, top_k: int = TOP_K,
                 max_pool: int = 200_000, seed: int = RANDOM_SEED):
        self.batch_size, self.top_k, self.max_pool = batch_size, top_k, max_pool
        self.rng = np.random.default_rng(seed)

    def propose(self, measured: pd.DataFrame, model, allowed: set[str],
                round_id: int = 1, verbose: bool = True) -> SimpleProposal:
        pool = sorted(allowed - set(measured.variant))
        if len(pool) > self.max_pool:
            pool = list(self.rng.choice(pool, size=self.max_pool, replace=False))
        # 先用快速预测扫全空间，只对头部候选估计不确定度（否则 15 万 × 60 棵树太慢）
        mu = model.predict(pool)
        df = pd.DataFrame({"variant": pool, "pred_log_fitness": mu})
        df = df.nlargest(self.batch_size, "pred_log_fitness").reset_index(drop=True)
        _, sigma = model.predict_with_uncertainty(df.variant.tolist())
        df["uncertainty"] = sigma
        df["pred_fitness"] = from_log_fitness(df.pred_log_fitness.values)
        df["mut_label"] = df.variant.map(mutation_label)
        df["n_mut"] = df.variant.map(hamming)
        df["hypothesis_id"] = "GREEDY"
        df["selection_reason"] = "模型预测适应度全局 Top-B（纯利用）"
        bi = measured.fitness.idxmax()
        if verbose:
            print(f"  [Model-Greedy] 对 {len(pool):,} 个未测候选打分，取 Top-{self.batch_size}")
        topk = _mark_topk(df, self.top_k)
        return SimpleProposal(round_id, self.name, df, topk,
                              len(pool), str(measured.loc[bi, "variant"]),
                              float(measured.loc[bi, "fitness"]),
                              critic_comment="纯模型贪心，不做规则审查与多样性控制。")
