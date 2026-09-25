"""虚拟定向进化实验：多轮"提出假设 → 设计突变 → 虚拟实验 → 反馈迭代"闭环。

每一轮的流程：
    1) 策略基于当前已测数据提出一批候选（batch_size 条）
    2) VirtualLabOracle 查出这批候选的真实 fitness（= 虚拟湿实验）
    3) 新数据并入训练集，重新训练适应度预测模型
    4) 记录本轮指标，进入下一轮

所有策略共享同一份初始数据、同一个候选空间、同一种模型超参，保证可比。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import (BATCH_SIZE, MUT_POSITIONS, N_ROUNDS, RANDOM_SEED, RESULT_DIR,
                     WT_COMBO, WT_FITNESS, rel)
from .data import VirtualLabOracle, combo_to_mutations, hamming, mutation_label
from .models import FitnessModel


@dataclass
class CampaignResult:
    strategy: str
    rounds: list[dict] = field(default_factory=list)
    measured_log: list[pd.DataFrame] = field(default_factory=list)
    proposals: list = field(default_factory=list)
    final_measured: pd.DataFrame = None
    run_id: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rounds)


def run_campaign(strategy, initial_measured: pd.DataFrame, oracle: VirtualLabOracle,
                 allowed: set[str], holdout: pd.DataFrame,
                 n_rounds: int = N_ROUNDS, model_kind: str = "xgboost",
                 seed: int = RANDOM_SEED, verbose: bool = True) -> CampaignResult:
    """跑完一个策略的完整定向进化 campaign。"""
    name = getattr(strategy, "name", None) or getattr(getattr(strategy, "cfg", None), "name", "strategy")
    client = getattr(strategy, "llm", None)
    run_id = client.begin_run("campaign") if client is not None else uuid4().hex
    measured = initial_measured.copy().reset_index(drop=True)
    oracle = VirtualLabOracle(table=dict(oracle.table), measured=set(measured.variant))

    model = FitnessModel(kind=model_kind, seed=seed).fit(measured)
    result = CampaignResult(strategy=name, run_id=run_id)

    baseline_best = float(measured.fitness.max())
    hold = holdout[~holdout.variant.isin(measured.variant)]
    m0 = model.evaluate(hold, ks=(10, 50, 100))

    result.rounds.append({
        "round": 0, "strategy": name,
        "n_measured": int(len(measured)),
        "n_new": 0,
        "best_fitness": baseline_best,
        "best_variant": str(measured.loc[measured.fitness.idxmax(), "variant"]),
        "batch_max_fitness": np.nan, "batch_mean_fitness": np.nan,
        "batch_median_fitness": np.nan,
        "n_above_wt": int((measured.fitness > WT_FITNESS).sum()),
        "n_hits_in_batch": 0, "hit_rate": np.nan,
        "cum_hits": 0,
        "n_above_wt_in_batch": 0, "frac_above_wt_in_batch": np.nan,
        "n_dead_in_batch": 0, "batch_pred_mae": np.nan, "n_unique_p40": 0,
        **{f"touch_pos{p}": 0 for p in MUT_POSITIONS},
        "model_spearman": m0["spearman"], "model_recall@100": m0["recall@100"],
        "improvement_over_start": 0.0,
    })

    if verbose:
        print(f"\n{'=' * 96}")
        print(f"策略 [{name}] campaign 开始：初始已测 {len(measured)}，"
              f"初始最优 {result.rounds[0]['best_variant']} = {baseline_best:.3f}，"
              f"初始模型 Spearman={m0['spearman']:.4f}")
        print(f"{'=' * 96}")

    cum_hits = 0
    hit_threshold = baseline_best          # "命中" 定义：真实 fitness 超过起始最优

    for r in range(1, n_rounds + 1):
        prop = strategy.propose(measured, model, allowed, round_id=r, verbose=verbose)
        batch_variants = prop.batch.variant.tolist()
        expected_batch = getattr(strategy, "batch_size", None) or getattr(
            getattr(strategy, "cfg", None), "batch_size", BATCH_SIZE)
        if len(batch_variants) != expected_batch or len(set(batch_variants)) != expected_batch:
            raise ValueError("送检批次必须满足约定预算且不包含重复候选")
        if not set(batch_variants) <= allowed - set(measured.variant):
            raise ValueError("送检候选必须位于尚未测量的允许集合中")
        if "verdict" in prop.batch and prop.batch.verdict.eq("reject").any():
            raise ValueError("被审稿模块拒绝的候选不可送检")

        # 虚拟湿实验
        assayed = oracle.assay(batch_variants)
        merged = prop.batch.merge(assayed[["variant", "fitness", "log_fitness"]],
                                  on="variant", how="left")
        merged["true_fitness"] = merged["fitness"]
        merged["pred_error"] = merged["pred_fitness"] - merged["true_fitness"]
        merged["is_hit"] = merged["true_fitness"] > hit_threshold
        prop.batch = merged
        if hasattr(prop, "top_k") and prop.top_k is not None and len(prop.top_k):
            prop.top_k = prop.top_k.merge(assayed[["variant", "fitness"]], on="variant", how="left") \
                                   .rename(columns={"fitness": "true_fitness"})

        n_hits = int(merged.is_hit.sum())
        cum_hits += n_hits

        # 批次层面的探索行为刻画：改动了哪些位点、有多少条突破野生型
        pos_touch = {p: 0 for p in MUT_POSITIONS}
        for v in batch_variants:
            for i, p in enumerate(MUT_POSITIONS):
                if v[i] != WT_COMBO[i]:
                    pos_touch[p] += 1
        n_above_wt_batch = int((merged.true_fitness > WT_FITNESS).sum())
        n_dead_batch = int((merged.true_fitness < 0.1 * WT_FITNESS).sum())

        # 并入训练集，重新训练
        measured = pd.concat([measured, assayed], ignore_index=True)
        measured["mutations"] = measured.variant.map(combo_to_mutations)
        measured["n_mut"] = measured.variant.map(hamming)
        model = FitnessModel(kind=model_kind, seed=seed).fit(measured)

        hold = holdout[~holdout.variant.isin(measured.variant)]
        mm = model.evaluate(hold, ks=(10, 50, 100))

        best_i = measured.fitness.idxmax()
        row = {
            "round": r, "strategy": name,
            "backend": getattr(prop, "backend", "model"),
            "n_rejected": getattr(prop, "n_rejected", 0),
            "n_measured": int(len(measured)),
            "n_new": int(len(assayed)),
            "best_fitness": float(measured.fitness.max()),
            "best_variant": str(measured.loc[best_i, "variant"]),
            "batch_max_fitness": float(merged.true_fitness.max()),
            "batch_mean_fitness": float(merged.true_fitness.mean()),
            "batch_median_fitness": float(merged.true_fitness.median()),
            "n_above_wt": int((measured.fitness > WT_FITNESS).sum()),
            "n_hits_in_batch": n_hits,
            "hit_rate": n_hits / max(1, len(merged)),
            "cum_hits": cum_hits,
            "n_above_wt_in_batch": n_above_wt_batch,
            "frac_above_wt_in_batch": n_above_wt_batch / max(1, len(merged)),
            "n_dead_in_batch": n_dead_batch,
            "batch_pred_mae": float((merged.pred_fitness - merged.true_fitness).abs().mean()),
            "n_unique_p40": int(merged.variant.str[1].nunique()),
            **{f"touch_pos{p}": pos_touch[p] for p in MUT_POSITIONS},
            "model_spearman": mm["spearman"], "model_recall@100": mm["recall@100"],
            "improvement_over_start": float(measured.fitness.max() - baseline_best),
        }
        result.rounds.append(row)
        result.proposals.append(prop)

        if verbose:
            print(f"  ➤ 虚拟实验回读：批次最高真实适应度 {row['batch_max_fitness']:.3f}，"
                  f"均值 {row['batch_mean_fitness']:.3f}，命中(>{hit_threshold:.2f}) {n_hits}/{len(merged)}")
            print(f"  ➤ 累计最优 {row['best_variant']}({mutation_label(row['best_variant'])}) "
                  f"= {row['best_fitness']:.3f}  (相对起点 {row['improvement_over_start']:+.3f})")
            print(f"  ➤ 模型更新后 Spearman={mm['spearman']:.4f} (Δ{mm['spearman'] - m0['spearman']:+.4f})，"
                  f"recall@100={mm['recall@100']:.3f}")

    result.final_measured = measured
    return result


def campaigns_to_frame(results: list[CampaignResult]) -> pd.DataFrame:
    return pd.concat([r.to_frame() for r in results], ignore_index=True)


def summarize_campaigns(results: list[CampaignResult], verbose: bool = True) -> pd.DataFrame:
    rows = []
    for res in results:
        df = res.to_frame()
        start, end = df.iloc[0], df.iloc[-1]
        batch_rows = df[df["round"] > 0]
        rows.append({
            "strategy": res.strategy,
            "start_best": start.best_fitness,
            "final_best": end.best_fitness,
            "absolute_gain": end.best_fitness - start.best_fitness,
            "fold_improvement": end.best_fitness / max(start.best_fitness, 1e-9),
            "final_best_variant": end.best_variant,
            "total_assays": int(batch_rows.n_new.sum()),
            "total_hits": int(end.cum_hits),
            "overall_hit_rate": end.cum_hits / max(1, int(batch_rows.n_new.sum())),
            "mean_batch_fitness": float(batch_rows.batch_mean_fitness.mean()),
            "best_round1": float(batch_rows.iloc[0].batch_max_fitness),
            "frac_above_wt_in_batch": float(batch_rows.frac_above_wt_in_batch.mean()),
            "n_dead_total": int(batch_rows.n_dead_in_batch.sum()),
            "mean_batch_pred_mae": float(batch_rows.batch_pred_mae.mean()),
            "touch_pos41_total": int(batch_rows.get("touch_pos41", pd.Series([0])).sum()),
            "model_spearman_start": start.model_spearman,
            "model_spearman_end": end.model_spearman,
            "model_spearman_gain": end.model_spearman - start.model_spearman,
        })
    out = pd.DataFrame(rows).sort_values("final_best", ascending=False).reset_index(drop=True)
    if verbose:
        print("\n" + "=" * 110)
        print("四种策略的定向进化 campaign 总结（相同预算：每轮 %d 次虚拟实验 × %d 轮）"
              % (BATCH_SIZE, N_ROUNDS))
        print("=" * 110)
        for _, r in out.iterrows():
            print(f"  {r.strategy:<14} 最终最优 {r.final_best:6.3f} ({r.final_best_variant})  "
                  f"提升 {r.absolute_gain:+6.3f} ({r.fold_improvement:.2f}×)  "
                  f"命中率 {r.overall_hit_rate:5.1%}  "
                  f"批次均值 {r.mean_batch_fitness:6.3f}  "
                  f"模型 Spearman {r.model_spearman_start:.3f}→{r.model_spearman_end:.3f}")
    return out


def run_replicates(strategy_factory, initial_measured: pd.DataFrame, oracle: VirtualLabOracle,
                   allowed: set[str], holdout: pd.DataFrame, seeds: list[int],
                   n_rounds: int = N_ROUNDS, model_kind: str = "xgboost",
                   verbose: bool = True) -> pd.DataFrame:
    """同一策略在多个随机种子下重复实验，用于给出带误差棒的结论。

    strategy_factory(seed) -> 一个全新的策略实例（内部随机数与模型种子都由 seed 决定）。
    单次 campaign 只做 36 次虚拟实验，方差很大，因此结论必须建立在重复实验上。
    """
    rows = []
    for seed in seeds:
        strat = strategy_factory(seed)
        res = run_campaign(strat, initial_measured, oracle, allowed, holdout,
                           n_rounds=n_rounds, model_kind=model_kind, seed=seed, verbose=False)
        df = res.to_frame()
        df["seed"] = seed
        rows.append(df)
        if verbose:
            end = df.iloc[-1]
            print(f"    seed={seed:<4} 最终最优 {end.best_fitness:6.3f} ({end.best_variant})  "
                  f"累计命中 {int(end.cum_hits):>2}  模型 Spearman {end.model_spearman:.4f}")
    return pd.concat(rows, ignore_index=True)


def aggregate_replicates(rep: pd.DataFrame, target_fitness: float | None = None,
                         batch_size: int = BATCH_SIZE, verbose: bool = True) -> pd.DataFrame:
    """把多种子结果聚合成 均值 ± 标准差。

    target_fitness 给定时，额外统计两个定向进化最关心的指标：
        success_rate    —— 多少比例的重复实验最终达到了该适应度（通常取全局最优）
        assays_to_target—— 成功重复首次达到目标时的平均累计实验数，按整批计数
    """
    final = rep[rep["round"] == rep["round"].max()]
    agg = final.groupby("strategy").agg(
        n_seeds=("seed", "nunique"),
        final_best_mean=("best_fitness", "mean"),
        final_best_std=("best_fitness", "std"),
        final_best_max=("best_fitness", "max"),
        final_best_min=("best_fitness", "min"),
        cum_hits_mean=("cum_hits", "mean"),
        cum_hits_std=("cum_hits", "std"),
        model_spearman_mean=("model_spearman", "mean"),
    ).reset_index()

    batch = rep[rep["round"] > 0].groupby("strategy").agg(
        batch_mean_fitness=("batch_mean_fitness", "mean"),
        frac_above_wt=("frac_above_wt_in_batch", "mean"),
        n_dead_per_batch=("n_dead_in_batch", "mean"),
        touch_pos41=("touch_pos41", "mean"),
        pred_mae=("batch_pred_mae", "mean"),
    ).reset_index()

    out = agg.merge(batch, on="strategy")

    if target_fitness is not None:
        max_round = int(rep["round"].max())
        budget = max_round * batch_size
        srows = []
        for (strat, seed), g in rep.groupby(["strategy", "seed"]):
            g = g.sort_values("round")
            reached = g[g.best_fitness >= target_fitness - 1e-9]
            first = int(reached["round"].iloc[0]) if len(reached) else None
            srows.append({"strategy": strat, "seed": seed,
                          "reached": first is not None,
                          "assays": first * batch_size if first is not None else np.nan})
        s = pd.DataFrame(srows).groupby("strategy").agg(
            success_rate=("reached", "mean"),
            n_success=("reached", "sum"),
            assays_to_target=("assays", "mean")).reset_index()
        out = out.merge(s, on="strategy")

    out = out.sort_values(["final_best_mean", "cum_hits_mean"], ascending=False).reset_index(drop=True)

    if verbose:
        n = int(out.n_seeds.iloc[0])
        print("\n" + "=" * 118)
        print(f"多种子重复实验汇总（{n} 个随机种子 × 每轮 {batch_size} 次虚拟实验，均值 ± 标准差）")
        print("=" * 118)
        head = (f"  {'策略':<14}{'最终最优':>17}{'累计命中':>13}{'批次均值':>11}"
                f"{'超野生型':>11}{'批次低功能':>10}{'触及位点41':>12}")
        if target_fitness is not None:
            head += f"{'成功率':>10}{'成功时实验数':>12}"
        print(head)
        for _, r in out.iterrows():
            line = (f"  {r.strategy:<14}"
                    f"{r.final_best_mean:8.3f} ± {r.final_best_std:5.3f}"
                    f"{r.cum_hits_mean:7.1f} ± {r.cum_hits_std:4.1f}"
                    f"{r.batch_mean_fitness:11.3f}"
                    f"{r.frac_above_wt:11.1%}"
                    f"{r.n_dead_per_batch:10.2f}"
                    f"{r.touch_pos41:12.2f}")
            if target_fitness is not None:
                line += f"{r.success_rate:10.0%}{r.assays_to_target:12.1f}"
            print(line)
        if target_fitness is not None:
            print(f"\n  注：成功率 = 在 {max_round} 轮 × {batch_size} 次实验的预算内找到真实全局最优"
                  f"(fitness={target_fitness:.3f}) 的重复实验比例；"
                  "实验数只对成功重复求平均，按首次达到目标的整批实验计数；"
                  "全部未找到时记为缺失，需结合成功率解读。")
    return out


def save_campaign_artifacts(results: list[CampaignResult], summary: pd.DataFrame,
                            path=None) -> str:
    """导出实验结果 JSON。"""
    path = path or (RESULT_DIR / "campaigns.json")
    payload = {"summary": summary.to_dict("records"), "rounds": [], "proposals": [], "runs": []}
    for res in results:
        payload["runs"].append({"run_id": res.run_id, "phase": "campaign", "strategy": res.strategy})
        payload["rounds"].extend(res.to_frame().assign(run_id=res.run_id).to_dict("records"))
        for p in res.proposals:
            cols = [c for c in ["variant", "mut_label", "n_mut", "pred_fitness", "uncertainty",
                                "true_fitness", "is_hit", "is_topk", "topk_rank",
                                "hypothesis_id", "source_type", "selection_reason",
                                "verdict", "recommendation_reason", "risk", "design_note",
                                "rule_pass", "rule_warnings"] if c in p.batch.columns]
            payload["proposals"].append({
                "strategy": res.strategy,
                "run_id": res.run_id,
                "phase": "campaign",
                "backend": getattr(p, "backend", "model"),
                "n_rejected": getattr(p, "n_rejected", 0),
                "round": p.round_id,
                "best_variant_before": p.best_variant_before,
                "best_fitness_before": p.best_fitness_before,
                "n_candidates": p.n_candidates,
                "critic_comment": p.critic_comment,
                "hypotheses": getattr(p, "hypotheses", None) or [],
                "analysis_summary": (getattr(p, "analysis", None) or {}).get("llm", {}).get("summary", ""),
                "analysis_observations": (getattr(p, "analysis", None) or {}).get("llm", {}).get("observations", []),
                # 统一把 np.nan 与可空整型的 pd.NA 转成 None，便于 JSON 序列化
                "batch": (p.batch[cols].astype(object)
                          .where(p.batch[cols].notna(), None).to_dict("records")),
            })
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(finite_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[保存] campaign 结果 -> {rel(path)}")
    return str(path)
