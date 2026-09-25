"""科学智能体：模拟蛋白质工程师的定向进化思维流程。

五个模块：
    DataAnalyst         读取当前实验数据与 top variants，做统计分析
    HypothesisGenerator 总结哪些突变位点/残基可能有益，形成可证伪的假设
    MutationDesigner    把假设翻译成具体的候选突变序列
    FitnessEvaluator    调用适应度预测模型给候选打分(含不确定度)
    ScientificCritic    用知识库规则审查候选、保证批次多样性、给出推荐理由

知识增强开关 use_knowledge:
    False -> 只看历史数据的边际统计，prompt 里不含任何理化性质/结构/规则信息
    True  -> 注入氨基酸理化性质、GB1 结构注释、突变规则库、知识图谱事实，
             并额外挖掘双突变中的上位效应(epistasis)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
import pandas as pd

from .config import (AA_ALPHABET, BATCH_SIZE, MUT_POSITIONS, RANDOM_SEED, TOP_K,
                     WT_COMBO, WT_FITNESS)
from .data import (combo_to_mutations, from_log_fitness, hamming, mutation_label,
                   mutations_to_combo, to_log_fitness)
from .knowledge import (AA_PROPERTIES, STRUCTURAL_CONTEXT, DesignRuleBook,
                        KnowledgeGraph, describe_substitution, substitution_class)
from .llm import LLMClient
from .schemas import (matches_hypothesis, valid_analysis, valid_candidates,
                      valid_hypotheses, valid_reviews)


# 分析工具
def marginal_table(measured: pd.DataFrame) -> pd.DataFrame:
    """每个 (位点, 氨基酸) 在已测数据中的边际表现。"""
    rows = []
    for i, pos in enumerate(MUT_POSITIONS):
        for aa in AA_ALPHABET:
            sub = measured[measured.variant.str[i] == aa]
            if len(sub) == 0:
                continue
            rows.append({
                "position": pos, "aa": aa, "n": len(sub),
                "mean_log_fitness": float(sub.log_fitness.mean()),
                "max_fitness": float(sub.fitness.max()),
                "mean_fitness": float(sub.fitness.mean()),
                "is_wt": aa == WT_COMBO[i],
            })
    return pd.DataFrame(rows)


def position_sensitivity(measured: pd.DataFrame) -> pd.DataFrame:
    """位点画像。

    这里刻意区分两个容易混淆的概念：

    * tolerance_range —— 该位点上不同残基造成的适应度波动幅度（"这个位点重不重要"）。
      它同时被有益替换和低功能替换拉大，因此**不能**直接用来决定往哪里优化：
      一个全是低功能替换的位点同样会有很大的波动。
    * opt_potential   —— 该位点最好的单点替换相对野生型的 log 增益（"这个位点有没有上升空间"）。
      定向进化真正要排序的是这个指标。

    GB1 数据是一个很好的反例：位点 41 的波动极大（18/19 个单点低功能），
    但上升空间为负；位点 40 波动不大，却贡献了 15 个有益单点。
    """
    mt = marginal_table(measured)
    wt_log = float(to_log_fitness(WT_FITNESS))
    rows = []
    for i, pos in enumerate(MUT_POSITIONS):
        sub = mt[mt.position == pos]
        wt_val = sub[sub.is_wt].mean_log_fitness
        wt_val = float(wt_val.iloc[0]) if len(wt_val) else np.nan
        singles = measured[(measured.n_mut == 1) &
                           (measured.variant.str[i] != WT_COMBO[i])]
        non_wt = sub[~sub.is_wt]
        single_max = float(singles.fitness.max()) if len(singles) else np.nan
        rows.append({
            "position": pos,
            "wt_aa": WT_COMBO[i],
            "n_residues_seen": int(sub.n.gt(0).sum()),
            "std_log_fitness": float(sub.mean_log_fitness.std()),
            "tolerance_range": float(sub.mean_log_fitness.max() - sub.mean_log_fitness.min()),
            "opt_potential": float(to_log_fitness(single_max) - wt_log) if len(singles) else np.nan,
            "best_aa": non_wt.loc[non_wt.mean_log_fitness.idxmax(), "aa"] if len(non_wt) else None,
            "best_aa_mean_fitness": float(from_log_fitness(non_wt.mean_log_fitness.max())) if len(non_wt) else np.nan,
            "best_single_mutation": (f"{WT_COMBO[i]}{pos}"
                                     f"{singles.loc[singles.fitness.idxmax(), 'variant'][i]}"
                                     if len(singles) else None),
            "n_beneficial_singles": int((singles.fitness > WT_FITNESS).sum()) if len(singles) else 0,
            "n_lethal_singles": int((singles.fitness < 0.1 * WT_FITNESS).sum()) if len(singles) else 0,
            "single_max_fitness": single_max,
            "wt_mean_log_fitness": wt_val,
        })
    df = pd.DataFrame(rows)
    df["potential_rank"] = df.opt_potential.rank(ascending=False).astype(int)
    df["tolerance_rank"] = df.tolerance_range.rank(ascending=False).astype(int)
    return df.sort_values("opt_potential", ascending=False).reset_index(drop=True)


def epistasis_table(measured: pd.DataFrame, min_fitness: float = 0.0) -> pd.DataFrame:
    """从双突变体中提取上位效应。

    加性期望: log(AB) ≈ log(A) + log(B) - log(WT)
    epistasis = 观测 - 期望；> 0 表示正向上位(1+1>2)，是组合设计最有价值的信号。
    """
    singles = {}
    for _, r in measured[measured.n_mut == 1].iterrows():
        singles[r.mutations[0]] = r.log_fitness
    wt_log = float(to_log_fitness(WT_FITNESS))

    rows = []
    for _, r in measured[measured.n_mut == 2].iterrows():
        m1, m2 = r.mutations
        if m1 not in singles or m2 not in singles:
            continue
        expected = singles[m1] + singles[m2] - wt_log
        rows.append({
            "variant": r.variant, "mut_a": m1, "mut_b": m2,
            "pos_a": int(m1[1:-1]), "pos_b": int(m2[1:-1]),
            "obs_log_fitness": float(r.log_fitness),
            "exp_log_fitness": float(expected),
            "epistasis": float(r.log_fitness - expected),
            "obs_fitness": float(r.fitness),
            "single_a_fitness": float(from_log_fitness(singles[m1])),
            "single_b_fitness": float(from_log_fitness(singles[m2])),
        })
    df = pd.DataFrame(rows, columns=["variant", "mut_a", "mut_b", "pos_a", "pos_b",
                                    "obs_log_fitness", "exp_log_fitness", "epistasis",
                                    "obs_fitness", "single_a_fitness", "single_b_fitness"])
    if len(df):
        a, b, ab = df.single_a_fitness, df.single_b_fitness, df.obs_fitness
        df["sign_epistasis"] = (((a - WT_FITNESS) * (ab - b) < 0)
                                 | ((b - WT_FITNESS) * (ab - a) < 0))
        df = df[df.obs_fitness >= min_fitness]
    else:
        df["sign_epistasis"] = pd.Series(dtype=bool)
    return df.sort_values("epistasis", ascending=False).reset_index(drop=True)


def pair_preference(epi: pd.DataFrame, top: int = 40) -> list[dict]:
    """筛选真正值得作为设计骨架的突变对。

    只看 epistasis（观测 − 加性期望）会被"从低功能中被拯救"的组合刷屏：
    两个单点都接近 0 时，任何一点活性都会产生巨大的偏差值，但绝对适应度可能只有 0.7。
    因此这里用 design_score = 观测 log 适应度 + 0.5 × 上位效应 来排序，
    既要求组合本身好用，又奖励超出加性预期的协同。
    """
    if len(epi) == 0:
        return []
    good = epi[(epi.epistasis > 0) & (epi.obs_fitness > WT_FITNESS)].copy()
    if len(good) == 0:
        return []
    good["design_score"] = good.obs_log_fitness + 0.5 * good.epistasis
    good = good.sort_values("design_score", ascending=False).head(top)
    return good[["mut_a", "mut_b", "obs_fitness", "epistasis",
                 "design_score", "sign_epistasis", "single_a_fitness",
                 "single_b_fitness"]].to_dict("records")


# Agent 配置
@dataclass
class AgentConfig:
    name: str = "LLM-Agent"
    use_knowledge: bool = False
    batch_size: int = BATCH_SIZE
    top_k: int = TOP_K
    n_candidates: int = 600          # 每轮生成的候选池规模
    exploration_ratio: float = 0.25  # 批次中留给"高不确定度探索"的比例
    # 批次多样性惩罚：每有一个已入选候选与它汉明距离 <= 1，就扣这么多分。
    # 这是通用的批量实验设计原则（避免一批做 12 个几乎相同的构建体），
    # 不属于蛋白质领域知识，因此两个 Agent 变体都启用，保证对比只反映知识增强本身。
    diversity_penalty: float = 0.12
    seed: int = RANDOM_SEED
    max_review_batches: int = 5


# 模块 1
class DataAnalyst:
    """读取当前实验数据与 top variants，做统计分析并用自然语言总结。"""

    role = "Data Analyst（实验数据分析师）"

    SYSTEM = (
        "你是一名蛋白质工程实验室的数据分析师。你的任务是阅读一轮定向进化的实验结果，"
        "客观地总结数据里呈现的规律：哪些位点对适应度影响最大、哪些残基替换有益、"
        "哪些组合出现了超出加性预期的效应。"
        "只陈述数据支持的结论，不要臆测。"
        "必须只输出一个 JSON 对象，字段为 "
        '{"summary": str, "key_positions": [int], "beneficial_residues": [{"position": int, "aa": str, "evidence": str}], '
        '"risky_positions": [int], "observations": [str]}。'
        "篇幅要求：summary 不超过 200 字；beneficial_residues 最多 8 条，evidence 每条不超过 40 字；"
        "observations 最多 6 条，每条不超过 80 字。不要输出 JSON 以外的任何文字。"
    )

    def __init__(self, llm: LLMClient, cfg: AgentConfig):
        self.llm, self.cfg = llm, cfg

    # 统计
    def compute(self, measured: pd.DataFrame) -> dict:
        sens = position_sensitivity(measured)
        marg = marginal_table(measured)
        top_var = measured.nlargest(12, "fitness")
        epi = epistasis_table(measured) if self.cfg.use_knowledge else pd.DataFrame()

        best_per_pos = {}
        for i, pos in enumerate(MUT_POSITIONS):
            sub = marg[marg.position == pos].sort_values("mean_log_fitness", ascending=False)
            best_per_pos[pos] = sub.head(6)[["aa", "mean_fitness", "max_fitness", "n"]].to_dict("records")

        singles = measured[measured.n_mut == 1]
        beneficial = singles[singles.fitness > WT_FITNESS].sort_values("fitness", ascending=False)
        lethal = singles[singles.fitness < 0.1 * WT_FITNESS]

        return {
            "n_measured": int(len(measured)),
            "best_variant": str(measured.loc[measured.fitness.idxmax(), "variant"]),
            "best_fitness": float(measured.fitness.max()),
            "median_fitness": float(measured.fitness.median()),
            "frac_above_wt": float((measured.fitness > WT_FITNESS).mean()),
            "sensitivity": sens.to_dict("records"),
            "best_residues_per_position": best_per_pos,
            "top_variants": top_var[["variant", "mut_label", "n_mut", "fitness"]].to_dict("records"),
            "beneficial_singles": [{"mutation": r.mutations[0], "fitness": float(r.fitness)}
                                   for _, r in beneficial.head(15).iterrows()],
            "lethal_singles": [r.mutations[0] for _, r in lethal.iterrows()],
            "epistasis_top": pair_preference(epi) if len(epi) else [],
            "n_sign_epistasis": int(epi.sign_epistasis.sum()) if len(epi) else 0,
        }

    # prompt
    def build_prompt(self, stats: dict) -> str:
        lines = [
            f"## 当前实验数据概况",
            f"- 已测量变体数: {stats['n_measured']}",
            f"- 当前最优变体: {stats['best_variant']} ({mutation_label(stats['best_variant'])})，"
            f"适应度 {stats['best_fitness']:.3f}（野生型 = 1.0）",
            f"- 适应度中位数: {stats['median_fitness']:.4f}；超过野生型的比例: {stats['frac_above_wt']:.1%}",
            "",
            "## 四个可突变位点的画像",
            "（上升空间 = 该位点最优单点替换相对野生型的 log 增益；"
            "波动幅度 = 该位点不同残基造成的适应度极差，被有益和低功能替换共同拉大）",
            "| 位点 | 野生型残基 | 上升空间 | 波动幅度 | 最优替换残基 | 有益单点数 | 低功能单点数 | 单点最高适应度 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in stats["sensitivity"]:
            lines.append(f"| {s['position']} | {s['wt_aa']} | {s['opt_potential']:+.3f} | "
                         f"{s['tolerance_range']:.3f} | {s['best_aa']} | "
                         f"{s['n_beneficial_singles']} | {s['n_lethal_singles']} | "
                         f"{s['single_max_fitness']:.3f} |")

        lines += ["", "## 各位点表现最好的残基（按已测数据的平均适应度）"]
        for pos, recs in stats["best_residues_per_position"].items():
            frag = ", ".join(f"{r['aa']}(均值{r['mean_fitness']:.2f}/最高{r['max_fitness']:.2f}, n={r['n']})"
                             for r in recs)
            lines.append(f"- 位点 {pos}: {frag}")

        lines += ["", "## 当前 Top 变体"]
        for r in stats["top_variants"][:10]:
            lines.append(f"- {r['variant']} [{r['mut_label']}] 突变数={r['n_mut']} 适应度={r['fitness']:.3f}")

        lines += ["", "## 有益单点突变（适应度 > 野生型）"]
        lines.append(", ".join(f"{b['mutation']}({b['fitness']:.2f})" for b in stats["beneficial_singles"]) or "无")
        lines += ["", "## 低功能单点突变（适应度 < 0.1）"]
        lines.append(", ".join(stats["lethal_singles"][:30]) or "无")

        if self.cfg.use_knowledge and stats["epistasis_top"]:
            lines += ["", "## 双突变中的上位效应（观测值 − 加性期望，>0 表示协同）"]
            for e in stats["epistasis_top"][:12]:
                tag = "【符号上位：突变效应随背景发生正负翻转】" if e["sign_epistasis"] else ""
                lines.append(f"- {e['mut_a']} + {e['mut_b']}: 组合适应度 {e['obs_fitness']:.3f}，"
                             f"上位效应 {e['epistasis']:+.3f}，设计价值评分 {e['design_score']:.3f} {tag}")
            lines.append(f"- 其中呈现符号上位效应的双突变共 {stats['n_sign_epistasis']} 对。")

        lines += ["", "请据此输出 JSON 分析结论。"]
        return "\n".join(lines)

    # 离线引擎
    def _offline(self, stats: dict) -> dict:
        sens = stats["sensitivity"]
        # 关键位点按"上升空间"排序，而不是按波动幅度 —— 后者会把全是低功能替换的位点排到前面
        key_pos = [s["position"] for s in sens if s["opt_potential"] > 0][:3] or [sens[0]["position"]]
        risky = [s["position"] for s in sens
                 if s["n_beneficial_singles"] == 0 and s["n_lethal_singles"] > 0]

        beneficial_residues = []
        for pos in key_pos:
            for r in stats["best_residues_per_position"][pos][:3]:
                if r["aa"] == WT_COMBO[MUT_POSITIONS.index(pos)]:
                    continue
                beneficial_residues.append({
                    "position": pos, "aa": r["aa"],
                    "evidence": f"位点 {pos} 上含 {r['aa']} 的已测变体平均适应度 {r['mean_fitness']:.2f}，"
                                f"最高 {r['max_fitness']:.2f}（n={r['n']}）"})

        obs = [
            f"已测 {stats['n_measured']} 个变体，仅 {stats['frac_above_wt']:.1%} 超过野生型，"
            f"说明该组合空间整体崎岖、大部分替换有害。",
            "按上升空间排序，位点优先级为 " +
            " > ".join(f"{s['position']}({s['opt_potential']:+.2f})" for s in sens) +
            f"，位点 {sens[0]['position']} 提供了最大的可优化余量"
            f"（最优单点 {sens[0]['best_single_mutation']} 适应度 {sens[0]['single_max_fitness']:.2f}）。",
            "需要注意波动幅度与上升空间是两回事：" +
            "、".join(f"位点 {s['position']} 波动 {s['tolerance_range']:.2f}/低功能单点 {s['n_lethal_singles']} 个"
                      for s in sorted(sens, key=lambda x: -x["tolerance_range"])[:2]) +
            "，波动大只说明该位点敏感，不代表可以往好的方向改。",
            f"当前最优变体 {stats['best_variant']}（{mutation_label(stats['best_variant'])}）"
            f"适应度 {stats['best_fitness']:.3f}，是后续组合设计的起点。",
        ]
        if risky:
            obs.append(f"位点 {risky} 在已测数据中没有任何有益单点替换，属于高风险位点，"
                       f"单独改动几乎必然有害。")
        if self.cfg.use_knowledge and stats["epistasis_top"]:
            e = stats["epistasis_top"][0]
            obs.append(f"检测到显著正向上位效应：{e['mut_a']} + {e['mut_b']} 的组合适应度达 "
                       f"{e['obs_fitness']:.3f}，比加性期望高 {e['epistasis']:+.3f}，"
                       f"提示位点间存在结构耦合，不能只按单点加性来设计。")
            sign_cases = [x for x in stats["epistasis_top"] if x["sign_epistasis"]]
            if sign_cases:
                s0 = sign_cases[0]
                obs.append(f"其中 {s0['mut_a']} + {s0['mut_b']} 的单点适应度分别为 "
                           f"{s0['single_a_fitness']:.3f} 和 {s0['single_b_fitness']:.3f}，"
                           f"组合为 {s0['obs_fitness']:.3f}；至少一个突变的效应随背景翻转。")

        summary = (
            f"本轮共掌握 {stats['n_measured']} 条实测数据。适应度景观高度不均匀，"
            f"位点 {sens[0]['position']}（野生型 {sens[0]['wt_aa']}）是最主要的可优化位点，"
            f"拥有 {sens[0]['n_beneficial_singles']} 个有益单点替换，最优替换残基为 {sens[0]['best_aa']}；"
            f"位点 {sens[-1]['position']} 上升空间最小（{sens[-1]['opt_potential']:+.2f}），"
            f"{sens[-1]['n_lethal_singles']} 个单点替换低功能。"
            f"当前最优变体为 {stats['best_variant']}，适应度 {stats['best_fitness']:.3f}。"
        )
        if self.cfg.use_knowledge and stats["epistasis_top"]:
            summary += "双突变数据显示存在明显的位点耦合，后续设计必须考虑上位效应而非简单叠加有益单点。"
        else:
            summary += "后续设计将以叠加历史有益单点突变为主线。"

        return {"summary": summary, "key_positions": key_pos,
                "beneficial_residues": beneficial_residues,
                "risky_positions": risky, "observations": obs}

    # 主流程
    def run(self, measured: pd.DataFrame, round_id: int = 0) -> dict:
        stats = self.compute(measured)
        prompt = self.build_prompt(stats)
        parsed = self.llm.chat("data_analysis", self.role, self.SYSTEM, prompt,
                               offline_handler=lambda: self._offline(stats),
                               round_id=round_id,
                               validate=valid_analysis)
        return {"stats": stats, "llm": parsed}


# 模块 2
class HypothesisGenerator:
    """把数据分析结论转化为可证伪的突变假设。"""

    role = "Hypothesis Generator（假设生成器）"

    SYSTEM_BASE = (
        "你是一名资深蛋白质工程科学家，正在设计定向进化的下一轮突变文库。"
        "请基于上一步的数据分析结论，提出若干条具体、可证伪的突变假设。"
        "每条假设都要说明：针对哪些位点、倾向使用哪些氨基酸、预期为什么会提高适应度。"
        "必须只输出一个 JSON 对象，字段为 "
        '{"hypotheses": [{"id": str, "statement": str, "positions": [int], '
        '"target_aas": {"位点": ["氨基酸"]}, "rationale": str, "confidence": float, "strategy": str}]}。'
        'strategy 取值限定为 "exploit"(利用已知有益突变) / "recombine"(重组历史优势变体) / '
        '"epistasis"(利用上位效应) / "explore"(探索未知区域)。'
        "篇幅要求：最多 5 条假设，statement 每条不超过 60 字，rationale 每条不超过 120 字。"
        "不要输出 JSON 以外的任何文字。"
    )

    SYSTEM_KB = (
        "\n你还可以使用以下背景知识来论证假设："
        "氨基酸理化性质（疏水性、侧链体积、电荷、芳香性、二级结构倾向）、"
        "BLOSUM62 保守/激进替换、GB1 各位点的结构环境，以及实验室的突变设计规则。"
        "请在 rationale 中显式引用这些知识。"
    )

    def __init__(self, llm: LLMClient, cfg: AgentConfig, kg: KnowledgeGraph | None = None,
                 rulebook: DesignRuleBook | None = None):
        self.llm, self.cfg, self.kg = llm, cfg, kg
        self.rulebook = rulebook

    @property
    def SYSTEM(self) -> str:
        return self.SYSTEM_BASE + (self.SYSTEM_KB if self.cfg.use_knowledge else "")

    # prompt
    def build_prompt(self, analysis: dict, best_variant: str,
                     scaffold_fitness: float | None = None, scaffold_source: str = "实测值") -> str:
        a, stats = analysis["llm"], analysis["stats"]
        lines = [
            "## 上一步的数据分析结论",
            f"摘要: {a['summary']}",
            f"关键位点: {a['key_positions']}",
            f"高风险位点: {a['risky_positions']}",
            "观察:",
        ]
        lines += [f"  - {o}" for o in a["observations"]]
        lines += ["", "## 数据支持的有益残基"]
        for b in a["beneficial_residues"]:
            lines.append(f"  - 位点 {b['position']} -> {b['aa']}；证据: {b['evidence']}")
        lines += ["", f"## 历史已测最优: {stats['best_variant']}，适应度 {stats['best_fitness']:.3f}",
                  f"## 本轮设计骨架: {best_variant} ({mutation_label(best_variant)})"]
        if scaffold_fitness is not None:
            lines.append(f"骨架适应度参照: {scaffold_fitness:.3f}，来源: {scaffold_source}。")
        lines.append("请从本轮设计骨架出发，区分该骨架与历史最优变体。")

        if self.cfg.use_knowledge:
            lines += ["", "## 位点结构环境（知识库）"]
            for pos in MUT_POSITIONS:
                c = STRUCTURAL_CONTEXT[pos]
                lines.append(f"  - 位点 {pos}（野生型 {c['wt']}）: {c['element']}，{c['burial']}。{c['note']}")

            lines += ["", "## 候选残基的理化性质（知识库）"]
            interesting = sorted({b["aa"] for b in a["beneficial_residues"]} |
                                 {"W", "Y", "F", "A", "L", "M", "C"})
            for aa in interesting:
                p = AA_PROPERTIES[aa]
                lines.append(f"  - {aa} ({p['tla']}): 疏水性 {p['hydropathy']:+.1f}, 体积 {p['volume']:.0f} Å³, "
                             f"电荷 {p['charge']:+.1f}, {'芳香族, ' if p['aromatic'] else ''}"
                             f"β股倾向 {p['sheet']:.2f}")

            if stats["epistasis_top"]:
                lines += ["", "## 已观测到的正向上位效应（知识图谱事实）"]
                for e in stats["epistasis_top"][:8]:
                    lines.append(f"  - ({e['mut_a']}) +（{e['mut_b']}）-> 适应度 {e['obs_fitness']:.3f}，"
                                 f"上位效应 {e['epistasis']:+.3f}"
                                 f"{'，属于符号上位效应' if e['sign_epistasis'] else ''}")

            lines += ["", "## 设计规则（硬约束必须满足，软约束用于风险与评分）"]
            lines += [f"  - {rule}" for rule in (self.rulebook or DesignRuleBook()).prompt_rules()]

        lines += ["", "请输出 4-5 条假设的 JSON。"]
        return "\n".join(lines)

    # 离线引擎
    def _offline(self, analysis: dict, best_variant: str) -> dict:
        a, stats = analysis["llm"], analysis["stats"]
        sens = stats["sensitivity"]
        hyps = []

        # H1 利用主导位点的最优残基
        top_pos = sens[0]["position"]
        idx = MUT_POSITIONS.index(top_pos)
        best_aas = [r["aa"] for r in stats["best_residues_per_position"][top_pos][:4]
                    if r["aa"] != WT_COMBO[idx]][:3]
        rat = (f"位点 {top_pos} 的上升空间为 {sens[0]['opt_potential']:+.2f}（四个位点中最高），"
               f"已有 {sens[0]['n_beneficial_singles']} 个有益单点替换，"
               f"其中最优的 {sens[0]['best_single_mutation']} 单独就把适应度带到 "
               f"{sens[0]['single_max_fitness']:.2f}。")
        if self.cfg.use_knowledge and best_aas:
            rat += "从理化性质看，" + describe_substitution(WT_COMBO[idx], top_pos, best_aas[0])
            rat += f"位点 {top_pos} 处于{STRUCTURAL_CONTEXT[top_pos]['element']}且{STRUCTURAL_CONTEXT[top_pos]['burial']}，能够容纳这类替换。"
        hyps.append(dict(id="H1", statement=f"固定位点 {top_pos} 为 {'/'.join(best_aas)} 可稳定提升适应度",
                         positions=[top_pos], target_aas={str(top_pos): best_aas},
                         rationale=rat, confidence=0.85, strategy="exploit"))

        # H2 叠加次优位点
        second = [s for s in sens[1:] if s["n_beneficial_singles"] > 0]
        if second:
            p2 = second[0]["position"]
            i2 = MUT_POSITIONS.index(p2)
            aas2 = [r["aa"] for r in stats["best_residues_per_position"][p2][:4]
                    if r["aa"] != WT_COMBO[i2]][:3]
            rat2 = (f"位点 {p2} 存在 {second[0]['n_beneficial_singles']} 个有益单点（最高 "
                    f"{second[0]['single_max_fitness']:.2f}），与位点 {top_pos} 的有益替换叠加"
                    f"有望获得加性增益。")
            if self.cfg.use_knowledge:
                rat2 += f"位点 {p2} 位于{STRUCTURAL_CONTEXT[p2]['element']}，{STRUCTURAL_CONTEXT[p2]['note']}"
            hyps.append(dict(id="H2", statement=f"在位点 {top_pos} 最优替换的基础上叠加位点 {p2} 的有益替换",
                             positions=[top_pos, p2],
                             target_aas={str(top_pos): best_aas, str(p2): aas2},
                             rationale=rat2, confidence=0.7, strategy="exploit"))

        # H3 重组历史优势变体
        top_muts = []
        for r in stats["top_variants"][:8]:
            top_muts.extend(combo_to_mutations(r["variant"]))
        uniq = sorted(set(top_muts))
        hyps.append(dict(id="H3", statement=f"重组当前 Top 变体中反复出现的突变 {uniq[:6]}",
                         positions=sorted({int(m[1:-1]) for m in uniq}),
                         target_aas={str(p): sorted({m[-1] for m in uniq if int(m[1:-1]) == p})
                                     for p in sorted({int(m[1:-1]) for m in uniq})},
                         rationale=f"Top 变体 {[r['variant'] for r in stats['top_variants'][:5]]} 中"
                                   f"高频出现这些突变，重组它们是定向进化中最经典的 DNA shuffling 思路。",
                         confidence=0.65, strategy="recombine"))

        # H4 上位效应（仅知识增强版本）
        if self.cfg.use_knowledge and stats["epistasis_top"]:
            sign_cases = [e for e in stats["epistasis_top"] if e["sign_epistasis"]] or stats["epistasis_top"]
            e = sign_cases[0]
            pos_set = sorted({int(e["mut_a"][1:-1]), int(e["mut_b"][1:-1])})
            rat4 = (f"{e['mut_a']} 与 {e['mut_b']} 的组合适应度 {e['obs_fitness']:.3f}，"
                    f"比加性期望高 {e['epistasis']:+.3f}。")
            if e["sign_epistasis"]:
                rat4 += ("这是典型的符号上位效应：单点有害的替换在特定伙伴存在时被补偿。"
                         "仅根据单点效应排序可能忽略这一组合，可将它作为待检验的设计骨架。")
            hyps.append(dict(id="H4", statement=f"以上位效应对 {e['mut_a']}+{e['mut_b']} 为骨架继续扩展",
                             positions=pos_set,
                             target_aas={str(int(e["mut_a"][1:-1])): [e["mut_a"][-1]],
                                         str(int(e["mut_b"][1:-1])): [e["mut_b"][-1]]},
                             rationale=rat4, confidence=0.6, strategy="epistasis"))

            # H5 高风险位点的条件性解锁
            risky = a["risky_positions"]
            if risky:
                rp = risky[0]
                ir = MUT_POSITIONS.index(rp)
                rescue = [e2 for e2 in stats["epistasis_top"]
                          if rp in (int(e2["mut_a"][1:-1]), int(e2["mut_b"][1:-1]))]
                aas_r = sorted({(e2["mut_a"] if int(e2["mut_a"][1:-1]) == rp else e2["mut_b"])[-1]
                                for e2 in rescue}) or ["A", "L", "S"]
                hyps.append(dict(
                    id="H5",
                    statement=f"在有上位效应证据的前提下，有条件地解锁高风险位点 {rp}",
                    positions=[rp],
                    target_aas={str(rp): aas_r[:4]},
                    rationale=(f"位点 {rp}（野生型 {WT_COMBO[ir]}）所有单点替换都有害，"
                               f"{STRUCTURAL_CONTEXT[rp]['note']}"
                               f"但双突变数据中已出现 {len(rescue)} 例被伙伴突变补偿的情况，"
                               f"说明该位点的有害效应是上下文依赖的。只有在与已验证的有益突变共同引入时才尝试。"),
                    confidence=0.45, strategy="epistasis"))
        else:
            # 无知识增强时的探索假设
            hyps.append(dict(id="H4", statement="在敏感性中等的位点上做随机探索以补充数据",
                             positions=[s["position"] for s in sens[1:3]],
                             target_aas={str(s["position"]): [r["aa"] for r in
                                                              stats["best_residues_per_position"][s["position"]][:4]]
                                         for s in sens[1:3]},
                             rationale="当前模型在这些位点的数据覆盖不足，增加样本有助于下一轮模型精度。",
                             confidence=0.4, strategy="explore"))
        return {"hypotheses": hyps}

    def run(self, analysis: dict, best_variant: str, round_id: int = 0,
            scaffold_fitness: float | None = None, scaffold_source: str = "实测值") -> list[dict]:
        prompt = self.build_prompt(analysis, best_variant, scaffold_fitness, scaffold_source)
        parsed = self.llm.chat("hypothesis", self.role, self.SYSTEM, prompt,
                               offline_handler=lambda: self._offline(analysis, best_variant),
                               round_id=round_id,
                               validate=valid_hypotheses)
        hyps = parsed.get("hypotheses", []) if isinstance(parsed, dict) else []
        # LLM 可能省掉 id / 用非法 strategy，这里补全，保证下游可用
        valid_strategy = {"exploit", "recombine", "epistasis", "explore"}
        for i, h in enumerate(hyps, 1):
            h.setdefault("id", f"H{i}")
            if h.get("strategy") not in valid_strategy:
                h["strategy"] = "exploit"
            h["target_aas"] = {str(k): [str(x).strip().upper() for x in (v or [])]
                               for k, v in (h.get("target_aas") or {}).items()}
        return hyps


# 模块 3
class MutationDesigner:
    """把假设翻译成具体的候选突变序列。"""

    role = "Mutation Designer（突变设计师）"

    # 分工：LLM 给出体现科学判断的「种子设计」，
    # 确定性引擎再按同样的策略把它们扩展成完整候选池。
    N_SEED_DESIGNS = 24

    SYSTEM = (
        "你是一名突变文库设计工程师。请把给定的突变假设转化为具体的四位点组合序列。"
        "序列写法为 4 个大写字母，依次对应位点 39/40/41/54，野生型为 VDGV。"
        "每条候选必须标注它来自哪条假设，design_note 用一句话说明设计意图（不超过 30 字）。"
        "必须只输出 JSON: {\"candidates\": [{\"variant\": str, \"hypothesis_id\": str, \"design_note\": str}]}。"
        "不要输出 JSON 以外的任何文字。"
    )

    def __init__(self, llm: LLMClient, cfg: AgentConfig, rulebook: DesignRuleBook | None = None):
        self.llm, self.cfg, self.rulebook = llm, cfg, rulebook
        self.rng = np.random.default_rng(cfg.seed)

    def build_prompt(self, hypotheses: list[dict], best_variant: str, n_measured: int) -> str:
        lines = [f"## 需要落地的假设（共 {len(hypotheses)} 条）"]
        for h in hypotheses:
            lines.append(f"- [{h['id']}|{h.get('strategy', 'exploit')}|置信度 {h.get('confidence', 0.5)}] "
                         f"{h['statement']}")
            lines.append(f"    目标位点 {h.get('positions')}，候选残基 {h.get('target_aas')}")
            lines.append(f"    依据: {h.get('rationale', '')}")
        lines += ["",
                  f"## 约束",
                  f"- 野生型组合: {WT_COMBO}（位点顺序 39/40/41/54）",
                  f"- 本轮设计骨架: {best_variant}",
                  f"- 已测量 {n_measured} 个变体，请勿重复提交已测过的组合",
                  f"- 只需给出 {self.N_SEED_DESIGNS} 条最有代表性的「种子设计」，"
                  f"每条假设至少覆盖 1 条；后续会由确定性枚举器按同样的策略把它们"
                  f"扩展到 {self.cfg.n_candidates} 条候选，因此这里**不要**罗列大量组合，"
                  f"把名额留给你认为最值得一做的设计",
                  ""]
        if self.cfg.use_knowledge:
            lines += ["## 设计规则（硬约束必须满足，软约束用于风险与评分）"]
            lines += [f"- {rule}" for rule in (self.rulebook or DesignRuleBook()).prompt_rules()]
            lines += ["- 批次内保持残基多样性，避免所有候选共享同一个 40 位残基。",
                      "- 已测低阶组合不得重复送检；从 WT 出发时可以提出带超限风险说明的高阶探索候选。", ""]
        lines.append("请输出候选序列的 JSON。")
        return "\n".join(lines)

    # 离线引擎
    def _offline(self, hypotheses: list[dict], analysis: dict, best_variant: str,
                 allowed: set[str], measured: set[str]) -> dict:
        stats = analysis["stats"]
        cfg = self.cfg
        cands: dict[str, dict] = {}

        hypotheses_by_id = {h["id"]: h for h in hypotheses}

        def add(v: str, hid: str | None, note: str, source: str = "hypothesis_expansion"):
            if len(v) != 4 or any(c not in AA_ALPHABET for c in v):
                return
            if v in measured or v not in allowed or v in cands:
                return
            linked = hid if hid in hypotheses_by_id and matches_hypothesis(v, hypotheses_by_id[hid]) else None
            cands[v] = {"variant": v, "hypothesis_id": linked,
                        "source_type": source, "design_note": note}

        # 每个位点可用的残基池（按假设汇总）
        pools: dict[int, list[str]] = defaultdict(list)
        for h in hypotheses:
            for k, v in (h.get("target_aas") or {}).items():
                pools[int(k)].extend([x for x in v if x in AA_ALPHABET])
        for pos in MUT_POSITIONS:
            i = MUT_POSITIONS.index(pos)
            extra = [r["aa"] for r in stats["best_residues_per_position"][pos][:5]]
            pools[pos] = list(dict.fromkeys(pools[pos] + extra + [WT_COMBO[i]]))

        # 策略 A: 假设驱动的定向组合（以当前最优变体为骨架，小步改动）
        for h in hypotheses:
            hid, positions = h["id"], [int(p) for p in (h.get("positions") or [])]
            targets = {int(k): [x for x in v if x in AA_ALPHABET]
                       for k, v in (h.get("target_aas") or {}).items()}
            if not targets:
                continue
            pos_list = [p for p in MUT_POSITIONS if p in targets] or positions
            for r in range(1, min(len(pos_list), 3) + 1):
                for subset in combinations(pos_list, r):
                    choices = [targets.get(p, [WT_COMBO[MUT_POSITIONS.index(p)]]) for p in subset]
                    for combo_aas in product(*choices):
                        base = list(best_variant)
                        for p, aa in zip(subset, combo_aas):
                            base[MUT_POSITIONS.index(p)] = aa
                        add("".join(base), hid,
                            f"在当前最优骨架 {best_variant} 上按假设 {hid} 改动位点 {list(subset)}")
                        base = list(WT_COMBO)
                        for p, aa in zip(subset, combo_aas):
                            base[MUT_POSITIONS.index(p)] = aa
                        add("".join(base), hid,
                            f"在野生型骨架上按假设 {hid} 引入位点 {list(subset)} 的替换")

        # 策略 B: 重组历史 Top 变体的突变
        top_variants = [r["variant"] for r in stats["top_variants"][:10]]
        for a, b in combinations(top_variants, 2):
            for mask in range(1, 15):
                v = "".join(a[i] if (mask >> i) & 1 else b[i] for i in range(4))
                add(v, None, f"重组 Top 变体 {a} 与 {b}（DNA shuffling 思路）", "recombine")

        # 策略 C: 上位效应骨架扩展（仅知识增强）
        if cfg.use_knowledge and stats["epistasis_top"]:
            for e in stats["epistasis_top"][:15]:
                scaffold = mutations_to_combo([e["mut_a"], e["mut_b"]])
                add(scaffold, None, f"上位效应骨架 {e['mut_a']}+{e['mut_b']}", "epistasis_extension")
                for pos in MUT_POSITIONS:
                    i = MUT_POSITIONS.index(pos)
                    if scaffold[i] != WT_COMBO[i]:
                        continue
                    for aa in pools[pos][:6]:
                        v = scaffold[:i] + aa + scaffold[i + 1:]
                        add(v, None, f"在上位效应骨架 {scaffold}（{e['mut_a']}+{e['mut_b']}）上"
                                     f"追加位点 {pos}->{aa}", "epistasis_extension")

        # 策略 D: 位点残基池的笛卡尔组合（受控枚举）
        grids = [pools[p][:5] for p in MUT_POSITIONS]
        for combo_aas in product(*grids):
            v = "".join(combo_aas)
            if hamming(v, best_variant) <= 2 or hamming(v) <= 3:
                add(v, None, f"位点残基池组合枚举 {v}", "residue_enumeration")
            if len(cands) > cfg.n_candidates * 2:
                break

        # 策略 E: 探索（在最优骨架附近随机扰动）
        n_explore = int(cfg.n_candidates * cfg.exploration_ratio)
        tries = 0
        while sum(c["source_type"] == "random_exploration" for c in cands.values()) < n_explore and tries < 20000:
            tries += 1
            base = list(best_variant if self.rng.random() < 0.6 else WT_COMBO)
            k = int(self.rng.integers(1, 3))
            for i in self.rng.choice(4, size=k, replace=False):
                base[int(i)] = AA_ALPHABET[int(self.rng.integers(0, 20))]
            add("".join(base), None, "探索型随机扰动，用于补充模型在未知区域的信息", "random_exploration")

        out = list(cands.values())
        return {"candidates": out[:cfg.n_candidates]}

    def run(self, hypotheses: list[dict], analysis: dict, best_variant: str,
            allowed: set[str], measured: set[str], round_id: int = 0) -> pd.DataFrame:
        prompt = self.build_prompt(hypotheses, best_variant, len(measured))
        parsed = self.llm.chat("mutation_design", self.role, self.SYSTEM, prompt,
                               offline_handler=lambda: self._offline(
                                   hypotheses, analysis, best_variant, allowed, measured),
                               round_id=round_id,
                               validate=lambda p: valid_candidates(p, {h["id"] for h in hypotheses}))
        cands = parsed.get("candidates", []) if isinstance(parsed, dict) else []

        # 真实 LLM 可能只给出少量候选或非法序列 -> 清洗并用离线引擎补足
        clean = []
        seen = set()
        hypothesis_map = {h["id"]: h for h in hypotheses}
        turn = self.llm.last("mutation_design")
        from_api = turn is not None and turn.backend.startswith("api")
        for c in cands:
            v = str(c.get("variant", "")).strip().upper()
            if (len(v) == 4 and all(ch in AA_ALPHABET for ch in v)
                    and v in allowed and v not in measured and v not in seen):
                seen.add(v)
                hid = c.get("hypothesis_id")
                linked = hid if hid in hypothesis_map and matches_hypothesis(v, hypothesis_map[hid]) else None
                clean.append({"variant": v, "hypothesis_id": linked,
                              "source_type": "llm_seed" if from_api else c.get("source_type", "hypothesis_expansion"),
                              "design_note": str(c.get("design_note", ""))})
        if from_api and len(clean) < self.cfg.n_candidates:
            backup = self._offline(hypotheses, analysis, best_variant, allowed, measured)["candidates"]
            for c in backup:
                if c["variant"] not in seen:
                    seen.add(c["variant"])
                    clean.append(c)
                if len(clean) >= self.cfg.n_candidates:
                    break
        return pd.DataFrame(clean[:self.cfg.n_candidates],
                            columns=["variant", "hypothesis_id", "source_type", "design_note"])


# 模块 4
class FitnessEvaluator:
    """调用适应度预测模型为候选打分（纯计算模块，不消耗 LLM）。"""

    role = "Fitness Evaluator（适应度评估器）"

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    def run(self, candidates: pd.DataFrame, model, best_known_log: float) -> pd.DataFrame:
        if len(candidates) == 0:
            return candidates.assign(pred_log_fitness=[], pred_fitness=[], uncertainty=[], ucb=[])
        variants = candidates.variant.tolist()
        mu, sigma = model.predict_with_uncertainty(variants)
        df = candidates.copy()
        df["pred_log_fitness"] = mu
        df["pred_fitness"] = from_log_fitness(mu)
        df["uncertainty"] = sigma
        # 置信上界：兼顾利用与探索（批量贝叶斯优化的常用采集函数）
        df["ucb"] = mu + 1.0 * sigma
        # 相对当前最优的预期提升
        df["expected_gain"] = df.pred_log_fitness - best_known_log
        df["n_mut"] = df.variant.map(hamming)
        df["mut_label"] = df.variant.map(mutation_label)
        return df.sort_values("pred_log_fitness", ascending=False).reset_index(drop=True)


# 模块 5
class ScientificCritic:
    """审查候选是否合理，保证批次多样性，并给出可读的推荐理由。"""

    role = "Scientific Critic（科学审稿人）"

    SYSTEM = (
        "你是一名严格的蛋白质工程审稿人。下游模型已经给出候选突变的预测适应度，"
        "你需要判断这些候选在结构与工程意义上是否合理，指出风险，"
        "并为最终推荐的候选写出清晰的推荐理由。"
        "必须只输出 JSON: {\"reviews\": [{\"variant\": str, \"verdict\": str, "
        "\"reason\": str, \"risk\": str}], \"batch_comment\": str}。"
        "verdict 取值限定为 \"accept\" / \"accept_with_caution\" / \"reject\"。"
        "reject 必须依据候选本身明确、可核验的不合理之处；不能仅因预测低于亲本、"
        "模型不确定度高或存在尚待验证的上位效应而否决探索候选，这些情况使用 accept_with_caution。"
        "只根据提供的数据和规则审查，未知的结构机制必须表述为假设。"
        "软规则告警本身不是硬性禁令，须区分单点效应与组合效应。"
        "提供的历史实测值优先于结构先验；已有单点或双突变实测证据不得称为未知或未测。"
        "补偿证据仅支持对应背景，不保证高阶组合有益。"
        "必须逐条审查全部候选，返回每条候选的 verdict、reason 和 risk。"
        "篇幅要求：reason 每条不超过 60 字，risk 每条不超过 30 字，batch_comment 不超过 120 字。"
        "不要输出 JSON 以外的任何文字。"
    )

    def __init__(self, llm: LLMClient, cfg: AgentConfig,
                 rulebook: DesignRuleBook | None = None,
                 kg: KnowledgeGraph | None = None):
        self.llm, self.cfg, self.rulebook, self.kg = llm, cfg, rulebook, kg

    # 规则审查 + 多样性挑选
    def screen(self, scored: pd.DataFrame, best_variant: str) -> pd.DataFrame:
        df = scored.copy()
        if self.cfg.use_knowledge and self.rulebook is not None:
            evals = [self.rulebook.evaluate(v, parent=best_variant) for v in df.variant]
            df["feasible"] = [e["feasible"] for e in evals]
            df["rule_penalty"] = [e["penalty"] for e in evals]
            df["rule_pass"] = [f"{e['n_pass']}/{e['n_total']}" for e in evals]
            df["rule_warnings"] = ["；".join(e["soft_warnings"]) for e in evals]
            df["rule_notes"] = ["；".join(e["passed_notes"]) for e in evals]
            df["hard_violations"] = ["；".join(e["hard_violations"]) for e in evals]
            df = df[df.feasible]
            df["adjusted_score"] = df.pred_log_fitness - df.rule_penalty
            df["adjusted_ucb"] = df.ucb - df.rule_penalty
        else:
            df["feasible"] = True
            df["rule_penalty"] = 0.0
            df["rule_pass"] = "-"
            df["rule_warnings"] = ""
            df["rule_notes"] = ""
            df["hard_violations"] = ""
            df["adjusted_score"] = df.pred_log_fitness
            df["adjusted_ucb"] = df.ucb
        return df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)

    def select_batch(self, screened: pd.DataFrame) -> pd.DataFrame:
        """批次选择：贪心地在"利用"和"探索"两个目标上取候选，
        每次选择都对与已入选候选过于相似（汉明距离 <= 1）的候选扣分，
        避免一个批次里全是几乎相同的构建体。"""
        cfg = self.cfg
        n_explore = max(1, int(round(cfg.batch_size * cfg.exploration_ratio)))
        n_exploit = cfg.batch_size - n_explore

        picked: list[dict] = []
        used: set[str] = set()
        pool = screened.reset_index(drop=True)

        def greedy_fill(score_col: str, limit: int, tag: str):
            while len(picked) < limit:
                best_row, best_val = None, -np.inf
                for row in pool.itertuples(index=False):
                    if row.variant in used:
                        continue
                    sim = sum(1 for p in picked if hamming(row.variant, p["variant"]) <= 1)
                    val = getattr(row, score_col) - cfg.diversity_penalty * sim
                    if val > best_val:
                        best_val, best_row = val, row
                if best_row is None:
                    break
                used.add(best_row.variant)
                r = best_row._asdict()
                r["selection_reason"] = tag
                r["diversity_adjusted_score"] = float(best_val)
                picked.append(r)

        greedy_fill("adjusted_score", n_exploit, "利用：模型预测适应度最高（已做批次多样性调整）")
        greedy_fill("adjusted_ucb", cfg.batch_size, "探索：置信上界高（预测不确定度大，信息增益高）")
        return pd.DataFrame(picked)

    # prompt
    def build_prompt(self, batch: pd.DataFrame, best_variant: str, best_fitness: float,
                     measured: pd.DataFrame | None = None, scaffold_source: str = "实测值") -> str:
        history = dict(zip(measured.variant, measured.fitness)) if measured is not None else {}
        lines = [f"## 本轮设计骨架: {best_variant}（{mutation_label(best_variant)}），"
                 f"适应度参照 {best_fitness:.3f}，来源: {scaffold_source}"]
        if self.cfg.use_knowledge:
            lines += ["", "## 设计规则（硬约束必须满足，软约束用于风险与评分）"]
            lines += (self.rulebook or DesignRuleBook()).prompt_rules()
        lines += ["", f"## 待审查的 {len(batch)} 条候选"]
        for _, r in batch.iterrows():
            lines.append(f"- {r.variant} [{r.mut_label}] 预测适应度 {r.pred_fitness:.3f} "
                         f"(log {r.pred_log_fitness:+.3f}, log 空间不确定度代理 {r.uncertainty:.3f})，"
                         f"来源 {getattr(r, 'source_type', 'model')}，关联假设 {r.hypothesis_id or '无直接关联'}；"
                         f"{r.selection_reason}")
            lines.append(f"    设计依据: {getattr(r, 'design_note', '')}")
            if self.cfg.use_knowledge:
                lines.append(f"    规则检查: 通过 {r.rule_pass}"
                             + (f"；警告: {r.rule_warnings}" if r.rule_warnings else "；无警告"))
                if r.rule_notes:
                    lines.append(f"    已通过规则的依据: {r.rule_notes}")
                muts = combo_to_mutations(r.variant)
                for m in muts:
                    single = mutations_to_combo([m])
                    if single in history:
                        lines.append(f"    历史单点实测: {m} ({single}) fitness={history[single]:.6f}（WT 背景）")
                    elif measured is not None:
                        lines.append(f"    历史单点记录: {m} 尚未单独测量")
                for a, b in combinations(muts, 2):
                    pair = mutations_to_combo([a, b])
                    if pair in history:
                        lines.append(f"    历史双突变实测: {a} + {b} ({pair}) fitness={history[pair]:.6f}；"
                                     "其余位点为 WT，高阶背景仍待验证")
                if self.kg is not None:
                    for m in muts:
                        lines.append(f"    知识图谱: {self.kg.explain_mutation(m)}")
        lines += ["", "请逐条给出审查结论和推荐理由，并对整批候选给一句总体评价。"]
        return "\n".join(lines)

    # 离线引擎
    def _offline(self, batch: pd.DataFrame, best_variant: str, best_fitness: float,
                 scaffold_source: str = "实测值") -> dict:
        reviews = []
        for _, r in batch.iterrows():
            muts = combo_to_mutations(r.variant)
            gain = r.pred_fitness - best_fitness
            bits = []
            if self.cfg.use_knowledge and self.kg is not None:
                for m in muts:
                    bits.append(self.kg.explain_mutation(m))
            else:
                for m in muts:
                    bits.append(f"{m} 的组合效应需要通过本轮虚拟实验检验。")

            reason = (f"预测适应度 {r.pred_fitness:.3f}"
                      f"（相对骨架参照 {best_fitness:.3f} 的预测变化 {gain:+.3f}，参照来源: {scaffold_source}）。"
                      f"该候选携带 {len(muts)} 个突变：{'、'.join(muts) if muts else '无（野生型）'}。"
                      + (" " + " ".join(bits) if bits else "")
                      + f" 入选原因：{r.selection_reason}。")
            if self.cfg.use_knowledge and r.rule_notes:
                reason += f" 规则库确认：{r.rule_notes}"

            risk_bits = []
            if self.cfg.use_knowledge and r.rule_warnings:
                risk_bits.append(r.rule_warnings)
            if r.uncertainty > float(batch.uncertainty.median()):
                risk_bits.append(f"模型不确定度偏高(±{r.uncertainty:.3f})，预测可能不可靠")
            if r.n_mut >= 3:
                risk_bits.append(f"共 {r.n_mut} 个突变，高阶组合的上位效应难以外推")
            risk = "；".join(risk_bits) if risk_bits else "未发现明显风险"

            if self.cfg.use_knowledge and r.rule_penalty > 0.2:
                verdict = "accept_with_caution"
            elif r.uncertainty > float(batch.uncertainty.quantile(0.8)):
                verdict = "accept_with_caution"
            else:
                verdict = "accept"
            reviews.append({"variant": r.variant, "verdict": verdict,
                            "reason": reason, "risk": risk})

        n_caution = sum(1 for x in reviews if x["verdict"] != "accept")
        n_unique_p40 = batch.variant.str[1].nunique()
        comment = (f"本批 {len(batch)} 条候选中 {n_caution} 条需谨慎对待；"
                   f"40 位残基共 {n_unique_p40} 种，批次多样性"
                   f"{'充足' if n_unique_p40 >= 3 else '偏低，存在冗余风险'}；"
                   f"预测适应度区间 {batch.pred_fitness.min():.2f} ~ {batch.pred_fitness.max():.2f}。")
        if self.cfg.use_knowledge:
            comment += "所有候选均已通过突变规则库的硬性约束检查。"
        else:
            comment += "本批次未使用结构/理化知识约束，完全依赖数据驱动的模型排序。"
        return {"reviews": reviews, "batch_comment": comment}

    def run(self, scored: pd.DataFrame, best_variant: str, best_fitness: float,
            round_id: int = 0, measured: pd.DataFrame | None = None,
            scaffold_source: str = "实测值") -> tuple[pd.DataFrame, dict]:
        screened = self.screen(scored, best_variant)
        reviews, comments, rejected = {}, [], set()
        for _ in range(self.cfg.max_review_batches):
            eligible = screened[~screened.variant.isin(rejected)]
            batch = self.select_batch(eligible)
            if len(batch) < self.cfg.batch_size:
                raise RuntimeError("通过筛选的候选不足实验预算，本轮未送检")
            pending = batch[~batch.variant.isin(reviews)]
            if len(pending):
                prompt = self.build_prompt(pending, best_variant, best_fitness, measured, scaffold_source)
                parsed = self.llm.chat(
                    "critic", self.role, self.SYSTEM, prompt,
                    offline_handler=lambda: self._offline(pending, best_variant, best_fitness, scaffold_source),
                    round_id=round_id,
                    validate=lambda p: valid_reviews(p, set(pending.variant)))
                reviews.update({r["variant"]: r for r in parsed["reviews"]})
                comments.append(parsed["batch_comment"])
                rejected.update(r["variant"] for r in parsed["reviews"] if r["verdict"] == "reject")
            if not set(batch.variant) & rejected:
                batch["verdict"] = batch.variant.map(lambda v: reviews[v]["verdict"])
                batch["recommendation_reason"] = batch.variant.map(lambda v: reviews[v]["reason"])
                batch["risk"] = batch.variant.map(lambda v: reviews[v]["risk"])
                return batch, {"batch_comment": "；".join(comments),
                               "n_screened": int(len(screened)), "n_rejected": len(rejected)}
        raise RuntimeError("审查后的候选不足实验预算，本轮未送检")


# 编排器
@dataclass
class RoundProposal:
    round_id: int
    strategy: str
    analysis: dict
    hypotheses: list[dict]
    n_candidates: int
    batch: pd.DataFrame
    top_k: pd.DataFrame
    critic_comment: str
    best_variant_before: str
    best_fitness_before: float
    backend: str = ""
    n_rejected: int = 0
    run_id: str = ""
    phase: str = ""


class DirectedEvolutionAgent:
    """把五个模块串成 "分析 -> 假设 -> 设计 -> 评估 -> 审查" 的闭环。"""

    def __init__(self, llm: LLMClient, cfg: AgentConfig,
                 rulebook: DesignRuleBook | None = None,
                 kg: KnowledgeGraph | None = None):
        self.llm, self.cfg = llm, cfg
        if llm.backend == "offline" and cfg.name.startswith("LLM-Agent"):
            cfg.name = cfg.name.replace("LLM-Agent", "Rule-Agent", 1)
        self.rulebook = rulebook
        self.kg = kg
        self.analyst = DataAnalyst(llm, cfg)
        self.hypo = HypothesisGenerator(llm, cfg, kg, rulebook)
        self.designer = MutationDesigner(llm, cfg, rulebook)
        self.evaluator = FitnessEvaluator(cfg)
        self.critic = ScientificCritic(llm, cfg, rulebook, kg)

    def propose(self, measured: pd.DataFrame, model, allowed: set[str],
                round_id: int = 1, verbose: bool = True,
                start_variant: str | None = None) -> RoundProposal:
        """提出下一轮候选突变。

        start_variant 给定时，用它作为本轮设计的骨架（"从这条序列出发帮我改"），
        否则默认以当前已测数据中的最优变体为骨架。
        """
        best_idx = measured.fitness.idxmax()
        best_variant = str(measured.loc[best_idx, "variant"])
        best_fitness = float(measured.loc[best_idx, "fitness"])
        best_log = float(measured.loc[best_idx, "log_fitness"])
        scaffold_source = "实测值"
        turn_start = len(self.llm.transcript)
        self.llm.strategy, self.llm.seed = self.cfg.name, self.cfg.seed

        if start_variant:
            best_variant = start_variant.strip().upper()
            hit = measured[measured.variant == best_variant]
            if len(hit):                       # 已测过 -> 用实测值
                best_fitness = float(hit.fitness.iloc[0])
                best_log = float(hit.log_fitness.iloc[0])
            else:                              # 未测过 -> 只能用模型预测值作为参照
                scaffold_source = "模型预测值，尚未实验测量"
                best_log = float(model.predict([best_variant])[0])
                best_fitness = float(from_log_fitness(best_log))

        measured_set = set(measured.variant)

        if verbose:
            print(f"\n{'─' * 96}")
            print(f"【{self.cfg.name}】第 {round_id} 轮  |  已测 {len(measured)} 个变体  |  "
                  f"设计骨架 {best_variant}({mutation_label(best_variant)})，参照 {best_fitness:.3f}（{scaffold_source}）")
            print(f"{'─' * 96}")

        # 1 数据分析
        analysis = self.analyst.run(measured, round_id=round_id)

        # 规则库从本轮数据中刷新经验规则；上位效应证据会用于覆盖过于保守的先验规则
        if self.cfg.use_knowledge and self.rulebook is not None:
            self.rulebook.learn_from_history(
                measured, epistasis_records=analysis["stats"]["epistasis_top"])
            if verbose and self.rulebook.rescue_partners:
                print(f"  [规则库] 已验证有益单点 {len(self.rulebook.known_good_mutations)} 个，"
                      f"低功能单点 {len(self.rulebook.dead_mutations)} 个，"
                      f"具备上位效应补偿证据的突变 {len(self.rulebook.rescue_partners)} 个")

        if verbose:
            print(f"  [1/5 Data Analyst] {analysis['llm']['summary']}")
            for o in analysis["llm"]["observations"]:
                print(f"        · {o}")

        # 2 假设生成
        hyps = self.hypo.run(analysis, best_variant, round_id=round_id,
                             scaffold_fitness=best_fitness, scaffold_source=scaffold_source)
        if verbose:
            print(f"  [2/5 Hypothesis Generator] 生成 {len(hyps)} 条假设：")
            for h in hyps:
                print(f"        [{h['id']}|{h.get('strategy')}|conf={h.get('confidence')}] {h['statement']}")
                print(f"              理由: {h.get('rationale', '')}")

        # 3 突变设计
        cands = self.designer.run(hyps, analysis, best_variant, allowed, measured_set, round_id=round_id)
        if verbose:
            src = cands.source_type.value_counts().to_dict() if len(cands) else {}
            print(f"  [3/5 Mutation Designer] 生成候选 {len(cands)} 条，来源分布: {src}")

        # 4 模型打分
        scored = self.evaluator.run(cands, model, best_log)
        if verbose and len(scored):
            print(f"  [4/5 Fitness Evaluator] 预测适应度范围 "
                  f"{scored.pred_fitness.min():.3f} ~ {scored.pred_fitness.max():.3f}，"
                  f"预测超过骨架参照的候选 {int((scored.pred_fitness > best_fitness).sum())} 条")

        # 5 审查 + 选批
        batch, meta = self.critic.run(scored, best_variant, best_fitness, round_id=round_id,
                                    measured=measured, scaffold_source=scaffold_source)
        # 批次内混有探索位，顺序并非按预测值排列，因此显式标注 Top-k
        topk = batch.nlargest(self.cfg.top_k, "pred_log_fitness").reset_index(drop=True)
        batch["is_topk"] = batch.variant.isin(set(topk.variant))
        batch["topk_rank"] = batch.variant.map(
            {v: i + 1 for i, v in enumerate(topk.variant)}).astype("Int64")
        if verbose:
            print(f"  [5/5 Scientific Critic] 规则筛选后 {meta['n_screened']} 条，"
                  f"最终选出 {len(batch)} 条送入虚拟实验")
            print(f"        批次评价: {meta['batch_comment']}")

        sources = {"api" if t.backend.startswith("api") else "offline"
                   for t in self.llm.transcript[turn_start:]}
        backend = next(iter(sources)) if len(sources) == 1 else "mixed"
        return RoundProposal(round_id=round_id, strategy=self.cfg.name, analysis=analysis,
                             hypotheses=hyps, n_candidates=int(len(cands)),
                             batch=batch, top_k=topk, critic_comment=meta["batch_comment"],
                             best_variant_before=best_variant, best_fitness_before=best_fitness,
                             backend=backend, n_rejected=meta["n_rejected"],
                             run_id=self.llm.run_id, phase=self.llm.phase)
