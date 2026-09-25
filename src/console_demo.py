"""Console workflow for the GB1 directed-evolution demo.

This module is an application adapter.  Model, Oracle, knowledge, Agent and
LLM behavior remain implemented in their existing ``src`` modules.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass

import pandas as pd

from .config import AA_ALPHABET, BATCH_SIZE, MUT_POSITIONS, RESULT_DIR, TOP_K, WT_COMBO
from .data import (VirtualLabOracle, combo_to_mutations, hamming, load_dataset,
                   mutation_label, read_wt_sequence)
from .knowledge import DesignRuleBook, KnowledgeGraph
from .llm import LLMClient
from .models import FitnessModel


def parse_start_sequence(text: str) -> dict:
    """Parse a GB1 starting sequence into the four-site variant notation.

    Accepted forms include blank/WT,
    four-site combinations, mutation notation, 56-aa GB1 sequences, and
    265-aa FLIP fusion sequences.
    """
    if not isinstance(text, str):
        return {"ok": False, "error": "序列必须是文本"}
    raw = text.strip()
    if not raw or raw.upper() in ("WT", "野生型", "WILDTYPE", "WILD-TYPE"):
        return {"ok": True, "variant": WT_COMBO, "how": "未指定起始序列，默认从野生型 VDGV 出发"}

    cleaned = "".join(ch for ch in raw.upper() if not ch.isspace())

    if len(cleaned) >= max(MUT_POSITIONS):
        bad = [c for c in cleaned if c not in AA_ALPHABET]
        if bad:
            return {"ok": False, "error": f"序列中含有非标准氨基酸字符：{sorted(set(bad))[:5]}"}
        references = [read_wt_sequence(), read_wt_sequence(full_construct=True)]
        reference = next((seq for seq in references if len(seq) == len(cleaned)), None)
        if reference is None:
            return {"ok": False, "error": "仅支持 56 aa GB1 结构域或 265 aa FLIP 融合序列表示"}
        changed = [i + 1 for i, (a, b) in enumerate(zip(cleaned, reference))
                   if a != b and i + 1 not in MUT_POSITIONS]
        if changed:
            return {"ok": False, "error": f"模型仅支持 GB1 四位点文库；其他位点与参考序列不符：{changed[:8]}"}
        combo = "".join(cleaned[p - 1] for p in MUT_POSITIONS)
        return {"ok": True, "variant": combo,
                "how": f"已校验长度 {len(cleaned)} 的 GB1 背景，抽取位点 "
                        f"{'/'.join(str(p) for p in MUT_POSITIONS)} 得到组合 {combo}"}

    if len(cleaned) == len(MUT_POSITIONS) and all(c in AA_ALPHABET for c in cleaned):
        return {"ok": True, "variant": cleaned,
                "how": f"识别为四位点组合，依次对应位点 {'/'.join(str(p) for p in MUT_POSITIONS)}"}

    toks = [t for t in re.split(r"[+,;/、]", raw.upper()) if t.strip()]
    muts, chars = [], list(WT_COMBO)
    seen_positions = set()
    if not toks:
        return {"ok": False, "error": "请输入序列或完整突变记号"}
    for t in toks:
        m = re.fullmatch(r"\s*([A-Z])\s*(\d{1,3})\s*([A-Z])\s*", t)
        if not m:
            return {"ok": False,
                    "error": f"无法解析 “{t.strip()}”。请使用 VDGV 这样的四位点组合、"
                             f"D40W + V54F 这样的突变记号，或直接粘贴完整蛋白序列。"}
        wt_aa, pos, new_aa = m.group(1), int(m.group(2)), m.group(3)
        if pos not in MUT_POSITIONS:
            return {"ok": False,
                    "error": f"位点 {pos} 不可突变，本文库只覆盖位点 "
                             f"{'/'.join(str(p) for p in MUT_POSITIONS)}。"}
        idx = MUT_POSITIONS.index(pos)
        if pos in seen_positions:
            return {"ok": False, "error": f"位点 {pos} 重复指定，请每个位点只填写一个替换"}
        seen_positions.add(pos)
        if wt_aa != WT_COMBO[idx]:
            return {"ok": False, "error": f"位点 {pos} 的野生型残基是 {WT_COMBO[idx]} 而不是 {wt_aa}。"}
        if new_aa not in AA_ALPHABET:
            return {"ok": False, "error": f"{new_aa} 不是标准氨基酸。"}
        chars[idx] = new_aa
        muts.append(f"{wt_aa}{pos}{new_aa}")
    combo = "".join(chars)
    return {"ok": True, "variant": combo,
            "how": f"识别为突变记号 {' + '.join(muts)}，对应组合 {combo}"}


@dataclass
class Lab:
    """Domain objects needed by one or more console rounds."""

    df: pd.DataFrame
    known: pd.DataFrame
    allowed: set[str]
    oracle: VirtualLabOracle
    model: FitnessModel
    kg: KnowledgeGraph
    rulebook: DesignRuleBook
    llm: LLMClient


def build_lab(*, backend: str | None = None, model_kind: str | None = None) -> Lab:
    """Build the experiment environment for interactive rounds."""
    df = load_dataset(verbose=False)
    from .data import make_splits

    splits = make_splits(df, verbose=False)
    known = splits["known"]
    best_model = model_kind or "xgboost"
    overview_path = RESULT_DIR / "overview.json"
    if overview_path.exists():
        import json
        try:
            if model_kind is None:
                best_model = json.loads(overview_path.read_text(encoding="utf-8")).get("best_model", best_model)
        except (OSError, ValueError):
            pass
    model = FitnessModel(kind=best_model).fit(known)
    return Lab(
        df=df,
        known=known,
        allowed=set(df.variant),
        oracle=VirtualLabOracle.from_dataframe(df, already_measured=known.variant),
        model=model,
        kg=KnowledgeGraph.build(known, verbose=False),
        rulebook=DesignRuleBook(),
        llm=LLMClient(backend=backend or os.environ.get("LLM_BACKEND", "auto")),
    )


def inspect_variant(lab: Lab, sequence: str) -> dict:
    """Return normalized input and fitness information."""
    result = parse_start_sequence(sequence)
    if not result["ok"]:
        return result
    variant = result["variant"]
    known_rows = lab.known[lab.known.variant == variant]
    measured = not known_rows.empty
    result.update({
        "label": mutation_label(variant),
        "n_mut": sum(a != b for a, b in zip(variant, WT_COMBO)),
        "in_library": variant in lab.allowed,
        "measured": measured,
        "measured_fitness": float(known_rows.fitness.iloc[0]) if measured else None,
        "predicted_fitness": float(lab.model.predict_fitness([variant])[0]),
    })
    return result


def run_round(lab: Lab, sequence: str, *, use_knowledge: bool = True,
              batch_size: int = BATCH_SIZE, round_id: int = 1) -> dict:
    """Run one interactive Agent round and return a stable structured result."""
    if type(use_knowledge) is not bool:
        raise ValueError("知识增强开关必须为布尔值")
    if type(batch_size) is not int or not 4 <= batch_size <= 24:
        raise ValueError("推荐数量必须为 4 至 24 之间的整数")
    parsed = parse_start_sequence(sequence)
    if not parsed["ok"]:
        raise ValueError(parsed["error"])

    from .agent import AgentConfig, DirectedEvolutionAgent

    reference_client = lab.llm
    client = LLMClient(backend=reference_client.backend, model=reference_client.model,
                       base_url=reference_client.base_url, api_key=reference_client.api_key,
                       phase="interactive")
    cfg = AgentConfig(name="LLM-Agent+KB" if use_knowledge else "LLM-Agent",
                      use_knowledge=use_knowledge, batch_size=batch_size,
                      top_k=min(TOP_K, batch_size))
    agent = DirectedEvolutionAgent(client, cfg,
                                   DesignRuleBook() if use_knowledge else None,
                                   lab.kg if use_knowledge else None)
    proposal = agent.propose(lab.known, lab.model, lab.allowed, round_id=round_id,
                             verbose=False, start_variant=parsed["variant"])

    batch = proposal.batch.copy()
    batch["true_fitness"] = batch.variant.map(lab.oracle.true_fitness)
    cols = ["variant", "mut_label", "n_mut", "pred_fitness", "uncertainty", "true_fitness",
            "hypothesis_id", "source_type", "is_topk", "topk_rank", "selection_reason",
            "verdict", "recommendation_reason", "risk", "design_note", "rule_pass",
            "rule_warnings"]
    cols = [column for column in cols if column in batch.columns]
    batch = batch[cols].astype(object).where(pd.notnull(batch[cols]), None)

    start_variant = proposal.best_variant_before
    start_measured = start_variant in set(lab.known.variant)
    n_above = int((batch.true_fitness > proposal.best_fitness_before).sum())
    return {
        "ok": True,
        "strategy": cfg.name,
        "backend": proposal.backend,
        "run_id": client.run_id,
        "phase": client.phase,
        "usage": client.usage_report(),
        "start": {"variant": start_variant, "label": mutation_label(start_variant),
                  "how": parsed["how"], "measured": start_measured,
                  "source": "实测值" if start_measured else "模型预测值（该组合尚未测量）",
                  "true_fitness": proposal.best_fitness_before if start_measured else None},
        "best_before": {"variant": start_variant, "label": mutation_label(start_variant),
                        "fitness": proposal.best_fitness_before},
        "n_candidates": proposal.n_candidates,
        "analysis": proposal.analysis["llm"],
        "hypotheses": proposal.hypotheses,
        "critic_comment": proposal.critic_comment,
        "batch": batch.to_dict("records"),
        "turns": [turn.to_record() for turn in client.transcript],
        "stats": {
            "max_true": float(batch.true_fitness.max()) if batch.true_fitness.notna().any() else None,
            "mean_true": float(batch.true_fitness.mean()) if batch.true_fitness.notna().any() else None,
            "n_above_start": n_above if start_measured else None,
            "n_above_reference": n_above,
            "comparison_basis": "measured" if start_measured else "predicted",
            "n_above_wt": int((batch.true_fitness > 1.0).sum()),
        },
    }


def advance_lab(lab: Lab, result: dict) -> None:
    """Commit the last batch as measured data and refit for the next round."""
    variants = [row["variant"] for row in result["batch"]]
    assayed = lab.oracle.assay(variants)
    lab.known = pd.concat([lab.known, assayed], ignore_index=True)
    lab.known["mutations"] = lab.known.variant.map(combo_to_mutations)
    lab.known["n_mut"] = lab.known.variant.map(hamming)
    lab.model = FitnessModel(kind=lab.model.kind, seed=lab.model.seed).fit(lab.known)
    lab.kg = KnowledgeGraph.build(lab.known, verbose=False)


def _print_variant_info(info: dict) -> None:
    if not info.get("ok"):
        print(f"输入错误：{info['error']}")
        return
    print(f"variant: {info['variant']}  |  {info['label']}  |  mutations={info['n_mut']}")
    print(f"解析: {info['how']}")
    print(f"文库: {'yes' if info['in_library'] else 'no'}  |  measured: {'yes' if info['measured'] else 'no'}")
    if info["measured"]:
        print(f"真实 fitness: {info['measured_fitness']:.6f}")
    else:
        print(f"预测 fitness: {info['predicted_fitness']:.6f}")


def print_round_result(result: dict) -> None:
    """Render a structured round result without changing its fields."""
    start, stats = result["start"], result["stats"]
    print(f"\n[{result['strategy']}] backend={result['backend']} phase={result['phase']}")
    print(f"起点: {start['variant']} ({start['label']})，{start['source']}，fitness={result['best_before']['fitness']:.6f}")
    print("\n[1/5 Data Analyst]")
    print(result["analysis"].get("summary", ""))
    for item in result["analysis"].get("observations", []):
        print(f"  - {item}")
    print(f"\n[2/5 Hypothesis Generator] {len(result['hypotheses'])} hypotheses")
    for hypothesis in result["hypotheses"]:
        print(f"  [{hypothesis.get('id')}] {hypothesis.get('statement', '')}")
    print(f"\n[3/5 Mutation Designer] generated candidates={result['n_candidates']}")
    for turn in result["turns"]:
        if turn["step"] == "mutation_design":
            try:
                design = json.loads(turn["response"])
            except (TypeError, ValueError):
                design = turn["response"]
            candidates = design.get("candidates", []) if isinstance(design, dict) else []
            print(f"  source={turn['backend']}  proposed={len(candidates)}")
            for candidate in candidates[:8]:
                print(f"    {candidate.get('variant')}: {candidate.get('design_note', '')}")
            if len(candidates) > 8:
                print(f"    ... 另有 {len(candidates) - 8} 条设计候选，完整输出保留在 turns 中")
    print("\n[4/5 Fitness Evaluator + Oracle]")
    for candidate in result["batch"]:
        print(f"  {candidate['variant']} ({candidate.get('mut_label', '')})  "
              f"pred={candidate.get('pred_fitness'):.6f}  true={candidate.get('true_fitness'):.6f}  "
              f"uncertainty={candidate.get('uncertainty'):.3f}")
        print(f"    source={candidate.get('source_type')}  selection={candidate.get('selection_reason')}")
        print(f"    design={candidate.get('design_note')}  verdict={candidate.get('verdict')}")
        print(f"    recommendation={candidate.get('recommendation_reason')}")
        print(f"    risk={candidate.get('risk')}")
    print("\n[5/5 Scientific Critic]")
    print(result["critic_comment"])
    print(f"\n统计: max_true={stats['max_true']} mean_true={stats['mean_true']} "
          f"above_reference={stats['n_above_reference']} above_wt={stats['n_above_wt']} "
          f"basis={stats['comparison_basis']}")
    usage = result["usage"]
    print(f"LLM usage: backend={result['backend']} calls={usage['total_calls']} "
          f"api={usage['llm_calls']} offline={usage['offline_fallbacks']} "
          f"tokens={usage['prompt_tokens'] + usage['completion_tokens']}")


def _predict_flow(lab: Lab) -> None:
    text = input("输入起始 variant/序列（空值=WT，q 返回）：").strip()
    if text.lower() == "q":
        return
    _print_variant_info(inspect_variant(lab, text))


def interactive_demo(lab: Lab | None = None) -> None:
    """Run the menu-driven console demo."""
    lab = lab or build_lab()
    lab.llm.banner()
    current_start = ""
    round_id = 1
    while True:
        print("\n=== GB1 Directed Evolution Console Demo ===")
        print("[1] Start interactive evolution")
        print("[2] Predict fitness")
        print("[3] Model information")
        print("[0] Exit")
        choice = input("Select: ").strip()
        if choice == "0":
            print("退出。")
            return
        if choice == "2":
            _predict_flow(lab)
            continue
        if choice == "3":
            print(f"model={lab.model.kind}  known={len(lab.known)}  library={len(lab.allowed)}")
            print(f"positions={MUT_POSITIONS}  WT={WT_COMBO}  batch_default={BATCH_SIZE}")
            continue
        if choice != "1":
            print("请输入 0、1、2 或 3。")
            continue

        text = input("输入起始 variant/序列（空值=WT）：").strip()
        info = inspect_variant(lab, text)
        _print_variant_info(info)
        if not info.get("ok") or not info.get("in_library"):
            continue
        current_start = text
        while True:
            kb_text = input("启用 knowledge？[Y/n]: ").strip().lower()
            use_knowledge = kb_text not in {"n", "no", "否"}
            batch_text = input(f"batch size [4-24，默认 {BATCH_SIZE}]: ").strip()
            try:
                batch_size = int(batch_text) if batch_text else BATCH_SIZE
                result = run_round(lab, current_start, use_knowledge=use_knowledge,
                                   batch_size=batch_size, round_id=round_id)
            except (ValueError, RuntimeError) as exc:
                print(f"本轮未完成：{exc}")
                break
            print_round_result(result)
            round_id += 1
            next_action = input("继续下一轮请输入 variant（空值使用本批真实最优，q 返回主菜单）：").strip()
            if next_action.lower() == "q":
                break
            if not next_action:
                next_action = max(result["batch"], key=lambda row: row["true_fitness"] or float("-inf"))["variant"]
            if next_action:
                next_info = parse_start_sequence(next_action)
                if next_info.get("ok") and next_info["variant"] not in lab.allowed:
                    next_info = {"ok": False, "error": "起始序列不在 GB1 允许文库中"}
                if next_info.get("ok"):
                    advance_lab(lab, result)
                    next_info = inspect_variant(lab, next_action)
                _print_variant_info(next_info)
                if not next_info.get("ok"):
                    break
                current_start = next_action
                print(f"已送检 {len(result['batch'])} 条候选；累计已测 {len(lab.known)} 条。")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GB1 蛋白质定向进化 Console Demo")
    parser.add_argument("--backend", choices=("auto", "api", "offline"), default=None,
                        help="LLM 后端；默认读取 LLM_BACKEND 或自动选择")
    args = parser.parse_args(argv)
    interactive_demo(build_lab(backend=args.backend))


if __name__ == "__main__":
    main()
