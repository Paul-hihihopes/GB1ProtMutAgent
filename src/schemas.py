"""Agent 模块之间的结构化数据契约。"""
from __future__ import annotations

import math

from .config import AA_ALPHABET, MUT_POSITIONS


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positions(value, *, nonempty=False) -> bool:
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(type(p) is int and p in MUT_POSITIONS for p in value)
            and len(value) == len(set(value)))


def valid_analysis(value) -> bool:
    if not isinstance(value, dict):
        return False
    residues = value.get("beneficial_residues")
    observations = value.get("observations")
    return (_text(value.get("summary"))
            and _positions(value.get("key_positions"), nonempty=True)
            and _positions(value.get("risky_positions"))
            and isinstance(observations, list) and all(_text(x) for x in observations)
            and isinstance(residues, list)
            and all(isinstance(r, dict)
                    and type(r.get("position")) is int and r["position"] in MUT_POSITIONS
                    and isinstance(r.get("aa"), str) and len(r["aa"]) == 1
                    and r["aa"] in AA_ALPHABET and _text(r.get("evidence")) for r in residues))


def valid_hypotheses(value) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("hypotheses"), list):
        return False
    hypotheses = value["hypotheses"]
    if not 2 <= len(hypotheses) <= 6:
        return False
    ids = set()
    for h in hypotheses:
        if not isinstance(h, dict) or not _text(h.get("id")) or h["id"] in ids:
            return False
        ids.add(h["id"])
        if not (_text(h.get("statement")) and _text(h.get("rationale"))
                and _positions(h.get("positions"), nonempty=True)
                and h.get("strategy") in {"exploit", "recombine", "epistasis", "explore"}):
            return False
        confidence = h.get("confidence")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            return False
        targets = h.get("target_aas")
        if not isinstance(targets, dict) or not targets:
            return False
        for position, aas in targets.items():
            if str(position) not in {str(p) for p in h["positions"]}:
                return False
            if not (isinstance(aas, list) and aas and all(
                    isinstance(a, str) and len(a) == 1 and a in AA_ALPHABET for a in aas)):
                return False
    return True


def matches_hypothesis(variant: str, hypothesis: dict) -> bool:
    """候选满足该假设指定的全部目标残基约束。"""
    targets = hypothesis.get("target_aas", {})
    return bool(targets) and all(
        variant[MUT_POSITIONS.index(int(p))] in aas for p, aas in targets.items())


def valid_candidates(value, hypothesis_ids: set[str]) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("candidates"), list):
        return False
    candidates = value["candidates"]
    return len(candidates) >= 3 and all(
        isinstance(c, dict) and isinstance(c.get("variant"), str)
        and len(c["variant"]) == 4 and all(a in AA_ALPHABET for a in c["variant"])
        and (c.get("hypothesis_id") is None or c.get("hypothesis_id") in hypothesis_ids)
        and _text(c.get("design_note")) for c in candidates)


def valid_reviews(value, variants: set[str]) -> bool:
    if not isinstance(value, dict) or not _text(value.get("batch_comment")):
        return False
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != len(variants):
        return False
    seen = set()
    for r in reviews:
        if not isinstance(r, dict):
            return False
        v = r.get("variant")
        if not isinstance(v, str) or v not in variants or v in seen:
            return False
        seen.add(v)
        if (r.get("verdict") not in {"accept", "accept_with_caution", "reject"}
                or not _text(r.get("reason")) or not _text(r.get("risk"))):
            return False
    return seen == variants
