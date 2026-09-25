"""定向进化知识库 + 突变规则库 + 轻量知识图谱。

三层内容：
1. 氨基酸理化性质表（疏水性、体积、电荷、极性、芳香性、二级结构倾向、柔性）
2. 替换矩阵 BLOSUM62 -> 保守替换 / 激进替换判定
3. 蛋白质工程经验规则（突变数量、结构位置、半胱氨酸、脯氨酸、电荷平衡等）
   以及由 "氨基酸-性质-位点-突变-变体" 构成的知识图谱

数据来源见 README：Kyte & Doolittle (1982)、Zamyatnin (1972)、
Chou & Fasman (1978)、Vihinen et al. (1994)、Henikoff & Henikoff (1992)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import AA_ALPHABET, MUT_POSITIONS, RESULT_DIR, WT_COMBO, rel
from .data import combo_to_mutations, hamming, validate_combo

# 1. 理化性质
# hydropathy: Kyte-Doolittle 疏水性指数 (越大越疏水, rel)
# volume     : 侧链体积 Å^3 (Zamyatnin 1972)
# charge     : pH 7.0 形式电荷 (His 按部分质子化记 0.5)
# helix/sheet: Chou-Fasman 二级结构倾向 (>1 表示倾向性强)
# flexibility: 平均主链柔性 (Vihinen 1994)
AA_PROPERTIES: dict[str, dict] = {
    "A": dict(name="Alanine",       tla="Ala", hydropathy=1.8,  volume=88.6,  charge=0.0,
              mw=89.09,  polarity="nonpolar",  aromatic=False, helix=1.42, sheet=0.83, flexibility=0.36),
    "C": dict(name="Cysteine",      tla="Cys", hydropathy=2.5,  volume=108.5, charge=0.0,
              mw=121.16, polarity="polar",     aromatic=False, helix=0.70, sheet=1.19, flexibility=0.35),
    "D": dict(name="Aspartate",     tla="Asp", hydropathy=-3.5, volume=111.1, charge=-1.0,
              mw=133.10, polarity="negative",  aromatic=False, helix=1.01, sheet=0.54, flexibility=0.51),
    "E": dict(name="Glutamate",     tla="Glu", hydropathy=-3.5, volume=138.4, charge=-1.0,
              mw=147.13, polarity="negative",  aromatic=False, helix=1.51, sheet=0.37, flexibility=0.50),
    "F": dict(name="Phenylalanine", tla="Phe", hydropathy=2.8,  volume=189.9, charge=0.0,
              mw=165.19, polarity="nonpolar",  aromatic=True,  helix=1.13, sheet=1.38, flexibility=0.31),
    "G": dict(name="Glycine",       tla="Gly", hydropathy=-0.4, volume=60.1,  charge=0.0,
              mw=75.07,  polarity="nonpolar",  aromatic=False, helix=0.57, sheet=0.75, flexibility=0.54),
    "H": dict(name="Histidine",     tla="His", hydropathy=-3.2, volume=153.2, charge=0.5,
              mw=155.16, polarity="positive",  aromatic=True,  helix=1.00, sheet=0.87, flexibility=0.32),
    "I": dict(name="Isoleucine",    tla="Ile", hydropathy=4.5,  volume=166.7, charge=0.0,
              mw=131.17, polarity="nonpolar",  aromatic=False, helix=1.08, sheet=1.60, flexibility=0.46),
    "K": dict(name="Lysine",        tla="Lys", hydropathy=-3.9, volume=168.6, charge=1.0,
              mw=146.19, polarity="positive",  aromatic=False, helix=1.16, sheet=0.74, flexibility=0.47),
    "L": dict(name="Leucine",       tla="Leu", hydropathy=3.8,  volume=166.7, charge=0.0,
              mw=131.17, polarity="nonpolar",  aromatic=False, helix=1.21, sheet=1.30, flexibility=0.37),
    "M": dict(name="Methionine",    tla="Met", hydropathy=1.9,  volume=162.9, charge=0.0,
              mw=149.21, polarity="nonpolar",  aromatic=False, helix=1.45, sheet=1.05, flexibility=0.30),
    "N": dict(name="Asparagine",    tla="Asn", hydropathy=-3.5, volume=114.1, charge=0.0,
              mw=132.12, polarity="polar",     aromatic=False, helix=0.67, sheet=0.89, flexibility=0.46),
    "P": dict(name="Proline",       tla="Pro", hydropathy=-1.6, volume=112.7, charge=0.0,
              mw=115.13, polarity="nonpolar",  aromatic=False, helix=0.57, sheet=0.55, flexibility=0.51),
    "Q": dict(name="Glutamine",     tla="Gln", hydropathy=-3.5, volume=143.8, charge=0.0,
              mw=146.15, polarity="polar",     aromatic=False, helix=1.11, sheet=1.10, flexibility=0.49),
    "R": dict(name="Arginine",      tla="Arg", hydropathy=-4.5, volume=173.4, charge=1.0,
              mw=174.20, polarity="positive",  aromatic=False, helix=0.98, sheet=0.93, flexibility=0.53),
    "S": dict(name="Serine",        tla="Ser", hydropathy=-0.8, volume=89.0,  charge=0.0,
              mw=105.09, polarity="polar",     aromatic=False, helix=0.77, sheet=0.75, flexibility=0.51),
    "T": dict(name="Threonine",     tla="Thr", hydropathy=-0.7, volume=116.1, charge=0.0,
              mw=119.12, polarity="polar",     aromatic=False, helix=0.83, sheet=1.19, flexibility=0.44),
    "V": dict(name="Valine",        tla="Val", hydropathy=4.2,  volume=140.0, charge=0.0,
              mw=117.15, polarity="nonpolar",  aromatic=False, helix=1.06, sheet=1.70, flexibility=0.39),
    "W": dict(name="Tryptophan",    tla="Trp", hydropathy=-0.9, volume=227.8, charge=0.0,
              mw=204.23, polarity="nonpolar",  aromatic=True,  helix=1.08, sheet=1.37, flexibility=0.31),
    "Y": dict(name="Tyrosine",      tla="Tyr", hydropathy=-1.3, volume=193.6, charge=0.0,
              mw=181.19, polarity="polar",     aromatic=True,  helix=0.69, sheet=1.47, flexibility=0.42),
}

NUMERIC_PROPS = ["hydropathy", "volume", "charge", "mw", "helix", "sheet", "flexibility"]

POLARITY_GROUPS = {
    "nonpolar": "疏水/非极性",
    "polar": "极性不带电",
    "positive": "正电荷",
    "negative": "负电荷",
}


def aa_table() -> pd.DataFrame:
    """氨基酸性质表(DataFrame 形式，便于展示和特征构建)。"""
    rows = []
    for aa in AA_ALPHABET:
        p = dict(AA_PROPERTIES[aa])
        p["aa"] = aa
        p["aromatic"] = bool(p["aromatic"])
        rows.append(p)
    df = pd.DataFrame(rows).set_index("aa").loc[list(AA_ALPHABET)]
    return df


# 2. BLOSUM62
_B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62_RAW = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _build_blosum62() -> dict[tuple[str, str], int]:
    mat = {}
    rows = [r for r in _B62_RAW.strip().splitlines() if r.strip()]
    for i, row in enumerate(rows):
        vals = [int(x) for x in row.split()]
        for j, v in enumerate(vals):
            mat[(_B62_ORDER[i], _B62_ORDER[j])] = v
    return mat


BLOSUM62 = _build_blosum62()


def blosum_score(a: str, b: str) -> int:
    return BLOSUM62.get((a, b), BLOSUM62.get((b, a), 0))


def substitution_class(a: str, b: str) -> str:
    """基于 BLOSUM62 把替换分为 保守 / 中性 / 激进。"""
    if a == b:
        return "同一残基"
    s = blosum_score(a, b)
    if s >= 1:
        return "保守替换"
    if s == 0:
        return "中性替换"
    if s >= -2:
        return "偏激进替换"
    return "激进替换"


def property_delta(a: str, b: str) -> dict[str, float]:
    """替换引起的理化性质变化 (b - a)。"""
    pa, pb = AA_PROPERTIES[a], AA_PROPERTIES[b]
    return {k: round(pb[k] - pa[k], 3) for k in ("hydropathy", "volume", "charge", "helix", "sheet", "flexibility")}


def describe_substitution(a: str, pos: int, b: str) -> str:
    """生成一句人类可读的替换描述，供 Agent 说明理由。"""
    d = property_delta(a, b)
    cls = substitution_class(a, b)
    bits = [f"{a}{pos}{b} 属于{cls}(BLOSUM62={blosum_score(a, b):+d})"]
    if abs(d["volume"]) >= 25:
        bits.append("侧链体积" + ("显著增大" if d["volume"] > 0 else "显著减小") + f"({d['volume']:+.1f} Å³)")
    if abs(d["hydropathy"]) >= 2.0:
        bits.append("疏水性" + ("明显增强" if d["hydropathy"] > 0 else "明显减弱") + f"({d['hydropathy']:+.1f})")
    if abs(d["charge"]) >= 1.0:
        bits.append(f"净电荷变化 {d['charge']:+.1f}")
    if AA_PROPERTIES[b]["aromatic"] and not AA_PROPERTIES[a]["aromatic"]:
        bits.append("引入芳香环，可能形成新的堆积/阳离子-π 相互作用")
    return "，".join(bits) + "。"


# 3. 位点结构注释
# GB1 (56 aa) 二级结构: β1 2-8, β2 13-19, α-helix 23-36, β3 42-46, β4 51-55
STRUCTURAL_CONTEXT: dict[int, dict] = {
    39: dict(wt="V", element="α螺旋→β3 连接环 (37-41)", burial="部分埋藏",
             note="靠近 IgG Fc 结合界面边缘，疏水侧链有助于界面堆积，对体积变化中等敏感。"),
    40: dict(wt="D", element="α螺旋→β3 连接环 (37-41)", burial="表面暴露",
             note="完全溶剂暴露的界面热点，野生型带负电；替换为大芳香残基常能新增结合界面接触。"),
    41: dict(wt="G", element="α螺旋→β3 连接环 (37-41)", burial="主链受限",
             note="甘氨酸处于常规残基不利的主链二面角区域，任何侧链都可能造成局部张力，单点替换风险极高。"),
    54: dict(wt="V", element="β4 折叠股 (51-55)", burial="埋藏于疏水核",
             note="位于 β 折叠内部，需要维持 β 股氢键与疏水堆积，忌引入脯氨酸或强极性残基。"),
}


# 4. 规则库
@dataclass
class RuleCheck:
    rule_id: str
    name: str
    severity: str          # "hard" = 直接否决, "soft" = 扣分提示
    passed: bool
    message: str
    penalty: float = 0.0   # 施加在预测 fitness(log 空间)上的惩罚


@dataclass
class DesignRuleBook:
    """蛋白质工程经验规则库。

    每条规则输入一个四位点组合(如 'FWAA')与上下文，输出 RuleCheck。
    """
    max_total_mutations: int = 4          # 文库本身只允许 4 个位点
    max_new_mutations_per_round: int = 2  # 单轮相对于当前最优变体的最大跨度
    max_radical_substitutions: int = 2    # 最多几个激进替换
    forbid_proline_in_strand: bool = True
    penalize_free_cysteine: bool = True
    max_abs_net_charge_change: float = 2.0
    high_risk_positions: tuple[int, ...] = (41,)   # 主链受限位点
    use_rescue_evidence: bool = True
    known_good_mutations: list[str] = field(default_factory=list)  # 由历史数据填充
    dead_mutations: set[str] = field(default_factory=set)          # 历史上低功能的单点
    # 有正向上位效应证据的突变对: {mutation: {可以补偿它的伙伴突变}}
    rescue_partners: dict[str, set[str]] = field(default_factory=dict)

    def prompt_rules(self) -> list[str]:
        """提示词与实际规则共用配置；软规则只增加风险和评分惩罚。"""
        rules = [
            "R1 硬约束：只允许 20 种标准氨基酸。",
            f"R2 硬约束：相对野生型的总突变数不超过 {self.max_total_mutations}。",
            f"R3 软约束：相对本轮设计骨架通常改动不超过 {self.max_new_mutations_per_round} 个位点；"
            "超限候选允许探索，但会扣分并须说明风险；所有突变都有有益单点或补偿证据时，跨度惩罚减半。",
            f"R4 软约束：相对野生型的激进替换（BLOSUM62 ≤ -3）建议不超过 {self.max_radical_substitutions} 个。",
        ]
        if self.forbid_proline_in_strand:
            rules.append("R5 硬约束：不在 β4 折叠股（位点 54）引入脯氨酸。")
        if self.penalize_free_cysteine:
            rules.append("R6 软约束：自由半胱氨酸增加风险和惩罚，不直接否决。")
        if self.high_risk_positions:
            rules.append(f"R7 软约束：位点 {list(self.high_risk_positions)} 的替换增加风险；"
                         "已测补偿证据可降低相应惩罚，但不能保证高阶组合仍然有益。")
        rules += [
            f"R8 软约束：净电荷变化建议在 ±{self.max_abs_net_charge_change:g} 内。",
            "R9 软约束：优先复用历史有益单点，组合收益仍需实验验证。",
            "R10 软约束：缺少补偿证据的历史低功能单点增加惩罚，不等于组合必然失活。",
        ]
        return rules

    # 规则注入
    def learn_from_history(self, measured: pd.DataFrame, wt_fitness: float = 1.0,
                           epistasis_records: list[dict] | None = None,
                           verbose: bool = False) -> "DesignRuleBook":
        """从"已完成实验"里抽取经验性规则。

        除了"哪些单点有益 / 低功能"，还要记录**上位效应证据**：
        哪些本身有害的突变，在特定伙伴突变存在时反而变得有益。
        这类证据的作用是"覆盖先验规则" —— 见 check() 中 R3/R7 的处理。
        """
        singles = measured[measured.n_mut == 1]
        good = singles[singles.fitness > wt_fitness].sort_values("fitness", ascending=False)
        self.known_good_mutations = [m[0] for m in good.mutations.tolist()]
        dead = singles[singles.fitness < 0.1 * wt_fitness]
        self.dead_mutations = {m[0] for m in dead.mutations.tolist()}

        self.rescue_partners = {}
        for rec in (epistasis_records or []) if self.use_rescue_evidence else []:
            if rec.get("obs_fitness", 0) <= wt_fitness or rec.get("epistasis", 0) <= 0:
                continue
            a, b = rec["mut_a"], rec["mut_b"]
            fa, fb = rec.get("single_a_fitness"), rec.get("single_b_fitness")
            if fa is None or fb is None:
                continue
            if fa < wt_fitness and rec["obs_fitness"] > fb:
                self.rescue_partners.setdefault(a, set()).add(b)
            if fb < wt_fitness and rec["obs_fitness"] > fa:
                self.rescue_partners.setdefault(b, set()).add(a)

        if verbose:
            print(f"[规则库] 从历史数据学到 {len(self.known_good_mutations)} 个有益单点突变，"
                  f"{len(self.dead_mutations)} 个低功能单点突变，"
                  f"{len(self.rescue_partners)} 个突变具备上位效应补偿证据")
            print(f"        有益单点 Top8: {self.known_good_mutations[:8]}")
            if self.rescue_partners:
                k = next(iter(self.rescue_partners))
                print(f"        例: {k} 可被 {sorted(self.rescue_partners[k])[:4]} 补偿")
        return self

    # 证据判定
    def has_epistasis_support(self, mut: str, muts: list[str]) -> str | None:
        """若 mut 在本变体内找到有实验证据的补偿伙伴，返回该伙伴，否则 None。"""
        partners = self.rescue_partners.get(mut, set())
        for other in muts:
            if other != mut and other in partners:
                return other
        return None

    def is_evidence_backed(self, mut: str, muts: list[str]) -> bool:
        """该突变要么本身已被验证有益，要么有上位效应补偿证据。"""
        return mut in self.known_good_mutations or self.has_epistasis_support(mut, muts) is not None

    # 逐条规则
    def check(self, combo: str, parent: str = WT_COMBO) -> list[RuleCheck]:
        checks: list[RuleCheck] = []
        try:
            validate_combo(combo)
            validate_combo(parent)
        except ValueError as exc:
            return [RuleCheck("R1", "序列合法性", "hard", False, str(exc), 99.0)]
        muts = combo_to_mutations(combo)
        n_mut = len(muts)

        # R1 合法字符
        illegal = [c for c in combo if c not in AA_ALPHABET]
        checks.append(RuleCheck(
            "R1", "序列合法性", "hard", not illegal,
            "全部为 20 种标准氨基酸，无终止密码子/非标准残基。" if not illegal
            else f"含非法残基 {illegal}，会引入终止密码子或非标准氨基酸。",
            penalty=0.0 if not illegal else 99.0))

        # R2 总突变数量
        ok2 = n_mut <= self.max_total_mutations
        checks.append(RuleCheck(
            "R2", "突变数量上限", "hard", ok2,
            f"共 {n_mut} 个突变，未超过文库允许的 {self.max_total_mutations} 个位点。" if ok2
            else f"共 {n_mut} 个突变，超过上限 {self.max_total_mutations}。",
            penalty=0.0 if ok2 else 99.0))

        # R3 单轮跨度（避免一次引入过多突变）
        # 例外：如果所有突变都有实验证据支撑（本身有益，或有上位效应补偿证据），
        # 那么"跨度大"不再等同于"不可预测"，惩罚减半。
        step = hamming(combo, parent)
        all_backed = bool(muts) and all(self.is_evidence_backed(m, muts) for m in muts)
        ok3 = step <= self.max_new_mutations_per_round
        raw_pen3 = 0.15 * max(0, step - self.max_new_mutations_per_round)
        if ok3:
            msg3 = f"相对当前最优变体 {parent} 只改动 {step} 个位点，符合小步迭代策略。"
        elif all_backed:
            msg3 = (f"相对 {parent} 改动了 {step} 个位点，超出小步迭代的常规跨度；"
                    f"但每个突变都有实验证据支撑（已验证有益或有上位效应补偿），风险可控，惩罚减半。")
            raw_pen3 *= 0.5
        else:
            msg3 = f"相对 {parent} 一次改动 {step} 个位点，跨度过大，上位效应不可预测、失败风险升高。"
        checks.append(RuleCheck("R3", "单轮突变跨度", "soft", ok3, msg3, penalty=raw_pen3))

        # R4 激进替换数量
        radical = [m for m in muts if substitution_class(m[0], m[-1]) == "激进替换"]
        ok4 = len(radical) <= self.max_radical_substitutions
        checks.append(RuleCheck(
            "R4", "保守/激进替换平衡", "soft", ok4,
            (f"激进替换 {len(radical)} 个({','.join(radical) if radical else '无'})，在可接受范围内。") if ok4
            else f"激进替换过多({','.join(radical)})，结构扰动风险高。",
            penalty=0.0 if ok4 else 0.12 * (len(radical) - self.max_radical_substitutions)))

        # R5 β 股中的脯氨酸
        bad_pro = [m for m in muts
                   if m[-1] == "P" and STRUCTURAL_CONTEXT.get(int(m[1:-1]), {}).get("element", "").startswith("β")]
        ok5 = not (self.forbid_proline_in_strand and bad_pro)
        checks.append(RuleCheck(
            "R5", "β 折叠股禁脯氨酸", "hard" if self.forbid_proline_in_strand else "soft", ok5,
            "未在 β 折叠股内引入脯氨酸。" if ok5
            else f"{','.join(bad_pro)} 在 β 股内引入脯氨酸，会破坏主链氢键网络。",
            penalty=0.0 if ok5 else 99.0))

        # R6 自由半胱氨酸
        new_cys = [m for m in muts if m[-1] == "C"]
        ok6 = not (self.penalize_free_cysteine and new_cys)
        checks.append(RuleCheck(
            "R6", "自由半胱氨酸风险", "soft", ok6,
            "未引入额外的自由半胱氨酸。" if ok6
            else f"{','.join(new_cys)} 引入自由巯基，可能造成错误二硫键或聚集(仅提示，不否决)。",
            penalty=0.0 if ok6 else 0.08 * len(new_cys)))

        # R7 高风险主链受限位点
        # 实测双突变中的效应方向翻转可降低对应组合的先验风险，
        # 但不保证这一补偿关系在更高阶背景下仍然成立。
        risky = [m for m in muts if int(m[1:-1]) in self.high_risk_positions]
        rescued = {m: self.has_epistasis_support(m, muts) for m in risky}
        unsupported = [m for m, p in rescued.items() if p is None]
        ok7 = not unsupported
        if not risky:
            msg7 = "未改动主链构象受限的高风险位点。"
        elif ok7:
            msg7 = ("；".join(f"{m} 虽位于主链受限位点，但与 {p} 的组合已有正向上位效应实测证据，"
                             f"属于有条件放行" for m, p in rescued.items()) + "。")
        else:
            msg7 = (f"{','.join(unsupported)} 位于主链受限位点"
                    f"({'/'.join(str(p) for p in self.high_risk_positions)})，"
                    f"单点替换历史上几乎全部有害，且本变体中找不到有证据的补偿伙伴。")
        checks.append(RuleCheck("R7", "主链受限位点", "soft", ok7, msg7,
                                penalty=0.10 * len(unsupported)))

        # R8 电荷平衡
        dq = sum(AA_PROPERTIES[m[-1]]["charge"] - AA_PROPERTIES[m[0]]["charge"] for m in muts)
        ok8 = abs(dq) <= self.max_abs_net_charge_change
        checks.append(RuleCheck(
            "R8", "净电荷平衡", "soft", ok8,
            f"净电荷变化 {dq:+.1f}，在可接受范围内。" if ok8
            else f"净电荷变化 {dq:+.1f}，过大的静电扰动可能破坏结合界面。",
            penalty=0.0 if ok8 else 0.10 * (abs(dq) - self.max_abs_net_charge_change)))

        # R9 优先组合历史上表现好的单点
        hit = [m for m in muts if m in self.known_good_mutations]
        ok9 = bool(hit) or n_mut == 0
        checks.append(RuleCheck(
            "R9", "复用历史有益突变", "soft", ok9,
            f"包含历史验证的有益单点 {','.join(hit)}，组合效应仍需实验确认。" if hit
            else "未包含任何历史验证过的有益单点突变，属于纯探索型设计。",
            penalty=0.0 if ok9 else 0.10))

        # R10 规避历史低功能突变
        lethal = [m for m in muts if m in self.dead_mutations
                  and self.has_epistasis_support(m, muts) is None]
        ok10 = not lethal
        checks.append(RuleCheck(
            "R10", "规避已知低功能突变", "soft", ok10,
            "未包含缺少组合补偿证据的历史低功能单点突变。" if ok10
            else (f"包含历史低功能单点 {','.join(lethal)}；除非有明确的上位效应证据，"
                  f"否则成功概率很低。"),
            penalty=0.0 if ok10 else 0.20 * len(lethal)))

        return checks

    # 汇总
    def evaluate(self, combo: str, parent: str = WT_COMBO) -> dict:
        checks = self.check(combo, parent)
        hard_fail = [c for c in checks if c.severity == "hard" and not c.passed]
        soft_fail = [c for c in checks if c.severity == "soft" and not c.passed]
        penalty = sum(c.penalty for c in soft_fail)
        return {
            "variant": combo,
            "feasible": len(hard_fail) == 0,
            "penalty": round(float(min(penalty, 1.5)), 4),
            "n_pass": sum(1 for c in checks if c.passed),
            "n_total": len(checks),
            "hard_violations": [c.message for c in hard_fail],
            "soft_warnings": [c.message for c in soft_fail],
            "passed_notes": [c.message for c in checks if c.passed],
            "checks": [c.__dict__ for c in checks],
        }


# 5. 知识图谱
@dataclass
class KnowledgeGraph:
    """三元组形式的小型知识图谱。

    关系类型:
        AminoAcid --has_property--> Property
        AminoAcid --conservative_to--> AminoAcid
        Mutation  --occurs_at--> Position
        Mutation  --introduces--> AminoAcid
        Mutation  --improves / impairs--> Fitness
        Variant   --contains--> Mutation
        Position  --located_in--> StructuralElement
    """
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    node_types: dict[str, str] = field(default_factory=dict)

    def add(self, h: str, r: str, t: str, ht: str = "Entity", tt: str = "Entity") -> None:
        self.triples.append((h, r, t))
        self.node_types.setdefault(h, ht)
        self.node_types.setdefault(t, tt)

    # 构建
    @classmethod
    def build(cls, measured: pd.DataFrame, wt_fitness: float = 1.0,
              top_variants: int = 15, verbose: bool = True) -> "KnowledgeGraph":
        kg = cls()

        # (1) 氨基酸 -> 理化性质
        for aa, p in AA_PROPERTIES.items():
            kg.add(aa, "has_property", POLARITY_GROUPS[p["polarity"]], "AminoAcid", "Property")
            kg.add(aa, "has_property", "疏水" if p["hydropathy"] > 0 else "亲水", "AminoAcid", "Property")
            kg.add(aa, "has_property", "大体积(>160Å³)" if p["volume"] > 160 else
                   ("小体积(<110Å³)" if p["volume"] < 110 else "中等体积"), "AminoAcid", "Property")
            if p["aromatic"]:
                kg.add(aa, "has_property", "芳香族", "AminoAcid", "Property")
            if p["sheet"] > 1.2:
                kg.add(aa, "has_property", "β股倾向强", "AminoAcid", "Property")

        # (2) 保守替换关系
        for a in AA_ALPHABET:
            for b in AA_ALPHABET:
                if a < b and blosum_score(a, b) >= 1:
                    kg.add(a, "conservative_to", b, "AminoAcid", "AminoAcid")

        # (3) 位点 -> 结构元件
        for pos, ctx in STRUCTURAL_CONTEXT.items():
            kg.add(f"Pos{pos}", "located_in", ctx["element"], "Position", "StructuralElement")
            kg.add(f"Pos{pos}", "has_property", ctx["burial"], "Position", "Property")
            kg.add(f"Pos{pos}", "wild_type_is", ctx["wt"], "Position", "AminoAcid")

        # (4) 从实验数据学到的突变效应
        singles = measured[measured.n_mut == 1]
        for _, r in singles.iterrows():
            m = r.mutations[0]
            pos = int(m[1:-1])
            kg.add(m, "occurs_at", f"Pos{pos}", "Mutation", "Position")
            kg.add(m, "introduces", m[-1], "Mutation", "AminoAcid")
            if r.fitness > wt_fitness:
                kg.add(m, "improves", "Fitness", "Mutation", "Phenotype")
            elif r.fitness < 0.1 * wt_fitness:
                kg.add(m, "abolishes", "Fitness", "Mutation", "Phenotype")
            else:
                kg.add(m, "impairs", "Fitness", "Mutation", "Phenotype")

        # (5) 高适应度变体 -> 所含突变
        for _, r in measured.nlargest(top_variants, "fitness").iterrows():
            if r.n_mut == 0:
                continue
            for m in r.mutations:
                kg.add(r.variant, "contains", m, "Variant", "Mutation")
            kg.add(r.variant, "has_phenotype", "高适应度", "Variant", "Phenotype")

        if verbose:
            print(f"[知识图谱] 三元组 {len(kg.triples):,} 条，节点 {len(kg.node_types):,} 个")
            print(f"[知识图谱] 关系类型: {sorted(set(r for _, r, _ in kg.triples))}")
        return kg

    # 查询
    def query(self, head: str | None = None, rel: str | None = None,
              tail: str | None = None) -> list[tuple[str, str, str]]:
        return [t for t in self.triples
                if (head is None or t[0] == head)
                and (rel is None or t[1] == rel)
                and (tail is None or t[2] == tail)]

    def beneficial_mutations(self) -> list[str]:
        return [h for h, _, _ in self.query(rel="improves", tail="Fitness")]

    def lethal_mutations(self) -> list[str]:
        return [h for h, _, _ in self.query(rel="abolishes", tail="Fitness")]

    def mutations_at(self, pos: int) -> list[str]:
        return [h for h, _, _ in self.query(rel="occurs_at", tail=f"Pos{pos}")]

    def properties_of(self, aa: str) -> list[str]:
        return [t for _, _, t in self.query(head=aa, rel="has_property")]

    def explain_mutation(self, mut: str) -> str:
        """用图谱里的路径解释一个突变，供 Agent 生成理由。"""
        wt_aa, pos, new_aa = mut[0], int(mut[1:-1]), mut[-1]
        ctx = STRUCTURAL_CONTEXT.get(pos, {})
        effects = [r for _, r, t in self.query(head=mut) if t in ("Fitness",)]
        eff_txt = {"improves": "历史数据中提升适应度",
                   "impairs": "历史数据中降低适应度",
                   "abolishes": "历史数据中导致功能丧失"}.get(effects[0] if effects else "", "历史数据中未单独测量")
        props_new = "、".join(self.properties_of(new_aa)[:3])
        return (f"{mut}: 位点 {pos} 位于{ctx.get('element', '未知结构元件')}({ctx.get('burial', '')})；"
                f"引入的 {new_aa} 具有[{props_new}]；{describe_substitution(wt_aa, pos, new_aa)}"
                f"该单点{eff_txt}。")

    # 导出
    def to_records(self) -> list[dict]:
        return [{"head": h, "relation": r, "tail": t,
                 "head_type": self.node_types.get(h, "Entity"),
                 "tail_type": self.node_types.get(t, "Entity")}
                for h, r, t in self.triples]

    def save(self, path=None) -> str:
        path = path or (RESULT_DIR / "knowledge_graph.json")
        payload = {"n_triples": len(self.triples),
                   "n_nodes": len(self.node_types),
                   "triples": self.to_records()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[保存] 知识图谱 -> {rel(path)}")
        return str(path)
