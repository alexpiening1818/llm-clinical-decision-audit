"""
Guideline-Grounded Case Generator for Early-Stage Breast Cancer Adjuvant RT Decisions

Generates synthetic but clinically realistic patient cases sampled from plausible
population distributions, then maps each case to guideline-adherent reference
answers derived from NCCN Breast Cancer Guidelines v5.2025 and ASTRO consensus.

The reference logic is intentionally explicit and auditable. Every reference answer
is accompanied by the rule that produced it, so the decision tree is inspectable
and pre-registrable as a study protocol.

Decision tasks evaluated:
  1. Adjuvant RT recommendation (offer / conditional / omission acceptable)
  2. Target volume (whole breast / partial breast / chest wall / + regional nodes)
  3. Fractionation (conventional / moderate hypo / ultra-hypo / APBI)
  4. Tumor bed boost (indicated / not indicated / optional)
  5. RNI indication (yes / strongly consider / no)

Author: [your name]
Version: 0.1 (pre-registration draft)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional
import argparse


# ---------------------------------------------------------------------------
# Patient data model
# ---------------------------------------------------------------------------

@dataclass
class BreastCancerCase:
    """A single synthetic patient case. All fields are what a clinician would
    see at the adjuvant RT consultation after primary surgery."""

    case_id: str

    # Demographics
    age: int
    menopausal_status: str          # "premenopausal" | "postmenopausal"
    ecog: int                        # 0-4

    # Comorbidities / frailty
    significant_comorbidity: bool    # cardiac, pulmonary, etc that affect RT tolerance
    life_expectancy_lt_10yr: bool    # clinical judgment proxy

    # Tumor features
    histology: str                   # "IDC" | "ILC" | "DCIS" | "mixed"
    grade: int                       # 1-3
    tumor_size_cm: float
    lvi_present: bool
    multifocal: bool

    # Receptor / molecular
    er_positive: bool
    pr_positive: bool
    her2_positive: bool
    oncotype_score: Optional[int]    # 0-100 or None
    brca_mutation: bool

    # Surgical / pathologic
    surgery_type: str                # "BCS" | "mastectomy"
    margin_status: str               # "negative" | "close" | "positive"
    margin_mm: Optional[float]       # mm clearance if known
    nodes_examined: int
    nodes_positive: int
    sentinel_only: bool

    # Staging
    clinical_t: str                  # "cT1" | "cT2" | ...
    clinical_n: str                  # "cN0" | "cN1" | ...
    pathologic_t: str                # "pT1" | "pT1mic" | "pTis" | ...
    pathologic_n: str                # "pN0" | "pN1mi" | "pN1" | ...

    # Systemic therapy plan
    planned_endocrine_therapy: bool
    planned_chemotherapy: bool
    planned_her2_therapy: bool

    # Reference answers (populated by guideline engine)
    reference: dict = field(default_factory=dict)

    def to_clinical_vignette(self) -> str:
        """Render case as a clinical vignette for LLM prompting."""
        lines = []
        lines.append(f"A {self.age}-year-old {self.menopausal_status} woman with ECOG {self.ecog}.")

        comorbid_text = ""
        if self.significant_comorbidity:
            comorbid_text = " She has significant cardiopulmonary comorbidity."
        if self.life_expectancy_lt_10yr:
            comorbid_text += " Clinical life expectancy is estimated under 10 years."
        if comorbid_text:
            lines.append(comorbid_text.strip())

        # Tumor
        tumor = (
            f"She was diagnosed with {self.histology} of the breast, grade {self.grade}, "
            f"{self.tumor_size_cm:.1f} cm, {'with' if self.lvi_present else 'without'} LVI, "
            f"{'multifocal' if self.multifocal else 'unifocal'}."
        )
        lines.append(tumor)

        # Receptors
        rec = (
            f"ER {'positive' if self.er_positive else 'negative'}, "
            f"PR {'positive' if self.pr_positive else 'negative'}, "
            f"HER2 {'positive' if self.her2_positive else 'negative'}."
        )
        if self.oncotype_score is not None:
            rec += f" Oncotype DX recurrence score {self.oncotype_score}."
        if self.brca_mutation:
            rec += " She is a known BRCA mutation carrier."
        lines.append(rec)

        # Surgery
        if self.surgery_type == "BCS":
            surg = "She underwent breast-conserving surgery"
        else:
            surg = "She underwent mastectomy"
        surg += f" with {self.margin_status} margins"
        if self.margin_mm is not None:
            surg += f" ({self.margin_mm:.1f} mm clearance)"
        surg += "."
        lines.append(surg)

        # Nodes
        if self.sentinel_only and self.nodes_examined <= 3:
            node_text = (
                f"Sentinel lymph node biopsy examined {self.nodes_examined} node(s) "
                f"with {self.nodes_positive} positive."
            )
        else:
            node_text = (
                f"Axillary evaluation examined {self.nodes_examined} nodes "
                f"with {self.nodes_positive} positive."
            )
        lines.append(node_text)

        # Staging
        lines.append(
            f"Staging: clinical {self.clinical_t}{self.clinical_n}, "
            f"pathologic {self.pathologic_t}{self.pathologic_n}."
        )

        # Planned systemic
        planned = []
        if self.planned_endocrine_therapy:
            planned.append("adjuvant endocrine therapy")
        if self.planned_chemotherapy:
            planned.append("adjuvant chemotherapy")
        if self.planned_her2_therapy:
            planned.append("HER2-directed therapy")
        if planned:
            lines.append("Planned adjuvant systemic therapy: " + ", ".join(planned) + ".")
        else:
            lines.append("No adjuvant systemic therapy is currently planned.")

        return " ".join(lines)


# ---------------------------------------------------------------------------
# Sampling distributions
# ---------------------------------------------------------------------------

def sample_case(rng: random.Random, case_id: str) -> BreastCancerCase:
    """Sample a single case from realistic distributions. Distributions are
    calibrated to approximate the SEER early-stage breast cancer population
    weighted toward cases where the RT decision is clinically interesting
    (i.e., stage 0-IIA BCS or mastectomy candidates, with some oversampling
    of older patients and edge cases where guidelines bifurcate)."""

    # Age: bimodal to capture both screening-detected and elderly populations
    if rng.random() < 0.35:
        # Older population (enriched to stress-test omission decisions)
        age = int(rng.gauss(72, 7))
    else:
        age = int(rng.gauss(58, 10))
    age = max(35, min(92, age))

    menopausal = "postmenopausal" if age >= 52 else ("premenopausal" if age < 48 else rng.choice(["premenopausal", "postmenopausal"]))

    # ECOG skewed healthy but with tail
    ecog = rng.choices([0, 1, 2, 3], weights=[55, 30, 12, 3])[0]

    # Comorbidity correlated with age and ECOG
    comorb_prob = 0.05 + 0.012 * max(0, age - 50) + 0.15 * ecog
    significant_comorbidity = rng.random() < min(0.7, comorb_prob)

    life_expectancy_lt_10yr = (
        (age >= 80) or
        (age >= 70 and significant_comorbidity) or
        (ecog >= 2)
    ) and (rng.random() < 0.7)

    # Histology
    histology = rng.choices(
        ["IDC", "ILC", "DCIS", "mixed"],
        weights=[70, 12, 15, 3]
    )[0]

    # Size — biased toward early stage
    if histology == "DCIS":
        size = rng.uniform(0.3, 3.5)
    else:
        size = abs(rng.gauss(1.8, 1.2))
        size = max(0.2, min(size, 7.0))

    grade = rng.choices([1, 2, 3], weights=[25, 45, 30])[0]

    # LVI more common with higher grade / larger tumors
    lvi_prob = 0.05 + 0.05 * grade + 0.03 * size
    lvi_present = rng.random() < min(0.5, lvi_prob) and histology != "DCIS"

    multifocal = rng.random() < 0.08

    # Receptors
    er_positive = rng.random() < (0.78 if histology != "DCIS" else 0.85)
    pr_positive = er_positive and rng.random() < 0.85
    her2_positive = rng.random() < (0.15 if histology != "DCIS" else 0.05)

    # Oncotype only in ER+/HER2-/node-negative invasive
    oncotype_score = None

    brca_mutation = rng.random() < 0.04

    # Surgery — biased toward BCS for early stage
    if size > 5.0:
        surgery_type = "mastectomy"
    elif size > 3.0:
        surgery_type = rng.choices(["BCS", "mastectomy"], weights=[45, 55])[0]
    else:
        surgery_type = rng.choices(["BCS", "mastectomy"], weights=[80, 20])[0]

    # Margins
    margin_status_roll = rng.random()
    if margin_status_roll < 0.82:
        margin_status = "negative"
        margin_mm = round(rng.uniform(2.0, 10.0), 1)
    elif margin_status_roll < 0.93:
        margin_status = "close"
        margin_mm = round(rng.uniform(0.5, 1.9), 1)
    else:
        margin_status = "positive"
        margin_mm = 0.0

    # Nodal evaluation
    if histology == "DCIS":
        # DCIS usually doesn't get axillary surgery unless mastectomy
        if surgery_type == "mastectomy":
            nodes_examined = rng.randint(1, 3)
            sentinel_only = True
        else:
            nodes_examined = 0
            sentinel_only = True
        nodes_positive = 0
    else:
        # Most early breast cancer: SLNB first
        sentinel_only = rng.random() < 0.75
        if sentinel_only:
            nodes_examined = rng.randint(1, 4)
        else:
            nodes_examined = rng.randint(8, 22)

        # Nodal positivity — depends on size/grade/LVI
        node_pos_prob = 0.05 + 0.04 * size + 0.03 * grade + (0.10 if lvi_present else 0)
        node_pos_prob = min(0.55, node_pos_prob)
        if rng.random() < node_pos_prob:
            if sentinel_only:
                nodes_positive = min(nodes_examined, rng.choices([1, 2, 3], weights=[60, 30, 10])[0])
            else:
                nodes_positive = min(nodes_examined, rng.choices([1, 2, 3, 4, 5, 8], weights=[30, 20, 15, 15, 10, 10])[0])
        else:
            nodes_positive = 0

    # Oncotype: order in ER+/HER2-/node-neg invasive
    if (
        histology not in ("DCIS",)
        and er_positive and not her2_positive
        and nodes_positive == 0 and size <= 5.0
    ):
        if rng.random() < 0.6:
            oncotype_score = rng.choices(
                [rng.randint(0, 10), rng.randint(11, 25), rng.randint(26, 100)],
                weights=[35, 45, 20]
            )[0]

    # Staging
    if histology == "DCIS":
        pathologic_t = "pTis"
        clinical_t = "cTis"
    elif size < 0.1:
        pathologic_t = "pT1mic"
        clinical_t = "cT1"
    elif size <= 2.0:
        pathologic_t = "pT1"
        clinical_t = "cT1"
    elif size <= 5.0:
        pathologic_t = "pT2"
        clinical_t = "cT2"
    else:
        pathologic_t = "pT3"
        clinical_t = "cT3"

    if nodes_positive == 0:
        pathologic_n = "pN0"
        clinical_n = "cN0"
    elif nodes_positive <= 3:
        pathologic_n = "pN1"
        clinical_n = "cN1"
    elif nodes_positive <= 9:
        pathologic_n = "pN2"
        clinical_n = "cN1"
    else:
        pathologic_n = "pN3"
        clinical_n = "cN2"

    # Micrometastasis override
    if nodes_positive == 1 and rng.random() < 0.15 and sentinel_only:
        pathologic_n = "pN1mi"

    # Systemic therapy plans
    planned_endocrine_therapy = er_positive
    planned_chemotherapy = (
        nodes_positive >= 1 or
        (oncotype_score is not None and oncotype_score >= 26) or
        size >= 2.0 or
        grade == 3 or
        her2_positive or
        (not er_positive and not her2_positive and histology != "DCIS")  # TNBC
    )
    # Downweight chemo in elderly/frail
    if age >= 75 or life_expectancy_lt_10yr:
        if rng.random() < 0.5:
            planned_chemotherapy = False
    planned_her2_therapy = her2_positive

    return BreastCancerCase(
        case_id=case_id,
        age=age,
        menopausal_status=menopausal,
        ecog=ecog,
        significant_comorbidity=significant_comorbidity,
        life_expectancy_lt_10yr=life_expectancy_lt_10yr,
        histology=histology,
        grade=grade,
        tumor_size_cm=round(size, 1),
        lvi_present=lvi_present,
        multifocal=multifocal,
        er_positive=er_positive,
        pr_positive=pr_positive,
        her2_positive=her2_positive,
        oncotype_score=oncotype_score,
        brca_mutation=brca_mutation,
        surgery_type=surgery_type,
        margin_status=margin_status,
        margin_mm=margin_mm,
        nodes_examined=nodes_examined,
        nodes_positive=nodes_positive,
        sentinel_only=sentinel_only,
        clinical_t=clinical_t,
        clinical_n=clinical_n,
        pathologic_t=pathologic_t,
        pathologic_n=pathologic_n,
        planned_endocrine_therapy=planned_endocrine_therapy,
        planned_chemotherapy=planned_chemotherapy,
        planned_her2_therapy=planned_her2_therapy,
    )


# ---------------------------------------------------------------------------
# Guideline engine: NCCN Breast Cancer v5.2025 + ASTRO consensus
# ---------------------------------------------------------------------------
#
# Every reference answer is accompanied by the rule ID that produced it.
# Rules are numbered R### and cited in the output. Reviewers can audit the
# mapping from clinical input to reference answer by following rule IDs.
# ---------------------------------------------------------------------------

def apply_guidelines(case: BreastCancerCase) -> dict:
    """Apply NCCN-derived guideline logic to produce a reference decision set."""

    rules_triggered = []
    reference = {}

    # -----------------------------------------------------------------------
    # Decision 1: Should adjuvant RT be offered?
    # -----------------------------------------------------------------------

    rt_recommendation = None
    rt_reason = None

    # CALGB 9343 / PRIME II omission criteria
    omission_eligible = (
        case.age >= 70
        and case.histology != "DCIS"
        and case.surgery_type == "BCS"
        and case.margin_status == "negative"
        and case.tumor_size_cm <= 2.0
        and case.pathologic_t == "pT1"
        and case.nodes_positive == 0
        and case.er_positive
        and case.planned_endocrine_therapy
        and not case.her2_positive
    )

    if omission_eligible:
        rt_recommendation = "omission acceptable"
        rt_reason = "R001: meets CALGB 9343 / PRIME II omission criteria (age ≥70, pT1N0, ER+, BCS with negative margins, planned endocrine therapy)"
        rules_triggered.append("R001")
    elif case.surgery_type == "BCS" and case.histology == "DCIS":
        # DCIS post-BCS: RT generally offered, but omission can be considered in low-risk
        # NCCN accepts RT or RT-omission with endocrine therapy in low-risk DCIS
        if (
            case.age >= 50
            and case.tumor_size_cm <= 2.5
            and case.margin_status == "negative"
            and case.margin_mm is not None and case.margin_mm >= 3.0
            and case.grade <= 2
        ):
            rt_recommendation = "conditional"
            rt_reason = "R002: DCIS post-BCS with low-risk features (age ≥50, ≤2.5 cm, grade 1-2, wide margins); RT or RT-omission with endocrine therapy are both acceptable per NCCN"
            rules_triggered.append("R002")
        else:
            rt_recommendation = "offer"
            rt_reason = "R003: DCIS post-BCS without low-risk features; adjuvant RT recommended"
            rules_triggered.append("R003")
    elif case.surgery_type == "BCS":
        rt_recommendation = "offer"
        rt_reason = "R004: invasive breast cancer post-BCS; adjuvant RT recommended"
        rules_triggered.append("R004")
    elif case.surgery_type == "mastectomy":
        # PMRT indications
        strong_pmrt = (
            case.nodes_positive >= 4
            or case.margin_status == "positive"
            or case.pathologic_t in ("pT3", "pT4")
            or case.tumor_size_cm > 5.0
        )
        moderate_pmrt = (
            1 <= case.nodes_positive <= 3
        )
        if strong_pmrt:
            rt_recommendation = "offer"
            rt_reason = "R005: classic PMRT indication (≥4 positive nodes, positive margins, pT3/T4, or tumor >5 cm)"
            rules_triggered.append("R005")
        elif moderate_pmrt:
            rt_recommendation = "offer"
            rt_reason = "R006: 1-3 positive nodes post-mastectomy; PMRT strongly recommended per NCCN"
            rules_triggered.append("R006")
        elif case.histology == "DCIS":
            rt_recommendation = "omission acceptable"
            rt_reason = "R007: DCIS post-mastectomy with negative margins; RT typically not indicated"
            rules_triggered.append("R007")
        else:
            rt_recommendation = "omission acceptable"
            rt_reason = "R008: node-negative invasive cancer post-mastectomy with negative margins, no high-risk features; PMRT not routinely indicated"
            rules_triggered.append("R008")

    reference["rt_recommendation"] = {
        "answer": rt_recommendation,
        "rule": rt_reason,
    }

    # -----------------------------------------------------------------------
    # Decision 2: Target volume
    # -----------------------------------------------------------------------

    target_volume = None
    target_reason = None

    if rt_recommendation == "omission acceptable" and case.surgery_type == "BCS":
        target_volume = "not applicable (omission)"
        target_reason = "R010: RT omission acceptable per decision 1"
        rules_triggered.append("R010")
    elif rt_recommendation == "omission acceptable" and case.surgery_type == "mastectomy":
        target_volume = "not applicable (no PMRT indicated)"
        target_reason = "R011: PMRT not indicated per decision 1"
        rules_triggered.append("R011")
    elif case.surgery_type == "BCS":
        # APBI suitability (ASTRO consensus criteria, adopted by NCCN)
        apbi_suitable = (
            case.age >= 50
            and case.histology != "ILC"
            and case.tumor_size_cm <= 2.0
            and case.pathologic_t in ("pT1", "pTis")
            and case.nodes_positive == 0
            and case.margin_status == "negative"
            and case.margin_mm is not None and case.margin_mm >= 2.0
            and not case.lvi_present
            and case.er_positive
            and not case.brca_mutation
            and not case.multifocal
        )
        # DCIS APBI has slightly different criteria
        dcis_apbi_suitable = (
            case.histology == "DCIS"
            and case.age >= 50
            and case.tumor_size_cm <= 2.5
            and case.grade <= 2
            and case.margin_status == "negative"
            and case.margin_mm is not None and case.margin_mm >= 3.0
        )

        rni_answer = _check_rni_indication(case)[0]

        if rni_answer in ("yes", "strongly consider"):
            target_volume = "whole breast plus regional nodal irradiation"
            target_reason = "R012: RNI indicated (see RNI decision); target is whole breast + RNI"
            rules_triggered.append("R012")
        elif apbi_suitable or dcis_apbi_suitable:
            target_volume = "partial breast (APBI) or whole breast"
            target_reason = "R013: meets ASTRO APBI suitability criteria; APBI or WBI both acceptable"
            rules_triggered.append("R013")
        else:
            target_volume = "whole breast"
            target_reason = "R014: standard whole-breast irradiation post-BCS when APBI criteria not met and RNI not indicated"
            rules_triggered.append("R014")
    elif case.surgery_type == "mastectomy":
        rni_answer = _check_rni_indication(case)[0]
        if rni_answer in ("yes", "strongly consider"):
            target_volume = "chest wall plus regional nodal irradiation"
            target_reason = "R015: PMRT indicated with RNI (≥1 positive node or high-risk node-negative)"
            rules_triggered.append("R015")
        else:
            target_volume = "chest wall"
            target_reason = "R016: PMRT indicated without RNI"
            rules_triggered.append("R016")

    reference["target_volume"] = {
        "answer": target_volume,
        "rule": target_reason,
    }

    # -----------------------------------------------------------------------
    # Decision 3: Fractionation
    # -----------------------------------------------------------------------

    fractionation = None
    frac_reason = None

    if rt_recommendation == "omission acceptable":
        fractionation = "not applicable"
        frac_reason = "R020: RT omission per decision 1"
        rules_triggered.append("R020")
    elif target_volume and "partial breast" in target_volume:
        fractionation = "APBI (e.g., 30 Gy in 5 fx or 38.5 Gy in 10 fx BID)"
        frac_reason = "R021: APBI regimen per ASTRO-accepted schedules"
        rules_triggered.append("R021")
    elif target_volume and "whole breast" in target_volume and "regional nodal" not in target_volume:
        # FAST-Forward eligibility: pT1-T2 pN0, BCS, WBI alone
        fast_forward_eligible = (
            case.surgery_type == "BCS"
            and case.pathologic_t in ("pT1", "pT2", "pTis")
            and case.nodes_positive == 0
        )
        if fast_forward_eligible:
            fractionation = "moderate hypofractionation (40 Gy / 15 fx) preferred; ultra-hypofractionation (26 Gy / 5 fx per FAST-Forward) acceptable"
            frac_reason = "R022: WBI alone post-BCS in node-negative disease; moderate hypofractionation preferred, ultra-hypofractionation per FAST-Forward acceptable"
            rules_triggered.append("R022")
        else:
            fractionation = "moderate hypofractionation (40 Gy / 15 fx) preferred"
            frac_reason = "R023: WBI with moderate hypofractionation preferred per NCCN"
            rules_triggered.append("R023")
    elif target_volume and "regional nodal" in target_volume:
        # RNI: NCCN accepts moderate hypofractionation based on recent data (RT CHARM, Alliance A221505)
        fractionation = "moderate hypofractionation (40 Gy / 15 fx) acceptable for WBI+RNI or CW+RNI; conventional fractionation (50 Gy / 25 fx) also acceptable"
        frac_reason = "R024: hypofractionation increasingly accepted for RNI; both moderate hypo and conventional acceptable"
        rules_triggered.append("R024")
    elif target_volume == "chest wall":
        fractionation = "moderate hypofractionation (40 Gy / 15 fx) or conventional (50 Gy / 25 fx) acceptable"
        frac_reason = "R025: PMRT to chest wall alone; hypofractionation acceptable"
        rules_triggered.append("R025")

    reference["fractionation"] = {
        "answer": fractionation,
        "rule": frac_reason,
    }

    # -----------------------------------------------------------------------
    # Decision 4: Tumor bed boost
    # -----------------------------------------------------------------------

    boost = None
    boost_reason = None

    if rt_recommendation == "omission acceptable":
        boost = "not applicable"
        boost_reason = "R030: RT omission per decision 1"
        rules_triggered.append("R030")
    elif case.surgery_type == "mastectomy":
        # Boost to scar/chest wall sometimes used but not standard
        boost = "not routinely indicated"
        boost_reason = "R031: tumor bed boost not routinely used post-mastectomy"
        rules_triggered.append("R031")
    elif target_volume and "partial breast" in target_volume:
        boost = "not applicable (APBI already treats tumor bed)"
        boost_reason = "R032: APBI targets tumor bed; separate boost not used"
        rules_triggered.append("R032")
    elif case.histology == "DCIS":
        # DCIS boost: consider in younger patients, close margins, high grade
        if case.age < 50 or case.grade == 3 or case.margin_status == "close":
            boost = "consider"
            boost_reason = "R033: DCIS with risk factors for local recurrence (young age, high grade, or close margins); boost may be considered"
            rules_triggered.append("R033")
        else:
            boost = "optional"
            boost_reason = "R034: DCIS without high-risk features; boost optional"
            rules_triggered.append("R034")
    else:
        # Invasive post-BCS
        # EORTC 22881/10882 established boost benefit most pronounced in <50; ongoing benefit to older
        if case.age < 50:
            boost = "indicated"
            boost_reason = "R035: age <50 with invasive disease post-BCS; boost indicated per EORTC 22881/10882"
            rules_triggered.append("R035")
        elif case.age >= 50 and (case.grade == 3 or case.lvi_present or case.margin_status == "close"):
            boost = "indicated"
            boost_reason = "R036: age ≥50 with high-risk features (grade 3, LVI, or close margins); boost indicated"
            rules_triggered.append("R036")
        elif case.age >= 70 and case.tumor_size_cm <= 2.0 and case.nodes_positive == 0 and case.er_positive:
            boost = "optional"
            boost_reason = "R037: age ≥70 with low-risk invasive disease; boost optional, minimal absolute benefit"
            rules_triggered.append("R037")
        else:
            boost = "consider"
            boost_reason = "R038: invasive disease post-BCS without high-risk features; boost may be considered based on individual risk"
            rules_triggered.append("R038")

    reference["boost"] = {
        "answer": boost,
        "rule": boost_reason,
    }

    # -----------------------------------------------------------------------
    # Decision 5: Regional nodal irradiation
    # -----------------------------------------------------------------------

    needs_rni, rni_reason, rni_rule = _check_rni_indication(case)
    rules_triggered.append(rni_rule)

    reference["rni"] = {
        "answer": needs_rni,
        "rule": rni_reason,
    }

    reference["rules_triggered"] = rules_triggered
    return reference


def _check_rni_indication(case: BreastCancerCase) -> tuple[str, str, str]:
    """Return (rni_answer, rule_text, rule_id). Factored out because the
    target volume decision also depends on this."""

    if case.histology == "DCIS":
        return ("no", "R040: DCIS does not warrant RNI", "R040")

    if case.nodes_positive >= 4:
        return ("yes", "R041: ≥4 positive nodes; RNI indicated per NCCN", "R041")

    if 1 <= case.nodes_positive <= 3:
        # Per MA.20, EORTC 22922, and NCCN: strongly consider RNI for 1-3 positive nodes
        # Most centers favor RNI in this group, especially with high-risk features
        return ("strongly consider", "R042: 1-3 positive nodes; RNI strongly considered per MA.20/EORTC 22922 (and NCCN)", "R042")

    # Node-negative
    high_risk_node_neg = (
        case.tumor_size_cm > 5.0
        or (case.tumor_size_cm >= 2.0 and case.lvi_present and case.grade == 3)
    )
    if high_risk_node_neg:
        return ("consider", "R043: node-negative but high-risk features (large tumor, LVI, high grade); RNI may be considered", "R043")

    return ("no", "R044: node-negative without high-risk features; RNI not indicated", "R044")


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_cohort(n: int, seed: int = 42) -> list[BreastCancerCase]:
    """Generate a cohort of n cases with pre-registered seed for reproducibility."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        case_id = f"BC-{seed:04d}-{i+1:04d}"
        case = sample_case(rng, case_id)
        case.reference = apply_guidelines(case)
        cases.append(case)
    return cases


def cohort_summary(cases: list[BreastCancerCase]) -> dict:
    """Summarize the generated cohort for pre-registration reporting."""
    from collections import Counter

    n = len(cases)
    summary = {
        "n_total": n,
        "age_median": sorted(c.age for c in cases)[n // 2],
        "age_range": (min(c.age for c in cases), max(c.age for c in cases)),
        "pct_age_geq_70": round(100 * sum(1 for c in cases if c.age >= 70) / n, 1),
        "histology": dict(Counter(c.histology for c in cases)),
        "surgery": dict(Counter(c.surgery_type for c in cases)),
        "node_positive": round(100 * sum(1 for c in cases if c.nodes_positive > 0) / n, 1),
        "er_positive": round(100 * sum(1 for c in cases if c.er_positive) / n, 1),
        "her2_positive": round(100 * sum(1 for c in cases if c.her2_positive) / n, 1),
        "rt_recommendation_distribution": dict(Counter(
            c.reference["rt_recommendation"]["answer"] for c in cases
        )),
        "target_volume_distribution": dict(Counter(
            c.reference["target_volume"]["answer"] for c in cases
        )),
        "fractionation_distribution": dict(Counter(
            c.reference["fractionation"]["answer"][:40] + "..." if c.reference["fractionation"]["answer"] and len(c.reference["fractionation"]["answer"]) > 40 else c.reference["fractionation"]["answer"]
            for c in cases
        )),
        "boost_distribution": dict(Counter(c.reference["boost"]["answer"] for c in cases)),
        "rni_distribution": dict(Counter(c.reference["rni"]["answer"] for c in cases)),
    }
    return summary


def serialize_case(case: BreastCancerCase) -> dict:
    """Serialize case to JSON-compatible dict including vignette and reference."""
    d = asdict(case)
    d["clinical_vignette"] = case.to_clinical_vignette()
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate breast cancer RT decision cases.")
    parser.add_argument("--n", type=int, default=120, help="Number of cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="cases.jsonl", help="Output JSONL path")
    parser.add_argument("--summary", type=str, default="cohort_summary.json", help="Summary output path")
    args = parser.parse_args()

    cases = generate_cohort(args.n, args.seed)

    with open(args.output, "w") as f:
        for c in cases:
            f.write(json.dumps(serialize_case(c)) + "\n")

    summary = cohort_summary(cases)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(cases)} cases → {args.output}")
    print(f"Cohort summary → {args.summary}")
    print()
    print("=== Cohort Summary ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
