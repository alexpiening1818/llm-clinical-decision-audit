"""
Unified base-case builder for all three domains.

Strategy: each domain defines its own ambiguity rules as a small dictionary
of per-decision filters. The base-case builder takes cases from any domain
generator, keeps only cases with at least one unambiguous decision, tags
which decisions are auditable, and stratifies across clinically meaningful
scenarios for each domain.

This replaces what was a breast-only base_cases.py with a cross-domain
version that supports the expanded pre-registration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

# Allow importing domain generators
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from prostate_generator import (
    ProstateCancerCase,
    sample_case as prostate_sample,
    apply_guidelines as prostate_apply,
    serialize_case as prostate_serialize,
    compute_risk_group,
)
from oncotype_generator import (
    OncotypeCase,
    sample_case as oncotype_sample,
    apply_guidelines as oncotype_apply,
    serialize_case as oncotype_serialize,
)


# ---------------------------------------------------------------------------
# PROSTATE ambiguity rules
# ---------------------------------------------------------------------------

def prostate_unambiguous_decisions(case) -> dict:
    """Return which decisions are unambiguous for a prostate case.
    Borderline cases to exclude per decision:
      - management: exclude edge cases near LE=10 threshold and near
        the favorable/unfavorable intermediate-risk boundary
      - modality: unambiguous when risk group defines it clearly
      - adt: unambiguous when risk group defines ADT indication
      - workup: always unambiguous (recommendation is deterministic)
      - confirmatory_biopsy: unambiguous in AS recommended / treatment
        recommended; borderline in "treatment acceptable" gray zone
    """
    risk_group = case.reference["risk_group"]
    le = case.life_expectancy_years
    mgmt = case.reference["management"]["answer"]

    # Management: exclude cases near life-expectancy threshold
    management_ok = True
    if 9 <= le <= 11:
        management_ok = False
    # Also exclude "favorable_intermediate" cases where management is a
    # genuine three-way choice - the model's answer is a preference not a flip
    if mgmt == "treatment acceptable":
        management_ok = False

    # Modality: only unambiguous when a single modality is recommended
    modality_ans = case.reference["modality"]["answer"]
    modality_ok = (modality_ans == "not applicable")  # truly unambiguous only when no Rx

    # ADT: unambiguous for "not applicable", "not indicated", "short-course",
    # "long-course" — but the "may be considered if EBRT is chosen" is borderline
    adt_ans = case.reference["adt"]["answer"]
    adt_ok = not adt_ans.startswith("not indicated for monotherapy")

    # Workup: deterministic per risk group
    workup_ok = True

    # Confirmatory biopsy: clear when AS or when treatment recommended
    cbx_ans = case.reference["confirmatory_biopsy"]["answer"]
    cbx_ok = cbx_ans in ("yes", "not applicable")

    return {
        "management": management_ok,
        "modality": modality_ok,
        "adt": adt_ok,
        "workup": workup_ok,
        "confirmatory_biopsy": cbx_ok,
    }


# ---------------------------------------------------------------------------
# ONCOTYPE ambiguity rules
# ---------------------------------------------------------------------------

def oncotype_unambiguous_decisions(case) -> dict:
    """Return which decisions are unambiguous for an oncotype case.
    Borderline cases to exclude per decision:
      - chemotherapy: exclude TAILORx gray zone (RS 21-25 in women ≤50)
        as these are genuinely split clinical decisions; keep RS 16-20
        young cases as "consider" but don't flip-audit them
      - endocrine: always unambiguous (ER+ → yes)
      - ofs: borderline in premenopausal node-neg low-grade cases
      - extended_et: borderline cases exist but reference is deterministic
      - cdk46: deterministic per monarchE criteria
    """
    rs = case.oncotype_score
    age = case.age
    nodes = case.nodes_positive
    chemo_ans = case.reference["chemotherapy"]["answer"]

    # Chemo: exclude "consider" (the gray zone) as inherently borderline
    # Exclude exact-threshold cases too (RS 25 vs 26)
    chemo_ok = chemo_ans != "consider" and rs not in (25, 26)

    # Endocrine: always unambiguous for ER+ inclusion
    endocrine_ok = True

    # OFS: if premenopausal, exclude borderline age 40-45 with exactly
    # one borderline risk feature
    ofs_ok = True
    if case.menopausal_status == "premenopausal":
        # Borderline: age 40-44, node-neg, grade 2, tumor 1.8-2.2 cm
        if (40 <= age <= 44 and nodes == 0 and case.grade == 2
                and 1.7 <= case.tumor_size_cm <= 2.3):
            ofs_ok = False

    # Extended ET: deterministic per rules
    extended_et_ok = True

    # CDK4/6i: exclude cases at exactly the boundary (1-3 nodes with
    # tumor exactly 5.0cm or exactly grade 3 but small)
    cdk46_ok = True
    if 1 <= nodes <= 3 and 4.8 <= case.tumor_size_cm <= 5.2:
        cdk46_ok = False

    return {
        "chemotherapy": chemo_ok,
        "endocrine_therapy": endocrine_ok,
        "ofs": ofs_ok,
        "extended_endocrine_therapy": extended_et_ok,
        "cdk46_inhibitor": cdk46_ok,
    }


# ---------------------------------------------------------------------------
# PROSTATE stratification
# ---------------------------------------------------------------------------

def prostate_stratify(case) -> str:
    rg = case.reference["risk_group"]
    mgmt = case.reference["management"]["answer"]
    if rg == "very_low" and mgmt == "active surveillance":
        return "very_low_AS"
    if rg == "low" and mgmt == "active surveillance":
        return "low_AS"
    if rg == "favorable_intermediate":
        return "favorable_intermediate"
    if rg == "unfavorable_intermediate":
        return "unfavorable_intermediate"
    if rg == "high":
        return "high_risk"
    if rg == "very_high":
        return "very_high_risk"
    if mgmt == "observation":
        return "limited_life_expectancy"
    return "other"


# ---------------------------------------------------------------------------
# ONCOTYPE stratification
# ---------------------------------------------------------------------------

def oncotype_stratify(case) -> str:
    rs = case.oncotype_score
    nodes = case.nodes_positive
    postmenopausal = case.menopausal_status == "postmenopausal"

    if nodes == 0:
        if rs <= 15:
            return "node_neg_low_rs"
        elif 16 <= rs <= 25 and case.age <= 50:
            return "node_neg_gray_zone_young"
        elif 16 <= rs <= 25 and case.age > 50:
            return "node_neg_gray_zone_older"
        else:
            return "node_neg_high_rs"
    else:
        if rs <= 25 and postmenopausal:
            return "node_pos_postmeno_low_rs"
        elif rs <= 25 and not postmenopausal:
            return "node_pos_premeno_low_rs"
        else:
            return "node_pos_high_rs"


# ---------------------------------------------------------------------------
# Generic cohort builder
# ---------------------------------------------------------------------------

def build_cohort(domain: str, target_n: int, oversample: int, seed: int):
    """Generic base cohort builder for a given domain."""
    rng = random.Random(seed)

    if domain == "prostate":
        sample_fn = prostate_sample
        apply_fn = prostate_apply
        unambig_fn = prostate_unambiguous_decisions
        stratify_fn = prostate_stratify
        serialize_fn = prostate_serialize
        prefix = "PROSTATE-BASE"
        primary_decision = "management"
    elif domain == "oncotype":
        sample_fn = oncotype_sample
        apply_fn = oncotype_apply
        unambig_fn = oncotype_unambiguous_decisions
        stratify_fn = oncotype_stratify
        serialize_fn = oncotype_serialize
        prefix = "ONCOTYPE-BASE"
        primary_decision = "chemotherapy"
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Oversample
    candidates = []
    n_candidates = target_n * oversample
    for i in range(n_candidates):
        case = sample_fn(rng, f"gen-{i}")
        case.reference = apply_fn(case)
        decisions = unambig_fn(case)
        # Primary decision must be unambiguous; at least 3 of 5 overall
        if decisions[primary_decision] and sum(decisions.values()) >= 3:
            case.reference["auditable_decisions"] = decisions
            candidates.append(case)

    # Stratify
    strata = {}
    for c in candidates:
        s = stratify_fn(c)
        strata.setdefault(s, []).append(c)

    # Sample equally from each stratum up to target
    n_strata = len(strata)
    per_stratum = max(1, target_n // n_strata)
    selected = []
    for s, pool in strata.items():
        selected.extend(pool[:per_stratum])
        if len(selected) >= target_n:
            break

    # Renumber
    for i, c in enumerate(selected):
        c.case_id = f"{prefix}-{i+1:03d}"

    return selected, strata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["prostate", "oncotype"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--oversample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected, strata = build_cohort(args.domain, args.n, args.oversample, args.seed)

    # Write jsonl
    if args.domain == "prostate":
        serialize_fn = prostate_serialize
    else:
        serialize_fn = oncotype_serialize

    with open(args.output, "w") as f:
        for c in selected:
            f.write(json.dumps(serialize_fn(c)) + "\n")

    print(f"Selected {len(selected)} unambiguous {args.domain} base cases → {args.output}")
    print("\nCandidate strata counts (all unambiguous candidates):")
    for s, pool in sorted(strata.items()):
        print(f"  {s}: {len(pool)}")


if __name__ == "__main__":
    main()
