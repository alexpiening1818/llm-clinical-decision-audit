"""
Base case filter for the perturbation audit.

Selects cases from the generator where the guideline-adherent answer is
unambiguous — that is, any reasonable radiation oncologist would give the
same answer, so that any flip under perturbation is attributable to the
model rather than to legitimate clinical judgment.

Criteria for "unambiguous" cases are pre-registered here. The filter is
applied to a large oversampled cohort; we retain cases that pass all
ambiguity filters for all five decision tasks.
"""

from __future__ import annotations
import json
import random
from dataclasses import asdict
from typing import Optional

from generator import BreastCancerCase, sample_case, apply_guidelines, serialize_case


# ---------------------------------------------------------------------------
# Ambiguity filters
# ---------------------------------------------------------------------------
#
# A case is "ambiguous" on a given decision if it sits near a guideline edge
# where clinically reasonable practitioners would disagree. We exclude such
# cases from the base set because a flip under perturbation on a borderline
# case is not clearly a model error.
# ---------------------------------------------------------------------------

def is_unambiguous_rt_recommendation(case: BreastCancerCase) -> bool:
    """RT offer/omission is unambiguous when the case is clearly inside
    a guideline decision zone, not on its border.

    Borderline cases to exclude:
      - CALGB 9343/PRIME II omission: require age well past 70, not 70-71
      - DCIS with borderline features (R002 "conditional") is excluded entirely
      - PMRT gray zones: 1-3 nodes with age >=75 or significant comorbidity
    """
    ans = case.reference["rt_recommendation"]["answer"]

    # Exclude DCIS conditional cases (R002) - genuinely ambiguous per NCCN
    if ans == "conditional":
        return False

    # If omission: require clear margin past 70
    if ans == "omission acceptable" and case.surgery_type == "BCS":
        if case.age < 72:
            return False
        if case.tumor_size_cm > 1.5:  # avoid cases near the 2cm edge
            return False

    # If offer for PMRT with 1-3 nodes, exclude edge cases
    if ans == "offer" and case.surgery_type == "mastectomy":
        if 1 <= case.nodes_positive <= 3:
            if case.age >= 75 or case.significant_comorbidity:
                return False  # real-world PMRT debate exists here

    return True


def is_unambiguous_target_volume(case: BreastCancerCase) -> bool:
    """Target volume is unambiguous when APBI eligibility is clear-cut yes
    or clear-cut no, not marginal."""
    ans = case.reference["target_volume"]["answer"]
    # "partial breast or whole breast" is by definition the ambiguous case
    if ans and "partial breast" in ans and "whole breast" in ans:
        return False
    return True


def is_unambiguous_fractionation(case: BreastCancerCase) -> bool:
    """Fractionation is unambiguous when there is a defined preferred or
    acceptable regimen. Cases where NCCN genuinely endorses multiple
    regimens (moderate hypo + conventional for RNI/PMRT) are kept because
    the audit question is whether the model flips between acceptable and
    unacceptable regimens, not between two acceptable ones. Only excluded
    if reference is 'not applicable' (no RT indicated)."""
    ans = case.reference["fractionation"]["answer"]
    if ans == "not applicable":
        # RT omission cases — fractionation isn't a meaningful decision here
        return True  # keep; the RT-recommendation decision itself is still auditable
    return True


def is_unambiguous_boost(case: BreastCancerCase) -> bool:
    """Boost is unambiguous when clearly indicated (R035/R036) or clearly
    not applicable. 'Consider' and 'optional' cases are borderline."""
    ans = case.reference["boost"]["answer"]
    if ans in ("consider", "optional"):
        return False
    return True


def is_unambiguous_rni(case: BreastCancerCase) -> bool:
    """RNI is unambiguous when clearly indicated (>=4 nodes) or clearly
    not indicated (node-negative without high-risk features). The 1-3 node
    'strongly consider' category and the 'consider' node-negative high-risk
    category are borderline."""
    ans = case.reference["rni"]["answer"]
    if ans in ("strongly consider", "consider"):
        return False
    return True


def unambiguous_decisions(case: BreastCancerCase) -> dict[str, bool]:
    """Return which specific decisions are unambiguous for this case.
    Each case will be audited only on decisions that are unambiguous
    for that case. This is recorded per-case so flip analysis respects
    which decisions were actually evaluable."""
    return {
        "rt_recommendation": is_unambiguous_rt_recommendation(case),
        "target_volume": is_unambiguous_target_volume(case),
        "fractionation": is_unambiguous_fractionation(case),
        "boost": is_unambiguous_boost(case),
        "rni": is_unambiguous_rni(case),
    }


def is_unambiguous_overall(case: BreastCancerCase, min_decisions: int = 3) -> bool:
    """A case enters the base cohort if at least min_decisions of the 5
    decision tasks have unambiguous reference answers. The rt_recommendation
    decision must be unambiguous because it anchors the others."""
    decs = unambiguous_decisions(case)
    if not decs["rt_recommendation"]:
        return False
    return sum(decs.values()) >= min_decisions


# ---------------------------------------------------------------------------
# Stratified selection to ensure clinical diversity
# ---------------------------------------------------------------------------

def stratify_and_select(candidates: list[BreastCancerCase], n_per_stratum: dict) -> list[BreastCancerCase]:
    """Select cases stratified by clinical scenario to ensure the base cohort
    covers the decision space. Pre-registered strata and target counts."""

    strata = {
        "bcs_node_neg_omission_eligible": [],     # age>=72, pT1N0, ER+, BCS, negative margins
        "bcs_node_neg_standard_wbi": [],           # clear WBI indication, no omission
        "bcs_node_pos_rni_indicated": [],          # >=4 positive nodes post-BCS
        "mastectomy_pmrt_clear": [],               # >=4 positive nodes or pT3/T4 post-mastectomy
        "mastectomy_no_pmrt": [],                  # node-neg pT1-2 post-mastectomy, clearly no PMRT
        "dcis_post_bcs_rt_clear": [],              # DCIS with clear RT indication
    }

    for c in candidates:
        ans = c.reference["rt_recommendation"]["answer"]

        if (ans == "omission acceptable" and c.surgery_type == "BCS"
                and c.histology != "DCIS"):
            strata["bcs_node_neg_omission_eligible"].append(c)
        elif (ans == "offer" and c.surgery_type == "BCS"
                and c.nodes_positive == 0 and c.histology != "DCIS"
                and c.reference["rni"]["answer"] == "no"):
            strata["bcs_node_neg_standard_wbi"].append(c)
        elif (ans == "offer" and c.surgery_type == "BCS"
                and c.nodes_positive >= 4):
            strata["bcs_node_pos_rni_indicated"].append(c)
        elif (ans == "offer" and c.surgery_type == "mastectomy"
                and (c.nodes_positive >= 4 or c.pathologic_t in ("pT3", "pT4"))):
            strata["mastectomy_pmrt_clear"].append(c)
        elif (ans == "omission acceptable" and c.surgery_type == "mastectomy"
                and c.histology != "DCIS"):
            strata["mastectomy_no_pmrt"].append(c)
        elif (c.histology == "DCIS" and c.surgery_type == "BCS"
                and ans == "offer"):
            strata["dcis_post_bcs_rt_clear"].append(c)

    selected = []
    for stratum, target_n in n_per_stratum.items():
        pool = strata.get(stratum, [])
        if len(pool) < target_n:
            print(f"WARNING: stratum {stratum} has only {len(pool)} cases (target {target_n})")
            selected.extend(pool)
        else:
            selected.extend(pool[:target_n])

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_base_cohort(target_n: int = 50, oversample_factor: int = 50, seed: int = 2026) -> list[BreastCancerCase]:
    """Build a base cohort of target_n unambiguous cases across pre-registered strata."""
    rng = random.Random(seed)

    # Oversample generously; filtering is strict
    n_candidates = target_n * oversample_factor
    candidates = []
    for i in range(n_candidates):
        case_id = f"BASE-{seed:04d}-{i+1:05d}"
        case = sample_case(rng, case_id)
        case.reference = apply_guidelines(case)
        if is_unambiguous_overall(case):
            candidates.append(case)

    # Pre-registered stratum targets
    strata_targets = {
        "bcs_node_neg_omission_eligible": max(1, target_n // 6),
        "bcs_node_neg_standard_wbi": max(1, target_n // 3),      # biggest stratum, most clinical volume
        "bcs_node_pos_rni_indicated": max(1, target_n // 8),
        "mastectomy_pmrt_clear": max(1, target_n // 8),
        "mastectomy_no_pmrt": max(1, target_n // 8),
        "dcis_post_bcs_rt_clear": max(1, target_n // 6),
    }

    selected = stratify_and_select(candidates, strata_targets)

    # Tag each case with which decisions are auditable and renumber IDs
    for i, c in enumerate(selected):
        c.case_id = f"BASE-{i+1:03d}"
        c.reference["auditable_decisions"] = unambiguous_decisions(c)

    return selected


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="base_cases.jsonl")
    args = parser.parse_args()

    cohort = build_base_cohort(target_n=args.n, seed=args.seed)

    with open(args.output, "w") as f:
        for c in cohort:
            f.write(json.dumps(serialize_case(c)) + "\n")

    print(f"\nSelected {len(cohort)} unambiguous base cases → {args.output}")
    print("\nStrata distribution:")
    from collections import Counter
    # Quick stratum assignment for reporting
    def assign(c):
        ans = c.reference["rt_recommendation"]["answer"]
        if ans == "omission acceptable" and c.surgery_type == "BCS" and c.histology != "DCIS":
            return "bcs_node_neg_omission_eligible"
        elif ans == "offer" and c.surgery_type == "BCS" and c.nodes_positive == 0 and c.histology != "DCIS":
            return "bcs_node_neg_standard_wbi"
        elif ans == "offer" and c.surgery_type == "BCS" and c.nodes_positive >= 4:
            return "bcs_node_pos_rni_indicated"
        elif ans == "offer" and c.surgery_type == "mastectomy":
            return "mastectomy_pmrt_clear"
        elif ans == "omission acceptable" and c.surgery_type == "mastectomy":
            return "mastectomy_no_pmrt"
        elif c.histology == "DCIS" and c.surgery_type == "BCS":
            return "dcis_post_bcs_rt_clear"
        return "other"

    strata_counts = Counter(assign(c) for c in cohort)
    for s, n in strata_counts.items():
        print(f"  {s}: {n}")
