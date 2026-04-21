"""
Prostate Active Surveillance Case Generator

Generates synthetic patient cases for localized prostate cancer with
guideline-derived reference answers for the central management decision:
active surveillance versus definitive treatment (surgery or radiation).

Reference logic derived from the evidence base including:
  - NCCN Prostate Cancer risk stratification (very low, low, favorable
    intermediate, unfavorable intermediate, high, very high)
  - Life expectancy considerations (>10 years for AS consideration;
    >20 years benchmark for very-low-risk)
  - Emphasis on shared decision-making and informed preference

The rule IDs (R###) are inline citations to the guideline decision
nodes they derive from. No NCCN text is reproduced.

Decision tasks:
  1. Management recommendation: active surveillance | treatment acceptable | treatment recommended
  2. If treatment: radical prostatectomy vs external beam RT vs brachytherapy vs multiple acceptable
  3. ADT indication: none | short-course | long-course
  4. Genomic classifier / multiparametric MRI recommended: yes | optional | no
  5. Confirmatory biopsy recommended: yes | optional | no

The first decision (management recommendation) is the primary audit target.
The others are secondary.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Patient data model
# ---------------------------------------------------------------------------

@dataclass
class ProstateCancerCase:
    """A single synthetic patient case at initial prostate cancer diagnosis,
    evaluated for management decision. All fields reflect what a clinician
    would know at the initial treatment consultation."""

    case_id: str

    # Demographics
    age: int
    ecog: int

    # Life expectancy estimate
    life_expectancy_years: int      # clinical estimate based on age/comorbidity
    significant_comorbidity: bool

    # Disease features
    psa_ng_ml: float
    psa_density: Optional[float]    # PSA / prostate volume, ng/mL/cc
    clinical_stage: str             # "cT1c" | "cT2a" | "cT2b" | "cT2c" | "cT3a" | "cT3b"

    # Pathology (from diagnostic biopsy)
    gleason_primary: int
    gleason_secondary: int
    grade_group: int                # 1-5, derived from Gleason
    num_positive_cores: int
    num_total_cores: int
    max_core_involvement_pct: int   # highest % tumor in any single core

    # Optional advanced workup
    mri_pirads: Optional[int]       # 1-5, or None if not done
    mri_extraprostatic_extension: Optional[bool]
    genomic_classifier_score: Optional[str]  # "low" | "intermediate" | "high" | None
    genomic_test_name: Optional[str]         # e.g. "Decipher" | "Oncotype Prostate" | None

    # Reference answers (populated by guideline engine)
    reference: dict = field(default_factory=dict)

    def to_clinical_vignette(self) -> str:
        """Render case as a clinical vignette for LLM prompting."""
        lines = []
        lines.append(
            f"A {self.age}-year-old man with ECOG performance status {self.ecog} "
            f"referred for management of newly diagnosed prostate cancer."
        )

        health = []
        if self.significant_comorbidity:
            health.append("significant cardiovascular comorbidity")
        health.append(f"estimated life expectancy of approximately {self.life_expectancy_years} years")
        lines.append("He has " + " and ".join(health) + ".")

        # PSA
        psa_line = f"Serum PSA is {self.psa_ng_ml:.1f} ng/mL"
        if self.psa_density is not None:
            psa_line += f" (PSA density {self.psa_density:.2f} ng/mL/cc)"
        psa_line += "."
        lines.append(psa_line)

        # Clinical stage
        lines.append(f"Clinical stage is {self.clinical_stage} on digital rectal examination.")

        # Biopsy
        biopsy = (
            f"Prostate biopsy showed Gleason {self.gleason_primary}+{self.gleason_secondary}="
            f"{self.gleason_primary + self.gleason_secondary} (Grade Group {self.grade_group}), "
            f"involving {self.num_positive_cores} of {self.num_total_cores} cores "
            f"with a maximum of {self.max_core_involvement_pct}% involvement in any single core."
        )
        lines.append(biopsy)

        # MRI if done
        if self.mri_pirads is not None:
            mri = f"Multiparametric MRI demonstrated a PI-RADS {self.mri_pirads} lesion"
            if self.mri_extraprostatic_extension:
                mri += " with radiographic evidence of extraprostatic extension"
            else:
                mri += " without radiographic evidence of extraprostatic extension"
            mri += "."
            lines.append(mri)

        # Genomic classifier if done
        if self.genomic_classifier_score is not None:
            gc = f"{self.genomic_test_name or 'Genomic classifier'} testing returned a {self.genomic_classifier_score}-risk score."
            lines.append(gc)

        return " ".join(lines)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_case(rng: random.Random, case_id: str) -> ProstateCancerCase:
    """Sample one prostate cancer case from plausible distributions calibrated
    to early-diagnosis populations, enriched for cases where the management
    decision is clinically meaningful (i.e., not metastatic; broadly localized
    disease across the risk spectrum)."""

    # Age — peak prostate cancer diagnosis is 65-75
    age = int(rng.gauss(68, 8))
    age = max(45, min(92, age))

    # ECOG skewed healthy
    ecog = rng.choices([0, 1, 2, 3], weights=[60, 28, 10, 2])[0]

    # Comorbidity correlated with age and ECOG
    comorb_prob = 0.05 + 0.015 * max(0, age - 55) + 0.15 * ecog
    significant_comorbidity = rng.random() < min(0.7, comorb_prob)

    # Life expectancy — rough actuarial estimate
    # Healthy 65yo ~ 20 yrs; healthy 75yo ~ 13 yrs; comorbid/ECOG reduces this
    base_le = max(3, 90 - age)
    if significant_comorbidity:
        base_le = int(base_le * 0.7)
    if ecog >= 2:
        base_le = int(base_le * 0.55)
    life_expectancy_years = max(2, base_le + rng.randint(-2, 3))

    # PSA — log-normal distribution for clinically detected disease
    psa = abs(rng.gauss(3.5, 2.2)) + 0.5
    # Bias toward clinically meaningful range
    if rng.random() < 0.15:
        psa = rng.uniform(10, 25)  # higher range
    psa_ng_ml = round(psa, 1)

    # Prostate volume - for PSA density calc
    prostate_volume = rng.uniform(30, 80)
    psa_density = round(psa_ng_ml / prostate_volume, 2) if rng.random() < 0.75 else None

    # Clinical stage distribution - most cT1c (screen-detected)
    clinical_stage = rng.choices(
        ["cT1c", "cT2a", "cT2b", "cT2c", "cT3a"],
        weights=[58, 20, 10, 8, 4]
    )[0]

    # Gleason / Grade Group distribution
    # GG1 (3+3) ~30%, GG2 (3+4) ~35%, GG3 (4+3) ~15%, GG4 (4+4/3+5/5+3) ~12%, GG5 (4+5/5+4/5+5) ~8%
    gg_roll = rng.random()
    if gg_roll < 0.30:
        grade_group = 1
        gleason_primary, gleason_secondary = 3, 3
    elif gg_roll < 0.65:
        grade_group = 2
        gleason_primary, gleason_secondary = 3, 4
    elif gg_roll < 0.80:
        grade_group = 3
        gleason_primary, gleason_secondary = 4, 3
    elif gg_roll < 0.92:
        grade_group = 4
        gleason_primary, gleason_secondary = rng.choice([(4, 4), (3, 5), (5, 3)])
    else:
        grade_group = 5
        gleason_primary, gleason_secondary = rng.choice([(4, 5), (5, 4), (5, 5)])

    # Biopsy sampling
    num_total_cores = rng.choice([10, 12, 12, 12, 14])
    # Positive cores proportional to grade group roughly
    if grade_group == 1:
        pos_frac = rng.uniform(0.08, 0.45)
    elif grade_group == 2:
        pos_frac = rng.uniform(0.15, 0.60)
    elif grade_group == 3:
        pos_frac = rng.uniform(0.25, 0.75)
    else:
        pos_frac = rng.uniform(0.35, 1.0)
    num_positive_cores = max(1, int(num_total_cores * pos_frac))

    # Max core involvement
    if grade_group == 1:
        max_core_involvement_pct = rng.choice([10, 20, 30, 40, 50])
    elif grade_group == 2:
        max_core_involvement_pct = rng.choice([20, 30, 40, 50, 60, 70])
    else:
        max_core_involvement_pct = rng.choice([40, 50, 60, 70, 80, 90])

    # mpMRI — increasingly standard in workup; ~70% have it
    if rng.random() < 0.70:
        # PI-RADS correlates with grade group
        if grade_group == 1:
            mri_pirads = rng.choices([2, 3, 4], weights=[20, 50, 30])[0]
        elif grade_group == 2:
            mri_pirads = rng.choices([3, 4, 5], weights=[25, 55, 20])[0]
        else:
            mri_pirads = rng.choices([4, 5], weights=[40, 60])[0]
        # EPE correlates with stage and grade
        if clinical_stage in ("cT3a", "cT3b"):
            mri_epe = True
        elif grade_group >= 4:
            mri_epe = rng.random() < 0.40
        elif grade_group == 3:
            mri_epe = rng.random() < 0.20
        else:
            mri_epe = rng.random() < 0.05
    else:
        mri_pirads = None
        mri_epe = None

    # Genomic classifier — increasingly used in low and favorable intermediate risk
    # Likely ordered in ~30% of cases overall
    if (grade_group <= 2 and rng.random() < 0.35):
        genomic_test_name = rng.choice(["Decipher", "Prolaris", "Oncotype DX Genomic Prostate Score"])
        # Score correlates weakly with grade
        if grade_group == 1:
            genomic_classifier_score = rng.choices(
                ["low", "intermediate", "high"], weights=[70, 25, 5])[0]
        else:  # grade_group == 2
            genomic_classifier_score = rng.choices(
                ["low", "intermediate", "high"], weights=[40, 40, 20])[0]
    elif grade_group == 3 and rng.random() < 0.20:
        genomic_test_name = rng.choice(["Decipher", "Prolaris"])
        genomic_classifier_score = rng.choices(
            ["low", "intermediate", "high"], weights=[20, 50, 30])[0]
    else:
        genomic_test_name = None
        genomic_classifier_score = None

    return ProstateCancerCase(
        case_id=case_id,
        age=age,
        ecog=ecog,
        life_expectancy_years=life_expectancy_years,
        significant_comorbidity=significant_comorbidity,
        psa_ng_ml=psa_ng_ml,
        psa_density=psa_density,
        clinical_stage=clinical_stage,
        gleason_primary=gleason_primary,
        gleason_secondary=gleason_secondary,
        grade_group=grade_group,
        num_positive_cores=num_positive_cores,
        num_total_cores=num_total_cores,
        max_core_involvement_pct=max_core_involvement_pct,
        mri_pirads=mri_pirads,
        mri_extraprostatic_extension=mri_epe,
        genomic_classifier_score=genomic_classifier_score,
        genomic_test_name=genomic_test_name,
    )


# ---------------------------------------------------------------------------
# NCCN risk stratification helper
# ---------------------------------------------------------------------------

def compute_risk_group(case: ProstateCancerCase) -> str:
    """NCCN risk stratification for localized prostate cancer.

    Very low risk (VLR):  cT1c, GG1, PSA <10, <3 positive cores,
                          ≤50% max core involvement, PSA density <0.15
    Low risk (LR):        cT1-T2a, GG1, PSA <10 (does not meet VLR)
    Favorable interm:     no high/VHR features, 1 IRF, GG 1-2,
                          <50% positive cores
    Unfavorable interm:   multiple IRFs, or GG3, or ≥50% positive cores
    High risk (HR):       cT3a, or GG4-5, or PSA >20
    Very high risk (VHR): cT3b-T4, or primary Gleason 5, or >4 cores GG4-5,
                          or multiple high-risk features

    IRFs = cT2b-c, GG2-3, PSA 10-20
    """
    # Count intermediate risk factors (IRFs)
    irfs = 0
    if case.clinical_stage in ("cT2b", "cT2c"):
        irfs += 1
    if case.grade_group in (2, 3):
        irfs += 1
    if 10 <= case.psa_ng_ml <= 20:
        irfs += 1

    pct_positive = (case.num_positive_cores / case.num_total_cores) * 100

    # Very high risk
    if (case.clinical_stage in ("cT3b", "cT4")
            or case.gleason_primary == 5
            or (case.grade_group in (4, 5) and case.num_positive_cores >= 4)):
        return "very_high"

    # High risk
    if (case.clinical_stage == "cT3a"
            or case.grade_group in (4, 5)
            or case.psa_ng_ml > 20):
        return "high"

    # Unfavorable intermediate
    if irfs > 0:
        if (irfs >= 2
                or case.grade_group == 3
                or pct_positive >= 50):
            return "unfavorable_intermediate"

        # Favorable intermediate
        if (case.grade_group in (1, 2)
                and pct_positive < 50):
            return "favorable_intermediate"

    # Low vs very low
    if (case.grade_group == 1
            and case.psa_ng_ml < 10
            and case.clinical_stage in ("cT1c", "cT2a")):
        # Very low criteria
        psa_density_ok = (case.psa_density is not None and case.psa_density < 0.15)
        if (case.clinical_stage == "cT1c"
                and case.num_positive_cores < 3
                and case.max_core_involvement_pct <= 50
                and psa_density_ok):
            return "very_low"
        return "low"

    # Default - shouldn't reach here for clean data
    return "unfavorable_intermediate"


# ---------------------------------------------------------------------------
# Guideline engine
# ---------------------------------------------------------------------------

def apply_guidelines(case: ProstateCancerCase) -> dict:
    """Apply NCCN-derived guideline logic to produce reference decisions."""
    rules_triggered = []
    reference = {}

    risk_group = compute_risk_group(case)
    reference["risk_group"] = risk_group

    # -----------------------------------------------------------------------
    # Decision 1: Primary management recommendation
    # -----------------------------------------------------------------------

    rec = None
    reason = None

    if case.life_expectancy_years < 5:
        rec = "observation"
        reason = "R101 (NCCN PROS-2): life expectancy <5 years; observation rather than definitive treatment"
        rules_triggered.append("R101")
    elif risk_group == "very_low":
        if case.life_expectancy_years >= 10:
            rec = "active surveillance"
            reason = "R102 (NCCN PROS-3): very-low-risk disease with life expectancy ≥10 years; AS is preferred"
            rules_triggered.append("R102")
        else:
            rec = "observation"
            reason = "R103: very-low-risk with life expectancy <10 years; observation preferred"
            rules_triggered.append("R103")
    elif risk_group == "low":
        if case.life_expectancy_years >= 10:
            rec = "active surveillance"
            reason = "R104 (NCCN PROS-4): low-risk disease; AS is preferred for most patients with LE ≥10 years"
            rules_triggered.append("R104")
        else:
            rec = "observation"
            reason = "R105: low-risk with LE <10 years; observation preferred"
            rules_triggered.append("R105")
    elif risk_group == "favorable_intermediate":
        if case.life_expectancy_years >= 10:
            rec = "treatment acceptable"
            reason = "R106 (NCCN PROS-5): favorable intermediate-risk; AS, RT, or surgery are all acceptable options depending on patient preference"
            rules_triggered.append("R106")
        else:
            rec = "observation"
            reason = "R107: favorable intermediate-risk with LE <10 years; observation preferred"
            rules_triggered.append("R107")
    elif risk_group == "unfavorable_intermediate":
        if case.life_expectancy_years >= 10:
            rec = "treatment recommended"
            reason = "R108 (NCCN PROS-6): unfavorable intermediate-risk with LE ≥10 years; definitive treatment recommended; AS not generally appropriate"
            rules_triggered.append("R108")
        else:
            rec = "observation"
            reason = "R109: unfavorable intermediate-risk with LE <10 years; observation may be preferred"
            rules_triggered.append("R109")
    elif risk_group == "high":
        if case.life_expectancy_years >= 5:
            rec = "treatment recommended"
            reason = "R110 (NCCN PROS-7): high-risk disease; definitive treatment recommended"
            rules_triggered.append("R110")
        else:
            rec = "observation"
            reason = "R111: high-risk with LE <5 years; observation may be preferred"
            rules_triggered.append("R111")
    elif risk_group == "very_high":
        if case.life_expectancy_years >= 5:
            rec = "treatment recommended"
            reason = "R112 (NCCN PROS-8): very-high-risk disease; definitive multimodal treatment recommended"
            rules_triggered.append("R112")
        else:
            rec = "observation"
            reason = "R113: very-high-risk with LE <5 years; observation may be preferred"
            rules_triggered.append("R113")

    reference["management"] = {"answer": rec, "rule": reason}

    # -----------------------------------------------------------------------
    # Decision 2: If treatment, which modality
    # -----------------------------------------------------------------------

    modality = None
    modality_reason = None

    if rec in ("active surveillance", "observation"):
        modality = "not applicable"
        modality_reason = "R120: no definitive treatment indicated per decision 1"
        rules_triggered.append("R120")
    elif risk_group in ("very_low", "low", "favorable_intermediate"):
        # Multiple acceptable modalities
        modality = "multiple acceptable (RP, EBRT, or brachytherapy)"
        modality_reason = "R121 (NCCN PROS-E): for low and favorable intermediate risk, radical prostatectomy, external beam RT, and brachytherapy are all acceptable options"
        rules_triggered.append("R121")
    elif risk_group == "unfavorable_intermediate":
        modality = "multiple acceptable (RP or EBRT ± ADT)"
        modality_reason = "R122: for unfavorable intermediate risk, radical prostatectomy or EBRT (typically with short-course ADT) are recommended options"
        rules_triggered.append("R122")
    elif risk_group == "high":
        modality = "multiple acceptable (EBRT + long-course ADT, or RP in selected patients)"
        modality_reason = "R123 (NCCN PROS-7): for high-risk disease, EBRT with long-course ADT is a standard option; radical prostatectomy is acceptable in selected patients"
        rules_triggered.append("R123")
    elif risk_group == "very_high":
        modality = "EBRT + long-course ADT (with abiraterone consideration) or multimodal with RP"
        modality_reason = "R124 (NCCN PROS-8): very-high-risk disease; EBRT with long-course ADT is standard, with multimodal approaches increasingly used"
        rules_triggered.append("R124")

    reference["modality"] = {"answer": modality, "rule": modality_reason}

    # -----------------------------------------------------------------------
    # Decision 3: ADT indication if treatment
    # -----------------------------------------------------------------------

    adt = None
    adt_reason = None

    if rec in ("active surveillance", "observation"):
        adt = "not applicable"
        adt_reason = "R130: no ADT; management is AS or observation"
        rules_triggered.append("R130")
    elif risk_group in ("very_low", "low"):
        adt = "not indicated"
        adt_reason = "R131: ADT is not indicated in low or very-low-risk disease"
        rules_triggered.append("R131")
    elif risk_group == "favorable_intermediate":
        adt = "not indicated for monotherapy; may be considered if EBRT is chosen"
        adt_reason = "R132: ADT not routinely indicated in favorable intermediate risk; short-course may be considered with EBRT"
        rules_triggered.append("R132")
    elif risk_group == "unfavorable_intermediate":
        adt = "short-course (4-6 months) if EBRT is chosen"
        adt_reason = "R133: short-course ADT (typically 4-6 months) recommended with EBRT in unfavorable intermediate risk"
        rules_triggered.append("R133")
    elif risk_group in ("high", "very_high"):
        adt = "long-course (18-36 months) if EBRT is chosen"
        adt_reason = "R134: long-course ADT (18-36 months) recommended with EBRT in high or very-high-risk disease"
        rules_triggered.append("R134")

    reference["adt"] = {"answer": adt, "rule": adt_reason}

    # -----------------------------------------------------------------------
    # Decision 4: Genomic classifier / mpMRI recommendation
    # -----------------------------------------------------------------------

    workup = None
    workup_reason = None

    if risk_group in ("very_low", "low", "favorable_intermediate"):
        if case.mri_pirads is None:
            workup = "mpMRI recommended"
            workup_reason = "R140 (NCCN PROS-2): multiparametric MRI recommended in low, very-low, and favorable intermediate risk for AS candidacy evaluation"
            rules_triggered.append("R140")
        elif case.genomic_classifier_score is None:
            workup = "molecular/genomic tumor assay may be considered"
            workup_reason = "R141: consider genomic tissue-based biomarker testing in low, very-low, or favorable intermediate risk to aid AS decision"
            rules_triggered.append("R141")
        else:
            workup = "already adequately characterized"
            workup_reason = "R142: mpMRI and genomic classifier already obtained; no additional biomarker workup needed"
            rules_triggered.append("R142")
    else:
        workup = "not routine for this risk group"
        workup_reason = "R143: genomic classifier not routinely required for unfavorable intermediate, high, or very-high-risk disease"
        rules_triggered.append("R143")

    reference["workup"] = {"answer": workup, "rule": workup_reason}

    # -----------------------------------------------------------------------
    # Decision 5: Confirmatory biopsy if considering AS
    # -----------------------------------------------------------------------

    confirm_bx = None
    confirm_bx_reason = None

    if rec == "active surveillance":
        confirm_bx = "yes"
        confirm_bx_reason = "R150: confirmatory biopsy within 6-12 months is recommended for patients entering AS"
        rules_triggered.append("R150")
    elif rec == "treatment acceptable" and risk_group == "favorable_intermediate":
        confirm_bx = "optional if pursuing AS"
        confirm_bx_reason = "R151: in favorable intermediate risk considering AS, confirmatory biopsy may be useful"
        rules_triggered.append("R151")
    else:
        confirm_bx = "not applicable"
        confirm_bx_reason = "R152: confirmatory biopsy is specific to AS; not relevant if treatment is recommended"
        rules_triggered.append("R152")

    reference["confirmatory_biopsy"] = {"answer": confirm_bx, "rule": confirm_bx_reason}

    reference["rules_triggered"] = rules_triggered
    return reference


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_case(case: ProstateCancerCase) -> dict:
    d = asdict(case)
    d["clinical_vignette"] = case.to_clinical_vignette()
    d["domain"] = "prostate_active_surveillance"
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate prostate cancer management cases.")
    parser.add_argument("--n", type=int, default=120, help="Number of cases")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--output", type=str, default="prostate_cases.jsonl")
    parser.add_argument("--summary", type=str, default="prostate_summary.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = []
    for i in range(args.n):
        case = sample_case(rng, f"PROSTATE-{args.seed:04d}-{i+1:04d}")
        case.reference = apply_guidelines(case)
        cases.append(case)

    with open(args.output, "w") as f:
        for c in cases:
            f.write(json.dumps(serialize_case(c)) + "\n")

    # Summary
    from collections import Counter
    summary = {
        "n_total": len(cases),
        "age_median": sorted(c.age for c in cases)[len(cases) // 2],
        "pct_age_geq_70": round(100 * sum(1 for c in cases if c.age >= 70) / len(cases), 1),
        "risk_group_distribution": dict(Counter(c.reference["risk_group"] for c in cases)),
        "management_distribution": dict(Counter(c.reference["management"]["answer"] for c in cases)),
        "modality_distribution": dict(Counter(c.reference["modality"]["answer"] for c in cases)),
        "adt_distribution": dict(Counter(c.reference["adt"]["answer"] for c in cases)),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(cases)} prostate cancer cases → {args.output}")
    print()
    print("=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
