"""
Oncotype-Guided Adjuvant Chemotherapy Case Generator

Generates synthetic patient cases for early-stage, hormone receptor-positive,
HER2-negative breast cancer in which the adjuvant chemotherapy decision is
informed by the 21-gene recurrence score (Oncotype DX).

Reference logic derived from the evidence base including:
  - TAILORx (NEJM 2018): RS ≤25 cutoff for chemotherapy omission in
    node-negative disease, with age-stratified exception for women
    ≤50 years with RS 16-25
  - RxPONDER (NEJM 2021): chemotherapy omission acceptable in
    postmenopausal women with 1-3 positive nodes and RS ≤25
  - NCCN Breast Cancer current version: Oncotype as category 1 test for
    guiding chemotherapy in eligible populations

Decision tasks:
  1. Adjuvant chemotherapy recommendation: recommend | consider | not indicated
  2. Endocrine therapy recommendation: yes | not applicable
  3. Ovarian suppression / OFS recommendation (premenopausal): yes | no | not applicable
  4. Extended endocrine therapy (beyond 5 years) consideration: yes | no | not applicable
  5. CDK4/6 inhibitor (abemaciclib) eligibility: yes | no | not applicable

Rule IDs (R###) cite guideline decision nodes and underlying trials.
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
class OncotypeCase:
    """A patient with HR+/HER2- early breast cancer at the point of the
    adjuvant therapy decision, with Oncotype DX recurrence score result."""

    case_id: str

    # Demographics
    age: int
    menopausal_status: str              # "premenopausal" | "postmenopausal"
    ecog: int

    # Tumor / pathology
    tumor_size_cm: float
    grade: int                          # 1-3
    lvi_present: bool
    histology: str                      # "IDC" | "ILC"

    # Receptors
    er_positive: bool                   # inclusion requires True
    pr_positive: bool
    her2_positive: bool                 # inclusion requires False

    # Surgical / nodal
    surgery_type: str                   # "BCS" | "mastectomy"
    nodes_positive: int                 # must be 0-3 for Oncotype indication
    nodes_examined: int

    # Oncotype DX
    oncotype_score: int                 # 0-100

    # Staging
    pathologic_t: str
    pathologic_n: str

    # Planned endocrine therapy adherence / receptor positivity
    planned_endocrine_therapy: bool

    # Reference answers
    reference: dict = field(default_factory=dict)

    def to_clinical_vignette(self) -> str:
        lines = []
        lines.append(
            f"A {self.age}-year-old {self.menopausal_status} woman with ECOG {self.ecog} "
            f"who was recently diagnosed with breast cancer and is being evaluated "
            f"for adjuvant systemic therapy."
        )

        tumor = (
            f"Pathology showed {self.histology} of the breast, grade {self.grade}, "
            f"{self.tumor_size_cm:.1f} cm, "
            f"{'with' if self.lvi_present else 'without'} lymphovascular invasion."
        )
        lines.append(tumor)

        receptors = (
            f"Tumor is ER-{'positive' if self.er_positive else 'negative'}, "
            f"PR-{'positive' if self.pr_positive else 'negative'}, "
            f"HER2-{'positive' if self.her2_positive else 'negative'}."
        )
        lines.append(receptors)

        surgery = (
            f"She underwent {'breast-conserving surgery' if self.surgery_type == 'BCS' else 'mastectomy'}."
        )
        lines.append(surgery)

        if self.nodes_positive == 0:
            nodes = f"Axillary evaluation examined {self.nodes_examined} sentinel nodes, all negative."
        else:
            nodes = (
                f"Axillary evaluation examined {self.nodes_examined} nodes "
                f"with {self.nodes_positive} positive."
            )
        lines.append(nodes)

        lines.append(
            f"Staging: {self.pathologic_t}{self.pathologic_n}. "
            f"Oncotype DX 21-gene recurrence score returned a value of {self.oncotype_score}."
        )

        if self.planned_endocrine_therapy:
            lines.append("She is planning to receive adjuvant endocrine therapy.")

        return " ".join(lines)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_case(rng: random.Random, case_id: str) -> OncotypeCase:
    """Sample a patient eligible for the Oncotype DX chemotherapy decision:
    HR+, HER2-, node-negative or 1-3 positive nodes, no distant metastases.
    Sampling enriches the age-stratified gray zone (RS 16-25 in women ≤50)
    and the clean-answer regions so the base cohort covers the decision space."""

    # Age — broad distribution spanning pre and post-menopause
    if rng.random() < 0.40:
        # Younger subset to ensure adequate gray-zone representation
        age = int(rng.gauss(45, 7))
    else:
        age = int(rng.gauss(62, 10))
    age = max(30, min(80, age))

    if age < 50:
        menopausal_status = "premenopausal"
    elif age > 55:
        menopausal_status = "postmenopausal"
    else:
        menopausal_status = rng.choice(["premenopausal", "postmenopausal"])

    ecog = rng.choices([0, 1, 2], weights=[70, 25, 5])[0]

    # Histology
    histology = rng.choices(["IDC", "ILC"], weights=[85, 15])[0]

    # Receptors — by design, all ER+ HER2-; most also PR+
    er_positive = True
    her2_positive = False
    pr_positive = rng.random() < 0.85

    # Tumor size — biased toward T1b-T2 since Oncotype is primarily used there
    size = abs(rng.gauss(1.7, 0.9))
    size = max(0.3, min(size, 5.0))

    # Grade distribution
    grade = rng.choices([1, 2, 3], weights=[30, 55, 15])[0]

    # LVI
    lvi_present = rng.random() < (0.15 if grade <= 2 else 0.30)

    # Surgery — mostly BCS for early stage
    if size > 4.0:
        surgery_type = "mastectomy"
    else:
        surgery_type = rng.choices(["BCS", "mastectomy"], weights=[80, 20])[0]

    # Nodes — Oncotype decision in 0-3 positive. Bias toward node-negative.
    nodes_examined = rng.choice([2, 3, 4, 12, 15])
    if rng.random() < 0.70:
        nodes_positive = 0
    else:
        nodes_positive = rng.choices([1, 2, 3], weights=[60, 25, 15])[0]
        # Ensure enough nodes examined to have positives
        nodes_examined = max(nodes_examined, nodes_positive + 2)

    # Staging
    if size <= 2.0:
        pathologic_t = "pT1"
    elif size <= 5.0:
        pathologic_t = "pT2"
    else:
        pathologic_t = "pT3"

    if nodes_positive == 0:
        pathologic_n = "pN0"
    elif nodes_positive <= 3:
        pathologic_n = "pN1"
    else:
        pathologic_n = "pN2"

    # Oncotype score — realistic distribution with emphasis on decision-relevant ranges
    # Per TAILORx ~ 17% RS ≥ 26, ~69% RS 11-25, ~14% RS 0-10
    # We'll enrich the gray zone and high-score cases to ensure stratum coverage
    score_bucket = rng.choices(
        ["low", "low_gray", "mid_gray", "high"],
        weights=[25, 20, 30, 25]
    )[0]
    if score_bucket == "low":
        oncotype_score = rng.randint(0, 15)
    elif score_bucket == "low_gray":
        oncotype_score = rng.randint(16, 20)
    elif score_bucket == "mid_gray":
        oncotype_score = rng.randint(21, 25)
    else:
        oncotype_score = rng.randint(26, 55)

    planned_endocrine_therapy = True  # ER+ so endocrine therapy is always planned

    return OncotypeCase(
        case_id=case_id,
        age=age,
        menopausal_status=menopausal_status,
        ecog=ecog,
        tumor_size_cm=round(size, 1),
        grade=grade,
        lvi_present=lvi_present,
        histology=histology,
        er_positive=er_positive,
        pr_positive=pr_positive,
        her2_positive=her2_positive,
        surgery_type=surgery_type,
        nodes_positive=nodes_positive,
        nodes_examined=nodes_examined,
        oncotype_score=oncotype_score,
        pathologic_t=pathologic_t,
        pathologic_n=pathologic_n,
        planned_endocrine_therapy=planned_endocrine_therapy,
    )


# ---------------------------------------------------------------------------
# Guideline engine
# ---------------------------------------------------------------------------

def apply_guidelines(case: OncotypeCase) -> dict:
    """Apply evidence-based (TAILORx, RxPONDER, NCCN) rules to produce
    reference decisions."""
    rules_triggered = []
    reference = {}

    rs = case.oncotype_score
    age = case.age
    nodes = case.nodes_positive
    postmenopausal = case.menopausal_status == "postmenopausal"

    # -----------------------------------------------------------------------
    # Decision 1: Chemotherapy recommendation
    # -----------------------------------------------------------------------

    chemo = None
    chemo_reason = None

    if nodes == 0:
        # Node-negative: TAILORx framework
        if rs <= 15:
            chemo = "not indicated"
            chemo_reason = "R201 (TAILORx / NCCN): RS 0-15 in node-negative HR+/HER2- disease; chemotherapy not indicated; endocrine therapy alone recommended"
            rules_triggered.append("R201")
        elif 16 <= rs <= 25:
            if age <= 50:
                # Gray zone — some benefit per TAILORx exploratory analysis
                chemo = "consider"
                chemo_reason = "R202 (TAILORx exploratory): RS 16-25 in women ≤50 with node-negative disease; chemotherapy may be considered based on exploratory subgroup analysis showing possible benefit, particularly at RS 21-25"
                rules_triggered.append("R202")
            else:
                chemo = "not indicated"
                chemo_reason = "R203 (TAILORx): RS 16-25 in women >50 with node-negative disease; endocrine therapy alone is non-inferior to chemoendocrine therapy"
                rules_triggered.append("R203")
        else:  # rs >= 26
            chemo = "recommend"
            chemo_reason = "R204 (TAILORx / NCCN): RS ≥26 in node-negative HR+/HER2- disease; chemotherapy is recommended"
            rules_triggered.append("R204")

    else:
        # Node-positive (1-3 nodes): RxPONDER framework
        if rs <= 25:
            if postmenopausal:
                chemo = "not indicated"
                chemo_reason = "R205 (RxPONDER): RS ≤25 in postmenopausal women with 1-3 positive nodes; chemotherapy does not improve invasive disease-free survival; endocrine therapy alone recommended"
                rules_triggered.append("R205")
            else:
                # Premenopausal 1-3 nodes with RS ≤25: RxPONDER showed chemo benefit
                chemo = "recommend"
                chemo_reason = "R206 (RxPONDER): RS ≤25 in premenopausal women with 1-3 positive nodes; RxPONDER demonstrated chemotherapy benefit in this subgroup"
                rules_triggered.append("R206")
        else:  # rs >= 26
            chemo = "recommend"
            chemo_reason = "R207 (NCCN): RS ≥26 in node-positive disease; chemotherapy recommended regardless of menopausal status"
            rules_triggered.append("R207")

    reference["chemotherapy"] = {"answer": chemo, "rule": chemo_reason}

    # -----------------------------------------------------------------------
    # Decision 2: Endocrine therapy
    # -----------------------------------------------------------------------

    endocrine = "yes"
    endocrine_reason = "R210: ER+ disease; adjuvant endocrine therapy recommended for minimum 5 years"
    rules_triggered.append("R210")
    reference["endocrine_therapy"] = {"answer": endocrine, "rule": endocrine_reason}

    # -----------------------------------------------------------------------
    # Decision 3: Ovarian function suppression (premenopausal)
    # -----------------------------------------------------------------------

    ofs = None
    ofs_reason = None

    if postmenopausal:
        ofs = "not applicable"
        ofs_reason = "R220: postmenopausal; OFS not applicable"
        rules_triggered.append("R220")
    else:
        # SOFT/TEXT: OFS + AI benefit in higher-risk premenopausal patients
        high_risk_premenopausal = (
            case.nodes_positive >= 1
            or (case.grade == 3 and case.tumor_size_cm >= 2.0)
            or rs >= 26
            or age < 40
        )
        if high_risk_premenopausal:
            ofs = "yes"
            ofs_reason = "R221 (SOFT/TEXT): premenopausal with high-risk features; OFS plus aromatase inhibitor or tamoxifen recommended"
            rules_triggered.append("R221")
        else:
            ofs = "no"
            ofs_reason = "R222: premenopausal low-risk; tamoxifen monotherapy without OFS is appropriate"
            rules_triggered.append("R222")

    reference["ofs"] = {"answer": ofs, "rule": ofs_reason}

    # -----------------------------------------------------------------------
    # Decision 4: Extended endocrine therapy (beyond 5 years)
    # -----------------------------------------------------------------------

    extended_et = None
    extended_et_reason = None

    # Extended endocrine therapy benefits are clearest in node-positive
    # disease, larger tumors, higher grade. Decision typically revisited
    # at year 5, but forward-looking recommendation can be given.
    if case.nodes_positive >= 1 or (case.tumor_size_cm >= 2.0 and case.grade >= 2):
        extended_et = "yes"
        extended_et_reason = "R230 (MA.17 / aTTom / ATLAS): consider extended endocrine therapy (beyond 5 years, typically to 7-10 years) given node-positive or high-risk node-negative disease"
        rules_triggered.append("R230")
    else:
        extended_et = "no"
        extended_et_reason = "R231: low-risk profile (node-negative, small, low-grade); standard 5-year endocrine therapy duration"
        rules_triggered.append("R231")

    reference["extended_endocrine_therapy"] = {"answer": extended_et, "rule": extended_et_reason}

    # -----------------------------------------------------------------------
    # Decision 5: Abemaciclib (CDK4/6i) eligibility
    # -----------------------------------------------------------------------

    cdk46 = None
    cdk46_reason = None

    # monarchE: HR+/HER2- high-risk EBC; abemaciclib x 2 years
    # High risk = ≥4 positive nodes, OR 1-3 nodes + [grade 3, OR tumor ≥5 cm, OR Ki-67 ≥20%]
    # We don't sample Ki-67; use node+grade+size as proxy
    if case.nodes_positive >= 4:
        cdk46 = "yes"
        cdk46_reason = "R240 (monarchE): ≥4 positive nodes; eligible for adjuvant abemaciclib"
        rules_triggered.append("R240")
    elif (1 <= case.nodes_positive <= 3) and (case.grade == 3 or case.tumor_size_cm >= 5.0):
        cdk46 = "yes"
        cdk46_reason = "R241 (monarchE): 1-3 positive nodes with high-risk features (grade 3 or tumor ≥5 cm); eligible for adjuvant abemaciclib"
        rules_triggered.append("R241")
    else:
        cdk46 = "no"
        cdk46_reason = "R242: does not meet monarchE high-risk criteria; abemaciclib not indicated"
        rules_triggered.append("R242")

    reference["cdk46_inhibitor"] = {"answer": cdk46, "rule": cdk46_reason}

    reference["rules_triggered"] = rules_triggered
    return reference


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_case(case: OncotypeCase) -> dict:
    d = asdict(case)
    d["clinical_vignette"] = case.to_clinical_vignette()
    d["domain"] = "oncotype_chemotherapy"
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="oncotype_cases.jsonl")
    parser.add_argument("--summary", default="oncotype_summary.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = []
    for i in range(args.n):
        case = sample_case(rng, f"ONCOTYPE-{args.seed:04d}-{i+1:04d}")
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
        "menopausal_split": dict(Counter(c.menopausal_status for c in cases)),
        "node_positive_rate": round(100 * sum(1 for c in cases if c.nodes_positive > 0) / len(cases), 1),
        "oncotype_distribution": {
            "0-15": sum(1 for c in cases if c.oncotype_score <= 15),
            "16-20": sum(1 for c in cases if 16 <= c.oncotype_score <= 20),
            "21-25": sum(1 for c in cases if 21 <= c.oncotype_score <= 25),
            "26+": sum(1 for c in cases if c.oncotype_score >= 26),
        },
        "chemotherapy_distribution": dict(Counter(c.reference["chemotherapy"]["answer"] for c in cases)),
        "ofs_distribution": dict(Counter(c.reference["ofs"]["answer"] for c in cases)),
        "cdk46_distribution": dict(Counter(c.reference["cdk46_inhibitor"]["answer"] for c in cases)),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(cases)} Oncotype cases → {args.output}")
    print()
    print("=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
