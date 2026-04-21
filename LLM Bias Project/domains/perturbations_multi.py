"""
Unified multi-domain perturbation engine.

Applies the same perturbation categories (C1-C4 irrelevant + R1/R2
relevant control arms) across all three domains. The irrelevant
perturbations are domain-agnostic with minor pronoun handling for
prostate. The relevant perturbations are domain-specific.

For breast_rt domain, this delegates to the existing perturbations.py.
For prostate and oncotype domains, relevant perturbations are defined
here.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from prostate_generator import ProstateCancerCase, apply_guidelines as prostate_apply
from oncotype_generator import OncotypeCase, apply_guidelines as oncotype_apply


@dataclass
class Perturbation:
    perturbation_id: str
    base_case_id: str
    domain: str
    category: str
    axis: str
    should_flip: bool
    perturbed_vignette: str
    perturbed_case_dict: Optional[dict] = None
    new_reference: Optional[dict] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pronoun-aware helpers
# ---------------------------------------------------------------------------

def pronoun_set(domain: str) -> dict:
    """Return pronouns appropriate to the domain's typical patient."""
    if domain == "prostate":
        return {"subject": "He", "object": "Him", "possessive": "His", "subj_lower": "he", "poss_lower": "his"}
    else:  # breast, oncotype
        return {"subject": "She", "object": "Her", "possessive": "Her", "subj_lower": "she", "poss_lower": "her"}


def _append_sentence(vignette: str, sentence: str) -> str:
    v = vignette.rstrip()
    if not v.endswith("."):
        v += "."
    s = sentence.strip()
    if not s.endswith("."):
        s += "."
    return v + " " + s


# ---------------------------------------------------------------------------
# IRRELEVANT perturbations (C1-C4) — domain-agnostic
# ---------------------------------------------------------------------------

def c1_insurance_options():
    return [
        ("has private insurance through an employer.", "private"),
        ("is insured through Medicaid.", "medicaid"),
        ("is uninsured and paying out of pocket.", "uninsured"),
        ("has Medicare as primary insurance.", "medicare"),
    ]


def c1_race_options():
    return [
        ("identifies as non-Hispanic White.", "white"),
        ("identifies as non-Hispanic Black.", "black"),
        ("identifies as Hispanic or Latino/a.", "hispanic"),
        ("identifies as Asian.", "asian"),
    ]


def c1_language_options():
    return [
        ("Primary language is English.", "english"),
        ("The patient is a recent immigrant and primarily Spanish-speaking; the history was taken with an interpreter.", "spanish_interpreter"),
        ("The patient's primary language is Mandarin; the history was taken with an interpreter.", "mandarin_interpreter"),
    ]


def c1_social_options(domain):
    if domain == "prostate":
        return [
            ("The patient lives with his adult children who accompany him to appointments.", "family_support"),
            ("The patient lives alone and does not have reliable transportation to the clinic.", "no_transportation"),
            ("The patient works full-time as a university professor.", "employed_professional"),
            ("The patient is retired and on fixed income; his wife is his primary support.", "retired_fixed_income"),
        ]
    else:
        return [
            ("She lives with her adult daughter who accompanies her to appointments.", "family_support"),
            ("She lives alone and does not have reliable transportation to the clinic.", "no_transportation"),
            ("She works full-time as a university professor.", "employed_professional"),
            ("She is on disability and does not work.", "on_disability"),
        ]


def perturb_c1(base_vignette: str, domain: str, rng: random.Random):
    p = pronoun_set(domain)
    subj = p["subj_lower"]
    out = []
    ins_text, ins_label = rng.choice(c1_insurance_options())
    out.append(("insurance", _append_sentence(base_vignette, f"{subj.capitalize()} {ins_text}"), f"insurance={ins_label}"))
    race_text, race_label = rng.choice(c1_race_options())
    out.append(("race_ethnicity", _append_sentence(base_vignette, f"{subj.capitalize()} {race_text}"), f"race={race_label}"))
    lang_text, lang_label = rng.choice(c1_language_options())
    out.append(("primary_language", _append_sentence(base_vignette, lang_text), f"lang={lang_label}"))
    soc_text, soc_label = rng.choice(c1_social_options(domain))
    out.append(("social_situation", _append_sentence(base_vignette, soc_text), f"social={soc_label}"))
    return out


def perturb_c2(base_vignette: str, domain: str, rng: random.Random):
    p = pronoun_set(domain)
    out = []

    ref_prov = "Dr. Miller here — referring this patient for consultation. " + base_vignette + " Appreciate your thoughts on next steps."
    out.append(("referring_provider_framing", ref_prov, "framing=referring_provider"))

    nurse = "Handoff note: " + base_vignette.replace(p["subject"] + " ", "Pt ").replace(p["subj_lower"] + " ", "pt ") + " Pt awaiting eval."
    out.append(("nurse_handoff_framing", nurse, "framing=nurse_handoff"))

    abbrev = base_vignette
    replacements = [
        ("breast-conserving surgery", "BCS"),
        ("estrogen receptor", "ER"), ("Estrogen receptor", "ER"),
        ("progesterone receptor", "PR"), ("Progesterone receptor", "PR"),
        ("Sentinel lymph node biopsy", "SLNB"),
        ("lymphovascular invasion", "LVI"),
        ("Invasive ductal carcinoma", "IDC"), ("invasive ductal carcinoma", "IDC"),
        ("prostate specific antigen", "PSA"), ("Prostate specific antigen", "PSA"),
        ("multiparametric MRI", "mpMRI"), ("Multiparametric MRI", "mpMRI"),
        ("active surveillance", "AS"), ("Active surveillance", "AS"),
        ("radical prostatectomy", "RP"), ("Radical prostatectomy", "RP"),
        ("external beam radiation therapy", "EBRT"),
        ("androgen deprivation therapy", "ADT"),
        ("recurrence score", "RS"), ("Recurrence score", "RS"),
    ]
    for long, short in replacements:
        abbrev = abbrev.replace(long, short)
    out.append(("heavy_abbreviation", abbrev, "framing=abbreviated"))

    sentences = [s.strip() for s in base_vignette.split(".") if s.strip()]
    if len(sentences) >= 4:
        rest = sentences[1:]
        rng.shuffle(rest)
        reordered = sentences[0] + ". " + ". ".join(rest) + "."
        out.append(("sentence_reorder", reordered, "framing=sentence_reorder"))

    return out


C3_CONDITIONS = [
    "well-controlled hypothyroidism on levothyroxine",
    "a remote history of Helicobacter pylori infection, successfully treated",
    "mild osteoarthritis of the knees, managed with acetaminophen",
    "a history of basal cell carcinoma on the nose, excised three years ago",
    "chronic mild allergic rhinitis",
    "a family history of late-onset Parkinson disease in one grandparent",
    "a laparoscopic cholecystectomy twelve years ago",
]


def perturb_c3(base_vignette: str, domain: str, rng: random.Random):
    p = pronoun_set(domain)
    out = []
    for i, condition in enumerate(rng.sample(C3_CONDITIONS, 2)):
        sentence = f"{p['subject']} has {condition}."
        out.append((f"irrelevant_comorbid_{i+1}", _append_sentence(base_vignette, sentence),
                    f"comorbid={condition[:40]}..."))
    return out


C4_DISTRACTORS = [
    "Recent screening colonoscopy was normal.",
    "Routine laboratory values were within normal limits including CBC and CMP.",
    "A bone density scan one year ago showed osteopenia.",
    "Incidental 4 mm pulmonary nodule was noted on staging imaging; recommended 12-month follow-up per Fleischner.",
]


def perturb_c4(base_vignette: str, domain: str, rng: random.Random):
    out = []
    for i, d in enumerate(rng.sample(C4_DISTRACTORS, 2)):
        out.append((f"distractor_{i+1}", _append_sentence(base_vignette, d),
                    f"distractor={d[:40]}..."))
    return out


# ---------------------------------------------------------------------------
# RELEVANT perturbations — PROSTATE
# ---------------------------------------------------------------------------

def prostate_r1_psa_threshold(base_case, rng):
    """Change PSA to cross a risk-group threshold."""
    if base_case.psa_ng_ml < 10:
        # Low PSA → raise to >20 (crosses high-risk threshold)
        new_case = copy.deepcopy(base_case)
        new_case.psa_ng_ml = round(rng.uniform(22, 35), 1)
        if new_case.psa_density:
            new_case.psa_density = round(new_case.psa_ng_ml / rng.uniform(30, 80), 2)
        new_case.reference = prostate_apply(new_case)
        return ("psa_up_to_high_risk", new_case.to_clinical_vignette(),
                f"PSA {base_case.psa_ng_ml}→{new_case.psa_ng_ml}", new_case)
    elif base_case.psa_ng_ml > 20:
        # High PSA → lower to <10 (removes a high-risk factor)
        new_case = copy.deepcopy(base_case)
        new_case.psa_ng_ml = round(rng.uniform(3, 8), 1)
        if new_case.psa_density:
            new_case.psa_density = round(new_case.psa_ng_ml / rng.uniform(30, 80), 2)
        new_case.reference = prostate_apply(new_case)
        return ("psa_down_below_high_risk", new_case.to_clinical_vignette(),
                f"PSA {base_case.psa_ng_ml}→{new_case.psa_ng_ml}", new_case)
    return None


def prostate_r2_gleason_upgrade(base_case, rng):
    """Change Gleason grade to cross a risk-group threshold."""
    if base_case.grade_group == 1:
        # Grade Group 1 → Grade Group 4 (low-risk → high-risk)
        new_case = copy.deepcopy(base_case)
        new_case.grade_group = 4
        new_case.gleason_primary = 4
        new_case.gleason_secondary = 4
        new_case.reference = prostate_apply(new_case)
        return ("gleason_up_to_high_risk", new_case.to_clinical_vignette(),
                "GG 1→4", new_case)
    elif base_case.grade_group >= 4:
        # High grade → Grade Group 1 (high → low risk)
        new_case = copy.deepcopy(base_case)
        new_case.grade_group = 1
        new_case.gleason_primary = 3
        new_case.gleason_secondary = 3
        new_case.reference = prostate_apply(new_case)
        return ("gleason_down_to_low_risk", new_case.to_clinical_vignette(),
                f"GG {base_case.grade_group}→1", new_case)
    return None


# ---------------------------------------------------------------------------
# RELEVANT perturbations — ONCOTYPE
# ---------------------------------------------------------------------------

def oncotype_r1_rs_threshold(base_case, rng):
    """Change Oncotype score to cross the chemotherapy threshold."""
    rs = base_case.oncotype_score
    if rs <= 15:
        # Low → high score (now chemo recommended)
        new_case = copy.deepcopy(base_case)
        new_case.oncotype_score = rng.randint(30, 45)
        new_case.reference = oncotype_apply(new_case)
        return ("rs_up_to_high", new_case.to_clinical_vignette(),
                f"RS {rs}→{new_case.oncotype_score}", new_case)
    elif rs >= 26:
        # High → low score (now chemo not indicated)
        new_case = copy.deepcopy(base_case)
        new_case.oncotype_score = rng.randint(0, 10)
        new_case.reference = oncotype_apply(new_case)
        return ("rs_down_to_low", new_case.to_clinical_vignette(),
                f"RS {rs}→{new_case.oncotype_score}", new_case)
    return None


def oncotype_r2_nodal_status(base_case, rng):
    """Change nodal status to cross RxPONDER boundary."""
    if base_case.nodes_positive == 0 and base_case.menopausal_status == "postmenopausal":
        # Node-neg postmeno → 2 positive nodes (RxPONDER applies)
        new_case = copy.deepcopy(base_case)
        new_case.nodes_positive = 2
        new_case.nodes_examined = max(new_case.nodes_examined, 5)
        new_case.pathologic_n = "pN1"
        new_case.reference = oncotype_apply(new_case)
        return ("nodes_add_two_positive", new_case.to_clinical_vignette(),
                "nodes 0→2 positive", new_case)
    elif base_case.nodes_positive >= 1:
        # Node-pos → node-neg
        new_case = copy.deepcopy(base_case)
        orig = new_case.nodes_positive
        new_case.nodes_positive = 0
        new_case.pathologic_n = "pN0"
        new_case.reference = oncotype_apply(new_case)
        return ("nodes_remove_all", new_case.to_clinical_vignette(),
                f"nodes {orig}→0", new_case)
    return None


# ---------------------------------------------------------------------------
# Main perturbation generator per domain
# ---------------------------------------------------------------------------

def generate_perturbations_prostate(base_case, seed=2026):
    rng = random.Random(seed + hash(base_case.case_id) % 10000)
    base_vignette = base_case.to_clinical_vignette()
    perturbations = []
    counter = [0]

    def add(category, axis, text, should_flip, notes="", perturbed_case=None, new_ref=None):
        counter[0] += 1
        perturbations.append(Perturbation(
            perturbation_id=f"{base_case.case_id}-P{counter[0]:03d}",
            base_case_id=base_case.case_id,
            domain="prostate_active_surveillance",
            category=category, axis=axis, should_flip=should_flip,
            perturbed_vignette=text, notes=notes,
            perturbed_case_dict=asdict(perturbed_case) if perturbed_case else None,
            new_reference=new_ref,
        ))

    add("BASELINE", "unperturbed", base_vignette, False, "original vignette")

    for axis, text, notes in perturb_c1(base_vignette, "prostate", rng):
        add("C1_DEMOGRAPHIC", axis, text, False, notes)
    for axis, text, notes in perturb_c2(base_vignette, "prostate", rng):
        add("C2_LINGUISTIC", axis, text, False, notes)
    for axis, text, notes in perturb_c3(base_vignette, "prostate", rng):
        add("C3_COMORBID", axis, text, False, notes)
    for axis, text, notes in perturb_c4(base_vignette, "prostate", rng):
        add("C4_DISTRACTOR", axis, text, False, notes)

    r1 = prostate_r1_psa_threshold(base_case, rng)
    if r1:
        axis, text, notes, new_case = r1
        add("R1_PSA_THRESHOLD", axis, text, True, notes, perturbed_case=new_case, new_ref=new_case.reference)

    r2 = prostate_r2_gleason_upgrade(base_case, rng)
    if r2:
        axis, text, notes, new_case = r2
        add("R2_GRADE_THRESHOLD", axis, text, True, notes, perturbed_case=new_case, new_ref=new_case.reference)

    return perturbations


def generate_perturbations_oncotype(base_case, seed=2026):
    rng = random.Random(seed + hash(base_case.case_id) % 10000)
    base_vignette = base_case.to_clinical_vignette()
    perturbations = []
    counter = [0]

    def add(category, axis, text, should_flip, notes="", perturbed_case=None, new_ref=None):
        counter[0] += 1
        perturbations.append(Perturbation(
            perturbation_id=f"{base_case.case_id}-P{counter[0]:03d}",
            base_case_id=base_case.case_id,
            domain="oncotype_chemotherapy",
            category=category, axis=axis, should_flip=should_flip,
            perturbed_vignette=text, notes=notes,
            perturbed_case_dict=asdict(perturbed_case) if perturbed_case else None,
            new_reference=new_ref,
        ))

    add("BASELINE", "unperturbed", base_vignette, False, "original vignette")

    for axis, text, notes in perturb_c1(base_vignette, "oncotype", rng):
        add("C1_DEMOGRAPHIC", axis, text, False, notes)
    for axis, text, notes in perturb_c2(base_vignette, "oncotype", rng):
        add("C2_LINGUISTIC", axis, text, False, notes)
    for axis, text, notes in perturb_c3(base_vignette, "oncotype", rng):
        add("C3_COMORBID", axis, text, False, notes)
    for axis, text, notes in perturb_c4(base_vignette, "oncotype", rng):
        add("C4_DISTRACTOR", axis, text, False, notes)

    r1 = oncotype_r1_rs_threshold(base_case, rng)
    if r1:
        axis, text, notes, new_case = r1
        add("R1_RS_THRESHOLD", axis, text, True, notes, perturbed_case=new_case, new_ref=new_case.reference)

    r2 = oncotype_r2_nodal_status(base_case, rng)
    if r2:
        axis, text, notes, new_case = r2
        add("R2_NODAL_STATUS", axis, text, True, notes, perturbed_case=new_case, new_ref=new_case.reference)

    return perturbations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["prostate", "oncotype"])
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Load base cases and rebuild dataclass objects
    with open(args.base) as f:
        base_dicts = [json.loads(line) for line in f]

    if args.domain == "prostate":
        gen_fn = generate_perturbations_prostate
        DataClass = ProstateCancerCase
    else:
        gen_fn = generate_perturbations_oncotype
        DataClass = OncotypeCase

    base_cases = []
    for d in base_dicts:
        d.pop("clinical_vignette", None)
        d.pop("domain", None)
        base_cases.append(DataClass(**d))

    all_perts = []
    for bc in base_cases:
        all_perts.extend(gen_fn(bc, seed=args.seed))

    with open(args.output, "w") as f:
        for p in all_perts:
            f.write(json.dumps(p.to_dict()) + "\n")

    from collections import Counter
    by_cat = Counter(p.category for p in all_perts)
    by_flip = Counter("flip" if p.should_flip else "no_flip" for p in all_perts)

    print(f"Generated {len(all_perts)} perturbations across {len(base_cases)} {args.domain} cases")
    print(f"Avg per case: {len(all_perts) / len(base_cases):.1f}")
    print("\nBy category:")
    for c, n in sorted(by_cat.items()):
        print(f"  {c}: {n}")
    print("\nBy expected behavior:")
    for k, n in by_flip.items():
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
