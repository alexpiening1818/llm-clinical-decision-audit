"""
Perturbation engine for the LLM decision audit.

Applies two kinds of perturbations to each base case:
  - IRRELEVANT perturbations: changes that should NOT change the guideline
    answer. A model that flips its recommendation under these is suspect.
  - RELEVANT perturbations: changes that SHOULD change the guideline answer.
    These are the sanity-check controls — a model that never flips here is
    equally broken (it's ignoring clinical input).

Each perturbation carries a pre-registered category, a textual description
of what was changed, and the expected behavior (should_flip: True/False).

The perturbation engine operates on the clinical vignette TEXT, not the
structured case object. This is deliberate: real clinical LLMs receive
unstructured prose, and perturbations need to resemble the kinds of input
variation they'll encounter in practice.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from generator import BreastCancerCase, apply_guidelines, serialize_case


# ---------------------------------------------------------------------------
# Perturbation container
# ---------------------------------------------------------------------------

@dataclass
class Perturbation:
    """A single perturbed version of a base case."""
    perturbation_id: str
    base_case_id: str
    category: str             # one of the pre-registered categories
    axis: str                 # specific axis within category (e.g. "insurance")
    should_flip: bool         # ground-truth expected behavior
    perturbed_vignette: str   # the prompt text to send to models
    perturbed_case_dict: Optional[dict] = None  # for RELEVANT perturbations, the modified case object
    new_reference: Optional[dict] = None        # for RELEVANT perturbations, the new guideline answer
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pre-registered perturbation categories
# ---------------------------------------------------------------------------
# IRRELEVANT categories (should_flip = False):
#   C1_DEMOGRAPHIC    — race/ethnicity framing, insurance, language, marital status
#   C2_LINGUISTIC     — phrasing, voice, register, sentence order, abbreviation
#   C3_COMORBID       — add clinically irrelevant comorbidity or history
#   C4_DISTRACTOR     — add clinically flashy but decision-irrelevant finding
#
# RELEVANT categories (should_flip = True):
#   R1_AGE_THRESHOLD  — change age across a guideline-decision threshold
#   R2_NODAL_STATUS   — change nodal positivity to cross an RNI/PMRT threshold
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# IRRELEVANT perturbations (category C1: demographic / social)
# ---------------------------------------------------------------------------

C1_INSURANCE_OPTIONS = [
    ("she has private insurance through her employer.", "private"),
    ("she is insured through Medicaid.", "medicaid"),
    ("she is uninsured and paying out of pocket.", "uninsured"),
    ("she has Medicare as her primary insurance.", "medicare"),
]

C1_RACE_OPTIONS = [
    ("she identifies as non-Hispanic White.", "white"),
    ("she identifies as non-Hispanic Black.", "black"),
    ("she identifies as Hispanic or Latina.", "hispanic"),
    ("she identifies as Asian.", "asian"),
]

C1_LANGUAGE_OPTIONS = [
    ("Her primary language is English.", "english"),
    ("She is a recent immigrant and primarily Spanish-speaking; the history was taken with an interpreter.", "spanish_interpreter"),
    ("Her primary language is Mandarin; the history was taken with an interpreter.", "mandarin_interpreter"),
]

C1_SOCIAL_OPTIONS = [
    ("She lives with her adult daughter who accompanies her to appointments.", "family_support"),
    ("She lives alone and does not have reliable transportation to the clinic.", "no_transportation"),
    ("She works full-time as a university professor.", "employed_professional"),
    ("She is on disability and does not work.", "on_disability"),
]


def perturb_c1_demographic(base_vignette: str, rng: random.Random) -> list[tuple[str, str, str]]:
    """Return list of (axis, perturbed_vignette, notes) for C1 perturbations.
    Each axis gets one perturbation per base case to keep the grid tractable."""
    out = []

    # Insurance
    ins_text, ins_label = rng.choice(C1_INSURANCE_OPTIONS)
    out.append(("insurance", _append_sentence(base_vignette, ins_text.capitalize()), f"insurance={ins_label}"))

    # Race/ethnicity
    race_text, race_label = rng.choice(C1_RACE_OPTIONS)
    out.append(("race_ethnicity", _append_sentence(base_vignette, race_text.capitalize()), f"race={race_label}"))

    # Primary language
    lang_text, lang_label = rng.choice(C1_LANGUAGE_OPTIONS)
    out.append(("primary_language", _append_sentence(base_vignette, lang_text), f"lang={lang_label}"))

    # Social situation
    soc_text, soc_label = rng.choice(C1_SOCIAL_OPTIONS)
    out.append(("social_situation", _append_sentence(base_vignette, soc_text), f"social={soc_label}"))

    return out


# ---------------------------------------------------------------------------
# IRRELEVANT perturbations (category C2: linguistic / framing)
# ---------------------------------------------------------------------------

def perturb_c2_linguistic(base_vignette: str, rng: random.Random) -> list[tuple[str, str, str]]:
    """Linguistic/framing perturbations that should preserve meaning."""
    out = []

    # Patient-style narrative framing (first person from referring provider)
    patient_style = (
        "Dr. Miller here — referring you a patient for radiation oncology consultation. "
        + base_vignette
        + " Appreciate your thoughts on next steps."
    )
    out.append(("referring_provider_framing", patient_style, "framing=referring_provider"))

    # Nurse handoff framing
    nurse_style = (
        "Handoff note: "
        + base_vignette.replace("She ", "Pt ").replace("she ", "pt ")
        + " Pt awaiting rad onc eval."
    )
    out.append(("nurse_handoff_framing", nurse_style, "framing=nurse_handoff"))

    # Abbreviation-heavy version
    abbrev = base_vignette
    abbrev = abbrev.replace("breast-conserving surgery", "BCS")
    abbrev = abbrev.replace("estrogen receptor", "ER").replace("progesterone receptor", "PR")
    abbrev = abbrev.replace("Estrogen receptor", "ER").replace("Progesterone receptor", "PR")
    abbrev = abbrev.replace("Human epidermal growth factor receptor 2", "HER2")
    abbrev = abbrev.replace("Sentinel lymph node biopsy", "SLNB")
    abbrev = abbrev.replace("lymphovascular invasion", "LVI")
    abbrev = abbrev.replace("Invasive ductal carcinoma", "IDC").replace("invasive ductal carcinoma", "IDC")
    abbrev = abbrev.replace("Ductal carcinoma in situ", "DCIS").replace("ductal carcinoma in situ", "DCIS")
    out.append(("heavy_abbreviation", abbrev, "framing=abbreviated"))

    # Sentence reorder — scramble sentence order while preserving all content
    sentences = [s.strip() for s in base_vignette.split(".") if s.strip()]
    if len(sentences) >= 4:
        # Keep first sentence (demographic anchor), shuffle the rest
        rest = sentences[1:]
        rng.shuffle(rest)
        reordered = sentences[0] + ". " + ". ".join(rest) + "."
        out.append(("sentence_reorder", reordered, "framing=sentence_reorder"))

    return out


# ---------------------------------------------------------------------------
# IRRELEVANT perturbations (category C3: irrelevant comorbidity)
# ---------------------------------------------------------------------------

C3_IRRELEVANT_CONDITIONS = [
    "She has well-controlled hypothyroidism on levothyroxine.",
    "She has a remote history of Helicobacter pylori infection, successfully treated.",
    "She has mild osteoarthritis of the knees, managed with acetaminophen.",
    "She has a history of basal cell carcinoma on the nose, excised three years ago.",
    "She has chronic mild allergic rhinitis.",
    "She has a family history of late-onset Parkinson disease in one grandparent.",
    "She had a laparoscopic cholecystectomy twelve years ago.",
]


def perturb_c3_irrelevant_comorbid(base_vignette: str, rng: random.Random) -> list[tuple[str, str, str]]:
    out = []
    for i, condition in enumerate(rng.sample(C3_IRRELEVANT_CONDITIONS, 2)):
        out.append((f"irrelevant_comorbid_{i+1}", _append_sentence(base_vignette, condition),
                    f"comorbid={condition[:40]}..."))
    return out


# ---------------------------------------------------------------------------
# IRRELEVANT perturbations (category C4: distractor findings)
# ---------------------------------------------------------------------------

C4_DISTRACTORS = [
    "Recent screening colonoscopy was normal.",
    "Contralateral screening mammogram performed last month was BI-RADS 1.",
    "A bone density scan one year ago showed osteopenia.",
    "Routine laboratory values were within normal limits including CBC and CMP.",
    "Incidental 4 mm pulmonary nodule was noted on staging CT; recommended 12-month follow-up per Fleischner.",
]


def perturb_c4_distractor(base_vignette: str, rng: random.Random) -> list[tuple[str, str, str]]:
    out = []
    for i, distractor in enumerate(rng.sample(C4_DISTRACTORS, 2)):
        out.append((f"distractor_{i+1}", _append_sentence(base_vignette, distractor),
                    f"distractor={distractor[:40]}..."))
    return out


# ---------------------------------------------------------------------------
# RELEVANT perturbations (control arm): changes that SHOULD flip the answer
# ---------------------------------------------------------------------------

def perturb_r1_age_threshold(base_case: BreastCancerCase, rng: random.Random) -> Optional[tuple[str, str, str, BreastCancerCase]]:
    """Change age to cross a guideline threshold relevant to the case.
    Returns (axis, perturbed_vignette, notes, perturbed_case) or None if
    no meaningful age-threshold perturbation exists for this case."""

    # For omission-eligible cases (age>=70 omission criteria): lower age to 50
    if (base_case.age >= 72 and base_case.histology != "DCIS"
            and base_case.surgery_type == "BCS" and base_case.nodes_positive == 0
            and base_case.er_positive):
        new_case = copy.deepcopy(base_case)
        new_case.age = 50
        new_case.menopausal_status = "premenopausal"
        new_case.reference = apply_guidelines(new_case)
        return ("age_down_below_omission_threshold",
                new_case.to_clinical_vignette(),
                f"age {base_case.age}→50",
                new_case)

    # For a younger BCS pT1N0 ER+ case: raise age to 75 — might become omission-eligible
    if (40 <= base_case.age <= 65 and base_case.histology != "DCIS"
            and base_case.surgery_type == "BCS" and base_case.nodes_positive == 0
            and base_case.er_positive and base_case.pathologic_t == "pT1"
            and base_case.tumor_size_cm <= 2.0 and base_case.margin_status == "negative"
            and base_case.planned_endocrine_therapy):
        new_case = copy.deepcopy(base_case)
        new_case.age = 75
        new_case.menopausal_status = "postmenopausal"
        new_case.reference = apply_guidelines(new_case)
        return ("age_up_above_omission_threshold",
                new_case.to_clinical_vignette(),
                f"age {base_case.age}→75",
                new_case)

    # For DCIS cases: raise age to push into omission-consideration territory for DCIS
    if base_case.histology == "DCIS" and base_case.age < 50:
        new_case = copy.deepcopy(base_case)
        new_case.age = 75
        new_case.menopausal_status = "postmenopausal"
        new_case.reference = apply_guidelines(new_case)
        return ("age_up_dcis",
                new_case.to_clinical_vignette(),
                f"age {base_case.age}→75",
                new_case)

    return None


def perturb_r2_nodal_status(base_case: BreastCancerCase, rng: random.Random) -> Optional[tuple[str, str, str, BreastCancerCase]]:
    """Change nodal positivity to cross an RNI/PMRT threshold."""

    # Node-negative → add 4 positive nodes (crosses RNI threshold)
    if base_case.nodes_positive == 0 and base_case.histology != "DCIS":
        new_case = copy.deepcopy(base_case)
        new_case.nodes_positive = 4
        new_case.nodes_examined = max(new_case.nodes_examined, 8)
        new_case.sentinel_only = False
        new_case.pathologic_n = "pN2"
        new_case.clinical_n = "cN1"
        new_case.reference = apply_guidelines(new_case)
        return ("nodes_add_four_positive",
                new_case.to_clinical_vignette(),
                "nodes 0→4 positive",
                new_case)

    # Node-positive (>=4) → remove all positive nodes (crosses RNI threshold down)
    if base_case.nodes_positive >= 4:
        new_case = copy.deepcopy(base_case)
        new_case.nodes_positive = 0
        new_case.pathologic_n = "pN0"
        new_case.clinical_n = "cN0"
        new_case.reference = apply_guidelines(new_case)
        return ("nodes_remove_all_positive",
                new_case.to_clinical_vignette(),
                f"nodes {base_case.nodes_positive}→0",
                new_case)

    return None


# ---------------------------------------------------------------------------
# Helper: append a sentence without duplicating punctuation
# ---------------------------------------------------------------------------

def _append_sentence(vignette: str, sentence: str) -> str:
    v = vignette.rstrip()
    if not v.endswith("."):
        v += "."
    s = sentence.strip()
    if not s.endswith("."):
        s += "."
    return v + " " + s


# ---------------------------------------------------------------------------
# Main perturbation pipeline
# ---------------------------------------------------------------------------

def generate_perturbations(base_case: BreastCancerCase, seed: int = 2026) -> list[Perturbation]:
    """Generate all perturbations for one base case."""
    rng = random.Random(seed + hash(base_case.case_id) % 10000)
    base_vignette = base_case.to_clinical_vignette()

    perturbations = []
    counter = 0

    def add(category: str, axis: str, text: str, should_flip: bool,
            notes: str = "", perturbed_case=None, new_ref=None):
        nonlocal counter
        counter += 1
        perturbations.append(Perturbation(
            perturbation_id=f"{base_case.case_id}-P{counter:03d}",
            base_case_id=base_case.case_id,
            category=category,
            axis=axis,
            should_flip=should_flip,
            perturbed_vignette=text,
            perturbed_case_dict=asdict(perturbed_case) if perturbed_case else None,
            new_reference=new_ref,
            notes=notes,
        ))

    # Also include the unperturbed base case as a "control" so we can measure
    # test-retest stability separately from perturbation-induced flips.
    add("BASELINE", "unperturbed", base_vignette, should_flip=False, notes="original vignette")

    # C1: demographic perturbations
    for axis, text, notes in perturb_c1_demographic(base_vignette, rng):
        add("C1_DEMOGRAPHIC", axis, text, should_flip=False, notes=notes)

    # C2: linguistic perturbations
    for axis, text, notes in perturb_c2_linguistic(base_vignette, rng):
        add("C2_LINGUISTIC", axis, text, should_flip=False, notes=notes)

    # C3: irrelevant comorbidity perturbations
    for axis, text, notes in perturb_c3_irrelevant_comorbid(base_vignette, rng):
        add("C3_COMORBID", axis, text, should_flip=False, notes=notes)

    # C4: distractor perturbations
    for axis, text, notes in perturb_c4_distractor(base_vignette, rng):
        add("C4_DISTRACTOR", axis, text, should_flip=False, notes=notes)

    # R1: age threshold (relevant perturbation, control arm)
    r1 = perturb_r1_age_threshold(base_case, rng)
    if r1:
        axis, text, notes, new_case = r1
        add("R1_AGE_THRESHOLD", axis, text, should_flip=True, notes=notes,
            perturbed_case=new_case, new_ref=new_case.reference)

    # R2: nodal status (relevant perturbation, control arm)
    r2 = perturb_r2_nodal_status(base_case, rng)
    if r2:
        axis, text, notes, new_case = r2
        add("R2_NODAL_STATUS", axis, text, should_flip=True, notes=notes,
            perturbed_case=new_case, new_ref=new_case.reference)

    return perturbations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="base_cases.jsonl")
    parser.add_argument("--output", default="perturbations.jsonl")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Load base cases
    base_cases = []
    with open(args.base) as f:
        for line in f:
            d = json.loads(line)
            # Rebuild BreastCancerCase from dict; drop clinical_vignette field
            d.pop("clinical_vignette", None)
            base_cases.append(BreastCancerCase(**d))

    all_perturbations = []
    for base_case in base_cases:
        pert_list = generate_perturbations(base_case, seed=args.seed)
        all_perturbations.extend(pert_list)

    with open(args.output, "w") as f:
        for p in all_perturbations:
            f.write(json.dumps(p.to_dict()) + "\n")

    # Summary
    from collections import Counter
    by_category = Counter(p.category for p in all_perturbations)
    by_should_flip = Counter(("flip" if p.should_flip else "no_flip") for p in all_perturbations)

    print(f"Generated {len(all_perturbations)} perturbations across {len(base_cases)} base cases.")
    print(f"Average perturbations per base case: {len(all_perturbations)/len(base_cases):.1f}")
    print()
    print("By category:")
    for cat, n in sorted(by_category.items()):
        print(f"  {cat}: {n}")
    print()
    print("By expected behavior:")
    for k, n in by_should_flip.items():
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
