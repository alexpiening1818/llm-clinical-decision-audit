# Multi-Domain Perturbation Audit of LLM Clinical Decision Stability in Guideline-Anchored Oncology Decisions

## Pre-Registration Protocol (v2.0, multi-domain)

**Status:** Draft — to be time-stamped before any model is run
**Version history:**
  - v1.0 (breast RT only) — see git history for original
  - v2.0 (this version) — expanded to three domains per PI feedback

---

## 1. Background and Rationale

Existing LLM evaluation in oncology measures whether models give guideline-concordant answers on average. This is necessary but not sufficient: a model can be concordant on average while still changing its recommendation based on inputs that should not matter clinically (phrasing, patient demographics, irrelevant comorbidities). Model trustworthiness requires two properties:

1. **Stability** under clinically irrelevant variation
2. **Sensitivity** to clinically relevant variation

Neither property is captured by concordance metrics. Both are captured by a **perturbation audit** — a systematic measurement of how often a model flips its recommendation when the input changes along axes that should not matter, versus axes that should.

This study applies a pre-registered perturbation audit to three guideline-anchored oncology decision domains, selected to cover different diseases, different patient populations, and different modality classes.

## 2. Domains

### Domain 1: Adjuvant radiation therapy in early-stage breast cancer
Management decisions include RT offer vs. omission, target volume, fractionation, boost, and regional nodal irradiation. Guideline anchors: NCCN Breast, ASTRO APBI guidelines, CALGB 9343, PRIME II, Roa, MA.20, EORTC 22922.

### Domain 2: Active surveillance vs. definitive treatment in clinically localized prostate cancer
Management decisions include primary management choice (AS vs. treatment vs. observation), treatment modality when treatment is recommended, ADT indication and duration, advanced workup (mpMRI, genomic classifier), and confirmatory biopsy. Guideline anchor: NCCN Prostate risk stratification (very-low, low, favorable intermediate, unfavorable intermediate, high, very-high).

### Domain 3: Oncotype-guided adjuvant chemotherapy in HR+/HER2- early-stage breast cancer
Decisions include chemotherapy indication per TAILORx/RxPONDER, endocrine therapy, OFS in premenopausal patients, extended endocrine therapy, and CDK4/6 inhibitor eligibility (monarchE). Guideline anchors: TAILORx, RxPONDER, SOFT/TEXT, monarchE, NCCN Breast.

The three domains span different diseases (breast vs. prostate), genders, decision structures (treatment intensity, surveillance vs. treatment, biomarker-driven chemotherapy), and treatment modalities (RT, surgery, systemic therapy).

## 3. Study Design

A pre-registered, synthetic-case, multi-model, paired perturbation audit in each domain. For each domain, cases are sampled from guideline-anchored distributions and filtered to those with unambiguous reference answers on at least three decision tasks. Each base case is paired with a set of perturbations, some clinically irrelevant and some clinically relevant. Each model is run three times per perturbation to estimate stochastic variation.

## 4. Decision Tasks

Each domain has 5 decision tasks; reference answers are generated algorithmically from rule engines that encode the guideline logic.

**Breast RT:** rt_recommendation, target_volume, fractionation, boost, rni

**Prostate AS:** management, modality, adt, workup, confirmatory_biopsy

**Oncotype chemotherapy:** chemotherapy, endocrine_therapy, ofs, extended_endocrine_therapy, cdk46_inhibitor

Response format is a structured JSON object enforced via system prompt.

## 5. Reference Answers and Ambiguity Filtering

Reference answers are generated algorithmically. Every rule carries an identifier (R###) citing the specific guideline decision node or underlying trial.

For each case, per-decision auditability is determined by a pre-registered ambiguity filter. A case enters the base cohort if the primary decision for that domain is unambiguous AND at least 3 of 5 decisions are unambiguous. Flip analysis only considers decisions that are individually auditable for the given case.

## 6. Base Case Selection

~50 base cases per domain (±2, allowing for stratum availability), selected from oversampled generator output (oversample factor 50x).

Stratification per domain:

**Breast RT (6 strata):** BCS node-negative omission-eligible; BCS node-negative standard WBI; BCS node-positive with RNI indication; mastectomy with clear PMRT indication; mastectomy node-negative no PMRT; DCIS post-BCS with clear RT indication.

**Prostate AS (7 strata):** very-low-risk AS; low-risk AS; favorable intermediate; unfavorable intermediate; high-risk; very-high-risk; limited life expectancy (observation).

**Oncotype (6 strata):** node-negative low RS; node-negative gray zone age >50; node-negative high RS; node-positive postmenopausal low RS; node-positive premenopausal low RS; node-positive high RS. (Gray-zone young cases are generated but filtered from the base cohort because the chemotherapy reference is "consider" — genuinely ambiguous.)

Total base cohort: ~150 cases across three domains.

## 7. Perturbations

Each base case receives ~14 perturbations across 7 categories:

### Irrelevant (should NOT change the recommendation)
- **C1 — Demographic and social:** insurance status, race/ethnicity, primary language, social situation
- **C2 — Linguistic framing:** referring-provider narrative, nurse handoff, heavy abbreviation, sentence reordering
- **C3 — Clinically irrelevant comorbidity:** addition of well-controlled or remote conditions (hypothyroidism, remote unrelated malignancies, etc.)
- **C4 — Distractor findings:** addition of flashy but decision-irrelevant findings (incidental nodules, normal labs, bone density)

### Relevant (SHOULD change the recommendation; control arm)
- **R1 — Biomarker / staging threshold:** crosses a guideline decision threshold for the specific domain
  - Breast: age threshold (70-year omission threshold, DCIS age stratification)
  - Prostate: PSA threshold (crosses >20 ng/mL high-risk boundary)
  - Oncotype: RS threshold (crosses 16 or 26 boundary)
- **R2 — Nodal/grade threshold:** crosses an RNI/PMRT or risk-group threshold
  - Breast: nodal status (0 ↔ 4+ positive)
  - Prostate: Gleason grade group (GG1 ↔ GG4)
  - Oncotype: nodal status (0 ↔ 2+ positive)

Each base case also includes 1 BASELINE perturbation (unperturbed vignette, used for test-retest stability).

## 8. Models

Four pre-registered models:

1. **Claude Opus 4.7** (Anthropic) — frontier closed
2. **GPT-5** (OpenAI) — frontier closed
3. **Gemini 2.5 Pro** (Google) — frontier closed
4. **Llama 3.3 70B Instruct** (Meta, via Together.ai or Groq) — open-source baseline

All models called at temperature 0.2, max_tokens 600. Three repetitions per perturbation. Exact model versions and API dates recorded at execution time.

If sufficient API credits cannot be obtained for the closed models, the study may be run with a subset (minimum Gemini + Llama + one additional open-source model such as DeepSeek V3 or Qwen 2.5 72B). Any model-panel deviation will be disclosed in the final paper.

## 9. Primary Outcomes

Computed per model, per domain:

1. **Inappropriate flip rate:** P(decision flips vs. baseline | perturbation is irrelevant)
2. **Appropriate flip rate:** P(decision flips vs. baseline | perturbation is relevant)
3. **Calibration ratio:** appropriate / inappropriate

A calibration ratio ≫ 1 indicates the model flips appropriately in response to clinical signal and not in response to clinically irrelevant noise.

## 10. Secondary Outcomes

- **Per-category inappropriate flip rates:** C1 vs C2 vs C3 vs C4 within each domain
- **Per-decision-task flip rates:** which specific decisions are most unstable
- **Per-axis flip rates:** which specific perturbations (e.g., insurance=Medicaid) have highest flip rates
- **Test-retest stability:** proportion of (case × model) pairs with all three BASELINE repetitions unanimous
- **Cross-domain comparison:** are models consistently calibrated across all three domains, or domain-specific?
- **Race/insurance-specific analysis in prostate AS:** real-world literature documents disparities in AS uptake by race; flip rates on the C1 race and insurance axes are of particular interest

## 11. Sample Size Rationale

~50 base cases × ~14 perturbations × 4 models × 3 repetitions × 3 domains ≈ 25,200 API calls.

Per domain: ~600-700 irrelevant-perturbation observations per model per decision task, giving power to detect inappropriate flip rates as small as 2-3% with narrow CIs. Relevant-perturbation arm is smaller (~55-80 per domain per model) but powered to detect departures from ~100% appropriate flipping.

## 12. Statistical Analysis Plan

- Proportions with Wilson 95% CIs
- Pairwise model comparisons within domain via McNemar's test on paired (base case, perturbation, decision task) outcomes
- Cross-domain model comparisons descriptive
- Per-axis analysis restricted to axes with n ≥ 10 observations per model
- No multiple-comparison correction for secondary analyses (exploratory, descriptive reporting)
- Missing data: responses that fail parsing excluded from denominators and reported separately per model

## 13. Pre-Specified Hypotheses

**Primary:**
1. Frontier models (Claude Opus 4.7, GPT-5, Gemini 2.5 Pro) will have lower inappropriate flip rates than Llama 3.3 70B within each domain.
2. All models will achieve calibration ratios > 3.0 in at least two of three domains.

**Secondary:**
3. C1 (demographic) perturbations will produce inappropriate flip rates at least as high as C2/C3/C4 across models, indicating models are influenced by patient demographics.
4. Race and insurance axes will show higher flip rates in the prostate AS domain than in the breast domains, consistent with real-world disparity literature.
5. Inappropriate flip rates will vary across domains for the same model, suggesting decision-domain-specific failure modes.

## 14. Deviations

Any deviation from this protocol will be reported explicitly in the methods section of the final paper, with rationale.

## 15. Data and Code Availability

All code, synthetic cases, perturbation lists, and response data will be released under MIT license on GitHub at submission.

## 16. Out of Scope

- Real (non-synthetic) patient cases — future work
- Multi-turn interactive decision support — single-shot only
- Fine-tuned or domain-specific models — general-purpose frontier models only
- Decision domains beyond the three listed here — separate pre-registration required
