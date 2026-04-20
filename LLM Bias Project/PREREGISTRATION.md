# Perturbation-Based Audit of LLM Decision Stability in Adjuvant Radiation Therapy for Early-Stage Breast Cancer

## Pre-Registration Protocol

**Version:** 1.0
**Status:** Draft — to be time-stamped before any model is run
**Date of finalization:** 4/20/2026
**Authors:** Alexander Piening

---

## 1. Background and Rationale

Large language models are being studied as decision-support tools in oncology, with growing literature on guideline concordance for treatment recommendations. Published work in this area focuses almost exclusively on whether the model gives the guideline-adherent answer on average. This metric is necessary but not sufficient for safe clinical deployment. A model can be concordant on average and still be dangerous if its recommendations shift in response to inputs that should not matter clinically, such as how the case is phrased, the patient's insurance status, or the inclusion of decision-irrelevant comorbidities.

Equivalently, a model must demonstrate two properties to be trustworthy as decision support: **stability** under clinically irrelevant variation, and **sensitivity** to clinically relevant variation. These properties are not captured by concordance. They are captured by what we term a **perturbation audit**: a systematic assessment of how often a model changes its recommendation when the input changes along axes that should not change the answer, versus axes that should.

No published study in radiation oncology has conducted a perturbation audit of LLM decision-making with a pre-registered design, multi-model comparison, and an explicit calibration metric. This study fills that gap in the setting of adjuvant radiation therapy for early-stage breast cancer.

## 2. Study Design

### 2.1 Overall Design

A pre-registered, synthetic-case, multi-model, paired perturbation audit. Each base case is a guideline-anchored synthetic patient vignette with an unambiguous reference answer for at least three of five RT decision tasks. Each base case is paired with a set of perturbations: some clinically irrelevant (target: recommendation should not change), some clinically relevant (target: recommendation should change). Each model is run three times per perturbation to estimate stochastic variation.

### 2.2 Target Population

Synthetic patients with early-stage breast cancer who have undergone primary surgical treatment and are referred for adjuvant radiation therapy consultation. Patient characteristics are sampled from distributions calibrated to the SEER early-stage breast cancer population with deliberate enrichment for clinically relevant scenarios (elderly patients eligible for RT omission; node-positive patients eligible for RNI; mastectomy patients with and without classic PMRT indications; DCIS post-BCS with clear RT indication).

## 3. Decision Tasks

Each model is asked to provide five decisions per case:

1. **RT recommendation** — offer RT, conditional, or omission acceptable
2. **Target volume** — whole breast, APBI, APBI-or-whole-breast, whole breast plus RNI, chest wall, chest wall plus RNI, or not applicable
3. **Fractionation** — conventional, moderate hypofractionation, ultra-hypofractionation, APBI, multiple acceptable, or not applicable
4. **Tumor bed boost** — indicated, not indicated, consider, optional, or not applicable
5. **Regional nodal irradiation** — yes, no, strongly consider, or consider

Response format is a structured JSON object (enforced via system prompt).

## 4. Reference Answers

Reference answers are generated algorithmically from a guideline engine anchored to NCCN Breast Cancer Guidelines (current version at time of execution, to be cited specifically in the final paper with version number and date accessed). Every reference answer carries a rule identifier traceable to a specific guideline decision node.

## 5. Base Case Selection

Base cases are selected from an oversampled cohort by a pre-registered filter that requires the reference answer to be unambiguous. "Unambiguous" is operationalized per decision task:

- RT recommendation: excludes DCIS "conditional" cases; excludes RT omission at age 70–71; excludes PMRT gray zones (1–3 positive nodes with age ≥75 or significant comorbidity)
- Target volume: excludes APBI-or-WBI cases where NCCN endorses both
- Fractionation: kept if there is a defined preferred or multiply-acceptable regimen; excluded only when RT is not indicated
- Boost: excludes "consider" and "optional" cases
- RNI: excludes "strongly consider" (1–3 nodes) and "consider" (node-negative high-risk)

A base case enters the cohort if the RT recommendation decision is unambiguous and at least 3 of 5 decisions are unambiguous. The audit only evaluates flips on decisions that are individually auditable for each case.

Base cases are stratified across six pre-registered clinical scenarios to ensure coverage:

| Stratum | Target count |
|---|---|
| BCS, node-negative, omission-eligible | ~8 |
| BCS, node-negative, standard WBI | ~16 |
| BCS, node-positive with RNI indication | ~6 |
| Mastectomy with classic PMRT indication | ~6 |
| Mastectomy, node-negative, no PMRT | ~6 |
| DCIS post-BCS with clear RT indication | ~8 |

Target total: 50 base cases.

## 6. Perturbations

### 6.1 Categories of Irrelevant Perturbations (should not change recommendation)

- **C1 — Demographic and social factors:** insurance status, race/ethnicity, primary language, social/living situation
- **C2 — Linguistic framing:** referring-provider framing, nurse-handoff framing, heavy abbreviation, sentence reordering
- **C3 — Clinically irrelevant comorbidity:** addition of well-controlled conditions unrelated to the RT decision (hypothyroidism, remote unrelated malignancies, etc.)
- **C4 — Distractor findings:** addition of clinically flashy but decision-irrelevant findings (incidental imaging findings, normal labs, contralateral screening results)

### 6.2 Categories of Relevant Perturbations (control arm; should change recommendation)

- **R1 — Age threshold:** modification of age to cross a guideline-relevant decision threshold (e.g., crossing the 70-year CALGB 9343 omission threshold, or moving a DCIS case between age groups)
- **R2 — Nodal status:** modification of nodal positivity to cross an RNI or PMRT threshold (e.g., 0 → 4 positive nodes, or 4+ → 0)

### 6.3 Perturbations Per Case

Each base case produces approximately 14 perturbations:
- 1 BASELINE (unperturbed; used for test-retest stability)
- 4 C1 (demographic)
- 4 C2 (linguistic)
- 2 C3 (comorbidity)
- 2 C4 (distractor)
- 0–1 R1 (age; only when a meaningful threshold exists for the case)
- 0–1 R2 (nodal; only when a meaningful threshold exists for the case)

Relevant perturbations (R1, R2) are generated only when they cross a guideline-relevant threshold for the specific base case. This means the relevant-perturbation arm will be smaller than the irrelevant arm, with asymmetric statistical power by design. This is noted as a limitation; the study is not under-powered for the primary analysis because the irrelevant arm is large.

## 7. Models

Four models, pre-registered:

1. **Claude Opus 4.7** (Anthropic) — frontier closed
2. **GPT-5** (OpenAI) — frontier closed
3. **Gemini 2.5 Pro** (Google) — frontier closed
4. **Llama 3.3 70B Instruct** (Meta, via Together.ai) — open-source baseline

All models called at temperature 0.2 with max_tokens 600. Exact model versions and API dates recorded at execution time.

Each model is called three times per perturbation to capture stochastic variation.

## 8. Primary Outcomes

### 8.1 Inappropriate Flip Rate (per model)

P(decision flips vs. baseline | perturbation is clinically irrelevant)

Computed as a proportion with Wilson 95% confidence interval. Lower is better. Only auditable decisions are included in the denominator.

### 8.2 Appropriate Flip Rate (per model)

P(decision flips vs. baseline | perturbation is clinically relevant)

Higher is better.

### 8.3 Calibration Ratio (per model)

(appropriate flip rate) / (inappropriate flip rate)

A ratio ≫ 1 indicates the model flips when it should and not when it shouldn't. A ratio near 1 indicates the model is flipping approximately at random with respect to clinical relevance.

## 9. Secondary Outcomes

- **Per-category inappropriate flip rates:** flip rate stratified by C1 / C2 / C3 / C4 to identify which class of irrelevant variation the model is most sensitive to
- **Per-decision-task flip rates:** stratified by which of the five RT decisions flipped, identifying task-specific weaknesses
- **Per-axis flip rates:** fine-grained flip rates for specific perturbation axes (e.g., insurance = Medicaid vs. private) to identify specific triggers
- **Test-retest stability on BASELINE:** proportion of (base case × model) pairs where all three BASELINE repetitions agree on each decision. This separates stochastic variation from perturbation-induced flipping
- **Pairwise model comparisons** via McNemar's test on paired perturbation outcomes

## 10. Sample Size Rationale

- 50 base cases × ~14 perturbations × 4 models × 3 repetitions ≈ 8,400 model calls
- With ~600 irrelevant-perturbation observations per model per decision task, we have statistical power to detect inappropriate flip rates as small as 2–3% with narrow confidence intervals
- Relevant-perturbation arm is smaller (~55 per model) but still powered to detect departures from near-100% appropriate flipping

## 11. Statistical Analysis Plan

- Proportions reported with Wilson 95% confidence intervals
- Pairwise model comparisons on inappropriate flip rate via McNemar's test, paired at the (base case, perturbation, decision task) level
- No multiple-comparison correction applied to exploratory secondary analyses; all secondary findings reported as descriptive
- No missing-data imputation; responses that fail parsing are excluded from the relevant denominator and reported separately

## 12. Hypotheses (Pre-Specified)

1. **Primary:** Frontier models (Claude Opus 4.7, GPT-5, Gemini 2.5 Pro) will have lower inappropriate flip rates than the open-source baseline (Llama 3.3 70B).
2. **Primary:** All models will have calibration ratios meaningfully greater than 1 (at least 3.0), indicating they do respond to clinically relevant information.
3. **Secondary:** C1 (demographic) perturbations will produce inappropriate flip rates at least as high as C2 (linguistic), C3 (comorbidity), and C4 (distractor) perturbations across models.
4. **Secondary:** Specific demographic axes (insurance, primary language) will emerge as higher-flip-rate axes than others within the C1 category.

## 13. Deviations from Pre-Registration

Any deviation from this protocol will be reported explicitly in the methods section of the final paper, with rationale.

## 14. Data and Code Availability

All code (generator, perturbation engine, evaluation harness, analysis) will be released under an MIT license on GitHub at submission. The synthetic case set and the anonymized response data will be released as JSONL alongside.

## 15. Not Covered by This Pre-Registration

- Any future perturbation audit in other tumor sites (separate pre-registration for each)
- Real-case (as opposed to synthetic) evaluation
- Fine-tuned or domain-specific models
- Multi-turn interactive decision support (this study evaluates single-shot decisions only)
