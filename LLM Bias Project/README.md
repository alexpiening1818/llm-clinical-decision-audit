# Multi-Domain Perturbation Audit of LLM Clinical Decision Stability

A pre-registered framework for measuring whether large language models give
stable recommendations under clinically irrelevant variation and responsive
recommendations under clinically relevant variation, across three
guideline-anchored oncology decision domains.

## What this is

Existing LLM evaluations in oncology measure whether models give
guideline-concordant answers on average. This doesn't tell you whether a
model is stable: a model can be concordant on average and still change its
recommendation based on patient insurance status, phrasing, or irrelevant
comorbidity. This pipeline measures that directly.

Three decision domains are covered:

1. **Adjuvant radiation therapy in early-stage breast cancer** — NCCN-anchored; decisions include RT offer, target volume, fractionation, boost, RNI
2. **Active surveillance vs. treatment in localized prostate cancer** — NCCN risk stratification; decisions include management, modality, ADT, workup, confirmatory biopsy
3. **Oncotype-guided adjuvant chemotherapy in HR+/HER2- breast cancer** — TAILORx + RxPONDER + monarchE; decisions include chemotherapy, endocrine therapy, OFS, extended ET, CDK4/6i

For each case with an unambiguous guideline answer, we apply two kinds of
perturbations:

- **Irrelevant** — should not change the recommendation (insurance, race,
  phrasing, irrelevant comorbidities, distractor findings)
- **Relevant** — should change the recommendation (crossing guideline
  thresholds for age, PSA, Oncotype RS, nodal status, or grade group)

The headline metric is a **calibration ratio** per model per domain — how
often it flips when it should, divided by how often it flips when it
shouldn't.

## Repository structure

```
perturbation_audit/
├── PREREGISTRATION.md         # full protocol (v2.0, multi-domain)
├── README.md                  # this file
├── generator.py               # breast RT case generator
├── base_cases.py              # breast RT base case filter
├── perturbations.py           # breast RT perturbation engine
├── evaluate.py                # evaluation harness (domain-agnostic)
├── analyze.py                 # flip-rate analysis (domain-agnostic)
├── make_report.py             # publication tables (domain-agnostic)
└── domains/
    ├── prostate_generator.py      # prostate AS case generator
    ├── oncotype_generator.py      # Oncotype chemotherapy case generator
    ├── base_cases_multi.py        # unified base-case filter (prostate + oncotype)
    └── perturbations_multi.py     # unified perturbation engine (prostate + oncotype)
```

## Quickstart

### 1. Generate base cases for each domain

```bash
# Breast RT (already integrated in top-level base_cases.py)
python base_cases.py --n 50 --seed 2026 --output base_cases.jsonl

# Prostate AS
cd domains
python base_cases_multi.py --domain prostate --n 50 --seed 2026 --output prostate_base.jsonl

# Oncotype
python base_cases_multi.py --domain oncotype --n 50 --seed 2026 --output oncotype_base.jsonl
```

### 2. Generate perturbations for each domain

```bash
# Breast RT
python perturbations.py --base base_cases.jsonl --output perturbations.jsonl

# Prostate AS
cd domains
python perturbations_multi.py --domain prostate --base prostate_base.jsonl --output prostate_perts.jsonl

# Oncotype
python perturbations_multi.py --domain oncotype --base oncotype_base.jsonl --output oncotype_perts.jsonl
```

Each domain produces ~700 perturbations (50 baseline + ~600 irrelevant + ~80 relevant control).

### 3. Dry-run the evaluator

```bash
python evaluate.py --perturbations perturbations.jsonl --output responses.jsonl
```

Without `--execute`, this prints the plan and estimated cost. Review before
committing API spend.

### 4. Set API keys and run evaluation

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...  # or GROQ_API_KEY
```

Test with a tiny subset:

```bash
python evaluate.py --limit 5 --repetitions 1 --execute
```

Then run the full audit for each domain. You can combine the three
perturbation files into one and run the harness once, or run three
separate campaigns. Checkpoint files are resumable.

Estimated cost for full 3-domain audit: $200-600 at frontier-model
pricing; under $25 using only open-source + Gemini free tier.

### 5. Analyze

```bash
python analyze.py --responses responses.jsonl --output analysis_summary.json
```

Prints a text report to stdout and saves full summary. For per-domain
stratification, pass `--domain` if supported.

### 6. Generate report tables

```bash
python make_report.py --summary analysis_summary.json --output report.md
```

## Pre-registration

The protocol is in `PREREGISTRATION.md` (v2.0, multi-domain). Time-stamp
this file (by pushing the repo to public GitHub with a dated commit, or
uploading to OSF) **before running any real API calls**. This is the
single most important step for the paper's credibility.

## Dependencies

```bash
pip install anthropic openai google-genai
```

Python 3.10+.

## Customizing for additional domains

The architecture is portable. To add a fourth domain:

1. Write a `{domain}_generator.py` following the pattern of
   `prostate_generator.py`: a dataclass with `to_clinical_vignette()`, a
   `sample_case(rng, id)` function, and an `apply_guidelines(case)`
   function returning a reference dict.
2. Add ambiguity rules and stratification logic to
   `domains/base_cases_multi.py`.
3. Add domain-specific relevant (R1/R2) perturbations to
   `domains/perturbations_multi.py`. C1-C4 irrelevant perturbations
   transfer unchanged.

`evaluate.py`, `analyze.py`, and `make_report.py` are domain-agnostic.

## Citation

When ready, cite the pre-registration and the final paper (to be inserted).

## License

MIT.
