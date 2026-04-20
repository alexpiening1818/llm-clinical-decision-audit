# LLM Perturbation Audit for Adjuvant Radiation Therapy Decisions in Early-Stage Breast Cancer

A pre-registered framework for measuring whether large language models give
stable recommendations under clinically irrelevant variation and responsive
recommendations under clinically relevant variation.

## What this is

Existing LLM evaluations in radiation oncology measure whether models give
guideline-concordant answers on average. This doesn't tell you whether a
model is stable: a model can be concordant on average and still change its
recommendation based on a patient's insurance status, a different phrasing
of the same case, or an irrelevant comorbidity. This pipeline measures that
directly.

For each base case with an unambiguous guideline answer, we apply two kinds
of perturbations:

- **Irrelevant** — should not change the recommendation (insurance, race,
  phrasing, irrelevant comorbidities, distractor findings)
- **Relevant** — should change the recommendation (age crossing an omission
  threshold, new nodal involvement)

The headline metric is a **calibration ratio** per model — how often it flips
when it should, divided by how often it flips when it shouldn't. A well-
calibrated model has a ratio much greater than 1.

## Pipeline overview

```
generator.py           → synthetic cases + guideline-derived reference answers
base_cases.py          → filter to unambiguous cases, stratify across scenarios
perturbations.py       → apply irrelevant + relevant perturbations per case
evaluate.py            → run each perturbation through each model, 3 reps
analyze.py             → compute flip rates, calibration ratios, breakdowns
```

All scripts are standalone Python, no external framework dependencies
beyond the API client libraries.

## Quickstart

### 1. Generate base cases

```bash
python base_cases.py --n 50 --seed 2026 --output base_cases.jsonl
```

Produces 50 base cases across 6 pre-registered clinical strata, each tagged
with which of the 5 decisions are auditable.

### 2. Generate perturbations

```bash
python perturbations.py --base base_cases.jsonl --output perturbations.jsonl
```

Produces ~705 perturbations (1 baseline + ~13 perturbations per case).

### 3. Dry-run the evaluation

```bash
python evaluate.py --perturbations perturbations.jsonl --output responses.jsonl
```

Without `--execute`, this only prints the plan and estimated cost. Review
the prompt template and cost estimate before committing.

### 4. Run the evaluation

Set API keys:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
```

Test with a small subset first:

```bash
python evaluate.py --limit 5 --repetitions 1 --execute
```

When that looks right, run the full audit:

```bash
python evaluate.py --execute --repetitions 3
```

The script checkpoints after every API call and can be resumed. Estimated
runtime on a single machine: 6–12 hours depending on rate limits. Estimated
cost: $200–600 at April 2026 pricing.

### 5. Analyze

```bash
python analyze.py --responses responses.jsonl --output analysis_summary.json
```

Prints a text report to stdout and saves the full summary as JSON.

## Pre-registration

The protocol is in `PREREGISTRATION.md`. Time-stamp this file (OSF or a
dated public GitHub commit) **before running any real API calls**. This is
the single most important thing for the paper's credibility.

## Dependencies

Install as needed:

```bash
pip install anthropic openai google-genai
```

No other hard dependencies. Python 3.10+.

## Files

| File | Purpose |
|------|---------|
| `generator.py`        | Guideline-anchored case generator |
| `base_cases.py`       | Base case filter and stratification |
| `perturbations.py`    | Perturbation engine |
| `evaluate.py`         | Model evaluation harness |
| `analyze.py`          | Flip-rate analysis and reporting |
| `PREREGISTRATION.md`  | Pre-registration protocol |
| `README.md`           | This file |

## Customizing for a different tumor site

The architecture is portable. To audit a different site, replace or extend:

1. `generator.py` — the `BreastCancerCase` dataclass, `sample_case`, and
   `apply_guidelines` functions
2. `base_cases.py` — the ambiguity filters and stratification
3. `perturbations.py` — the `R1_AGE_THRESHOLD` and `R2_NODAL_STATUS` rules
   (the C1–C4 irrelevant perturbations mostly transfer unchanged)

`evaluate.py` and `analyze.py` are site-agnostic.

## Citation

If you use this framework, please cite the protocol pre-registration and
the final paper (to be inserted).

## License

MIT.
