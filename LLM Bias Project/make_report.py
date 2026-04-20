"""
Publication-ready report generator.

Reads analysis_summary.json and produces markdown tables formatted for
direct inclusion in a manuscript. Generates:
  - Table 1: per-model primary outcomes (inappropriate flip rate,
             appropriate flip rate, calibration ratio)
  - Table 2: per-category inappropriate flip rates by model
  - Table 3: per-decision-task inappropriate flip rates by model
  - Table 4: top-N highest-flip axes across models
  - Appendix: full per-axis flip rate tables per model

Designed to be drop-in usable in a methods paper.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fmt_pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def fmt_ci(low: float, high: float) -> str:
    return f"[{100 * low:.1f}, {100 * high:.1f}]"


def fmt_ratio(r: float) -> str:
    if isinstance(r, str):
        return r
    if math.isinf(r):
        return "∞"
    if math.isnan(r):
        return "undefined"
    return f"{r:.2f}"


def table1_primary(summary: dict, test_retest: dict) -> str:
    """Per-model primary outcomes."""
    lines = [
        "### Table 1. Primary outcomes by model",
        "",
        "| Model | Inappropriate flip rate (95% CI) | Appropriate flip rate (95% CI) | Calibration ratio | Baseline stability |",
        "|---|---|---|---|---|",
    ]
    models = sorted(summary["per_model_irrelevant"].keys())
    for m in models:
        irrel = summary["per_model_irrelevant"][m]
        rel = summary["per_model_relevant"][m]
        cr = summary["per_model_calibration_ratio"][m]
        # Handle possible string values from JSON deserialization
        if isinstance(cr, str):
            try:
                cr = float(cr)
            except ValueError:
                pass
        stab = test_retest.get(m, {}).get("unanimous_agreement_rate", 0)
        lines.append(
            f"| {m} | {fmt_pct(irrel['inappropriate_flip_rate'])} {fmt_ci(irrel['ci_low'], irrel['ci_high'])} "
            f"| {fmt_pct(rel['appropriate_flip_rate'])} {fmt_ci(rel['ci_low'], rel['ci_high'])} "
            f"| {fmt_ratio(cr)} "
            f"| {fmt_pct(stab)} |"
        )
    lines.append("")
    return "\n".join(lines)


def table2_categories(summary: dict) -> str:
    """Per-category inappropriate flip rates."""
    models = sorted(summary["per_model_per_category_irrelevant"].keys())
    all_categories = set()
    for m in models:
        all_categories.update(summary["per_model_per_category_irrelevant"][m].keys())
    cats = sorted(c for c in all_categories if c != "BASELINE")

    lines = [
        "### Table 2. Inappropriate flip rate by perturbation category",
        "",
        "Each cell shows the flip rate for that model on perturbations of that category.",
        "All values are proportions; lower is better. Irrelevant perturbations only.",
        "",
        "| Model | " + " | ".join(cats) + " |",
        "|---|" + "---|" * len(cats),
    ]
    for m in models:
        row = [m]
        by_cat = summary["per_model_per_category_irrelevant"][m]
        for c in cats:
            if c in by_cat:
                row.append(fmt_pct(by_cat[c]["flip_rate"]))
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def table3_tasks(summary: dict) -> str:
    """Per-decision-task inappropriate flip rates."""
    models = sorted(summary["per_model_per_task_irrelevant"].keys())
    tasks = ["rt_recommendation", "target_volume", "fractionation", "boost", "rni"]

    lines = [
        "### Table 3. Inappropriate flip rate by decision task",
        "",
        "| Model | " + " | ".join(tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    for m in models:
        row = [m]
        by_task = summary["per_model_per_task_irrelevant"][m]
        for t in tasks:
            if t in by_task and by_task[t]["n"] > 0:
                row.append(fmt_pct(by_task[t]["flip_rate"]))
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def table4_top_axes(summary: dict, n: int = 15) -> str:
    """Top N highest-flip axes across models."""
    rows = []
    for model, by_axis in summary["per_model_per_axis_irrelevant"].items():
        for axis, d in by_axis.items():
            rows.append((model, axis, d["flip_rate"], d["n"]))
    rows.sort(key=lambda x: -x[2])

    lines = [
        f"### Table 4. Top {n} highest-flip perturbation axes across models",
        "",
        "| Model | Perturbation axis | Flip rate | n |",
        "|---|---|---|---|",
    ]
    for model, axis, rate, nn in rows[:n]:
        lines.append(f"| {model} | {axis} | {fmt_pct(rate)} | {nn} |")
    lines.append("")
    return "\n".join(lines)


def appendix_per_model_axes(summary: dict) -> str:
    """Full per-axis flip rate tables, one per model."""
    lines = ["### Appendix A. Per-axis inappropriate flip rate, by model", ""]
    for model in sorted(summary["per_model_per_axis_irrelevant"].keys()):
        lines.append(f"#### {model}")
        lines.append("")
        lines.append("| Axis | Flip rate (95% CI) | n |")
        lines.append("|---|---|---|")
        by_axis = summary["per_model_per_axis_irrelevant"][model]
        sorted_axes = sorted(by_axis.items(), key=lambda kv: -kv[1]["flip_rate"])
        for axis, d in sorted_axes:
            lines.append(
                f"| {axis} | {fmt_pct(d['flip_rate'])} {fmt_ci(d['ci_low'], d['ci_high'])} | {d['n']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="analysis_summary.json")
    parser.add_argument("--output", default="report.md")
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)
    summary = data["flip_summary"]
    test_retest = data["test_retest"]

    sections = [
        "# Perturbation Audit Results",
        "",
        table1_primary(summary, test_retest),
        table2_categories(summary),
        table3_tasks(summary),
        table4_top_axes(summary),
        appendix_per_model_axes(summary),
    ]

    report = "\n".join(sections)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Wrote report → {args.output}")
    print()
    print(report)


if __name__ == "__main__":
    main()
