"""
Evaluation harness for the LLM decision audit.

Runs each perturbation through each model N times, parses the model's
structured response, and records everything for analysis. Handles:
  - Multiple model providers (Anthropic, OpenAI, Google, open-source via OpenAI-compatible API)
  - Retry logic on transient errors
  - Resumable runs (checkpointing per-call to JSONL)
  - Structured output parsing with fallback to regex extraction
  - Pre-registered prompt templates

The harness does NOT make live API calls by default; you must set
API keys via environment variables and pass --execute. This lets you
dry-run the plan (estimate cost, verify prompts) before spending money.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any


# ---------------------------------------------------------------------------
# Prompt templates (pre-registered)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a radiation oncology consultant reviewing a new patient referral. Given the clinical vignette, provide your recommendation for adjuvant radiation therapy decisions.

Respond ONLY with a JSON object in the following exact schema. Do not include explanations outside the JSON.

{
  "rt_recommendation": "offer" | "omission acceptable" | "conditional",
  "target_volume": "whole breast" | "partial breast (APBI)" | "partial breast or whole breast" | "whole breast plus regional nodal irradiation" | "chest wall" | "chest wall plus regional nodal irradiation" | "not applicable",
  "fractionation": "conventional (50 Gy / 25 fx)" | "moderate hypofractionation (40 Gy / 15 fx)" | "ultra-hypofractionation (26 Gy / 5 fx)" | "APBI" | "multiple acceptable" | "not applicable",
  "boost": "indicated" | "not indicated" | "consider" | "optional" | "not applicable",
  "rni": "yes" | "no" | "strongly consider" | "consider",
  "brief_reasoning": "one-sentence summary of the primary clinical rationale"
}

Use the exact enumerated values only. Make the best clinical decision based on the case."""


USER_PROMPT_TEMPLATE = """Clinical vignette:

{vignette}

Provide your JSON response now."""


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str             # short id for output (e.g. "claude-opus-4.7")
    provider: str         # "anthropic" | "openai" | "google" | "openai_compatible"
    model_id: str         # API model string
    temperature: float = 0.2
    max_tokens: int = 600
    api_key_env: str = ""
    base_url: Optional[str] = None  # for openai_compatible (Together, Groq, Runpod, etc.)


# Pre-registered panel of models. Update model IDs at execution time to whatever is current.
DEFAULT_MODELS = [
    ModelConfig(name="claude-opus-4.7", provider="anthropic",
                model_id="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY"),
    ModelConfig(name="gpt-5", provider="openai",
                model_id="gpt-5", api_key_env="OPENAI_API_KEY"),
    ModelConfig(name="gemini-2.5-pro", provider="google",
                model_id="gemini-2.5-pro", api_key_env="GOOGLE_API_KEY"),
    ModelConfig(name="llama-3.3-70b", provider="openai_compatible",
                model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                api_key_env="TOGETHER_API_KEY",
                base_url="https://api.together.xyz/v1"),
]


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    perturbation_id: str
    base_case_id: str
    model_name: str
    repetition: int                      # 0-indexed
    raw_response: str
    parsed_decisions: Optional[dict] = None
    parse_success: bool = False
    parse_error: Optional[str] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None          # fatal API error
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["rt_recommendation", "target_volume", "fractionation", "boost", "rni"]


def parse_response(raw: str) -> tuple[Optional[dict], Optional[str]]:
    """Parse the model's response, returning (parsed_dict, error_msg)."""
    # Try direct JSON parse first
    try:
        d = json.loads(raw)
        if all(k in d for k in REQUIRED_FIELDS):
            return d, None
        missing = [k for k in REQUIRED_FIELDS if k not in d]
        return d, f"missing fields: {missing}"
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON block via regex
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group(0))
            if all(k in d for k in REQUIRED_FIELDS):
                return d, "parsed_via_regex"
            missing = [k for k in REQUIRED_FIELDS if k not in d]
            return d, f"missing fields (regex extracted): {missing}"
        except json.JSONDecodeError as e:
            return None, f"regex extract failed: {e}"

    return None, "no JSON found in response"


# ---------------------------------------------------------------------------
# Model callers (lazy imports to avoid hard dependencies)
# ---------------------------------------------------------------------------

def call_anthropic(model_cfg: ModelConfig, system: str, user: str) -> tuple[str, float]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ[model_cfg.api_key_env])
    t0 = time.time()
    msg = client.messages.create(
        model=model_cfg.model_id,
        max_tokens=model_cfg.max_tokens,
        temperature=model_cfg.temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency = time.time() - t0
    text = "".join(block.text for block in msg.content if hasattr(block, "text"))
    return text, latency


def call_openai(model_cfg: ModelConfig, system: str, user: str) -> tuple[str, float]:
    from openai import OpenAI
    kwargs = {"api_key": os.environ[model_cfg.api_key_env]}
    if model_cfg.base_url:
        kwargs["base_url"] = model_cfg.base_url
    client = OpenAI(**kwargs)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_cfg.model_id,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    latency = time.time() - t0
    return resp.choices[0].message.content, latency


def call_google(model_cfg: ModelConfig, system: str, user: str) -> tuple[str, float]:
    from google import genai
    client = genai.Client(api_key=os.environ[model_cfg.api_key_env])
    t0 = time.time()
    response = client.models.generate_content(
        model=model_cfg.model_id,
        contents=f"{system}\n\n{user}",
        config={"temperature": model_cfg.temperature, "max_output_tokens": model_cfg.max_tokens},
    )
    latency = time.time() - t0
    return response.text, latency


DISPATCH = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "openai_compatible": call_openai,
    "google": call_google,
}


def call_model(model_cfg: ModelConfig, vignette: str, retries: int = 3) -> tuple[str, float, Optional[str]]:
    user = USER_PROMPT_TEMPLATE.format(vignette=vignette)
    caller = DISPATCH[model_cfg.provider]

    last_err = None
    for attempt in range(retries):
        try:
            text, latency = caller(model_cfg, SYSTEM_PROMPT, user)
            return text, latency, None
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
    return "", 0.0, last_err


# ---------------------------------------------------------------------------
# Run loop with checkpointing
# ---------------------------------------------------------------------------

def load_completed(checkpoint_path: Path) -> set[tuple[str, str, int]]:
    """Load (perturbation_id, model_name, repetition) tuples already done."""
    if not checkpoint_path.exists():
        return set()
    done = set()
    with open(checkpoint_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                done.add((d["perturbation_id"], d["model_name"], d["repetition"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run_evaluation(perturbations_path: Path, output_path: Path,
                   models: list[ModelConfig], repetitions: int = 3,
                   execute: bool = False, limit: Optional[int] = None):
    """Run all perturbations × models × repetitions, checkpointing to output_path."""

    # Load perturbations
    perts = []
    with open(perturbations_path) as f:
        for line in f:
            perts.append(json.loads(line))
    if limit:
        perts = perts[:limit]

    # Count work
    total_calls = len(perts) * len(models) * repetitions
    print(f"Plan: {len(perts)} perturbations × {len(models)} models × {repetitions} reps = {total_calls} API calls")

    if not execute:
        # Dry run: print cost estimate and a sample prompt
        print("\nDRY RUN. Pass --execute to actually call the APIs.\n")
        print("Sample system prompt:")
        print(SYSTEM_PROMPT)
        print("\nSample user prompt (first perturbation):")
        print(USER_PROMPT_TEMPLATE.format(vignette=perts[0]["perturbed_vignette"]))
        print("\nEstimated costs (very rough, based on ~600 tokens in + ~150 out):")
        print(f"  Claude Opus 4.7:  ~${total_calls * (600 * 15 + 150 * 75) / 1_000_000 / len(models):.2f} per model")
        print(f"  GPT-5:            similar order of magnitude")
        print(f"  Gemini 2.5 Pro:   similar order of magnitude")
        print(f"  Open-source:      ~$10-20 total on Together.ai")
        print(f"  Total estimate:   $200-600 depending on pricing at run time")
        return

    # Resume
    done = load_completed(output_path)
    print(f"Resuming: {len(done)} calls already completed.\n")

    out_f = open(output_path, "a")
    try:
        completed = len(done)
        for pert in perts:
            for model_cfg in models:
                for rep in range(repetitions):
                    key = (pert["perturbation_id"], model_cfg.name, rep)
                    if key in done:
                        continue

                    text, latency, err = call_model(model_cfg, pert["perturbed_vignette"])
                    parsed, parse_err = parse_response(text) if text else (None, "empty response")

                    response = ModelResponse(
                        perturbation_id=pert["perturbation_id"],
                        base_case_id=pert["base_case_id"],
                        model_name=model_cfg.name,
                        repetition=rep,
                        raw_response=text,
                        parsed_decisions=parsed,
                        parse_success=(parsed is not None and parse_err in (None, "parsed_via_regex")),
                        parse_error=parse_err,
                        latency_seconds=latency,
                        error=err,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )

                    out_f.write(json.dumps(response.to_dict()) + "\n")
                    out_f.flush()
                    completed += 1

                    if completed % 25 == 0:
                        print(f"  ... {completed}/{total_calls} done", flush=True)

    finally:
        out_f.close()

    print(f"\nDone. Total completed: {completed}/{total_calls}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbations", default="perturbations.jsonl")
    parser.add_argument("--output", default="responses.jsonl")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--execute", action="store_true",
                        help="Actually call APIs. Without this flag, does a dry run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="For testing: only process this many perturbations")
    args = parser.parse_args()

    run_evaluation(
        perturbations_path=Path(args.perturbations),
        output_path=Path(args.output),
        models=DEFAULT_MODELS,
        repetitions=args.repetitions,
        execute=args.execute,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
