#!/usr/bin/env python3
"""
CLI front-end for the structured persona generation pipeline.

Usage
-----

    # Stage 1+2 — pure math, no LLM. Produces a fully valid population.
    python generate_population.py sample --n 200 --seed 42 --out population.json

    # Stage 3 — LLM bios only. Reads an existing JSON, fills bio fields.
    python generate_population.py bios --input population.json --model gpt-4o-mini

    # Validation diagnostics (range checks + summary stats).
    python generate_population.py validate --input population.json

    # Smoke test: load via load_population and print the first 3 agents.
    python generate_population.py load-check --input population.json

    # Legacy one-shot LLM persona generator (deprecated; gated for parity).
    python generate_population.py legacy --n 500 --out agents_500_pro_max.json

The four primary subcommands are independent. Running `bios` without
having first run `sample` errors clearly. The legacy command is kept
for short-term comparison runs only and prints a deprecation notice.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

# Allow `python generate_population.py ...` from anywhere.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persona_pipeline import (  # noqa: E402
    PIPELINE_STAGE_BIOS,
    PIPELINE_STAGE_PARAMETERS,
    LLMClient,
    generate_bios,
    generate_population_json,
    load_population,
    validate_population,
)

# ---------------------------------------------------------------------------
# LLM provider adapters
#
# generate_bios takes a Callable[[prompt, model], str]. We provide thin
# adapters for OpenAI and Gemini so the CLI stays provider-agnostic.
# Both are lazy-imported so the CLI runs on machines without either SDK.
# ---------------------------------------------------------------------------


def _build_openai_client(api_key: str) -> LLMClient:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover — exercised only with provider=openai
        raise RuntimeError(
            "Missing dependency `openai`. Install it with `pip install openai`."
        ) from e

    client = OpenAI(api_key=api_key)

    def call(prompt: str, model: str) -> str:
        resp = client.responses.create(  # type: ignore[attr-defined]
            model=model,
            input=prompt,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            raise RuntimeError(f"OpenAI returned empty output for model {model}")
        return text

    return call


def _build_gemini_client(api_key: str) -> LLMClient:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as e:  # pragma: no cover — exercised only with provider=gemini
        raise RuntimeError(
            "Missing dependency `google-genai`. Install it with `pip install google-genai`."
        ) from e

    client = genai.Client(api_key=api_key)

    def call(prompt: str, model: str) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7),
        )
        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError(f"Gemini returned empty output for model {model}")
        return text

    return call


def _build_llm_client(provider: str, api_key: str) -> LLMClient:
    if provider == "openai":
        return _build_openai_client(api_key)
    if provider == "gemini":
        return _build_gemini_client(api_key)
    raise ValueError(f"unknown LLM provider: {provider!r}")


def _resolve_api_key(provider: str, explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    env_var = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    val = os.environ.get(env_var, "").strip()
    if not val:
        # RuntimeError (not SystemExit) so the top-level `main()` exception
        # handler converts it to a clean exit code 1 with a printed message.
        raise RuntimeError(
            f"No {provider} API key. Pass --api-key or set {env_var}."
        )
    return val


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_sample(args: argparse.Namespace) -> int:
    out = generate_population_json(n=args.n, seed=args.seed, output_path=args.out)
    print(
        f"Wrote {args.n} agents to {out} "
        f"(stage={PIPELINE_STAGE_PARAMETERS}, no LLM used)"
    )
    return 0


def _cmd_bios(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args.provider, args.api_key)
    client = _build_llm_client(args.provider, api_key)
    result = generate_bios(
        json_path=args.input,
        llm_client=client,
        model=args.model,
        delay_between_calls=args.delay,
        max_retries=args.max_retries,
    )
    print(
        f"Bios filled: {result['filled']}, skipped: {result['skipped']}"
    )
    if result["skipped_ids"]:
        print("Skipped agent ids:")
        for aid in result["skipped_ids"]:
            print(f"  - {aid}")
    return 0 if result["skipped"] == 0 else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    validate_population(args.input)
    return 0


def _cmd_load_check(args: argparse.Namespace) -> int:
    agents = load_population(args.input)
    print(f"Loaded {len(agents)} agents from {args.input}.")
    print("First 3 agents (engine-shaped dicts):")
    for a in agents[:3]:
        # Keep the print compact — show key engine fields only.
        compact = {
            k: a[k]
            for k in (
                "id", "name", "initial_belief", "susceptibility",
                "activity", "confidence_epsilon",
                "selective_exposure_beta", "repulsion_threshold_rho",
                "repulsion_strength_gamma",
            )
            if k in a
        }
        print(f"  {compact}")
    return 0


def _cmd_legacy(args: argparse.Namespace) -> int:
    """Run the old single-shot LLM persona generator (deprecated)."""
    print(
        "DEPRECATION: `legacy` invokes the pre-pipeline persona generator. "
        "Prefer `sample` + `bios` for reproducible runs.",
        file=sys.stderr,
    )
    # Local import so persona_gen's optional Gemini dependency does not
    # affect non-legacy runs.
    from persona_gen import generate_agents

    api_key = _resolve_api_key("gemini", args.api_key)
    generate_agents(
        n=args.n,
        out_path=args.out,
        model=args.model,
        api_key=api_key,
        seed=args.seed,
        temperature=args.temperature,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
    )
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_population.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # sample
    s = sub.add_parser(
        "sample",
        help="Run stages 1+2 (sampling + mapping). No LLM call.",
    )
    s.add_argument("--n", type=int, required=True, help="Population size.")
    s.add_argument("--seed", type=int, required=True, help="RNG seed for LHS.")
    s.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    s.set_defaults(func=_cmd_sample)

    # bios
    b = sub.add_parser(
        "bios",
        help="Run stage 3 (LLM bio generation). Reads --input, fills bios.",
    )
    b.add_argument("--input", type=Path, required=True, help="Pipeline JSON.")
    b.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model name (e.g. gpt-4o-mini, gemini-1.5-flash).",
    )
    b.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=("openai", "gemini"),
        help="LLM provider (default: openai).",
    )
    b.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Override OPENAI_API_KEY / GEMINI_API_KEY env var.",
    )
    b.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between successful LLM calls (default 0.5).",
    )
    b.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Per-agent retry budget on LLM error (default 3).",
    )
    b.set_defaults(func=_cmd_bios)

    # validate
    v = sub.add_parser(
        "validate",
        help="Run validation diagnostics on a pipeline JSON.",
    )
    v.add_argument("--input", type=Path, required=True, help="Pipeline JSON.")
    v.set_defaults(func=_cmd_validate)

    # load-check
    l = sub.add_parser(
        "load-check",
        help="Smoke-test load_population: print the first 3 engine-shaped agents.",
    )
    l.add_argument("--input", type=Path, required=True, help="Pipeline JSON.")
    l.set_defaults(func=_cmd_load_check)

    # legacy
    g = sub.add_parser(
        "legacy",
        help="Run the deprecated one-shot LLM persona generator.",
    )
    g.add_argument("--n", type=int, default=500, help="Number of personas.")
    g.add_argument(
        "--out", type=Path, default=Path("agents_legacy.json"), help="Output JSON."
    )
    g.add_argument("--model", type=str, default="gemini-3-flash-preview")
    g.add_argument("--api-key", type=str, default="")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--max-retries", type=int, default=4)
    g.add_argument("--max-output-tokens", type=int, default=8192)
    g.set_defaults(func=_cmd_legacy)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
