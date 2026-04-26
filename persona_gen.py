"""
DEPRECATED legacy persona generator.

This module produced agents by asking an LLM to invent both the bio AND the
numerical fields (initial_belief, susceptibility) in a single call. That
violates the architectural invariant of the new pipeline:

    The LLM is never allowed to produce numerical values.

The replacement is the three-stage pipeline in `backend.persona_pipeline`,
driven via `generate_population.py`:

    python generate_population.py sample --n 500 --seed 0 --out population.json
    python generate_population.py bios --input population.json --model gpt-4o-mini

This file is kept for short-term comparison runs only and will be removed
once the new pipeline is fully validated. New code should not import from
here.

Setting `USE_LEGACY_PERSONA_GENERATION = True` and running this module
(or `generate_population.py legacy`) re-enables the old behaviour.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional


# Master toggle. Defaults to False so accidental imports don't quietly
# resurrect the old code path. Set True to allow the CLI to run.
USE_LEGACY_PERSONA_GENERATION: bool = False


DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_INITIAL_BELIEF = 0.0
DEFAULT_SUSCEPTIBILITY = 0.5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def _ensure_susceptibility(x: Any) -> Optional[float]:
    v = _coerce_float(x)
    if v is None or math.isnan(v) or math.isinf(v):
        return None
    if v <= 0:
        return None
    return _clamp(v, 1e-6, 1.0)


def _ensure_belief(x: Any) -> Optional[float]:
    v = _coerce_float(x)
    if v is None or math.isnan(v) or math.isinf(v):
        return None
    return _clamp(v, -1.0, 1.0)


def _validate_persona_relaxed(obj: Any) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    name = obj.get("name")
    bio = obj.get("bio")
    belief = _ensure_belief(obj.get("initial_belief"))
    susceptibility = _ensure_susceptibility(obj.get("susceptibility"))

    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(bio, str) or not bio.strip():
        return None

    return {
        "name": name.strip(),
        "bio": bio.strip(),
        "initial_belief": belief if belief is not None else DEFAULT_INITIAL_BELIEF,
        "susceptibility": susceptibility if susceptibility is not None else DEFAULT_SUSCEPTIBILITY,
    }


def _build_gemini_client(api_key: str):
    try:
        from google import genai  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `google-genai`. Install it with `pip install google-genai`."
        ) from e

    return genai.Client(api_key=api_key)


def _personas_prompt(n: int) -> str:
    return f"""
You are generating random initial agents for a social simulation. The agents should be diverse and realistic.

Return JSON only (no markdown, no commentary). Prefer this shape:

Top-level schema:
{{
  "personas": [ Persona, ... ]
}}

Persona schema:
{{
  "name": string,
  "bio": string,
  "initial_belief": number in [-1, 1],
  "susceptibility": number in (0, 1]
}}

Guidelines:
- You MUST generate {n} personas in the "personas" array.
- Every persona MUST have a unique "name".
- Bios should be distinct; do not reuse the same bio template.
- Keep numeric fields within bounds, generate to the thousandth decimal place.
- Keep each bio concise (1-3 sentences) and realistic, and vary demographic details (age range, region, occupation).

Generate {n} personas now.
""".strip()


def _extract_json(text: str) -> Any:
    """
    Gemini usually returns JSON when response_mime_type is application/json, but in practice
    we may still see leading/trailing text. This extracts the first JSON object/array.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    try:
        return json.loads(text)
    except Exception:
        pass

    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if not starts:
        raise ValueError("Model response did not contain JSON.")

    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_str = False
    esc = False
    end: Optional[int] = None

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError("Model response contained JSON start but no balanced end.")

    candidate = text[start:end]
    return json.loads(candidate)


def _generate_personas_gemini(
    client,
    *,
    model: str,
    n: int,
    temperature: float,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    msg = _personas_prompt(n)
    try:
        from google.genai import types  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `google-genai`. Install it with `pip install google-genai`."
        ) from e

    resp = client.models.generate_content(
        model=model,
        contents=msg,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    parsed = _extract_json(getattr(resp, "text", "") or "")
    personas_raw: Any
    if isinstance(parsed, dict):
        if "personas" in parsed:
            personas_raw = parsed["personas"]
        elif "agents" in parsed:
            personas_raw = parsed["agents"]
        else:
            personas_raw = None
    elif isinstance(parsed, list):
        personas_raw = parsed
    else:
        personas_raw = None

    if not isinstance(personas_raw, list):
        raise ValueError('Model returned JSON without a "personas" array (or array root).')

    personas: list[dict[str, Any]] = []
    for i, item in enumerate(personas_raw):
        p = _validate_persona_relaxed(item)
        if p is None:
            # Very relaxed fallback: keep generation moving.
            p = {
                "name": f"Agent {i}",
                "bio": "A participant in the simulation.",
                "initial_belief": DEFAULT_INITIAL_BELIEF,
                "susceptibility": DEFAULT_SUSCEPTIBILITY,
            }
        personas.append(p)

    # Ensure we return exactly n personas: trim or pad with simple fallbacks.
    if len(personas) > n:
        personas = personas[:n]
    while len(personas) < n:
        i = len(personas)
        personas.append(
            {
                "name": f"Agent {i}",
                "bio": "A participant in the simulation.",
                "initial_belief": DEFAULT_INITIAL_BELIEF,
                "susceptibility": DEFAULT_SUSCEPTIBILITY,
            }
        )

    # De-dupe names deterministically.
    seen_names: set[str] = set()
    for i, p in enumerate(personas):
        base = (p.get("name") or f"Agent {i}").strip() or f"Agent {i}"
        name = base
        if name in seen_names:
            suffix = 2
            while f"{base} ({suffix})" in seen_names:
                suffix += 1
            name = f"{base} ({suffix})"
        p["name"] = name
        seen_names.add(name)

    return personas


def generate_agents(
    *,
    n: int,
    out_path: Path,
    model: str,
    api_key: str,
    seed: Optional[int],
    temperature: float,
    max_retries: int,
    max_output_tokens: int,
) -> None:
    if n <= 0:
        raise ValueError("--n must be > 0")

    if seed is not None:
        # Gemini doesn't guarantee deterministic sampling across calls, but we keep this
        # for parity with the CLI and potential future use.
        pass

    client = _build_gemini_client(api_key)

    personas: Optional[list[dict[str, Any]]] = None
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            personas = _generate_personas_gemini(
                client,
                model=model,
                n=n,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            break
        except Exception as e:
            last_err = e

    if personas is None:
        raise RuntimeError(
            f"Failed to generate {n} personas after {max_retries} attempts: "
            f"{type(last_err).__name__ if last_err else 'UnknownError'}"
        )

    agents: list[dict[str, Any]] = [{"id": i, **p} for i, p in enumerate(personas)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(agents)} personas to {out_path}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Generate EchoChamber personas JSON via Gemini API.")
    p.add_argument("--n", type=int, default=500, help="Number of personas to generate (default: 500).")
    p.add_argument("--out", type=Path, default=Path("agents.json"), help="Output JSON path (default: agents.json).")
    p.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL, help=f'Gemini model (default: "{DEFAULT_GEMINI_MODEL}").')
    p.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Gemini API key. If omitted, reads from GEMINI_API_KEY env var.",
    )
    p.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility (default: 1337).")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7).")
    p.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max LLM retries if output is invalid (default: 4).",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Max output tokens for the Gemini response (default: 8192).",
    )

    p.add_argument(
        "--allow-legacy",
        action="store_true",
        help=(
            "Acknowledge that this is the deprecated one-shot generator and run "
            "anyway. Without this flag the CLI exits with an instruction to use "
            "the new pipeline."
        ),
    )

    args = p.parse_args(argv)

    if not (USE_LEGACY_PERSONA_GENERATION or args.allow_legacy):
        print(
            "persona_gen.py is deprecated. Use the structured pipeline instead:\n"
            "  python generate_population.py sample --n 500 --seed 0 --out population.json\n"
            "  python generate_population.py bios --input population.json --model gpt-4o-mini\n\n"
            "If you really want the legacy behaviour, pass --allow-legacy or set "
            "USE_LEGACY_PERSONA_GENERATION=True at the top of persona_gen.py.",
            file=sys.stderr,
        )
        return 2

    warnings.warn(
        "persona_gen.py is deprecated; use generate_population.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        api_key = (args.api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("No API key found. Provide --api-key or set GEMINI_API_KEY.")

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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))