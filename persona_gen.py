from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


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


def _validate_persona(obj: Any) -> Optional[dict[str, Any]]:
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
    if belief is None or susceptibility is None:
        return None

    return {
        "name": name.strip(),
        "bio": bio.strip(),
        "initial_belief": belief,
        "susceptibility": susceptibility,
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
You are generating initial agents for a social simulation.

Return STRICT JSON ONLY (no markdown, no commentary). The JSON must validate against the provided schema.

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

Hard constraints:
- Output EXACTLY {n} personas in the "personas" array.
- Every persona MUST have a unique "name".
- Bios should be distinct; do not reuse the same bio template.
- initial_belief MUST be within [-1, 1] (inclusive).
- susceptibility MUST be > 0 and <= 1.
- Keep each bio concise (1-3 sentences) and realistic, and vary demographic details (age range, region, occupation).

Generate {n} personas now.
""".strip()


def _personas_json_schema() -> dict[str, Any]:
    # Note: JSON Schema can't enforce "exactly N items" (since N is dynamic),
    # so we validate count in code.
    return {
        "type": "object",
        "properties": {
            "personas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "bio": {"type": "string"},
                        "initial_belief": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                        "susceptibility": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["name", "bio", "initial_belief", "susceptibility"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["personas"],
        "additionalProperties": False,
    }


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
            response_json_schema=_personas_json_schema(),
        ),
    )
    parsed = json.loads(resp.text)
    if not isinstance(parsed, dict) or "personas" not in parsed:
        raise ValueError('Model returned JSON without top-level key "personas".')

    personas_raw = parsed["personas"]
    if not isinstance(personas_raw, list):
        raise ValueError('"personas" must be a JSON array.')
    if len(personas_raw) != n:
        raise ValueError(f"Expected exactly {n} personas, got {len(personas_raw)}.")

    personas: list[dict[str, Any]] = []
    for item in personas_raw:
        p = _validate_persona(item)
        if p is None:
            raise ValueError("One or more personas failed validation.")
        personas.append(p)

    seen_names: set[str] = set()
    for p in personas:
        name = p["name"]
        if name in seen_names:
            raise ValueError(f'Duplicate persona name generated: "{name}".')
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

    args = p.parse_args(argv)
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