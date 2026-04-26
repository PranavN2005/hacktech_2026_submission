"""
Three-stage persona generation pipeline for the opinion-dynamics simulation.

    Stage 1 — sample          (pure math, no LLM)
    Stage 2 — map to params   (pure math, no LLM)
    Stage 3 — narrate (bio)   (LLM-only; numerical fields are read-only)

The architectural invariant for this module:

    The LLM is never allowed to produce numerical values.

Stages 1 and 2 produce a complete, valid agent population. Stage 3 is
purely cosmetic — if it never runs, the simulation can still load and
execute the population.

Public API
==========
    sample_trait_vectors(n, seed)      → (n, 6) ndarray
    traits_to_parameters(trait_vector) → dict  (one agent's params)
    generate_population_json(n, seed, output_path)
    generate_bios(json_path, llm_client, model, ...)
    load_population(json_path)         → list[dict] in engine format
    validate_population(json_path)     → dict of summary stats (raises on errors)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import beta as scipy_beta
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

from backend.persona_config import (
    AXIS_KEYS,
    CORRELATION_MATRIX,
    LABEL_FUNCTIONS,
    MARGINAL_DISTRIBUTIONS,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — quasi-random correlated population sampling
# ---------------------------------------------------------------------------

# Numerical guard for the inverse normal CDF — Φ⁻¹(0) = -∞ and Φ⁻¹(1) = +∞.
_LHS_CLIP_LO: float = 1e-6
_LHS_CLIP_HI: float = 1.0 - _LHS_CLIP_LO


def _lhs_uniform(*, n: int, dim: int, seed: int) -> np.ndarray:
    """
    Latin Hypercube sample of `n` points in [0, 1]^dim with a fixed seed.

    Each axis is partitioned into `n` strata of width 1/n; exactly one
    sample lands in each stratum per axis. This gives uniform marginals
    with guaranteed coverage of fringe values, which is critical for the
    polarisation dynamics (rare extremists must always be present).

    Returned values are strictly in (0, 1), so they can be passed to the
    inverse normal CDF without clipping (clipping is still applied later
    after the copula step for safety).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    sampler = LatinHypercube(d=dim, seed=seed)
    return sampler.random(n=n)


def sample_trait_vectors(n: int, seed: int) -> np.ndarray:
    """
    Draw `n` correlated trait vectors in [0, 1]^6.

    Algorithm (see module docstring + spec for full justification):

        1. Latin Hypercube sample u ~ U([0, 1]^6) — guarantees stratification.
        2. Gaussian copula:
             z       = Φ⁻¹(clip(u))
             z_corr  = z @ L.T               where  L L.T = Σ
             u_corr  = Φ(z_corr)
        3. Apply each axis's Beta inverse CDF to u_corr column-wise.

    The output column order matches `AXIS_KEYS` exactly:
        [lr, auth, conf, open, react, active].

    Returns
    -------
    ndarray of shape (n, 6) with values in [0, 1].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    dim = len(AXIS_KEYS)
    u = _lhs_uniform(n=n, dim=dim, seed=seed)

    # Gaussian copula: clip → ppf → correlate → cdf.
    u_clipped = np.clip(u, _LHS_CLIP_LO, _LHS_CLIP_HI)
    z = norm.ppf(u_clipped)
    L = np.linalg.cholesky(CORRELATION_MATRIX)
    z_corr = z @ L.T
    u_corr = norm.cdf(z_corr)

    # Apply per-axis Beta inverse CDF.
    out = np.empty_like(u_corr)
    for k, key in enumerate(AXIS_KEYS):
        params = MARGINAL_DISTRIBUTIONS[key]
        out[:, k] = scipy_beta.ppf(u_corr[:, k], params["a"], params["b"])

    # Beta support is [0, 1]; explicit clip guards against ppf returning
    # values microscopically outside the unit interval near the boundaries.
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 2 — deterministic mapping from traits to numerical agent parameters
# ---------------------------------------------------------------------------

# Floors prevent pathological cases (immovable agents, zero activity, etc.)
# that break the dynamics or are physically uninteresting.
_S_FLOOR: float = 0.05
_SIGMA_FLOOR: float = 0.05
_ACTIVITY_FLOOR: float = 0.01

# ε formula (Fix 1 — compressed range, tied to belief extremity):
#
#   extremity       = |2·lr − 1|                       ∈ [0, 1]
#   base_tolerance  = ε_base + ε_range · open · (1 − 0.5·conf)
#                                                       ∈ [0.05, 0.40]
#   ε               = base_tolerance · (1 − ε_ext_damp · extremity)
#   ε               = max(ε, ε_floor)
#
# With median traits the typical value is ≈ 0.19, placing most of the
# population in the Hegselmann-Krause fragmentation / two-cluster zone
# (ε ≈ 0.1–0.35 on a diameter-2 belief axis), compared to the old formula
# which produced a median ε ≈ 0.84 — firmly in the consensus zone.
_EPSILON_BASE_MIN: float = 0.05    # base_tolerance lower bound (open=0)
_EPSILON_BASE_RANGE: float = 0.35  # variable part → ε_max base = 0.40
_EPSILON_EXT_DAMP: float = 0.40   # extremist narrowing factor
_EPSILON_FLOOR: float = 0.05      # hard floor on ε

# Repulsion gain cap (Fix 3).  Raised from 0.5 to 0.8 so repulsion is a
# meaningful counter-force to attraction; with the old cap (γ ≤ 0.5) and
# mean s ≈ 0.41, attraction was 6–7× stronger than repulsion for typical g.
_GAMMA_CAP: float = 0.8

# β (selective exposure strength) max ≈ 5 — empirically a regime where the
# feed becomes effectively a hard filter without numerical overflow in
# softmax-style scoring downstream.
_BETA_RIGIDITY_W: float = 3.0
_BETA_CLOSED_W: float = 2.0

# Rigidity extremity weight (Fix 2): ensures extreme believers always have
# non-trivial rigidity, independent of auth/conf/open.  With this term the
# correlation between |x| and g rises from ≈0 to ≈+0.5–0.6.
_G_EXTREMITY_W: float = 0.25

# Susceptibility extremity damping (Fix 4): reduces s for fringe agents by
# up to 30% so the centripetal pull of nearby moderates does not trivially
# overwhelm the fixed-attractor role of extremists.
_S_EXTREMITY_DAMP: float = 0.30


def traits_to_parameters(trait_vector: np.ndarray) -> dict[str, Any]:
    """
    Map one trait vector → full numerical agent parameter dict.

    The returned dict is deterministic, stateless, and contains no LLM-derived
    fields. See the spec for the closed-form mapping; comments inline below
    document the domain assumption behind each formula.

    Parameters
    ----------
    trait_vector : ndarray of shape (6,)
        Values in [0, 1], in AXIS_KEYS order.

    Returns
    -------
    dict with keys:
        x, s, sigma, epsilon, g, beta, rho, gamma, a, e, q, traits.
    """
    v = np.asarray(trait_vector, dtype=np.float64)
    if v.shape != (len(AXIS_KEYS),):
        raise ValueError(
            f"trait_vector must have shape ({len(AXIS_KEYS)},), got {v.shape}"
        )

    lr, auth, conf, open_, react, active = (float(x) for x in v)

    # Belief: linear rescale of left-right axis to [-1, +1].
    x = 2.0 * lr - 1.0

    # Extremity ∈ [0, 1] — shared input for several downstream formulas.
    # 0 = centrist, 1 = far left or far right.
    extremity = abs(x)

    # ── Fix 4: Susceptibility (damped by extremity) ───────────────────────
    # Extreme believers are up to 30% less susceptible — they occupy the
    # fringe and should resist centripetal pull more than centrists do.
    s = max(open_ * (1.0 - 0.5 * conf) * (1.0 - _S_EXTREMITY_DAMP * extremity), _S_FLOOR)

    # Belief uncertainty σ: distinct from s. Governs how confidently the
    # agent holds its current position rather than how easily it can move.
    sigma = max(1.0 - conf, _SIGMA_FLOOR)

    # ── Fix 1: Confidence bound ε (compressed, tied to belief extremity) ──
    # base_tolerance ∈ [0.05, 0.40]: open + uncertain agents accept a wider
    # window.  Extremity then shrinks that window proportionally so fringe
    # agents only update from their close ideological neighbours.
    # Result: median ε ≈ 0.19 (vs 0.84 before), placing the population in
    # the HK/DW fragmentation/two-cluster zone rather than the consensus zone.
    base_tolerance = _EPSILON_BASE_MIN + _EPSILON_BASE_RANGE * open_ * (1.0 - 0.5 * conf)
    epsilon = max(base_tolerance * (1.0 - _EPSILON_EXT_DAMP * extremity), _EPSILON_FLOOR)

    # ── Fix 2: Rigidity g (extremity as direct driver) ────────────────────
    # The original formula (auth · (1−open) · conf) had near-zero correlation
    # with |x| (≈ 0.002).  Adding an extremity term ensures fringe agents
    # always have meaningful rigidity — the fixed-attractor role they must
    # play for stable bimodal polarisation.
    g = min(auth * (1.0 - open_) * conf + _G_EXTREMITY_W * extremity, 1.0)

    # Selective exposure β: feed sorting strength toward aligned content.
    # Driven mostly by rigidity, secondarily by closed-mindedness alone.
    beta = _BETA_RIGIDITY_W * g + _BETA_CLOSED_W * (1.0 - open_)

    # Repulsion threshold ρ: distance beyond which an interaction flips
    # sign. By construction ρ ≥ ε; when g = 0 the neutral zone collapses
    # and distant opinions just don't influence (no repulsion).
    rho = epsilon + g * (2.0 - epsilon)

    # ── Fix 3: Repulsion gain γ (raised cap 0.5 → 0.8) ───────────────────
    # With mean s ≈ 0.41 and old γ_max = 0.5, attraction dominated repulsion
    # 6–7×.  At 0.8 the forces are closer to parity for high-rigidity agents,
    # allowing the repulsion zone to actually push beliefs apart.
    gamma = g * _GAMMA_CAP

    # Activity rate a: direct mapping with a tiny floor so even the
    # quietest agent has a nonzero chance of appearing in feeds.
    a = max(active, _ACTIVITY_FLOOR)

    # Emotional reactivity e: direct passthrough; consumed by the future
    # content-spreading layer, not by the core opinion dynamics.
    e = react

    # Topic salience q: extremists care more, centrists less.
    q = 0.5 + 0.5 * abs(x)

    return {
        "x": x,
        "s": s,
        "sigma": sigma,
        "epsilon": epsilon,
        "g": g,
        "beta": beta,
        "rho": rho,
        "gamma": gamma,
        "a": a,
        "e": e,
        "q": q,
        "traits": {
            "lr": lr,
            "auth": auth,
            "conf": conf,
            "open": open_,
            "react": react,
            "active": active,
        },
    }


# ---------------------------------------------------------------------------
# Stage 2b — JSON I/O
# ---------------------------------------------------------------------------

PIPELINE_STAGE_PARAMETERS: str = "parameters_complete"
PIPELINE_STAGE_BIOS: str = "bios_complete"

# Round all numerical values to this many decimals in the persisted JSON.
# 4 places is a good readability/accuracy trade-off — the simulation is
# robust to ~1e-4 perturbations.
_JSON_NUM_DECIMALS: int = 4


def _round_floats(obj: Any, ndigits: int = _JSON_NUM_DECIMALS) -> Any:
    """Deep-round all floats in a nested JSON-serialisable structure."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _agent_record(agent_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Assemble a single agent JSON record from a parameter dict."""
    return {
        "id": agent_id,
        "bio": None,
        "x": params["x"],
        "s": params["s"],
        "sigma": params["sigma"],
        "epsilon": params["epsilon"],
        "g": params["g"],
        "beta": params["beta"],
        "rho": params["rho"],
        "gamma": params["gamma"],
        "a": params["a"],
        "e": params["e"],
        "q": params["q"],
        "traits": params["traits"],
    }


def generate_population_json(
    n: int,
    seed: int,
    output_path: str | Path,
) -> Path:
    """
    Run stages 1 and 2 end-to-end and write the agent population to disk.

    The resulting file is the deterministic hand-off point between the
    numerical pipeline and the LLM bio stage. It can be inspected,
    diffed, and reproduced from `(n, seed)` alone.

    Returns the resolved output path.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    out_path = Path(output_path)
    traits = sample_trait_vectors(n=n, seed=seed)

    agents: list[dict[str, Any]] = []
    for i, row in enumerate(traits):
        params = traits_to_parameters(row)
        agents.append(_agent_record(f"agent_{i:04d}", params))

    payload = {
        "metadata": {
            "n": n,
            "seed": seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_stage": PIPELINE_STAGE_PARAMETERS,
            "bio_complete": False,
            "axis_order": list(AXIS_KEYS),
            "correlation_matrix": CORRELATION_MATRIX.tolist(),
        },
        "agents": agents,
    }

    payload = _round_floats(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Stage 3 — LLM bio generation
# ---------------------------------------------------------------------------

# Type alias for a minimal LLM client interface used by `generate_bios`. Any
# callable matching this signature works (real Gemini/OpenAI client, mock,
# offline stub). This deliberately keeps the bio stage provider-agnostic.
LLMClient = Callable[[str, str], str]


_BIO_PROMPT_TEMPLATE: str = """\
You are writing a brief background description for a fictional social media user in a computational simulation.

This person has the following measured characteristics:

Political-economic orientation: {lr_label}  ({lr:.2f} on a 0=far-left to 1=far-right scale)
Social orientation: {auth_label}  ({auth:.2f} on a 0=libertarian to 1=authoritarian scale)
Epistemic confidence: {conf_label}  ({conf:.2f} on a 0=very uncertain to 1=highly certain scale)
Openness to opposing views: {open_label}  ({open_:.2f} on a 0=closed to 1=very open scale)
Emotional reactivity: {react_label}  ({react:.2f} on a 0=calm to 1=highly reactive scale)
Online activity level: {active_label}  ({active:.2f} on a 0=rarely active to 1=very active scale)

Derived simulation parameters (read-only, for context):
  Belief score: {x:.3f}  (-1 = far left, +1 = far right)
  Susceptibility to influence: {s:.3f}
  Topic salience: {q:.3f}

Write a 2-3 sentence bio for this person. The bio should describe:
  - Their general social or professional background (age range, rough occupation area, lifestyle context).
  - A sentence that reflects their political or social outlook, consistent with the values above.
  - A sentence that reflects how they behave online, consistent with their activity and reactivity levels.

Do not invent numerical values. Do not use jargon like "epistemic" or "authoritarian." Write in plain, realistic language as if describing a real person on a social media profile.

Return only the bio text. No preamble, no explanation, no quotation marks."""


def _build_bio_prompt(agent: dict[str, Any]) -> str:
    """Render the bio prompt for one agent record."""
    t = agent["traits"]
    return _BIO_PROMPT_TEMPLATE.format(
        lr=t["lr"],
        auth=t["auth"],
        conf=t["conf"],
        open_=t["open"],
        react=t["react"],
        active=t["active"],
        lr_label=LABEL_FUNCTIONS["lr"](t["lr"]),
        auth_label=LABEL_FUNCTIONS["auth"](t["auth"]),
        conf_label=LABEL_FUNCTIONS["conf"](t["conf"]),
        open_label=LABEL_FUNCTIONS["open"](t["open"]),
        react_label=LABEL_FUNCTIONS["react"](t["react"]),
        active_label=LABEL_FUNCTIONS["active"](t["active"]),
        x=agent["x"],
        s=agent["s"],
        q=agent["q"],
    )


def _read_population(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"population file not found: {path}")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "agents" not in payload or "metadata" not in payload:
        raise ValueError(
            f"{path} is not a valid persona-pipeline file: missing metadata/agents."
        )
    return payload


def _write_population(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate_bios(
    json_path: str | Path,
    llm_client: LLMClient,
    model: str,
    *,
    delay_between_calls: float = 0.5,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """
    Fill in the `bio` field for every agent whose bio is currently null.

    The function is *resumable*: agents already carrying a non-null bio are
    skipped, so the LLM stage can be interrupted and rerun freely.

    The numerical fields of every agent record are touched ONLY when the
    file is round-tripped to disk (json.load → json.dumps), and even then
    no value is recomputed — only `bio` and `metadata.{pipeline_stage,
    bio_complete}` change between runs.

    Parameters
    ----------
    json_path : str | Path
        Path to a population JSON written by `generate_population_json`.
    llm_client : Callable[[str, str], str]
        `(prompt, model) -> str` returning the bio text. Wrap any provider
        SDK in a small lambda to fit this signature.
    model : str
        Forwarded to `llm_client` so the same callable can serve multiple
        models.
    delay_between_calls : float
        Seconds to sleep between successful calls (rate-limit friendliness).
    max_retries : int
        Per-agent retry budget on exception. Exponential backoff with base
        `initial_backoff`. Skipped agents are logged but do not abort the
        run.
    initial_backoff : float
        Seconds for the first retry; doubles each subsequent retry.
    sleep_fn : Callable
        Override `time.sleep` for testability.

    Returns
    -------
    A dict with run statistics: {"filled": int, "skipped": int, "skipped_ids": list[str]}.
    """
    path = Path(json_path)
    payload = _read_population(path)

    stage = payload["metadata"].get("pipeline_stage")
    if stage not in (PIPELINE_STAGE_PARAMETERS, PIPELINE_STAGE_BIOS):
        raise ValueError(
            f"Cannot run bios on {path}: pipeline_stage={stage!r}, "
            f"expected {PIPELINE_STAGE_PARAMETERS!r} (or {PIPELINE_STAGE_BIOS!r} for resume)."
        )

    filled = 0
    skipped_ids: list[str] = []

    for agent in payload["agents"]:
        if agent.get("bio") is not None:
            continue
        prompt = _build_bio_prompt(agent)
        bio_text: Optional[str] = None
        backoff = initial_backoff
        for attempt in range(max_retries):
            try:
                bio_text = llm_client(prompt, model)
                if not isinstance(bio_text, str) or not bio_text.strip():
                    raise ValueError("LLM returned empty bio text")
                bio_text = bio_text.strip()
                break
            except Exception as exc:
                logger.warning(
                    "LLM call failed for %s on attempt %d/%d: %s",
                    agent["id"], attempt + 1, max_retries, exc,
                )
                if attempt < max_retries - 1:
                    sleep_fn(backoff)
                    backoff *= 2.0
        if bio_text is None:
            logger.warning("Skipping bio for %s after %d failed attempts",
                           agent["id"], max_retries)
            skipped_ids.append(agent["id"])
            continue
        agent["bio"] = bio_text
        filled += 1
        if delay_between_calls > 0:
            sleep_fn(delay_between_calls)

    all_complete = all(a.get("bio") is not None for a in payload["agents"])
    if all_complete:
        payload["metadata"]["pipeline_stage"] = PIPELINE_STAGE_BIOS
        payload["metadata"]["bio_complete"] = True

    _write_population(path, payload)

    return {
        "filled": filled,
        "skipped": len(skipped_ids),
        "skipped_ids": skipped_ids,
    }


# ---------------------------------------------------------------------------
# Step 6 — load_population (engine-shaped output)
# ---------------------------------------------------------------------------

# Map from new pipeline field names → engine-expected field names. The
# engine's `_resolve_per_agent` already reads the right-hand-side names for
# per-agent overrides, so populating those keys is sufficient to wire up
# heterogeneous agent parameters in any model_type.
_PIPELINE_TO_ENGINE_FIELDS: Mapping[str, str] = {
    "x": "initial_belief",
    "s": "susceptibility",
    "a": "activity",
    "epsilon": "confidence_epsilon",
    "beta": "selective_exposure_beta",
    "rho": "repulsion_threshold_rho",
    "gamma": "repulsion_strength_gamma",
    "sigma": "belief_uncertainty",
    "g": "rigidity",
    "e": "emotional_reactivity",
    "q": "topic_salience",
}


def _agent_to_engine_dict(agent: dict[str, Any]) -> dict[str, Any]:
    """Convert one pipeline agent record to a dict the engine consumes."""
    out: dict[str, Any] = {
        # Engine reassigns id = i internally; preserve original as a
        # diagnostic / inspector field.
        "id": agent["id"],
        "name": agent.get("name") or str(agent["id"]),
        "bio": agent.get("bio") or "",
    }
    for src, dst in _PIPELINE_TO_ENGINE_FIELDS.items():
        if src in agent:
            out[dst] = float(agent[src])
    if "traits" in agent:
        out["traits"] = agent["traits"]
    return out


def load_population(json_path: str | Path) -> list[dict[str, Any]]:
    """
    Read a pipeline JSON and return a list of engine-format agent dicts.

    No LLM is consulted. No sampling is run. This is pure deserialisation
    plus a name-mapping translation step.

    Backward compatibility: when handed a flat-array JSON file (the legacy
    persona format), the function returns it untouched so existing test
    fixtures and `agents_500_pro_max.json` keep working unchanged.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"population file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        if not raw:
            raise ValueError("agent JSON list is empty")
        return raw

    if not isinstance(raw, dict) or "agents" not in raw:
        raise ValueError(
            f"{path} is neither a flat-array JSON nor a pipeline payload"
        )

    stage = raw.get("metadata", {}).get("pipeline_stage")
    if stage not in (PIPELINE_STAGE_PARAMETERS, PIPELINE_STAGE_BIOS):
        raise ValueError(
            f"Cannot load population from {path}: pipeline_stage={stage!r}"
        )

    return [_agent_to_engine_dict(a) for a in raw["agents"]]


# ---------------------------------------------------------------------------
# Step 7 — validation + diagnostics
# ---------------------------------------------------------------------------

# Tolerances for floating-point round-trip comparisons. Anything within this
# margin of a hard bound is treated as "in range" — needed because the JSON
# is rounded to 4 decimals so e.g. ε = 0.2 may serialise as 0.2 then come
# back exactly equal but margin checks like `> 0.05` need a small epsilon
# to absorb rounding.
_RANGE_TOL: float = 1e-6

# Warning threshold for right-skew detection. >60% of x > 0 suggests a bug.
_RIGHT_SKEW_WARN_FRACTION: float = 0.6


def validate_population(json_path: str | Path, *, log: bool = True) -> dict[str, Any]:
    """
    Validate per-agent invariants and print a population-level summary.

    Raises
    ------
    ValueError on any range violation.

    Warns (logger.warning, no exception) if more than 60% of agents have
    `x > 0`, which is the canonical smoke-test for a sampling/mapping bug
    that produced a right-skewed population.

    Returns a dict of computed summary statistics.
    """
    path = Path(json_path)
    payload = _read_population(path)
    agents = payload["agents"]
    if not agents:
        raise ValueError("population is empty")

    errors: list[str] = []
    for a in agents:
        aid = a.get("id", "<unknown>")
        x = a["x"]
        s = a["s"]
        sigma = a["sigma"]
        epsilon = a["epsilon"]
        g = a["g"]
        beta = a["beta"]
        rho = a["rho"]
        gamma = a["gamma"]
        act = a["a"]

        if not (-1.0 - _RANGE_TOL <= x <= 1.0 + _RANGE_TOL):
            errors.append(f"{aid}: x={x} out of [-1, 1]")
        if s <= 0.0:
            errors.append(f"{aid}: s={s} not strictly positive")
        if not (0.0 <= sigma <= 1.0 + _RANGE_TOL):
            errors.append(f"{aid}: sigma={sigma} out of (0, 1]")
        if not (_EPSILON_FLOOR - _RANGE_TOL <= epsilon <= 2.0 + _RANGE_TOL):
            errors.append(f"{aid}: epsilon={epsilon} out of [{_EPSILON_FLOOR}, 2]")
        if not (0.0 - _RANGE_TOL <= g <= 1.0 + _RANGE_TOL):
            errors.append(f"{aid}: g={g} out of [0, 1]")
        if beta < 0.0 - _RANGE_TOL:
            errors.append(f"{aid}: beta={beta} negative")
        if rho < epsilon - _RANGE_TOL:
            errors.append(f"{aid}: rho={rho} < epsilon={epsilon}")
        if rho > 2.0 + _RANGE_TOL:
            errors.append(f"{aid}: rho={rho} > 2")
        if gamma > _GAMMA_CAP + _RANGE_TOL:
            errors.append(f"{aid}: gamma={gamma} > {_GAMMA_CAP}")
        if gamma < 0.0 - _RANGE_TOL:
            errors.append(f"{aid}: gamma={gamma} negative")
        if act <= 0.0:
            errors.append(f"{aid}: a={act} not strictly positive")

    if errors:
        sample = "\n  ".join(errors[:10])
        more = "" if len(errors) <= 10 else f"\n  ... and {len(errors) - 10} more"
        raise ValueError(
            f"validate_population: {len(errors)} invariant violation(s):\n  {sample}{more}"
        )

    arrays = {
        k: np.array([float(a[k]) for a in agents], dtype=np.float64)
        for k in ("x", "s", "sigma", "epsilon", "beta", "a")
    }

    stats: dict[str, Any] = {
        "n": len(agents),
        "summary": {
            k: {
                "mean": float(arrays[k].mean()),
                "std": float(arrays[k].std()),
                "min": float(arrays[k].min()),
                "max": float(arrays[k].max()),
            }
            for k in arrays
        },
        "x_histogram_bins": np.linspace(-1.0, 1.0, 11).tolist(),
        "x_histogram_counts": [
            int(c) for c in np.histogram(arrays["x"], bins=10, range=(-1.0, 1.0))[0]
        ],
        "strong_believers": int((np.abs(arrays["x"]) > 0.6).sum()),
        "near_lurkers": int((arrays["a"] < 0.1).sum()),
    }

    right_skew_fraction = float((arrays["x"] > 0.0).mean())
    stats["right_skew_fraction"] = right_skew_fraction
    if right_skew_fraction > _RIGHT_SKEW_WARN_FRACTION:
        logger.warning(
            "validate_population: %.0f%% of agents have x > 0 (>%.0f%% threshold). "
            "Likely a sampling or mapping bug producing a right-skewed population.",
            right_skew_fraction * 100,
            _RIGHT_SKEW_WARN_FRACTION * 100,
        )

    if log:
        _log_validation_summary(stats)

    return stats


def _log_validation_summary(stats: dict[str, Any]) -> None:
    """Pretty-print a population summary table to stdout."""
    print(f"\n=== Population summary (n={stats['n']}) ===")
    header = f"{'field':<10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"
    print(header)
    print("-" * len(header))
    for field, s in stats["summary"].items():
        print(
            f"{field:<10} {s['mean']:>10.4f} {s['std']:>10.4f} "
            f"{s['min']:>10.4f} {s['max']:>10.4f}"
        )
    print("\nx histogram (bins from -1 to +1, 10 bins):")
    edges = stats["x_histogram_bins"]
    for i, c in enumerate(stats["x_histogram_counts"]):
        print(f"  [{edges[i]:+.2f}, {edges[i+1]:+.2f})  {c}")
    print(f"\nstrong believers (|x| > 0.6): {stats['strong_believers']}")
    print(f"near lurkers   (a   < 0.1):    {stats['near_lurkers']}")
    print(f"right-skew fraction (x > 0):   {stats['right_skew_fraction']:.2%}")


__all__ = [
    "PIPELINE_STAGE_PARAMETERS",
    "PIPELINE_STAGE_BIOS",
    "_lhs_uniform",
    "sample_trait_vectors",
    "traits_to_parameters",
    "generate_population_json",
    "generate_bios",
    "load_population",
    "validate_population",
    "_build_bio_prompt",
]
