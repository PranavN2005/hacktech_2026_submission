"""
Configuration for the structured persona-generation pipeline.

This module is the single source of truth for:

    - which continuous trait axes the population is sampled along,
    - the marginal Beta distribution per axis,
    - the symmetric correlation matrix Σ that ties axes together,
    - the plain-English label functions used by the LLM bio prompt.

The values below are *defaults*. Everything is plain data so a downstream
caller (or a future calibration script) can override marginals or replace
Σ with survey-derived values without touching the sampler or mapper.

No randomness, no I/O, no LLM. This module is pure configuration.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Axis order — this is the canonical column order used by every downstream
# tensor in the pipeline (sampler output, JSON `traits` blocks, mapper input).
# ---------------------------------------------------------------------------

AXIS_KEYS: tuple[str, ...] = ("lr", "auth", "conf", "open", "react", "active")

AXIS_DESCRIPTIONS: Mapping[str, str] = {
    "lr": "Left-right political-economic axis (0 = far left, 1 = far right).",
    "auth": "Social-libertarian to authoritarian axis (0 = libertarian, 1 = authoritarian).",
    "conf": "Epistemic confidence in own beliefs (0 = very uncertain, 1 = maximally certain).",
    "open": "Openness to opposing views (0 = closed, 1 = very open).",
    "react": "Emotional reactivity to charged content (0 = dispassionate, 1 = highly reactive).",
    "active": "Social media activity rate (0 = rarely active, 1 = extremely active).",
}


# ---------------------------------------------------------------------------
# Marginal distributions
#
# Each axis is sampled from a Beta(a, b) distribution on [0, 1]. The shape
# parameters are chosen to match domain intuition — see comments below.
# Override this dict to recalibrate the population's marginal mood.
# ---------------------------------------------------------------------------

MARGINAL_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    # peaks at center, tails at extremes
    "lr": {"a": 2.0, "b": 2.0},
    # broadly uniform with mild center peak
    "auth": {"a": 1.5, "b": 1.5},
    # skewed toward moderate uncertainty (mean ≈ 0.4)
    "conf": {"a": 2.0, "b": 3.0},
    # symmetric peak at 0.5
    "open": {"a": 2.0, "b": 2.0},
    # mild left skew (most agents not very reactive)
    "react": {"a": 1.5, "b": 2.0},
    # heavily skewed toward low activity (most users are lurkers)
    "active": {"a": 1.0, "b": 3.0},
}


# ---------------------------------------------------------------------------
# Correlation matrix Σ
#
# Built from a sparse list of (axis_a, axis_b, value) triples and validated
# at module import time. Anything not listed is implicitly 0; the diagonal
# is 1.
# ---------------------------------------------------------------------------

# (axis_a, axis_b, ρ) — order within a pair does not matter. See the docstring
# for justification of each value.
_CORRELATION_PAIRS: Sequence[tuple[str, str, float]] = (
    # right-authoritarianism cluster (Altemeyer, Stenner)
    ("lr", "auth", 0.45),
    # high certainty ↔ low openness (epistemic closure literature)
    ("conf", "open", -0.55),
    # authoritarianism ↔ epistemic closure
    ("conf", "auth", 0.30),
    # reactivity ↔ defensiveness toward challenge
    ("react", "open", -0.25),
    # reactivity predicts online activity (engagement literature)
    ("react", "active", 0.35),
    # weak: right-leaning slightly less open on average
    ("lr", "open", -0.20),
)


def _build_correlation_matrix(
    keys: Sequence[str],
    pairs: Sequence[tuple[str, str, float]],
) -> np.ndarray:
    """Assemble a symmetric Σ from the sparse pair list and validate PSD."""
    n = len(keys)
    idx = {k: i for i, k in enumerate(keys)}
    sigma = np.eye(n, dtype=np.float64)
    for a, b, v in pairs:
        if a not in idx or b not in idx:
            raise KeyError(f"Unknown axis in correlation pair: ({a!r}, {b!r})")
        sigma[idx[a], idx[b]] = v
        sigma[idx[b], idx[a]] = v
    eigs = np.linalg.eigvalsh(sigma)
    if eigs.min() < -1e-9:
        raise ValueError(
            "Correlation matrix Σ is not positive semi-definite "
            f"(min eigenvalue = {eigs.min():.6g}). "
            "Reduce the magnitude of one or more off-diagonal entries."
        )
    return sigma


CORRELATION_MATRIX: np.ndarray = _build_correlation_matrix(
    AXIS_KEYS, _CORRELATION_PAIRS
)


# ---------------------------------------------------------------------------
# Plain-English label functions used by the bio prompt
#
# These are the only "knob" the LLM gets at stage 3: it sees readable
# labels (and the underlying scalar) instead of jargon, and is told to write
# in plain language consistent with both.
# ---------------------------------------------------------------------------


def lr_label(v: float) -> str:
    if v < 0.2:
        return "strongly left-leaning"
    if v < 0.4:
        return "moderately left-leaning"
    if v < 0.6:
        return "centrist"
    if v < 0.8:
        return "moderately right-leaning"
    return "strongly right-leaning"


def auth_label(v: float) -> str:
    if v < 0.25:
        return "strongly libertarian"
    if v < 0.5:
        return "mildly libertarian"
    if v < 0.75:
        return "mildly traditional/authoritarian"
    return "strongly traditional/authoritarian"


def conf_label(v: float) -> str:
    if v < 0.33:
        return "quite uncertain in their views"
    if v < 0.66:
        return "moderately confident"
    return "very sure of their views"


def open_label(v: float) -> str:
    if v < 0.33:
        return "resistant to changing their mind"
    if v < 0.66:
        return "somewhat open to other perspectives"
    return "genuinely curious about opposing views"


def react_label(v: float) -> str:
    if v < 0.33:
        return "emotionally even-keeled"
    if v < 0.66:
        return "occasionally reactive"
    return "strongly emotional in responses to provocative content"


def active_label(v: float) -> str:
    if v < 0.2:
        return "rarely uses social media"
    if v < 0.5:
        return "occasional social media user"
    if v < 0.8:
        return "fairly active social media user"
    return "very heavy social media user"


LABEL_FUNCTIONS: Mapping[str, Callable[[float], str]] = {
    "lr": lr_label,
    "auth": auth_label,
    "conf": conf_label,
    "open": open_label,
    "react": react_label,
    "active": active_label,
}


__all__ = [
    "AXIS_KEYS",
    "AXIS_DESCRIPTIONS",
    "MARGINAL_DISTRIBUTIONS",
    "CORRELATION_MATRIX",
    "LABEL_FUNCTIONS",
    "lr_label",
    "auth_label",
    "conf_label",
    "open_label",
    "react_label",
    "active_label",
]
