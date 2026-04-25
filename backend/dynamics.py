"""
Pure, stateless helpers for the modular opinion-dynamics layer.

These functions operate purely on NumPy arrays so they can be unit-tested
independently of `SimulationEngine` and reused by future model variants.

Math overview
=============

Static substrate (unchanged each tick):

    A_{ij} ∈ {0, 1}     i follows j
    W_{ij} ≥ 0          baseline edge weight (defaults to A_{ij})
    T_{ij}              optional trust factor; reserved, defaults to 1

Per-step quantities:

    d_{ij}(t) = |x_i(t) - x_j(t)|                         opinion distance
    M_{ij}(t)                                             exposure mask
    C_{ij}(t)                                             compatibility weight

Effective influence (matrix-form modes):

    u_{ij}(t)  = A_{ij} · W_{ij} · M_{ij}(t) · C_{ij}(t) · T_{ij}
    Ŵ_{ij}(t) = u_{ij}(t) / Σ_k u_{ik}(t)                 row-normalised

Belief update (matrix form, used by degroot / confirmation_bias /
bounded_confidence):

    x_i(t+1) = x_i(t) + s_i · Σ_j Ŵ_{ij}(t) · (x_j(t) - x_i(t)) + ξ_i(t)

Belief update (incremental form, used by repulsive_bc):

    x_i(t+1) = x_i(t) + s_i · Σ_j Ŵ_{ij}(t) · φ_i(x_j - x_i) + ξ_i(t)

with the piecewise φ defined in `phi_repulsive`.

The follower graph A is static; everything that varies tick-to-tick lives in
M(t) and C(t), so the *effective* influence network is dynamic even though
the substrate is not.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Distance + safe normalisation
# ---------------------------------------------------------------------------

def compute_distances(B: np.ndarray) -> np.ndarray:
    """Return the |b_i - b_j| matrix for a 1-D belief vector."""
    return np.abs(B[:, None] - B[None, :])


def safe_row_normalize(X: np.ndarray) -> np.ndarray:
    """
    Row-normalise non-negative X; rows that sum to zero are returned as zero
    rows so callers can detect "no-influence" agents and freeze them.
    """
    row_sums = X.sum(axis=1, keepdims=True)
    out = np.zeros_like(X, dtype=np.float64)
    np.divide(X, row_sums, where=row_sums > 0, out=out)
    return out


# ---------------------------------------------------------------------------
# Compatibility weights C(t)
# ---------------------------------------------------------------------------

def compatibility(
    model_type: str,
    dist: np.ndarray,
    *,
    alpha_decay: np.ndarray | float = 2.0,
    epsilon: np.ndarray | float = 0.4,
) -> np.ndarray:
    """
    Per-edge compatibility weight C_{ij}(t) given current opinion distance.

    Parameters
    ----------
    model_type : str
        - "degroot"            → C ≡ 1 (no compatibility filter)
        - "confirmation_bias"  → C_{ij} = exp(-α_i · d_{ij})
        - "bounded_confidence" → C_{ij} = 1[d_{ij} ≤ ε_i]
        - "repulsive_bc"       → C ≡ 1 (the φ function handles attract/repel)

    `alpha_decay` and `epsilon` may be scalars (homogeneous) or 1-D arrays of
    length N (per-agent heterogeneous, broadcast over the rows).
    """
    if model_type in ("degroot", "repulsive_bc"):
        return np.ones_like(dist)

    if model_type == "confirmation_bias":
        a = np.asarray(alpha_decay, dtype=np.float64)
        if a.ndim == 0:
            return np.exp(-a * dist)
        return np.exp(-a[:, None] * dist)

    if model_type == "bounded_confidence":
        e = np.asarray(epsilon, dtype=np.float64)
        if e.ndim == 0:
            return (dist <= e).astype(np.float64)
        return (dist <= e[:, None]).astype(np.float64)

    raise ValueError(f"unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# Exposure mask M(t)
# ---------------------------------------------------------------------------

def exposure_mask(
    mode: str,
    A: np.ndarray,
    dist: np.ndarray,
    *,
    activity: np.ndarray,
    selective_beta: np.ndarray | float = 2.0,
    top_k: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Per-step visibility mask M_{ij}(t) ∈ {0, 1} restricted to followed edges.

    Modes
    -----
    "all_followed"
        M = A. Every followed account is visible every tick.

    "top_k"
        For each i, rank j ∈ followed(i) by
            score_{ij} = β_i · (1 - d_{ij}) + log(1 + a_j)
        and mark the top `top_k` as visible. Ties broken by the natural
        argsort stability of NumPy.

    "sampled"
        Sample (without replacement) up to `top_k` followed neighbours per i
        with probability proportional to
            p_{ij} ∝ exp(β_i · (1 - d_{ij})) · a_j
        Requires `rng`; reproducibility flows from the engine's seed.

    Parameters
    ----------
    A          : (N, N) follow adjacency, A[i,j]=1 ⟺ i follows j
    dist       : (N, N) current |x_i - x_j|
    activity   : (N,) sender activity a_j (default 1.0 if not provided)
    selective_beta : scalar or (N,) per-agent β_i
    top_k      : positive int
    rng        : numpy Generator (required for "sampled")
    """
    N = A.shape[0]
    A_bool = A > 0

    if mode == "all_followed":
        return A_bool.astype(np.float64)

    beta = np.asarray(selective_beta, dtype=np.float64)
    if beta.ndim == 0:
        beta_i = np.full(N, float(beta))
    else:
        beta_i = beta

    if mode == "top_k":
        # score_{ij} = β_i (1 - d_{ij}) + log(1 + a_j)
        score = beta_i[:, None] * (1.0 - dist) + np.log1p(activity)[None, :]
        # Mask non-followed edges with -inf so they never win the top-k race.
        score = np.where(A_bool, score, -np.inf)

        M = np.zeros_like(A, dtype=np.float64)
        for i in range(N):
            row_scores = score[i]
            followed_count = int(A_bool[i].sum())
            if followed_count == 0:
                continue
            keep = min(top_k, followed_count)
            # argpartition is O(N); we only need the top `keep` indices.
            top_idx = np.argpartition(-row_scores, keep - 1)[:keep]
            M[i, top_idx] = 1.0
        return M

    if mode == "sampled":
        if rng is None:
            raise ValueError("sampled exposure requires an rng for reproducibility")
        # Probability ∝ exp(β_i (1 - d_{ij})) · a_j on followed edges.
        logit = beta_i[:, None] * (1.0 - dist)
        # Subtract row-max for numerical stability before exp.
        logit_safe = logit - logit.max(axis=1, keepdims=True)
        prob = np.exp(logit_safe) * activity[None, :]
        prob = np.where(A_bool, prob, 0.0)

        M = np.zeros_like(A, dtype=np.float64)
        for i in range(N):
            row = prob[i]
            total = row.sum()
            if total <= 0:
                continue
            p = row / total
            followed_count = int(A_bool[i].sum())
            if followed_count == 0:
                continue
            sample_size = min(top_k, followed_count)
            # rng.choice supports weighted sampling without replacement.
            chosen = rng.choice(N, size=sample_size, replace=False, p=p)
            M[i, chosen] = 1.0
        return M

    raise ValueError(f"unknown exposure mode: {mode!r}")


# ---------------------------------------------------------------------------
# Effective row-normalised weights
# ---------------------------------------------------------------------------

def effective_weights(
    A: np.ndarray,
    W_base: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    *,
    T: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute u_{ij} = A · W · M · C · T and row-normalise.

    Returns an (N, N) matrix with rows summing to 1, except rows where every
    entry is zero (no compatible visible neighbour) which remain all-zero so
    the caller can detect "freeze this agent" rows.
    """
    u = A * W_base * M * C
    if T is not None:
        u = u * T
    return safe_row_normalize(u)


# ---------------------------------------------------------------------------
# Repulsion piecewise φ
# ---------------------------------------------------------------------------

def phi_repulsive(
    delta: np.ndarray,
    *,
    epsilon: np.ndarray | float,
    rho: np.ndarray | float,
    gamma: np.ndarray | float,
) -> np.ndarray:
    """
    Per-edge attraction / neutral / repulsion response to Δ = x_j - x_i.

        φ_i(Δ) =  Δ              if |Δ| ≤ ε_i      (attraction)
                  0              if ε_i < |Δ| ≤ ρ_i (mid-range ignore)
                 -γ_i · Δ        if |Δ| > ρ_i      (repulsion)

    All thresholds may be scalars or 1-D arrays broadcastable over the rows
    of `delta` (which is shape (N, N)).
    """
    abs_d = np.abs(delta)
    e = np.asarray(epsilon, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)

    if e.ndim == 1:
        e = e[:, None]
    if r.ndim == 1:
        r = r[:, None]
    if g.ndim == 1:
        g = g[:, None]

    out = np.zeros_like(delta, dtype=np.float64)
    attract = abs_d <= e
    repel = abs_d > r
    out = np.where(attract, delta, out)
    out = np.where(repel, -g * delta, out)
    return out
