"""
SimulationConfig: declarative knobs for the new dynamics layer.

All fields have sensible defaults so an engine constructed without an explicit
config behaves as a reasonable bounded-confidence simulation.

Notes on the math (full derivations live in `dynamics.py`):

    Effective influence weights for the matrix-form modes (degroot,
    confirmation_bias, bounded_confidence) are

        W̃_{ij}(t) ∝ A_{ij} · W_{ij} · M_{ij}(t) · C_{ij}(t) · T_{ij}

    where A is the static follow graph, M is the per-step exposure mask,
    C is the per-step compatibility weight, and T is an optional trust
    factor (currently disabled by default).

    For repulsive_bc the update is incremental and uses an attraction /
    neutral / repulsion piecewise function; see `dynamics.phi_repulsive`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModelType = Literal["degroot", "confirmation_bias", "bounded_confidence", "repulsive_bc"]
ExposureMode = Literal["all_followed", "top_k", "sampled"]


@dataclass
class SimulationConfig:
    """Declarative configuration for the modular dynamics path."""

    # Which influence rule to use.
    model_type: ModelType = "bounded_confidence"

    # How content is selected from followed accounts each tick.
    exposure_mode: ExposureMode = "all_followed"

    # `top_k` and `sampled` modes use this to bound how many sources reach i.
    top_k_visible: int = 10

    # Selective-exposure strength used inside the exposure score / probability.
    selective_exposure_beta: float = 2.0

    # Confirmation-bias smooth distance discount: C_ij = exp(-α · |x_i - x_j|).
    distance_decay_alpha: float = 2.0

    # Hard bounded-confidence threshold: C_ij = 1[ |x_i - x_j| ≤ ε ].
    confidence_epsilon: float = 0.4

    # Repulsion onset distance ρ; must satisfy ρ ≥ ε.
    repulsion_threshold_rho: float = 0.9

    # Repulsion gain γ used in φ(Δ) = -γ · Δ for |Δ| > ρ.
    repulsion_strength_gamma: float = 0.5

    # Reserved for a future per-edge trust factor T_ij. When False, T ≡ 1.
    trust_enabled: bool = False

    # Std of i.i.d. Gaussian noise added to each belief update.
    noise_sigma: float = 0.0

    # Always clip beliefs back to [-1, 1] after each step.
    clip_beliefs: bool = True

    # Optional per-mode metadata, useful for plotting / labelling experiments.
    label: str = ""

    # Reserved hook for future per-agent overrides keyed by agent id.
    # Values may include any of: confidence_epsilon, repulsion_threshold_rho,
    # repulsion_strength_gamma, selective_exposure_beta, distance_decay_alpha.
    per_agent_overrides: dict[int, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_type not in (
            "degroot",
            "confirmation_bias",
            "bounded_confidence",
            "repulsive_bc",
        ):
            raise ValueError(f"unknown model_type: {self.model_type!r}")
        if self.exposure_mode not in ("all_followed", "top_k", "sampled"):
            raise ValueError(f"unknown exposure_mode: {self.exposure_mode!r}")
        if self.top_k_visible <= 0:
            raise ValueError("top_k_visible must be a positive integer")
        if self.confidence_epsilon < 0:
            raise ValueError("confidence_epsilon must be ≥ 0")
        if self.repulsion_threshold_rho < self.confidence_epsilon:
            # Mid-range ignore zone (ε, ρ] is empty if ρ < ε; spec requires ρ ≥ ε.
            raise ValueError(
                "repulsion_threshold_rho must be ≥ confidence_epsilon "
                f"(got ρ={self.repulsion_threshold_rho}, ε={self.confidence_epsilon})"
            )
        if self.repulsion_strength_gamma < 0:
            raise ValueError("repulsion_strength_gamma must be ≥ 0")
        if self.distance_decay_alpha < 0:
            raise ValueError("distance_decay_alpha must be ≥ 0")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be ≥ 0")
