"""
Stage-2 (mapping) tests for the persona pipeline.

`traits_to_parameters` is a pure deterministic function: trait vector in,
parameter dict out. These tests pin down both the closed-form correctness
(belief sign, susceptibility floor, ρ ≥ ε, γ cap) and the qualitative
behaviour at the extreme corners of trait space (the "ideologue" and
"open lurker" archetypes).

No randomness, no LLM, no I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.persona_config import AXIS_KEYS
from backend.persona_pipeline import (
    sample_trait_vectors,
    traits_to_parameters,
)


def _random_trait_vectors(n: int, seed: int) -> np.ndarray:
    """Random trait vectors uniform in [0, 1]^6 (NOT the LHS sampler).

    For pure invariant testing we only need broad coverage, not stratified
    or correlated samples — so a plain uniform draw stresses the mapping
    over the full unit hypercube without coupling to the sampler.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, len(AXIS_KEYS)))


# ---------------------------------------------------------------------------
# Range invariants over a fuzzed input space
# ---------------------------------------------------------------------------


class TestRangeInvariants:
    def test_belief_range(self) -> None:
        traits = _random_trait_vectors(1000, seed=42)
        for row in traits:
            p = traits_to_parameters(row)
            assert -1.0 <= p["x"] <= 1.0

    def test_susceptibility_floor(self) -> None:
        traits = _random_trait_vectors(1000, seed=43)
        for row in traits:
            p = traits_to_parameters(row)
            assert p["s"] >= 0.05 - 1e-12

    def test_rho_geq_epsilon(self) -> None:
        traits = _random_trait_vectors(1000, seed=44)
        for row in traits:
            p = traits_to_parameters(row)
            assert p["rho"] >= p["epsilon"] - 1e-12

    def test_gamma_cap(self) -> None:
        # Cap raised from 0.5 to 0.8 so repulsion is a meaningful counterforce.
        traits = _random_trait_vectors(1000, seed=45)
        for row in traits:
            p = traits_to_parameters(row)
            assert p["gamma"] <= 0.8 + 1e-12
            assert p["gamma"] >= 0.0 - 1e-12

    def test_sigma_floor(self) -> None:
        """Internal uncertainty σ has a 0.05 floor analogous to s."""
        traits = _random_trait_vectors(1000, seed=46)
        for row in traits:
            p = traits_to_parameters(row)
            assert p["sigma"] >= 0.05 - 1e-12

    def test_activity_floor(self) -> None:
        """Activity rate has a small but nonzero floor (0.01) so no agent is
        completely invisible in feeds."""
        traits = _random_trait_vectors(1000, seed=47)
        for row in traits:
            p = traits_to_parameters(row)
            assert p["a"] >= 0.01 - 1e-12

    def test_topic_salience_in_range(self) -> None:
        traits = _random_trait_vectors(1000, seed=48)
        for row in traits:
            p = traits_to_parameters(row)
            assert 0.5 <= p["q"] <= 1.0


# ---------------------------------------------------------------------------
# Archetype tests at the corners of trait space
# ---------------------------------------------------------------------------


class TestArchetypes:
    def test_extreme_left_agent(self) -> None:
        """traits=[lr=0, auth=0, conf=0, open=1, react=0, active=0]

        x = -1.  Extremity = 1 so:
          s = open*(1-0.5*conf)*(1-0.3*extremity) = 1*1*0.7 = 0.70
          g = auth*(1-open)*conf + 0.25*extremity  = 0 + 0.25 = 0.25
          gamma = g * 0.8 = 0.20
        """
        v = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        p = traits_to_parameters(v)
        assert p["x"] == pytest.approx(-1.0)
        assert p["s"] == pytest.approx(0.70), f"got s={p['s']}"
        assert p["g"] == pytest.approx(0.25), f"extremity alone → g=0.25, got {p['g']}"
        assert p["gamma"] == pytest.approx(0.20), f"got gamma={p['gamma']}"

    def test_extreme_right_agent(self) -> None:
        """traits=[lr=1, auth=1, conf=1, open=0, react=1, active=1]
        → x = +1, s hits floor (closed+confident+extreme), g = 1 (capped),
          gamma = 0.8 (raised cap), rho = 2."""
        v = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        p = traits_to_parameters(v)
        assert p["x"] == pytest.approx(1.0)
        assert p["s"] == pytest.approx(0.05), (
            f"closed + confident + extreme ⇒ s should hit floor 0.05, got {p['s']}"
        )
        assert p["g"] == pytest.approx(1.0), f"capped at 1.0, got {p['g']}"
        assert p["gamma"] == pytest.approx(0.8), f"gamma=g*0.8=0.8, got {p['gamma']}"
        assert p["rho"] == pytest.approx(2.0)
        assert p["q"] == pytest.approx(1.0), "extreme |x| ⇒ max salience"

    def test_centrist_open_lurker(self) -> None:
        """traits=[0.5, 0.5, 0.5, 1.0, 0.0, 0.0] — the maximally tractable
        agent: belief 0, low rigidity, high openness, near-zero activity."""
        v = np.array([0.5, 0.5, 0.5, 1.0, 0.0, 0.0])
        p = traits_to_parameters(v)
        assert p["x"] == pytest.approx(0.0)
        assert p["g"] == pytest.approx(0.0)
        assert p["gamma"] == pytest.approx(0.0)
        assert p["a"] == pytest.approx(0.01), "activity floor should kick in"
        assert p["q"] == pytest.approx(0.5), "centrist ⇒ minimum salience"


# ---------------------------------------------------------------------------
# Closed-form formula sanity (spot-check exact values)
# ---------------------------------------------------------------------------


class TestClosedFormCorrectness:
    def test_belief_linear_rescale(self) -> None:
        # x = 2 * lr - 1
        for lr in (0.0, 0.25, 0.5, 0.75, 1.0):
            v = np.array([lr, 0.5, 0.5, 0.5, 0.5, 0.5])
            assert traits_to_parameters(v)["x"] == pytest.approx(2.0 * lr - 1.0)

    def test_epsilon_formula(self) -> None:
        # New formula (Fix 1):
        #   base_tolerance = 0.05 + 0.35 * open * (1 - 0.5 * conf)
        #   ε = base_tolerance * (1 - 0.4 * extremity);  ε ≥ 0.05
        #
        # Centrist (lr=0.5 → extremity=0), open=1, conf=0:
        #   base_tolerance = 0.05 + 0.35 = 0.40
        #   ε = 0.40 * 1.0 = 0.40
        v_max = np.array([0.5, 0.5, 0.0, 1.0, 0.5, 0.5])
        assert traits_to_parameters(v_max)["epsilon"] == pytest.approx(0.40)
        # Centrist, open=0 → base_tolerance = 0.05 → ε = 0.05 (floor)
        v_min = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.5])
        assert traits_to_parameters(v_min)["epsilon"] == pytest.approx(0.05)

    def test_traits_block_round_trips(self) -> None:
        """The `traits` sub-dict echoes the input verbatim."""
        v = np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
        t = traits_to_parameters(v)["traits"]
        for key, val in zip(AXIS_KEYS, v):
            assert t[key] == pytest.approx(float(val))

    def test_returned_keys_match_spec(self) -> None:
        v = np.array([0.5] * 6)
        p = traits_to_parameters(v)
        assert set(p.keys()) == {
            "x", "s", "sigma", "epsilon", "g", "beta", "rho", "gamma",
            "a", "e", "q", "traits",
        }


class TestMappingFromSampler:
    """End-to-end sanity: every sampled population maps cleanly."""

    def test_sampler_outputs_map_without_error(self) -> None:
        traits = sample_trait_vectors(n=200, seed=99)
        for row in traits:
            p = traits_to_parameters(row)
            assert p["rho"] >= p["epsilon"] - 1e-12
            assert 0.0 <= p["gamma"] <= 0.8 + 1e-12
            assert -1.0 <= p["x"] <= 1.0
