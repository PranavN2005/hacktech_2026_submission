"""
Unit + small integration tests for the new modular dynamics path.

Covers:
    - SimulationConfig validation
    - dynamics.py helpers (compatibility, exposure, effective_weights, phi)
    - SimulationEngine.step_with_config for each model_type
    - exposure modes (all_followed, top_k, sampled)
    - repulsion behaviour, no-compatible-neighbour edge case, clipping
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.config import SimulationConfig
from backend.dynamics import (
    compatibility,
    compute_distances,
    effective_weights,
    exposure_mask,
    phi_repulsive,
    safe_row_normalize,
)
from backend.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_agents(
    tmp_path: Path,
    n: int = 8,
    *,
    susceptibility: float = 0.4,
    activity: float | None = None,
) -> Path:
    """Write a small deterministic agents JSON for tests."""
    agents = []
    for i in range(n):
        agent = {
            "id": i,
            "name": f"Agent {i}",
            "bio": f"Bio {i}",
            "initial_belief": float(np.linspace(-0.9, 0.9, n)[i]),
            "susceptibility": susceptibility,
        }
        if activity is not None:
            agent["activity"] = activity
        agents.append(agent)
    path = tmp_path / "agents.json"
    path.write_text(json.dumps(agents), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# SimulationConfig validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults_construct(self) -> None:
        cfg = SimulationConfig()
        assert cfg.model_type == "bounded_confidence"
        assert cfg.exposure_mode == "all_followed"

    def test_rho_must_be_at_least_epsilon(self) -> None:
        with pytest.raises(ValueError):
            SimulationConfig(confidence_epsilon=0.5, repulsion_threshold_rho=0.4)

    def test_negative_noise_rejected(self) -> None:
        with pytest.raises(ValueError):
            SimulationConfig(noise_sigma=-0.1)

    def test_unknown_model_rejected(self) -> None:
        with pytest.raises(ValueError):
            SimulationConfig(model_type="not-a-real-model")  # type: ignore[arg-type]

    def test_unknown_exposure_rejected(self) -> None:
        with pytest.raises(ValueError):
            SimulationConfig(exposure_mode="something-else")  # type: ignore[arg-type]

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            SimulationConfig(top_k_visible=0)


# ---------------------------------------------------------------------------
# dynamics.py helpers
# ---------------------------------------------------------------------------

class TestDynamicsHelpers:
    def test_safe_row_normalize_zero_row(self) -> None:
        X = np.array([[1.0, 1.0], [0.0, 0.0]])
        Y = safe_row_normalize(X)
        assert np.allclose(Y, np.array([[0.5, 0.5], [0.0, 0.0]]))

    def test_compatibility_confirmation_bias_decays_with_distance(self) -> None:
        dist = np.array([[0.0, 0.1], [0.1, 0.0]])
        C = compatibility("confirmation_bias", dist, alpha_decay=2.0)
        assert C[0, 0] == pytest.approx(1.0)
        # Closer agents have larger weight than further ones (sanity check).
        assert C[0, 1] < 1.0
        assert C[0, 1] == pytest.approx(np.exp(-0.2))

    def test_compatibility_bounded_confidence_is_indicator(self) -> None:
        dist = np.array([[0.0, 0.3, 0.6]])
        C = compatibility("bounded_confidence", dist, epsilon=0.4)
        assert np.allclose(C, np.array([[1.0, 1.0, 0.0]]))

    def test_compatibility_degroot_is_all_ones(self) -> None:
        dist = np.array([[0.0, 0.9], [0.9, 0.0]])
        C = compatibility("degroot", dist)
        assert np.allclose(C, np.ones_like(dist))

    def test_phi_repulsive_three_zones(self) -> None:
        delta = np.array([[0.1, 0.5, 0.95]])  # within ε / mid / beyond ρ
        phi = phi_repulsive(delta, epsilon=0.2, rho=0.7, gamma=0.5)
        assert phi[0, 0] == pytest.approx(0.1)       # attract
        assert phi[0, 1] == pytest.approx(0.0)       # ignore
        assert phi[0, 2] == pytest.approx(-0.5 * 0.95)  # repel

    def test_exposure_all_followed_equals_A(self) -> None:
        A = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        dist = np.zeros((3, 3))
        activity = np.ones(3)
        M = exposure_mask("all_followed", A, dist, activity=activity)
        assert np.array_equal(M, A)

    def test_exposure_top_k_keeps_only_k(self) -> None:
        # 5 nodes, node 0 follows nodes 1..4. Distances increase with index.
        A = np.zeros((5, 5))
        A[0, 1:] = 1
        # Node 0's belief = -1.0; others spread out.
        B = np.array([-1.0, -0.9, -0.5, 0.5, 1.0])
        dist = compute_distances(B)
        activity = np.ones(5)
        M = exposure_mask(
            "top_k",
            A,
            dist,
            activity=activity,
            selective_beta=5.0,
            top_k=2,
        )
        # Node 0's row should keep exactly two of its four followed neighbours.
        assert int(M[0].sum()) == 2
        # The two selected should be the closest (smallest distance).
        kept = np.where(M[0] > 0)[0].tolist()
        assert set(kept) == {1, 2}

    def test_exposure_sampled_is_reproducible(self) -> None:
        A = np.zeros((6, 6))
        A[0, 1:] = 1
        B = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
        dist = compute_distances(B)
        activity = np.ones(6)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        M_a = exposure_mask(
            "sampled", A, dist, activity=activity, top_k=3, rng=rng_a
        )
        M_b = exposure_mask(
            "sampled", A, dist, activity=activity, top_k=3, rng=rng_b
        )
        assert np.array_equal(M_a, M_b)
        assert int(M_a[0].sum()) == 3

    def test_effective_weights_zero_row_when_no_compat(self) -> None:
        A = np.array([[0, 1], [1, 0]], dtype=np.float64)
        W_base = A
        M = A
        # Compatibility everywhere zero ⟹ row sums zero ⟹ row stays zero.
        C = np.zeros_like(A)
        W = effective_weights(A, W_base, M, C)
        assert np.allclose(W, 0.0)


# ---------------------------------------------------------------------------
# Engine: legacy preservation + new modes
# ---------------------------------------------------------------------------

class TestEngineModes:
    def test_step_with_config_runs_each_mode(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, n=12)
        for mt in ("degroot", "confirmation_bias", "bounded_confidence", "repulsive_bc"):
            cfg = SimulationConfig(model_type=mt)  # type: ignore[arg-type]
            eng = SimulationEngine(path, seed=11, min_out_degree=3)
            state = eng.step_with_config(cfg)
            assert np.isfinite(state.beliefs).all()
            assert np.all(state.beliefs >= -1.0) and np.all(state.beliefs <= 1.0)
            assert state.step == 1

    def test_clipping_keeps_beliefs_in_range_under_noise(self, tmp_path: Path) -> None:
        # Large noise should still produce in-range beliefs because clip_beliefs=True.
        path = write_agents(tmp_path, n=10, susceptibility=0.9)
        cfg = SimulationConfig(model_type="degroot", noise_sigma=2.0)
        eng = SimulationEngine(path, seed=3, min_out_degree=3)
        for _ in range(5):
            state = eng.step_with_config(cfg)
            assert np.all(state.beliefs >= -1.0) and np.all(state.beliefs <= 1.0)

    def test_confirmation_bias_far_neighbour_smaller_weight(
        self, tmp_path: Path
    ) -> None:
        # Two-followed-neighbour agent: one nearby, one far. Confirmation-bias
        # should pull the agent toward the near neighbour more than the far.
        agents = [
            {"id": 0, "name": "A", "bio": "", "initial_belief": 0.0, "susceptibility": 1.0},
            {"id": 1, "name": "B", "bio": "", "initial_belief": 0.05, "susceptibility": 1.0},  # near
            {"id": 2, "name": "C", "bio": "", "initial_belief": 0.95, "susceptibility": 1.0},  # far
        ]
        path = tmp_path / "agents.json"
        path.write_text(json.dumps(agents), encoding="utf-8")
        eng = SimulationEngine(path, seed=0, min_out_degree=2)
        # Override A so node 0 deterministically follows nodes 1 and 2 only,
        # and ensure no incoming influence pulls 1/2 in this single step.
        eng._A = np.array(  # noqa: SLF001 (deliberate test-time injection)
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float64
        )
        cfg = SimulationConfig(
            model_type="confirmation_bias",
            distance_decay_alpha=5.0,
        )
        state = eng.step_with_config(cfg)
        # The far neighbour (0.95) and near neighbour (0.05) both pull node 0
        # upward; with strong α the near one dominates, so node 0 ends up
        # closer to 0.05 than to (0.05+0.95)/2 = 0.5.
        assert state.beliefs[0] < 0.5
        assert state.beliefs[0] > 0.0

    def test_bounded_confidence_far_neighbour_zero_weight(
        self, tmp_path: Path
    ) -> None:
        agents = [
            {"id": 0, "name": "A", "bio": "", "initial_belief": 0.0, "susceptibility": 1.0},
            {"id": 1, "name": "B", "bio": "", "initial_belief": 0.05, "susceptibility": 1.0},
            {"id": 2, "name": "C", "bio": "", "initial_belief": 0.95, "susceptibility": 1.0},
        ]
        path = tmp_path / "agents.json"
        path.write_text(json.dumps(agents), encoding="utf-8")
        eng = SimulationEngine(path, seed=0, min_out_degree=2)
        eng._A = np.array(  # noqa: SLF001
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float64
        )
        # ε = 0.2 ⟹ neighbour at 0.05 is in, neighbour at 0.95 is out.
        cfg = SimulationConfig(
            model_type="bounded_confidence",
            confidence_epsilon=0.2,
            repulsion_threshold_rho=0.9,
        )
        state = eng.step_with_config(cfg)
        # Only the near neighbour contributes ⟹ new b[0] should equal 0.05.
        assert state.beliefs[0] == pytest.approx(0.05, abs=1e-9)

    def test_no_compatible_neighbours_freezes_agent(self, tmp_path: Path) -> None:
        agents = [
            {"id": 0, "name": "A", "bio": "", "initial_belief": 0.0, "susceptibility": 1.0},
            {"id": 1, "name": "B", "bio": "", "initial_belief": 0.99, "susceptibility": 1.0},
        ]
        path = tmp_path / "agents.json"
        path.write_text(json.dumps(agents), encoding="utf-8")
        eng = SimulationEngine(path, seed=0, min_out_degree=1)
        eng._A = np.array([[0, 1], [0, 0]], dtype=np.float64)  # noqa: SLF001
        cfg = SimulationConfig(
            model_type="bounded_confidence",
            confidence_epsilon=0.1,
            repulsion_threshold_rho=0.5,
        )
        state = eng.step_with_config(cfg)
        # Distance is 0.99, ε is 0.1 ⟹ no compatible neighbour ⟹ b[0] frozen.
        assert state.beliefs[0] == pytest.approx(0.0, abs=1e-9)
        assert np.isfinite(state.beliefs).all()
        assert state.frac_no_compatible is not None
        assert state.frac_no_compatible > 0.0

    def test_top_k_exposure_only_top_k_visible(self, tmp_path: Path) -> None:
        # Engine-level: with top_k=1 and a clear ordering, agent 0 keeps just
        # the closest of its many followed neighbours.
        n = 8
        path = write_agents(tmp_path, n=n)
        eng = SimulationEngine(path, seed=0, min_out_degree=3)
        # Force agent 0 to follow everyone else.
        eng._A = np.zeros((n, n), dtype=np.float64)  # noqa: SLF001
        eng._A[0, 1:] = 1.0
        cfg = SimulationConfig(
            model_type="bounded_confidence",
            exposure_mode="top_k",
            top_k_visible=1,
            confidence_epsilon=2.0,  # never triggers — isolate exposure effect
            repulsion_threshold_rho=2.0,
            selective_exposure_beta=10.0,
        )
        # Dry-run the exposure helper directly to inspect M.
        dist = compute_distances(eng._B)  # noqa: SLF001
        M = exposure_mask(
            "top_k",
            eng._A,  # noqa: SLF001
            dist,
            activity=eng._activity,  # noqa: SLF001
            selective_beta=10.0,
            top_k=1,
        )
        assert int(M[0].sum()) == 1

    def test_repulsion_pushes_agent_away(self, tmp_path: Path) -> None:
        # Two agents far apart, mutual follow. Repulsion should push them
        # further apart than they already are, not closer.
        agents = [
            {"id": 0, "name": "A", "bio": "", "initial_belief": -0.8, "susceptibility": 0.5},
            {"id": 1, "name": "B", "bio": "", "initial_belief":  0.8, "susceptibility": 0.5},
        ]
        path = tmp_path / "agents.json"
        path.write_text(json.dumps(agents), encoding="utf-8")
        eng = SimulationEngine(path, seed=0, min_out_degree=1)
        eng._A = np.array([[0, 1], [1, 0]], dtype=np.float64)  # noqa: SLF001
        cfg = SimulationConfig(
            model_type="repulsive_bc",
            confidence_epsilon=0.1,
            repulsion_threshold_rho=0.5,
            repulsion_strength_gamma=0.5,
            clip_beliefs=True,
        )
        state = eng.step_with_config(cfg)
        # |Δ| = 1.6 > ρ = 0.5 ⟹ repulsion in effect ⟹ each moves away from
        # the other (subject to clipping at ±1).
        assert state.beliefs[0] < -0.8 + 1e-12
        assert state.beliefs[1] >  0.8 - 1e-12

    def test_legacy_step_unchanged_after_refactor(self, tmp_path: Path) -> None:
        # The legacy `step(alpha, beta, epsilon)` API must still produce
        # identical output to a fresh engine on the same seed.
        path = write_agents(tmp_path, n=20)
        a = SimulationEngine(path, seed=99, min_out_degree=3)
        b = SimulationEngine(path, seed=99, min_out_degree=3)
        for _ in range(4):
            sa = a.step(alpha=0.6, beta=0.1, epsilon=0.5)
            sb = b.step(alpha=0.6, beta=0.1, epsilon=0.5)
            assert np.allclose(sa.beliefs, sb.beliefs)

    def test_step_with_config_deterministic_on_same_seed(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, n=15)
        cfg = SimulationConfig(
            model_type="bounded_confidence",
            exposure_mode="sampled",
            top_k_visible=3,
            noise_sigma=0.05,
        )
        a = SimulationEngine(path, seed=7, min_out_degree=4)
        b = SimulationEngine(path, seed=7, min_out_degree=4)
        for _ in range(3):
            sa = a.step_with_config(cfg)
            sb = b.step_with_config(cfg)
            assert np.allclose(sa.beliefs, sb.beliefs)

    def test_diagnostics_populated(self, tmp_path: Path) -> None:
        path = write_agents(tmp_path, n=12)
        eng = SimulationEngine(path, seed=0, min_out_degree=3)
        cfg = SimulationConfig(model_type="bounded_confidence")
        state = eng.step_with_config(cfg)
        assert state.mean_pairwise_distance is not None
        assert state.active_exposures is not None
        assert state.active_exposures > 0
        assert state.frac_no_compatible is not None
        assert state.mean_exposure_similarity is not None
