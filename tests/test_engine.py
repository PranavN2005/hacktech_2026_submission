import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.engine import SimulationEngine
from backend.main import app, engine as app_engine


def test_step_keeps_beliefs_finite_and_bounded() -> None:
    engine = SimulationEngine(n=120, m=3, seed=7)
    state = engine.step(alpha=0.5, beta=0.2, epsilon=0.4)

    assert len(state.beliefs) == 120
    assert np.isfinite(state.beliefs).all()
    assert np.all(state.beliefs >= -1.0)
    assert np.all(state.beliefs <= 1.0)


def test_fixed_seed_is_deterministic() -> None:
    eng_a = SimulationEngine(n=80, m=3, seed=99)
    eng_b = SimulationEngine(n=80, m=3, seed=99)

    for _ in range(5):
        state_a = eng_a.step(alpha=0.6, beta=0.1, epsilon=0.5)
        state_b = eng_b.step(alpha=0.6, beta=0.1, epsilon=0.5)
        assert np.allclose(state_a.beliefs, state_b.beliefs)
        assert state_a.polarization == state_b.polarization
        assert state_a.echo_coefficient == state_b.echo_coefficient


def test_stream_emits_valid_sse_payloads() -> None:
    client = TestClient(app)
    response = client.get(
        "/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=2&interval=0"
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    data_lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert len(data_lines) == 2

    payload = json.loads(data_lines[0].replace("data: ", "", 1))
    assert {"step", "beliefs", "polarization", "echo_coefficient"} <= payload.keys()
    assert len(payload["beliefs"]) == app_engine.n


def test_engine_loads_minimal_persona_fields_from_json(tmp_path: Path) -> None:
    agents = [
        {"id": "a1", "initial_belief": -0.6, "susceptibility": 0.25},
        {"id": "a2", "initial_belief": 0.1, "susceptibility": 0.7},
        {"id": "a3", "initial_belief": 0.9, "susceptibility": 0.4},
        {"id": "a4", "initial_belief": -0.2, "susceptibility": 0.55},
    ]
    agents_path = tmp_path / "agents.json"
    agents_path.write_text(json.dumps(agents), encoding="utf-8")

    engine = SimulationEngine(m=2, seed=1, agents_path=str(agents_path))
    state = engine.step(alpha=0.5, beta=0.2, epsilon=0.4)

    assert engine.n == 4
    assert engine.agent_ids == ["a1", "a2", "a3", "a4"]
    assert len(state.beliefs) == 4


def test_engine_rejects_missing_required_persona_fields(tmp_path: Path) -> None:
    broken_agents = [
        {"id": "a1", "initial_belief": 0.2},
        {"id": "a2", "initial_belief": -0.1, "susceptibility": 0.4},
    ]
    agents_path = tmp_path / "agents.json"
    agents_path.write_text(json.dumps(broken_agents), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required key 'susceptibility'"):
        SimulationEngine(m=1, seed=1, agents_path=str(agents_path))
