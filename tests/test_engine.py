import json

import numpy as np
from fastapi.testclient import TestClient

from backend.engine import SimulationEngine
from backend.main import app


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
    assert len(payload["beliefs"]) == 500
