import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.engine import SimulationEngine
from backend.main import app
import backend.main as main_module


def write_agents(tmp_path: Path, n: int = 8) -> Path:
    agents = [
        {
            "id": i,
            "name": f"Agent {i}",
            "bio": f"Bio {i}",
            "initial_belief": float(np.linspace(-0.9, 0.9, n)[i]),
            "susceptibility": 0.4,
        }
        for i in range(n)
    ]
    path = tmp_path / "agents.json"
    path.write_text(json.dumps(agents), encoding="utf-8")
    return path


def test_step_keeps_beliefs_finite_and_bounded(tmp_path: Path) -> None:
    agents_path = write_agents(tmp_path, n=20)
    engine = SimulationEngine(agents_path, seed=7, min_out_degree=3)
    state = engine.step(alpha=0.5, beta=0.2, epsilon=0.4)

    assert len(state.beliefs) == 20
    assert np.isfinite(state.beliefs).all()
    assert np.all(state.beliefs >= -1.0)
    assert np.all(state.beliefs <= 1.0)


def test_fixed_seed_is_deterministic(tmp_path: Path) -> None:
    agents_path = write_agents(tmp_path, n=20)
    eng_a = SimulationEngine(agents_path, seed=99, min_out_degree=3)
    eng_b = SimulationEngine(agents_path, seed=99, min_out_degree=3)

    for _ in range(5):
        state_a = eng_a.step(alpha=0.6, beta=0.1, epsilon=0.5)
        state_b = eng_b.step(alpha=0.6, beta=0.1, epsilon=0.5)
        assert np.allclose(state_a.beliefs, state_b.beliefs)
        assert state_a.polarization == state_b.polarization
        assert state_a.echo_coefficient == state_b.echo_coefficient


def test_init_and_stream_contracts() -> None:
    client = TestClient(app)

    init_resp = client.get("/init")
    assert init_resp.status_code == 200
    init_payload = init_resp.json()
    assert {"agent_count", "nodes", "edges", "defaults"} <= init_payload.keys()
    assert init_payload["agent_count"] == len(init_payload["nodes"])

    stream_resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=2&interval=0")
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]

    data_lines = [line for line in stream_resp.text.splitlines() if line.startswith("data: ")]
    assert len(data_lines) == 2

    payload = json.loads(data_lines[0].replace("data: ", "", 1))
    assert {"step", "beliefs", "polarization", "echo_coefficient"} <= payload.keys()
    assert len(payload["beliefs"]) == init_payload["agent_count"]


def test_stream_rejects_invalid_alpha_beta_sum() -> None:
    client = TestClient(app)
    response = client.get("/stream?alpha=0.9&beta=0.3&epsilon=0.4")
    assert response.status_code == 422
    assert "alpha + beta must be ≤ 1.0" in response.json()["detail"]


def test_engine_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        SimulationEngine(missing)

def test_init_endpoint_node_contract() -> None:
    client = TestClient(app)
    data = client.get("/init").json()
    required = {
        "id",
        "name",
        "bio",
        "initial_belief",
        "susceptibility",
        "social_capital",
    }
    assert {"agent_count", "nodes", "edges", "defaults"} <= data.keys()
    assert data["agent_count"] == main_module.engine.N
    assert len(data["nodes"]) == data["agent_count"]
    for i, node in enumerate(data["nodes"]):
        assert required <= node.keys()
        assert node["id"] == i


def test_agent_quantity_bounds_and_stream_alignment() -> None:
    client = TestClient(app)
    init = client.get("/init?agent_quantity=37")
    assert init.status_code == 200
    init_data = init.json()
    assert init_data["agent_count"] == 37
    assert len(init_data["nodes"]) == 37

    stream = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0")
    events = [
        json.loads(line[len("data: "):])
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert len(events[0]["beliefs"]) == 37

    assert client.get("/init?agent_quantity=0").status_code == 422
    assert client.get("/init?agent_quantity=501").status_code == 422
