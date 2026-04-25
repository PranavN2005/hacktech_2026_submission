import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.engine import SimulationEngine
from backend.main import app
import backend.main as main_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODE_REQUIRED_KEYS = {
    "id",
    "name",
    "bio",
    "initial_belief",
    "susceptibility",
    "social_capital",
}

_DEFAULTS_REQUIRED_KEYS = {"alpha", "beta", "epsilon", "steps", "interval"}

_STREAM_REQUIRED_KEYS = {"step", "beliefs", "polarization", "echo_coefficient"}


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


def parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


# ---------------------------------------------------------------------------
# Engine unit tests
# ---------------------------------------------------------------------------

def test_step_keeps_beliefs_finite(tmp_path: Path) -> None:
    engine = SimulationEngine(write_agents(tmp_path, n=20), seed=7, min_out_degree=3)
    state = engine.step(alpha=0.5, beta=0.2, epsilon=0.4)
    assert np.isfinite(state.beliefs).all()


def test_step_keeps_beliefs_bounded(tmp_path: Path) -> None:
    engine = SimulationEngine(write_agents(tmp_path, n=20), seed=7, min_out_degree=3)
    state = engine.step(alpha=0.5, beta=0.2, epsilon=0.4)
    assert np.all(state.beliefs >= -1.0)
    assert np.all(state.beliefs <= 1.0)


def test_step_output_length_matches_agent_count(tmp_path: Path) -> None:
    engine = SimulationEngine(write_agents(tmp_path, n=20), seed=7, min_out_degree=3)
    state = engine.step(alpha=0.5, beta=0.2, epsilon=0.4)
    assert len(state.beliefs) == 20


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


def test_engine_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SimulationEngine(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# /init — response contract
# ---------------------------------------------------------------------------

class TestInit:
    def test_status_200(self) -> None:
        assert TestClient(app).get("/init").status_code == 200

    def test_top_level_keys(self) -> None:
        data = TestClient(app).get("/init").json()
        assert {"agent_count", "nodes", "edges", "defaults"} <= data.keys()

    def test_agent_count_matches_engine(self) -> None:
        data = TestClient(app).get("/init").json()
        assert data["agent_count"] == main_module.engine.N

    def test_nodes_length_matches_agent_count(self) -> None:
        data = TestClient(app).get("/init").json()
        assert len(data["nodes"]) == data["agent_count"]

    def test_node_schema(self) -> None:
        data = TestClient(app).get("/init").json()
        for node in data["nodes"]:
            assert _NODE_REQUIRED_KEYS <= node.keys(), f"node {node.get('id')} missing keys"

    def test_node_id_is_index_aligned(self) -> None:
        """nodes[i]['id'] must equal i so belief arrays can be index-matched."""
        data = TestClient(app).get("/init").json()
        for i, node in enumerate(data["nodes"]):
            assert node["id"] == i

    def test_edges_have_from_and_to(self) -> None:
        data = TestClient(app).get("/init").json()
        assert len(data["edges"]) > 0
        assert "from" in data["edges"][0] and "to" in data["edges"][0]

    def test_defaults_keys(self) -> None:
        data = TestClient(app).get("/init").json()
        assert _DEFAULTS_REQUIRED_KEYS <= data["defaults"].keys()


# ---------------------------------------------------------------------------
# /init — agent_quantity parameter
# ---------------------------------------------------------------------------

class TestAgentQuantity:
    def test_custom_quantity_returns_correct_count(self) -> None:
        data = TestClient(app).get("/init?agent_quantity=50").json()
        assert data["agent_count"] == 50
        assert len(data["nodes"]) == 50

    def test_custom_quantity_nodes_are_index_aligned(self) -> None:
        data = TestClient(app).get("/init?agent_quantity=50").json()
        for i, node in enumerate(data["nodes"]):
            assert node["id"] == i

    def test_stream_belief_count_matches_init_quantity(self) -> None:
        client = TestClient(app)
        init_data = client.get("/init?agent_quantity=37").json()
        events = parse_sse(
            client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0").text
        )
        assert len(events[0]["beliefs"]) == init_data["agent_count"] == 37

    @pytest.mark.parametrize("bad", [0, 501])
    def test_out_of_bounds_quantity_rejected(self, bad: int) -> None:
        assert TestClient(app).get(f"/init?agent_quantity={bad}").status_code == 422


# ---------------------------------------------------------------------------
# /stream — response contract + validation
# ---------------------------------------------------------------------------

class TestStream:
    def test_status_200(self) -> None:
        resp = TestClient(app).get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0")
        assert resp.status_code == 200

    def test_content_type_is_event_stream(self) -> None:
        resp = TestClient(app).get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0")
        assert "text/event-stream" in resp.headers["content-type"]

    def test_correct_number_of_events_emitted(self) -> None:
        resp = TestClient(app).get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=3&interval=0")
        assert len(parse_sse(resp.text)) == 3

    def test_event_payload_keys(self) -> None:
        resp = TestClient(app).get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0")
        payload = parse_sse(resp.text)[0]
        assert _STREAM_REQUIRED_KEYS <= payload.keys()

    def test_belief_count_matches_agent_count(self) -> None:
        client = TestClient(app)
        n = client.get("/init").json()["agent_count"]
        payload = parse_sse(
            client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0").text
        )[0]
        assert len(payload["beliefs"]) == n

    def test_alpha_plus_beta_gt_1_rejected(self) -> None:
        resp = TestClient(app).get("/stream?alpha=0.8&beta=0.5&epsilon=0.4&steps=1&interval=0")
        assert resp.status_code == 422

    def test_alpha_plus_beta_gt_1_error_message(self) -> None:
        resp = TestClient(app).get("/stream?alpha=0.9&beta=0.3&epsilon=0.4")
        assert "alpha + beta must be ≤ 1.0" in resp.json()["detail"]
