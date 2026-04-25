"""
API-level tests for EchoChamber backend.

Install:  pip install pytest httpx fastapi[all]
Run:      pytest tests/test_engine.py -v
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app, engine as app_engine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /init
# ---------------------------------------------------------------------------

class TestInit:
    def test_status_200(self, client):
        resp = client.get("/init")
        assert resp.status_code == 200

    def test_top_level_keys(self, client):
        data = client.get("/init").json()
        assert {"agent_count", "nodes", "edges", "defaults"} <= data.keys()

    def test_agent_count_matches_engine(self, client):
        data = client.get("/init").json()
        assert data["agent_count"] == app_engine.N

    def test_nodes_length_matches_agent_count(self, client):
        data = client.get("/init").json()
        assert len(data["nodes"]) == data["agent_count"]

    def test_node_shape(self, client):
        data = client.get("/init").json()
        required = {"id", "name", "bio", "initial_belief", "susceptibility", "social_capital"}
        for node in data["nodes"]:
            assert required <= node.keys(), f"node {node.get('id')} missing keys"

    def test_node_id_is_index_aligned(self, client):
        """nodes[i]['id'] must equal i so belief arrays can be indexed directly."""
        data = client.get("/init").json()
        for i, node in enumerate(data["nodes"]):
            assert node["id"] == i

    def test_edges_have_from_to(self, client):
        data = client.get("/init").json()
        assert len(data["edges"]) > 0
        edge = data["edges"][0]
        assert "from" in edge and "to" in edge

    def test_defaults_keys(self, client):
        data = client.get("/init").json()
        assert {"alpha", "beta", "epsilon", "steps", "interval"} <= data["defaults"].keys()


# ---------------------------------------------------------------------------
# /stream
# ---------------------------------------------------------------------------

class TestStream:
    def _parse_sse(self, text: str) -> list[dict]:
        return [
            json.loads(line[len("data: "):])
            for line in text.splitlines()
            if line.startswith("data: ")
        ]

    def test_status_200_and_content_type(self, client):
        resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_emits_correct_number_of_events(self, client):
        resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=3&interval=0")
        events = self._parse_sse(resp.text)
        assert len(events) == 3

    def test_payload_keys(self, client):
        resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0")
        events = self._parse_sse(resp.text)
        assert len(events) >= 1
        assert {"step", "beliefs", "polarization", "echo_coefficient"} <= events[0].keys()

    def test_beliefs_length_matches_init_agent_count(self, client):
        init = client.get("/init").json()
        resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=1&interval=0")
        events = self._parse_sse(resp.text)
        assert len(events[0]["beliefs"]) == init["agent_count"]

    def test_no_nan_or_inf_in_beliefs(self, client):
        resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=5&interval=0")
        events = self._parse_sse(resp.text)
        for ev in events:
            beliefs = np.array(ev["beliefs"])
            assert np.isfinite(beliefs).all(), f"NaN/inf detected at step {ev['step']}"

    def test_beliefs_bounded(self, client):
        resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=5&interval=0")
        events = self._parse_sse(resp.text)
        for ev in events:
            beliefs = np.array(ev["beliefs"])
            assert (beliefs >= -1.0).all() and (beliefs <= 1.0).all()

    def test_step_counter_increments(self, client):
        resp = client.get("/stream?alpha=0.5&beta=0.2&epsilon=0.4&steps=3&interval=0")
        events = self._parse_sse(resp.text)
        assert [ev["step"] for ev in events] == [1, 2, 3]

    def test_alpha_plus_beta_gt_1_rejected(self, client):
        resp = client.get("/stream?alpha=0.8&beta=0.5&epsilon=0.4&steps=1&interval=0")
        assert resp.status_code == 422
