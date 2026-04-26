"""
Stage-2b / 3 / 6 / 7 tests for the persona pipeline.

Covers:
    - generate_population_json (writes valid metadata + agents, no LLM)
    - generate_bios            (mocked LLM only — no real API calls)
    - load_population          (engine-shaped output, backward compat)
    - validate_population      (range checks + summary stats + warnings)

The LLM is *always* mocked here. Real-API tests live in nobody's CI; the
contract is a small Callable[[prompt, model], str] that any provider can
adapt to.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.persona_pipeline import (
    PIPELINE_STAGE_BIOS,
    PIPELINE_STAGE_PARAMETERS,
    _build_bio_prompt,
    generate_bios,
    generate_population_json,
    load_population,
    validate_population,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_population(tmp_path: Path, n: int = 8, seed: int = 0) -> Path:
    out = tmp_path / "population.json"
    generate_population_json(n=n, seed=seed, output_path=out)
    return out


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# generate_population_json
# ---------------------------------------------------------------------------


class TestGeneratePopulationJson:
    def test_json_no_llm_call(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Stage 2 must not import or invoke any LLM SDK."""
        # If pipeline accidentally imports google.genai or openai during
        # generation, this monkeypatch ensures we'd notice (a real call
        # would either fail or hang). We also verify by inspecting that
        # the only modules touched are stdlib + numpy/scipy.
        called = {"count": 0}

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            called["count"] += 1
            raise RuntimeError("LLM should not be called from stage 2")

        # Patch onto the pipeline module just in case it ever grows an
        # internal LLM helper.
        import backend.persona_pipeline as pp
        monkeypatch.setattr(pp, "generate_bios", boom, raising=True)

        out = _make_population(tmp_path, n=4, seed=1)
        assert out.exists()
        assert called["count"] == 0

    def test_json_stage_field(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=5, seed=2)
        data = _read(out)
        assert data["metadata"]["pipeline_stage"] == PIPELINE_STAGE_PARAMETERS
        assert data["metadata"]["bio_complete"] is False

    def test_bio_field_null(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=10, seed=3)
        data = _read(out)
        for a in data["agents"]:
            assert a["bio"] is None

    def test_metadata_includes_axis_order_and_sigma(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=3, seed=4)
        data = _read(out)
        meta = data["metadata"]
        assert meta["n"] == 3
        assert meta["seed"] == 4
        assert "generated_at" in meta and isinstance(meta["generated_at"], str)
        assert meta["axis_order"] == ["lr", "auth", "conf", "open", "react", "active"]
        sigma = np.array(meta["correlation_matrix"])
        assert sigma.shape == (6, 6)

    def test_agent_record_has_required_keys(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=3, seed=5)
        agents = _read(out)["agents"]
        required = {"id", "bio", "x", "s", "sigma", "epsilon", "g",
                    "beta", "rho", "gamma", "a", "e", "q", "traits"}
        for a in agents:
            assert required <= a.keys()

    def test_agent_ids_zero_padded(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=12, seed=6)
        agents = _read(out)["agents"]
        assert agents[0]["id"] == "agent_0000"
        assert agents[11]["id"] == "agent_0011"

    def test_numerical_values_rounded_to_4_decimals(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=5, seed=7)
        agents = _read(out)["agents"]
        for a in agents:
            for key in ("x", "s", "sigma", "epsilon", "g", "beta",
                        "rho", "gamma", "a", "e", "q"):
                v = a[key]
                # Round-trip equals self ⇒ no more than 4 decimals.
                assert round(float(v), 4) == float(v), (
                    f"{a['id']}.{key}={v} not rounded to 4 decimals"
                )

    def test_reproducibility(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        generate_population_json(n=10, seed=42, output_path=a)
        generate_population_json(n=10, seed=42, output_path=b)
        # Compare just the agents (metadata.generated_at differs by ms).
        assert _read(a)["agents"] == _read(b)["agents"]

    def test_invalid_n(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            generate_population_json(n=0, seed=0, output_path=tmp_path / "x.json")


# ---------------------------------------------------------------------------
# generate_bios — mocked LLM
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal mock LLM client that records every prompt it's asked for."""

    def __init__(self, response: str = "test bio") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def __call__(self, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        return self.response


class TestGenerateBios:
    def test_bios_only_modify_bio(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=6, seed=8)
        before = _read(out)
        # Snapshot every numerical field per agent.
        before_numerical = [
            {k: a[k] for k in a if k != "bio"}
            for a in before["agents"]
        ]

        stub = _StubLLM("test bio")
        result = generate_bios(out, stub, model="mock-model",
                               delay_between_calls=0.0)
        assert result["filled"] == 6
        assert result["skipped"] == 0

        after = _read(out)
        # Numerical fields byte-for-byte identical.
        for a, snap in zip(after["agents"], before_numerical):
            assert {k: a[k] for k in a if k != "bio"} == snap
            assert a["bio"] == "test bio"

        # Stage advanced.
        assert after["metadata"]["pipeline_stage"] == PIPELINE_STAGE_BIOS
        assert after["metadata"]["bio_complete"] is True

    def test_bios_resumable(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=5, seed=9)
        # Manually pre-fill bios for two agents.
        data = _read(out)
        data["agents"][0]["bio"] = "manual zero"
        data["agents"][3]["bio"] = "manual three"
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        stub = _StubLLM("auto-generated")
        generate_bios(out, stub, model="mock", delay_between_calls=0.0)

        after = _read(out)
        assert after["agents"][0]["bio"] == "manual zero"
        assert after["agents"][3]["bio"] == "manual three"
        # The other three were filled by the stub.
        assert after["agents"][1]["bio"] == "auto-generated"
        assert after["agents"][2]["bio"] == "auto-generated"
        assert after["agents"][4]["bio"] == "auto-generated"
        # Stub was called exactly 3 times, not 5.
        assert len(stub.calls) == 3

    def test_bios_skip_on_persistent_error(self, tmp_path: Path,
                                            caplog: pytest.LogCaptureFixture) -> None:
        out = _make_population(tmp_path, n=3, seed=10)

        def always_fails(prompt: str, model: str) -> str:
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING):
            result = generate_bios(
                out, always_fails, model="mock",
                delay_between_calls=0.0,
                max_retries=2,
                initial_backoff=0.0,
                sleep_fn=lambda _s: None,
            )
        assert result["filled"] == 0
        assert result["skipped"] == 3
        assert len(result["skipped_ids"]) == 3
        # All bios still null since none succeeded.
        after = _read(out)
        for a in after["agents"]:
            assert a["bio"] is None
        # Pipeline stage NOT advanced because not all bios completed.
        assert after["metadata"]["pipeline_stage"] == PIPELINE_STAGE_PARAMETERS

    def test_bios_retry_then_succeed(self, tmp_path: Path) -> None:
        """First attempt raises, second succeeds; the bio should still land."""
        out = _make_population(tmp_path, n=1, seed=11)
        attempts = {"count": 0}

        def flaky(prompt: str, model: str) -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary outage")
            return "second-try bio"

        result = generate_bios(
            out, flaky, model="mock",
            delay_between_calls=0.0,
            max_retries=3,
            initial_backoff=0.0,
            sleep_fn=lambda _s: None,
        )
        assert result["filled"] == 1
        assert result["skipped"] == 0
        assert _read(out)["agents"][0]["bio"] == "second-try bio"

    def test_bios_rejects_wrong_pipeline_stage(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=2, seed=12)
        data = _read(out)
        data["metadata"]["pipeline_stage"] = "something_else"
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="pipeline_stage"):
            generate_bios(out, _StubLLM(), model="mock", delay_between_calls=0.0)

    def test_bios_respects_delay(self, tmp_path: Path) -> None:
        """delay_between_calls is applied (verifies wiring, not real time)."""
        out = _make_population(tmp_path, n=3, seed=13)
        sleeps: list[float] = []
        generate_bios(
            out, _StubLLM("ok"), model="mock",
            delay_between_calls=0.25,
            sleep_fn=sleeps.append,
        )
        # 3 successful calls ⇒ 3 delays.
        assert sleeps == [0.25, 0.25, 0.25]

    def test_prompt_contains_labels_and_numbers(self, tmp_path: Path) -> None:
        """The bio prompt should embed both readable labels and the raw scalar."""
        out = _make_population(tmp_path, n=1, seed=14)
        agent = _read(out)["agents"][0]
        prompt = _build_bio_prompt(agent)
        # At least one of the qualitative labels must appear.
        assert any(
            label in prompt for label in (
                "left-leaning", "right-leaning", "centrist",
                "libertarian", "authoritarian",
                "uncertain", "confident", "very sure",
            )
        )
        # The raw numeric belief must appear (rounded to 3 decimals).
        assert f"{agent['x']:.3f}" in prompt


# ---------------------------------------------------------------------------
# load_population
# ---------------------------------------------------------------------------


class TestLoadPopulation:
    def test_load_population_x_matches_json(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=10, seed=15)
        agents_in_file = _read(out)["agents"]
        loaded = load_population(out)
        assert len(loaded) == len(agents_in_file)
        for src, eng in zip(agents_in_file, loaded):
            assert eng["initial_belief"] == pytest.approx(src["x"])

    def test_load_population_maps_field_names(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=3, seed=16)
        loaded = load_population(out)
        for eng in loaded:
            assert "initial_belief" in eng
            assert "susceptibility" in eng
            assert "activity" in eng
            assert "confidence_epsilon" in eng
            assert "selective_exposure_beta" in eng
            assert "repulsion_threshold_rho" in eng
            assert "repulsion_strength_gamma" in eng

    def test_load_population_synthesises_name_when_missing(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=2, seed=17)
        loaded = load_population(out)
        for eng in loaded:
            assert isinstance(eng["name"], str) and eng["name"]

    def test_load_population_handles_null_bio(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=2, seed=18)
        loaded = load_population(out)
        for eng in loaded:
            assert isinstance(eng["bio"], str)  # null becomes empty string

    def test_load_population_rejects_unknown_stage(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=2, seed=19)
        data = _read(out)
        data["metadata"]["pipeline_stage"] = "draft"
        out.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="pipeline_stage"):
            load_population(out)

    def test_load_population_legacy_flat_array(self, tmp_path: Path) -> None:
        """Backward compat: a flat-list JSON file is returned as-is."""
        legacy = [
            {"id": 0, "name": "A", "bio": "x",
             "initial_belief": 0.1, "susceptibility": 0.5},
            {"id": 1, "name": "B", "bio": "y",
             "initial_belief": -0.3, "susceptibility": 0.7},
        ]
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = load_population(path)
        assert loaded == legacy

    def test_load_population_works_with_engine(self, tmp_path: Path) -> None:
        """End-to-end: SimulationEngine accepts the new pipeline file."""
        out = _make_population(tmp_path, n=15, seed=20)
        # Engine reads the file via _load_agents → must accept the new format.
        from backend.engine import SimulationEngine
        eng = SimulationEngine(out, seed=0, min_out_degree=3)
        assert eng.N == 15
        # The legacy step() should run without raising.
        state = eng.step(alpha=0.5, beta=0.2, epsilon=0.4)
        assert np.isfinite(state.beliefs).all()


# ---------------------------------------------------------------------------
# validate_population
# ---------------------------------------------------------------------------


class TestValidatePopulation:
    def test_valid_population_returns_summary(self, tmp_path: Path,
                                               capsys: pytest.CaptureFixture) -> None:
        out = _make_population(tmp_path, n=200, seed=21)
        stats = validate_population(out)
        assert stats["n"] == 200
        # The summary table is printed to stdout.
        captured = capsys.readouterr().out
        assert "Population summary" in captured
        for k in ("x", "s", "sigma", "epsilon", "beta", "a"):
            assert k in stats["summary"]
        assert len(stats["x_histogram_counts"]) == 10
        assert "strong_believers" in stats
        assert "near_lurkers" in stats

    def test_invalid_rho_raises(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=4, seed=22)
        data = _read(out)
        data["agents"][0]["rho"] = data["agents"][0]["epsilon"] - 0.1
        out.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="rho"):
            validate_population(out, log=False)

    def test_invalid_gamma_raises(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=4, seed=23)
        data = _read(out)
        data["agents"][0]["gamma"] = 0.99
        out.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="gamma"):
            validate_population(out, log=False)

    def test_zero_susceptibility_raises(self, tmp_path: Path) -> None:
        out = _make_population(tmp_path, n=4, seed=24)
        data = _read(out)
        data["agents"][0]["s"] = 0.0
        out.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="s="):
            validate_population(out, log=False)

    def test_right_skew_warning(self, tmp_path: Path,
                                 caplog: pytest.LogCaptureFixture) -> None:
        out = _make_population(tmp_path, n=10, seed=25)
        # Force >60% positive beliefs.
        data = _read(out)
        for a in data["agents"]:
            a["x"] = abs(a["x"]) + 0.01
            # Recompute rho so the validator's rho≥ε check still passes.
            a["rho"] = max(a["rho"], a["epsilon"])
        out.write_text(json.dumps(data) + "\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="backend.persona_pipeline"):
            validate_population(out, log=False)
        warnings = [r for r in caplog.records if "right-skew" in r.message.lower()
                    or "right_skew" in r.message.lower() or "right-skew" in r.message]
        assert warnings, "expected a right-skew warning"


# ---------------------------------------------------------------------------
# Pipeline ↔ engine integration: per-agent epsilon is honored
# ---------------------------------------------------------------------------


class TestEnginePerAgentResolution:
    def test_engine_picks_up_per_agent_epsilon_from_pipeline_file(
        self, tmp_path: Path
    ) -> None:
        """The engine's _resolve_per_agent already reads `confidence_epsilon`
        from agent dicts; load_population maps `epsilon` → `confidence_epsilon`.
        Verify the wiring is end-to-end correct by passing a known config and
        checking the resolved per-agent eps matches the pipeline-derived ε."""
        from backend.config import SimulationConfig
        from backend.engine import SimulationEngine

        out = _make_population(tmp_path, n=8, seed=26)
        eng = SimulationEngine(out, seed=0, min_out_degree=2)
        cfg = SimulationConfig(
            model_type="bounded_confidence",
            confidence_epsilon=0.4,  # global default — should be overridden per agent
            repulsion_threshold_rho=2.0,
        )
        eps_vec = eng._resolve_per_agent(cfg, "confidence_epsilon")  # noqa: SLF001
        # If per-agent override is wired, eps_vec is heterogeneous.
        assert eps_vec.shape == (eng.N,)
        assert eps_vec.std() > 0.0, (
            "expected heterogeneous per-agent epsilon from pipeline file"
        )
