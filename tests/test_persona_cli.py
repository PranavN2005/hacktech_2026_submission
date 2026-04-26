"""
Tests for the `generate_population.py` CLI.

These exercise the argparse wiring + subcommand handlers without making
any real LLM calls. The bios subcommand is covered in
`test_persona_pipeline.py` via direct invocation of `generate_bios`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add the repo root to sys.path so we can import the CLI module by name.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_population  # noqa: E402


class TestSampleCommand:
    def test_sample_writes_population(self, tmp_path: Path,
                                       capsys: pytest.CaptureFixture) -> None:
        out = tmp_path / "pop.json"
        rc = generate_population.main(
            ["sample", "--n", "5", "--seed", "0", "--out", str(out)]
        )
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["metadata"]["pipeline_stage"] == "parameters_complete"
        assert len(data["agents"]) == 5
        captured = capsys.readouterr().out
        assert "no LLM used" in captured

    def test_sample_requires_n_seed_out(self) -> None:
        with pytest.raises(SystemExit):
            generate_population.main(["sample"])


class TestValidateCommand:
    def test_validate_runs(self, tmp_path: Path,
                            capsys: pytest.CaptureFixture) -> None:
        out = tmp_path / "pop.json"
        generate_population.main(
            ["sample", "--n", "20", "--seed", "1", "--out", str(out)]
        )
        capsys.readouterr()  # clear
        rc = generate_population.main(["validate", "--input", str(out)])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "Population summary" in captured


class TestLoadCheckCommand:
    def test_load_check_runs(self, tmp_path: Path,
                              capsys: pytest.CaptureFixture) -> None:
        out = tmp_path / "pop.json"
        generate_population.main(
            ["sample", "--n", "8", "--seed", "2", "--out", str(out)]
        )
        capsys.readouterr()
        rc = generate_population.main(["load-check", "--input", str(out)])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "Loaded 8 agents" in captured
        assert "First 3 agents" in captured


class TestBiosCommandErrors:
    def test_bios_without_api_key_errors(self, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "pop.json"
        generate_population.main(
            ["sample", "--n", "2", "--seed", "3", "--out", str(out)]
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # No API key + no --api-key should exit with code 1.
        rc = generate_population.main(
            ["bios", "--input", str(out), "--model", "gpt-4o-mini"]
        )
        assert rc == 1

    def test_bios_rejects_unsampled_input(self, tmp_path: Path) -> None:
        # Make a population whose pipeline_stage is bogus.
        out = tmp_path / "pop.json"
        generate_population.main(
            ["sample", "--n", "2", "--seed", "4", "--out", str(out)]
        )
        data = json.loads(out.read_text())
        data["metadata"]["pipeline_stage"] = "draft"
        out.write_text(json.dumps(data), encoding="utf-8")

        # Wire in a stub LLM by monkeypatching the provider builder.
        import generate_population as gp

        def stub_client(*_args, **_kwargs):
            return lambda prompt, model: "x"

        # Override the API-key resolver so it doesn't try to read env.
        gp._build_llm_client = lambda *a, **kw: (lambda p, m: "x")  # type: ignore
        gp._resolve_api_key = lambda *a, **kw: "fake"  # type: ignore

        rc = gp.main([
            "bios", "--input", str(out), "--model", "stub",
            "--api-key", "fake",
        ])
        assert rc == 1  # ValueError raised → returned as 1
