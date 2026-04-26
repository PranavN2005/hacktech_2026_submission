#!/usr/bin/env python3
"""
Reproducible side-by-side comparison of the four model modes on the
*same* initial graph and *same* initial beliefs.

What it prints
--------------
For each model_type in {degroot, confirmation_bias, bounded_confidence,
repulsive_bc} the script reports:
    - final Esteban-Ray polarisation,
    - the time series of Esteban-Ray,
    - the time series of mean pairwise belief distance.

Usage
-----
    .venv/bin/python scripts/experiment_modes.py
    .venv/bin/python scripts/experiment_modes.py --steps 50 --agents 100
    .venv/bin/python scripts/experiment_modes.py --save-csv runs.csv

If matplotlib is installed, pass --plot to dump a PNG per metric. The
script intentionally does not import matplotlib at the top level so
the default run has zero extra dependencies.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script: `python scripts/experiment_modes.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import SimulationConfig  # noqa: E402
from backend.engine import SimulationEngine  # noqa: E402


MODES: list[SimulationConfig] = [
    SimulationConfig(
        model_type="degroot",
        exposure_mode="all_followed",
        label="DeGroot baseline (all_followed)",
    ),
    SimulationConfig(
        model_type="confirmation_bias",
        exposure_mode="all_followed",
        distance_decay_alpha=3.0,
        label="Confirmation bias",
    ),
    SimulationConfig(
        model_type="bounded_confidence",
        exposure_mode="all_followed",
        confidence_epsilon=0.4,
        label="Bounded confidence",
    ),
    SimulationConfig(
        model_type="repulsive_bc",
        exposure_mode="all_followed",
        confidence_epsilon=0.3,
        repulsion_threshold_rho=0.7,
        repulsion_strength_gamma=0.4,
        label="Repulsive bounded confidence",
    ),
]


def run_mode(
    agents_path: Path,
    *,
    config: SimulationConfig,
    seed: int,
    agent_count: int,
    steps: int,
) -> dict:
    """Run a single mode on a fresh engine and collect time series."""
    engine = SimulationEngine(
        agents_path,
        agent_count=agent_count,
        seed=seed,
        min_out_degree=10,
    )
    polarisation: list[float] = []
    mean_distance: list[float] = []
    no_compat: list[float] = []

    for _ in range(steps):
        state = engine.step_with_config(config)
        polarisation.append(state.polarization)
        mean_distance.append(state.mean_pairwise_distance or 0.0)
        no_compat.append(state.frac_no_compatible or 0.0)

    return {
        "label": config.label or config.model_type,
        "model_type": config.model_type,
        "polarization_series": polarisation,
        "mean_distance_series": mean_distance,
        "frac_no_compatible_series": no_compat,
        "final_polarization": polarisation[-1] if polarisation else float("nan"),
        "final_mean_distance": mean_distance[-1] if mean_distance else float("nan"),
        "final_beliefs": engine._B.copy(),  # noqa: SLF001 (one-off script)
    }


def print_summary(results: list[dict]) -> None:
    print("\n=== Final-step summary ===")
    print(f"{'Mode':<35}  {'final ER':>10}  {'final ⟨d⟩':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['label']:<35}  "
            f"{r['final_polarization']:>10.4f}  "
            f"{r['final_mean_distance']:>10.4f}"
        )


def write_csv(path: Path, results: list[dict]) -> None:
    """One row per (mode, step) for easy plotting in any tool."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["mode", "step", "polarization", "mean_distance", "frac_no_compatible"]
        )
        for r in results:
            for t, (p, d, nc) in enumerate(
                zip(
                    r["polarization_series"],
                    r["mean_distance_series"],
                    r["frac_no_compatible_series"],
                )
            ):
                writer.writerow([r["model_type"], t, p, d, nc])
    print(f"\nwrote {path}")


def maybe_plot(results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib not installed; skipping --plot", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for r in results:
        ax.plot(r["polarization_series"], label=r["label"])
    ax.set_xlabel("step")
    ax.set_ylabel("Esteban-Ray polarisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "polarization.png", dpi=150)

    fig, ax = plt.subplots()
    for r in results:
        ax.plot(r["mean_distance_series"], label=r["label"])
    ax.set_xlabel("step")
    ax.set_ylabel("mean pairwise belief distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_distance.png", dpi=150)

    print(f"wrote plots to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--agents-path",
        type=Path,
        default=Path(os.environ.get("AGENTS_PATH", "pop100.json")),
    )
    ap.add_argument("--agents", type=int, default=200, help="how many agents to sample")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-csv", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-dir", type=Path, default=Path("scripts/plots"))
    args = ap.parse_args()

    if not args.agents_path.exists():
        raise SystemExit(f"agents file not found: {args.agents_path}")

    results: list[dict] = []
    for cfg in MODES:
        print(f"running {cfg.label or cfg.model_type} ...")
        results.append(
            run_mode(
                args.agents_path,
                config=cfg,
                seed=args.seed,
                agent_count=args.agents,
                steps=args.steps,
            )
        )

    print_summary(results)

    if args.save_csv:
        write_csv(args.save_csv, results)

    if args.plot:
        maybe_plot(results, args.plot_dir)


if __name__ == "__main__":
    main()
