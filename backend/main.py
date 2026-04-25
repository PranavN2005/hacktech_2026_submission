from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import SimulationConfig
from backend.engine import SimulationEngine

# ---------------------------------------------------------------------------
# App + engine setup
# ---------------------------------------------------------------------------

app = FastAPI(title="EchoChamber Backend", version="0.3.0")

_AGENTS_PATH = Path(os.environ.get("AGENTS_PATH", "agents_500_pro_max.json"))
if not _AGENTS_PATH.exists():
    raise RuntimeError(
        f"Agents file not found at {_AGENTS_PATH.resolve()}. "
        "Set the AGENTS_PATH env var or ensure agents_500_pro_max.json is present."
    )

def _make_engine(agent_quantity: int) -> SimulationEngine:
    return SimulationEngine(
        _AGENTS_PATH,
        agent_count=agent_quantity,
        seed=42,
        min_out_degree=10,
    )


engine = _make_engine(agent_quantity=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "n_agents": engine.N}


# ---------------------------------------------------------------------------
# Init – static graph + agent metadata (called once at frontend startup)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "alpha": 0.5,
    "beta": 0.2,
    "epsilon": 0.4,
    "steps": 100,
    "interval": 0.1,
    # New dynamics defaults (echoes SimulationConfig defaults)
    "model_type": "degroot",
    "exposure_mode": "all_followed",
    "top_k_visible": 10,
    "selective_exposure_beta": 2.0,
    "distance_decay_alpha": 2.0,
    "repulsion_threshold_rho": 0.9,
    "repulsion_strength_gamma": 0.5,
    "noise_sigma": 0.0,
}


@app.get("/init")
async def init(
    agent_quantity: int = Query(
        500,
        ge=1,
        le=500,
        description="Number of agents to simulate (randomly sampled up to 500)",
    )
) -> dict:
    """
    Return agent metadata, follow-graph topology, and simulation defaults.

    Node ordering in `nodes` is identical to the belief-array index order
    emitted by `/stream`, so `nodes[i].id == i` always holds.
    """
    global engine
    engine = _make_engine(agent_quantity=agent_quantity)

    sc = engine.social_capital  # shape (N,), index-aligned with beliefs
    nodes = [
        {
            # `id` is graph index-aligned with `/stream` beliefs[i].
            "id": i,
            # Preserve original persona id for inspector/debugging.
            "persona_id": a["id"],
            "name": a["name"],
            "bio": a["bio"],
            "initial_belief": float(a["initial_belief"]),
            "susceptibility": float(a["susceptibility"]),
            "social_capital": int(sc[i]),
        }
        for i, a in enumerate(engine.agents)
    ]
    edges = [{"from": int(i), "to": int(j)} for i, j in engine.graph_edges]
    return {
        "agent_count": engine.N,
        "nodes": nodes,
        "edges": edges,
        "defaults": _DEFAULTS,
    }


# ---------------------------------------------------------------------------
# Simulation stream (SSE)
# ---------------------------------------------------------------------------

@app.get("/stream")
async def stream(
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Echo-chamber strength (platform curation)"),
    beta: float = Query(0.2, ge=0.0, le=1.0, description="Virality / outrage bias (platform curation)"),
    epsilon: float = Query(0.4, ge=0.0, le=2.0, description="Bounded-confidence / repulsion onset threshold ε"),
    steps: int = Query(100, ge=1, le=1000, description="Number of simulation steps"),
    interval: float = Query(0.1, ge=0.0, le=10.0, description="Seconds between SSE events"),
    # ── Dynamics model ──────────────────────────────────────────────────
    model_type: str = Query(
        "degroot",
        description="degroot | confirmation_bias | bounded_confidence | repulsive_bc",
    ),
    exposure_mode: str = Query(
        "all_followed",
        description="all_followed | top_k | sampled",
    ),
    top_k_visible: int = Query(10, ge=1, le=100, description="Visible neighbours per tick (top_k / sampled)"),
    selective_exposure_beta: float = Query(2.0, ge=0.0, le=20.0, description="Selective-exposure strength β"),
    distance_decay_alpha: float = Query(2.0, ge=0.0, le=20.0, description="Confirmation-bias decay α"),
    repulsion_threshold_rho: float = Query(0.9, ge=0.0, le=2.0, description="Repulsion onset ρ (≥ ε)"),
    repulsion_strength_gamma: float = Query(0.5, ge=0.0, le=5.0, description="Repulsion gain γ"),
    noise_sigma: float = Query(0.0, ge=0.0, le=1.0, description="Std of i.i.d. Gaussian belief noise"),
):
    """
    Run the simulation and stream each step as an SSE event.

    When `model_type="degroot"` (the default) the legacy platform-curation
    path is used and `alpha`/`beta` drive feed scoring.  For all other
    model types the modular `step_with_config` path is used; `alpha`/`beta`
    have no effect and `epsilon` maps to `confidence_epsilon`.

    Each event payload:
        {
            "step":                    int,
            "beliefs":                 float[N],
            "polarization":            float,
            "polarization_normalized": float,
            "echo_coefficient":        float,
            // present when model_type != "degroot":
            "mean_pairwise_distance":  float | null,
            "frac_no_compatible":      float | null,
            "mean_exposure_similarity": float | null,
            "active_exposures":        int | null
        }
    """
    if model_type == "degroot" and alpha + beta > 1.0:
        raise HTTPException(
            status_code=422,
            detail="alpha + beta must be ≤ 1.0 (the remainder is the baseline/chronological weight).",
        )

    # Build config for the modular path (used for every model_type except degroot).
    # epsilon is re-used as confidence_epsilon so the existing UI slider maps naturally.
    rho = max(repulsion_threshold_rho, epsilon)  # enforce ρ ≥ ε silently on the server
    config: SimulationConfig | None = None
    if model_type != "degroot":
        try:
            config = SimulationConfig(
                model_type=model_type,  # type: ignore[arg-type]
                exposure_mode=exposure_mode,  # type: ignore[arg-type]
                top_k_visible=top_k_visible,
                selective_exposure_beta=selective_exposure_beta,
                distance_decay_alpha=distance_decay_alpha,
                confidence_epsilon=epsilon,
                repulsion_threshold_rho=rho,
                repulsion_strength_gamma=repulsion_strength_gamma,
                noise_sigma=noise_sigma,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def event_generator():
        engine.reset()
        for _ in range(steps):
            if config is None:
                # Legacy DeGroot path: platform-curation feed scoring.
                state = engine.step(alpha=alpha, beta=beta, epsilon=epsilon)
                payload: dict = {
                    "step": state.step,
                    "beliefs": state.beliefs.tolist(),
                    "polarization": state.polarization,
                    "polarization_normalized": state.polarization_normalized,
                    "echo_coefficient": state.echo_coefficient,
                    "mean_pairwise_distance": None,
                    "frac_no_compatible": None,
                    "mean_exposure_similarity": None,
                    "active_exposures": None,
                }
            else:
                state = engine.step_with_config(config)
                payload = {
                    "step": state.step,
                    "beliefs": state.beliefs.tolist(),
                    "polarization": state.polarization,
                    "polarization_normalized": state.polarization_normalized,
                    "echo_coefficient": state.echo_coefficient,
                    "mean_pairwise_distance": state.mean_pairwise_distance,
                    "frac_no_compatible": state.frac_no_compatible,
                    "mean_exposure_similarity": state.mean_exposure_similarity,
                    "active_exposures": state.active_exposures,
                }
            yield f"data: {json.dumps(payload)}\n\n"
            if interval > 0:
                await asyncio.sleep(interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
