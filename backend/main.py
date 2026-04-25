from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

engine = SimulationEngine(_AGENTS_PATH, seed=42, min_out_degree=10)

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
}


@app.get("/init")
async def init() -> dict:
    """
    Return agent metadata, follow-graph topology, and simulation defaults.

    Node ordering in `nodes` is identical to the belief-array index order
    emitted by `/stream`, so `nodes[i].id == i` always holds.
    """
    sc = engine.social_capital  # shape (N,), index-aligned with beliefs
    nodes = [
        {
            "id": int(a["id"]),
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
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Echo-chamber strength"),
    beta: float = Query(0.2, ge=0.0, le=1.0, description="Virality / outrage bias"),
    epsilon: float = Query(0.4, ge=0.0, le=2.0, description="Bounded-confidence threshold"),
    steps: int = Query(100, ge=1, le=1000, description="Number of simulation steps"),
    interval: float = Query(0.1, ge=0.0, le=10.0, description="Seconds between SSE events"),
):
    """
    Run the simulation and stream each step as an SSE event.

    Each event payload:
        {
            "step":             int,
            "beliefs":          float[N],
            "polarization":     float,
            "echo_coefficient": float
        }
    """
    if alpha + beta > 1.0:
        raise HTTPException(
            status_code=422,
            detail="alpha + beta must be ≤ 1.0 (the remainder is the baseline/chronological weight).",
        )

    async def event_generator():
        engine.reset()
        for _ in range(steps):
            state = engine.step(alpha=alpha, beta=beta, epsilon=epsilon)
            payload = {
                "step": state.step,
                "beliefs": state.beliefs.tolist(),
                "polarization": state.polarization,
                "echo_coefficient": state.echo_coefficient,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if interval > 0:
                await asyncio.sleep(interval)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
