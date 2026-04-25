from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.engine import SimulationEngine

# ---------------------------------------------------------------------------
# App + engine setup
# ---------------------------------------------------------------------------

app = FastAPI(title="EchoChamber Backend", version="0.2.0")

_AGENTS_PATH = Path("agents.json")
if not _AGENTS_PATH.exists():
    raise RuntimeError(
        f"agents.json not found at {_AGENTS_PATH.resolve()}. "
        "Run persona_gen.py first to generate agents."
    )

engine = SimulationEngine(_AGENTS_PATH, seed=42)

# Allow local frontend dev servers during development.
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
# Static graph data (called once by the frontend at startup)
# ---------------------------------------------------------------------------

@app.get("/agents")
async def get_agents() -> list[dict]:
    """Return all agent metadata for the node inspector panel."""
    return engine.agents


@app.get("/graph")
async def get_graph() -> dict:
    """
    Return the static follow graph topology for vis.js initialisation.

    nodes  – list of { id, label, social_capital }
    edges  – list of { from, to }  (from follows to)
    """
    nodes = [
        {
            "id": a["id"],
            "label": a["name"],
            "social_capital": int(engine._C[a["id"]]),
        }
        for a in engine.agents
    ]
    edges = [{"from": i, "to": j} for i, j in engine.graph_edges]
    return {"nodes": nodes, "edges": edges}


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
            "beliefs":          float[N],   // current belief of every agent
            "polarization":     float,      // Esteban-Ray index
            "echo_coefficient": float       // echo-chamber coefficient
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
