from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.engine import SimulationEngine


app = FastAPI(title="EchoChamber Backend", version="0.1.0")
engine = SimulationEngine(n=500, m=3, seed=42)

# Allow local frontend dev servers to call this API during development.
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stream")
async def stream(
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    beta: float = Query(0.2, ge=0.0, le=1.0),
    epsilon: float = Query(0.4, ge=0.0, le=2.0),
    steps: int = Query(100, ge=1, le=1000),
    interval: float = Query(0.1, ge=0.0, le=10.0),
):
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
