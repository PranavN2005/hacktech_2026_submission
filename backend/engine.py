from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class StepState:
    step: int
    beliefs: np.ndarray
    polarization: float
    echo_coefficient: float


class SimulationEngine:
    def __init__(
        self,
        n: int = 500,
        m: int = 3,
        seed: int = 42,
        agents_path: str | None = None,
    ) -> None:
        if agents_path is not None:
            self.agent_ids, initial_beliefs, susceptibility = self._load_agents(agents_path)
            n = int(initial_beliefs.shape[0])
        else:
            if n <= 1:
                raise ValueError("n must be > 1")
            self.agent_ids = [str(i) for i in range(n)]
            initial_beliefs = None
            susceptibility = None

        if m <= 0 or m >= n:
            raise ValueError("m must be in [1, n-1]")

        self.n = n
        self.m = m
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        graph = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
        self._adjacency = nx.to_numpy_array(graph, dtype=np.float64)
        np.fill_diagonal(self._adjacency, 0.0)

        self._social_capital = self._adjacency.sum(axis=0)
        cap_max = float(self._social_capital.max())
        if cap_max > 0:
            self._social_capital = self._social_capital / cap_max
        else:
            self._social_capital = np.ones(n, dtype=np.float64)

        if initial_beliefs is None:
            self._initial_beliefs = self._rng.uniform(-1.0, 1.0, size=n)
        else:
            self._initial_beliefs = initial_beliefs

        if susceptibility is None:
            self._susceptibility = self._rng.uniform(0.1, 0.9, size=n)
        else:
            self._susceptibility = susceptibility

        self.current_step = 0
        self.beliefs = self._initial_beliefs.copy()

    def reset(self) -> None:
        self.current_step = 0
        self.beliefs = self._initial_beliefs.copy()

    def step(self, alpha: float, beta: float, epsilon: float) -> StepState:
        beliefs_col = self.beliefs[:, None]
        beliefs_row = self.beliefs[None, :]
        diff = np.abs(beliefs_col - beliefs_row)

        alpha = float(np.clip(alpha, 0.0, 1.0))
        beta = float(np.clip(beta, 0.0, 1.0))
        epsilon = float(np.clip(epsilon, 0.0, 2.0))

        baseline = max(0.0, 1.0 - alpha - beta)
        homophily = 1.0 - (diff / 2.0)
        extremity = np.abs(self.beliefs)[None, :]

        raw_scores = (
            self._social_capital[None, :]
            * (alpha * homophily + beta * extremity + baseline)
            * self._adjacency
        )

        feed = self._row_normalize(raw_scores)

        confidence_mask = (diff <= epsilon).astype(np.float64)
        effective = feed * confidence_mask
        influence = self._row_normalize(effective)

        delta = influence @ self.beliefs - self.beliefs
        updated = self.beliefs + self._susceptibility * delta
        self.beliefs = np.clip(updated, -1.0, 1.0)

        polarization = float(np.var(self.beliefs))
        echo_coefficient = self._echo_coefficient(diff=diff, feed=feed)

        state = StepState(
            step=self.current_step,
            beliefs=self.beliefs.copy(),
            polarization=polarization,
            echo_coefficient=echo_coefficient,
        )
        self.current_step += 1
        return state

    @staticmethod
    def _row_normalize(matrix: np.ndarray) -> np.ndarray:
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums > 0,
        )

    def _echo_coefficient(self, diff: np.ndarray, feed: np.ndarray) -> float:
        off_diag = ~np.eye(self.n, dtype=bool)
        population_distance = float(diff[off_diag].mean())
        if population_distance == 0.0:
            return 0.0

        feed_distance = float((feed * diff).sum(axis=1).mean())
        raw_echo = 1.0 - (feed_distance / population_distance)
        return float(np.clip(raw_echo, -1.0, 1.0))

    @staticmethod
    def _load_agents(agents_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        path = Path(agents_path)
        if not path.exists():
            raise ValueError(f"agents file not found: {agents_path}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("agents file must contain a JSON array")
        if len(raw) < 2:
            raise ValueError("agents file must contain at least 2 agents")

        agent_ids: list[str] = []
        beliefs: list[float] = []
        susceptibility: list[float] = []
        seen_ids: set[str] = set()

        for idx, agent in enumerate(raw):
            if not isinstance(agent, dict):
                raise ValueError(f"agent at index {idx} must be an object")

            agent_id = SimulationEngine._require_value(agent, "id", idx)
            belief_raw = SimulationEngine._require_value(agent, "initial_belief", idx)
            susceptibility_raw = SimulationEngine._require_value(agent, "susceptibility", idx)

            agent_id = str(agent_id)
            if not agent_id:
                raise ValueError(f"agent at index {idx} has empty id")
            if agent_id in seen_ids:
                raise ValueError(f"duplicate agent id: {agent_id}")
            seen_ids.add(agent_id)

            belief = float(belief_raw)
            sigma = float(susceptibility_raw)
            if belief < -1.0 or belief > 1.0:
                raise ValueError(f"agent {agent_id} has out-of-range initial_belief: {belief}")
            if sigma <= 0.0 or sigma > 1.0:
                raise ValueError(f"agent {agent_id} has out-of-range susceptibility: {sigma}")

            agent_ids.append(agent_id)
            beliefs.append(belief)
            susceptibility.append(sigma)

        return (
            agent_ids,
            np.array(beliefs, dtype=np.float64),
            np.array(susceptibility, dtype=np.float64),
        )

    @staticmethod
    def _require_value(agent: dict[str, Any], key: str, index: int) -> Any:
        if key not in agent:
            raise ValueError(f"agent at index {index} missing required key '{key}'")
        return agent[key]
