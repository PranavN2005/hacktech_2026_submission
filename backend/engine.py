from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class StepState:
    step: int
    beliefs: np.ndarray
    polarization: float
    echo_coefficient: float


class SimulationEngine:
    def __init__(self, n: int = 500, m: int = 3, seed: int = 42) -> None:
        if n <= 1:
            raise ValueError("n must be > 1")
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

        self._initial_beliefs = self._rng.uniform(-1.0, 1.0, size=n)
        self._susceptibility = self._rng.uniform(0.1, 0.9, size=n)

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
