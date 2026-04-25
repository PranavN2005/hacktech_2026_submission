"""
SimulationEngine: directed follow graph + opinion dynamics.

Graph convention
----------------
    Edge i → j  means "agent i follows agent j".
    A[i, j] = 1  ⟹  i follows j  ⟹  j's posts can appear in i's feed.
    Social capital  c_j = in-degree(j) = number of followers of j.

Two-layer system
----------------
    Layer 1 - Follow graph A (static):
        Binary adjacency matrix built once via scale_free_graph.
        A[i,j] = 0 means j can never influence i, regardless of parameters.

    Layer 2 - Feed matrix F(t) (recomputed each step):
        Continuous curation weights over the follow graph.
        F[i,j] > 0  only where  A[i,j] = 1.
        Controlled by platform parameters alpha (echo-chamber strength)
        and beta (virality / outrage bias).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

# Esteban-Ray constants
_RHO: float = 1.3   # polarization sensitivity (literature standard)
_K_ER: float = 1.0  # scale constant


@dataclass(frozen=True)
class SimState:
    step: int
    beliefs: np.ndarray      # shape (N,), values ∈ [-1, 1]
    polarization: float      # Esteban-Ray index ≥ 0
    echo_coefficient: float  # ∈ [0, 1]


class SimulationEngine:
    """
    Loads agents from agents.json, builds a directed scale-free follow graph,
    and exposes a step() method that advances the opinion dynamics by one tick.
    """

    def __init__(
        self,
        agents_path: str | Path,
        agent_count: int | None = None,
        seed: Optional[int] = None,
        min_out_degree: int = 10,
    ):
        self._seed = seed
        all_agents: list[dict] = _load_agents(Path(agents_path))
        if agent_count is None:
            agent_count = len(all_agents)
        if not (1 <= agent_count <= len(all_agents)):
            raise ValueError(
                f"agent_count must be within [1, {len(all_agents)}], got {agent_count}"
            )

        if agent_count == len(all_agents):
            sampled = [dict(a) for a in all_agents]
        else:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(all_agents), size=agent_count, replace=False)
            sampled = [dict(all_agents[int(i)]) for i in chosen]

        # Re-assign ids to keep the invariant nodes[i].id == i for the API.
        for i, a in enumerate(sampled):
            a["id"] = i

        self._agents = sampled
        self.N: int = len(self._agents)

        # ── Build directed follow graph ──────────────────────────────────
        self._G: nx.DiGraph = _build_follow_graph(self.N, min_out_degree, seed)

        # A[i, j] = 1  iff  i follows j  (follows = row, followed = column)
        self._A: np.ndarray = (
            nx.adjacency_matrix(self._G, nodelist=range(self.N))
            .toarray()
            .astype(np.float64)
        )

        # Social capital c_j = in-degree of j = column sum of A
        self._C: np.ndarray = self._A.sum(axis=0)  # shape (N,)

        # Per-agent parameters
        self._B0: np.ndarray = np.array(
            [a["initial_belief"] for a in self._agents], dtype=np.float64
        )
        self._sigma: np.ndarray = np.array(
            [a["susceptibility"] for a in self._agents], dtype=np.float64
        )

        # Mutable state
        self._B: np.ndarray = self._B0.copy()
        self._current_step: int = 0

        # Cached from the most recent step() call (used by get_metrics)
        self._last_F: np.ndarray = np.zeros((self.N, self.N))

    # ── Public API ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore beliefs to initial values and reset the step counter."""
        self._B = self._B0.copy()
        self._current_step = 0
        self._last_F = np.zeros((self.N, self.N))

    def step(self, alpha: float, beta: float, epsilon: float) -> SimState:
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        alpha   : echo-chamber strength ∈ [0, 1]
        beta    : virality / outrage bias ∈ [0, 1]
        epsilon : bounded-confidence threshold ∈ (0, 2]
        """
        B = self._B

        # ── Layer 2: Feed matrix F(t) ─────────────────────────────────
        # Raw algorithmic score:
        #   S[i,j] = c_j · [α(1 − |b_i−b_j|/2) + β|b_j| + (1−α−β)]
        belief_diff = np.abs(B[:, None] - B[None, :])      # (N, N)

        S = self._C[None, :] * (
            alpha * (1.0 - belief_diff / 2.0)              # homophily
            + beta * np.abs(B[None, :])                    # virality
            + (1.0 - alpha - beta)                         # baseline / chronological
        )
        S *= self._A        # enforce follow graph: S[i,j]=0 where A[i,j]=0

        # Row-normalise → feed probability F[i,j]
        row_sums = S.sum(axis=1, keepdims=True)
        F = np.divide(S, row_sums, where=row_sums > 0, out=np.zeros_like(S))
        self._last_F = F

        # ── Bounded confidence (Deffuant-Weisbuch) ───────────────────
        # W̃[i,j] = F[i,j] if |b_i−b_j| ≤ ε, else 0
        W_tilde = np.where(belief_diff <= epsilon, F, 0.0)

        # Row-normalise W̃ → W* (agents with no reachable neighbours get 0 row)
        wt_sums = W_tilde.sum(axis=1, keepdims=True)
        W_star = np.divide(
            W_tilde, wt_sums, where=wt_sums > 0, out=np.zeros_like(W_tilde)
        )

        # ── Belief update ─────────────────────────────────────────────
        # b_i(t+1) = b_i(t) + σ_i · Σ_j W*_ij · (b_j − b_i)
        #          = b_i(t) + σ_i · [(W*·B)[i] − b_i · rowsum(W*)[i]]
        wstar_row_sum = W_star.sum(axis=1)          # 1 where node has influence, 0 otherwise
        delta = self._sigma * (W_star @ B - B * wstar_row_sum)
        self._B = np.clip(B + delta, -1.0, 1.0)
        self._current_step += 1

        pol, echo = self._compute_metrics(F, belief_diff)
        return SimState(
            step=self._current_step,
            beliefs=self._B.copy(),
            polarization=pol,
            echo_coefficient=echo,
        )

    def get_metrics(self) -> dict:
        """
        Return current polarization metrics plus per-cluster belief centroids.
        Safe to call between step() calls.
        """
        B = self._B
        belief_diff = np.abs(B[:, None] - B[None, :])
        pol, echo = self._compute_metrics(self._last_F, belief_diff)
        communities = self._louvain_communities()
        centroids = {
            f"cluster_{k}": float(B[list(c)].mean())
            for k, c in enumerate(communities)
        }
        return {
            "step": self._current_step,
            "polarization": pol,
            "echo_coefficient": echo,
            "cluster_centroids": centroids,
        }

    @property
    def graph_edges(self) -> list[tuple[int, int]]:
        """All directed edges (i, j) meaning i follows j."""
        return list(self._G.edges())

    @property
    def agents(self) -> list[dict]:
        return self._agents

    @property
    def social_capital(self) -> np.ndarray:
        """In-degree of each agent (shape (N,)). Index i matches beliefs[i]."""
        return self._C

    # ── Internal helpers ────────────────────────────────────────────────

    def _compute_metrics(
        self, F: np.ndarray, belief_diff: np.ndarray
    ) -> tuple[float, float]:
        return self._esteban_ray(self._B), self._echo_coefficient(F, belief_diff)

    def _esteban_ray(self, B: np.ndarray) -> float:
        """
        Esteban-Ray polarization index.
        P = K · Σ_k Σ_m π_k^(1+ρ) · π_m · |C_k − C_m|

        Communities are detected on the follow graph topology via Louvain,
        then characterised by their mean belief.
        """
        communities = self._louvain_communities()
        if len(communities) < 2:
            return 0.0

        pop = float(self.N)
        pi = np.array([len(c) / pop for c in communities])
        centroids = np.array([B[list(c)].mean() for c in communities])

        # Vectorised double sum
        pi_row = pi[:, None]   # (K, 1)
        pi_col = pi[None, :]   # (1, K)
        c_row = centroids[:, None]
        c_col = centroids[None, :]
        total = float(
            ((pi_row ** (1.0 + _RHO)) * pi_col * np.abs(c_row - c_col)).sum()
        )
        return _K_ER * total

    def _echo_coefficient(self, F: np.ndarray, belief_diff: np.ndarray) -> float:
        """
        E(t) = 1 − mean_feed_distance(t) / mean_population_distance(t)

        mean_feed_distance   : average |b_i−b_j| weighted by F[i,j] over all edges
        mean_population_dist : average |b_i−b_j| over all (i,j) pairs
        """
        mean_pop = float(belief_diff.mean())
        if mean_pop == 0.0:
            return 0.0

        feed_weights = F * self._A          # non-zero only on actual edges
        total_weight = float(feed_weights.sum())
        if total_weight == 0.0:
            return 0.0

        mean_feed = float((feed_weights * belief_diff).sum()) / total_weight
        return float(1.0 - mean_feed / mean_pop)

    def _louvain_communities(self):
        """
        Louvain community detection on the undirected projection of the follow
        graph. Seed is fixed for reproducibility.
        """
        G_und = self._G.to_undirected()
        return list(nx.community.louvain_communities(G_und, seed=self._seed))


# ── Utility ─────────────────────────────────────────────────────────────

def _build_follow_graph(N: int, min_out_degree: int, seed: Optional[int]) -> nx.DiGraph:
    """
    Build a directed scale-free follow graph with a guaranteed minimum out-degree.

    Step 1 – scale_free_graph with beta-heavy parameterisation:
        Raising beta (edge-between-existing-nodes probability) increases the
        average degree while preserving the power-law in-degree distribution.
        alpha + beta + gamma must equal 1.0.

    Step 2 – Post-process minimum out-degree:
        For every node whose out-degree is still below `min_out_degree`, sample
        uniformly from the pool of nodes it does not yet follow and add edges.
        This ensures bounded confidence always has reachable neighbours while
        leaving the celebrity in-degree structure untouched.
    """
    rng = np.random.default_rng(seed)

    # Denser parameterisation: more beta = more inter-existing-node edges
    raw = nx.scale_free_graph(
        N,
        alpha=0.30,   # prob: new node added, follows an existing node
        beta=0.60,    # prob: new edge between two existing nodes (↑ density)
        gamma=0.10,   # prob: new node added, gets followed by an existing node
        delta_in=0.2,
        delta_out=0.2,
        seed=seed,
    )
    G: nx.DiGraph = nx.DiGraph(raw)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Post-process: enforce minimum out-degree for every node
    all_nodes = np.arange(N)
    for node in range(N):
        current_out = set(G.successors(node))
        deficit = min_out_degree - len(current_out)
        if deficit <= 0:
            continue
        # Candidate targets: nodes not already followed and not self
        candidates = np.array(
            [j for j in all_nodes if j != node and j not in current_out]
        )
        if len(candidates) == 0:
            continue
        chosen = rng.choice(
            candidates,
            size=min(deficit, len(candidates)),
            replace=False,
        )
        G.add_edges_from((node, int(j)) for j in chosen)

    return G


def _load_agents(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"agents file not found: {path}")
    with path.open(encoding="utf-8") as f:
        agents = json.load(f)
    if not isinstance(agents, list) or not agents:
        raise ValueError("agents.json must be a non-empty JSON array.")
    return agents
