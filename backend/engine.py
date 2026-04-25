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

from backend.config import SimulationConfig
from backend.dynamics import (
    compatibility,
    compute_distances,
    effective_weights,
    exposure_mask,
    phi_repulsive,
    safe_row_normalize,
)

# Esteban-Ray constants
_RHO: float = 1.3   # polarization sensitivity (literature standard)
_K_ER: float = 1.0  # scale constant


@dataclass(frozen=True)
class SimState:
    step: int
    beliefs: np.ndarray      # shape (N,), values ∈ [-1, 1]
    polarization: float      # Esteban-Ray index ≥ 0
    echo_coefficient: float  # ∈ [0, 1]
    # Raw ER is kept for continuity; this display-oriented value scales it by
    # the max possible ER for the current Louvain community sizes.
    polarization_normalized: float = 0.0
    # Optional auxiliary diagnostics populated by step_with_config().
    # Legacy step() leaves them None to preserve the existing payload shape.
    mean_pairwise_distance: Optional[float] = None
    frac_no_compatible: Optional[float] = None
    mean_exposure_similarity: Optional[float] = None
    active_exposures: Optional[int] = None


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
        # Activity is an optional per-agent trait used by the new exposure
        # layer (top_k / sampled). Defaults to 1.0 if absent in the JSON so
        # legacy agent files continue to work unchanged.
        self._activity: np.ndarray = np.array(
            [float(a.get("activity", 1.0)) for a in self._agents], dtype=np.float64
        )

        # Mutable state
        self._B: np.ndarray = self._B0.copy()
        self._current_step: int = 0

        # Cached from the most recent step() call (used by get_metrics)
        self._last_F: np.ndarray = np.zeros((self.N, self.N))

        # Dedicated RNG for stochastic exposure / noise. Re-seeded on reset()
        # so two engines with the same seed produce identical trajectories.
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ── Public API ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore beliefs to initial values and reset the step counter."""
        self._B = self._B0.copy()
        self._current_step = 0
        self._last_F = np.zeros((self.N, self.N))
        self._rng = np.random.default_rng(self._seed)

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

        pol, pol_norm, echo = self._compute_metrics(F, belief_diff)
        return SimState(
            step=self._current_step,
            beliefs=self._B.copy(),
            polarization=pol,
            polarization_normalized=pol_norm,
            echo_coefficient=echo,
        )

    # ── New modular dynamics path ───────────────────────────────────────
    #
    # Math summary (full derivations live in `dynamics.py`):
    #
    #   Baseline DeGroot:
    #       x(t+1) = (I - S) x(t) + S Ŵ(t) x(t)
    #       Ŵ(t)   = row-normalize( A · M(t) )
    #
    #   Confirmation bias:
    #       Ŵ(t)   ∝ A · exp(-α · |x_i - x_j|) · M(t)
    #
    #   Bounded confidence:
    #       Ŵ(t)   ∝ A · 1[|x_i - x_j| ≤ ε] · M(t)
    #
    #   Repulsive bounded confidence (incremental form):
    #       x_i(t+1) = x_i(t) + s_i Σ_j Ŵ_{ij}(t) · φ_i(x_j - x_i)
    #
    # The follow graph A is static in all four modes. The exposure layer M(t)
    # picks which followed accounts are visible this tick, so the *effective*
    # influence network is dynamic even though the substrate is not.

    def step_with_config(self, config: SimulationConfig) -> SimState:
        """
        Advance one tick using the modular config-driven dynamics.

        Honours `config.model_type` (degroot, confirmation_bias,
        bounded_confidence, repulsive_bc) and `config.exposure_mode`
        (all_followed, top_k, sampled). Falls back to noise-free, no-clip
        behaviour if those flags are off.
        """
        B = self._B
        dist = compute_distances(B)
        A = self._A

        # ── Per-agent thresholds with optional overrides ──────────────
        # Future-proofing: per-agent overrides resolve from the agent dict
        # first, then config.per_agent_overrides, then the global default.
        # Today the homogeneous global path is the fast common case.
        eps = self._resolve_per_agent(config, "confidence_epsilon")
        alpha_decay = self._resolve_per_agent(config, "distance_decay_alpha")
        beta_sel = self._resolve_per_agent(config, "selective_exposure_beta")
        rho = self._resolve_per_agent(config, "repulsion_threshold_rho")
        gamma = self._resolve_per_agent(config, "repulsion_strength_gamma")

        # ── Layer 1: exposure mask M(t) over the static follow graph ──
        M = exposure_mask(
            config.exposure_mode,
            A,
            dist,
            activity=self._activity,
            selective_beta=beta_sel,
            top_k=config.top_k_visible,
            rng=self._rng,
        )

        # ── Layer 2: compatibility C(t) ──────────────────────────────
        C = compatibility(
            config.model_type,
            dist,
            alpha_decay=alpha_decay,
            epsilon=eps,
        )

        # Baseline edge weight for the new framework is the raw graph edge.
        # The legacy step() retains the platform-curation weighting; the new
        # path keeps the substrate clean so the four modes are directly
        # comparable.
        W_base = A
        T = None  # config.trust_enabled is reserved for a future feature.

        if config.model_type == "repulsive_bc":
            W_hat = self._repulsive_weights(A, W_base, M, T)
            new_B = self._update_repulsive(B, W_hat, dist, eps, rho, gamma, config)
        else:
            W_hat = effective_weights(A, W_base, M, C, T=T)
            new_B = self._update_matrix_form(B, W_hat, config)

        if config.clip_beliefs:
            new_B = np.clip(new_B, -1.0, 1.0)

        self._B = new_B
        self._current_step += 1
        # Cache an F-shaped object so get_metrics()/echo coefficient still
        # have something meaningful to work with after a config-driven step.
        self._last_F = W_hat

        # ── Diagnostics ───────────────────────────────────────────────
        # Echo / Esteban-Ray are computed exactly as before so the primary
        # metrics remain comparable across modes.
        new_dist = compute_distances(new_B)
        pol, pol_norm, echo = self._compute_metrics(W_hat, new_dist)
        diagnostics = self._compute_diagnostics(W_hat, new_dist, M)

        return SimState(
            step=self._current_step,
            beliefs=self._B.copy(),
            polarization=pol,
            polarization_normalized=pol_norm,
            echo_coefficient=echo,
            mean_pairwise_distance=diagnostics["mean_pairwise_distance"],
            frac_no_compatible=diagnostics["frac_no_compatible"],
            mean_exposure_similarity=diagnostics["mean_exposure_similarity"],
            active_exposures=diagnostics["active_exposures"],
        )

    # ── Modular update sub-routines ─────────────────────────────────────

    def _update_matrix_form(
        self,
        B: np.ndarray,
        W_hat: np.ndarray,
        config: SimulationConfig,
    ) -> np.ndarray:
        """
        Standard DeGroot-style update using a per-tick row-stochastic Ŵ.

            x_i(t+1) = x_i(t) + s_i · Σ_j Ŵ_{ij} · (x_j - x_i) + ξ_i

        Rows of Ŵ that are all-zero (no compatible visible neighbour) freeze
        the agent at its current belief, which avoids divide-by-zero NaNs.
        """
        wstar_row_sum = W_hat.sum(axis=1)
        delta = self._sigma * (W_hat @ B - B * wstar_row_sum)
        new_B = B + delta
        if config.noise_sigma > 0:
            new_B = new_B + self._rng.normal(0.0, config.noise_sigma, size=B.shape)
        return new_B

    def _repulsive_weights(
        self,
        A: np.ndarray,
        W_base: np.ndarray,
        M: np.ndarray,
        T: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        For repulsive_bc we do *not* apply a compatibility filter when
        normalising — the φ piecewise function handles attract / ignore /
        repel internally. We row-normalise over visible followed neighbours
        so the total interaction strength stays bounded regardless of how
        many extreme-distant neighbours are in view.
        """
        u = A * W_base * M
        if T is not None:
            u = u * T
        return safe_row_normalize(u)

    def _update_repulsive(
        self,
        B: np.ndarray,
        W_hat: np.ndarray,
        dist: np.ndarray,
        eps: np.ndarray | float,
        rho: np.ndarray | float,
        gamma: np.ndarray | float,
        config: SimulationConfig,
    ) -> np.ndarray:
        """
        Incremental update for the repulsive bounded-confidence model:

            x_i(t+1) = x_i(t) + s_i · Σ_j Ŵ_{ij}(t) · φ_i(x_j - x_i) + ξ_i

        A convex weighted average cannot encode negative influence, so we
        sum φ(Δ) per-edge and weight by Ŵ — visible-followed-only.
        """
        delta_mat = B[None, :] - B[:, None]               # Δ_{ij} = x_j - x_i
        phi = phi_repulsive(delta_mat, epsilon=eps, rho=rho, gamma=gamma)
        update = self._sigma * (W_hat * phi).sum(axis=1)
        new_B = B + update
        if config.noise_sigma > 0:
            new_B = new_B + self._rng.normal(0.0, config.noise_sigma, size=B.shape)
        return new_B

    def _resolve_per_agent(
        self, config: SimulationConfig, field_name: str
    ) -> np.ndarray:
        """
        Build a length-N vector for a parameter, allowing per-agent overrides.

        Resolution order per agent:
            1. agent dict has the field directly,
            2. config.per_agent_overrides[i][field_name],
            3. global config default.
        """
        default = float(getattr(config, field_name))
        out = np.full(self.N, default, dtype=np.float64)
        for i, agent in enumerate(self._agents):
            if field_name in agent:
                out[i] = float(agent[field_name])
                continue
            override = config.per_agent_overrides.get(i)
            if override and field_name in override:
                out[i] = float(override[field_name])
        return out

    def _compute_diagnostics(
        self,
        W_hat: np.ndarray,
        dist: np.ndarray,
        M: np.ndarray,
    ) -> dict:
        """Auxiliary metrics for the modular path."""
        # Off-diagonal distances only (skip self-pairs).
        N = self.N
        if N > 1:
            mask = ~np.eye(N, dtype=bool)
            mean_pair = float(dist[mask].mean())
        else:
            mean_pair = 0.0

        row_sums = W_hat.sum(axis=1)
        frac_no_comp = float((row_sums == 0).mean())

        # Average distance over visible (non-zero exposure) edges.
        active_count = int((M > 0).sum())
        if active_count > 0:
            mean_exp_sim = float(1.0 - (dist * (M > 0)).sum() / active_count)
        else:
            mean_exp_sim = 0.0

        return {
            "mean_pairwise_distance": mean_pair,
            "frac_no_compatible": frac_no_comp,
            "mean_exposure_similarity": mean_exp_sim,
            "active_exposures": active_count,
        }

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
    ) -> tuple[float, float, float]:
        raw_er, normalized_er = self._esteban_ray_values(self._B)
        return raw_er, normalized_er, self._echo_coefficient(F, belief_diff)

    def _esteban_ray(self, B: np.ndarray) -> float:
        """Return the raw Esteban-Ray polarization index."""
        return self._esteban_ray_values(B)[0]

    def _esteban_ray_values(self, B: np.ndarray) -> tuple[float, float]:
        """
        Raw and normalized Esteban-Ray polarization index.

        P = K · Σ_k Σ_m π_k^(1+ρ) · π_m · |C_k − C_m|

        Communities are detected on the follow graph topology via Louvain,
        then characterised by their mean belief.

        The normalized value divides raw ER by the maximum possible ER under
        the current community sizes and belief range [-1, 1]:

            P_max = 2 · Σ_{k≠m} π_k^(1+ρ) · π_m

        This preserves raw ER for scientific continuity while giving the UI a
        stable 0-1 display scale. With two equally sized communities at
        centroids -1 and 1, normalized ER is exactly 1.
        """
        communities = self._louvain_communities()
        if len(communities) < 2:
            return 0.0, 0.0

        pop = float(self.N)
        pi = np.array([len(c) / pop for c in communities])
        centroids = np.array([B[list(c)].mean() for c in communities])

        # Vectorised double sum
        pi_row = pi[:, None]   # (K, 1)
        pi_col = pi[None, :]   # (1, K)
        c_row = centroids[:, None]
        c_col = centroids[None, :]
        raw = float(
            ((pi_row ** (1.0 + _RHO)) * pi_col * np.abs(c_row - c_col)).sum()
        )
        raw *= _K_ER

        off_diag = ~np.eye(len(communities), dtype=bool)
        max_possible = float(
            (2.0 * ((pi_row ** (1.0 + _RHO)) * pi_col)[off_diag]).sum()
        )
        if max_possible <= 0.0:
            return raw, 0.0
        normalized = float(np.clip(raw / max_possible, 0.0, 1.0))
        return raw, normalized

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
