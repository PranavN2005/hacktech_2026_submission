import type { SimParams, DynamicsConfig } from './store';

export interface InitPayload {
  agent_count: number;
  nodes: Array<{
    id: number;
    name: string;
    bio: string;
    initial_belief: number;
    susceptibility: number;
    social_capital: number;
    persona_id?: number | string;
  }>;
  edges: Array<{ from: number; to: number }>;
  defaults: SimParams;
}

const API_BASE = '/api';

export async function fetchInit(agentQuantity?: number): Promise<InitPayload> {
  const query = typeof agentQuantity === 'number'
    ? `?${new URLSearchParams({ agent_quantity: String(agentQuantity) }).toString()}`
    : '';
  const response = await fetch(`${API_BASE}/init${query}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch /init (${response.status} ${response.statusText})`);
  }
  return response.json() as Promise<InitPayload>;
}

export function createStreamUrl(params: SimParams, dynamics: DynamicsConfig): string {
  const searchParams = new URLSearchParams({
    // Platform / legacy params
    alpha: String(params.alpha),
    beta: String(params.beta),
    epsilon: String(params.epsilon),
    steps: String(params.steps),
    interval: String(params.interval),
    // Dynamics model
    model_type: dynamics.model_type,
    exposure_mode: dynamics.exposure_mode,
    top_k_visible: String(dynamics.top_k_visible),
    selective_exposure_beta: String(dynamics.selective_exposure_beta),
    distance_decay_alpha: String(dynamics.distance_decay_alpha),
    repulsion_threshold_rho: String(dynamics.repulsion_threshold_rho),
    repulsion_strength_gamma: String(dynamics.repulsion_strength_gamma),
    noise_sigma: String(dynamics.noise_sigma),
  });
  return `${API_BASE}/stream?${searchParams.toString()}`;
}
