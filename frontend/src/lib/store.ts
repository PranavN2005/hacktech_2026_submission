import { writable, derived } from 'svelte/store';

// Types
export interface Agent {
  id: number;
  persona_id?: number | string;
  name: string;
  bio: string;
  initial_belief: number;
  susceptibility: number;
  social_capital: number;
}

export interface GraphEdge {
  from: number;
  to: number;
}

export interface SimParams {
  alpha: number;
  beta: number;
  epsilon: number;
  steps: number;
  interval: number;
}

export type ModelType = 'degroot' | 'confirmation_bias' | 'bounded_confidence' | 'repulsive_bc';
export type ExposureMode = 'all_followed' | 'top_k' | 'sampled';

export interface DynamicsConfig {
  model_type: ModelType;
  exposure_mode: ExposureMode;
  top_k_visible: number;
  selective_exposure_beta: number;
  distance_decay_alpha: number;
  repulsion_threshold_rho: number;
  repulsion_strength_gamma: number;
  noise_sigma: number;
}

export interface MetricPoint {
  step: number;
  polarization: number;
  polarization_normalized: number;
  echo: number;
  mean_pairwise_distance?: number;
  frac_no_compatible?: number;
  mean_exposure_similarity?: number;
  active_exposures?: number;
}

export interface SimulationState {
  isInitialized: boolean;
  isPlaying: boolean;
  isLoading: boolean;
  error: string | null;
  initVersion: number;
  // Increments every time the user resets — consumers (e.g. NetworkGraph)
  // subscribe to this to trigger a layout re-fit without rebuilding state.
  resetVersion: number;
  currentStep: number;
  totalSteps: number;
  agentQuantity: number;
  agents: Agent[];
  edges: GraphEdge[];
  beliefs: number[];
  metricsHistory: MetricPoint[];
  selectedAgent: Agent | null;
  params: SimParams;
  dynamics: DynamicsConfig;
}

// Default parameters
const defaultParams: SimParams = {
  alpha: 0.5,
  beta: 0.2,
  epsilon: 0.4,
  steps: 100,
  interval: 0.3,
};

const defaultDynamics: DynamicsConfig = {
  model_type: 'degroot',
  exposure_mode: 'all_followed',
  top_k_visible: 10,
  selective_exposure_beta: 2.0,
  distance_decay_alpha: 2.0,
  repulsion_threshold_rho: 0.9,
  repulsion_strength_gamma: 0.5,
  noise_sigma: 0.0,
};

// Initial state
const initialState: SimulationState = {
  isInitialized: false,
  isPlaying: false,
  isLoading: false,
  error: null,
  initVersion: 0,
  resetVersion: 0,
  currentStep: 0,
  totalSteps: 100,
  agentQuantity: 500,
  agents: [],
  edges: [],
  beliefs: [],
  metricsHistory: [],
  selectedAgent: null,
  params: { ...defaultParams },
  dynamics: { ...defaultDynamics },
};

// Create the main store
function createSimStore() {
  const { subscribe, set, update } = writable<SimulationState>(initialState);

  return {
    subscribe,

    // Initialize from /init endpoint data
    initialize: (data: {
      agent_count: number;
      nodes: Agent[];
      edges: GraphEdge[];
      defaults: SimParams;
    }) => {
      update(state => {
        const mergedParams = {
          ...data.defaults,
          ...state.params,
        };
        return {
          ...state,
          isInitialized: true,
          isLoading: false,
          error: null,
          initVersion: state.initVersion + 1,
          agents: data.nodes,
          edges: data.edges,
          beliefs: data.nodes.map(n => n.initial_belief),
          agentQuantity: data.agent_count,
          totalSteps: mergedParams.steps,
          params: mergedParams,
        };
      });
    },

    setLoading: (isLoading: boolean) => {
      update(state => ({ ...state, isLoading }));
    },

    setError: (error: string | null) => {
      update(state => ({ ...state, error, isLoading: false }));
    },

    setParams: (params: Partial<SimParams>) => {
      update(state => ({
        ...state,
        params: { ...state.params, ...params },
        totalSteps:
          typeof params.steps === 'number' && !state.isPlaying
            ? params.steps
            : state.totalSteps,
      }));
    },

    setDynamics: (dynamics: Partial<DynamicsConfig>) => {
      update(state => {
        const merged = { ...state.dynamics, ...dynamics };
        // Keep ρ ≥ ε (mirror the server-side guard so the UI stays consistent).
        const eps = typeof dynamics.model_type !== 'undefined'
          ? state.params.epsilon
          : state.params.epsilon;
        if (merged.repulsion_threshold_rho < eps) {
          merged.repulsion_threshold_rho = eps;
        }
        return { ...state, dynamics: merged };
      });
    },

    startSimulation: () => {
      update(state => ({
        ...state,
        isPlaying: true,
        currentStep: 0,
        totalSteps: state.params.steps,
        metricsHistory: [],
      }));
    },

    stopSimulation: () => {
      update(state => ({ ...state, isPlaying: false }));
    },

    ingestStep: (payload: {
      step: number;
      beliefs: number[];
      polarization: number;
      polarization_normalized?: number | null;
      echo_coefficient: number;
      mean_pairwise_distance?: number | null;
      frac_no_compatible?: number | null;
      mean_exposure_similarity?: number | null;
      active_exposures?: number | null;
    }) => {
      update(state => ({
        ...state,
        currentStep: payload.step,
        beliefs: payload.beliefs,
        metricsHistory: [
          ...state.metricsHistory,
          {
            step: payload.step,
            polarization: payload.polarization,
            polarization_normalized: payload.polarization_normalized ?? 0,
            echo: payload.echo_coefficient,
            mean_pairwise_distance: payload.mean_pairwise_distance ?? undefined,
            frac_no_compatible: payload.frac_no_compatible ?? undefined,
            mean_exposure_similarity: payload.mean_exposure_similarity ?? undefined,
            active_exposures: payload.active_exposures ?? undefined,
          },
        ],
      }));
    },

    completeSimulation: () => {
      update(state => ({ ...state, isPlaying: false }));
    },

    reset: () => {
      update(state => ({
        ...state,
        isPlaying: false,
        currentStep: 0,
        beliefs: state.agents.map(a => a.initial_belief),
        metricsHistory: [],
        resetVersion: state.resetVersion + 1,
      }));
    },

    selectAgent: (agent: Agent | null) => {
      update(state => ({ ...state, selectedAgent: agent }));
    },

    applyPreset: (presetName: string) => {
      type Preset = { params: Partial<SimParams>; dynamics: Partial<DynamicsConfig> };
      const presets: Record<string, Preset> = {
        chronological: {
          params: { alpha: 0.0, beta: 0.0, epsilon: 0.5 },
          dynamics: { model_type: 'degroot', exposure_mode: 'all_followed' },
        },
        engagement: {
          params: { alpha: 0.75, beta: 0.2, epsilon: 0.3 },
          dynamics: { model_type: 'degroot', exposure_mode: 'all_followed' },
        },
        diversity: {
          params: { alpha: 0.1, beta: 0.1, epsilon: 0.8 },
          dynamics: { model_type: 'bounded_confidence', exposure_mode: 'all_followed' },
        },
        echo_bubble: {
          params: { alpha: 0.5, beta: 0.2, epsilon: 0.35 },
          dynamics: {
            model_type: 'confirmation_bias',
            exposure_mode: 'top_k',
            top_k_visible: 8,
            distance_decay_alpha: 4.0,
          },
        },
        polarized: {
          params: { alpha: 0.5, beta: 0.2, epsilon: 0.25 },
          dynamics: {
            model_type: 'repulsive_bc',
            exposure_mode: 'all_followed',
            repulsion_threshold_rho: 0.7,
            repulsion_strength_gamma: 0.6,
          },
        },
      };
      const preset = presets[presetName];
      if (!preset) return;
      update(state => ({
        ...state,
        params: { ...state.params, ...preset.params },
        dynamics: { ...state.dynamics, ...preset.dynamics },
      }));
    },
  };
}

export const simStore = createSimStore();

// Derived stores
export const currentPolarization = derived(simStore, $store => {
  const history = $store.metricsHistory;
  return history.length > 0 ? history[history.length - 1].polarization : 0;
});

export const currentEcho = derived(simStore, $store => {
  const history = $store.metricsHistory;
  return history.length > 0 ? history[history.length - 1].echo : 0;
});

export const progress = derived(simStore, $store => {
  return $store.totalSteps > 0 ? ($store.currentStep / $store.totalSteps) * 100 : 0;
});

export const latestMetric = derived(simStore, $store => {
  const h = $store.metricsHistory;
  return h.length > 0 ? h[h.length - 1] : null;
});
