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

export interface MetricPoint {
  step: number;
  polarization: number;
  echo: number;
}

export interface SimulationState {
  isInitialized: boolean;
  isPlaying: boolean;
  isLoading: boolean;
  error: string | null;
  currentStep: number;
  totalSteps: number;
  agents: Agent[];
  edges: GraphEdge[];
  beliefs: number[];
  metricsHistory: MetricPoint[];
  selectedAgent: Agent | null;
  params: SimParams;
}

// Default parameters
const defaultParams: SimParams = {
  alpha: 0.5,
  beta: 0.2,
  epsilon: 0.4,
  steps: 100,
  interval: 0.1
};

// Initial state
const initialState: SimulationState = {
  isInitialized: false,
  isPlaying: false,
  isLoading: false,
  error: null,
  currentStep: 0,
  totalSteps: 100,
  agents: [],
  edges: [],
  beliefs: [],
  metricsHistory: [],
  selectedAgent: null,
  params: { ...defaultParams }
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
      update(state => ({
        ...state,
        isInitialized: true,
        isLoading: false,
        error: null,
        agents: data.nodes,
        edges: data.edges,
        beliefs: data.nodes.map(n => n.initial_belief),
        totalSteps: data.defaults.steps,
        params: { ...data.defaults }
      }));
    },

    // Set loading state
    setLoading: (isLoading: boolean) => {
      update(state => ({ ...state, isLoading }));
    },

    // Set error
    setError: (error: string | null) => {
      update(state => ({ ...state, error, isLoading: false }));
    },

    // Update simulation parameters
    setParams: (params: Partial<SimParams>) => {
      update(state => ({
        ...state,
        params: { ...state.params, ...params }
      }));
    },

    // Start simulation
    startSimulation: () => {
      update(state => ({
        ...state,
        isPlaying: true,
        currentStep: 0,
        metricsHistory: []
      }));
    },

    // Stop simulation
    stopSimulation: () => {
      update(state => ({ ...state, isPlaying: false }));
    },

    // Ingest a simulation step from SSE
    ingestStep: (payload: {
      step: number;
      beliefs: number[];
      polarization: number;
      echo_coefficient: number;
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
            echo: payload.echo_coefficient
          }
        ]
      }));
    },

    // Complete simulation
    completeSimulation: () => {
      update(state => ({ ...state, isPlaying: false }));
    },

    // Reset simulation
    reset: () => {
      update(state => ({
        ...state,
        isPlaying: false,
        currentStep: 0,
        beliefs: state.agents.map(a => a.initial_belief),
        metricsHistory: []
      }));
    },

    // Select an agent
    selectAgent: (agent: Agent | null) => {
      update(state => ({ ...state, selectedAgent: agent }));
    },

    // Apply a preset
    applyPreset: (presetName: 'chronological' | 'engagement' | 'diversity') => {
      const presets: Record<string, Partial<SimParams>> = {
        chronological: { alpha: 0.0, beta: 0.0, epsilon: 0.5 },
        engagement: { alpha: 0.75, beta: 0.2, epsilon: 0.3 },
        diversity: { alpha: 0.1, beta: 0.1, epsilon: 0.8 }
      };
      update(state => ({
        ...state,
        params: { ...state.params, ...presets[presetName] }
      }));
    }
  };
}

export const simStore = createSimStore();

// Derived stores for convenience
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
