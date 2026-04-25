<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# EchoChamber: Project Specification \& Architecture

This document is the complete, self-contained blueprint for building the EchoChamber simulation platform. It covers the theoretical mathematics, system architecture, data models, and step-by-step implementation guide required to complete the project in 36 hours.

## 1. Mathematical Framework (Paper-Ready Specification)

The simulation engine is grounded in established opinion dynamics literature, specifically extending the DeGroot model with bounded confidence and dynamic algorithmic rewiring.

### 1.1 The State Space

Let the society be represented by a directed graph $G = (V, E)$ with $N$ agents.
At any time step $t$, each agent $i \in V$ holds a belief $b_i(t) \in [-1, 1]$ on a continuous ideological spectrum (e.g., -1 is extreme left, 1 is extreme right).
Each agent also has an inherent susceptibility score $\sigma_i \in (0, 1]$ determining their baseline willingness to update their beliefs, and a social capital score $c_i$ corresponding to their node degree (follower count).

### 1.2 Baseline DeGroot Updating

In the classical DeGroot model, agent $i$'s belief at $t+1$ is a weighted average of their neighbors' beliefs:

$$
b_i(t+1) = \sum_{j \in N(i)} W_{ij} b_j(t)
$$

where $W_{ij}$ is the static trust weight agent $i$ places on agent $j$, and $\sum_j W_{ij} = 1$.

### 1.3 Algorithmic Feed Modulation (The Novel Contribution)

In social media, the static adjacency matrix is replaced by a dynamic algorithmic feed matrix $F(t)$. The platform does not show $i$ all content from their neighbors $N(i)$; it curates a specific subset.

The probability $P_{ij}(t)$ that the feed algorithm surfaces agent $j$'s content to agent $i$ is governed by two platform parameters:

1. **Echo Chamber Strength ($\alpha \in [0, 1]$)**: The platform's preference for homophily (showing users content they agree with).
2. **Virality/Outrage Bias ($\beta \in [0, 1]$)**: The platform's preference for surfacing extreme or high-engagement content, independent of agreement.

We define the raw algorithmic score $S_{ij}(t)$ for candidate post $j$ shown to $i$:

$$
S_{ij}(t) = c_j \cdot \left[ \alpha \left( 1 - \frac{|b_i(t) - b_j(t)|}{2} \right) + \beta |b_j(t)| + (1 - \alpha - \beta) \right]
$$

The feed matrix $F_{ij}(t)$ is then constructed by normalizing $S_{ij}(t)$ across all $j \in N(i)$.

### 1.4 Bounded Confidence (Deffuant-Weisbuch)

Humans exhibit confirmation bias and backfire effects. Agents only incorporate opinions that fall within their tolerance threshold $\epsilon$. If the distance exceeds $\epsilon$, the weight drops to zero.
Let the effective influence weight be:

$$
\tilde{W}_{ij}(t) = \begin{cases} F_{ij}(t) & \text{if } |b_i(t) - b_j(t)| \leq \epsilon \\ 0 & \text{otherwise} \end{cases}
$$

The final belief update equation for the simulation is:

$$
b_i(t+1) = b_i(t) + \sigma_i \sum_{j \in N(i)} \tilde{W}^*_{ij}(t) (b_j(t) - b_i(t))
$$

where $\tilde{W}^*_{ij}(t)$ is the row-normalized effective influence matrix.

### 1.5 Quantification Metrics

**Esteban-Ray Polarization Index ($P$)**:
Quantifies the emergence of distinct, antagonistic clusters:

$$
P(t) = K \sum_{k} \sum_{m} \pi_k^{1+\rho} \pi_m |C_k - C_m|
$$

where $\pi_k$ is the population fraction of cluster $k$, $C_k$ is the centroid belief of cluster $k$, and $\rho \approx 1.3$ is the polarization sensitivity parameter.

**Echo Chamber Coefficient ($E$)**:

$$
E(t) = 1 - \frac{\text{Mean Distance of Feed}(t)}{\text{Mean Distance of Population}(t)}
$$

***

## 2. System Architecture

The system uses a unidirectional real-time data flow.

1. **Initialization**: The FastAPI backend generates LLM personas via Ollama, builds the NetworkX graph, and seeds initial beliefs.
2. **Simulation Loop**: The NumPy engine computes matrix operations for $b(t+1)$.
3. **Streaming**: FastAPI pushes the new state matrix via Server-Sent Events (SSE) to the client.
4. **State Management**: The React frontend receives the SSE payload and writes it to a global Zustand store.
5. **Re-rendering**: vis.js (network) and Vega-Lite (charts) independently subscribe to the Zustand store and update at 60 FPS.

***

## 3. Backend Specification (Python)

### 3.1 Dependencies

```text
fastapi==0.109.2
uvicorn==0.27.0
networkx==3.2.1
numpy==1.26.4
scipy==1.12.0
pandas==2.2.0
ollama==0.1.6
pydantic==2.6.1
```


### 3.2 Core Modules

**`persona_gen.py`**
Responsible for creating the agents before the server starts. Uses `ollama.chat` targeting `qwen:3.6b`.

- **Input**: Pew Research demographic distributions (e.g., 20% rural/conservative, 30% urban/liberal).
- **Prompt**: *"Generate a JSON persona for a 45-year-old rural nurse. Include: 'id', 'name', 'bio', 'susceptibility' (0.1-0.9), and 'initial_belief' (-1.0 to 1.0). Ensure the belief aligns with demographic averages."*
- **Output**: Writes `agents.json`.

**`engine.py`**
Contains the `SimulationEngine` class.

- Uses `networkx.barabasi_albert_graph(N=500, m=3)` for realistic scale-free follower networks.
- Maintains $\mathbf{B}$ (belief vector, $N \times 1$), $\mathbf{C}$ (follower count vector, $N \times 1$), and $\mathbf{A}$ (binary adjacency matrix, $N \times N$).
- Method `step(alpha, beta, epsilon)` computes the $F(t)$ matrix and executes the DeGroot update equation using `scipy.sparse` matrix multiplication for performance.
- Method `get_metrics()` calculates the Esteban-Ray index and cluster centroids using `networkx.algorithms.community.louvain_communities()`.

**`main.py`**
The FastAPI entry point exposing the SSE endpoint.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()
engine = SimulationEngine("agents.json")

@app.get("/stream")
async def stream_simulation(alpha: float = 0.5, beta: float = 0.2, epsilon: float = 0.4):
    async def event_generator():
        engine.reset()
        for step in range(100):
            state = engine.step(alpha, beta, epsilon)
            payload = {
                "step": step,
                "beliefs": state.beliefs.tolist(),
                "polarization": state.polarization,
                "echo_coefficient": state.echo_coefficient
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.1) # Throttle for UI rendering
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```


***

## 4. Frontend Specification (React/Vite)

### 4.1 Dependencies

```json
"dependencies": {
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "zustand": "^4.5.0",
  "react-vega": "^7.6.0",
  "vega": "^5.25.0",
  "vega-lite": "^5.16.3",
  "vis-network-react": "^3.0.0",
  "vis-network": "^9.1.9",
  "lucide-react": "^0.330.0"
},
"devDependencies": {
  "vite": "^5.0.8",
  "tailwindcss": "^3.4.1"
}
```


### 4.2 Zustand Store (`store.js`)

This is the central nervous system of the frontend.

```javascript
import { create } from 'zustand';

export const useSimStore = create((set) => ({
  // Simulation State
  isPlaying: false,
  currentStep: 0,
  nodes: [], // agent data for vis.js
  metricsHistory: [], // array of { step, polarization, echo_coefficient } for Vega
  
  // Platform Parameters (Controls)
  params: { alpha: 0.5, beta: 0.2, epsilon: 0.4 },
  setParams: (newParams) => set((state) => ({ params: { ...state.params, ...newParams } })),
  
  // SSE Update Handler
  ingestStep: (payload) => set((state) => ({
    currentStep: payload.step,
    nodes: state.nodes.map((node, i) => ({
      ...node,
      belief: payload.beliefs[i],
      color: payload.beliefs[i] > 0 ? '#ef4444' : '#3b82f6' // Red/Blue mapping
    })),
    metricsHistory: [...state.metricsHistory, {
      step: payload.step,
      polarization: payload.polarization,
      echo: payload.echo_coefficient
    }]
  })),
  
  // Transport control
  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
  reset: () => set({ currentStep: 0, metricsHistory: [] })
}));
```


### 4.3 SSE Integration Component (`SimController.jsx`)

```javascript
import { useEffect } from 'react';
import { useSimStore } from './store';

export default function SimController() {
  const { isPlaying, params, ingestStep } = useSimStore();

  useEffect(() => {
    if (!isPlaying) return;
    
    const url = `http://localhost:8000/stream?alpha=${params.alpha}&beta=${params.beta}&epsilon=${params.epsilon}`;
    const eventSource = new EventSource(url);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      ingestStep(data);
      if (data.step >= 99) eventSource.close();
    };
    
    return () => eventSource.close();
  }, [isPlaying, params, ingestStep]);
  
  return null; // Logic-only component
}
```


### 4.4 Visualizations

- **Network Graph (`vis-network-react`)**: Bind `nodes` and `edges` to the component. Configure the physics engine with `barnesHut: { gravitationalConstant: -2000 }` to keep nodes clustered tightly based on connection strength. Map node colors to the belief scale (-1 to 1 interpolation).
- **Polarization Chart (`react-vega`)**: Use a line chart tracking the `metricsHistory` array. Map `step` to X and `polarization` to Y.
- **Belief Histogram (`react-vega`)**: A binned bar chart displaying the raw `payload.beliefs` array at the current step. Watching this shift from a bell curve to a bimodal distribution is the visual proof of polarization.

***

## 5. Execution Timeline (36 Hours)

### Hours 1-4: Foundation \& Data Seeding (You)

- Download the Pew demographic CSV.
- Write the Qwen 3.6 script via Ollama to generate 500 agents and save to `agents.json`.
- Partner installs React+Vite, Tailwind, and Zustand, and builds the static UI shell (sidebar for parameters, main grid for charts).


### Hours 4-12: The Simulation Engine (You)

- Implement `engine.py`. Translate the equations from Section 1 into NumPy matrix operations.
- Write tests confirming that with $\alpha = 1.0$, polarization diverges, and with $\alpha = 0.0$, consensus forms.
- Partner wires up dummy data to `vis-network-react` and `react-vega` to ensure charts render correctly.


### Hours 12-18: The Bridge (Joint)

- Wrap `engine.py` in FastAPI.
- Implement the SSE `/stream` endpoint.
- Partner implements `EventSource` in React and routes the incoming data to the Zustand store.
- **Milestone**: Pressing "Play" in the UI triggers the backend, data flows through SSE, and the frontend charts animate.


### Hours 18-28: Scientific Grounding \& UI Polish (Joint)

- Implement the Esteban-Ray polarization index in the backend.
- Partner refines the visualization: mapping node colors to ideology, smoothing network physics, and adding hover tooltips (Agent Inspector) to show the LLM persona.
- Add preset scenario buttons: "Chronological Feed", "Engagement-Maximized", "Diversity-Nudged".


### Hours 28-36: The Narrative \& Buffer

- Run the full simulations to capture stable data points.
- Implement the Null Hypothesis baseline (random beliefs).
- Prepare the pitch: Focus entirely on how identical populations fracture differently based purely on algorithmic design parameters.

