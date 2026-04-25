<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Network, DataSet } from 'vis-network/standalone';
  import { simStore, type Agent, type GraphEdge } from './store';

  let container: HTMLDivElement;
  let network: Network | null = null;
  let nodesDataset: DataSet<any> | null = null;
  let edgesDataset: DataSet<any> | null = null;
  let isStabilized = false;
  let isInitializing = false;
  let lastRenderedStep = 0;
  const RENDER_INTERVAL = 1; // Update visuals every step (cheap with batched DataSet.update)

  // Convert belief (-1 to 1) to color - matches histogram gradient
  function beliefToColor(belief: number): string {
    // Blue (-1) -> Purple (0) -> Red (+1)
    if (belief <= 0) {
      const t = (belief + 1); // 0 to 1 as belief goes from -1 to 0
      const r = Math.round(59 + (168 - 59) * t);
      const g = Math.round(130 + (85 - 130) * t);
      const b = Math.round(247 + (85 - 247) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const t = belief; // 0 to 1 as belief goes from 0 to 1
      const r = Math.round(168 + (239 - 168) * t);
      const g = Math.round(85 + (68 - 85) * t);
      const b = Math.round(85 + (68 - 85) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  function getNodeSize(socialCapital: number, maxCapital: number): number {
    // Reduced node sizes for better edge visibility
    const minSize = 4;
    const maxSize = 12;
    const normalized = Math.sqrt(socialCapital / Math.max(maxCapital, 1));
    return minSize + (maxSize - minSize) * normalized;
  }

  function initializeNetwork(agents: Agent[], edges: GraphEdge[], beliefs: number[]) {
    if (!container || agents.length === 0 || isInitializing) return;
    if (network) return; // Already initialized
    
    isInitializing = true;

    const maxCapital = Math.max(...agents.map(a => a.social_capital || 100));

    // Create nodes with belief-based colors - smaller sizes
    const nodes = agents.map((agent, i) => {
      const belief = beliefs[i] ?? agent.initial_belief;
      const color = beliefToColor(belief);
      return {
        id: agent.id,
        color: {
          background: color,
          border: 'rgba(255,255,255,0.5)',
          highlight: { background: color, border: '#ffffff' },
          hover: { background: color, border: '#ffffff' }
        },
        size: getNodeSize(agent.social_capital || 100, maxCapital),
        title: `${agent.name}\nBelief: ${belief.toFixed(3)}\nFollowers: ${agent.social_capital || 0}`
      };
    });

    // Create directed edges - no smoothing or arrows for performance
    const visEdges = edges.map((edge, i) => ({
      id: i,
      from: edge.from,
      to: edge.to,
      arrows: { to: { enabled: true, scaleFactor: 0.35, type: 'arrow' } },
      color: {
        color: 'rgba(147, 157, 177, 0.2)',
        highlight: '#60a5fa',
        hover: '#60a5fa'
      },
      width: 0.5,
      smooth: false,
      hoverWidth: 1.2,
      selectionWidth: 1.5
    }));

    nodesDataset = new DataSet(nodes);
    edgesDataset = new DataSet(visEdges);

    const options = {
      nodes: {
        shape: 'dot',
        borderWidth: 1.5,
        borderWidthSelected: 3,
        font: { size: 0 }, // No labels for performance
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.3)',
          size: 4,
          x: 0,
          y: 2
        }
      },
      edges: {
        smooth: false,
        selectionWidth: 1.5,
        hoverWidth: 1.2
      },
      physics: {
        enabled: true,
        solver: 'barnesHut',
        barnesHut: {
          gravitationalConstant: -3000,
          centralGravity: 0.4,
          springLength: 120,
          springConstant: 0.06,
          damping: 0.15,
          avoidOverlap: 0.3
        },
        stabilization: {
          enabled: true,
          iterations: 200,
          updateInterval: 25,
          fit: true
        },
        maxVelocity: 80,
        minVelocity: 0.1,
        timestep: 0.5
      },
      interaction: {
        hover: true,
        tooltipDelay: 50,
        hideEdgesOnDrag: false,
        hideEdgesOnZoom: false,
        dragNodes: true,
        dragView: true,
        zoomView: true,
        multiselect: true,
        navigationButtons: false,
        keyboard: {
          enabled: true,
          speed: { x: 10, y: 10, zoom: 0.02 }
        }
      },
      layout: {
        improvedLayout: false,
        randomSeed: 42
      }
    };

    network = new Network(container, { nodes: nodesDataset, edges: edgesDataset }, options);

    // Handle node selection
    network.on('selectNode', (params) => {
      const nodeId = params.nodes[0];
      const agent = agents.find(a => a.id === nodeId);
      if (agent) {
        simStore.selectAgent(agent);
      }
    });

    network.on('deselectNode', () => {
      simStore.selectAgent(null);
    });

    // No physics re-enable on drag - user maintains direct control
    // Nodes move only where dragged, no global physics simulation

    // Handle stabilization complete
    network.once('stabilizationIterationsDone', () => {
      isStabilized = true;
      isInitializing = false;
      network?.setOptions({ physics: { enabled: false } });
      network?.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
    });
  }

  function updateBeliefs(beliefs: number[], agents: Agent[]) {
    if (!nodesDataset || beliefs.length === 0 || !isStabilized) return;

    const maxCapital = Math.max(...agents.map(a => a.social_capital || 100));

    const updates = agents.map((agent, i) => {
      const color = beliefToColor(beliefs[i]);
      return {
        id: agent.id,
        color: {
          background: color,
          border: 'rgba(255,255,255,0.5)',
          highlight: { background: color, border: '#ffffff' },
          hover: { background: color, border: '#ffffff' }
        },
        size: getNodeSize(agent.social_capital || 100, maxCapital),
        title: `${agent.name}\nBelief: ${beliefs[i].toFixed(3)}\nFollowers: ${agent.social_capital || 0}`
      };
    });

    nodesDataset.update(updates);
  }

  // Watch for initialization
  $: if ($simStore.isInitialized && $simStore.agents.length > 0 && container && !network) {
    initializeNetwork($simStore.agents, $simStore.edges, $simStore.beliefs);
  }

  // Watch for belief updates during simulation - throttled to every RENDER_INTERVAL steps
  $: if (network && isStabilized && $simStore.beliefs.length > 0 && $simStore.currentStep > 0) {
    const stepDiff = $simStore.currentStep - lastRenderedStep;
    if (stepDiff >= RENDER_INTERVAL || $simStore.currentStep === $simStore.totalSteps) {
      updateBeliefs($simStore.beliefs, $simStore.agents);
      lastRenderedStep = $simStore.currentStep;
    }
  }

  // Reset network colors when simulation resets
  $: if ($simStore.currentStep === 0 && $simStore.beliefs.length > 0 && network && isStabilized) {
    lastRenderedStep = 0;
    updateBeliefs($simStore.beliefs, $simStore.agents);
  }

  onDestroy(() => {
    if (network) {
      network.destroy();
      network = null;
      nodesDataset = null;
      edgesDataset = null;
    }
  });
</script>

<div class="network-container">
  <div bind:this={container} class="network-graph"></div>
  
  <div class="network-overlay">
    <div class="network-legend">
      <div class="legend-title">Belief Spectrum</div>
      <div class="legend-gradient">
        <div class="gradient-bar"></div>
        <div class="gradient-labels">
          <span>-1 Liberal</span>
          <span>0</span>
          <span>+1 Conservative</span>
        </div>
      </div>
    </div>
    <div class="network-controls">
      <span class="control-hint">Drag nodes to interact</span>
    </div>
  </div>

  {#if !$simStore.isInitialized}
    <div class="loading-overlay">
      <div class="loading-spinner"></div>
      <span>Waiting for simulation data...</span>
    </div>
  {:else if !isStabilized && !network}
    <div class="loading-overlay">
      <div class="loading-spinner"></div>
      <span>Building network graph...</span>
    </div>
  {/if}
</div>

<style>
  .network-container {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 400px;
    background: linear-gradient(135deg, #0a0c10 0%, #111318 100%);
    border-radius: 8px;
    overflow: hidden;
  }

  .network-graph {
    width: 100%;
    height: 100%;
  }

  .network-overlay {
    position: absolute;
    bottom: 12px;
    left: 12px;
    right: 12px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    pointer-events: none;
  }

  .network-legend {
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: rgba(15, 17, 23, 0.92);
    padding: 12px 14px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(8px);
  }

  .legend-title {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .legend-gradient {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .gradient-bar {
    width: 160px;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #3b82f6, #a855f7, #ef4444);
  }

  .gradient-labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: rgba(255, 255, 255, 0.5);
  }

  .network-controls {
    background: rgba(15, 17, 23, 0.92);
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(8px);
  }

  .control-hint {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
  }

  .loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: linear-gradient(135deg, #0a0c10 0%, #111318 100%);
    color: rgba(255, 255, 255, 0.6);
    font-size: 14px;
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
