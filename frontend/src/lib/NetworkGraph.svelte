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

  // Convert belief (-1 to 1) to color - matches histogram gradient
  function beliefToColor(belief: number): string {
    // Blue (-1) -> Purple (0) -> Red (+1)
    if (belief <= 0) {
      const t = (belief + 1); // 0 to 1 as belief goes from -1 to 0
      const r = Math.round(59 + (168 - 59) * t);
      const g = Math.round(130 + (85 - 130) * t);
      const b = Math.round(246 + (247 - 246) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const t = belief; // 0 to 1 as belief goes from 0 to 1
      const r = Math.round(168 + (239 - 168) * t);
      const g = Math.round(85 + (68 - 85) * t);
      const b = Math.round(247 + (68 - 247) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  function getNodeSize(socialCapital: number, maxCapital: number): number {
    const minSize = 8;
    const maxSize = 20;
    const normalized = Math.sqrt(socialCapital / Math.max(maxCapital, 1));
    return minSize + (maxSize - minSize) * normalized;
  }

  function initializeNetwork(agents: Agent[], edges: GraphEdge[], beliefs: number[]) {
    if (!container || agents.length === 0 || isInitializing) return;
    if (network) return; // Already initialized
    
    isInitializing = true;
    console.log('[v0] Initializing network with', agents.length, 'agents and', edges.length, 'edges');

    const maxCapital = Math.max(...agents.map(a => a.social_capital || 100));

    // Create nodes with belief-based colors
    const nodes = agents.map((agent, i) => {
      const belief = beliefs[i] ?? agent.initial_belief;
      return {
        id: agent.id,
        color: {
          background: beliefToColor(belief),
          border: 'rgba(255,255,255,0.3)',
          highlight: { background: beliefToColor(belief), border: '#ffffff' },
          hover: { background: beliefToColor(belief), border: '#ffffff' }
        },
        size: getNodeSize(agent.social_capital || 100, maxCapital),
        title: `${agent.name}\nBelief: ${belief.toFixed(3)}`
      };
    });

    // Create directed edges
    const visEdges = edges.map((edge, i) => ({
      id: i,
      from: edge.from,
      to: edge.to,
      arrows: { to: { enabled: true, scaleFactor: 0.3 } },
      color: { color: 'rgba(100, 116, 139, 0.15)', highlight: '#3b82f6' },
      width: 0.5,
      smooth: false
    }));

    console.log('[v0] Creating DataSets...');
    nodesDataset = new DataSet(nodes);
    edgesDataset = new DataSet(visEdges);

    const options = {
      nodes: {
        shape: 'dot',
        borderWidth: 1,
        borderWidthSelected: 2,
        font: { size: 0 } // No labels for performance
      },
      edges: {
        smooth: false,
        selectionWidth: 1
      },
      physics: {
        enabled: true,
        solver: 'barnesHut',
        barnesHut: {
          gravitationalConstant: -2000,
          centralGravity: 0.3,
          springLength: 80,
          springConstant: 0.04,
          damping: 0.09,
          avoidOverlap: 0.1
        },
        stabilization: {
          enabled: true,
          iterations: 150,
          updateInterval: 25,
          fit: true
        },
        maxVelocity: 50,
        minVelocity: 0.75
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        hideEdgesOnDrag: true,
        hideEdgesOnZoom: true,
        dragNodes: true,
        dragView: true,
        zoomView: true
      },
      layout: {
        improvedLayout: false,
        randomSeed: 42
      }
    };

    console.log('[v0] Creating Network...');
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

    // Disable physics after stabilization
    network.once('stabilizationIterationsDone', () => {
      console.log('[v0] Network stabilized, disabling physics');
      isStabilized = true;
      isInitializing = false;
      network?.setOptions({ physics: { enabled: false } });
    });
  }

  function updateBeliefs(beliefs: number[], agents: Agent[]) {
    if (!nodesDataset || beliefs.length === 0 || !isStabilized) return;

    const updates = agents.map((agent, i) => ({
      id: agent.id,
      color: {
        background: beliefToColor(beliefs[i]),
        border: 'rgba(255,255,255,0.3)',
        highlight: { background: beliefToColor(beliefs[i]), border: '#ffffff' },
        hover: { background: beliefToColor(beliefs[i]), border: '#ffffff' }
      },
      title: `${agent.name}\nBelief: ${beliefs[i].toFixed(3)}`
    }));

    nodesDataset.update(updates);
  }

  // Watch for initialization
  $: if ($simStore.isInitialized && $simStore.agents.length > 0 && container && !network) {
    initializeNetwork($simStore.agents, $simStore.edges, $simStore.beliefs);
  }

  // Watch for belief updates during simulation
  $: if (network && isStabilized && $simStore.beliefs.length > 0 && $simStore.currentStep > 0) {
    updateBeliefs($simStore.beliefs, $simStore.agents);
  }

  // Reset network colors when simulation resets
  $: if ($simStore.currentStep === 0 && $simStore.beliefs.length > 0 && network && isStabilized) {
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
      <div class="legend-item">
        <span class="legend-dot" style="background: #3b82f6;"></span>
        <span>Liberal (-1)</span>
      </div>
      <div class="legend-item">
        <span class="legend-dot" style="background: #a855f7;"></span>
        <span>Moderate (0)</span>
      </div>
      <div class="legend-item">
        <span class="legend-dot" style="background: #ef4444;"></span>
        <span>Conservative (+1)</span>
      </div>
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
    background: var(--bg-primary);
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
    pointer-events: none;
  }

  .network-legend {
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(15, 17, 23, 0.9);
    padding: 10px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
    font-size: 11px;
    color: var(--text-secondary);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: var(--bg-primary);
    color: var(--text-secondary);
    font-size: 14px;
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
