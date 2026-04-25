<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Network, DataSet } from 'vis-network/standalone';
  import { simStore, type Agent, type GraphEdge } from './store';

  let container: HTMLDivElement;
  let network: Network | null = null;
  let nodesDataset: DataSet<any> | null = null;
  let edgesDataset: DataSet<any> | null = null;
  let isStabilized = false;

  // Convert belief (-1 to 1) to color - matches histogram gradient
  function beliefToColor(belief: number): string {
    // Blue (-1) -> Purple (0) -> Red (+1)
    // Using same colors as histogram: #3b82f6 -> #a855f7 -> #ef4444
    if (belief <= 0) {
      // Blue (#3b82f6) to Purple (#a855f7)
      const t = (belief + 1); // 0 to 1 as belief goes from -1 to 0
      const r = Math.round(59 + (168 - 59) * t);
      const g = Math.round(130 + (85 - 130) * t);
      const b = Math.round(246 + (247 - 246) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Purple (#a855f7) to Red (#ef4444)
      const t = belief; // 0 to 1 as belief goes from 0 to 1
      const r = Math.round(168 + (239 - 168) * t);
      const g = Math.round(85 + (68 - 85) * t);
      const b = Math.round(247 + (68 - 247) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  // Node size based on social capital
  function getNodeSize(socialCapital: number, maxCapital: number): number {
    const minSize = 6;
    const maxSize = 18;
    const normalized = Math.sqrt(socialCapital / Math.max(maxCapital, 1));
    return minSize + (maxSize - minSize) * normalized;
  }

  function initializeNetwork(agents: Agent[], edges: GraphEdge[], beliefs: number[]) {
    if (!container || agents.length === 0) return;
    if (network) return; // Already initialized

    console.log('[v0] Initializing network with', agents.length, 'agents and', edges.length, 'edges');

    const maxCapital = Math.max(...agents.map(a => a.social_capital || 100));

    // Create nodes
    const nodes = agents.map((agent, i) => {
      const belief = beliefs[i] ?? agent.initial_belief;
      return {
        id: agent.id,
        label: '',
        title: `${agent.name}\nBelief: ${belief.toFixed(3)}`,
        color: beliefToColor(belief),
        size: getNodeSize(agent.social_capital || 100, maxCapital),
        borderWidth: 1,
        borderWidthSelected: 2
      };
    });

    // Sample edges for performance - show max 2000 edges
    const maxEdges = 2000;
    const sampleRate = edges.length > maxEdges ? Math.ceil(edges.length / maxEdges) : 1;
    const sampledEdges = edges.filter((_, i) => i % sampleRate === 0);

    console.log('[v0] Using', sampledEdges.length, 'sampled edges');

    const visEdges = sampledEdges.map((edge, i) => ({
      id: i,
      from: edge.from,
      to: edge.to,
      arrows: { to: { enabled: true, scaleFactor: 0.4 } },
      color: { 
        color: 'rgba(100, 116, 139, 0.2)', 
        highlight: 'rgba(59, 130, 246, 0.6)' 
      },
      width: 0.5,
      smooth: false
    }));

    nodesDataset = new DataSet(nodes);
    edgesDataset = new DataSet(visEdges);

    const options = {
      nodes: {
        shape: 'dot',
        font: { color: '#ffffff', size: 10 },
        shadow: false,
        borderColor: 'rgba(255,255,255,0.3)'
      },
      edges: {
        smooth: false,
        selectionWidth: 1.5
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -30,
          centralGravity: 0.01,
          springLength: 50,
          springConstant: 0.08,
          damping: 0.4,
          avoidOverlap: 0.5
        },
        stabilization: {
          enabled: true,
          iterations: 100,
          updateInterval: 50
        },
        maxVelocity: 50,
        minVelocity: 0.1
      },
      interaction: {
        hover: true,
        tooltipDelay: 50,
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

    // Disable physics after stabilization for better performance
    network.once('stabilizationIterationsDone', () => {
      console.log('[v0] Network stabilized');
      isStabilized = true;
      network?.setOptions({ physics: { enabled: false } });
    });

    // Also handle stabilization progress
    network.on('stabilizationProgress', (params) => {
      const progress = Math.round((params.iterations / params.total) * 100);
      if (progress % 25 === 0) {
        console.log('[v0] Stabilization:', progress + '%');
      }
    });
  }

  function updateBeliefs(beliefs: number[], agents: Agent[]) {
    if (!nodesDataset || beliefs.length === 0 || !isStabilized) return;

    const updates = agents.map((agent, i) => ({
      id: agent.id,
      color: beliefToColor(beliefs[i]),
      title: `${agent.name}\nBelief: ${beliefs[i].toFixed(3)}`
    }));

    nodesDataset.update(updates);
  }

  // Watch for initialization
  $: if ($simStore.isInitialized && $simStore.agents.length > 0 && container && !network) {
    initializeNetwork($simStore.agents, $simStore.edges, $simStore.beliefs);
  }

  // Watch for belief updates during simulation
  $: if (network && $simStore.beliefs.length > 0 && $simStore.currentStep > 0) {
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
      <span>Initializing network...</span>
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
    background: rgba(15, 17, 23, 0.85);
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
</style>
