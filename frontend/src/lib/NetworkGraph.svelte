<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Network, type Options, type Node, type Edge, type DataSet } from 'vis-network';
  import { DataSet as VisDataSet } from 'vis-network/standalone';
  import { simStore, type Agent, type GraphEdge } from './store';

  let container: HTMLDivElement;
  let network: Network | null = null;
  let nodesDataset: DataSet<Node> | null = null;
  let edgesDataset: DataSet<Edge> | null = null;

  // Convert belief (-1 to 1) to color
  function beliefToColor(belief: number): string {
    // Blue (-1) -> Purple (0) -> Red (1)
    if (belief < 0) {
      // Blue to purple
      const t = (belief + 1); // 0 to 1
      const r = Math.round(59 + (168 - 59) * t);
      const g = Math.round(130 + (85 - 130) * t);
      const b = Math.round(246 + (247 - 246) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Purple to red
      const t = belief; // 0 to 1
      const r = Math.round(168 + (239 - 168) * t);
      const g = Math.round(85 + (68 - 85) * t);
      const b = Math.round(247 + (68 - 247) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  // Node size based on social capital
  function getNodeSize(socialCapital: number, maxCapital: number): number {
    const minSize = 8;
    const maxSize = 25;
    const normalized = Math.sqrt(socialCapital / Math.max(maxCapital, 1));
    return minSize + (maxSize - minSize) * normalized;
  }

  function initializeNetwork(agents: Agent[], edges: GraphEdge[], beliefs: number[]) {
    if (!container || agents.length === 0) return;

    const maxCapital = Math.max(...agents.map(a => a.social_capital));

    // Create nodes
    const nodes: Node[] = agents.map((agent, i) => ({
      id: agent.id,
      label: '',
      title: `${agent.name}\nBelief: ${beliefs[i]?.toFixed(3) || agent.initial_belief.toFixed(3)}`,
      color: {
        background: beliefToColor(beliefs[i] ?? agent.initial_belief),
        border: 'rgba(255,255,255,0.2)',
        highlight: {
          background: beliefToColor(beliefs[i] ?? agent.initial_belief),
          border: '#ffffff'
        }
      },
      size: getNodeSize(agent.social_capital, maxCapital),
      borderWidth: 1,
      borderWidthSelected: 3
    }));

    // Create edges - only use a sample to avoid overwhelming the visualization
    const sampledEdges = edges.length > 3000 
      ? edges.filter((_, i) => i % Math.ceil(edges.length / 3000) === 0)
      : edges;

    const visEdges: Edge[] = sampledEdges.map((edge, i) => ({
      id: i,
      from: edge.from,
      to: edge.to,
      arrows: { to: { enabled: true, scaleFactor: 0.3 } },
      color: { color: 'rgba(100, 116, 139, 0.15)', highlight: 'rgba(59, 130, 246, 0.5)' },
      width: 0.5
    }));

    nodesDataset = new VisDataSet(nodes);
    edgesDataset = new VisDataSet(visEdges);

    const options: Options = {
      nodes: {
        shape: 'dot',
        font: { color: '#ffffff', size: 12 },
        shadow: false
      },
      edges: {
        smooth: {
          enabled: true,
          type: 'continuous',
          roundness: 0.5
        },
        selectionWidth: 2
      },
      physics: {
        enabled: true,
        solver: 'barnesHut',
        barnesHut: {
          gravitationalConstant: -3000,
          centralGravity: 0.5,
          springLength: 80,
          springConstant: 0.02,
          damping: 0.4,
          avoidOverlap: 0.3
        },
        stabilization: {
          enabled: true,
          iterations: 150,
          updateInterval: 25
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        hideEdgesOnDrag: true,
        hideEdgesOnZoom: true
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

    // Stabilize then disable physics for performance
    network.once('stabilizationIterationsDone', () => {
      network?.setOptions({ physics: { enabled: false } });
    });
  }

  function updateBeliefs(beliefs: number[], agents: Agent[]) {
    if (!nodesDataset || beliefs.length === 0) return;

    const maxCapital = Math.max(...agents.map(a => a.social_capital));

    const updates = agents.map((agent, i) => ({
      id: agent.id,
      color: {
        background: beliefToColor(beliefs[i]),
        border: 'rgba(255,255,255,0.2)',
        highlight: {
          background: beliefToColor(beliefs[i]),
          border: '#ffffff'
        }
      },
      title: `${agent.name}\nBelief: ${beliefs[i].toFixed(3)}`,
      size: getNodeSize(agent.social_capital, maxCapital)
    }));

    nodesDataset.update(updates);
  }

  // Reactive updates
  $: if ($simStore.isInitialized && $simStore.agents.length > 0 && container && !network) {
    initializeNetwork($simStore.agents, $simStore.edges, $simStore.beliefs);
  }

  $: if (network && $simStore.beliefs.length > 0 && $simStore.currentStep > 0) {
    updateBeliefs($simStore.beliefs, $simStore.agents);
  }

  // Reset network when simulation resets
  $: if ($simStore.currentStep === 0 && $simStore.beliefs.length > 0 && network) {
    updateBeliefs($simStore.beliefs, $simStore.agents);
  }

  onDestroy(() => {
    if (network) {
      network.destroy();
      network = null;
    }
  });
</script>

<div class="network-container">
  <div bind:this={container} class="network-graph"></div>
  
  <div class="network-overlay">
    <div class="network-legend">
      <div class="legend-item">
        <span class="legend-dot liberal"></span>
        <span>Liberal (-1)</span>
      </div>
      <div class="legend-item">
        <span class="legend-dot moderate"></span>
        <span>Moderate (0)</span>
      </div>
      <div class="legend-item">
        <span class="legend-dot conservative"></span>
        <span>Conservative (+1)</span>
      </div>
    </div>
  </div>

  {#if !$simStore.isInitialized}
    <div class="loading-container" style="position: absolute; inset: 0;">
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
  }

  .network-graph {
    width: 100%;
    height: 100%;
  }
</style>
