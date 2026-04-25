<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import embed, { type VisualizationSpec, type Result } from 'vega-embed';
  import { simStore } from './store';

  let container: HTMLDivElement;
  let vegaResult: Result | null = null;

  const spec: VisualizationSpec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    width: 'container',
    height: 120,
    padding: { left: 5, right: 5, top: 5, bottom: 5 },
    background: 'transparent',
    config: {
      axis: {
        labelColor: '#9ca3af',
        titleColor: '#9ca3af',
        gridColor: '#2e3340',
        domainColor: '#2e3340',
        tickColor: '#2e3340',
        labelFont: 'Inter, system-ui, sans-serif',
        titleFont: 'Inter, system-ui, sans-serif',
        labelFontSize: 10,
        titleFontSize: 11
      },
      view: {
        stroke: 'transparent'
      }
    },
    data: { name: 'beliefs' },
    mark: {
      type: 'bar',
      cornerRadiusTopLeft: 3,
      cornerRadiusTopRight: 3
    },
    encoding: {
      x: {
        bin: { maxbins: 20 },
        field: 'belief',
        type: 'quantitative',
        title: 'Belief',
        scale: { domain: [-1, 1] },
        axis: { tickCount: 5 }
      },
      y: {
        aggregate: 'count',
        type: 'quantitative',
        title: null
      },
      color: {
        field: 'belief',
        type: 'quantitative',
        scale: {
          domain: [-1, 0, 1],
          range: ['#3b82f6', '#a855f7', '#ef4444']
        },
        legend: null
      }
    }
  };

  async function initChart() {
    if (!container) return;

    try {
      vegaResult = await embed(container, spec, {
        actions: false,
        renderer: 'canvas'
      });
    } catch (e) {
      console.error('Failed to initialize histogram:', e);
    }
  }

  function updateChart(beliefs: number[]) {
    if (!vegaResult || beliefs.length === 0) return;

    const data = beliefs.map(b => ({ belief: b }));

    vegaResult.view.change('beliefs',
      vegaResult.view.changeset()
        .remove(() => true)
        .insert(data)
    ).run();
  }

  onMount(() => {
    initChart();
  });

  $: if (vegaResult && $simStore.beliefs.length > 0) {
    updateChart($simStore.beliefs);
  }

  onDestroy(() => {
    if (vegaResult) {
      vegaResult.finalize();
    }
  });

  // Compute bimodality indicator
  $: bimodality = (() => {
    const beliefs = $simStore.beliefs;
    if (beliefs.length === 0) return 0;
    
    const leftCount = beliefs.filter(b => b < -0.3).length;
    const rightCount = beliefs.filter(b => b > 0.3).length;
    const centerCount = beliefs.filter(b => b >= -0.3 && b <= 0.3).length;
    
    // Simple bimodality score: high when edges are populated, center is empty
    const edgeRatio = (leftCount + rightCount) / beliefs.length;
    const centerRatio = centerCount / beliefs.length;
    
    return Math.max(0, edgeRatio - centerRatio);
  })();
</script>

<div class="chart-container">
  <div class="chart-header">
    <span class="chart-title">Belief Distribution</span>
    <span class="chart-value" title="Bimodality indicator">{bimodality.toFixed(2)}</span>
  </div>
  <div bind:this={container} class="chart-wrapper"></div>
</div>
