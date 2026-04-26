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

  // Sarle's Bimodality Coefficient: BC = (skewness² + 1) / kurtosis
  // Threshold: BC > 5/9 ≈ 0.556 indicates bimodal tendency.
  // A single cluster (low variance) correctly returns 0; a perfect half-at-±1
  // split returns 1.0; a normal distribution returns ~0.333.
  $: bimodality = (() => {
    const beliefs = $simStore.beliefs;
    const n = beliefs.length;
    if (n < 3) return 0;

    const mean = beliefs.reduce((s, b) => s + b, 0) / n;
    const m2 = beliefs.reduce((s, b) => s + (b - mean) ** 2, 0) / n;
    if (m2 === 0) return 0; // constant distribution → unimodal

    const m3 = beliefs.reduce((s, b) => s + (b - mean) ** 3, 0) / n;
    const m4 = beliefs.reduce((s, b) => s + (b - mean) ** 4, 0) / n;

    const skewness = m3 / m2 ** 1.5;
    const kurtosis = m4 / m2 ** 2; // Pearson kurtosis (not excess)

    if (kurtosis === 0) return 0;
    return Math.min(1, (skewness ** 2 + 1) / kurtosis);
  })();
</script>

<div class="chart-container">
  <div class="chart-header">
    <span class="chart-title">Belief Distribution</span>
    <div class="chart-value-group">
      <span class="chart-value">{bimodality.toFixed(2)}</span>
      <button
        type="button"
        class="info-tooltip"
        aria-label="Metric explanation"
      >
        i
        <span class="tooltip-content">
          Sarle's Bimodality Coefficient = (skewness² + 1) / kurtosis. Values above 0.556 indicate a bimodal split. Perfect two-camp split ≈ 1.0; single consensus cluster ≈ 0; normal spread ≈ 0.33.
        </span>
      </button>
    </div>
  </div>
  <div bind:this={container} class="chart-wrapper"></div>
</div>
