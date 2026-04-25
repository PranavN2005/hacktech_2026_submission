<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import embed, { type VisualizationSpec, type Result } from 'vega-embed';
  import { simStore, type MetricPoint } from './store';

  export let title = 'Polarization Index';
  export let metric: 'polarization' | 'echo' = 'polarization';
  
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
    data: { name: 'metrics' },
    layer: [
      {
        mark: {
          type: 'area',
          line: { color: '#3b82f6', strokeWidth: 2 },
          color: {
            x1: 1,
            y1: 1,
            x2: 1,
            y2: 0,
            gradient: 'linear',
            stops: [
              { offset: 0, color: 'rgba(59, 130, 246, 0)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.3)' }
            ]
          },
          interpolate: 'monotone'
        },
        encoding: {
          x: {
            field: 'step',
            type: 'quantitative',
            title: 'Step',
            scale: { domain: [0, 100] }
          },
          y: {
            field: 'value',
            type: 'quantitative',
            title: null,
            scale: { domain: [0, 1] }
          }
        }
      },
      {
        mark: {
          type: 'point',
          filled: true,
          color: '#3b82f6',
          size: 30
        },
        encoding: {
          x: { field: 'step', type: 'quantitative' },
          y: { field: 'value', type: 'quantitative' },
          opacity: {
            condition: { test: 'datum.step === datum.maxStep', value: 1 },
            value: 0
          }
        }
      }
    ]
  };

  async function initChart() {
    if (!container) return;

    try {
      vegaResult = await embed(container, spec, {
        actions: false,
        renderer: 'canvas'
      });
    } catch (e) {
      console.error('Failed to initialize Vega chart:', e);
    }
  }

  function updateChart(history: MetricPoint[]) {
    if (!vegaResult) return;

    const data = history.map(h => ({
      step: h.step,
      value: metric === 'polarization' ? h.polarization : h.echo,
      maxStep: history.length > 0 ? history[history.length - 1].step : 0
    }));

    vegaResult.view.change('metrics', 
      vegaResult.view.changeset()
        .remove(() => true)
        .insert(data)
    ).run();
  }

  onMount(() => {
    initChart();
  });

  $: if (vegaResult && $simStore.metricsHistory) {
    updateChart($simStore.metricsHistory);
  }

  // Reset chart when simulation resets
  $: if (vegaResult && $simStore.currentStep === 0 && $simStore.metricsHistory.length === 0) {
    updateChart([]);
  }

  onDestroy(() => {
    if (vegaResult) {
      vegaResult.finalize();
    }
  });

  // Current value
  $: currentValue = $simStore.metricsHistory.length > 0 
    ? (metric === 'polarization' 
        ? $simStore.metricsHistory[$simStore.metricsHistory.length - 1].polarization 
        : $simStore.metricsHistory[$simStore.metricsHistory.length - 1].echo)
    : 0;
</script>

<div class="chart-container">
  <div class="chart-header">
    <span class="chart-title">{title}</span>
    <span class="chart-value">{currentValue.toFixed(3)}</span>
  </div>
  <div bind:this={container} class="chart-wrapper"></div>
</div>
