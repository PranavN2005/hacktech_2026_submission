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
          type: 'line',
          strokeWidth: 2,
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
          },
          color: {
            field: 'series',
            type: 'nominal',
            legend: metric === 'polarization'
              ? {
                  orient: 'bottom',
                  direction: 'horizontal',
                  title: null,
                  labelColor: '#9ca3af',
                  labelFont: 'Inter, system-ui, sans-serif',
                  labelFontSize: 10,
                  symbolSize: 80
                }
              : null,
            scale: {
              domain: ['Normalized ER', 'Raw ER', 'Echo'],
              range: ['#3b82f6', '#f59e0b', '#3b82f6']
            }
          }
        }
      },
      {
        mark: {
          type: 'point',
          filled: true,
          size: 30
        },
        encoding: {
          x: { field: 'step', type: 'quantitative' },
          y: { field: 'value', type: 'quantitative' },
          color: {
            field: 'series',
            type: 'nominal',
            legend: null,
            scale: {
              domain: ['Normalized ER', 'Raw ER', 'Echo'],
              range: ['#3b82f6', '#f59e0b', '#3b82f6']
            }
          },
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

    const maxStep = history.length > 0 ? history[history.length - 1].step : 0;
    const data = metric === 'polarization'
      ? history.flatMap(h => [
          {
            step: h.step,
            value: h.polarization_normalized,
            series: 'Normalized ER',
            maxStep,
          },
          {
            step: h.step,
            value: h.polarization,
            series: 'Raw ER',
            maxStep,
          },
        ])
      : history.map(h => ({
          step: h.step,
          value: h.echo,
          series: 'Echo',
          maxStep,
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
  $: currentPoint = $simStore.metricsHistory.length > 0
    ? $simStore.metricsHistory[$simStore.metricsHistory.length - 1]
    : null;

  $: currentValue = currentPoint
    ? (metric === 'polarization' 
        ? currentPoint.polarization_normalized
        : currentPoint.echo)
    : 0;

  $: metricHelpText =
    metric === 'polarization'
      ? 'Esteban-Ray polarization shown two ways: normalized ER (blue, 0-1 display scale) and raw ER (amber, literature formula). Normalized ER divides raw ER by the max possible value for the current graph-community sizes.'
      : 'Echo chamber coefficient = 1 - (mean feed belief distance / mean population belief distance). Higher means more like-minded feeds.';
</script>

<div class="chart-container">
  <div class="chart-header">
    <span class="chart-title">{title}</span>
    <div class="chart-value-group">
      <span class="chart-value">{currentValue.toFixed(3)}</span>
      {#if metric === 'polarization' && currentPoint}
        <span class="chart-subvalue">raw {currentPoint.polarization.toFixed(4)}</span>
      {/if}
      <button
        type="button"
        class="info-tooltip"
        aria-label="Metric explanation"
      >
        i
        <span class="tooltip-content">{metricHelpText}</span>
      </button>
    </div>
  </div>
  <div bind:this={container} class="chart-wrapper"></div>
</div>
