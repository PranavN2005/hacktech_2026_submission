<script lang="ts">
  import { simStore } from './store';

  let eventSource: EventSource | null = null;

  // Local copies for slider binding
  let alpha = $simStore.params.alpha;
  let beta = $simStore.params.beta;
  let epsilon = $simStore.params.epsilon;

  // Sync with store
  $: alpha = $simStore.params.alpha;
  $: beta = $simStore.params.beta;
  $: epsilon = $simStore.params.epsilon;

  function updateParam(param: 'alpha' | 'beta' | 'epsilon', value: number) {
    // Ensure alpha + beta <= 1
    if (param === 'alpha' && value + $simStore.params.beta > 1) {
      simStore.setParams({ alpha: value, beta: 1 - value });
    } else if (param === 'beta' && $simStore.params.alpha + value > 1) {
      simStore.setParams({ beta: value, alpha: 1 - value });
    } else {
      simStore.setParams({ [param]: value });
    }
  }

  async function startSimulation() {
    if ($simStore.isPlaying) {
      stopSimulation();
      return;
    }

    simStore.startSimulation();

    const params = $simStore.params;
    const url = `http://localhost:8000/stream?alpha=${params.alpha}&beta=${params.beta}&epsilon=${params.epsilon}&steps=${params.steps}&interval=${params.interval}`;

    eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        simStore.ingestStep(data);

        if (data.step >= $simStore.totalSteps) {
          stopSimulation();
        }
      } catch (e) {
        console.error('Failed to parse SSE data:', e);
      }
    };

    eventSource.onerror = (err) => {
      console.error('SSE error:', err);
      stopSimulation();
      simStore.setError('Connection to simulation server lost');
    };
  }

  function stopSimulation() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    simStore.completeSimulation();
  }

  function resetSimulation() {
    stopSimulation();
    simStore.reset();
  }

  function applyPreset(preset: 'chronological' | 'engagement' | 'diversity') {
    simStore.applyPreset(preset);
  }

  // Derived values for display
  $: baselineWeight = Math.max(0, 1 - $simStore.params.alpha - $simStore.params.beta).toFixed(2);
</script>

<aside class="control-panel">
  <!-- Transport Controls -->
  <section class="panel-section">
    <div class="panel-section-header">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      <h2>Simulation</h2>
    </div>

    <div class="step-display">
      <span class="step-label">Step</span>
      <span class="step-value">{$simStore.currentStep}</span>
      <span class="step-total">/ {$simStore.totalSteps}</span>
    </div>

    <div class="progress-bar">
      <div 
        class="progress-fill" 
        style="width: {($simStore.currentStep / $simStore.totalSteps) * 100}%"
      ></div>
    </div>

    <div class="transport-controls">
      <button 
        class="btn btn-primary"
        on:click={startSimulation}
        disabled={!$simStore.isInitialized}
      >
        {#if $simStore.isPlaying}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="4" width="4" height="16"></rect>
            <rect x="14" y="4" width="4" height="16"></rect>
          </svg>
          Pause
        {:else}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          {$simStore.currentStep > 0 ? 'Resume' : 'Start'}
        {/if}
      </button>
      <button 
        class="btn btn-secondary btn-icon"
        on:click={resetSimulation}
        disabled={!$simStore.isInitialized}
        title="Reset simulation"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
          <path d="M3 3v5h5"></path>
        </svg>
      </button>
    </div>
  </section>

  <!-- Parameters -->
  <section class="panel-section">
    <div class="panel-section-header">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="4" y1="21" x2="4" y2="14"></line>
        <line x1="4" y1="10" x2="4" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12" y2="3"></line>
        <line x1="20" y1="21" x2="20" y2="16"></line>
        <line x1="20" y1="12" x2="20" y2="3"></line>
        <line x1="1" y1="14" x2="7" y2="14"></line>
        <line x1="9" y1="8" x2="15" y2="8"></line>
        <line x1="17" y1="16" x2="23" y2="16"></line>
      </svg>
      <h2>Platform Parameters</h2>
    </div>

    <div class="parameter-group">
      <!-- Alpha -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="alpha">Echo Chamber Strength (α)</label>
          <span class="parameter-value">{$simStore.params.alpha.toFixed(2)}</span>
        </div>
        <input 
          type="range" 
          id="alpha"
          min="0" 
          max="1" 
          step="0.01"
          bind:value={alpha}
          on:input={() => updateParam('alpha', alpha)}
          disabled={$simStore.isPlaying}
        />
        <p class="parameter-description">
          Platform preference for showing users content they agree with (homophily).
        </p>
      </div>

      <!-- Beta -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="beta">Virality Bias (β)</label>
          <span class="parameter-value">{$simStore.params.beta.toFixed(2)}</span>
        </div>
        <input 
          type="range" 
          id="beta"
          min="0" 
          max="1" 
          step="0.01"
          bind:value={beta}
          on:input={() => updateParam('beta', beta)}
          disabled={$simStore.isPlaying}
        />
        <p class="parameter-description">
          Platform preference for extreme/outrage content regardless of agreement.
        </p>
      </div>

      <!-- Epsilon -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="epsilon">Confidence Threshold (ε)</label>
          <span class="parameter-value">{$simStore.params.epsilon.toFixed(2)}</span>
        </div>
        <input 
          type="range" 
          id="epsilon"
          min="0" 
          max="1" 
          step="0.01"
          bind:value={epsilon}
          on:input={() => updateParam('epsilon', epsilon)}
          disabled={$simStore.isPlaying}
        />
        <p class="parameter-description">
          Maximum belief distance for agents to influence each other.
        </p>
      </div>

      <!-- Baseline display -->
      <div class="metric-card" style="margin-top: 8px;">
        <span class="metric-label">Baseline / Chronological Weight</span>
        <span class="metric-value">{baselineWeight}</span>
      </div>
    </div>
  </section>

  <!-- Presets -->
  <section class="panel-section">
    <div class="panel-section-header">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="7" height="7"></rect>
        <rect x="14" y="3" width="7" height="7"></rect>
        <rect x="14" y="14" width="7" height="7"></rect>
        <rect x="3" y="14" width="7" height="7"></rect>
      </svg>
      <h2>Scenario Presets</h2>
    </div>

    <div class="preset-buttons">
      <button 
        class="btn btn-preset"
        on:click={() => applyPreset('chronological')}
        disabled={$simStore.isPlaying}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        Chronological Feed
      </button>
      <button 
        class="btn btn-preset"
        on:click={() => applyPreset('engagement')}
        disabled={$simStore.isPlaying}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path>
        </svg>
        Engagement Maximized
      </button>
      <button 
        class="btn btn-preset"
        on:click={() => applyPreset('diversity')}
        disabled={$simStore.isPlaying}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="2" y1="12" x2="22" y2="12"></line>
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
        </svg>
        Diversity Nudged
      </button>
    </div>
  </section>

  <!-- Metrics Summary -->
  <section class="panel-section">
    <div class="panel-section-header">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="20" x2="18" y2="10"></line>
        <line x1="12" y1="20" x2="12" y2="4"></line>
        <line x1="6" y1="20" x2="6" y2="14"></line>
      </svg>
      <h2>Current Metrics</h2>
    </div>

    <div class="metrics-bar" style="flex-direction: column; gap: 8px;">
      <div class="metric-card">
        <span class="metric-label">Polarization Index</span>
        <span class="metric-value">
          {$simStore.metricsHistory.length > 0 
            ? $simStore.metricsHistory[$simStore.metricsHistory.length - 1].polarization.toFixed(4)
            : '0.0000'}
        </span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Echo Chamber Coefficient</span>
        <span class="metric-value">
          {$simStore.metricsHistory.length > 0 
            ? $simStore.metricsHistory[$simStore.metricsHistory.length - 1].echo.toFixed(4)
            : '0.0000'}
        </span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Active Agents</span>
        <span class="metric-value">{$simStore.agents.length}</span>
      </div>
    </div>
  </section>
</aside>
