<script lang="ts">
  import { onDestroy } from 'svelte';
  import { simStore, latestMetric, type ModelType, type ExposureMode } from './store';
  import { createStreamUrl, fetchInit } from './api';

  let eventSource: EventSource | null = null;

  // ── Local copies for two-way binding ───────────────────────────────
  let alpha       = $simStore.params.alpha;
  let beta        = $simStore.params.beta;
  let epsilon     = $simStore.params.epsilon;
  let steps       = $simStore.params.steps;
  let interval    = $simStore.params.interval;
  let agentQuantity = $simStore.agentQuantity || 500;
  let seedInput: string = '';

  let modelType         = $simStore.dynamics.model_type;
  let exposureMode      = $simStore.dynamics.exposure_mode;
  let topK              = $simStore.dynamics.top_k_visible;
  let selBeta           = $simStore.dynamics.selective_exposure_beta;
  let distAlpha         = $simStore.dynamics.distance_decay_alpha;
  let repRho            = $simStore.dynamics.repulsion_threshold_rho;
  let repGamma          = $simStore.dynamics.repulsion_strength_gamma;
  let noiseSigma        = $simStore.dynamics.noise_sigma;

  // Sync store → local when external changes arrive (e.g. preset applied)
  $: alpha         = $simStore.params.alpha;
  $: beta          = $simStore.params.beta;
  $: epsilon       = $simStore.params.epsilon;
  $: steps         = $simStore.params.steps;
  $: interval      = $simStore.params.interval;
  $: agentQuantity = $simStore.agentQuantity || 500;
  $: modelType     = $simStore.dynamics.model_type;
  $: exposureMode  = $simStore.dynamics.exposure_mode;
  $: topK          = $simStore.dynamics.top_k_visible;
  $: selBeta       = $simStore.dynamics.selective_exposure_beta;
  $: distAlpha     = $simStore.dynamics.distance_decay_alpha;
  $: repRho        = $simStore.dynamics.repulsion_threshold_rho;
  $: repGamma      = $simStore.dynamics.repulsion_strength_gamma;
  $: noiseSigma    = $simStore.dynamics.noise_sigma;

  // ── Derived visibility flags ────────────────────────────────────────
  $: useTopK       = exposureMode === 'top_k' || exposureMode === 'sampled';
  $: showDecayAlpha = modelType === 'confirmation_bias';
  $: showRepulsion = modelType === 'repulsive_bc';
  // epsilon is always visible (it's the BC threshold in all non-degroot modes)

  // ── Platform param helpers ──────────────────────────────────────────
  function updateParam(param: 'alpha' | 'beta' | 'epsilon', value: number) {
    if (param === 'alpha' && value + $simStore.params.beta > 1) {
      simStore.setParams({ alpha: value, beta: 1 - value });
    } else if (param === 'beta' && $simStore.params.alpha + value > 1) {
      simStore.setParams({ beta: value, alpha: 1 - value });
    } else {
      simStore.setParams({ [param]: value });
    }
    // Keep ρ ≥ ε in the dynamics config
    if (param === 'epsilon') {
      const rho = $simStore.dynamics.repulsion_threshold_rho;
      if (rho < value) simStore.setDynamics({ repulsion_threshold_rho: value });
    }
  }

  function updateSteps(value: number) {
    simStore.setParams({ steps: Math.max(1, Math.min(1000, Math.round(value))) });
  }

  // ── Dynamics helpers ────────────────────────────────────────────────
  function setModel(m: ModelType) {
    simStore.setDynamics({ model_type: m });
  }

  function setExposure(e: ExposureMode) {
    simStore.setDynamics({ exposure_mode: e });
  }

  // ── Transport ───────────────────────────────────────────────────────
  async function startSimulation() {
    if ($simStore.isPlaying) { stopSimulation(); return; }
    simStore.startSimulation();
    const url = createStreamUrl($simStore.params, $simStore.dynamics);
    eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        simStore.ingestStep(data);
        if (data.step >= $simStore.totalSteps) stopSimulation();
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
    eventSource?.close();
    eventSource = null;
    simStore.completeSimulation();
  }

  function resetSimulation() {
    stopSimulation();
    simStore.reset();
  }

  function parsedSeed(): number | undefined {
    const trimmed = seedInput.trim();
    if (trimmed === '') return undefined;
    const n = parseInt(trimmed, 10);
    return Number.isFinite(n) ? n : undefined;
  }

  async function applyAgentQuantity() {
    const clamped = Math.max(1, Math.min(500, Math.round(agentQuantity)));
    stopSimulation();
    simStore.setLoading(true);
    simStore.setError(null);
    try {
      const data = await fetchInit(clamped, parsedSeed());
      simStore.initialize(data);
    } catch (err) {
      console.error('Failed to refresh init data:', err);
      simStore.setError('Failed to reload graph with requested agent count.');
    }
  }

  async function randomizeSeed() {
    const newSeed = Math.floor(Math.random() * 1_000_000);
    seedInput = String(newSeed);
    const clamped = Math.max(1, Math.min(500, Math.round(agentQuantity)));
    stopSimulation();
    simStore.setLoading(true);
    simStore.setError(null);
    try {
      const data = await fetchInit(clamped, newSeed);
      simStore.initialize(data);
    } catch (err) {
      console.error('Failed to randomize graph seed:', err);
      simStore.setError('Failed to rebuild graph with new seed.');
    }
  }

  onDestroy(() => stopSimulation());

  // ── Derived display values ──────────────────────────────────────────
  $: baselineWeight = Math.max(0, 1 - $simStore.params.alpha - $simStore.params.beta).toFixed(2);

  const MODEL_LABELS: Record<ModelType, string> = {
    degroot:           'DeGroot',
    confirmation_bias: 'Conf. Bias',
    bounded_confidence:'Bounded Conf.',
    repulsive_bc:      'Repulsive BC',
  };
  const EXPOSURE_LABELS: Record<ExposureMode, string> = {
    all_followed: 'All Followed',
    top_k:        'Top K',
    sampled:      'Sampled',
  };
</script>

<aside class="control-panel">

  <!-- ── Transport ─────────────────────────────────────────────────── -->
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
      <div class="progress-fill" style="width: {($simStore.currentStep / $simStore.totalSteps) * 100}%"></div>
    </div>

    <div class="transport-controls">
      <button class="btn btn-primary" on:click={startSimulation} disabled={!$simStore.isInitialized}>
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

  <!-- ── Platform Parameters ───────────────────────────────────────── -->
  <section class="panel-section">
    <div class="panel-section-header">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line>
        <line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line>
        <line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line>
        <line x1="17" y1="16" x2="23" y2="16"></line>
      </svg>
      <h2>Platform Parameters</h2>
    </div>

    <div class="parameter-group">
      <!-- Alpha — only meaningful for DeGroot mode -->
      <div class="parameter" class:param-dimmed={modelType !== 'degroot'}>
        <div class="parameter-header">
          <label class="parameter-label" for="alpha">
            Echo Chamber Strength (α)
            {#if modelType !== 'degroot'}<span class="param-tag">DeGroot only</span>{/if}
          </label>
          <span class="parameter-value">{$simStore.params.alpha.toFixed(2)}</span>
        </div>
        <input type="range" id="alpha" min="0" max="1" step="0.01"
          bind:value={alpha} on:input={() => updateParam('alpha', alpha)}
          disabled={$simStore.isPlaying || modelType !== 'degroot'} />
        <p class="parameter-description">Platform preference for homophily (feed curation scoring).</p>
      </div>

      <!-- Beta — only meaningful for DeGroot mode -->
      <div class="parameter" class:param-dimmed={modelType !== 'degroot'}>
        <div class="parameter-header">
          <label class="parameter-label" for="beta">
            Virality Bias (β)
            {#if modelType !== 'degroot'}<span class="param-tag">DeGroot only</span>{/if}
          </label>
          <span class="parameter-value">{$simStore.params.beta.toFixed(2)}</span>
        </div>
        <input type="range" id="beta" min="0" max="1" step="0.01"
          bind:value={beta} on:input={() => updateParam('beta', beta)}
          disabled={$simStore.isPlaying || modelType !== 'degroot'} />
        <p class="parameter-description">Platform preference for outrage content (feed curation scoring).</p>
      </div>

      <!-- Epsilon — confidence threshold, used by all non-degroot modes -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="epsilon">Confidence Threshold (ε)</label>
          <span class="parameter-value">{$simStore.params.epsilon.toFixed(2)}</span>
        </div>
        <input type="range" id="epsilon" min="0" max="1" step="0.01"
          bind:value={epsilon} on:input={() => updateParam('epsilon', epsilon)}
          disabled={$simStore.isPlaying} />
        <p class="parameter-description">
          Maximum belief distance for mutual influence (BC / repulsive modes).
          Also sets the DeGroot feed BC filter.
        </p>
      </div>

      <!-- Steps -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="steps">Simulation Steps</label>
          <span class="parameter-value">{$simStore.params.steps}</span>
        </div>
        <input type="range" id="steps" min="10" max="1000" step="10"
          bind:value={steps} on:input={() => updateSteps(steps)}
          disabled={$simStore.isPlaying} />
        <p class="parameter-description">Timesteps per run.</p>
      </div>

      <!-- Step Interval -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="interval">Step Interval</label>
          <span class="parameter-value">
            {interval === 0 ? 'max speed' : `${interval.toFixed(2)}s`}
          </span>
        </div>
        <input type="range" id="interval" min="0" max="1" step="0.01"
          bind:value={interval}
          on:input={() => simStore.setParams({ interval })}
          disabled={$simStore.isPlaying} />
        <p class="parameter-description">Delay between steps. Set to 0 for maximum speed.</p>
      </div>

      <!-- Agent Count + Graph Seed — shared Apply button -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="agentQuantity">Agent Count</label>
          <span class="parameter-value">{$simStore.agentQuantity}</span>
        </div>
        <div class="transport-controls">
          <input id="agentQuantity" class="parameter-input" type="number"
            min="1" max="500" step="1" bind:value={agentQuantity}
            disabled={$simStore.isPlaying} />
        </div>
      </div>

      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="graphSeed">Graph Seed</label>
          <span class="parameter-value">{seedInput || 'random'}</span>
        </div>
        <div class="transport-controls">
          <input id="graphSeed" class="parameter-input" type="text" inputmode="numeric"
            pattern="[0-9]*" placeholder="Random"
            bind:value={seedInput}
            disabled={$simStore.isPlaying} />
          <button class="btn btn-secondary btn-icon" on:click={randomizeSeed}
            disabled={$simStore.isPlaying} title="Generate a random seed and rebuild">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="2" y="2" width="20" height="20" rx="4" ry="4"></rect>
              <circle cx="8" cy="8" r="1.5" fill="currentColor"></circle>
              <circle cx="16" cy="8" r="1.5" fill="currentColor"></circle>
              <circle cx="8" cy="16" r="1.5" fill="currentColor"></circle>
              <circle cx="16" cy="16" r="1.5" fill="currentColor"></circle>
              <circle cx="12" cy="12" r="1.5" fill="currentColor"></circle>
            </svg>
          </button>
          <button class="btn btn-secondary" on:click={applyAgentQuantity}
            disabled={$simStore.isPlaying} title="Rebuild graph with current agent count and seed">
            Apply
          </button>
        </div>
        <p class="parameter-description">
          Enter a number for a reproducible topology, or leave blank for a new random graph on each Apply.
        </p>
      </div>

      {#if modelType === 'degroot'}
        <div class="metric-card" style="margin-top: 4px;">
          <span class="metric-label">Baseline / Chronological Weight</span>
          <span class="metric-value">{baselineWeight}</span>
          <p class="parameter-description" style="margin-top: 6px;">
            Remaining feed weight: <code>1 − α − β</code>
          </p>
        </div>
      {/if}
    </div>
  </section>

  <!-- ── Dynamics Model ───────────────────────────────────────────── -->
  <section class="panel-section">
    <div class="panel-section-header">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
        <path d="M4.93 4.93a10 10 0 0 0 0 14.14"></path>
      </svg>
      <h2>Dynamics Model</h2>
    </div>

    <!-- Model type pill selector -->
    <div class="field-label">Influence rule</div>
    <div class="pill-group">
      {#each (['degroot', 'confirmation_bias', 'bounded_confidence', 'repulsive_bc'] as const) as m}
        <button
          class="pill"
          class:pill-active={modelType === m}
          on:click={() => setModel(m)}
          disabled={$simStore.isPlaying}
          title={m}
        >
          {MODEL_LABELS[m]}
        </button>
      {/each}
    </div>

    <!-- Model-type description -->
    <p class="parameter-description model-desc">
      {#if modelType === 'degroot'}
        Classic weighted average with platform-curated feed weights. α and β drive curation.
      {:else if modelType === 'confirmation_bias'}
        Influence decays smoothly with opinion distance: W̃ ∝ exp(−α·|Δ|).
      {:else if modelType === 'bounded_confidence'}
        Hard cutoff: agents only influence each other when |Δ| ≤ ε.
      {:else}
        Close opinions attract, mid-range are ignored, extreme opinions repel.
      {/if}
    </p>

    <!-- Exposure mode pill selector -->
    <div class="field-label" style="margin-top: 8px;">Exposure (feed) mode</div>
    <div class="pill-group">
      {#each (['all_followed', 'top_k', 'sampled'] as const) as e}
        <button
          class="pill"
          class:pill-active={exposureMode === e}
          on:click={() => setExposure(e)}
          disabled={$simStore.isPlaying}
          title={e}
        >
          {EXPOSURE_LABELS[e]}
        </button>
      {/each}
    </div>
    <p class="parameter-description model-desc">
      {#if exposureMode === 'all_followed'}
        Every followed account is visible every tick.
      {:else if exposureMode === 'top_k'}
        Each agent sees only the top-K followed accounts by homophily + activity score.
      {:else}
        Followed accounts are sampled probabilistically, weighted by similarity and activity.
      {/if}
    </p>

    <!-- Conditional sliders -->
    <div class="parameter-group" style="margin-top: 8px;">

      {#if useTopK}
        <!-- Top K -->
        <div class="parameter">
          <div class="parameter-header">
            <label class="parameter-label" for="topK">Visible Neighbours (K)</label>
            <span class="parameter-value">{$simStore.dynamics.top_k_visible}</span>
          </div>
          <input type="range" id="topK" min="1" max="30" step="1"
            bind:value={topK}
            on:input={() => simStore.setDynamics({ top_k_visible: topK })}
            disabled={$simStore.isPlaying} />
          <p class="parameter-description">Max followed accounts visible each tick.</p>
        </div>

        <!-- Selective exposure β -->
        <div class="parameter">
          <div class="parameter-header">
            <label class="parameter-label" for="selBeta">Selective Exposure (β<sub>s</sub>)</label>
            <span class="parameter-value">{$simStore.dynamics.selective_exposure_beta.toFixed(1)}</span>
          </div>
          <input type="range" id="selBeta" min="0" max="10" step="0.1"
            bind:value={selBeta}
            on:input={() => simStore.setDynamics({ selective_exposure_beta: selBeta })}
            disabled={$simStore.isPlaying} />
          <p class="parameter-description">
            Higher values increase preference for like-minded accounts in the feed ranking.
          </p>
        </div>
      {/if}

      {#if showDecayAlpha}
        <!-- Confirmation-bias decay α -->
        <div class="parameter">
          <div class="parameter-header">
            <label class="parameter-label" for="distAlpha">Distance Decay (α<sub>d</sub>)</label>
            <span class="parameter-value">{$simStore.dynamics.distance_decay_alpha.toFixed(1)}</span>
          </div>
          <input type="range" id="distAlpha" min="0" max="10" step="0.1"
            bind:value={distAlpha}
            on:input={() => simStore.setDynamics({ distance_decay_alpha: distAlpha })}
            disabled={$simStore.isPlaying} />
          <p class="parameter-description">
            Controls how sharply influence falls with opinion distance: W̃ ∝ exp(−α<sub>d</sub>·|Δ|).
          </p>
        </div>
      {/if}

      {#if showRepulsion}
        <!-- Repulsion threshold ρ -->
        <div class="parameter">
          <div class="parameter-header">
            <label class="parameter-label" for="repRho">Repulsion Onset (ρ)</label>
            <span class="parameter-value">{$simStore.dynamics.repulsion_threshold_rho.toFixed(2)}</span>
          </div>
          <input type="range" id="repRho"
            min={$simStore.params.epsilon} max="2" step="0.01"
            bind:value={repRho}
            on:input={() => simStore.setDynamics({ repulsion_threshold_rho: repRho })}
            disabled={$simStore.isPlaying} />
          <p class="parameter-description">
            Opinions further than ρ push agents apart. Must be ≥ ε ({$simStore.params.epsilon.toFixed(2)}).
          </p>
        </div>

        <!-- Repulsion strength γ -->
        <div class="parameter">
          <div class="parameter-header">
            <label class="parameter-label" for="repGamma">Repulsion Strength (γ)</label>
            <span class="parameter-value">{$simStore.dynamics.repulsion_strength_gamma.toFixed(2)}</span>
          </div>
          <input type="range" id="repGamma" min="0" max="2" step="0.01"
            bind:value={repGamma}
            on:input={() => simStore.setDynamics({ repulsion_strength_gamma: repGamma })}
            disabled={$simStore.isPlaying} />
          <p class="parameter-description">Gain applied to repulsive interactions (φ = −γ·Δ for |Δ| > ρ).</p>
        </div>
      {/if}

      <!-- Noise σ — always shown, but small -->
      <div class="parameter">
        <div class="parameter-header">
          <label class="parameter-label" for="noiseSigma">Belief Noise (σ)</label>
          <span class="parameter-value">{$simStore.dynamics.noise_sigma.toFixed(3)}</span>
        </div>
        <input type="range" id="noiseSigma" min="0" max="0.1" step="0.001"
          bind:value={noiseSigma}
          on:input={() => simStore.setDynamics({ noise_sigma: noiseSigma })}
          disabled={$simStore.isPlaying} />
        <p class="parameter-description">
          Std of i.i.d. Gaussian noise added each step. Non-zero noise can break
          hard BC symmetry and produce richer dynamics.
        </p>
      </div>
    </div>
  </section>

  <!-- ── Scenario Presets ──────────────────────────────────────────── -->
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
      <button class="btn btn-preset" class:active={$simStore.activePreset === 'chronological'}
        on:click={() => simStore.applyPreset('chronological')}
        disabled={$simStore.isPlaying}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        Chronological Feed
      </button>
      <button class="btn btn-preset" class:active={$simStore.activePreset === 'engagement'}
        on:click={() => simStore.applyPreset('engagement')}
        disabled={$simStore.isPlaying}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path>
        </svg>
        Engagement Maximized
      </button>
      <button class="btn btn-preset" class:active={$simStore.activePreset === 'diversity'}
        on:click={() => simStore.applyPreset('diversity')}
        disabled={$simStore.isPlaying}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="2" y1="12" x2="22" y2="12"></line>
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
        </svg>
        Diversity Nudged
      </button>
      <button class="btn btn-preset" class:active={$simStore.activePreset === 'echo_bubble'}
        on:click={() => simStore.applyPreset('echo_bubble')}
        disabled={$simStore.isPlaying}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <circle cx="12" cy="12" r="4"></circle>
        </svg>
        Echo Bubble
      </button>
      <button class="btn btn-preset" class:active={$simStore.activePreset === 'polarized'}
        on:click={() => simStore.applyPreset('polarized')}
        disabled={$simStore.isPlaying}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="2" y1="12" x2="8" y2="12"></line>
          <line x1="16" y1="12" x2="22" y2="12"></line>
          <circle cx="11" cy="12" r="3"></circle>
          <circle cx="13" cy="12" r="3"></circle>
        </svg>
        Repulsive Polarization
      </button>
    </div>
  </section>

  <!-- ── Current Metrics ───────────────────────────────────────────── -->
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
        <span class="metric-label">Polarization Index (Normalized)</span>
        <span class="metric-value">
          {$latestMetric ? $latestMetric.polarization_normalized.toFixed(4) : '0.0000'}
        </span>
        <p class="parameter-description" style="margin-top: 4px;">
          Raw ER: {$latestMetric ? $latestMetric.polarization.toFixed(4) : '0.0000'}
        </p>
      </div>
      <div class="metric-card">
        <span class="metric-label">Echo Chamber Coefficient</span>
        <span class="metric-value">
          {$latestMetric ? $latestMetric.echo.toFixed(4) : '0.0000'}
        </span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Active Agents</span>
        <span class="metric-value">{$simStore.agents.length}</span>
      </div>

      <!-- Diagnostics — only populated by the modular path -->
      {#if $latestMetric?.mean_pairwise_distance !== undefined}
        <div class="metric-card diagnostic-card">
          <span class="metric-label">Mean Pairwise Distance</span>
          <span class="metric-value metric-value-sm">
            {$latestMetric.mean_pairwise_distance!.toFixed(4)}
          </span>
        </div>
      {/if}
      {#if $latestMetric?.frac_no_compatible !== undefined}
        <div class="metric-card diagnostic-card">
          <span class="metric-label">Fraction Isolated (no compat. neighbour)</span>
          <span class="metric-value metric-value-sm"
            class:metric-warn={($latestMetric.frac_no_compatible ?? 0) > 0.3}>
            {($latestMetric.frac_no_compatible! * 100).toFixed(1)}%
          </span>
        </div>
      {/if}
      {#if $latestMetric?.mean_exposure_similarity !== undefined}
        <div class="metric-card diagnostic-card">
          <span class="metric-label">Mean Exposure Similarity</span>
          <span class="metric-value metric-value-sm">
            {$latestMetric.mean_exposure_similarity!.toFixed(4)}
          </span>
        </div>
      {/if}
      {#if $latestMetric?.active_exposures !== undefined}
        <div class="metric-card diagnostic-card">
          <span class="metric-label">Active Exposures / Step</span>
          <span class="metric-value metric-value-sm">{$latestMetric.active_exposures}</span>
        </div>
      {/if}
    </div>
  </section>
</aside>

<style>
  /* Pill selector */
  .field-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }

  .pill-group {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .pill {
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 999px;
    border: 1px solid var(--border-color);
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.12s ease;
    white-space: nowrap;
    line-height: 1.4;
  }

  .pill:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--text-primary);
  }

  .pill-active {
    background: rgba(59, 130, 246, 0.15);
    border-color: var(--accent);
    color: var(--accent);
  }

  .pill:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Model description blurb */
  .model-desc {
    margin-top: 4px;
    padding: 6px 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    border-left: 2px solid var(--accent);
  }

  /* Dimmed / inactive params */
  .param-dimmed {
    opacity: 0.45;
  }

  /* Tag next to dimmed labels */
  .param-tag {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 3px;
    padding: 1px 5px;
    margin-left: 6px;
    vertical-align: middle;
  }

  /* Diagnostic metric cards */
  .diagnostic-card {
    border-left: 2px solid var(--border-color);
  }

  .metric-value-sm {
    font-size: 15px;
  }

  .metric-warn {
    color: var(--warning);
  }
</style>
