<script lang="ts">
  import { onMount } from 'svelte';
  import { onDestroy } from 'svelte';
  import { simStore } from './lib/store';
  import { fetchInit } from './lib/api';
  import ControlPanel from './lib/ControlPanel.svelte';
  import NetworkGraph from './lib/NetworkGraph.svelte';
  import PolarizationChart from './lib/PolarizationChart.svelte';
  import BeliefHistogram from './lib/BeliefHistogram.svelte';
  import AgentInspector from './lib/AgentInspector.svelte';

  let connectionStatus: 'connecting' | 'connected' | 'error' = 'connecting';
  let healthPoll: number | null = null;

  async function checkHealth() {
    try {
      const res = await fetch('/api/health', { cache: 'no-store' });
      if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
      if (connectionStatus !== 'connected') connectionStatus = 'connected';
    } catch {
      connectionStatus = 'error';
    }
  }

  onMount(async () => {
    simStore.setLoading(true);

    try {
      const data = await fetchInit();
      simStore.initialize(data);
      connectionStatus = 'connected';
    } catch (err) {
      console.error('Failed to initialize:', err);
      simStore.setError(
        'Failed to connect to simulation server. Ensure backend is running on port 8000.'
      );
      connectionStatus = 'error';
    }

    // Keep header status updated if backend goes down (or comes back).
    await checkHealth();
    healthPoll = window.setInterval(checkHealth, 3000);
  });

  onDestroy(() => {
    if (healthPoll !== null) window.clearInterval(healthPoll);
  });
</script>

<div class="app-container">
  <!-- Header -->
  <header class="header">
    <div class="header-title">
      <div class="logo">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <path d="M12 16v-4"></path>
          <path d="M12 8h.01"></path>
        </svg>
      </div>
      <h1>EchoChamber</h1>
    </div>
    
    <div class="header-status">
      <div class="status-badge">
        <span 
          class="status-indicator" 
          class:connected={connectionStatus === 'connected'}
          class:running={$simStore.isPlaying}
        ></span>
        <span>
          {#if connectionStatus === 'connecting'}
            Connecting...
          {:else if connectionStatus === 'error'}
            Disconnected
          {:else if $simStore.isPlaying}
            Running
          {:else}
            Ready
          {/if}
        </span>
      </div>
      <div class="status-badge">
        <span>{$simStore.agents.length} agents</span>
      </div>
    </div>
  </header>

  <!-- Control Panel (Sidebar) -->
  <ControlPanel />

  <!-- Main Content -->
  <main class="main-content">
    <!-- Network Graph -->
    <div class="network-container">
      {#if $simStore.error}
        <div class="error-container">
          <svg class="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <p class="error-message">{$simStore.error}</p>
          <button class="btn btn-secondary" on:click={() => window.location.reload()}>
            Retry Connection
          </button>
        </div>
      {:else if $simStore.isLoading}
        <div class="loading-container">
          <div class="loading-spinner"></div>
          <span>Loading simulation data...</span>
        </div>
      {:else}
        {#key $simStore.initVersion}
          <NetworkGraph />
        {/key}
      {/if}
    </div>

    <!-- Charts Panel -->
    <aside class="charts-panel">
      <PolarizationChart 
        title="Esteban-Ray Polarization" 
        metric="polarization" 
      />
      <PolarizationChart 
        title="Echo Chamber Coefficient" 
        metric="echo" 
      />
      <BeliefHistogram />
    </aside>

    <!-- Agent Inspector -->
    <AgentInspector />
  </main>
</div>
