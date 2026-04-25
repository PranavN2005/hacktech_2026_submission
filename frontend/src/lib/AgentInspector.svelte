<script lang="ts">
  import { simStore } from './store';

  // Get current belief for selected agent
  $: currentBelief = $simStore.selectedAgent 
    ? $simStore.beliefs[$simStore.selectedAgent.id] ?? $simStore.selectedAgent.initial_belief
    : 0;

  // Belief change from initial
  $: beliefChange = $simStore.selectedAgent 
    ? currentBelief - $simStore.selectedAgent.initial_belief
    : 0;

  // Color based on belief
  function getBeliefColor(belief: number): string {
    if (belief < -0.3) return '#3b82f6';
    if (belief > 0.3) return '#ef4444';
    return '#a855f7';
  }

  // Get initials for avatar
  function getInitials(name: string): string {
    return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  }
</script>

<footer class="agent-inspector">
  {#if $simStore.selectedAgent}
    <div class="inspector-content">
      <div 
        class="inspector-avatar" 
        style="background: {getBeliefColor(currentBelief)}"
      >
        {getInitials($simStore.selectedAgent.name)}
      </div>
      
      <div class="inspector-details">
        <div class="inspector-name">{$simStore.selectedAgent.name}</div>
        <div class="inspector-bio">{$simStore.selectedAgent.bio}</div>
      </div>

      <div class="inspector-stats">
        <div class="inspector-stat">
          <span class="inspector-stat-value" style="color: {getBeliefColor(currentBelief)}">
            {currentBelief >= 0 ? '+' : ''}{currentBelief.toFixed(3)}
          </span>
          <span class="inspector-stat-label">Belief</span>
        </div>
        <div class="inspector-stat">
          <span class="inspector-stat-value" style="color: {beliefChange >= 0 ? '#22c55e' : '#ef4444'}">
            {beliefChange >= 0 ? '+' : ''}{beliefChange.toFixed(3)}
          </span>
          <span class="inspector-stat-label">Change</span>
        </div>
        <div class="inspector-stat">
          <span class="inspector-stat-value">{$simStore.selectedAgent.susceptibility.toFixed(2)}</span>
          <span class="inspector-stat-label">Susceptibility</span>
        </div>
        <div class="inspector-stat">
          <span class="inspector-stat-value">{$simStore.selectedAgent.social_capital}</span>
          <span class="inspector-stat-label">Followers</span>
        </div>
      </div>
    </div>
  {:else}
    <div class="inspector-placeholder">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="opacity: 0.5">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
        <circle cx="12" cy="7" r="4"></circle>
      </svg>
      <span>Click on a node in the network to inspect an agent</span>
    </div>
  {/if}
</footer>

<style>
  .inspector-placeholder {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    justify-content: center;
  }
</style>
