"""Thermal Bandit Resource Orchestrator (TBRO).

Real-time multi-armed bandit implemented as thermal sampling to route GPU/bitrate
where risk is likely to spike.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm


class ThermalBanditResourceOrchestrator(ThermalAlgorithm):
    """Thermal Bandit Resource Orchestrator.
    
    **One-liner:** Real-time multi-armed bandit (traffic, weather, events)
    implemented as thermal sampling to route GPU/bitrate where risk is likely to spike.
    
    **Why it matters:** Keeps SLA during surges (stadiums, campuses) without
    over-provisioning across all sites.
    
    **KPIs:**
    - % incidents processed within latency SLO
    - GPU-hours saved at fixed SLA
    - Overflow drops
    
    **Attributes:**
    - `n_sites`: Number of sites/arms
    - `bandit_model`: Ising bandit with reward priors
    - `exploration_temperature`: Temperature controlling exploration vs exploitation
    """
    
    n_sites: int
    bandit_model: IsingEBM
    exploration_temperature: Array
    reward_priors: Array
    
    def __init__(
        self,
        n_sites: int = 10,
        exploration_temperature: float = 1.0,
        reward_priors: Array | None = None,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize TBRO.
        
        Args:
            n_sites: Number of sites/arms in bandit
            exploration_temperature: Temperature for exploration (higher = more exploration)
            reward_priors: Prior expected rewards per site [n_sites]
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        self.n_sites = n_sites
        self.exploration_temperature = jnp.array(exploration_temperature)
        
        if reward_priors is None:
            reward_priors = jnp.ones((n_sites,)) * 0.5  # Uniform prior
        
        self.reward_priors = reward_priors
        
        # Create nodes: one per site (arm)
        nodes = [SpinNode() for _ in range(n_sites)]
        
        # Couple sites (exploration coupling)
        edges = []
        for i in range(n_sites):
            for j in range(i + 1, min(i + 3, n_sites)):  # Local coupling
                edges.append((nodes[i], nodes[j]))
        
        # Biases based on reward priors
        biases = reward_priors * 1.0
        weights = jnp.ones((len(edges),)) * 0.2  # Weak coupling
        beta = jnp.array(1.0 / exploration_temperature)
        
        bandit_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.bandit_model = bandit_model
        
        super().__init__()
    
    def forward(
        self,
        key: Key[Array, ""],
        site_risk_scores: Array,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, dict[str, float]]:
        """Sample resource allocation across sites.
        
        Args:
            key: JAX random key
            site_risk_scores: Current risk scores [n_sites] (0-1)
            schedule: Sampling schedule
        
        Returns:
            Resource allocation probabilities [n_sites] and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=20, n_samples=50, steps_per_sample=1)
        
        # Update biases with risk scores and reward history
        # Higher risk -> more resources, but tempered by exploration
        risk_biases = site_risk_scores * 2.0
        reward_biases = self.reward_priors * 1.0
        biases = risk_biases + reward_biases
        
        model = IsingEBM(
            self.bandit_model.nodes,
            self.bandit_model.edges,
            biases,
            self.bandit_model.weights,
            self.exploration_temperature,
        )
        
        free_blocks = [Block(model.nodes)]
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        
        # Extract allocation probabilities
        site_samples = samples[0]  # [n_samples, n_sites]
        allocation_probs = jnp.mean(site_samples.astype(jnp.float32), axis=0)
        allocation_probs = jax.nn.softmax(allocation_probs * 3.0)  # Normalize to probabilities
        
        # Compute KPIs
        # SLO: process high-risk sites with high probability
        high_risk_mask = site_risk_scores > 0.7
        slo_compliance = jnp.mean(allocation_probs[high_risk_mask]) if jnp.any(high_risk_mask) else 0.0
        
        # GPU-hours saved: compare to uniform allocation
        uniform_allocation = jnp.ones((self.n_sites,)) / self.n_sites
        gpu_saved = 1.0 - jnp.sum(allocation_probs) / jnp.sum(uniform_allocation)  # Percent reduction
        
        # Overflow: sites that need resources but didn't get them
        overflow = jnp.sum(jnp.maximum(0.0, site_risk_scores - allocation_probs))
        
        kpis = {
            "slo_compliance": float(slo_compliance),
            "gpu_hours_saved_percent": float(gpu_saved * 100),
            "overflow_drops": float(overflow),
            "total_allocation": float(jnp.sum(allocation_probs)),
        }
        
        self.kpi_tracker.record("slo_compliance", float(slo_compliance))
        self.kpi_tracker.record("gpu_hours_saved", float(gpu_saved))
        
        return allocation_probs, kpis
    
    def update_rewards(self, site_idx: int, reward: float):
        """Update reward prior for a site (bandit update).
        
        Args:
            site_idx: Site index
            reward: Observed reward (0-1)
        """
        # Exponential moving average update
        alpha = 0.1  # Learning rate
        self.reward_priors = self.reward_priors.at[site_idx].set(
            (1 - alpha) * self.reward_priors[site_idx] + alpha * reward
        )
        
        # Update model biases
        biases = self.reward_priors * 1.0
        new_model = eqx.tree_at(
            lambda m: m.biases,
            self.bandit_model,
            biases,
        )
        self = eqx.tree_at(lambda x: x.bandit_model, self, new_model)
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_slo_compliance": self.kpi_tracker.get_mean("slo_compliance") or 0.0,
            "mean_gpu_hours_saved_percent": self.kpi_tracker.get_mean("gpu_hours_saved") or 0.0,
        }
