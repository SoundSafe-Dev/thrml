"""Landauer-Aware Bayesian Inference (LABI).

A Bayesian update rule that pays kT ln 2 per bit only when posterior entropy
drops enough—energy-optimal inference at the edge.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm, entropy_from_samples, landauer_energy


class LandauerAwareBayesianInference(ThermalAlgorithm):
    """Landauer-Aware Bayesian Inference.
    
    **One-liner:** A Bayesian update rule that pays kT ln 2 per bit only when
    posterior entropy drops enough—energy-optimal inference at the edge.
    
    **Why it matters:** Extends battery life and reduces heat while preserving
    decision quality.
    
    **KPIs:**
    - Joules per correct escalation
    - Posterior calibration
    - % updates skipped with ≤1% recall loss
    
    **Attributes:**
    - `posterior_model`: EBM representing posterior distribution
    - `prior_entropy`: Baseline entropy of prior
    - `energy_threshold`: Threshold for skipping updates (Joules)
    """
    
    posterior_model: IsingEBM
    prior_entropy: float
    energy_threshold: float
    update_count: int
    skip_count: int
    
    def __init__(
        self,
        n_variables: int = 16,
        energy_threshold: float = 1e-18,  # 1 aJ threshold
        key: Key[Array, ""] | None = None,
    ):
        """Initialize LABI.
        
        Args:
            n_variables: Number of variables in posterior
            energy_threshold: Minimum entropy reduction (in bits) to justify update
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        self.energy_threshold = energy_threshold
        self.update_count = 0
        self.skip_count = 0
        
        # Create EBM for posterior
        nodes = [SpinNode() for _ in range(n_variables)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(n_variables - 1)]
        
        # Initial prior: uniform (high entropy)
        biases = jnp.zeros((n_variables,))
        weights = jnp.ones((len(edges),)) * 0.1
        beta = jnp.array(1.0)
        
        posterior_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.posterior_model = posterior_model
        
        # Estimate prior entropy
        free_blocks = [Block(nodes)]
        schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, posterior_model, free_blocks, ())
        program = IsingSamplingProgram(posterior_model, free_blocks, clamped_blocks=[])
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        self.prior_entropy = entropy_from_samples(samples[0])
        
        super().__init__()
    
    def forward(
        self,
        key: Key[Array, ""],
        likelihood_scores: Array,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[tuple[bool, Array], dict[str, float]]:
        """Perform energy-aware Bayesian update.
        
        Args:
            key: JAX random key
            likelihood_scores: Likelihood scores for update [n_variables]
            schedule: Sampling schedule
        
        Returns:
            (should_update, posterior_samples) and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
        
        # Compute current posterior entropy
        free_blocks = [Block(self.posterior_model.nodes)]
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, self.posterior_model, free_blocks, ())
        program = IsingSamplingProgram(self.posterior_model, free_blocks, clamped_blocks=[])
        samples_before = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        entropy_before = entropy_from_samples(samples_before[0])
        
        # Simulate posterior after update
        # Update biases with likelihood
        updated_biases = self.posterior_model.biases + likelihood_scores * 0.5
        updated_model = IsingEBM(
            self.posterior_model.nodes,
            self.posterior_model.edges,
            updated_biases,
            self.posterior_model.weights,
            self.posterior_model.beta,
        )
        
        k_init2, k_samp2 = jax.random.split(key, 2)
        init_state2 = hinton_init(k_init2, updated_model, free_blocks, ())
        program2 = IsingSamplingProgram(updated_model, free_blocks, clamped_blocks=[])
        samples_after = sample_states(k_samp2, program2, schedule, init_state2, [], free_blocks)
        entropy_after = entropy_from_samples(samples_after[0])
        
        # Compute entropy reduction
        entropy_delta = entropy_before - entropy_after
        
        # Landauer cost
        landauer_cost = landauer_energy(entropy_delta)
        
        # Decision: update only if cost is justified
        should_update = landauer_cost >= self.energy_threshold
        
        if should_update:
            # Perform actual update
            self = eqx.tree_at(lambda x: x.posterior_model, self, updated_model)
            posterior_samples = samples_after
            self = eqx.tree_at(lambda x: x.update_count, self, self.update_count + 1)
        else:
            # Skip update
            posterior_samples = samples_before
            self = eqx.tree_at(lambda x: x.skip_count, self, self.skip_count + 1)
        
        # Compute KPIs
        joules_per_update = landauer_cost if should_update else 0.0
        skip_rate = self.skip_count / (self.update_count + self.skip_count + 1e-10)
        
        # Posterior calibration: entropy should match true uncertainty
        calibration_error = jnp.abs(entropy_after - entropy_before)
        
        kpis = {
            "should_update": should_update,
            "entropy_delta_bits": float(entropy_delta),
            "landauer_cost_joules": float(landauer_cost),
            "joules_per_update": float(joules_per_update),
            "skip_rate": float(skip_rate),
            "calibration_error": float(calibration_error),
        }
        
        self.kpi_tracker.record("landauer_cost", float(landauer_cost))
        self.kpi_tracker.record("skip_rate", float(skip_rate))
        
        return (should_update, posterior_samples[0]), kpis
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_landauer_cost_joules": self.kpi_tracker.get_mean("landauer_cost") or 0.0,
            "mean_skip_rate": self.kpi_tracker.get_mean("skip_rate") or 0.0,
            "update_count": float(self.update_count),
            "skip_count": float(self.skip_count),
        }
