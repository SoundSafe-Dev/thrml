"""Stochastic-Resonance Signal Lifter (SRSL).

Use controlled thermal noise to amplify sub-threshold patterns via stochastic
resonance instead of filtering them out.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm, mutual_information


class StochasticResonanceSignalLifter(ThermalAlgorithm):
    """Stochastic-Resonance Signal Lifter.
    
    **One-liner:** Use controlled thermal noise to amplify sub-threshold patterns
    (e.g., suppressed gunshots, whispered threats) via stochastic resonance.
    
    **Why it matters:** Pulls weak but meaningful signals above detector thresholds
    without cranking gain (less false alarm from broadband amplification).
    
    **KPIs:**
    - +20-40% recall on low-SNR events at ≤+5% false alarms
    - Energy per uplifted detection (J/event)
    
    **Attributes:**
    - `bistable_spin`: The bistable Ising unit coupled to audio features
    - `beta_range`: Range of beta values to sweep for optimal operating point
    - `signal_window_size`: Size of the feature window to process
    """
    
    bistable_spin: IsingEBM
    beta_range: Array
    signal_window_size: int
    optimal_beta: Array = eqx.field(default_factory=lambda: jnp.array(1.0))
    
    def __init__(
        self,
        signal_window_size: int = 32,
        beta_min: float = 0.1,
        beta_max: float = 5.0,
        n_beta_steps: int = 20,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize SRSL.
        
        Args:
            signal_window_size: Number of feature dimensions
            beta_min: Minimum beta (high temp) for sweep
            beta_max: Maximum beta (low temp) for sweep
            n_beta_steps: Number of beta values to test
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        # Create bistable Ising unit: two stable states
        nodes = [SpinNode() for _ in range(signal_window_size)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(signal_window_size - 1)]
        
        # Bistable potential: negative bias creates two wells
        biases = jnp.zeros((signal_window_size,))
        # Coupling creates collective behavior
        weights = jnp.ones((len(edges),)) * 0.5
        
        beta_init = jnp.array(1.0)
        bistable_spin = IsingEBM(nodes, edges, biases, weights, beta_init)
        
        self.bistable_spin = bistable_spin
        self.beta_range = jnp.linspace(beta_min, beta_max, n_beta_steps)
        self.signal_window_size = signal_window_size
        self.optimal_beta = beta_init
        
        super().__init__()
    
    def forward(
        self,
        key: Key[Array, ""],
        weak_features: Array,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, dict[str, float]]:
        """Process weak features through stochastic resonance.
        
        Args:
            key: JAX random key
            weak_features: Weak signal features [n_features] (sub-threshold)
            schedule: Sampling schedule (default: 50 warmup, 100 samples, 2 steps/sample)
        
        Returns:
            Amplified features and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=2)
        
        # Grid search beta to maximize mutual information
        best_mi = -jnp.inf
        best_beta = float(self.beta_range[0])
        best_amplified = None
        
        k_init, k_samp = jax.random.split(key, 2)
        
        for beta_val in self.beta_range:
            beta = float(beta_val)
            # Update model beta
            model = IsingEBM(
                self.bistable_spin.nodes,
                self.bistable_spin.edges,
                self.bistable_spin.biases,
                self.bistable_spin.weights,
                beta,
            )
            
            # Set up sampling with weak features as bias (coupled to signal)
            # Couple weak signal to spin biases
            signal_biases = weak_features * 0.1  # Scale coupling strength
            model_biased = IsingEBM(
                model.nodes,
                model.edges,
                self.bistable_spin.biases + signal_biases,
                model.weights,
                beta,
            )
            
            free_blocks = [Block(model_biased.nodes)]
            program = IsingSamplingProgram(model_biased, free_blocks, clamped_blocks=[])
            
            init_state = hinton_init(k_init, model_biased, free_blocks, ())
            samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
            
            # Extract amplified signal (mean magnetization)
            spin_samples = samples[0]  # [n_samples, n_spins]
            spin_values = 2 * spin_samples.astype(jnp.float32) - 1  # Convert to [-1, 1]
            amplified = jnp.mean(spin_values, axis=0)
            
            # Compute mutual information between weak_features and amplified
            mi = mutual_information(weak_features, amplified)
            
            if mi > best_mi:
                best_mi = mi
                best_beta = beta
                best_amplified = amplified
        
        # Update optimal_beta using eqx.tree_at (equinox modules are frozen)
        self = eqx.tree_at(lambda x: x.optimal_beta, self, jnp.array(best_beta))
        
        # Final amplification with optimal beta
        model_optimal = IsingEBM(
            self.bistable_spin.nodes,
            self.bistable_spin.edges,
            self.bistable_spin.biases + weak_features * 0.1,
            self.bistable_spin.weights,
            best_beta,
        )
        free_blocks = [Block(model_optimal.nodes)]
        program = IsingSamplingProgram(model_optimal, free_blocks, clamped_blocks=[])
        init_state = hinton_init(k_init, model_optimal, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        spin_samples = samples[0]
        spin_values = 2 * spin_samples.astype(jnp.float32) - 1
        final_amplified = jnp.mean(spin_values, axis=0)
        
        # Compute KPIs
        signal_gain = jnp.std(final_amplified) / (jnp.std(weak_features) + 1e-10)
        energy_per_event = float(len(self.bistable_spin.nodes) * schedule.n_samples * 1e-12)  # Rough estimate
        
        kpis = {
            "mutual_information": float(best_mi),
            "optimal_beta": float(best_beta),
            "signal_gain": float(signal_gain),
            "energy_per_event_joules": energy_per_event,
        }
        
        self.kpi_tracker.record("mutual_information", best_mi)
        self.kpi_tracker.record("signal_gain", float(signal_gain))
        
        return final_amplified, kpis
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_mutual_information": self.kpi_tracker.get_mean("mutual_information") or 0.0,
            "mean_signal_gain": self.kpi_tracker.get_mean("signal_gain") or 0.0,
            "optimal_beta": float(self.optimal_beta),
        }
