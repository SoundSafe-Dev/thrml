"""Reservoir-EBM Front-End (REF).

Low-power analog reservoir converts raw streams into rich states; a tiny
Ising prior regularizes readout for stable features under noise.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm, entropy_from_samples


class ReservoirEBMFrontEnd(ThermalAlgorithm):
    """Reservoir-EBM Front-End.
    
    **One-liner:** Low-power analog reservoir converts raw streams into rich states;
    a tiny Ising prior regularizes readout for stable features under noise.
    
    **Why it matters:** Cuts DSP/DNN flops while improving stability in wind, glare, echo.
    
    **KPIs:**
    - Feature stability (PSI) across conditions
    - Energy per feature (µJ)
    - End-to-end AUC vs. baseline CNN
    
    **Attributes:**
    - `readout_model`: Small Ising EBM for feature readout
    - `reservoir_size`: Size of simulated reservoir
    - `feature_size`: Output feature dimension
    """
    
    readout_model: IsingEBM
    reservoir_size: int
    feature_size: int
    reservoir_weights: Array
    
    def __init__(
        self,
        reservoir_size: int = 100,
        feature_size: int = 32,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize REF.
        
        Args:
            reservoir_size: Size of reservoir (analog state dimension)
            feature_size: Output feature dimension
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        self.reservoir_size = reservoir_size
        self.feature_size = feature_size
        
        # Simulate reservoir weights (in hardware, this would be analog)
        # Random sparse connectivity
        self.reservoir_weights = jax.random.normal(key, (reservoir_size, reservoir_size)) * 0.1
        # Add sparse connections
        mask = jax.random.bernoulli(key, p=0.1, shape=(reservoir_size, reservoir_size))
        self.reservoir_weights = self.reservoir_weights * mask.astype(self.reservoir_weights.dtype)
        
        # Create small Ising readout network
        nodes = [SpinNode() for _ in range(feature_size)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(feature_size - 1)]
        
        # Regularizing biases (prefer smooth features)
        biases = jnp.zeros((feature_size,))
        weights = jnp.ones((len(edges),)) * 0.2  # Smoothness regularization
        beta = jnp.array(1.0)
        
        readout_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.readout_model = readout_model
        
        super().__init__()
    
    def simulate_reservoir(self, input_signal: Array) -> Array:
        """Simulate analog reservoir dynamics.
        
        Args:
            input_signal: Input signal [n_timesteps, n_inputs]
        
        Returns:
            Reservoir states [n_timesteps, reservoir_size]
        """
        n_timesteps, n_inputs = input_signal.shape
        
        # Simple reservoir simulation (leaky integrator)
        # In real hardware, this would be analog
        reservoir_states = jnp.zeros((n_timesteps, self.reservoir_size))
        input_projection = jax.random.normal(
            jax.random.key(0), (n_inputs, self.reservoir_size)
        ) * 0.1
        
        leak_rate = 0.9  # Leakage factor
        current_state = jnp.zeros((self.reservoir_size,))
        
        for t in range(n_timesteps):
            input_projected = jnp.dot(input_signal[t], input_projection)
            current_state = leak_rate * current_state + (1 - leak_rate) * (
                jnp.tanh(jnp.dot(current_state, self.reservoir_weights) + input_projected)
            )
            reservoir_states = reservoir_states.at[t].set(current_state)
        
        return reservoir_states
    
    def forward(
        self,
        key: Key[Array, ""],
        raw_stream: Array,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, dict[str, float]]:
        """Process raw stream through reservoir + EBM readout.
        
        Args:
            key: JAX random key
            raw_stream: Raw input stream [n_timesteps, n_inputs]
            schedule: Sampling schedule for EBM readout
        
        Returns:
            Stable features [feature_size] and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
        
        # Step 1: Reservoir processing
        reservoir_states = self.simulate_reservoir(raw_stream)
        
        # Step 2: Aggregate reservoir (mean pooling)
        reservoir_aggregate = jnp.mean(reservoir_states, axis=0)  # [reservoir_size]
        
        # Step 3: Project to feature space and use EBM for regularization
        # Project reservoir to feature space
        projection = jax.random.normal(key, (self.reservoir_size, self.feature_size)) * 0.1
        projected_features = jnp.dot(reservoir_aggregate, projection)
        
        # Use EBM to regularize features
        # Set biases based on projected features
        feature_biases = jnp.tanh(projected_features) * 0.5  # Normalize to reasonable range
        
        readout_model = IsingEBM(
            self.readout_model.nodes,
            self.readout_model.edges,
            feature_biases,
            self.readout_model.weights,
            self.readout_model.beta,
        )
        
        free_blocks = [Block(readout_model.nodes)]
        program = IsingSamplingProgram(readout_model, free_blocks, clamped_blocks=[])
        
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, readout_model, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        
        # Extract stable features
        spin_samples = samples[0]  # [n_samples, feature_size]
        feature_probs = jnp.mean(spin_samples.astype(jnp.float32), axis=0)
        stable_features = jax.nn.sigmoid(feature_probs * 2.0)  # Convert to [0, 1]
        
        # Compute KPIs
        # Feature stability: measure across multiple runs (would need batch processing)
        feature_entropy = entropy_from_samples(stable_features[None, :])
        feature_stability = 1.0 / (feature_entropy + 1e-10)  # Lower entropy = more stable
        
        # Energy per feature (rough estimate)
        # Reservoir: minimal (analog)
        # EBM: sampling cost
        n_samples = schedule.n_samples
        energy_per_feature_uj = n_samples * self.feature_size * 1e-9  # Nanojoules per feature (rough)
        
        kpis = {
            "feature_stability": float(feature_stability),
            "feature_entropy": float(feature_entropy),
            "energy_per_feature_uj": float(energy_per_feature_uj),
            "reservoir_state_norm": float(jnp.linalg.norm(reservoir_aggregate)),
        }
        
        self.kpi_tracker.record("feature_stability", float(feature_stability))
        self.kpi_tracker.record("energy_per_feature", float(energy_per_feature_uj))
        
        return stable_features, kpis
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_feature_stability": self.kpi_tracker.get_mean("feature_stability") or 0.0,
            "mean_energy_per_feature_uj": self.kpi_tracker.get_mean("energy_per_feature") or 0.0,
        }
