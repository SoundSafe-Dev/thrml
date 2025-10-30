"""Energy-Fingerprinted Scene Memory (EFSM).

Learn a thermal energy fingerprint of each camera's normal environment; flag
anomalies as out-of-manifold energy spikes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm, entropy_from_samples


class EnergyFingerprintedSceneMemory(ThermalAlgorithm):
    """Energy-Fingerprinted Scene Memory.
    
    **One-liner:** Learn a thermal energy fingerprint of each camera's normal
    environment; flag anomalies as out-of-manifold energy spikes.
    
    **Why it matters:** Zero-shot perimeter/scene change detection (blocked exits,
    staged objects) without per-site labeling.
    
    **KPIs:**
    - mAP for scene anomalies
    - Adaptation half-life (hours) and drift stability
    - False alarm per day
    
    **Attributes:**
    - `baseline_model`: Site-specific EBM baseline learned from clean windows
    - `n_features`: Number of scene feature dimensions
    - `adaptation_rate`: Learning rate for online adaptation
    """
    
    baseline_model: IsingEBM
    n_features: int
    adaptation_rate: float
    baseline_energy: Array
    
    def __init__(
        self,
        n_features: int = 64,
        adaptation_rate: float = 0.01,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize EFSM.
        
        Args:
            n_features: Number of scene feature dimensions
            adaptation_rate: Learning rate for online adaptation
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        self.n_features = n_features
        self.adaptation_rate = adaptation_rate
        
        # Create EBM for scene features
        nodes = [SpinNode() for _ in range(n_features)]
        # Fully connected or sparse topology
        edges = []
        for i in range(n_features):
            for j in range(i + 1, min(i + 5, n_features)):  # Local connectivity
                edges.append((nodes[i], nodes[j]))
        
        # Initialize with neutral biases (will be learned from data)
        biases = jnp.zeros((n_features,))
        weights = jnp.ones((len(edges),)) * 0.1
        beta = jnp.array(1.0)
        
        baseline_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.baseline_model = baseline_model
        self.baseline_energy = jnp.array(0.0)
        
        super().__init__()
    
    def fit_baseline(
        self,
        key: Key[Array, ""],
        clean_windows: Array,
        schedule: SamplingSchedule | None = None,
    ):
        """Fit baseline model from clean scene windows.
        
        Args:
            key: JAX random key
            clean_windows: Clean scene feature windows [n_windows, n_features]
            schedule: Sampling schedule for learning
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)
        
        # Convert features to binary (threshold at median)
        thresholds = jnp.median(clean_windows, axis=0)
        clean_binary = (clean_windows > thresholds).astype(jnp.bool_)
        
        # Estimate moments from data
        # For simplicity, use sample means and correlations
        n_samples, n_features = clean_binary.shape
        spin_values = 2 * clean_binary.astype(jnp.float32) - 1  # [-1, 1]
        
        # First moments (biases)
        mean_spins = jnp.mean(spin_values, axis=0)
        # Convert to biases (rough ML estimate)
        biases = jnp.arctanh(jnp.clip(mean_spins, -0.99, 0.99)) * 0.5
        
        # Second moments (weights) - pairwise correlations
        # For now, use simple local correlations
        edge_weights = []
        node_map = {node: i for i, node in enumerate(self.baseline_model.nodes)}
        for edge in self.baseline_model.edges:
            i, j = node_map[edge[0]], node_map[edge[1]]
            corr = jnp.mean(spin_values[:, i] * spin_values[:, j])
            edge_weights.append(corr * 0.3)  # Scale down
        
        weights = jnp.array(edge_weights)
        
        # Update baseline model using eqx.tree_at
        new_model = IsingEBM(
            self.baseline_model.nodes,
            self.baseline_model.edges,
            biases,
            weights,
            self.baseline_model.beta,
        )
        self = eqx.tree_at(lambda x: x.baseline_model, self, new_model)
        
        # Estimate baseline energy
        free_blocks = [Block(self.baseline_model.nodes)]
        
        # Sample from baseline to estimate energy distribution
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, self.baseline_model, free_blocks, ())
        program = IsingSamplingProgram(self.baseline_model, free_blocks, clamped_blocks=[])
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        
        # Compute baseline energy from samples (simplified - use mean over all samples)
        # Take first sample as representative
        if len(samples[0]) > 0:
            sample_state = [samples[0][0:1]]  # Single sample with batch dim
            energy = self.baseline_model.energy(sample_state, free_blocks)
            baseline_energy = jnp.array(float(energy))
        else:
            baseline_energy = jnp.array(0.0)
        self = eqx.tree_at(lambda x: x.baseline_energy, self, baseline_energy)
    
    def forward(
        self,
        key: Key[Array, ""],
        scene_features: Array,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Detect anomalies in scene features.
        
        Args:
            key: JAX random key
            scene_features: Current scene features [n_features]
            schedule: Sampling schedule
        
        Returns:
            Anomaly score (higher = more anomalous) and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
        
        # Convert features to binary
        thresholds = jnp.median(scene_features) if len(scene_features.shape) == 0 else jnp.median(scene_features)
        scene_binary = (scene_features > thresholds).astype(jnp.bool_)
        
        # Clamp features and compute energy
        free_blocks = [Block(self.baseline_model.nodes)]
        clamped_blocks = [Block(self.baseline_model.nodes)]
        clamped_data = [scene_binary.reshape((-1,))]
        
        # Sample conditioned on features to get energy distribution
        program = IsingSamplingProgram(self.baseline_model, free_blocks, clamped_blocks)
        
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, self.baseline_model, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, clamped_data, free_blocks)
        
        # Compute energy of actual scene
        scene_state = [scene_binary.reshape((1, -1))]
        scene_energy = self.baseline_model.energy(scene_state, free_blocks)
        
        # Anomaly = energy spike above baseline
        energy_delta = float(scene_energy - self.baseline_energy)
        anomaly_score = jax.nn.sigmoid(energy_delta * 0.1)  # Normalize to [0, 1]
        
        # Online adaptation (update baseline slowly)
        if energy_delta < 0:  # Normal scene, adapt baseline
            # Simple gradient-based update
            new_biases = self.baseline_model.biases + self.adaptation_rate * jnp.sign(scene_features - jnp.mean(scene_features)) * 0.01
            new_model = eqx.tree_at(
                lambda m: m.biases,
                self.baseline_model,
                new_biases,
            )
            self = eqx.tree_at(lambda x: x.baseline_model, self, new_model)
        
        # Compute KPIs
        entropy = entropy_from_samples(samples[0])
        drift_stability = jnp.abs(energy_delta) / (jnp.abs(self.baseline_energy) + 1e-10)
        
        kpis = {
            "anomaly_score": float(anomaly_score),
            "energy_delta": float(energy_delta),
            "entropy": float(entropy),
            "drift_stability": float(drift_stability),
        }
        
        self.kpi_tracker.record("anomaly_score", float(anomaly_score))
        self.kpi_tracker.record("energy_delta", float(energy_delta))
        
        return float(anomaly_score), kpis
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_anomaly_score": self.kpi_tracker.get_mean("anomaly_score") or 0.0,
            "mean_energy_delta": self.kpi_tracker.get_mean("energy_delta") or 0.0,
            "baseline_energy": float(self.baseline_energy),
        }
