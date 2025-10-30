"""Probabilistic Phase & Time Sync (PPTS).

Use coupled thermal oscillators to auto-synchronize multi-camera/mic feeds
(phase/time) without GPS/NTP—probabilistic clocking.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm


class ProbabilisticPhaseTimeSync(ThermalAlgorithm):
    """Probabilistic Phase & Time Sync.
    
    **One-liner:** Use coupled thermal oscillators to auto-synchronize multi-camera/mic
    feeds (phase/time) without GPS/NTP—probabilistic clocking.
    
    **Why it matters:** Cleaner cross-view tracking and A/V triangulation when network
    time is messy; fewer false tracks.
    
    **KPIs:**
    - Sync error (ms) at 95th percentile
    - Multi-view re-ID accuracy
    - Triangulation error (m)
    
    **Attributes:**
    - `n_sensors`: Number of sensors to synchronize
    - `oscillator_model`: Coupled oscillator Ising model
    - `coupling_strength`: Kuramoto-style coupling strength
    """
    
    n_sensors: int
    oscillator_model: IsingEBM
    coupling_strength: float
    phase_estimates: Array
    
    def __init__(
        self,
        n_sensors: int = 6,
        coupling_strength: float = 0.8,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize PPTS.
        
        Args:
            n_sensors: Number of sensors (cameras/mics) to synchronize
            coupling_strength: Kuramoto coupling strength κ
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        self.n_sensors = n_sensors
        self.coupling_strength = coupling_strength
        
        # Create oscillator nodes: each sensor has phase represented as spin
        # Use multiple spins per sensor to encode phase discretely
        spins_per_phase = 8  # Discretize phase into 8 bins
        nodes = [SpinNode() for _ in range(n_sensors * spins_per_phase)]
        
        # Couple sensors (Kuramoto-style): sensors synchronize phases
        edges = []
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                # Couple phase bins across sensors
                for k in range(spins_per_phase):
                    idx_i = i * spins_per_phase + k
                    idx_j = j * spins_per_phase + k
                    edges.append((nodes[idx_i], nodes[idx_j]))
        
        # Natural frequencies (slight biases per sensor)
        natural_freqs = jax.random.uniform(key, (n_sensors,), minval=-0.1, maxval=0.1)
        biases = []
        for i in range(n_sensors):
            for k in range(spins_per_phase):
                # Circular bias (prefer certain phase bins)
                phase_pref = jnp.sin(2 * jnp.pi * k / spins_per_phase) * natural_freqs[i]
                biases.append(phase_pref)
        biases = jnp.array(biases)
        
        # Coupling weights (stronger for aligned phases)
        weights = jnp.ones((len(edges),)) * coupling_strength
        beta = jnp.array(1.0)  # Temperature controls sync tightness
        
        oscillator_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.oscillator_model = oscillator_model
        
        # Initial phase estimates (uniform)
        self.phase_estimates = jnp.zeros((n_sensors,))
        
        super().__init__()
    
    def forward(
        self,
        key: Key[Array, ""],
        observed_phases: Array | None = None,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, dict[str, float]]:
        """Synchronize sensor phases.
        
        Args:
            key: JAX random key
            observed_phases: Observed phase offsets [n_sensors] (radians, optional)
            schedule: Sampling schedule
        
        Returns:
            Synchronized phase estimates [n_sensors] (radians) and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=100, n_samples=500, steps_per_sample=2)
        
        spins_per_phase = 8
        
        # If observed phases provided, use as bias
        if observed_phases is not None:
            biases = []
            for i in range(self.n_sensors):
                phase = observed_phases[i]
                phase_bin = int((phase / (2 * jnp.pi)) * spins_per_phase) % spins_per_phase
                for k in range(spins_per_phase):
                    # Strong bias to observed phase bin
                    bias = 2.0 if k == phase_bin else 0.0
                    phase_pref = self.oscillator_model.biases[i * spins_per_phase + k]
                    biases.append(bias + phase_pref)
            biases = jnp.array(biases)
        else:
            biases = self.oscillator_model.biases
        
        model = IsingEBM(
            self.oscillator_model.nodes,
            self.oscillator_model.edges,
            biases,
            self.oscillator_model.weights,
            self.oscillator_model.beta,
        )
        
        free_blocks = [Block(model.nodes)]
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        
        # Extract phase estimates from spin patterns
        # Each sensor's phase is encoded in its spin pattern
        spin_samples = samples[0]  # [n_samples, n_nodes]
        spin_probs = jnp.mean(spin_samples.astype(jnp.float32), axis=0)  # Average over samples
        
        # Decode phases: find dominant phase bin per sensor
        phase_estimates = []
        for i in range(self.n_sensors):
            sensor_spins = spin_probs[i * spins_per_phase : (i + 1) * spins_per_phase]
            # Compute circular mean
            angles = 2 * jnp.pi * jnp.arange(spins_per_phase, dtype=jnp.float32) / float(spins_per_phase)
            # Weighted average angle
            cos_mean = jnp.sum(sensor_spins * jnp.cos(angles))
            sin_mean = jnp.sum(sensor_spins * jnp.sin(angles))
            phase = jnp.arctan2(sin_mean, cos_mean)
            phase_estimates.append(phase)
        
        phase_estimates = jnp.array(phase_estimates)
        self = eqx.tree_at(lambda x: x.phase_estimates, self, phase_estimates)
        
        # Compute sync error (variance of phases)
        sync_error_rad = jnp.std(phase_estimates)
        sync_error_ms = sync_error_rad * 1000.0 / (2 * jnp.pi)  # Rough conversion (assuming 1kHz)
        
        # Pairwise phase differences (for triangulation error estimation)
        phase_diffs = []
        for i in range(self.n_sensors):
            for j in range(i + 1, self.n_sensors):
                diff = jnp.abs(phase_estimates[i] - phase_estimates[j])
                diff = jnp.minimum(diff, 2 * jnp.pi - diff)  # Wrap to [0, π]
                phase_diffs.append(diff)
        avg_phase_diff = jnp.mean(jnp.array(phase_diffs))
        
        # Triangulation error estimate (simplified)
        # Assuming speed of sound ~343 m/s, 20kHz sampling → ~1.7cm per sample
        triangulation_error_m = avg_phase_diff * 343.0 / (2 * jnp.pi * 20000.0)
        
        kpis = {
            "sync_error_ms": float(sync_error_ms),
            "sync_error_rad": float(sync_error_rad),
            "avg_phase_diff_rad": float(avg_phase_diff),
            "triangulation_error_m": float(triangulation_error_m),
        }
        
        self.kpi_tracker.record("sync_error_ms", float(sync_error_ms))
        self.kpi_tracker.record("triangulation_error_m", float(triangulation_error_m))
        
        return phase_estimates, kpis
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_sync_error_ms": self.kpi_tracker.get_mean("sync_error_ms") or 0.0,
            "mean_triangulation_error_m": self.kpi_tracker.get_mean("triangulation_error_m") or 0.0,
        }
