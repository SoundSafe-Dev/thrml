"""Thermo-Verifiable Sensing (TVS).

Embed entropy tags (thermo-generated one-time fingerprints) into sensor streams
to prove live-ness & provenance against replay/deepfakes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm, entropy_from_samples


class ThermoVerifiableSensing(ThermalAlgorithm):
    """Thermo-Verifiable Sensing.
    
    **One-liner:** Embed entropy tags (thermo-generated one-time fingerprints) into
    sensor streams to prove live-ness & provenance against replay/deepfakes.
    
    **Why it matters:** Trustworthy feeds for courts/ops; blocks synthetic insertion
    attacks cheaply.
    
    **KPIs:**
    - Attack success rate ↓
    - Verification latency (ms)
    - Bitrate overhead (%)
    
    **Attributes:**
    - `rng_model`: Thermal RNG Ising model for generating nonces
    - `nonce_size`: Size of nonce in bits
    - `watermark_strength`: Strength of watermark embedding
    """
    
    rng_model: IsingEBM
    nonce_size: int
    watermark_strength: float
    nonce_history: list[Array]
    max_history_size: int
    
    def __init__(
        self,
        nonce_size: int = 32,
        watermark_strength: float = 0.1,
        max_history_size: int = 1000,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize TVS.
        
        Args:
            nonce_size: Size of nonce in bits
            watermark_strength: Strength of watermark (0-1)
            max_history_size: Maximum nonce history to keep
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        self.nonce_size = nonce_size
        self.watermark_strength = watermark_strength
        self.max_history_size = max_history_size
        self.nonce_history = []
        
        # Create thermal RNG: chaotic Ising system
        nodes = [SpinNode() for _ in range(nonce_size)]
        # Fully connected for high entropy
        edges = []
        for i in range(nonce_size):
            for j in range(i + 1, nonce_size):
                edges.append((nodes[i], nodes[j]))
        
        # Neutral biases (unbiased RNG)
        biases = jnp.zeros((nonce_size,))
        weights = jax.random.uniform(key, (len(edges),), minval=-0.5, maxval=0.5)  # Random coupling
        beta = jnp.array(0.5)  # Low temperature for randomness
        
        rng_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.rng_model = rng_model
        
        super().__init__()
    
    def generate_nonce(self, key: Key[Array, ""], schedule: SamplingSchedule | None = None) -> Array:
        """Generate a thermal nonce.
        
        Args:
            key: JAX random key
            schedule: Sampling schedule
        
        Returns:
            Nonce bits [nonce_size]
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=100, n_samples=1, steps_per_sample=5)
        
        free_blocks = [Block(self.rng_model.nodes)]
        program = IsingSamplingProgram(self.rng_model, free_blocks, clamped_blocks=[])
        
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, self.rng_model, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        
        nonce = samples[0][0]  # Take first sample
        return nonce
    
    def watermark_stream(
        self,
        key: Key[Array, ""],
        stream_data: Array,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, Array, dict[str, float]]:
        """Embed watermark nonce into stream.
        
        Args:
            key: JAX random key
            stream_data: Stream data to watermark [n_samples, n_features]
            schedule: Sampling schedule for nonce generation
        
        Returns:
            (watermarked_stream, nonce) and KPIs
        """
        # Generate nonce
        nonce = self.generate_nonce(key, schedule)
        self.nonce_history.append(nonce)
        if len(self.nonce_history) > self.max_history_size:
            self.nonce_history.pop(0)
        
        # Embed nonce as low-bit watermark
        # Simple: add nonce pattern scaled by watermark_strength
        nonce_expanded = nonce.astype(jnp.float32) * 2.0 - 1.0  # [-1, 1]
        
        # Repeat nonce to match stream length
        n_samples, n_features = stream_data.shape
        nonce_repeated = jnp.tile(nonce_expanded[None, :], (n_samples, 1))
        
        # Truncate or pad to match features
        if nonce_repeated.shape[1] > n_features:
            nonce_repeated = nonce_repeated[:, :n_features]
        elif nonce_repeated.shape[1] < n_features:
            padding = n_features - nonce_repeated.shape[1]
            nonce_repeated = jnp.pad(nonce_repeated, ((0, 0), (0, padding)), mode="constant")
        
        watermarked = stream_data + self.watermark_strength * nonce_repeated
        
        # Compute bitrate overhead
        nonce_bits = self.nonce_size
        stream_bits = n_samples * n_features * 8  # Assuming 8 bits per sample
        bitrate_overhead = (nonce_bits / stream_bits) * 100.0
        
        kpis = {
            "nonce_entropy": float(entropy_from_samples(nonce[None, :])),
            "bitrate_overhead_percent": float(bitrate_overhead),
            "watermark_strength": self.watermark_strength,
        }
        
        return watermarked, nonce, kpis
    
    def verify_stream(
        self,
        watermarked_stream: Array,
        candidate_nonce: Array | None = None,
        threshold: float = 0.8,
    ) -> tuple[bool, dict[str, float]]:
        """Verify watermark in stream.
        
        Args:
            watermarked_stream: Potentially watermarked stream [n_samples, n_features]
            candidate_nonce: Candidate nonce to check (if None, check against history)
            threshold: Correlation threshold for verification
        
        Returns:
            (is_valid, verification_metrics)
        """
        # Extract watermark pattern
        n_samples, n_features = watermarked_stream.shape
        
        # Simple extraction: correlate stream with nonce pattern
        if candidate_nonce is not None:
            nonce_to_check = candidate_nonce
        else:
            # Check against most recent nonce
            if len(self.nonce_history) == 0:
                return False, {"correlation": 0.0, "verified": False}
            nonce_to_check = self.nonce_history[-1]
        
        nonce_expanded = nonce_to_check.astype(jnp.float32) * 2.0 - 1.0  # [-1, 1]
        if nonce_expanded.shape[0] > n_features:
            nonce_expanded = nonce_expanded[:n_features]
        elif nonce_expanded.shape[0] < n_features:
            padding = n_features - nonce_expanded.shape[0]
            nonce_expanded = jnp.pad(nonce_expanded, (0, padding), mode="constant")
        
        # Extract watermark (simplified: assume we can isolate it)
        # In practice, would need more sophisticated extraction
        stream_normalized = (watermarked_stream - jnp.mean(watermarked_stream)) / (
            jnp.std(watermarked_stream) + 1e-10
        )
        correlation = jnp.mean(stream_normalized * nonce_expanded[None, :])
        
        is_valid = correlation > threshold
        
        kpis = {
            "correlation": float(correlation),
            "verified": is_valid,
            "threshold": threshold,
        }
        
        self.kpi_tracker.record("correlation", float(correlation))
        self.kpi_tracker.record("verified", float(is_valid))
        
        return is_valid, kpis
    
    def forward(
        self,
        key: Key[Array, ""],
        stream_data: Array,
        mode: str = "watermark",
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array | bool, dict[str, float]]:
        """Main forward pass: watermark or verify.
        
        Args:
            key: JAX random key
            stream_data: Stream data [n_samples, n_features]
            mode: "watermark" or "verify"
            schedule: Sampling schedule
        
        Returns:
            Watermarked stream (if mode="watermark") or verification result (if mode="verify")
        """
        if mode == "watermark":
            watermarked, nonce, kpis = self.watermark_stream(key, stream_data, schedule)
            return watermarked, kpis
        elif mode == "verify":
            is_valid, kpis = self.verify_stream(stream_data)
            return is_valid, kpis
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_correlation": self.kpi_tracker.get_mean("correlation") or 0.0,
            "mean_verified_rate": self.kpi_tracker.get_mean("verified") or 0.0,
            "nonce_history_size": float(len(self.nonce_history)),
        }
