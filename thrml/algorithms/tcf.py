"""Thermodynamic Causal Fusion (TCF).

Discover causal edges (not just correlations) among audio/video/sensors by
small energy perturbations and observing equilibrium shifts.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm


class ThermodynamicCausalFusion(ThermalAlgorithm):
    """Thermodynamic Causal Fusion.
    
    **One-liner:** Discover causal edges (not just correlations) among audio/video/sensors
    by small energy perturbations and observing equilibrium shifts.
    
    **Why it matters:** More robust fusion graphs that don't break under domain shift
    (e.g., fog kills video but boosts acoustic cues).
    
    **KPIs:**
    - Causal discovery F1 on semi-synthetic data
    - Robustness delta under modality failure
    - Transfer AUC across sites
    
    **Attributes:**
    - `fusion_model`: EBM coupling different sensor modalities
    - `n_modalities`: Number of sensor modalities
    - `causal_graph`: Learned causal graph structure
    """
    
    fusion_model: IsingEBM
    n_modalities: int
    causal_graph: Array
    
    def __init__(
        self,
        n_modalities: int = 4,
        n_nodes_per_modality: int = 8,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize TCF.
        
        Args:
            n_modalities: Number of sensor modalities (e.g., video, audio, thermal, motion)
            n_nodes_per_modality: Number of nodes per modality
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        self.n_modalities = n_modalities
        
        # Create nodes: one per modality
        nodes = [SpinNode() for _ in range(n_modalities)]
        
        # Initially fully connected (will learn structure)
        edges = []
        for i in range(n_modalities):
            for j in range(i + 1, n_modalities):
                edges.append((nodes[i], nodes[j]))
        
        # Neutral biases initially
        biases = jnp.zeros((n_modalities,))
        weights = jnp.ones((len(edges),)) * 0.1  # Weak initial coupling
        beta = jnp.array(1.0)
        
        fusion_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.fusion_model = fusion_model
        
        # Causal graph: adjacency matrix [n_modalities, n_modalities]
        self.causal_graph = jnp.zeros((n_modalities, n_modalities))
        
        super().__init__()
    
    def discover_causal_structure(
        self,
        key: Key[Array, ""],
        modality_readings: Array,
        perturbation_strength: float = 0.5,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, dict[str, float]]:
        """Discover causal structure via perturb-and-observe.
        
        Args:
            key: JAX random key
            modality_readings: Current modality readings [n_modalities]
            perturbation_strength: Strength of perturbations
            schedule: Sampling schedule
        
        Returns:
            Learned causal graph [n_modalities, n_modalities] and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
        
        # Update biases with current readings
        biases = modality_readings * 1.0
        model = IsingEBM(
            self.fusion_model.nodes,
            self.fusion_model.edges,
            biases,
            self.fusion_model.weights,
            self.fusion_model.beta,
        )
        
        free_blocks = [Block(model.nodes)]
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        # Baseline equilibrium
        samples_baseline = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        baseline_mag = jnp.mean(2 * samples_baseline[0].astype(jnp.float32) - 1, axis=0)
        
        # Perturb each modality and observe effects
        causal_matrix = jnp.zeros((self.n_modalities, self.n_modalities))
        
        keys = jax.random.split(key, self.n_modalities)
        for i in range(self.n_modalities):
            # Perturb modality i (do-intervention)
            perturbed_biases = biases.at[i].set(biases[i] + perturbation_strength)
            model_pert = IsingEBM(
                model.nodes,
                model.edges,
                perturbed_biases,
                model.weights,
                model.beta,
            )
            program_pert = IsingSamplingProgram(model_pert, free_blocks, clamped_blocks=[])
            
            k_init_i, k_samp_i = jax.random.split(keys[i], 2)
            init_state_i = hinton_init(k_init_i, model_pert, free_blocks, ())
            samples_pert = sample_states(k_samp_i, program_pert, schedule, init_state_i, [], free_blocks)
            perturbed_mag = jnp.mean(2 * samples_pert[0].astype(jnp.float32) - 1, axis=0)
            
            # Measure effect on all other modalities (causal effect)
            effect = perturbed_mag - baseline_mag
            causal_matrix = causal_matrix.at[i, :].set(effect)
        
        # Threshold to get binary causal graph
        threshold = 0.1
        causal_graph = (jnp.abs(causal_matrix) > threshold).astype(jnp.float32)
        self = eqx.tree_at(lambda x: x.causal_graph, self, causal_graph)
        
        # Compute KPIs
        # F1 score: would need ground truth for real evaluation
        # For now, compute graph properties
        n_causal_edges = jnp.sum(self.causal_graph) - jnp.trace(self.causal_graph)  # Exclude self-loops
        avg_causal_strength = jnp.mean(jnp.abs(causal_matrix))
        
        kpis = {
            "n_causal_edges": int(n_causal_edges),
            "avg_causal_strength": float(avg_causal_strength),
            "causal_matrix": causal_matrix,
        }
        
        self.kpi_tracker.record("n_causal_edges", float(n_causal_edges))
        self.kpi_tracker.record("causal_strength", float(avg_causal_strength))
        
        return self.causal_graph, kpis
    
    def forward(
        self,
        key: Key[Array, ""],
        modality_readings: Array,
        failed_modalities: Array | None = None,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, dict[str, float]]:
        """Fuse modalities with causal awareness.
        
        Args:
            key: JAX random key
            modality_readings: Current modality readings [n_modalities]
            failed_modalities: Mask of failed modalities [n_modalities] (bool)
            schedule: Sampling schedule
        
        Returns:
            Fused threat score and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
        
        if failed_modalities is None:
            failed_modalities = jnp.zeros((self.n_modalities,), dtype=jnp.bool_)
        
        # Update biases with readings, mask failed modalities
        readings_masked = modality_readings * (1.0 - failed_modalities.astype(jnp.float32))
        biases = readings_masked * 1.5
        
        # Adjust weights based on causal graph (stronger coupling for causal edges)
        # Extract edge weights from causal graph
        edge_weights = []
        node_map = {node: i for i, node in enumerate(self.fusion_model.nodes)}
        for edge in self.fusion_model.edges:
            i, j = node_map[edge[0]], node_map[edge[1]]
            is_causal = self.causal_graph[i, j] > 0 or self.causal_graph[j, i] > 0
            weight = 0.5 if is_causal else 0.1
            edge_weights.append(weight)
        
        weights = jnp.array(edge_weights)
        
        model = IsingEBM(
            self.fusion_model.nodes,
            self.fusion_model.edges,
            biases,
            weights,
            self.fusion_model.beta,
        )
        
        free_blocks = [Block(model.nodes)]
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)
        
        # Fused threat score = mean magnetization
        spin_values = 2 * samples[0].astype(jnp.float32) - 1
        fused_score = jnp.mean(spin_values, axis=0)
        threat_score = jnp.mean(fused_score)  # Aggregate across modalities
        
        # Robustness: measure performance under modality failure
        robustness_delta = jnp.std(fused_score) / (jnp.std(modality_readings) + 1e-10)
        
        kpis = {
            "threat_score": float(threat_score),
            "fused_scores": fused_score,
            "robustness_delta": float(robustness_delta),
            "n_failed_modalities": int(jnp.sum(failed_modalities)),
        }
        
        self.kpi_tracker.record("threat_score", float(threat_score))
        self.kpi_tracker.record("robustness_delta", float(robustness_delta))
        
        return fused_score, kpis
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_threat_score": self.kpi_tracker.get_mean("threat_score") or 0.0,
            "mean_robustness_delta": self.kpi_tracker.get_mean("robustness_delta") or 0.0,
            "mean_causal_edges": self.kpi_tracker.get_mean("n_causal_edges") or 0.0,
        }
