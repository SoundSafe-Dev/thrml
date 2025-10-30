"""Boltzmann Policy Planner (BPP) for Escalation.

Fuse detection and response policy in one energy model; temperature controls
risk tolerance, sampling yields the next action (notify, lock, dispatch).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.base import KPITracker, ThermalAlgorithm


class BoltzmannPolicyPlanner(ThermalAlgorithm):
    """Boltzmann Policy Planner for Escalation.
    
    **One-liner:** Fuse detection and response policy in one energy model;
    temperature controls risk tolerance, sampling yields the next action
    (notify, lock, dispatch).
    
    **Why it matters:** Faster, consistent escalations with calibrated risk;
    reduces operator load and decision lag.
    
    **KPIs:**
    - Time-to-escalation (TTE)
    - Intervention precision/recall
    - Operator interventions per incident
    
    **Attributes:**
    - `policy_model`: Ising model coupling threat/tactic nodes to policy actions
    - `action_names`: Names of policy actions (e.g., ["notify", "lock", "dispatch"])
    - `risk_temperature`: Temperature parameter controlling risk tolerance
    """
    
    n_actions: int
    policy_model: IsingEBM
    action_names: list[str]
    risk_temperature: Array
    
    def __init__(
        self,
        action_names: list[str] | None = None,
        playbook_biases: Array | None = None,
        risk_temperature: float = 1.0,
        key: Key[Array, ""] | None = None,
    ):
        """Initialize BPP.
        
        Args:
            action_names: List of policy action names (e.g., ["notify", "lock", "dispatch"])
            playbook_biases: Prior biases for each action from playbooks
            risk_temperature: Temperature for risk tolerance (higher = more conservative)
            key: JAX random key
        """
        if key is None:
            key = jax.random.key(0)
        
        if action_names is None:
            action_names = ["notify", "lock", "dispatch", "no_action"]
        
        self.action_names = action_names
        self.n_actions = len(action_names)
        self.risk_temperature = jnp.array(risk_temperature)
        
        # Create nodes: threat nodes + tactic nodes + policy action nodes
        # Threat nodes: severity indicators
        # Tactic nodes: detected patterns (e.g., "loitering", "crowd_gathering")
        # Policy nodes: response actions
        
        n_threat_levels = 3  # low, medium, high
        n_tactics = 5  # Various threat patterns
        
        nodes = []
        # Threat level nodes (one-hot)
        nodes.extend([SpinNode() for _ in range(n_threat_levels)])
        # Tactic indicator nodes
        nodes.extend([SpinNode() for _ in range(n_tactics)])
        # Policy action nodes (one-hot)
        nodes.extend([SpinNode() for _ in range(self.n_actions)])
        
        # Couple threat -> tactics -> policy
        edges = []
        
        # Threat-to-tactic coupling
        threat_start = 0
        tactic_start = n_threat_levels
        for t_i in range(n_threat_levels):
            for tac_j in range(n_tactics):
                edges.append((nodes[threat_start + t_i], nodes[tactic_start + tac_j]))
        
        # Tactic-to-policy coupling
        policy_start = n_threat_levels + n_tactics
        for tac_i in range(n_tactics):
            for act_j in range(self.n_actions):
                edges.append((nodes[tactic_start + tac_i], nodes[policy_start + act_j]))
        
        # Playbook biases on policy actions
        if playbook_biases is None:
            # Default: prefer notify over lock, lock over dispatch
            playbook_biases = jnp.array([0.5, 0.2, -0.2, 0.1])
        
        biases = []
        # Threat biases (neutral)
        biases.extend([0.0] * n_threat_levels)
        # Tactic biases (weak positive - encourage detection)
        biases.extend([0.1] * n_tactics)
        # Policy biases from playbook
        biases.extend(playbook_biases.tolist())
        biases = jnp.array(biases)
        
        weights = jnp.ones((len(edges),)) * 0.4
        beta = jnp.array(1.0 / risk_temperature)  # Inverse temperature
        
        policy_model = IsingEBM(nodes, edges, biases, weights, beta)
        self.policy_model = policy_model
        
        super().__init__()
    
    def forward(
        self,
        key: Key[Array, ""],
        threat_level: int,
        tactic_scores: Array,
        schedule: SamplingSchedule | None = None,
    ) -> tuple[Array, dict[str, float]]:
        """Sample escalation policy given threat and tactics.
        
        Args:
            key: JAX random key
            threat_level: Threat level (0=low, 1=medium, 2=high)
            tactic_scores: Tactic detection scores [n_tactics] (0-1)
            schedule: Sampling schedule
        
        Returns:
            Policy action probabilities [n_actions] and KPIs
        """
        if schedule is None:
            schedule = SamplingSchedule(n_warmup=30, n_samples=100, steps_per_sample=2)
        
        # Clamp threat level
        n_threat_levels = 3
        threat_biases = jnp.zeros((n_threat_levels,))
        threat_biases = threat_biases.at[threat_level].set(2.0)  # Strong bias to correct threat level
        
        # Update tactic biases with detection scores
        n_tactics = 5
        tactic_biases = tactic_scores * 1.5
        
        # Combine with playbook policy biases
        policy_start = n_threat_levels + n_tactics
        playbook_biases = self.policy_model.biases[policy_start:]
        
        biases = jnp.concatenate([threat_biases, tactic_biases, playbook_biases])
        
        # Adjust beta for risk tolerance
        model = IsingEBM(
            self.policy_model.nodes,
            self.policy_model.edges,
            biases,
            self.policy_model.weights,
            self.risk_temperature,
        )
        
        # Clamp threat and tactic nodes, sample only policy
        threat_nodes = [self.policy_model.nodes[i] for i in range(n_threat_levels)]
        tactic_nodes = [self.policy_model.nodes[i] for i in range(n_threat_levels, n_threat_levels + n_tactics)]
        policy_nodes = [self.policy_model.nodes[i] for i in range(policy_start, policy_start + self.n_actions)]
        
        free_blocks = [Block(policy_nodes)]
        clamped_blocks = [Block(threat_nodes), Block(tactic_nodes)]
        
        # Set clamped values
        threat_state = jnp.zeros((n_threat_levels,), dtype=jnp.bool_)
        threat_state = threat_state.at[threat_level].set(True)
        tactic_state = (tactic_scores > 0.5).astype(jnp.bool_)
        clamped_data = [threat_state, tactic_state]
        
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks)
        
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, ())
        samples = sample_states(k_samp, program, schedule, init_state, clamped_data, free_blocks)
        
        # Extract policy action probabilities
        policy_samples = samples[0]  # [n_samples, n_actions]
        action_probs = jnp.mean(policy_samples.astype(jnp.float32), axis=0)
        action_probs = jax.nn.softmax(action_probs * 3.0)  # Sharp probabilities
        
        # Compute KPIs
        action_idx = jnp.argmax(action_probs)
        time_to_escalation = float(action_probs[action_idx])  # Confidence as proxy for speed
        
        # Precision: if high threat, should escalate (lock/dispatch)
        is_escalation = float(action_idx >= 1)  # notify=0, lock=1, dispatch=2
        should_escalate = float(threat_level >= 1)
        precision = is_escalation * should_escalate / (is_escalation + 1e-10)
        recall = is_escalation * should_escalate / (should_escalate + 1e-10)
        
        kpis = {
            "time_to_escalation": time_to_escalation,
            "intervention_precision": float(precision),
            "intervention_recall": float(recall),
            "selected_action": self.action_names[int(action_idx)],
            "action_confidence": float(action_probs[action_idx]),
        }
        
        self.kpi_tracker.record("time_to_escalation", time_to_escalation)
        self.kpi_tracker.record("intervention_precision", float(precision))
        
        return action_probs, kpis
    
    def get_kpis(self) -> dict[str, float]:
        """Get current KPI values."""
        return {
            "mean_time_to_escalation": self.kpi_tracker.get_mean("time_to_escalation") or 0.0,
            "mean_intervention_precision": self.kpi_tracker.get_mean("intervention_precision") or 0.0,
        }
