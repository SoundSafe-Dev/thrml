#!/usr/bin/env python3
"""Demo script for all 10 thermal algorithms.

This script demonstrates each algorithm with synthetic data and prints KPIs.
Run with: python examples/demo_thermal_algorithms.py
"""

import jax
import jax.numpy as jnp
from thrml import SamplingSchedule
from thrml.algorithms import (
    StochasticResonanceSignalLifter,
    ThermodynamicActivePerceptionScheduling,
    BoltzmannPolicyPlanner,
    EnergyFingerprintedSceneMemory,
    ThermalBanditResourceOrchestrator,
    LandauerAwareBayesianInference,
    ThermodynamicCausalFusion,
    ProbabilisticPhaseTimeSync,
    ThermoVerifiableSensing,
    ReservoirEBMFrontEnd,
)

jax.config.update("jax_enable_x64", True)
key = jax.random.key(42)


def demo_srsl(key):
    """Demo SRSL - Stochastic-Resonance Signal Lifter."""
    print("\n=== 1. Stochastic-Resonance Signal Lifter (SRSL) ===")
    srsl = StochasticResonanceSignalLifter(signal_window_size=32, key=key)
    
    weak_signal = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 32)) * 0.3 + jax.random.normal(key, (32,)) * 0.5
    schedule = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=2)
    key, subkey = jax.random.split(key)
    amplified, kpis = srsl.forward(subkey, weak_signal, schedule)
    
    print(f"  Signal gain: {kpis['signal_gain']:.2f}x")
    print(f"  Mutual information: {kpis['mutual_information']:.4f}")
    print(f"  Optimal beta: {kpis['optimal_beta']:.3f}")
    return key


def demo_taps(key):
    """Demo TAPS - Thermodynamic Active Perception & Scheduling."""
    print("\n=== 2. Thermodynamic Active Perception & Scheduling (TAPS) ===")
    taps = ThermodynamicActivePerceptionScheduling(n_sensors=8, n_bitrate_levels=3, key=key)
    
    threat_scores = jax.random.uniform(key, (8,), minval=0.0, maxval=1.0)
    threat_scores = threat_scores.at[2].set(0.9)
    schedule = SamplingSchedule(n_warmup=20, n_samples=50, steps_per_sample=1)
    key, subkey = jax.random.split(key)
    actions, kpis = taps.forward(subkey, threat_scores, schedule)
    
    print(f"  Total energy: {kpis['total_energy_joules_per_sec']:.4f} J/sec")
    print(f"  Active sensors: {kpis['active_sensors']}")
    print(f"  Threat coverage: {kpis['threat_coverage']:.3f}")
    return key


def demo_bpp(key):
    """Demo BPP - Boltzmann Policy Planner."""
    print("\n=== 3. Boltzmann Policy Planner (BPP) ===")
    bpp = BoltzmannPolicyPlanner(risk_temperature=1.0, key=key)
    
    threat_level = 2  # High threat
    tactic_scores = jnp.array([0.1, 0.3, 0.9, 0.7, 0.2])
    schedule = SamplingSchedule(n_warmup=30, n_samples=100, steps_per_sample=2)
    key, subkey = jax.random.split(key)
    action_probs, kpis = bpp.forward(subkey, threat_level, tactic_scores, schedule)
    
    print(f"  Selected action: {kpis['selected_action']}")
    print(f"  Action confidence: {kpis['action_confidence']:.3f}")
    print(f"  Intervention precision: {kpis['intervention_precision']:.3f}")
    return key


def demo_efsm(key):
    """Demo EFSM - Energy-Fingerprinted Scene Memory."""
    print("\n=== 4. Energy-Fingerprinted Scene Memory (EFSM) ===")
    efsm = EnergyFingerprintedSceneMemory(n_features=64, adaptation_rate=0.01, key=key)
    
    clean_windows = jax.random.normal(key, (100, 64)) * 0.5
    key, subkey = jax.random.split(key)
    efsm.fit_baseline(subkey, clean_windows)
    
    normal_scene = jax.random.normal(key, (64,)) * 0.5
    anomalous_scene = jax.random.normal(key, (64,)) * 2.0 + 5.0
    schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
    
    key, subkey = jax.random.split(key)
    normal_score, _ = efsm.forward(subkey, normal_scene, schedule)
    key, subkey = jax.random.split(key)
    anomaly_score, anomaly_kpis = efsm.forward(subkey, anomalous_scene, schedule)
    
    print(f"  Normal scene score: {normal_score:.3f}")
    print(f"  Anomalous scene score: {anomaly_score:.3f}")
    print(f"  Energy delta: {anomaly_kpis['energy_delta']:.2f}")
    return key


def demo_tbro(key):
    """Demo TBRO - Thermal Bandit Resource Orchestrator."""
    print("\n=== 5. Thermal Bandit Resource Orchestrator (TBRO) ===")
    tbro = ThermalBanditResourceOrchestrator(n_sites=10, exploration_temperature=1.0, key=key)
    
    site_risk_scores = jax.random.uniform(key, (10,), minval=0.0, maxval=1.0)
    site_risk_scores = site_risk_scores.at[3].set(0.95)
    schedule = SamplingSchedule(n_warmup=20, n_samples=50, steps_per_sample=1)
    key, subkey = jax.random.split(key)
    allocation, kpis = tbro.forward(subkey, site_risk_scores, schedule)
    
    print(f"  SLO compliance: {kpis['slo_compliance']:.3f}")
    print(f"  GPU hours saved: {kpis['gpu_hours_saved_percent']:.1f}%")
    print(f"  Overflow drops: {kpis['overflow_drops']:.3f}")
    return key


def demo_labi(key):
    """Demo LABI - Landauer-Aware Bayesian Inference."""
    print("\n=== 6. Landauer-Aware Bayesian Inference (LABI) ===")
    labi = LandauerAwareBayesianInference(n_variables=16, energy_threshold=1e-18, key=key)
    
    likelihood_scores = jax.random.uniform(key, (16,), minval=-0.5, maxval=0.5)
    schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
    key, subkey = jax.random.split(key)
    (should_update, _), kpis = labi.forward(subkey, likelihood_scores, schedule)
    
    print(f"  Should update: {should_update}")
    print(f"  Landauer cost: {kpis['landauer_cost_joules']:.2e} J")
    print(f"  Skip rate: {kpis['skip_rate']:.3f}")
    return key


def demo_tcf(key):
    """Demo TCF - Thermodynamic Causal Fusion."""
    print("\n=== 7. Thermodynamic Causal Fusion (TCF) ===")
    tcf = ThermodynamicCausalFusion(n_modalities=4, n_nodes_per_modality=8, key=key)
    
    modality_readings = jax.random.uniform(key, (4,), minval=0.0, maxval=1.0)
    schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
    key, subkey = jax.random.split(key)
    causal_graph, discovery_kpis = tcf.discover_causal_structure(subkey, modality_readings, schedule=schedule)
    
    print(f"  Causal edges discovered: {discovery_kpis['n_causal_edges']}")
    print(f"  Average causal strength: {discovery_kpis['avg_causal_strength']:.3f}")
    
    failed_modalities = jnp.array([False, True, False, False])
    key, subkey = jax.random.split(key)
    fused_score, fusion_kpis = tcf.forward(subkey, modality_readings, failed_modalities, schedule)
    
    print(f"  Threat score: {fusion_kpis['threat_score']:.3f}")
    print(f"  Robustness delta: {fusion_kpis['robustness_delta']:.3f}")
    return key


def demo_ppts(key):
    """Demo PPTS - Probabilistic Phase & Time Sync."""
    print("\n=== 8. Probabilistic Phase & Time Sync (PPTS) ===")
    ppts = ProbabilisticPhaseTimeSync(n_sensors=6, coupling_strength=0.8, key=key)
    
    observed_phases = jax.random.uniform(key, (6,), minval=0.0, maxval=2 * jnp.pi)
    schedule = SamplingSchedule(n_warmup=100, n_samples=500, steps_per_sample=2)
    key, subkey = jax.random.split(key)
    synced_phases, kpis = ppts.forward(subkey, observed_phases, schedule)
    
    print(f"  Sync error: {kpis['sync_error_ms']:.2f} ms")
    print(f"  Triangulation error: {kpis['triangulation_error_m']:.4f} m")
    return key


def demo_tvs(key):
    """Demo TVS - Thermo-Verifiable Sensing."""
    print("\n=== 9. Thermo-Verifiable Sensing (TVS) ===")
    tvs = ThermoVerifiableSensing(nonce_size=32, watermark_strength=0.1, key=key)
    
    stream_data = jax.random.normal(key, (100, 64))
    schedule = SamplingSchedule(n_warmup=100, n_samples=1, steps_per_sample=5)
    key, subkey = jax.random.split(key)
    watermarked, watermark_kpis = tvs.forward(subkey, stream_data, mode="watermark", schedule=schedule)
    
    print(f"  Bitrate overhead: {watermark_kpis['bitrate_overhead_percent']:.4f}%")
    
    is_valid, verify_kpis = tvs.verify_stream(watermarked)
    print(f"  Verified: {is_valid}")
    print(f"  Correlation: {verify_kpis['correlation']:.3f}")
    return key


def demo_ref(key):
    """Demo REF - Reservoir-EBM Front-End."""
    print("\n=== 10. Reservoir-EBM Front-End (REF) ===")
    ref = ReservoirEBMFrontEnd(reservoir_size=100, feature_size=32, key=key)
    
    raw_stream = jax.random.normal(key, (50, 16))
    schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)
    key, subkey = jax.random.split(key)
    features, kpis = ref.forward(subkey, raw_stream, schedule)
    
    print(f"  Feature stability: {kpis['feature_stability']:.3f}")
    print(f"  Energy per feature: {kpis['energy_per_feature_uj']:.3f} µJ")
    return key


def main():
    """Run all algorithm demos."""
    print("=" * 60)
    print("10 Thermal Algorithms for Operational Security")
    print("=" * 60)
    
    key = jax.random.key(42)
    
    key = demo_srsl(key)
    key = demo_taps(key)
    key = demo_bpp(key)
    key = demo_efsm(key)
    key = demo_tbro(key)
    key = demo_labi(key)
    key = demo_tcf(key)
    key = demo_ppts(key)
    key = demo_tvs(key)
    key = demo_ref(key)
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
