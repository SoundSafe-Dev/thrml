# Thermal Algorithms Guide

Complete guide to the 10 thermodynamic algorithms implemented for SoundSafe integration with Extropic hardware.

## Overview

Ten thermal algorithms leverage THRML's blocked Gibbs sampling and discrete EBM utilities to process audio signals, detect threats, schedule resources, and fuse multimodal sensor data—all designed to minimize energy consumption (Joules per token/alert) while maximizing throughput on Extropic's thermodynamic compute chips.

## Algorithm Catalog

### 1. SRSL - Stochastic Resonance Signal Lifter

**Purpose**: Amplify weak signals buried in noise using optimal temperature (β = 1/T) selection.

**SoundSafe Application**: Deepfake voice detection (pre-processing). Lifts weak spoof artifacts without broadband gain.

**Key Features**:
- Sweeps β values to maximize mutual information I(X;Y)
- Optimal β* maximizes information transfer per unit energy
- No broadband amplification; only resonant frequencies enhanced

**Usage**:
```python
from thrml.algorithms import StochasticResonanceSignalLifter
import jax

key = jax.random.key(42)
srsl = StochasticResonanceSignalLifter(signal_window_size=32, key=key)
weak_features = ...  # Your audio features
amplified, kpis = srsl.forward(key, weak_features, schedule)
print(f"Optimal beta: {kpis['optimal_beta']}")
print(f"Mutual information: {kpis['mutual_information']}")
print(f"Signal gain: {kpis['signal_gain']:.2f}x")
```

**KPIs**:
- `mutual_information`: Information transfer (0-∞)
- `optimal_beta`: Discovered β = 1/T maximizing I(X;Y)
- `signal_gain`: Amplification factor
- `energy_per_event_joules`: Energy per processing event

**Extropic Silicon**: Chip's Gibbs dynamics at β* = 1/T* maximize I(X;Y) per Joule. Weak patterns amplify naturally via thermalization.

---

### 2. TAPS - Thermodynamic Active Perception & Scheduling

**Purpose**: Energy-aware sensor activation and bitrate scheduling based on threat scores.

**SoundSafe Application**: Weapon/Aggression/Loitering/Zone detection. Schedules sensors and bitrates to minimize Joules while maintaining coverage.

**Key Features**:
- Hamiltonian encodes sensor activation costs
- Samples optimal sensor×bitrate matrix from energy distribution
- Adapts to real-time threat scores

**Usage**:
```python
from thrml.algorithms import ThermodynamicActivePerceptionScheduling

taps = ThermodynamicActivePerceptionScheduling(n_sensors=8, n_bitrate_levels=3, key=key)
threat_scores = jnp.array([0.1, 0.3, 0.9, 0.2, ...])  # Per-sensor threat
actions, kpis = taps.forward(key, threat_scores, schedule)
print(f"Energy: {kpis['total_energy_joules_per_sec']:.3f} J/s")
print(f"Coverage: {kpis['threat_coverage']:.3f}")
```

**KPIs**:
- `total_energy_joules_per_sec`: Power consumption rate
- `threat_coverage`: Fraction of high-threat sensors monitored
- `active_sensors`: Number of active sensors

**Extropic Silicon**: Thermal scheduling minimizes Joules while maintaining coverage via low-energy sampling of sensor subsets.

---

### 3. BPP - Boltzmann Policy Planner

**Purpose**: Policy action selection with temperature-controlled risk tolerance, compatible with Rules of Engagement (ROE).

**SoundSafe Application**: Weapon/Aggression/Loitering/Zone - ROE-compatible escalation under thermal risk control.

**Key Features**:
- Actions encoded as EBM nodes
- Temperature parameter controls risk aversion
- Fuses playbook priors with live tactic scores

**Usage**:
```python
from thrml.algorithms import BoltzmannPolicyPlanner

bpp = BoltzmannPolicyPlanner(risk_temperature=1.0, key=key)
threat_level = 2  # Scale 0-4
tactic_scores = jnp.array([0.1, 0.3, 0.9, 0.6, 0.2])
action_probs, kpis = bpp.forward(key, threat_level, tactic_scores, schedule)
print(f"Selected: {kpis['selected_action']}")
print(f"Confidence: {kpis['action_confidence']:.3f}")
```

**KPIs**:
- `selected_action`: Chosen policy action
- `action_confidence`: Probability of selected action
- `time_to_escalation`: Estimated escalation delay

**Extropic Silicon**: Gibbs sampling over action space minimizes risky decisions while respecting ROE constraints.

---

### 4. EFSM - Energy-Fingerprinted Scene Memory

**Purpose**: Anomaly detection via per-site energy baselines learned from clean audio windows.

**SoundSafe Application**: Anomalous sound detection. Detects out-of-manifold audio via ΔE spikes.

**Key Features**:
- Learns baseline energy model from clean windows
- Computes ΔE = E(scene) - E(baseline)
- Higher ΔE indicates anomalies

**Usage**:
```python
from thrml.algorithms import EnergyFingerprintedSceneMemory

efsm = EnergyFingerprintedSceneMemory(n_features=64, adaptation_rate=0.01, key=key)
clean_windows = ...  # Training data: (n_windows, n_features)
efsm.fit_baseline(key, clean_windows, schedule)

anomaly_score, kpis = efsm.forward(key, scene_features, schedule)
print(f"Anomaly score: {anomaly_score:.2f}")
print(f"Energy delta: {kpis.get('energy_delta', 0):.2e} J")
```

**KPIs**:
- `anomaly_score`: Energy deviation from baseline (higher = more anomalous)
- `energy_delta`: ΔE in Joules
- `adaptation_rate`: Learning rate for baseline updates

**Extropic Silicon**: Chip thermalizes to baseline; anomalies push system to higher energy states, detected via ΔE monitoring.

---

### 5. TBRO - Thermal Bandit Resource Orchestrator

**Purpose**: Multi-armed bandit resource allocation with thermal exploration/exploitation.

**SoundSafe Application**: Routes compute resources to high-risk zones (weapon/aggression detection).

**Key Features**:
- Sites encoded as bandit arms with reward priors
- Temperature controls exploration vs exploitation
- Maximizes SLO compliance while minimizing GPU-hours

**Usage**:
```python
from thrml.algorithms import ThermalBanditResourceOrchestrator

tbro = ThermalBanditResourceOrchestrator(n_sites=10, exploration_temperature=1.0, key=key)
risk_scores = jnp.array([0.2, 0.8, 0.1, 0.95, ...])  # Per-site risk
allocation, kpis = tbro.forward(key, risk_scores, schedule)
print(f"SLO compliance: {kpis['slo_compliance']:.3f}")
print(f"GPU-hours saved: {kpis['gpu_hours_saved_percent']:.1f}%")
```

**KPIs**:
- `slo_compliance`: Service level objective compliance rate
- `gpu_hours_saved_percent`: Efficiency improvement
- `overflow_drops`: Overflow packet rate

**Extropic Silicon**: Bandit learns optimal allocation via thermal sampling, routing compute to highest-value sites.

---

### 6. LABI - Landauer-Aware Bayesian Inference

**Purpose**: Energy-aware Bayesian updates that skip expensive computations when entropy reduction doesn't justify the Landauer cost.

**SoundSafe Application**: Gates expensive inference updates in anomalous sound detection.

**Key Features**:
- Computes Landauer cost: kT ln 2 per bit erased
- Skips updates when ΔH (entropy reduction) < threshold
- Only pays energy when information gain warrants it

**Usage**:
```python
from thrml.algorithms import LandauerAwareBayesianInference

labi = LandauerAwareBayesianInference(n_variables=16, energy_threshold=1e-18, key=key)
likelihood_deltas = jnp.array([0.1, -0.2, 0.05, ...])
(updated_state, posterior), kpis = labi.forward(key, likelihood_deltas, schedule)
print(f"Decision: {'UPDATE' if kpis['should_update'] else 'SKIP'}")
print(f"Landauer cost: {kpis['landauer_cost_joules']:.2e} J")
print(f"Skip rate: {kpis['skip_rate']:.3f}")
```

**KPIs**:
- `should_update`: Boolean decision (update vs skip)
- `landauer_cost_joules`: Energy cost in Joules
- `skip_rate`: Fraction of skipped updates (efficiency metric)
- `entropy_delta_bits`: Entropy change in bits

**Extropic Silicon**: Skips thermal transitions when ΔE < threshold → fewer Joules per token while maintaining detection accuracy.

---

### 7. TCF - Thermodynamic Causal Fusion

**Purpose**: Multimodal sensor fusion with causal graph discovery via small perturbations (do-interventions).

**SoundSafe Application**: Environmental + Access Sensors - Robust fusion despite modality failures.

**Key Features**:
- Discovers causal structure via perturbation-observation
- Robust causal edges survive sensor failures
- Fuses audio/video/doors/temperature signals

**Usage**:
```python
from thrml.algorithms import ThermodynamicCausalFusion

tcf = ThermodynamicCausalFusion(n_modalities=4, key=key)
modality_scores = jnp.array([0.8, 0.3, 0.9, 0.1])  # A/V/Doors/Temp

# Discover causal structure
graph, kpis_discover = tcf.discover_causal_structure(key, modality_scores, schedule)
print(f"Causal edges: {kpis_discover['n_causal_edges']}")

# Fuse with discovered structure
fused_score, kpis_fuse = tcf.forward(key, modality_scores, schedule)
print(f"Fused threat: {kpis_fuse['threat_score']:.3f}")
print(f"Robustness Δ: {kpis_fuse['robustness_delta']:.3f}")
```

**KPIs**:
- `n_causal_edges`: Number of discovered causal connections
- `threat_score`: Fused threat score (0-1)
- `robustness_delta`: Stability under perturbations
- `n_failed_modalities`: Count of failed sensors

**Extropic Silicon**: Do-interventions shift equilibrium; causal edges remain stable under perturbations, reducing false positives and re-computes.

---

### 8. PPTS - Probabilistic Phase & Time Sync

**Purpose**: Low-overhead probabilistic synchronization of sensor phases/clocks using coupled oscillators.

**SoundSafe Application**: Environmental + Access Sensors - Probabilistic time synchronization.

**Key Features**:
- Coupled oscillator model for phase synchronization
- Minimal protocol overhead
- Triangulation-capable for location inference

**Usage**:
```python
from thrml.algorithms import ProbabilisticPhaseTimeSync

ppts = ProbabilisticPhaseTimeSync(n_sensors=6, coupling_strength=0.8, key=key)
observed_phases = jax.random.uniform(key, (6,), minval=0, maxval=2*jnp.pi)
synced_phases, kpis = ppts.forward(key, observed_phases, schedule)
print(f"Sync error: {kpis['sync_error_ms']:.2f} ms")
print(f"Triangulation error: {kpis['triangulation_error_m']:.4f} m")
```

**KPIs**:
- `sync_error_ms`: Synchronization error in milliseconds
- `triangulation_error_m`: Location triangulation error in meters

**Extropic Silicon**: Low-overhead clocking via thermal phase coupling; synchronized phases emerge naturally.

---

### 9. TVS - Thermo-Verifiable Sensing

**Purpose**: Hardware-verifiable watermarking using thermal RNG for provenance and content protection.

**SoundSafe Application**: Audio watermarking & content protection - Court-admissible provenance.

**Key Features**:
- Embeds nonce watermarks using thermal RNG
- Verification is cheap (correlation check)
- Provenance survives transit

**Usage**:
```python
from thrml.algorithms import ThermoVerifiableSensing

tvs = ThermoVerifiableSensing(nonce_size=32, watermark_strength=0.1, key=key)
stream = ...  # (n_frames, n_features) audio stream

# Watermark
watermarked, wm_kpis = tvs.forward(key, stream, mode="watermark", schedule)
print(f"Bitrate overhead: {wm_kpis['bitrate_overhead_percent']:.4f}%")

# Verify
is_valid, v_kpis = tvs.verify_stream(watermarked)
print(f"Verified: {v_kpis['verified']}")
print(f"Correlation: {v_kpis['correlation']:.3f}")
```

**KPIs**:
- `verified`: Boolean verification result
- `correlation`: Watermark correlation strength (0-1)
- `bitrate_overhead_percent`: Encoding overhead

**Extropic Silicon**: Thermal RNG embedded in Gibbs sampling → hardware verifiable watermarks with negligible overhead.

---

### 10. REF - Reservoir-EBM Front-End

**Purpose**: Ultra-low-power feature extraction using analog reservoir computing with EBM prior regularization.

**SoundSafe Application**: Low-power feature extraction for weapon/aggression detection.

**Key Features**:
- Analog reservoir for initial feature projection
- EBM prior stabilizes features
- Extremely low energy per feature

**Usage**:
```python
from thrml.algorithms import ReservoirEBMFrontEnd

ref = ReservoirEBMFrontEnd(reservoir_size=64, feature_size=24, key=key)
raw_input = ...  # (n_timesteps, input_dim)
features, kpis = ref.forward(key, raw_input, schedule)
print(f"Feature stability: {kpis['feature_stability']:.3f}")
print(f"Energy/feature: {kpis['energy_per_feature_uj']:.3e} µJ")
```

**KPIs**:
- `feature_stability`: Feature variance (lower = more stable)
- `energy_per_feature_uj`: Energy cost per feature in microjoules

**Extropic Silicon**: Reservoir + EBM prior → stable features at ultra-low cost; front-end for detection stacks.

---

## Common Patterns

### KPI Tracking

All algorithms inherit from `ThermalAlgorithm` and return KPIs:

```python
output, kpis = algorithm.forward(key, inputs, schedule)
# kpis is a dict[str, float] with algorithm-specific metrics
```

### Energy Measurement

All algorithms support energy accounting:

```python
from thrml.algorithms.energy import EnergyConfig, estimate_sampling_energy_j, summarize_energy

cfg = EnergyConfig()
sampling_j = estimate_sampling_energy_j(n_samples=100, n_nodes=32, config=cfg)
tokens = 32 * 8
summary = summarize_energy(tokens=tokens, sampling_j=sampling_j, alerts=1, config=cfg)
# summary contains: joules_per_token, joules_per_alert, etc.
```

### Power Measurement (macOS)

With `--measure-power` flag:

```python
from thrml.algorithms.power import PowerSession

ps = PowerSession()
ps.start()
# ... run algorithm ...
joules = ps.stop()  # Real measured energy
```

See `thrml/algorithms/energy.py` and `thrml/algorithms/power.py` for details.

---

## SoundSafe Integration

All algorithms map to SoundSafe capabilities:

- **Deepfake Voice Detection**: TVS + SRSL + LABI
- **Audio Watermarking**: TVS
- **Anomalous Sound Detection**: EFSM + SRSL + LABI
- **Weapon/Aggression/Loitering/Zone**: REF + TAPS + TBRO + BPP
- **Environmental + Access Sensors**: PPTS + TCF
- **Multimodal Event Fusion**: TCF

See `docs/SOUNDSAFE_MAPPING.md` for detailed mappings.

---

## Discovery Framework

Use `examples/run_discovery.py` to find optimal operating points:

- **SRSL**: Sweep β × SNR → find β* maximizing I(X;Y)
- **LABI**: Sweep threshold × likelihood scale → find skip/update frontier
- **TCF**: Sweep perturbation strength → find stable causal edges

Results saved to `results/discovery/{algorithm}/` as CSV/JSON + plots.

---

## Visualization

Use `examples/run_prototypes.py` for annotated plots:

```bash
python examples/run_prototypes.py --seed 42 --measure-power
```

Generates BEFORE/AFTER comparisons, KPI annotations, and SoundSafe capability labels.

---

## Extropic Hardware Mapping

All algorithms map naturally to Extropic's thermodynamic compute chips:

- **Blocked Gibbs** → Hardware sampler relaxes to equilibrium
- **Energy functions** → Encoded in chip weights
- **Temperature (β)** → Direct hardware control
- **Sampling schedule** → Native hardware operation

Discovery sweeps identify optimal β* and operating frontiers today; these values are pinned during hardware port for maximum energy efficiency.

See `docs/THERMAL_OVERVIEW.md` for detailed hardware mappings per algorithm.

