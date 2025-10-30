# 10 Thermal Algorithms for Operational Security

This module implements 10 thermal algorithms designed for real operational wins (safety, cost, latency, energy) using THRML's Ising/EBM simulation stack. Each algorithm is designed to prototype on GPU today and experiment with future Extropic hardware.

## Overview

All algorithms inherit from `ThermalAlgorithm` and implement:
- `forward()`: Main inference/processing step
- `get_kpis()`: Return current KPI values
- Built-in KPI tracking via `KPITracker`

## Algorithms

### 1. Stochastic-Resonance Signal Lifter (SRSL)

**File:** `srsl.py`

Use controlled thermal noise to amplify sub-threshold patterns via stochastic resonance instead of filtering them out.

**Why it matters:** Pulls weak but meaningful signals above detector thresholds without cranking gain (less false alarm from broadband amplification).

**Key Features:**
- Bistable Ising unit coupled to audio features
- Temperature sweep to find info-optimal operating point
- Maximizes mutual information between weak and amplified signals

**KPIs:**
- Signal gain (std(amplified) / std(weak))
- Mutual information I(X;Y)
- Energy per uplifted detection (J/event)

**Usage:**
```python
from thrml.algorithms import StochasticResonanceSignalLifter
from thrml import SamplingSchedule

srsl = StochasticResonanceSignalLifter(signal_window_size=32)
weak_features = ...  # [n_features] array
amplified, kpis = srsl.forward(key, weak_features, schedule)
```

---

### 2. Thermodynamic Active Perception & Scheduling (TAPS)

**File:** `taps.py`

A Landauer-aware sensor scheduler that thermally samples which cameras/mics to wake, when, and at what bitrate to maximize threat-information per joule.

**Why it matters:** Cuts edge power 30-60% while maintaining (or improving) detection outcomes—huge for battery, PoE budgets, and GPU duty cycle.

**Key Features:**
- Formulate sensing as Ising bandit with energy costs in Hamiltonian
- Blocked Gibbs ≈ cheap planner
- Temperature tunes exploration vs exploitation

**KPIs:**
- Threat coverage
- Total energy (Joules/sec)
- Active sensors count
- Landauer cost

**Usage:**
```python
from thrml.algorithms import ThermodynamicActivePerceptionScheduling

taps = ThermodynamicActivePerceptionScheduling(n_sensors=8, n_bitrate_levels=3)
threat_scores = ...  # [n_sensors] array
actions, kpis = taps.forward(key, threat_scores, schedule)
```

---

### 3. Boltzmann Policy Planner (BPP)

**File:** `bpp.py`

Fuse detection and response policy in one energy model; temperature controls risk tolerance, sampling yields the next action (notify, lock, dispatch).

**Why it matters:** Faster, consistent escalations with calibrated risk; reduces operator load and decision lag.

**Key Features:**
- Energy-based decision layer (not just detection)
- Priors from playbooks
- Temperature-controlled risk tolerance

**KPIs:**
- Time-to-escalation (TTE)
- Intervention precision/recall
- Action confidence

**Usage:**
```python
from thrml.algorithms import BoltzmannPolicyPlanner

bpp = BoltzmannPolicyPlanner(action_names=["notify", "lock", "dispatch"])
threat_level = 2  # 0=low, 1=medium, 2=high
tactic_scores = ...  # [n_tactics] array
action_probs, kpis = bpp.forward(key, threat_level, tactic_scores, schedule)
```

---

### 4. Energy-Fingerprinted Scene Memory (EFSM)

**File:** `efsm.py`

Learn a thermal energy fingerprint of each camera's normal environment; flag anomalies as out-of-manifold energy spikes.

**Why it matters:** Zero-shot perimeter/scene change detection (blocked exits, staged objects) without per-site labeling.

**Key Features:**
- Site-specific EBM baselines that self-calibrate
- Memory is an energy well, not a CNN embedding
- Online adaptation

**KPIs:**
- Anomaly score (0-1)
- Energy delta from baseline
- Drift stability

**Usage:**
```python
from thrml.algorithms import EnergyFingerprintedSceneMemory

efsm = EnergyFingerprintedSceneMemory(n_features=64)
# Fit on clean windows
efsm.fit_baseline(key, clean_windows)
# Detect anomalies
anomaly_score, kpis = efsm.forward(key, scene_features, schedule)
```

---

### 5. Thermal Bandit Resource Orchestrator (TBRO)

**File:** `tbro.py`

Real-time multi-armed bandit implemented as thermal sampling to route GPU/bitrate where risk is likely to spike.

**Why it matters:** Keeps SLA during surges (stadiums, campuses) without over-provisioning across all sites.

**Key Features:**
- Bandit arms as spins with reward priors
- Temperature tunes exploration vs exploitation
- On-device allocation each minute

**KPIs:**
- SLO compliance (%)
- GPU-hours saved (%)
- Overflow drops

**Usage:**
```python
from thrml.algorithms import ThermalBanditResourceOrchestrator

tbro = ThermalBanditResourceOrchestrator(n_sites=10)
site_risk_scores = ...  # [n_sites] array
allocation, kpis = tbro.forward(key, site_risk_scores, schedule)
# Update rewards after observation
tbro.update_rewards(site_idx=3, reward=0.8)
```

---

### 6. Landauer-Aware Bayesian Inference (LABI)

**File:** `labi.py`

A Bayesian update rule that pays kT ln 2 per bit only when posterior entropy drops enough—energy-optimal inference at the edge.

**Why it matters:** Extends battery life and reduces heat while preserving decision quality.

**Key Features:**
- Energy in the objective: maximize ΔInformation / Joule
- Skip or cache when not worth thermodynamic cost
- Posterior entropy as spin-entropy proxy

**KPIs:**
- Landauer cost (Joules)
- Skip rate (%)
- Entropy delta (bits)

**Usage:**
```python
from thrml.algorithms import LandauerAwareBayesianInference

labi = LandauerAwareBayesianInference(n_variables=16, energy_threshold=1e-18)
likelihood_scores = ...  # [n_variables] array
(should_update, samples), kpis = labi.forward(key, likelihood_scores, schedule)
```

---

### 7. Thermodynamic Causal Fusion (TCF)

**File:** `tcf.py`

Discover causal edges (not just correlations) among audio/video/sensors by small energy perturbations and observing equilibrium shifts.

**Why it matters:** More robust fusion graphs that don't break under domain shift (e.g., fog kills video but boosts acoustic cues).

**Key Features:**
- Perturb-and-observe in EBM to estimate do-effects
- Use for structural learning of fusion topology
- Robust to modality failure

**KPIs:**
- Number of causal edges
- Causal strength
- Robustness delta under modality failure

**Usage:**
```python
from thrml.algorithms import ThermodynamicCausalFusion

tcf = ThermodynamicCausalFusion(n_modalities=4)
# Discover causal structure
causal_graph, kpis = tcf.discover_causal_structure(key, modality_readings, schedule)
# Fuse with causal awareness
fused_score, kpis = tcf.forward(key, modality_readings, failed_modalities, schedule)
```

---

### 8. Probabilistic Phase & Time Sync (PPTS)

**File:** `ppts.py`

Use coupled thermal oscillators to auto-synchronize multi-camera/mic feeds (phase/time) without GPS/NTP—probabilistic clocking.

**Why it matters:** Cleaner cross-view tracking and A/V triangulation when network time is messy; fewer false tracks.

**Key Features:**
- Kuramoto-style thermal coupling for timebase alignment
- Solved by sampling rather than packets
- Phase estimation via spin patterns

**KPIs:**
- Sync error (ms) at 95th percentile
- Triangulation error (m)
- Average phase difference

**Usage:**
```python
from thrml.algorithms import ProbabilisticPhaseTimeSync

ppts = ProbabilisticPhaseTimeSync(n_sensors=6, coupling_strength=0.8)
observed_phases = ...  # [n_sensors] array (radians)
synced_phases, kpis = ppts.forward(key, observed_phases, schedule)
```

---

### 9. Thermo-Verifiable Sensing (TVS)

**File:** `tvs.py`

Embed entropy tags (thermo-generated one-time fingerprints) into sensor streams to prove live-ness & provenance against replay/deepfakes.

**Why it matters:** Trustworthy feeds for courts/ops; blocks synthetic insertion attacks cheaply.

**Key Features:**
- On-device thermal RNG drives watermarking/nonce scheduling
- Verification is energy-light
- Nonce history tracking

**KPIs:**
- Bitrate overhead (%)
- Verification correlation
- Attack success rate ↓

**Usage:**
```python
from thrml.algorithms import ThermoVerifiableSensing

tvs = ThermoVerifiableSensing(nonce_size=32)
# Watermark stream
watermarked, kpis = tvs.forward(key, stream_data, mode="watermark", schedule)
# Verify watermark
is_valid, kpis = tvs.forward(key, watermarked, mode="verify")
```

---

### 10. Reservoir-EBM Front-End (REF)

**File:** `ref.py`

Low-power analog reservoir converts raw streams into rich states; a tiny Ising prior regularizes readout for stable features under noise.

**Why it matters:** Cuts DSP/DNN flops while improving stability in wind, glare, echo.

**Key Features:**
- Hybrid analog (reservoir) + thermodynamic prior (EBM)
- Replacing heavy feature towers
- Smoothness regularization via Ising coupling

**KPIs:**
- Feature stability
- Energy per feature (µJ)
- Feature entropy

**Usage:**
```python
from thrml.algorithms import ReservoirEBMFrontEnd

ref = ReservoirEBMFrontEnd(reservoir_size=100, feature_size=32)
raw_stream = ...  # [n_timesteps, n_inputs] array
features, kpis = ref.forward(key, raw_stream, schedule)
```

---

## Base Utilities

### `KPITracker`

Tracks metrics over time with timestamps:
```python
tracker = KPITracker()
tracker.record("metric_name", value, timestamp=time.time())
mean_val = tracker.get_mean("metric_name")
latest_val = tracker.get_latest("metric_name")
```

### `ThermalAlgorithm`

Base class for all algorithms:
```python
class MyAlgorithm(ThermalAlgorithm):
    def forward(self, key, data, schedule=None):
        # Implementation
        return result, kpis
    
    def get_kpis(self):
        return self.kpi_tracker.get_mean(...)
```

### Utility Functions

- `mutual_information(x, y)`: Estimate I(X;Y)
- `entropy_from_samples(samples)`: Estimate H(X)
- `landauer_energy(entropy_delta, temperature_kelvin)`: Compute E = kT ln(2) * ΔH
- `beta_from_temperature(temperature)`: Convert temp to beta

## Running the Demos

See `examples/demo_thermal_algorithms.py` for complete examples of all 10 algorithms.

```bash
python examples/demo_thermal_algorithms.py
```

## Implementation Notes

1. **All algorithms use THRML's Ising/EBM infrastructure** - They're built on `IsingEBM`, `IsingSamplingProgram`, and block Gibbs sampling.

2. **GPU-accelerated** - JAX compilation enables efficient execution on GPUs.

3. **Extensible** - Each algorithm is designed to easily bind to future Extropic hardware thermal RNG / annealers.

4. **KPI-driven** - Built-in tracking makes it easy to measure operational wins.

5. **Fast prototyping** - All algorithms work with synthetic data today for experimentation.
