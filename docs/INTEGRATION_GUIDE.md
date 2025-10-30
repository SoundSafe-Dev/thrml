# Integration Guide

Guide to integrating thermal algorithms into SoundSafe and connecting to Extropic hardware.

## Architecture Overview

```
┌─────────────────┐
│  SoundSafe      │
│  Audio/Video    │
│  Sensors        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Thermal        │──────▶│   THRML      │──────▶│  Extropic       │
│  Algorithms     │       │  Blocked     │       │  Thermodynamic  │
│  (10 algos)     │       │  Gibbs + EBM │       │  Compute Chip   │
└─────────────────┘      └──────────────┘      └─────────────────┘
         │
         ▼
┌─────────────────┐
│  KPIs + Energy  │
│  Metrics        │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  CoA/ROE        │
│  Integration    │
└─────────────────┘
```

## SoundSafe Integration

### Capability Mapping

Each algorithm maps to SoundSafe capabilities:

| Algorithm | SoundSafe Capability | Purpose |
|-----------|---------------------|---------|
| **SRSL** | Deepfake Voice Detection | Pre-processing to lift weak spoof artifacts |
| **TVS** | Audio Watermarking & Content Protection | Provenance, court-admissible watermarks |
| **EFSM** | Anomalous Sound Detection | Per-site energy baselines, ΔE spikes |
| **LABI** | Anomalous Sound Detection | Gate expensive inference updates |
| **REF** | Weapon/Aggression/Loitering/Zone | Low-power feature extraction |
| **TAPS** | Weapon/Aggression/Loitering/Zone | Sensor scheduling, energy-aware |
| **TBRO** | Weapon/Aggression/Loitering/Zone | Resource routing to high-risk zones |
| **BPP** | Weapon/Aggression/Loitering/Zone | ROE-compatible escalation |
| **PPTS** | Environmental + Access Sensors | Probabilistic time synchronization |
| **TCF** | Environmental + Access Sensors, Multimodal Fusion | Causal fusion, robust to failures |

See `docs/SOUNDSAFE_MAPPING.md` for detailed mappings.

### Integration Points

#### 1. Audio Pre-processing (SRSL)

```python
from thrml.algorithms import StochasticResonanceSignalLifter
import jax

# Initialize with discovered optimal beta
key = jax.random.key(sound_id)
srsl = StochasticResonanceSignalLifter(
    signal_window_size=window_size,
    beta_min=optimal_beta,  # From discovery
    beta_max=optimal_beta,
    n_beta_steps=1,
    key=key
)

# Process audio stream
amplified_features, kpis = srsl.forward(
    key,
    raw_audio_features,
    schedule=SamplingSchedule(n_warmup=10, n_samples=50, steps_per_sample=1)
)

# Use amplified_features for downstream detection
```

#### 2. Anomaly Detection (EFSM + LABI)

```python
from thrml.algorithms import (
    EnergyFingerprintedSceneMemory,
    LandauerAwareBayesianInference
)

# Per-site baseline learning
efsm = EnergyFingerprintedSceneMemory(n_features=64, key=site_key)
efsm.fit_baseline(key, clean_training_windows, schedule)

# Real-time anomaly detection
anomaly_score, efsm_kpis = efsm.forward(key, live_window, schedule)

# Energy-aware inference gating
labi = LandauerAwareBayesianInference(
    n_variables=16,
    energy_threshold=discovered_threshold,  # From discovery
    key=key
)
likelihood_deltas = compute_likelihood_deltas(anomaly_score, ...)
(posterior, state), labi_kpis = labi.forward(key, likelihood_deltas, schedule)

if labi_kpis['should_update']:
    # Expensive inference update is justified
    trigger_detection_pipeline(posterior)
```

#### 3. Sensor Scheduling (TAPS)

```python
from thrml.algorithms import ThermodynamicActivePerceptionScheduling

taps = ThermodynamicActivePerceptionScheduling(
    n_sensors=num_sensors,
    n_bitrate_levels=3,
    key=key
)

# Get threat scores from SoundSafe
threat_scores = get_threat_scores_per_sensor()  # From your detection system

# Sample optimal activation matrix
activation_matrix, taps_kpis = taps.forward(key, threat_scores, schedule)

# Activate sensors based on matrix
for sensor_id, bitrate_level in enumerate(get_optimal_bitrates(activation_matrix)):
    activate_sensor(sensor_id, bitrate_level)

# Monitor energy
energy_rate = taps_kpis['total_energy_joules_per_sec']
```

#### 4. Multimodal Fusion (TCF)

```python
from thrml.algorithms import ThermodynamicCausalFusion

tcf = ThermodynamicCausalFusion(n_modalities=4, key=key)

# Collect scores from multiple sensors
audio_score = get_audio_threat_score()
video_score = get_video_threat_score()
door_score = get_access_sensor_score()
temp_score = get_temperature_anomaly()

modality_scores = jnp.array([audio_score, video_score, door_score, temp_score])

# Discover causal structure (offline or periodic)
causal_graph, discovery_kpis = tcf.discover_causal_structure(
    key, modality_scores, schedule
)

# Fuse with discovered structure
fused_threat, fusion_kpis = tcf.forward(key, modality_scores, schedule)

if fused_threat > threshold:
    trigger_alert(fusion_kpis['threat_score'])
```

### CoA/ROE Integration

Use the adapter to convert THRML KPIs to threat events:

```python
from thrml.algorithms.coa_roe_adapter import build_threat_event, attach_thrml_context
from thrml.coa_roe.scenarios import ComprehensiveCaseScenarios

# After running algorithm
output, kpis = algorithm.forward(key, inputs, schedule)

# Build threat event
threat_event = build_threat_event(
    threat_type="weapon_detection",  # or "deepfake", "anomaly", etc.
    threat_level=kpis.get('threat_score', 0.0),
    kpis=kpis
)

# Attach THRML context
threat_event = attach_thrml_context(
    threat_event,
    algorithm_name="taps",
    algorithm_kpis=kpis,
    energy_joules=kpis.get('energy_total_joules', 0.0)
)

# Match to scenario
scenarios = ComprehensiveCaseScenarios()
matched = scenarios.match_scenario_pattern(threat_event)

if matched:
    action = scenarios.get_response_action(matched[0], threat_event)
    execute_coa_roe_action(action)
```

See `examples/run_coa_integration.py` for a complete example.

## Extropic Hardware Integration

### Hardware Abstraction

Extropic chips implement massively parallel Gibbs sampling:

- **Blocked Gibbs** → Native hardware operation
- **Energy functions** → Encoded in chip weights
- **Temperature (β)** → Hardware control register
- **Sampling schedule** → Native hardware cycles

### Discovery → Hardware Workflow

1. **Discovery Phase** (Current: GPU/JAX):
   ```bash
   python examples/run_discovery.py --seed 42 --measure-power
   ```
   - Identifies optimal β* for SRSL
   - Finds energy threshold frontiers for LABI
   - Discovers stable perturbation strengths for TCF
   - Results in `results/discovery/`

2. **Pinning Operating Points**:
   ```python
   # Load discovered values
   from examples.run_prototypes import load_optimal_params
   optimal_params = load_optimal_params(Path("results/discovery"))
   
   # Use in algorithm instantiation
   srsl = StochasticResonanceSignalLifter(
       beta_min=optimal_params['srsl_beta'],
       beta_max=optimal_params['srsl_beta'],
       ...
   )
   ```

3. **Hardware Port** (Future):
   - Map THRML sampling programs to hardware samplers
   - Convert energy functions to chip weights
   - Set temperature registers to pinned β* values
   - Same blocked Gibbs structure runs on chip

### Energy Measurement

#### Software (Current)

```python
from thrml.algorithms.energy import EnergyConfig, estimate_sampling_energy_j, summarize_energy

cfg = EnergyConfig()
sampling_j = estimate_sampling_energy_j(n_samples=100, n_nodes=32, config=cfg)
tokens = 256
summary = summarize_energy(tokens=tokens, sampling_j=sampling_j, alerts=1, config=cfg)
# summary['joules_per_token'], summary['joules_per_alert']
```

#### Hardware (Future)

On Extropic chips, energy measurement becomes:
- Direct power telemetry from chip
- Native energy counters per sampling operation
- Real-time Joules/token tracking

### Algorithm-Specific Hardware Mappings

#### SRSL
- **Energy Function**: Weak signal embedding + noise model
- **Temperature Control**: β* register set from discovery
- **Output**: Amplified signal (optimal I(X;Y) per Joule)

#### LABI
- **Energy Function**: Posterior update energy landscape
- **Threshold Register**: Landauer threshold (from discovery)
- **Output**: Update/Skip decision (energy-aware)

#### TCF
- **Energy Function**: Multimodal fusion graph
- **Perturbation Control**: Intervention strength (from discovery)
- **Output**: Causal graph + fused scores

See `docs/THERMAL_OVERVIEW.md` for complete hardware mappings.

## Data Flow

### Input Formats

All algorithms accept JAX arrays:

```python
import jax.numpy as jnp

# SRSL: (window_size,) float32
weak_features = jnp.array([...])

# TAPS: (n_sensors,) float32, threat scores 0-1
threat_scores = jnp.array([...])

# TCF: (n_modalities,) float32, modality scores 0-1
modality_scores = jnp.array([...])
```

### Output Formats

All algorithms return:
- `output`: Algorithm-specific output (array or dict)
- `kpis`: Dictionary of metrics

```python
output, kpis = algorithm.forward(key, inputs, schedule)

# kpis always includes energy metrics (if energy tracking enabled):
# - energy_total_joules
# - energy_joules_per_token
# - energy_joules_per_alert
```

## Performance Optimization

### Sampling Schedule Tuning

```python
from thrml import SamplingSchedule

# Fast (prototypes)
fast = SamplingSchedule(n_warmup=10, n_samples=30, steps_per_sample=1)

# Production
production = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

# Discovery (thorough)
discovery = SamplingSchedule(n_warmup=50, n_samples=500, steps_per_sample=1)
```

### Batch Processing

For streaming data, batch multiple windows:

```python
# Process batch of windows
windows = jnp.array([...])  # (batch_size, window_size, n_features)
for i in range(batch_size):
    output, kpis = algorithm.forward(key, windows[i], schedule)
```

### JIT Compilation

JAX JIT compiles automatically, but you can force compilation:

```python
from functools import partial
import jax

@partial(jax.jit, static_argnames=['schedule'])
def jitted_forward(key, inputs, schedule):
    return algorithm.forward(key, inputs, schedule)
```

## Error Handling

All algorithms handle:
- Invalid input shapes → Raises `ValueError`
- NaN/Inf in inputs → Propagates (check with `jnp.isfinite()`)
- Missing discovery params → Falls back to defaults

Example:

```python
try:
    output, kpis = algorithm.forward(key, inputs, schedule)
    assert jnp.all(jnp.isfinite(output))
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    # Fall back to baseline processing
```

## Monitoring and Logging

### KPI Logging

```python
import logging

logger = logging.getLogger("thrml")

output, kpis = algorithm.forward(key, inputs, schedule)

logger.info(f"Algorithm: {algorithm.__class__.__name__}")
logger.info(f"Energy: {kpis.get('energy_total_joules', 0):.2e} J")
logger.info(f"Joules/token: {kpis.get('energy_joules_per_token', 0):.2e}")

# SoundSafe-specific metrics
if 'threat_score' in kpis:
    logger.info(f"Threat score: {kpis['threat_score']:.3f}")
```

### Real-time Dashboards

KPIs can be exported to:
- Prometheus metrics
- JSON logs
- Time-series databases

Example export:

```python
import json

kpis_json = json.dumps({
    k: float(v) if isinstance(v, (jnp.ndarray, float)) else v
    for k, v in kpis.items()
})
# Send to monitoring system
```

## Deployment Checklist

- [ ] Run discovery sweeps to identify optimal parameters
- [ ] Pin operating points (β*, thresholds) from discovery
- [ ] Validate algorithms with production-like data
- [ ] Integrate with SoundSafe audio/video pipelines
- [ ] Connect CoA/ROE adapter for threat events
- [ ] Set up energy monitoring (Joules/token tracking)
- [ ] Configure sampling schedules for latency requirements
- [ ] Enable error handling and fallbacks
- [ ] Set up logging/monitoring dashboards
- [ ] Plan hardware port (map to Extropic chip samplers)

## Support

- Algorithm details: `docs/THERMAL_ALGORITHMS_GUIDE.md`
- SoundSafe mappings: `docs/SOUNDSAFE_MAPPING.md`
- Testing: `docs/TESTING_GUIDE.md`
- Examples: `examples/run_*.py`

