# Quick Reference

Fast lookup for common tasks, commands, and patterns.

## Algorithm Imports

```python
from thrml.algorithms import (
    StochasticResonanceSignalLifter,      # SRSL
    ThermodynamicActivePerceptionScheduling,  # TAPS
    BoltzmannPolicyPlanner,                # BPP
    EnergyFingerprintedSceneMemory,        # EFSM
    ThermalBanditResourceOrchestrator,    # TBRO
    LandauerAwareBayesianInference,        # LABI
    ThermodynamicCausalFusion,            # TCF
    ProbabilisticPhaseTimeSync,           # PPTS
    ThermoVerifiableSensing,              # TVS
    ReservoirEBMFrontEnd,                 # REF
)
```

## Basic Usage Pattern

```python
import jax
from thrml import SamplingSchedule
from thrml.algorithms import YourAlgorithm

# Initialize
key = jax.random.key(42)
algo = YourAlgorithm(..., key=key)

# Prepare input
inputs = ...  # JAX array

# Run
schedule = SamplingSchedule(n_warmup=10, n_samples=30, steps_per_sample=1)
output, kpis = algo.forward(key, inputs, schedule)

# Use results
print(f"Output: {output}")
print(f"KPIs: {kpis}")
```

## Common KPIs

All algorithms return `kpis` dict with energy metrics:

- `energy_total_joules` - Total energy consumed
- `energy_joules_per_token` - Energy efficiency per token
- `energy_joules_per_alert` - Energy efficiency per alert (if applicable)
- Algorithm-specific KPIs (see THERMAL_ALGORITHMS_GUIDE.md)

## Sampling Schedules

```python
from thrml import SamplingSchedule

# Fast (prototypes)
fast = SamplingSchedule(n_warmup=10, n_samples=30, steps_per_sample=1)

# Production
prod = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

# Discovery
disc = SamplingSchedule(n_warmup=50, n_samples=500, steps_per_sample=1)
```

## Discovery Commands

```bash
# Run all discovery sweeps
python examples/run_discovery.py --seed 42 --measure-power

# Results in results/discovery/
# - srsl/srsl_sweep.csv + srsl_mi_heatmap.png
# - labi/labi_frontier.csv + labi_skip_rate.png
# - tcf/tcf_perturb.json + tcf_edges.png
```

## Prototype Commands

```bash
# Run all prototypes with visuals
python examples/run_prototypes.py --seed 42 --measure-power

# Results in results/prototypes_YYYYMMDD_HHMMSS/
# - One PNG per algorithm + core demos
# - summary.txt with all KPIs
```

## Report Generation

```bash
# Generate HTML report
python examples/build_report.py --results results --output results/report.html
```

## Testing

```bash
# All tests
pytest -q

# Algorithm tests only
pytest tests/test_thermal_smoke.py -v

# Specific algorithm
pytest tests/test_thermal_smoke.py::test_srsl_smoke -v
```

## SoundSafe Mappings

| Algorithm | SoundSafe Capability |
|-----------|---------------------|
| SRSL + TVS + LABI | Deepfake Voice Detection |
| TVS | Audio Watermarking |
| EFSM + SRSL + LABI | Anomalous Sound Detection |
| REF + TAPS + TBRO + BPP | Weapon/Aggression/Loitering/Zone |
| PPTS + TCF | Environmental + Access Sensors |
| TCF | Multimodal Event Fusion |

## Energy Measurement

```python
from thrml.algorithms.energy import EnergyConfig, estimate_sampling_energy_j, summarize_energy

cfg = EnergyConfig()
sampling_j = estimate_sampling_energy_j(n_samples=100, n_nodes=32, config=cfg)
summary = summarize_energy(tokens=256, sampling_j=sampling_j, alerts=1, config=cfg)
print(f"Joules/token: {summary['joules_per_token']:.2e}")
```

## Power Measurement (macOS)

```python
from thrml.algorithms.power import PowerSession

ps = PowerSession()
ps.start()
# ... run algorithm ...
real_joules = ps.stop()
```

## Troubleshooting

### Import Errors
```bash
pip install -e ".[development,testing,examples]"
```

### JAX Backend
```python
import jax
print(jax.default_backend())  # Should be 'cpu' or 'gpu'
```

### Shape Errors
- Check input shapes match algorithm expectations
- Use `jnp.shape()` to inspect
- See algorithm docstrings for expected shapes

### NaN/Inf
```python
assert jnp.all(jnp.isfinite(output))
```

## File Locations

- Algorithms: `thrml/algorithms/*.py`
- Examples: `examples/run_*.py`
- Tests: `tests/test_*.py`
- Docs: `docs/*.md`
- Results: `results/`

## Common Patterns

### Discovery → Production

```python
# 1. Run discovery
# python examples/run_discovery.py --seed 42

# 2. Load optimal params
from examples.run_prototypes import load_optimal_params
optimal = load_optimal_params(Path("results/discovery"))

# 3. Use in algorithm
algo = YourAlgorithm(beta=optimal['srsl_beta'], ...)
```

### Batch Processing

```python
def process_batch(algorithm, windows, key):
    outputs = []
    kpis_list = []
    for window in windows:
        output, kpis = algorithm.forward(key, window, schedule)
        outputs.append(output)
        kpis_list.append(kpis)
    return jnp.stack(outputs), kpis_list
```

### Error Handling

```python
try:
    output, kpis = algo.forward(key, inputs, schedule)
    assert jnp.all(jnp.isfinite(output))
except ValueError as e:
    logger.error(f"Error: {e}")
    # Fallback
```

