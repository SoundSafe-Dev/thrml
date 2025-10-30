# Testing Guide

Complete guide to testing the thermal algorithms and THRML extensions.

## Test Suite Overview

The test suite validates:
- Core THRML functionality (64 tests)
- Thermal algorithm smoke tests (10 tests)
- Integration and end-to-end workflows
- Performance and scaling characteristics

## Running Tests

### Full Test Suite

```bash
# Run all tests
pytest

# Quiet mode (summary only)
pytest -q

# With coverage
pytest --cov=thrml --cov-report=html
```

### Specific Test Categories

```bash
# Thermal algorithm smoke tests
pytest tests/test_thermal_smoke.py -v

# Core THRML tests
pytest tests/test_block_management.py tests/test_block_sampling.py -v

# Discrete EBM tests
pytest tests/test_discrete_ebm.py -v

# Ising model tests
pytest tests/test_ising.py -v
```

## Test Structure

### Thermal Algorithm Tests

`tests/test_thermal_smoke.py` contains smoke tests for all 10 algorithms:

- `test_srsl_smoke` - Stochastic Resonance Signal Lifter
- `test_taps_smoke` - Thermodynamic Active Perception Scheduling
- `test_bpp_smoke` - Boltzmann Policy Planner
- `test_efsm_smoke` - Energy-Fingerprinted Scene Memory (xfailed)
- `test_tbro_smoke` - Thermal Bandit Resource Orchestrator
- `test_labi_smoke` - Landauer-Aware Bayesian Inference
- `test_tcf_smoke` - Thermodynamic Causal Fusion
- `test_ppts_smoke` - Probabilistic Phase Time Sync
- `test_tvs_smoke` - Thermo-Verifiable Sensing
- `test_ref_smoke` - Reservoir-EBM Front-End

Each test:
1. Instantiates the algorithm
2. Generates synthetic input data
3. Runs a forward pass
4. Validates KPIs are returned
5. Checks for reasonable output ranges

### Core THRML Tests

Core functionality tests in `tests/`:
- **Block Management** (`test_block_management.py`): Block creation, shape validation, compatibility
- **Block Sampling** (`test_block_sampling.py`): Blocked Gibbs sampling, state management
- **Discrete EBM** (`test_discrete_ebm.py`): Factor-based EBMs, conditional samplers, heterogeneous models
- **Factor Interactions** (`test_factor.py`): Factor operations, energy computations
- **Ising Models** (`test_ising.py`): Ising EBM, KL divergence, gradients
- **PGM** (`test_pgm.py`): Probabilistic graph models, node types

## Known Test Issues

### EFSM Broadcasting (xfailed)

`test_efsm_smoke` is marked `xfail` due to JAX shape broadcasting issues in energy evaluation. The algorithm works for single-sample baseline fitting; batch evaluation requires shape alignment fixes.

**Status**: Known issue, documented, safe placeholder visualization in prototypes.

### Performance Scaling Test (xfail on macOS)

`test_discrete_ebm.py::TestBigGrid::test_big` is marked `xfail` on macOS due to timing variability in CPU-only JAX. The test validates quadratic scaling of compilation time; on macOS with varying CPU loads, timing can exceed the 1.1× threshold.

**Status**: Functional correctness validated; performance test is sensitive to environment.

## Integration Tests

### Example Script Validation

Validate example scripts compile and run:

```bash
# Check syntax
python -m py_compile examples/*.py

# Run discovery
python examples/run_discovery.py --seed 42 --results results/test_discovery

# Run prototypes
python examples/run_prototypes.py --seed 42 --results results/test_prototypes

# Build report
python examples/build_report.py --results results/test_prototypes --output results/test_report.html
```

### Module Import Tests

Validate all modules import correctly:

```bash
python -c "from thrml.algorithms import *; print('All imports successful')"
python -c "from thrml.algorithms.synthetic_data import run_all; print('Synthetic data OK')"
python -c "from thrml.algorithms.energy import EnergyConfig; print('Energy utils OK')"
```

## Performance Benchmarks

### Synthetic Data Benchmarks

The synthetic data generator (`thrml/algorithms/synthetic_data.py`) includes a benchmark harness:

```python
from thrml.algorithms.synthetic_data import run_all, BenchConfig

config = BenchConfig(n_iters=100, seed=42)
results = run_all(config)
# Returns per-algorithm KPI summaries
```

### Discovery Benchmarks

Discovery sweeps can be used for performance profiling:

```bash
python examples/run_discovery.py --seed 42 --measure-power
```

Results include energy measurements per algorithm.

## Continuous Integration

The test suite is designed for CI/CD:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest -q --tb=short

- name: Run algorithm tests
  run: pytest tests/test_thermal_smoke.py -v

- name: Validate examples
  run: |
    python examples/run_discovery.py --seed 42
    python examples/run_prototypes.py --seed 42
```

## Writing New Tests

### Algorithm Test Template

```python
import pytest
import jax
import jax.numpy as jnp
from thrml import SamplingSchedule
from thrml.algorithms import YourAlgorithm

def test_your_algorithm_smoke():
    key = jax.random.key(42)
    algo = YourAlgorithm(..., key=key)
    
    # Generate test input
    input_data = jax.random.normal(key, (32,))
    schedule = SamplingSchedule(n_warmup=10, n_samples=30, steps_per_sample=1)
    
    # Run forward pass
    output, kpis = algo.forward(key, input_data, schedule)
    
    # Validate output shape
    assert output.shape == expected_shape
    
    # Validate KPIs
    assert isinstance(kpis, dict)
    assert "key_metric" in kpis
    assert jnp.isfinite(kpis["key_metric"])
```

### Core Functionality Test Template

```python
import unittest
import jax
import jax.numpy as jnp
from thrml import ...

class TestYourFeature(unittest.TestCase):
    def test_basic_functionality(self):
        # Setup
        key = jax.random.key(42)
        # ... create test objects ...
        
        # Execute
        result = your_function(...)
        
        # Assert
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(jnp.all(jnp.isfinite(result)))
```

## Debugging Failed Tests

### Verbose Output

```bash
# Maximum verbosity
pytest -vv

# Print output (for print statements)
pytest -s

# Stop at first failure
pytest -x
```

### Specific Test

```bash
# Run specific test
pytest tests/test_thermal_smoke.py::test_srsl_smoke -v

# Run with pdb on failure
pytest --pdb tests/test_thermal_smoke.py::test_srsl_smoke
```

### JAX Debugging

Enable JAX checks for invalid operations:

```python
import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
```

## Coverage Goals

Target coverage:
- Core THRML: >90%
- Thermal algorithms: >80% (smoke tests cover main paths)
- Example scripts: Functional validation

Generate coverage report:

```bash
pytest --cov=thrml --cov-report=html --cov-report=term
open htmlcov/index.html  # macOS
```

## Test Data

Synthetic data generators in `thrml/algorithms/synthetic_data.py`:
- `gen_srsl_window()` - Weak signal + noise
- `gen_taps_inputs()` - Threat scores per sensor
- `gen_bpp_inputs()` - Threat level + tactic scores
- `gen_efsm_windows()` - Clean audio windows
- `gen_tbro_inputs()` - Risk scores per site
- `gen_labi_likelihood()` - Likelihood deltas
- `gen_tcf_modalities()` - Multimodal scores
- `gen_ppts_phases()` - Phase observations
- `gen_tvs_stream()` - Audio stream
- `gen_ref_stream()` - Raw input features

All generators use seeded randomness for reproducibility.

## Troubleshooting

### Import Errors

```bash
# Verify installation
pip install -e ".[development,testing,examples]"

# Check PYTHONPATH
python -c "import sys; print(sys.path)"
```

### JAX Backend Issues

```bash
# Check JAX backend
python -c "import jax; print(jax.default_backend())"

# CPU-only is fine for tests
# GPU requires CUDA/ROCm setup
```

### Memory Issues

For large tests, reduce sample counts:

```python
schedule = SamplingSchedule(n_warmup=10, n_samples=30, steps_per_sample=1)
```

---

## Summary

The test suite provides:
- ✅ Fast smoke tests for all algorithms
- ✅ Comprehensive core THRML validation
- ✅ Integration test workflows
- ✅ Performance benchmarks
- ✅ CI/CD ready

Run `pytest -q` for a quick health check of the entire system.

