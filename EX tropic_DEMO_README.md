# Extropic Team Demo: Complete Reproduction Guide

This guide enables the Extropic team to reproduce all tests, benchmarks, and demonstrations of the thermal algorithms built on THRML.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[development,testing,examples]"

# 2. Run full demo
python examples/extropic_full_demo.py --all

# 3. Review results
open results/extropic_demo/DEMO_REPORT.md
```

## What's Included

### 1. All 10 Thermal Algorithms
- **SRSL**: Stochastic Resonance Signal Lifter
- **TAPS**: Thermodynamic Active Perception Scheduling
- **BPP**: Boltzmann Policy Planner
- **EFSM**: Energy-Fingerprinted Scene Memory
- **TBRO**: Thermal Bandit Resource Orchestrator
- **LABI**: Landauer-Aware Bayesian Inference
- **TCF**: Thermodynamic Causal Fusion
- **PPTS**: Probabilistic Phase Time Sync
- **TVS**: Thermo-Verifiable Sensing
- **REF**: Reservoir-EBM Front-End

Each algorithm runs with synthetic data and produces real KPIs.

### 2. Discovery Sweeps
- **SRSL**: Beta × SNR optimization (finds optimal β*)
- **LABI**: Threshold × Likelihood scale (finds skip/update frontier)
- **TCF**: Perturbation strength (finds stable causal edges)

### 3. Benchmark Comparisons
- GPU (baseline) vs Thermal Algorithms vs Extropic hardware
- Joules/token metrics
- Intelligence per watt calculations
- Throughput comparisons

### 4. Thermodynamic Generation Demo
- Conceptual demonstration of generation without GANs/FLOPs
- Audio and video pattern generation
- Comparison visualizations

## Detailed Reproduction Steps

### Step 1: Environment Setup

```bash
# Clone/navigate to repo
cd thrml

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e ".[development,testing,examples]"

# Verify installation
python -c "from thrml.algorithms import *; print('✓ Imports successful')"
```

### Step 2: Run Full Demo

```bash
# Complete demo (all components)
python examples/extropic_full_demo.py --all --seed 42

# Or run components separately:
python examples/extropic_full_demo.py --algorithms  # Just algorithms
python examples/extropic_full_demo.py --discovery   # Just discovery
python examples/extropic_full_demo.py --benchmark   # Just benchmark
```

### Step 3: Run Individual Components

#### Algorithms Only
```bash
python examples/extropic_full_demo.py --algorithms --output results/algo_demo
```

#### Discovery Sweeps
```bash
python examples/extropic_full_demo.py --discovery --output results/discovery_demo
```

#### Benchmark Comparison
```bash
python examples/benchmark_comparison.py --tokens 10000
python examples/generate_comprehensive_report.py
```

#### Generation Demo
```bash
python examples/thermodynamic_generation_example.py
```

#### Prototype Visualizations
```bash
python examples/run_prototypes.py --seed 42
python examples/build_report.py --results results --output results/report.html
```

### Step 4: Run Discovery
```bash
python examples/run_discovery.py --seed 42 --measure-power
```

## Expected Outputs

### Directory Structure

```
results/extropic_demo/
├── DEMO_REPORT.md              # Comprehensive summary
├── algorithm_results.json       # All algorithm KPIs
├── discovery_results.json       # Discovery sweep data
├── benchmark_comparison/         # GPU vs Extropic comparison
│   ├── comparison_table.csv
│   ├── comparison_table.json
│   ├── comprehensive_report.html
│   └── *.png (visualizations)
└── generation/                  # Thermodynamic generation demo
    ├── thermodynamic_generation_demo.png
    └── generation_metrics.json
```

### Algorithm Results

Each algorithm returns KPIs including:
- Energy metrics (Joules/token, Joules/alert)
- Performance metrics (algorithm-specific)
- SoundSafe applicability
- Extropic hardware mapping

Example output:
```json
{
  "SRSL": {
    "mutual_information": 0.987,
    "optimal_beta": 1.582,
    "signal_gain": 2.34,
    "energy_per_event_joules": 1.6e-09
  },
  "LABI": {
    "skip_rate": 0.65,
    "landauer_cost_joules": 2.4e-25,
    "should_update": false
  },
  ...
}
```

### Discovery Results

Discovery sweeps identify optimal operating points:
- **SRSL**: Optimal β* maximizing mutual information
- **LABI**: Energy threshold for skip/update frontier
- **TCF**: Perturbation strength for stable causal edges

## Validation Checklist

### Algorithms
- [ ] All 10 algorithms run without errors
- [ ] KPIs are in expected ranges
- [ ] Energy metrics are reported
- [ ] Outputs match algorithm specifications

### Discovery
- [ ] SRSL finds optimal β in reasonable range (0.5-3.0)
- [ ] LABI shows skip rate frontier
- [ ] TCF shows stable edge counts

### Benchmarks
- [ ] Energy comparisons show 65%+ improvement (Thermal vs Baseline)
- [ ] Extropic projections show 99%+ improvement
- [ ] Intelligence/watt improvements are substantial

### Generation
- [ ] Audio patterns generated successfully
- [ ] Video frames show spatial coherence
- [ ] Comparisons show energy/latency advantages

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the repo root
pwd  # Should show .../thrml

# Reinstall
pip install -e ".[development,testing,examples]" --force-reinstall
```

### JAX Backend Issues
```bash
# Check JAX backend
python -c "import jax; print(jax.default_backend())"

# CPU-only is fine for demos
# GPU requires CUDA/ROCm setup
```

### Memory Issues
```bash
# Reduce sample counts
export JAX_PLATFORM_NAME=cpu
python examples/extropic_full_demo.py --all
```

### EFSM Issues
- EFSM has known broadcasting issues (documented)
- Demo handles this gracefully
- Full fix coming in evolution roadmap

## Performance Expectations

### Typical Run Times
- **All algorithms**: 10-30 seconds
- **Discovery sweeps**: 20-60 seconds
- **Benchmark comparison**: 5-10 seconds
- **Full demo**: 1-2 minutes total

### Hardware Requirements
- **CPU**: Any modern CPU (JAX works on CPU)
- **Memory**: 2-4 GB RAM minimum
- **GPU**: Optional (CPU-only works fine)
- **Storage**: ~100 MB for results

## Extropic Hardware Validation

### Current State
- All algorithms run on GPU/CPU (JAX)
- Discovery identifies optimal operating points
- Benchmarks project Extropic performance

### Hardware Readiness
- Algorithms designed for Extropic chips
- Energy functions map to chip weights
- Sampling schedules map to hardware cycles
- Temperature control maps to hardware registers

### Validation Steps
1. Run demos to establish baseline metrics
2. Map energy functions to chip weights
3. Validate sampling on hardware prototypes
4. Measure actual energy consumption
5. Compare to projections

## Additional Resources

### Documentation
- `docs/THERMAL_ALGORITHMS_GUIDE.md` - Complete algorithm guide
- `docs/INTEGRATION_GUIDE.md` - Integration instructions
- `docs/THERMODYNAMIC_GENERATION.md` - Generation concepts
- `docs/EVOLUTION_ROADMAP.md` - Future roadmap

### Examples
- `examples/run_prototypes.py` - Visualizations
- `examples/run_discovery.py` - Discovery sweeps
- `examples/benchmark_comparison.py` - Comparisons
- `examples/thermodynamic_generation_example.py` - Generation demo

### Tests
- `tests/test_thermal_smoke.py` - Algorithm smoke tests
- `pytest tests/test_thermal_smoke.py -v` - Run tests

## Contact & Support

For issues or questions:
1. Check this README
2. Review `docs/` documentation
3. Run tests: `pytest -v`
4. Check example outputs in `results/`

## Success Criteria

✅ **Demo is successful if:**
- All 10 algorithms run and produce KPIs
- Discovery sweeps identify reasonable operating points
- Benchmark comparisons show expected improvements
- No critical errors (warnings OK for known issues like EFSM)
- Output files are generated successfully

## Next Steps After Demo

1. **Review Results**: Check KPIs for expected ranges
2. **Validate Discovery**: Confirm optimal points are reasonable
3. **Hardware Mapping**: Map energy functions to chip weights
4. **Performance Validation**: Compare projections to hardware
5. **Integration Planning**: Plan production integration

---

**Last Updated**: 2025-10-29  
**Version**: v1.0  
**Status**: Ready for Extropic team reproduction and validation

