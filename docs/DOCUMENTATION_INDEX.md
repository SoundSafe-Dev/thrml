# Documentation Index

Complete index of all documentation for THRML thermal algorithms and extensions.

## Getting Started

1. **[README.md](../README.md)** - Quick start, installation, core THRML overview
2. **[THERMAL_OVERVIEW.md](THERMAL_OVERVIEW.md)** - High-level summary of thermodynamic extensions
3. **[SOUNDSAFE_MAPPING.md](SOUNDSAFE_MAPPING.md)** - Algorithm-to-capability mappings for SoundSafe

## Detailed Guides

### Algorithms

- **[THERMAL_ALGORITHMS_GUIDE.md](THERMAL_ALGORITHMS_GUIDE.md)** - Complete guide to all 10 thermal algorithms
  - Detailed API reference
  - Usage examples
  - KPI explanations
  - Extropic hardware mappings
  - Common patterns

### Testing

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive testing documentation
  - Test suite overview
  - Running tests
  - Writing new tests
  - Debugging
  - Coverage goals

### Integration

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Integration with SoundSafe and Extropic hardware
  - SoundSafe integration points
  - CoA/ROE adapter usage
  - Extropic hardware mapping
  - Data flow patterns
  - Deployment checklist

## Core THRML Documentation

### API Reference

- **[api/block_management.md](api/block_management.md)** - Block creation and management
- **[api/block_sampling.md](api/block_sampling.md)** - Blocked Gibbs sampling
- **[api/conditional_samplers.md](api/conditional_samplers.md)** - Conditional samplers (spin, categorical)
- **[api/factor.md](api/factor.md)** - Factor-based interactions
- **[api/interaction.md](api/interaction.md)** - Interaction utilities
- **[api/models/discrete_ebm.md](api/models/discrete_ebm.md)** - Discrete EBM models
- **[api/models/ebm.md](api/models/ebm.md)** - Energy-based models
- **[api/models/ising.md](api/models/ising.md)** - Ising model implementations
- **[api/observers.md](api/observers.md)** - Observer patterns
- **[api/pgm.md](api/pgm.md)** - Probabilistic graphical models

### Architecture

- **[architecture.md](architecture.md)** - THRML core architecture
- **[index.md](index.md)** - Main documentation index (MkDocs)

## Roadmap & Strategy

- **[EVOLUTION_ROADMAP.md](EVOLUTION_ROADMAP.md)** - Strategic roadmap for algorithm evolution and expansion
  - Phase 1: Enhanced algorithms (6 months)
  - Phase 2: New algorithm categories (6-12 months)
  - Phase 3: Integration & orchestration (12-18 months)
  - Phase 4: Advanced capabilities (18-24 months)
- **[THERMODYNAMIC_GENERATION.md](THERMODYNAMIC_GENERATION.md)** - Revolutionary synthetic data generation via Gibbs sampling
  - Why it's incredible: 100-1000x energy savings vs GANs
  - No FLOPs needed: Native probabilistic hardware operations
  - Applications: Audio, video, sensor data generation
  - Integration with thermal algorithms

## Examples

All examples are in `examples/`:

- **[run_prototypes.py](../examples/run_prototypes.py)** - Prototype runs with annotated plots
- **[run_discovery.py](../examples/run_discovery.py)** - Thermodynamic discovery sweeps
- **[run_coa_integration.py](../examples/run_coa_integration.py)** - CoA/ROE integration demo

## Quick Reference

### 10 Thermal Algorithms

| Algorithm | File | SoundSafe Use Case | Key KPI |
|-----------|------|-------------------|---------|
| **SRSL** | `thrml/algorithms/srsl.py` | Deepfake voice detection | Mutual information |
| **TAPS** | `thrml/algorithms/taps.py` | Sensor scheduling | Energy (J/s) |
| **BPP** | `thrml/algorithms/bpp.py` | Policy planning | Time-to-escalation |
| **EFSM** | `thrml/algorithms/efsm.py` | Anomaly detection | Anomaly score |
| **TBRO** | `thrml/algorithms/tbro.py` | Resource routing | SLO compliance |
| **LABI** | `thrml/algorithms/labi.py` | Inference gating | Skip rate |
| **TCF** | `thrml/algorithms/tcf.py` | Multimodal fusion | Threat score |
| **PPTS** | `thrml/algorithms/ppts.py` | Time sync | Sync error (ms) |
| **TVS** | `thrml/algorithms/tvs.py` | Watermarking | Correlation |
| **REF** | `thrml/algorithms/ref.py` | Feature extraction | Feature stability |

### Common Commands

```bash
# Run prototypes
python examples/run_prototypes.py --seed 42

# Run discovery
python examples/run_discovery.py --seed 42 --measure-power

# Run tests
pytest tests/test_thermal_smoke.py -v

# Build report
python examples/build_report.py --results results --output results/report.html
```

### Key Modules

```
thrml/
├── algorithms/          # 10 thermal algorithms
│   ├── base.py         # Base classes, KPITracker
│   ├── energy.py       # Energy accounting utilities
│   ├── power.py        # Power measurement (macOS)
│   ├── synthetic_data.py # Data generators
│   └── coa_roe_adapter.py # CoA/ROE integration
├── coa_roe/            # CoA/ROE scenarios
└── models/             # Core THRML models

examples/
├── run_prototypes.py   # Prototype visualizations
├── run_discovery.py    # Discovery sweeps
├── run_coa_integration.py # CoA/ROE demo
└── build_report.py     # HTML report generator

tests/
├── test_thermal_smoke.py # Algorithm smoke tests
└── test_*.py           # Core THRML tests
```

## Documentation Structure

```
docs/
├── DOCUMENTATION_INDEX.md       # This file
├── THERMAL_OVERVIEW.md          # High-level overview
├── THERMAL_ALGORITHMS_GUIDE.md  # Complete algorithm guide
├── TESTING_GUIDE.md             # Testing documentation
├── INTEGRATION_GUIDE.md         # Integration guide
├── SOUNDSAFE_MAPPING.md         # Capability mappings
├── api/                         # Core THRML API docs
└── architecture.md              # Architecture docs
```

## Getting Help

1. Check the relevant guide above
2. Review example scripts in `examples/`
3. Run tests to see usage patterns: `pytest tests/test_thermal_smoke.py -v`
4. Check algorithm docstrings: `python -c "from thrml.algorithms import *; help(StochasticResonanceSignalLifter)"`

## Contributing

When adding new features:

1. Update relevant guide in `docs/`
2. Add tests to `tests/test_thermal_smoke.py` or appropriate test file
3. Add example usage in `examples/`
4. Update this index if adding new documentation

---

**Last Updated**: 2025-10-29  
**Version**: Thermal Algorithms v1.0

