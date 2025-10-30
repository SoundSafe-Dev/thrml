<h1 align='center'>THRML</h1>

THRML is a JAX library for building and sampling probabilistic graphical models, with a focus on efficient block Gibbs sampling and energy-based models. Extropic is developing hardware to make sampling from certain classes of discrete PGMs massively more energy‑efficient; THRML provides GPU‑accelerated tools for block sampling on sparse, heterogeneous graphs, making it a natural place to prototype today and experiment with future Extropic hardware.

Features include:

- Blocked Gibbs sampling for PGMs
- Arbitrary PyTree node states
- Support for heterogeneous graphical models
- Discrete EBM utilities
- Enables early experimentation with future Extropic hardware

From a technical point of view, the internal structure compiles factor-based interactions to a compact "global" state representation, minimising Python loops and maximising array-level parallelism in JAX.

## Thermal algorithms, discovery, and integrations (what we added)

This extension adds thermodynamic algorithms, discovery tooling, and integrations built on top of THRML’s blocked Gibbs and discrete EBM infrastructure:

- 10 thermal algorithms with KPI tracking under `thrml/algorithms/`:
  SRSL, TAPS, BPP, EFSM, TBRO, LABI, TCF, PPTS, TVS, REF.
- Core demos illustrating two‑color blocked Gibbs and a heterogeneous spin+categorical discrete EBM compiled to a compact global state.
- Prototype runner with annotated plots (`examples/run_prototypes.py`) – figures saved to `results/`.
- Discovery sweeps for temperature/noise/interventions (`examples/run_discovery.py`) – CSV/JSON + plots in `results/discovery/`.
- Optional CoA/ROE integration demo (`examples/run_coa_integration.py`) with a lightweight adapter and scenario matcher under `thrml/coa_roe/`.

Quickstart for the add‑ons:

```bash
# Prototypes (reduced sampling, annotated plots → results/prototypes_*)
python examples/run_prototypes.py --seed 42

# Thermodynamic discovery sweeps (CSV/JSON + plots → results/discovery)
python examples/run_discovery.py --seed 42

# CoA/ROE integration demo (adapts THRML KPIs into threat_event + context)
python examples/run_coa_integration.py --seed 42
```

See `docs/THERMAL_OVERVIEW.md` for a detailed summary of algorithms, KPIs, and discovery methodology.

**Complete Documentation**:
- [Documentation Index](docs/DOCUMENTATION_INDEX.md) - Complete documentation index
- [Quick Reference](docs/QUICK_REFERENCE.md) - Fast lookup for common tasks
- [Thermal Algorithms Guide](docs/THERMAL_ALGORITHMS_GUIDE.md) - Detailed API and usage for all 10 algorithms
- [Testing Guide](docs/TESTING_GUIDE.md) - Test suite documentation and best practices
- [Integration Guide](docs/INTEGRATION_GUIDE.md) - SoundSafe and Extropic hardware integration
- [SoundSafe Mapping](docs/SOUNDSAFE_MAPPING.md) - Algorithm-to-capability mappings

## Extropic Demo (one‑command run)
## Extropic Demo (one‑command run)

Reproduce the complete Extropic demo (10 algorithms, discovery, benchmarks, thermodynamic generation) and generate reports:

```bash
python examples/extropic_full_demo.py --all --seed 42

# Outputs under results/extropic_demo/
#  - DEMO_REPORT.md (summary)
#  - benchmark_comparison/comprehensive_report.html (interactive visuals)
#  - generation/ (thermodynamic generation demo)
```

Automated test suite:

```bash
./run_extropic_tests.sh
```

Git LFS large assets (videos, numpy, zips) are tracked in `.gitattributes`. If you see LFS push errors, enable LFS on your org/repo and run:

```bash
git lfs install && git lfs push origin main
```

## Visual overview (what you’ll see)

- Prototypes: Annotated before/after figures per algorithm with SoundSafe labels (SRSL, LABI, TCF, TAPS, etc.).
- Discovery: Heatmaps/curves for SRSL (MI vs β×SNR), LABI (skip‑rate vs threshold×scale), TCF (edges vs perturbation).
- Benchmarks: Energy (J/token), throughput (tokens/sec), intelligence per watt comparisons (GPU vs Thermal vs Extropic).
- Generation: Thermodynamic generation demo (no GANs/FLOPs) for audio/video patterns.

To generate visuals locally:

```bash
# Prototypes
python examples/run_prototypes.py --seed 42 --results results/visuals

# Discovery
python examples/run_discovery.py --seed 42 --results results/visuals/discovery

# Benchmarks + HTML
python examples/benchmark_comparison.py --tokens 20000 --output results/visuals/benchmark
python examples/generate_comprehensive_report.py --results results/visuals/benchmark/comparison_table.json --output results/visuals/benchmark
```

Key outputs:
- `results/visuals/prototypes_*/*.png`
- `results/visuals/discovery/*/*.png`
- `results/visuals/benchmark/comprehensive_report.html`

## Core equations (reference)

Gibbs distribution and EBMs:

```
p(x) = (1/Z(β)) · exp(−β·E(x)),   β = 1/T
E(s) = −Σ_i b_i s_i − Σ_(i,j) w_ij s_i s_j,  s_i ∈ {−1,+1}
```

Mutual information (SRSL objective):

```
I(X;Y) = Σ_{x,y} p(x,y) log ( p(x,y) / (p(x)p(y)) )
```

Landauer energy (LABI gating criterion):

```
E_Landauer = k_B T ln 2 · ΔH,   ΔH = H(prior) − H(posterior)
```

Thermal scheduling (TAPS) energy budget:

```
E_sense = Σ_{i,l} A_{i,l} · C_{i,l}    (subject to coverage/latency constraints)
```

Causal fusion (TCF) stability under interventions:

```
|| ∂E[X_j]/∂X_i || ≈ const  over  δ ∈ [δ_min, δ_max]
```

LaTeX report (optional): `results/visuals/latex/THERMAL_REPORT.tex` (embeds figures + equations).

## Installation

Requires Python 3.10+.

```bash
pip install thrml
```

## Documentation

Available at [docs.thrml.ai](https://docs.thrml.ai/en/latest/).


## Quick example

Sampling a small Ising chain with two-color block Gibbs:

```python
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i+1]) for i in range(4)]
biases = jnp.zeros((5,))
weights = jnp.ones((4,)) * 0.5
beta = jnp.array(1.0)
model = IsingEBM(nodes, edges, biases, weights, beta)

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
```

## Developing

To get started, you'll need to create a virtual environment and install the requirements:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package with development dependencies
pip install -e ".[development,testing,examples]"

# Install pre-commit hooks
pre-commit install
```

The pre-commit hooks will automatically run code formatting and linting tools (ruff, black, isort, pyright) on every commit to ensure consistent style.

If you want to skip pre-commit (for a WIP commit), you can use the `--no-verify` flag:

```bash
git commit --no-verify -m "Your commit message"
```
