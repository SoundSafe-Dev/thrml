# Thermal Algorithms, Discovery, and Integrations

This document summarizes the thermodynamic extensions added to Extropic’s THRML repository, what they do, how to run them, and how to interpret outputs. The work preserves THRML’s core philosophy: compile factor-based interactions to compact global states and leverage JAX for array-level parallelism.

## What was added

- 10 thermodynamic algorithms with KPI tracking under `thrml/algorithms/`:
  - SRSL (Stochastic-Resonance Signal Lifter)
  - TAPS (Thermodynamic Active Perception & Scheduling)
  - BPP (Boltzmann Policy Planner)
  - EFSM (Energy-Fingerprinted Scene Memory)
  - TBRO (Thermal Bandit Resource Orchestrator)
  - LABI (Landauer-Aware Bayesian Inference)
  - TCF (Thermodynamic Causal Fusion)
  - PPTS (Probabilistic Phase & Time Sync)
  - TVS (Thermo-Verifiable Sensing)
  - REF (Reservoir-EBM Front-End)
- Core demos showing:
  - Two-color blocked Gibbs Ising (equal-sized blocks, batch_shape=()).
  - Heterogeneous discrete EBM (spin + categorical) compiled to a global state and sampled with `SpinGibbsConditional` and `CategoricalGibbsConditional`.
- Prototype runner with annotated plots: `examples/run_prototypes.py` (saves to `results/prototypes_*`).
- Discovery runner with temperature/noise/intervention sweeps: `examples/run_discovery.py` (saves CSV/JSON and plots under `results/discovery`).
- Optional CoA/ROE integration demo: `examples/run_coa_integration.py` with an adapter (`thrml/algorithms/coa_roe_adapter.py`) and a minimal scenario matcher (`thrml/coa_roe/scenarios.py`).

## How to run

```
# Prototypes (reduced sampling, annotated plots)
python examples/run_prototypes.py --seed 42

# Thermodynamic discovery sweeps
python examples/run_discovery.py --seed 42

# CoA/ROE integration demo
python examples/run_coa_integration.py --seed 42
```

## Algorithm highlights and KPIs

- SRSL: sweeps inverse temperature β = 1/T to amplify sub-threshold signals via stochastic resonance (KPIs: mutual information, gain, optimal β).
- TAPS: sensor scheduling with energy-aware Hamiltonian (KPIs: energy (J/s), coverage, active sensors).
- BPP: policy sampling with temperature-controlled risk tolerance (KPIs: time-to-escalation, action confidence, precision/recall proxy).
- EFSM: per-site energy baselines for anomaly detection (KPIs: anomaly score, ΔE, drift stability).
- TBRO: thermal bandit resource allocation (KPIs: SLO compliance, GPU-hours saved, overflow).
- LABI: kT ln 2 energy-aware Bayesian updates (KPIs: Landauer cost, skip rate, entropy delta).
- TCF: causal discovery via small interventions and equilibrium shifts (KPIs: number of causal edges, robustness delta, fused threat score).
- PPTS: probabilistic phase/time synchronization (KPIs: sync error ms, triangulation error m).
- TVS: thermal RNG watermarking/verification (KPIs: correlation, verified, bitrate overhead).
- REF: analog reservoir + EBM prior for stable features (KPIs: stability, energy per feature).

## Discovery methodology

- SRSL: grid over SNR (dB) × β, compute mutual information I(X;Y) and gain; plot MI heatmap to find resonance bands.
- LABI: grid over energy threshold (J) × likelihood magnitude; plot skip-rate surface to identify efficient operating frontiers.
- TCF: sweep perturbation strength; track causal-edge stability vs intervention amplitude.

Outputs live under `results/discovery/`:
- SRSL: `srsl_sweep.csv` + `srsl_mi_heatmap.png`
- LABI: `labi_frontier.csv` + `labi_skip_rate.png`
- TCF: `tcf_perturb.json` + `tcf_edges.png`

## Extending to Extropic hardware

- THRML’s blocked Gibbs and discrete-EBM utilities map naturally to probabilistic hardware. The discovery suite is designed to identify information-optimal β and regime frontiers now, and then pin those while porting the same blocked updates to hardware samplers.
- Hooks (e.g., LABI thresholding, SRSL β, TCF perturb strengths) are explicit knobs suitable for hardware control loops.

### Extropic silicon mapping (per algorithm)

- SRSL: The chip’s massively parallel Gibbs sampler relaxes toward the EBM’s equilibrium; tuning β directly sets the thermalization temperature, so weak-but-meaningful audio patterns get amplified at info‑optimal β.
- TAPS: Sensing decisions are random variables in an Ising Hamiltonian with cost terms; the hardware sampler finds low‑energy action subsets yielding maximal threat‑info per Joule.
- BPP: Policy actions are nodes in the energy model; temperature controls risk tolerance; sampling proposes coherent, ROE‑consistent escalations quickly.
- EFSM: A site‑specific energy well encodes “normal”; the sampler monitors ΔE as new frames arrive; low‑power continuous relaxation yields early anomaly cues.
- TBRO: Arms/sites are spins with reward priors; the sampler explores/exploits by temperature, allocating compute where it’s most valuable.
- LABI: The system pays energy (kT ln 2 per bit) only when entropy drops enough; in silicon, you literally thermalize fewer bits when not needed.
- TCF: Small do‑perturbations at nodes and observing equilibrium shifts estimate causal edges natively via hardware sampling.
- PPTS: Coupled oscillator phases are implemented as spin states; thermal coupling synchronizes phases (time/clock) without heavyweight protocols.
- TVS: Use on‑chip thermal randomness to drive nonce spins/watermark scheduling; verification is cheap while provenance remains strong.
- REF: The analog front‑end (reservoir) feeds a tiny Ising prior; the chip regularizes features into stable, low‑energy states with minimal power.

## Notes on performance and correctness

- All algorithms use the THRML global-state compilation path to minimize Python overhead.
- Two-color blocked Gibbs demo verifies equal block sizes and scalar-state expectations (batch_shape=()).
- EFSM paths include a safe placeholder visualization if a shape incompatibility is encountered; the positive path uses single-sample energy evaluation to avoid broadcasting issues.

## Files of interest

- Runners
  - `examples/run_prototypes.py`
  - `examples/run_discovery.py`
  - `examples/run_coa_integration.py`
- Algorithms: `thrml/algorithms/`
- Scenarios/adapter: `thrml/coa_roe/`
- Benchmarks: `thrml/benchmark_reports/` and `BENCHMARK_REPORT_SUMMARY.md`
