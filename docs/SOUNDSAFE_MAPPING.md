# SoundSafe Mapping to Thermodynamic Algorithms

This note maps the 10 thermal algorithms to SoundSafe capabilities and explains how Extropic’s silicon—massively parallel Gibbs dynamics relaxing to low‑energy states—makes them natural fits.

## Chip Narrative

Extropic’s thermodynamic chips implement massively parallel Gibbs sampling. The chip relaxes toward the equilibrium distribution of an energy function encoded by weights. Every algorithm below sets up an EBM or Hamiltonian so the chip’s native dynamics compute the desired inference/action at minimal energy.

## Capability → Algorithm

- Deepfake Voice Detection
  - TVS (watermarking) + SRSL (pre‑proc) + LABI (gating)
  - Verify provenance; surface weak spoof artifacts via optimal β; avoid wasteful updates when ΔH is small.
- Audio Watermarking & Content Protection
  - TVS nonce watermarks driven by thermal RNG; verification is cheap; provenance intact.
- Anomalous Sound Detection; Behavioral Sound Mapping
  - EFSM per‑site energy baselines (ΔE spikes = anomalies), SRSL lift early onsets, LABI energy‑aware updates.
- Weapon/Aggression/Loitering/Zone
  - REF stable features at low power; TAPS schedules sensors/bitrates; TBRO routes compute; BPP escalates under ROE.
- Smart Footage Tagging
  - REF+EFSM+LABI pipeline tags streams with minimal energy per token.
- Environmental + Access Sensors
  - PPTS synchronizes phases (probabilistic clocking) and TCF fuses cross‑modal signals causally.
- Multimodal Event Fusion
  - TCF causal edges remain robust under modality failures; drives better alerts.

## Why thermodynamic compute reduces Joules while increasing tokens

- SRSL finds β (§ discovery heatmaps) that maximizes information per unit energy—not raw gain.
- LABI enforces kT ln 2 per bit only when entropy reduces enough—skipping expensive updates.
- TAPS/TBRO set up Hamiltonians with explicit energy costs—chip finds low‑energy sensing/compute subsets.
- REF stabilizes feature extraction—fewer flops, smoother downstream sampling.
- PPTS synchronizes clocks without heavy protocols.
- TCF learns robust fusion with small do‑perturbations—fewer false positives, fewer re‑computes.

Run the demos and discovery sweeps to view annotated plots and energy/tokens KPIs. See `results/report.html` for a single‑page summary.
