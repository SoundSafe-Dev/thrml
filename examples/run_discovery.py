#!/usr/bin/env python3
"""Thermodynamic Discovery Runner

Runs parameter sweeps and perturbation experiments to discover
thermodynamic regimes with operational value. Saves CSV/JSON and plots.

Experiments:
- SRSL: beta (temperature) × SNR sweep → MI, gain
- LABI: energy threshold × likelihood magnitude → skip/update frontier
- TCF: do-intervention strength × graph topology → causal edge stability

Usage:
  python examples/run_discovery.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from thrml import SamplingSchedule
from thrml.algorithms import (
	StochasticResonanceSignalLifter,
	LandauerAwareBayesianInference,
	ThermodynamicCausalFusion,
)
from thrml.algorithms.energy import EnergyConfig, estimate_sampling_energy_j, summarize_energy
from thrml.algorithms.power import PowerSession

FAST = SamplingSchedule(n_warmup=10, n_samples=50, steps_per_sample=1)


def ensure_dir(p: Path):
	p.mkdir(parents=True, exist_ok=True)


def exp_srsl(key, outdir: Path):
	ensure_dir(outdir)
	snrs = np.linspace(-12, 6, 10)
	betas = np.linspace(0.2, 4.0, 12)
	rows = []
	cfg = EnergyConfig()
	for snr in snrs:
		for beta in betas:
			weak = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 32)) * 0.3
			noise = jax.random.normal(key, (32,))
			alpha = 10 ** (snr / 20.0)
			weak_features = jnp.clip(alpha * weak + noise * 1.0, -3.0, 3.0)
			# Instantiate SRSL with singleton beta sweep via constructor
			srsl = StochasticResonanceSignalLifter(signal_window_size=32, beta_min=float(beta), beta_max=float(beta), n_beta_steps=1, key=key)
			_, kpis = srsl.forward(key, weak_features, FAST)
			energy = summarize_energy(tokens=32, sampling_j=estimate_sampling_energy_j(FAST.n_samples, 32, cfg), alerts=0, cfg=cfg)
			rows.append({"snr_db": float(snr), "beta": float(beta), **kpis, **{f"energy_{k}": v for k, v in energy.items()}})
	# Save CSV
	csv_path = outdir / "srsl_sweep.csv"
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=rows[0].keys())
		writer.writeheader(); writer.writerows(rows)
	# Plot MI heatmap
	mi_grid = np.full((len(snrs), len(betas)), np.nan)
	for r in rows:
		i = np.where(snrs == r["snr_db"])[0][0]
		j = np.where(betas == r["beta"])[0][0]
		mi_grid[i, j] = r["mutual_information"]
	fig, ax = plt.subplots(figsize=(6, 4))
	im = ax.imshow(mi_grid, aspect="auto", origin="lower", cmap="magma",
			extent=[betas[0], betas[-1], snrs[0], snrs[-1]])
	ax.set_xlabel("beta (1/T)"); ax.set_ylabel("SNR (dB)"); ax.set_title("SRSL: Mutual Information")
	plt.colorbar(im, ax=ax, label="I(X;Y)")
	fig.savefig(outdir / "srsl_mi_heatmap.png", dpi=150, bbox_inches="tight")
	plt.close(fig)
	return csv_path


def exp_labi(key, outdir: Path):
	ensure_dir(outdir)
	thresholds = np.geomspace(1e-21, 1e-17, 8)
	likelihood_scales = np.geomspace(0.05, 0.8, 10)
	rows = []
	cfg = EnergyConfig()
	for th in thresholds:
		labi = LandauerAwareBayesianInference(n_variables=16, energy_threshold=float(th), key=key)
		for s in likelihood_scales:
			lh = jax.random.uniform(key, (16,), minval=-s, maxval=s)
			(_, _), kpis = labi.forward(key, lh, FAST)
			energy = summarize_energy(tokens=16, sampling_j=estimate_sampling_energy_j(FAST.n_samples, 16, cfg), alerts=0, cfg=cfg)
			rows.append({"threshold_j": float(th), "lh_scale": float(s), **kpis, **{f"energy_{k}": v for k, v in energy.items()}})
	# Save CSV and frontier
	csv_path = outdir / "labi_frontier.csv"
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=rows[0].keys())
		writer.writeheader(); writer.writerows(rows)
	# Plot skip rate surface
	surface = np.full((len(thresholds), len(likelihood_scales)), np.nan)
	for r in rows:
		i = np.where(thresholds == r["threshold_j"])[0][0]
		j = np.where(likelihood_scales == r["lh_scale"])[0][0]
		surface[i, j] = r["skip_rate"]
	fig, ax = plt.subplots(figsize=(6, 4))
	im = ax.imshow(surface, aspect="auto", origin="lower", cmap="viridis",
			extent=[likelihood_scales[0], likelihood_scales[-1], thresholds[0], thresholds[-1]])
	ax.set_xscale("log"); ax.set_yscale("log")
	ax.set_xlabel("likelihood scale"); ax.set_ylabel("energy threshold (J)"); ax.set_title("LABI: Skip Rate")
	plt.colorbar(im, ax=ax, label="skip_rate")
	fig.savefig(outdir / "labi_skip_rate.png", dpi=150, bbox_inches="tight")
	plt.close(fig)
	return csv_path


def exp_tcf(key, outdir: Path):
	ensure_dir(outdir)
	strengths = np.linspace(0.1, 1.0, 10)
	tcf = ThermodynamicCausalFusion(n_modalities=4, key=key)
	mods = jax.random.uniform(key, (4,))
	rows = []
	cfg = EnergyConfig()
	for s in strengths:
		# run discover with perturbed strength via internal method
		graph, kpi1 = tcf.discover_causal_structure(key, mods, perturbation_strength=float(s), schedule=FAST)
		_, kpi2 = tcf.forward(key, mods, schedule=FAST)
		energy = summarize_energy(tokens=4, sampling_j=estimate_sampling_energy_j(FAST.n_samples, 4, cfg), alerts=0, cfg=cfg)
		rows.append({"perturb_strength": float(s), **{k: float(v) if hasattr(v, "__float__") else v for k, v in {**kpi1, **kpi2}.items() if isinstance(v, (int, float))}, **{f"energy_{k}": v for k, v in energy.items()}})
	# Save JSON
	json_path = outdir / "tcf_perturb.json"
	with open(json_path, "w") as f:
		json.dump(rows, f, indent=2)
	# Plot causal edges vs strength
	x = [r["perturb_strength"] for r in rows]
	y = [r.get("n_causal_edges", 0.0) for r in rows]
	fig, ax = plt.subplots(figsize=(6, 3))
	ax.plot(x, y, marker="o"); ax.set_xlabel("perturbation strength"); ax.set_ylabel("# causal edges")
	ax.set_title("TCF: Causal edge stability")
	fig.savefig(outdir / "tcf_edges.png", dpi=150, bbox_inches="tight")
	plt.close(fig)
	return json_path


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--results", type=str, default="results/discovery")
	parser.add_argument("--measure-power", action="store_true")
	args = parser.parse_args()

	key = jax.random.key(args.seed)
	outdir = Path(args.results)
	ensure_dir(outdir)

	print("Running SRSL sweep...")
	ps = PowerSession() if args.measure_power else None
	if ps: ps.start()
	srsl_csv = exp_srsl(key, outdir / "srsl")
	meas_j_srsl = ps.stop() if ps else None
	print(f"  -> {srsl_csv}")

	print("Running LABI frontier...")
	ps = PowerSession() if args.measure_power else None
	if ps: ps.start()
	labi_csv = exp_labi(key, outdir / "labi")
	meas_j_labi = ps.stop() if ps else None
	print(f"  -> {labi_csv}")

	print("Running TCF perturbations...")
	ps = PowerSession() if args.measure_power else None
	if ps: ps.start()
	tcf_json = exp_tcf(key, outdir / "tcf")
	meas_j_tcf = ps.stop() if ps else None
	print(f"  -> {tcf_json}")

	# Write a small measured energy summary if any
	if args.measure_power:
		(outdir / "measured_energy.json").write_text(
			json.dumps({"srsl_j": meas_j_srsl, "labi_j": meas_j_labi, "tcf_j": meas_j_tcf}, indent=2)
		)

	print("Discovery complete.")


if __name__ == "__main__":
	main()
