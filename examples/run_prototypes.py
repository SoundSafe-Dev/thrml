#!/usr/bin/env python3
"""Prototype runs for 10 thermal algorithms (reduced sample counts).

- CPU-only JAX works out-of-the-box.
- Apple GPU (Metal) works if jax-metal is installed (optional).
- Saves figures under results/prototypes_YYYYmmdd_HHMMSS/.

Usage:
  python examples/run_prototypes.py --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
try:
	import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
	raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`. ")
import numpy as np
from typing import cast, Any

from thrml import (
	SpinNode,
	CategoricalNode,
	Block as BlockAlias,  # avoid name clash; we import concrete Block below
	SamplingSchedule,
	sample_states,
)
from thrml.algorithms import (
	BoltzmannPolicyPlanner,
	EnergyFingerprintedSceneMemory,
	LandauerAwareBayesianInference,
	ProbabilisticPhaseTimeSync,
	ReservoirEBMFrontEnd,
	StochasticResonanceSignalLifter,
	ThermoVerifiableSensing,
	ThermalBanditResourceOrchestrator,
	ThermodynamicActivePerceptionScheduling,
	ThermodynamicCausalFusion,
)
from thrml.models.discrete_ebm import SpinGibbsConditional, CategoricalGibbsConditional, SpinEBMFactor, CategoricalEBMFactor
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, BlockSamplingProgram, sample_states as sample_states_core
from thrml.factor import FactorSamplingProgram
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.algorithms.energy import EnergyConfig, estimate_sampling_energy_j, summarize_energy
from thrml.algorithms.power import PowerSession

# Reduced schedule (fast local dev)
FAST_SCHEDULE = SamplingSchedule(n_warmup=10, n_samples=30, steps_per_sample=1)

# SoundSafe capability mapping
SOUNDSAFE_CAPABILITIES = {
	"srsl": "Deepfake Voice Detection (pre‑proc)",
	"labi": "Anomalous Sound Detection (gating)",
	"tcf": "Multimodal Event Fusion",
	"taps": "Weapon/Aggression/Loitering/Zone",
	"bpp": "Weapon/Aggression/Loitering/Zone",
	"efsm": "Anomalous Sound Detection",
	"tbro": "Weapon/Aggression/Loitering/Zone",
	"ppts": "Environmental + Access Sensors",
	"tvs": "Audio Watermarking & Content Protection",
	"ref": "Weapon/Aggression/Loitering/Zone",
}


def load_optimal_params(discovery_dir: Path) -> dict[str, float | None]:
	"""Load optimal operating points from discovery results."""
	params = {}
	# SRSL: find β with max mutual information
	srsl_csv = discovery_dir / "srsl" / "srsl_sweep.csv"
	if srsl_csv.exists():
		import csv
		rows = list(csv.DictReader(open(srsl_csv)))
		if rows:
			best = max(rows, key=lambda r: float(r.get("mutual_information", 0)))
			params["srsl_beta"] = float(best.get("beta", 1.0))
	# LABI: find threshold with skip_rate ~0.5 and reasonable energy
	labi_csv = discovery_dir / "labi" / "labi_frontier.csv"
	if labi_csv.exists():
		import csv
		rows = list(csv.DictReader(open(labi_csv)))
		if rows:
			# Prefer moderate skip_rate (0.3-0.7) with lowest threshold that works
			candidates = [r for r in rows if 0.3 <= float(r.get("skip_rate", 0)) <= 0.7]
			if candidates:
				best = min(candidates, key=lambda r: float(r.get("threshold_j", 1e-18)))
				params["labi_threshold"] = float(best.get("threshold_j", 1e-18))
			else:
				# Fallback: lowest threshold that gives any updates
				candidates = [r for r in rows if float(r.get("skip_rate", 1)) < 0.9]
				if candidates:
					best = min(candidates, key=lambda r: float(r.get("threshold_j", 1e-18)))
					params["labi_threshold"] = float(best.get("threshold_j", 1e-18))
	# TCF: find perturbation strength with stable edge count
	tcf_json = discovery_dir / "tcf" / "tcf_perturb.json"
	if tcf_json.exists():
		import json
		data = json.loads(tcf_json.read_text())
		if data:
			# Find point with good edge count (not too low) and stable
			edge_counts = [float(r.get("n_causal_edges", 0)) for r in data]
			if edge_counts:
				median_edges = sorted(edge_counts)[len(edge_counts)//2]
				candidates = [r for r in data if abs(float(r.get("n_causal_edges", 0)) - median_edges) < 1.0]
				if candidates:
					best = candidates[len(candidates)//2]  # middle of stable region
					params["tcf_perturb"] = float(best.get("perturb_strength", 0.5))
	return params


def _savefig(fig, outdir: Path, name: str):
	outdir.mkdir(parents=True, exist_ok=True)
	fig.savefig(outdir / f"{name}.png", dpi=150, bbox_inches="tight")
	plt.close(fig)


def _textbox(ax, text: str):
	ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)


def _add_soundsafe_label(fig, algo_name: str):
	"""Add SoundSafe capability label at the top of the figure."""
	cap = SOUNDSAFE_CAPABILITIES.get(algo_name, "Thermal Algorithm")
	fig.suptitle(f"{algo_name.upper()}: {cap}", fontsize=10, fontweight="bold", y=0.995)


def run_srsl(key, outdir: Path, optimal_beta: float | None = None):
	# Use optimal beta if provided (from discovery), otherwise let algorithm find it
	if optimal_beta is not None:
		srsl = StochasticResonanceSignalLifter(signal_window_size=32, beta_min=float(optimal_beta), beta_max=float(optimal_beta), n_beta_steps=1, key=key)
	else:
		srsl = StochasticResonanceSignalLifter(signal_window_size=32, key=key)
	weak_signal = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 32)) * 0.3 + jax.random.normal(key, (32,)) * 0.4
	amplified, kpis = srsl.forward(key, weak_signal, FAST_SCHEDULE)
	
	# Enhanced before/after visualization
	fig = plt.figure(figsize=(14, 7))
	gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
	
	# BEFORE/AFTER side-by-side waveforms
	ax_before = fig.add_subplot(gs[0, 0])
	ax_after = fig.add_subplot(gs[0, 1])
	ax_before.plot(weak_signal, color="gray", lw=2, label="BEFORE: Weak Signal")
	ax_before.set_title("BEFORE: Weak Signal (SNR≈-6dB)", fontweight="bold", color="gray")
	ax_before.grid(True, alpha=0.3); ax_before.legend()
	ax_after.plot(amplified, color="tab:blue", lw=2, label="AFTER: Amplified")
	ax_after.set_title("AFTER: SRSL Amplified", fontweight="bold", color="tab:blue")
	ax_after.grid(True, alpha=0.3); ax_after.legend()
	
	# Overlay comparison
	ax_overlay = fig.add_subplot(gs[0, 2])
	ax_overlay.plot(weak_signal, label="Weak", lw=1.5, alpha=0.7, color="gray")
	ax_overlay.plot(amplified, label="Amplified", lw=1.5, alpha=0.9, color="tab:blue")
	ax_overlay.set_title("Overlay Comparison")
	ax_overlay.legend(); ax_overlay.grid(True, alpha=0.3)
	
	# Power spectrum comparison
	ax_power = fig.add_subplot(gs[0, 3])
	power_before = float(jnp.std(weak_signal))
	power_after = float(jnp.std(amplified))
	ax_power.barh(["BEFORE", "AFTER"], [power_before, power_after], color=["gray", "tab:blue"], alpha=0.7)
	ax_power.set_title(f"Signal Power\nGain: {power_after/power_before:.2f}x")
	ax_power.set_xlabel("Std Dev")
	
	# Histogram comparison
	ax_hist = fig.add_subplot(gs[1, 0:2])
	ax_hist.hist(np.array(weak_signal), bins=20, alpha=0.6, label="BEFORE", color="gray", density=True)
	ax_hist.hist(np.array(amplified), bins=20, alpha=0.6, label="AFTER", color="tab:blue", density=True)
	ax_hist.set_title("Distribution Shift (Normalized)", fontweight="bold")
	ax_hist.set_xlabel("Amplitude"); ax_hist.set_ylabel("Density"); ax_hist.legend()
	ax_hist.grid(True, alpha=0.3)
	
	# KPI box with optimal beta annotation
	ax_kpi = fig.add_subplot(gs[1, 2])
	ax_kpi.axis("off")
	beta_note = f" (DISCOVERY: {optimal_beta:.3f})" if optimal_beta is not None else ""
	_textbox(ax_kpi, f"Mutual Information: {kpis['mutual_information']:.3f}\n"
					  f"Optimal Beta: {kpis['optimal_beta']:.3f}{beta_note}\n"
					  f"Signal Gain: {kpis['signal_gain']:.3f}x\n"
					  f"Energy/Event: {kpis['energy_per_event_joules']:.2e} J")
	
	# SoundSafe explanation
	ax_exp = fig.add_subplot(gs[1, 3])
	ax_exp.axis("off")
	_textbox(ax_exp, "Extropic Chip:\nGibbs dynamics at β*=1/T*\nmaximize I(X;Y) per Joule.\nLifts weak spoof artifacts\nin deepfake voice detection.")
	
	_add_soundsafe_label(fig, "srsl")
	_savefig(fig, outdir, "srsl")
	return kpis


def run_taps(key, outdir: Path):
	taps = ThermodynamicActivePerceptionScheduling(n_sensors=8, n_bitrate_levels=3, key=key)
	threat_scores = jax.random.uniform(key, (8,)); threat_scores = threat_scores.at[2].set(0.9)
	actions, kpis = taps.forward(key, threat_scores, FAST_SCHEDULE)
	fig, ax = plt.subplots(2, 2, figsize=(10, 6))
	ax[0, 0].bar(range(8), threat_scores); ax[0, 0].set_title("Threat Scores by Sensor"); ax[0, 0].set_xlabel("Sensor")
	ax[0, 0].grid(True, alpha=0.3)
	im = ax[0, 1].imshow(actions, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
	ax[0, 1].set_title("Activation Probabilities (Sensor × Bitrate)")
	fig.colorbar(im, ax=ax[0, 1])
	# Bitrate distribution
	bitrate_mass = np.array(actions).sum(axis=0)
	ax[1, 0].bar(range(actions.shape[1]), bitrate_mass, color="tab:orange")
	ax[1, 0].set_title("Total Activation Mass per Bitrate Level")
	ax[1, 0].grid(True, alpha=0.3)
	# KPI box
	ax[1, 1].axis("off")
	_textbox(ax[1, 1], f"Energy: {kpis['total_energy_joules_per_sec']:.3f} J/s\n"
					  f"Coverage: {kpis['threat_coverage']:.3f}\n"
					  f"Active sensors: {kpis['active_sensors']}\n\n"
					  f"SoundSafe: Weapon/Aggression detection\n"
					  f"Extropic: Thermal scheduling minimizes\n"
					  f"Joules while maintaining coverage.")
	_add_soundsafe_label(fig, "taps")
	_savefig(fig, outdir, "taps")
	return kpis


def run_bpp(key, outdir: Path):
	bpp = BoltzmannPolicyPlanner(risk_temperature=1.0, key=key)
	threat_level = 2
	tactic_scores = jnp.array([0.1, 0.3, 0.9, 0.6, 0.2])
	action_probs, kpis = bpp.forward(key, threat_level, tactic_scores, FAST_SCHEDULE)
	fig, ax = plt.subplots(2, 2, figsize=(10, 6))
	# Action probabilities
	ax[0, 0].bar(bpp.action_names, np.array(action_probs), color="tab:green")
	ax[0, 0].set_title("Policy Action Probabilities")
	ax[0, 0].tick_params(axis='x', rotation=30)
	ax[0, 0].grid(True, alpha=0.3)
	# Tactic scores
	ax[0, 1].bar(range(len(tactic_scores)), tactic_scores, color="tab:purple")
	ax[0, 1].set_title("Tactic Scores")
	ax[0, 1].grid(True, alpha=0.3)
	# Highlight chosen action text
	ax[1, 0].axis("off")
	_textbox(ax[1, 0], f"Chosen: {kpis['selected_action']}\n"
					  f"Confidence: {kpis['action_confidence']:.3f}\n"
					  f"Time-to-escalation: {kpis['time_to_escalation']:.3f}\n\n"
					  f"SoundSafe: ROE-compatible escalation\n"
					  f"under thermal risk control.")
	# Legend box
	ax[1, 1].axis("off")
	_textbox(ax[1, 1], "Lower temperature = more risk-averse;\n"
					  "policy fuses playbook priors with live tactics.\n\n"
					  "Extropic: Gibbs sampling over action\n"
					  "space minimizes risky decisions.")
	_add_soundsafe_label(fig, "bpp")
	_savefig(fig, outdir, "bpp")
	return kpis


def run_efsm(key, outdir: Path):
	# EFSM currently sensitive to shapes; run with simplified windows
	try:
		efsm = EnergyFingerprintedSceneMemory(n_features=64, adaptation_rate=0.01, key=key)
		clean_windows = jax.random.normal(key, (40, 64)) * 0.4
		efsm.fit_baseline(key, clean_windows, FAST_SCHEDULE)
		normal_scene = jax.random.normal(key, (64,)) * 0.4
		anomalous_scene = normal_scene + 4.0 + jax.random.normal(key, (64,)) * 1.2
		n_score, _ = efsm.forward(key, normal_scene, FAST_SCHEDULE)
		a_score, _ = efsm.forward(key, anomalous_scene, FAST_SCHEDULE)
		fig, ax = plt.subplots(1, 2, figsize=(10, 4))
		ax[0].bar(["BEFORE\n(Normal)", "AFTER\n(Anomaly)"], [float(n_score), float(a_score)], 
				  color=["green", "red"], alpha=0.7, edgecolor="black", lw=2)
		ax[0].set_title("EFSM: Energy Fingerprint Scores", fontweight="bold")
		ax[0].set_ylabel("ΔE (Energy Deviation)")
		ax[0].grid(True, alpha=0.3)
		ax[1].axis("off")
		_textbox(ax[1], f"BEFORE (Normal): {float(n_score):.2f}\n"
						f"AFTER (Anomaly): {float(a_score):.2f}\n\n"
						f"ΔE Spike: {float(a_score - n_score):.2f}\n\n"
						f"SoundSafe: Anomalous sound detection\n"
						f"via per-site energy baselines.\n\n"
						f"Extropic: Chip thermalizes to baseline;\n"
						f"anomalies → higher energy states.")
		_add_soundsafe_label(fig, "efsm")
		_savefig(fig, outdir, "efsm")
		return {"normal_score": float(n_score), "anomaly_score": float(a_score)}
	except Exception as e:  # noqa: BLE001
		fig, ax = plt.subplots(figsize=(8, 3))
		ax.axis("off")
		_textbox(ax, "EFSM temporarily unavailable.\n"
				  "Action: fix broadcasting by aligning batch dims\n"
				  "or using non-batched energy eval for baseline.")
		_add_soundsafe_label(fig, "efsm")
		_savefig(fig, outdir, "efsm")
		return {"error": str(e)}


def run_tbro(key, outdir: Path):
	tbro = ThermalBanditResourceOrchestrator(n_sites=10, exploration_temperature=1.0, key=key)
	risk = jax.random.uniform(key, (10,)); risk = risk.at[3].set(0.95)
	allocation, kpis = tbro.forward(key, risk, FAST_SCHEDULE)
	fig, ax = plt.subplots(2, 2, figsize=(10, 6))
	ax[0, 0].bar(range(10), risk, color="tab:gray", alpha=0.7); ax[0, 0].set_title("BEFORE: Site Risk")
	ax[0, 0].grid(True, alpha=0.3)
	ax[0, 1].bar(range(10), allocation, color="tab:blue", alpha=0.8); ax[0, 1].set_title("AFTER: Resource Allocation")
	ax[0, 1].grid(True, alpha=0.3)
	# Risk vs allocation scatter
	ax[1, 0].scatter(np.array(risk), np.array(allocation), c="tab:blue", s=50, alpha=0.7)
	ax[1, 0].plot([0, 1], [0, 1], "k--", lw=1)
	ax[1, 0].set_xlabel("Risk"); ax[1, 0].set_ylabel("Allocation")
	ax[1, 0].set_title("Risk vs Allocation")
	ax[1, 0].grid(True, alpha=0.3)
	# KPI box
	ax[1, 1].axis("off")
	_textbox(ax[1, 1], f"SLO compliance: {kpis['slo_compliance']:.3f}\n"
					  f"GPU-hours saved: {kpis['gpu_hours_saved_percent']:.1f}%\n"
					  f"Overflow: {kpis['overflow_drops']:.3f}\n\n"
					  f"SoundSafe: Routes compute to high-risk\n"
					  f"zones (weapon/aggression detection).\n\n"
					  f"Extropic: Bandit learns optimal\n"
					  f"allocation via thermal sampling.")
	_add_soundsafe_label(fig, "tbro")
	_savefig(fig, outdir, "tbro")
	return kpis


def run_labi(key, outdir: Path, optimal_threshold: float | None = None):
	threshold = optimal_threshold if optimal_threshold is not None else 1e-18
	labi = LandauerAwareBayesianInference(n_variables=16, energy_threshold=float(threshold), key=key)
	lh = jax.random.uniform(key, (16,), minval=-0.4, maxval=0.4)
	(_, _), kpis = labi.forward(key, lh, FAST_SCHEDULE)
	
	fig = plt.figure(figsize=(14, 6))
	gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
	
	# BEFORE/AFTER: likelihood deltas vs posterior decision
	ax_before = fig.add_subplot(gs[0, 0])
	ax_before.bar(range(16), lh, color="gray", alpha=0.7)
	ax_before.set_title("BEFORE: Likelihood Deltas", fontweight="bold", color="gray")
	ax_before.set_xlabel("Variable"); ax_before.set_ylabel("Δ Likelihood")
	ax_before.axhline(0, color="k", linestyle="--", lw=0.5)
	ax_before.grid(True, alpha=0.3)
	
	ax_after = fig.add_subplot(gs[0, 1])
	decision = "UPDATE" if kpis.get("should_update", False) else "SKIP"
	entropy_delta = kpis.get("entropy_delta_bits", 0.0)
	landauer = kpis.get("landauer_cost_joules", 0.0)
	ax_after.axis("off")
	color = "tab:green" if decision == "UPDATE" else "tab:orange"
	ax_after.text(0.5, 0.7, decision, transform=ax_after.transAxes, ha="center", va="center",
				  fontsize=24, fontweight="bold", color=color)
	ax_after.text(0.5, 0.5, f"ΔH = {entropy_delta:.4f} bits", transform=ax_after.transAxes, ha="center", fontsize=10)
	ax_after.text(0.5, 0.4, f"Landauer = {landauer:.2e} J", transform=ax_after.transAxes, ha="center", fontsize=9)
	ax_after.set_title("AFTER: Decision", fontweight="bold")
	
	# Energy frontier visualization (skip vs update)
	ax_frontier = fig.add_subplot(gs[0, 2:4])
	threshold_note = f" (DISCOVERY: {optimal_threshold:.2e})" if optimal_threshold is not None else ""
	_textbox(ax_frontier, f"Threshold: {threshold:.2e}{threshold_note}\n"
						  f"Skip Rate: {kpis.get('skip_rate', 0.0):.3f}\n"
						  f"Updates Saved: {1.0 - kpis.get('skip_rate', 0.0):.1%}\n"
						  f"Energy Saved: {(1.0 - kpis.get('skip_rate', 0.0)) * landauer:.2e} J")
	ax_frontier.set_title("Energy Frontier (Skip vs Update)", fontweight="bold")
	
	# Explanation and SoundSafe mapping
	ax_exp1 = fig.add_subplot(gs[1, 0:2])
	ax_exp1.axis("off")
	_textbox(ax_exp1, "Rule: Pay kT ln 2 per bit ONLY when entropy reduction (ΔH) justifies it.\n\n"
					  "SoundSafe: Gates expensive inference updates in anomalous sound detection.\n"
					  "Extropic Chip: Skips thermal transitions when ΔE < threshold → fewer Joules/token.")
	
	ax_exp2 = fig.add_subplot(gs[1, 2:4])
	ax_exp2.axis("off")
	skip_pct = kpis.get('skip_rate', 0.0) * 100
	update_pct = (1.0 - kpis.get('skip_rate', 0.0)) * 100
	_textbox(ax_exp2, f"Decision Breakdown:\n"
					  f"  • Skip: {skip_pct:.1f}% (save energy)\n"
					  f"  • Update: {update_pct:.1f}% (when ΔH warrants)\n\n"
					  f"Operating Point: Threshold={threshold:.2e} J\n"
					  f"tuned to balance detection rate vs energy cost.")
	
	_add_soundsafe_label(fig, "labi")
	_savefig(fig, outdir, "labi")
	return kpis


def run_tcf(key, outdir: Path, optimal_perturb: float | None = None):
	perturb = optimal_perturb if optimal_perturb is not None else 0.5
	tcf = ThermodynamicCausalFusion(n_modalities=4, key=key)
	mods = jax.random.uniform(key, (4,))
	graph, kpi1 = tcf.discover_causal_structure(key, mods, perturbation_strength=float(perturb), schedule=FAST_SCHEDULE)
	fused, kpi2 = tcf.forward(key, mods, schedule=FAST_SCHEDULE)
	
	fig = plt.figure(figsize=(14, 6))
	gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
	
	# BEFORE: Raw modality readings
	ax_before = fig.add_subplot(gs[0, 0])
	ax_before.bar(range(4), mods, color="tab:gray", alpha=0.7)
	ax_before.set_title("BEFORE: Raw Modality Readings", fontweight="bold", color="gray")
	ax_before.set_xlabel("Modality (A/V/Doors/Temp)"); ax_before.set_ylabel("Score")
	ax_before.set_ylim([0, 1]); ax_before.grid(True, alpha=0.3)
	
	# AFTER: Fused threat scores
	ax_after = fig.add_subplot(gs[0, 1])
	ax_after.bar(range(4), fused, color="tab:blue", alpha=0.8)
	ax_after.set_title("AFTER: Causal Fusion", fontweight="bold", color="tab:blue")
	ax_after.set_xlabel("Modality"); ax_after.set_ylabel("Fused Threat Score")
	ax_after.set_ylim([0, 1]); ax_after.grid(True, alpha=0.3)
	
	# Causal graph discovery
	ax_graph = fig.add_subplot(gs[0, 2])
	im = ax_graph.imshow(graph, cmap="Blues", vmin=0, vmax=1, aspect="auto")
	ax_graph.set_title("Causal Graph (Discovered)", fontweight="bold")
	ax_graph.set_xlabel("To"); ax_graph.set_ylabel("From")
	fig.colorbar(im, ax=ax_graph, label="Edge Strength")
	
	# Comparison: raw vs fused
	ax_compare = fig.add_subplot(gs[0, 3])
	x = np.arange(4)
	width = 0.35
	ax_compare.bar(x - width/2, mods, width, label="Raw", color="tab:gray", alpha=0.7)
	ax_compare.bar(x + width/2, fused, width, label="Fused", color="tab:blue", alpha=0.8)
	ax_compare.set_title("Raw vs Fused Comparison")
	ax_compare.set_xlabel("Modality"); ax_compare.set_ylabel("Score")
	ax_compare.legend(); ax_compare.grid(True, alpha=0.3)
	
	# KPIs and discovery annotation
	ax_kpi = fig.add_subplot(gs[1, 0:2])
	ax_kpi.axis("off")
	perturb_note = f" (DISCOVERY: {optimal_perturb:.2f})" if optimal_perturb is not None else ""
	_textbox(ax_kpi, f"Discovery Results:\n"
					  f"  • Perturbation Strength: {perturb:.2f}{perturb_note}\n"
					  f"  • Causal Edges: {int(kpi1.get('n_causal_edges', 0))}\n"
					  f"  • Robustness Δ: {kpi2.get('robustness_delta', 0.0):.3f}\n"
					  f"  • Threat Score: {kpi2.get('threat_score', 0.0):.3f}\n\n"
					  f"SoundSafe: Multimodal fusion (A/V/doors)\n"
					  f"survives sensor failures via causal structure.")
	
	# Explanation
	ax_exp = fig.add_subplot(gs[1, 2:4])
	ax_exp.axis("off")
	_textbox(ax_exp, "Extropic Chip:\n"
					 "  • Do-interventions shift equilibrium\n"
					 "  • Causal edges = stable under perturbations\n"
					 "  • Fewer false positives → fewer re-computes\n\n"
					 "SoundSafe Application:\n"
					 "  • Environmental + Access Sensors\n"
					 "  • Robust fusion despite modality dropouts")
	
	_add_soundsafe_label(fig, "tcf")
	_savefig(fig, outdir, "tcf")
	kpis = {**kpi1, **kpi2}
	kpis.pop("causal_matrix", None)
	return kpis


def run_ppts(key, outdir: Path):
	ppts = ProbabilisticPhaseTimeSync(n_sensors=6, coupling_strength=0.8, key=key)
	obs = jax.random.uniform(key, (6,), minval=0.0, maxval=2*jnp.pi)
	phases, kpis = ppts.forward(key, obs, FAST_SCHEDULE)
	fig = plt.figure(figsize=(12, 5))
	# Polar plot
	ax1 = fig.add_subplot(1, 2, 1, projection='polar')
	for ph in phases:
		ax1.arrow(float(ph), 0, 0, 1.0, width=0.01, color='tab:blue', alpha=0.7)
	ax1.set_title("AFTER: Synchronized Phases (Polar)", fontweight="bold", pad=20)
	# Phase difference histogram
	diffs = []
	for i in range(len(phases)):
		for j in range(i+1, len(phases)):
			d = abs(float(phases[i] - phases[j]))
			d = min(d, 2*np.pi - d)
			diffs.append(d)
	ax2 = fig.add_subplot(1, 2, 2)
	ax2.hist(diffs, bins=10, color='tab:orange', alpha=0.8, edgecolor='black')
	ax2.set_title("Pairwise Phase Differences", fontweight="bold")
	ax2.set_xlabel("Phase Difference (rad)"); ax2.set_ylabel("Count")
	ax2.grid(True, alpha=0.3)
	_textbox(ax2, f"Sync error: {kpis['sync_error_ms']:.2f} ms\n"
				  f"Triangulation: {kpis['triangulation_error_m']:.4f} m\n\n"
				  f"SoundSafe: Environmental sensors\n"
				  f"probabilistic time sync.\n\n"
				  f"Extropic: Low-overhead clocking\n"
				  f"via thermal phase coupling.")
	_add_soundsafe_label(fig, "ppts")
	_savefig(fig, outdir, "ppts")
	return kpis


def run_tvs(key, outdir: Path):
	tvs = ThermoVerifiableSensing(nonce_size=32, watermark_strength=0.1, key=key)
	stream = jax.random.normal(key, (100, 64))
	watermarked, wm_kpis = tvs.forward(key, stream, mode="watermark", schedule=FAST_SCHEDULE)
	wm_array = jnp.array(watermarked)
	is_valid, v_kpis = tvs.verify_stream(wm_array)
	
	fig = plt.figure(figsize=(14, 6))
	gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
	
	# BEFORE: Original stream
	ax_before = fig.add_subplot(gs[0, 0])
	ax_before.plot(np.array(stream)[:30, 0], color="gray", lw=2, label="BEFORE")
	ax_before.set_title("BEFORE: Original Stream", fontweight="bold", color="gray")
	ax_before.set_xlabel("Frame"); ax_before.set_ylabel("Amplitude")
	ax_before.grid(True, alpha=0.3); ax_before.legend()
	
	# AFTER: Watermarked stream
	ax_after = fig.add_subplot(gs[0, 1])
	ax_after.plot(np.array(watermarked)[:30, 0], color="tab:blue", lw=2, label="AFTER", alpha=0.8)
	ax_after.set_title("AFTER: Watermarked Stream", fontweight="bold", color="tab:blue")
	ax_after.set_xlabel("Frame"); ax_after.set_ylabel("Amplitude")
	ax_after.grid(True, alpha=0.3); ax_after.legend()
	
	# Overlay comparison
	ax_overlay = fig.add_subplot(gs[0, 2])
	ax_overlay.plot(np.array(stream)[:30, 0], label="Original", alpha=0.7, color="gray")
	ax_overlay.plot(np.array(watermarked)[:30, 0], label="Watermarked", alpha=0.9, color="tab:blue")
	ax_overlay.set_title("Overlay Comparison")
	ax_overlay.legend(); ax_overlay.grid(True, alpha=0.3)
	
	# Residual (watermark effect)
	resid = np.array(watermarked - stream)
	ax_resid = fig.add_subplot(gs[0, 3])
	ax_resid.plot(resid[:30, 0], color='tab:red', lw=1.5)
	ax_resid.set_title("Residual (WM Effect)", fontweight="bold")
	ax_resid.axhline(0, color="k", linestyle="--", lw=0.5)
	ax_resid.set_xlabel("Frame"); ax_resid.grid(True, alpha=0.3)
	
	# Verification results
	ax_verify = fig.add_subplot(gs[1, 0:2])
	ax_verify.axis("off")
	verified = v_kpis.get('verified', False)
	corr = v_kpis.get('correlation', 0.0)
	overhead = wm_kpis.get('bitrate_overhead_percent', 0.0)
	color = "tab:green" if verified else "tab:red"
	ax_verify.text(0.5, 0.8, "VERIFIED" if verified else "NOT VERIFIED", 
				   transform=ax_verify.transAxes, ha="center", fontsize=20, fontweight="bold", color=color)
	_textbox(ax_verify, f"\n\nCorrelation: {corr:.3f}\n"
						f"Bitrate Overhead: {overhead:.4f}%\n"
						f"SoundSafe: Court-admissible provenance\n"
						f"Extropic: Thermal RNG embeds nonce\n"
						f"→ Hardware verifiable watermarks")
	
	# Energy and explanation
	ax_exp = fig.add_subplot(gs[1, 2:4])
	ax_exp.axis("off")
	_textbox(ax_exp, "SoundSafe Application:\n"
					 "  • Deepfake voice detection (provenance)\n"
					 "  • Content protection (litigation-ready)\n"
					 "  • Chain of custody verification\n\n"
					 f"Energy: {wm_kpis.get('energy_total_joules', 0.0):.2e} J per 100 frames\n"
					 f"Extropic advantage: Thermal RNG embedded\n"
					 f"in Gibbs sampling → negligible overhead.")
	
	_add_soundsafe_label(fig, "tvs")
	_savefig(fig, outdir, "tvs")
	return {"correlation": v_kpis.get("correlation", 0.0), "verified": v_kpis.get("verified", False), **wm_kpis}


def run_ref(key, outdir: Path):
	ref = ReservoirEBMFrontEnd(reservoir_size=64, feature_size=24, key=key)
	raw = jax.random.normal(key, (40, 16))
	features, kpis = ref.forward(key, raw, FAST_SCHEDULE)
	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].bar(range(len(features)), features, color="tab:blue", alpha=0.8)
	ax[0].set_title("AFTER: Stable Features", fontweight="bold")
	ax[0].set_xlabel("Feature Index"); ax[0].set_ylabel("Feature Value")
	ax[0].grid(True, alpha=0.3)
	ax[1].axis("off")
	_textbox(ax[1], f"Feature stability: {kpis.get('feature_stability', 0.0):.3f}\n"
				  f"Energy/feature: {kpis.get('energy_per_feature_uj', 0.0):.3e} µJ\n\n"
				  f"SoundSafe: Low-power feature extraction\n"
				  f"for weapon/aggression detection.\n\n"
				  f"Extropic: Reservoir + EBM prior\n"
				  f"→ stable features at ultra-low cost.")
	_add_soundsafe_label(fig, "ref")
	_savefig(fig, outdir, "ref")
	return kpis


def run_core_block_gibbs_two_color(key, outdir: Path):
	# Quick example from README: Ising chain with two-color blocked Gibbs
	nodes = [SpinNode() for _ in range(4)]
	edges = [(nodes[i], nodes[i+1]) for i in range(3)]
	nodes_abs = cast(list[Any], nodes)
	edges_abs = cast(list[Any], edges)
	biases = jnp.zeros((4,))
	weights = jnp.ones((3,)) * 0.5
	beta = jnp.array(1.0)
	model = IsingEBM(cast(list[Any], nodes_abs), cast(list[Any], edges_abs), biases, weights, beta)

	free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
	program = IsingSamplingProgram(model, cast(list[Any], free_blocks), clamped_blocks=[])

	k_init, k_samp = jax.random.split(key, 2)
	# Type checker expects tuple[int]; runtime accepts () for scalar batch
	init_state = hinton_init(k_init, model, cast(list[Any], free_blocks), ())  # type: ignore[arg-type]
	schedule = SamplingSchedule(n_warmup=20, n_samples=100, steps_per_sample=1)
	all_nodes_block = [Block(nodes)]
	samples = sample_states(k_samp, program, schedule, init_state, [], cast(list[Any], all_nodes_block))

	# Visualize magnetization over nodes
	spin = samples[0]  # [n_samples, n_nodes]
	mag = np.array(2*spin.astype(jnp.int32)-1).mean(axis=0)
	fig, ax = plt.subplots(1, 2, figsize=(9, 3))
	ax[0].imshow(np.array(spin).T, aspect='auto', cmap='Greys', vmin=0, vmax=1)
	ax[0].set_title('Samples (node × time)')
	ax[1].bar(range(len(mag)), mag)
	ax[1].set_title('Mean magnetization')
	_textbox(ax[1], 'Two-color blocked Gibbs\nMinimizes Python loops;\nmaximizes JAX parallelism.')
	_savefig(fig, outdir, 'core_block_gibbs')
	return {"mean_abs_mag": float(np.mean(np.abs(mag)))}


def run_core_heterogeneous_discrete(key, outdir: Path):
	# Heterogeneous PGM: one spin block + one categorical block coupled via discrete EBM utilities
	n_spins = 6
	K = 3  # categories
	spin_nodes = [SpinNode() for _ in range(n_spins)]
	cat_nodes = [CategoricalNode() for _ in range(n_spins)]  # one categorical per spin for simplicity
	block_spin = Block(spin_nodes)
	block_cat = Block(cat_nodes)

	# Factors: spin-only term and categorical-only term to keep simple, plus optional coupling via DiscreteEBMFactor
	# Here: use SpinEBMFactor (bias-like term) and CategoricalEBMFactor (preference over categories)
	key_w1, key_w2 = jax.random.split(key)
	spin_w = jax.random.normal(key_w1, (n_spins,)) * 0.3  # weights act as local fields
	cat_w = jnp.tile(jnp.linspace(0.2, -0.2, K)[None, :], (n_spins, 1))  # simple categorical preferences

	f_spin = SpinEBMFactor([block_spin], spin_w)
	f_cat = CategoricalEBMFactor([block_cat], cat_w)
	factors = [f_spin, f_cat]

	# Block Gibbs spec with two superblocks (spin, categorical)
	spec = BlockGibbsSpec([block_spin, block_cat], clamped_blocks=[])
	sp_samp = SpinGibbsConditional()
	cat_samp = CategoricalGibbsConditional(n_categories=K)
	program = FactorSamplingProgram(spec, [sp_samp, cat_samp], factors, [])

	# Sample
	init_spin = (jax.random.bernoulli(key, p=0.5, shape=(len(block_spin),)).astype(jnp.bool_),)
	init_cat = (jax.random.randint(key, (len(block_cat),), 0, K).astype(jnp.uint8),)
	init_state = [init_spin[0], init_cat[0]]
	schedule = SamplingSchedule(n_warmup=20, n_samples=100, steps_per_sample=1)
	samples = sample_states_core(key, program, schedule, init_state, [], [block_spin, block_cat])

	spin_samps = np.array(samples[0])
	cat_samps = np.array(samples[1])
	spin_mag = (2*spin_samps-1).mean(axis=0)
	cat_hist = np.stack([(cat_samps==c).mean(axis=0) for c in range(K)], axis=1)

	fig, ax = plt.subplots(1, 3, figsize=(12, 3))
	ax[0].imshow(spin_samps.T, aspect='auto', cmap='Greys', vmin=0, vmax=1)
	ax[0].set_title('Spin samples')
	ax[1].bar(range(n_spins), spin_mag)
	ax[1].set_title('Spin magnetization')
	im = ax[2].imshow(cat_hist.T, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
	ax[2].set_title('Categorical probs (K)')
	fig.colorbar(im, ax=ax[2])
	_textbox(ax[2], 'Heterogeneous graph (Spin+Categorical)\nDiscrete EBM utilities +\nfactorized interactions → global state')
	_savefig(fig, outdir, 'core_heterogeneous')
	return {"spin_mean_abs_mag": float(np.mean(np.abs(spin_mag)))}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--results", type=str, default="results")
	parser.add_argument("--measure-power", action="store_true")
	args = parser.parse_args()

	# CPU-only works out of the box; if using jax-metal, user can pin versions.
	jax.config.update("jax_enable_x64", True)

	base = Path(args.results)
	stamp = time.strftime("prototypes_%Y%m%d_%H%M%S")
	outdir = base / stamp
	outdir.mkdir(parents=True, exist_ok=True)

	print(f"Saving figures to {outdir}")
	key = jax.random.key(args.seed)

	# Load optimal parameters from discovery if available
	discovery_dir = base / "discovery"
	optimal_params = {}
	if discovery_dir.exists():
		optimal_params = load_optimal_params(discovery_dir)
		if optimal_params:
			print(f"Loaded optimal params from discovery: {optimal_params}")

	artifacts = {}
	cfg = EnergyConfig()
	ps = PowerSession() if args.measure_power else None
	# Run all 10; EFSM included with simplified path
	for name, fn in [
		("core_block_gibbs", run_core_block_gibbs_two_color),
		("core_hetero", run_core_heterogeneous_discrete),
		("srsl", lambda k, o: run_srsl(k, o, optimal_params.get("srsl_beta"))),
		("taps", run_taps),
		("bpp", run_bpp),
		("efsm", run_efsm),
		("tbro", run_tbro),
		("labi", lambda k, o: run_labi(k, o, optimal_params.get("labi_threshold"))),
		("tcf", lambda k, o: run_tcf(k, o, optimal_params.get("tcf_perturb"))),
		("ppts", run_ppts),
		("tvs", run_tvs),
		("ref", run_ref),
	]:
		try:
			k, key = key, jax.random.split(key, 2)[1]
			if ps: ps.start()
			kpis = fn(k, outdir)
			real_j = ps.stop() if ps else None
			# Attach simple energy summaries where possible
			if name in ("srsl", "taps", "bpp", "efsm", "tbro", "labi", "tcf", "ppts", "tvs", "ref"):
				n_nodes = 64 if name in ("ppts", "ref") else 32
				sampling_j = estimate_sampling_energy_j(FAST_SCHEDULE.n_samples, n_nodes, cfg)
				tokens = 32 * 8
				energy_summary = summarize_energy(tokens, sampling_j, alerts=1, cfg=cfg)
				if real_j is not None:
					energy_summary["measured_joules"] = real_j
				kpis.update({f"energy_{k}": v for k, v in energy_summary.items()})
			artifacts[name] = kpis
			print(f"✓ {name} done")
		except Exception as e:  # noqa: BLE001
			artifacts[name] = {"error": str(e)}
			print(f"✗ {name} error: {e}")

	# Save a small summary file
	summary = outdir / "summary.txt"
	with open(summary, "w") as f:
		for name, kpis in artifacts.items():
			f.write(f"[{name}]\n")
			for k, v in kpis.items():
				f.write(f"  {k}: {v}\n")
			f.write("\n")

	print(f"\nSummary written to {summary}")


if __name__ == "__main__":
	main()
