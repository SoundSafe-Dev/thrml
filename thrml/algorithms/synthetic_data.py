"""Synthetic data generators and a lightweight harness for thermal algorithms.

This module provides simple, parameterizable synthetic data sources for each
algorithm, and a small benchmark loop to automate KPI collection.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp

from thrml import SamplingSchedule
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


# -----------------------------
# Synthetic data generators
# -----------------------------

def gen_srsl_window(key: jax.Array, n: int = 32, snr: float = -5.0) -> jnp.ndarray:
	freq = 4.0
	t = jnp.linspace(0, 1.0, n)
	sig = jnp.sin(2 * jnp.pi * freq * t)
	noise = jax.random.normal(key, (n,))
	alpha = 10 ** (snr / 20.0)
	return jnp.clip(alpha * sig + noise, -3.0, 3.0)


def gen_taps_inputs(key: jax.Array, n_sensors: int = 8) -> Tuple[jnp.ndarray, jnp.ndarray]:
	base = jax.random.uniform(key, (n_sensors,))
	# Inject two spikes
	idx = jax.random.choice(key, n_sensors, (2,), replace=False)
	base = base.at[idx].set(jnp.array([0.9, 0.85]))
	costs = jnp.array([[0.0, 0.5, 2.0] for _ in range(n_sensors)])
	return base, costs


def gen_bpp_inputs(key: jax.Array, n_tactics: int = 5) -> Tuple[int, jnp.ndarray]:
	threat_level = int(jax.random.choice(key, 3))
	tactic_scores = jax.random.uniform(key, (n_tactics,))
	return threat_level, tactic_scores


def gen_efsm_windows(key: jax.Array, n_features: int = 64, n_clean: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
	k1, k2 = jax.random.split(key)
	clean = jax.random.normal(k1, (n_clean, n_features)) * 0.5
	# Normal vs anomalous
	normal = jax.random.normal(k2, (n_features,)) * 0.5
	# Outlier shift
	anom = normal + 5.0 + jax.random.normal(k2, (n_features,)) * 2.0
	return clean, normal, anom


def gen_tbro_inputs(key: jax.Array, n_sites: int = 10) -> jnp.ndarray:
	risk = jax.random.uniform(key, (n_sites,))
	# Ensure two surges
	idx = jax.random.choice(key, n_sites, (2,), replace=False)
	risk = risk.at[idx].set(jnp.array([0.95, 0.9]))
	return risk


def gen_labi_likelihood(key: jax.Array, n_variables: int = 16) -> jnp.ndarray:
	return jax.random.uniform(key, (n_variables,), minval=-0.5, maxval=0.5)


def gen_tcf_modalities(key: jax.Array, n_modalities: int = 4) -> jnp.ndarray:
	return jax.random.uniform(key, (n_modalities,))


def gen_ppts_phases(key: jax.Array, n_sensors: int = 6) -> jnp.ndarray:
	return jax.random.uniform(key, (n_sensors,), minval=0.0, maxval=2 * jnp.pi)


def gen_tvs_stream(key: jax.Array, t: int = 100, f: int = 64) -> jnp.ndarray:
	return jax.random.normal(key, (t, f))


def gen_ref_stream(key: jax.Array, t: int = 50, n_in: int = 16) -> jnp.ndarray:
	return jax.random.normal(key, (t, n_in))


# -----------------------------
# Lightweight benchmark harness
# -----------------------------

@dataclass
class BenchConfig:
	steps: int = 5
	warmup: int = 50
	samples: int = 100
	steps_per_sample: int = 2
	seed: int = 0

	def schedule(self) -> SamplingSchedule:
		return SamplingSchedule(
			n_warmup=self.warmup, n_samples=self.samples, steps_per_sample=self.steps_per_sample
		)


def run_srsl_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	algo = StochasticResonanceSignalLifter(signal_window_size=32, key=key)
	k = key
	for _ in range(cfg.steps):
		k, k1 = jax.random.split(k)
		features = gen_srsl_window(k1, 32)
		algo.forward(k1, features, cfg.schedule())
	return algo.get_kpis()


def run_taps_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	threat, costs = gen_taps_inputs(key)
	algo = ThermodynamicActivePerceptionScheduling(n_sensors=8, n_bitrate_levels=3, sensor_costs=costs, key=key)
	algo.forward(key, threat, cfg.schedule())
	return algo.get_kpis()


def run_bpp_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	lvl, tactics = gen_bpp_inputs(key)
	algo = BoltzmannPolicyPlanner(key=key)
	algo.forward(key, lvl, tactics, cfg.schedule())
	return algo.get_kpis()


def run_efsm_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	clean, normal, anom = gen_efsm_windows(key)
	algo = EnergyFingerprintedSceneMemory(n_features=64, key=key)
	algo.fit_baseline(key, clean, cfg.schedule())
	algo.forward(key, normal, cfg.schedule())
	algo.forward(key, anom, cfg.schedule())
	return algo.get_kpis()


def run_tbro_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	risk = gen_tbro_inputs(key)
	algo = ThermalBanditResourceOrchestrator(n_sites=10, key=key)
	algo.forward(key, risk, cfg.schedule())
	return algo.get_kpis()


def run_labi_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	lh = gen_labi_likelihood(key)
	algo = LandauerAwareBayesianInference(n_variables=16, key=key)
	algo.forward(key, lh, cfg.schedule())
	return algo.get_kpis()


def run_tcf_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	mods = gen_tcf_modalities(key)
	algo = ThermodynamicCausalFusion(key=key)
	algo.discover_causal_structure(key, mods, schedule=cfg.schedule())
	algo.forward(key, mods, schedule=cfg.schedule())
	return algo.get_kpis()


def run_ppts_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	ph = gen_ppts_phases(key)
	algo = ProbabilisticPhaseTimeSync(key=key)
	algo.forward(key, ph, cfg.schedule())
	return algo.get_kpis()


def run_tvs_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	stream = gen_tvs_stream(key)
	algo = ThermoVerifiableSensing(key=key)
	wm, _ = algo.forward(key, stream, mode="watermark", schedule=cfg.schedule())
	algo.verify_stream(wm)
	return algo.get_kpis()


def run_ref_loop(cfg: BenchConfig) -> Dict[str, float]:
	key = jax.random.key(cfg.seed)
	stream = gen_ref_stream(key)
	algo = ReservoirEBMFrontEnd(key=key)
	algo.forward(key, stream, cfg.schedule())
	return algo.get_kpis()


HARNESS: Dict[str, Callable[[BenchConfig], Dict[str, float]]] = {
	"SRSL": run_srsl_loop,
	"TAPS": run_taps_loop,
	"BPP": run_bpp_loop,
	# "EFSM": run_efsm_loop,  # Temporarily disabled due to shape issues
	"TBRO": run_tbro_loop,
	"LABI": run_labi_loop,
	"TCF": run_tcf_loop,
	"PPTS": run_ppts_loop,
	"TVS": run_tvs_loop,
	"REF": run_ref_loop,
}


def run_all(cfg: BenchConfig) -> Dict[str, Dict[str, float]]:
	results: Dict[str, Dict[str, float]] = {}
	for name, fn in HARNESS.items():
		results[name] = fn(cfg)
	return results
