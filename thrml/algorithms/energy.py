"""Energy and token analytics helpers.

This module provides simple, configurable energy proxies to estimate
Joules per token and Joules per alert during sampling-based inference.

These are proxies suitable for comparative benchmarking on the same host;
for precise metering, integrate with power sensors or job schedulers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EnergyConfig:
	# Unit energy proxies; tune to match your device/power profile.
	energy_per_sample_per_node_j: float = 1e-12  # J per (sample × node)
	energy_per_token_j: float = 5e-10           # J per token transform (feature op)
	energy_per_alert_overhead_j: float = 2e-6   # J per alert (notification/logging)


def estimate_sampling_energy_j(n_samples: int, n_nodes: int, cfg: EnergyConfig) -> float:
	return float(n_samples * n_nodes) * cfg.energy_per_sample_per_node_j


def estimate_tokens_from_audio(n_frames: int, n_bins: int) -> int:
	# Treat each feature bin per frame as a token analogue
	return int(n_frames * n_bins)


def estimate_joules_per_token(total_energy_j: float, num_tokens: int) -> float:
	return float("inf") if num_tokens <= 0 else total_energy_j / float(num_tokens)


def summarize_energy(tokens: int, sampling_j: float, alerts: int, cfg: EnergyConfig) -> Dict[str, float]:
	alert_j = alerts * cfg.energy_per_alert_overhead_j
	total_j = sampling_j + alert_j + tokens * cfg.energy_per_token_j
	return {
		"tokens": float(tokens),
		"alerts": float(alerts),
		"sampling_joules": sampling_j,
		"alert_overhead_joules": alert_j,
		"token_processing_joules": tokens * cfg.energy_per_token_j,
		"total_joules": total_j,
		"joules_per_token": estimate_joules_per_token(total_j, tokens),
		"joules_per_alert": float("inf") if alerts <= 0 else total_j / float(alerts),
	}
