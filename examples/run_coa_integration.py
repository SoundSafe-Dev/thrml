#!/usr/bin/env python3
"""Demo: Integrate THRML outputs with COA/ROE Engine.

- Runs a fast pass of BPP/TAPS/TCF/LABI to get KPIs
- Builds a threat_event dict and context via adapter
- Invokes COAROEEngine.generate_response_with_rule_correlations

Usage:
  python examples/run_coa_integration.py --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import importlib.util
import types
from datetime import datetime

import jax
import jax.numpy as jnp

from thrml import SamplingSchedule
from thrml.algorithms import (
	BoltzmannPolicyPlanner,
	ThermodynamicActivePerceptionScheduling,
	ThermodynamicCausalFusion,
	LandauerAwareBayesianInference,
)
from thrml.algorithms.coa_roe_adapter import build_threat_event, attach_thrml_context

# Try package import; fallback to direct file load from extracted zip
try:
	from thrml.coa_roe.coa_roe_engine import COAROEEngine  # type: ignore
except Exception:
	# Fallback lightweight stub if external engine cannot be imported
	from dataclasses import dataclass
	from typing import List, Dict, Any
	@dataclass
	class _Action:
		action_type: str
		description: str
	@dataclass
	class _Response:
		protocol: str
		actions: List[_Action]
	class COAROEEngine:  # type: ignore
		def __init__(self, *_args, **_kwargs):
			pass
		async def start(self):
			return
		async def stop(self):
			return
		async def generate_response_with_rule_correlations(self, threat_event: Dict[str, Any]):
			level = threat_event.get("threat_level", "medium")
			act = "coordinate" if level in ("high", "critical") else "monitor"
			return _Response(protocol=threat_event.get("threat_type", "unknown"), actions=[_Action(act, f"Auto-{act} by stub engine")])

from thrml.coa_roe.scenarios import ComprehensiveCaseScenarios, SectorType


FAST = SamplingSchedule(n_warmup=10, n_samples=30, steps_per_sample=1)


def run_thrml_pass(key):
	# BPP
	bpp = BoltzmannPolicyPlanner(risk_temperature=1.0, key=key)
	threat_level = 2
	tactic_scores = jnp.array([0.1, 0.2, 0.8, 0.5, 0.1])
	_, bpp_kpis = bpp.forward(key, threat_level, tactic_scores, FAST)
	# TAPS
	taps = ThermodynamicActivePerceptionScheduling(n_sensors=8, n_bitrate_levels=3, key=key)
	threat_scores = jax.random.uniform(key, (8,))
	_, taps_kpis = taps.forward(key, threat_scores, FAST)
	# TCF
	tcf = ThermodynamicCausalFusion(n_modalities=4, key=key)
	mods = jax.random.uniform(key, (4,))
	graph, tcf_disc = tcf.discover_causal_structure(key, mods, schedule=FAST)
	_, tcf_fuse = tcf.forward(key, mods, schedule=FAST)
	# LABI
	labi = LandauerAwareBayesianInference(n_variables=16, energy_threshold=1e-18, key=key)
	likelihood = jax.random.uniform(key, (16,), minval=-0.2, maxval=0.2)
	(_, _), labi_kpis = labi.forward(key, likelihood, FAST)
	return bpp_kpis, taps_kpis, {**tcf_disc, **tcf_fuse}, labi_kpis


async def main_async(seed: int):
	key = jax.random.key(seed)
	bpp_kpis, taps_kpis, tcf_kpis, labi_kpis = run_thrml_pass(key)

	# Build threat event and context (example targets gunshot response)
	threat_level = "medium"
	threat_type = "physical_security"
	# Optional: use scenario matcher to refine threat_type/level
	scenarios = ComprehensiveCaseScenarios()
	detections = [
		{"detection_type": "gunshot_detection", "confidence": 0.9, "timestamp": datetime.now().isoformat()},
		{"detection_type": "firearm_detection", "confidence": 0.85, "timestamp": datetime.now().isoformat()},
	]
	matches = scenarios.match(detections, SectorType.SCHOOLS)
	if matches:
		m = matches[0]
		threat_type = m.get("scenario_type", threat_type)
		if m.get("threat_level") == "critical":
			threat_level = "high"

	threat = build_threat_event(
		threat_type=threat_type,
		threat_level=threat_level,
		confidence=float(tcf_kpis.get("threat_score", 0.6)),
		description="THRML fused threat; BPP action candidate included",
		metadata={
			"bpp_selected_action": bpp_kpis.get("selected_action"),
			"bpp_confidence": bpp_kpis.get("action_confidence"),
			"taps_energy": taps_kpis.get("total_energy_joules_per_sec"),
			"tcf_causal_edges": tcf_kpis.get("n_causal_edges"),
			"labi_skip_rate": labi_kpis.get("skip_rate"),
		},
		context=attach_thrml_context(bpp=bpp_kpis, taps=taps_kpis, tcf=tcf_kpis, labi=labi_kpis),
	)

	engine = COAROEEngine()
	await engine.start()
	response = await engine.generate_response_with_rule_correlations(threat)
	await engine.stop()

	print("Generated response:")
	if response is None:
		print("  None (no protocol found)")
		return
	print(f"  Protocol: {getattr(response, 'protocol', None)}")
	print(f"  Actions: {len(getattr(response, 'actions', []))}")
	for a in getattr(response, "actions", [])[:5]:
		action_type = getattr(a.action_type, "value", a.action_type)
		print(f"   - {action_type}: {a.description}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	asyncio.run(main_async(args.seed))


if __name__ == "__main__":
	main()
