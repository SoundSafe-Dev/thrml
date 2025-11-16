"""Generate JSON payloads for the ThrML Next.js + Three.js demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp

from thrml.algorithms.bpp import BoltzmannPolicyPlanner
from thrml.algorithms.srsl import StochasticResonanceSignalLifter
from thrml.algorithms.taps import ThermodynamicActivePerceptionScheduling
from thrml.algorithms.synthetic_data import (
    BenchConfig,
    gen_bpp_inputs,
    gen_srsl_window,
    gen_taps_inputs,
)

DEFAULT_OUTPUT = Path("examples/nextjs-demo/public/data/algorithm_metrics.json")


def _vector_to_points(
    vector: jnp.ndarray,
    base_z: float,
    label: str,
    scale: float = 1.0,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for idx, value in enumerate(jnp.asarray(vector).tolist()):
        x_coord = (idx - len(vector) / 2) * 0.2
        y_coord = float(value) * scale
        z_coord = base_z + jnp.sin(jnp.array(idx / 3.0)) * 0.75
        points.append(
            {
                "x": float(x_coord),
                "y": y_coord,
                "z": float(z_coord),
                "label": label,
                "metrics": {"index": idx, "value": y_coord},
            }
        )
    return points


def _taps_points(actions: jnp.ndarray) -> List[Dict[str, Any]]:
    sensors, levels = actions.shape
    points: List[Dict[str, Any]] = []
    for sensor in range(sensors):
        for level in range(levels):
            probability = float(actions[sensor, level])
            points.append(
                {
                    "x": sensor - sensors / 2,
                    "y": level * 0.85,
                    "z": probability * 4.0,
                    "label": f"sensor_{sensor}_level_{level}",
                    "metrics": {
                        "sensor": sensor,
                        "level": level,
                        "probability": probability,
                    },
                }
            )
    return points


def _bpp_points(probs: jnp.ndarray, threat_level: int) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for idx, probability in enumerate(jnp.asarray(probs).tolist()):
        angle = idx / max(len(probs), 1) * jnp.pi * 2.0
        radius = 2.2 + probability * 2.5
        x_coord = float(radius * jnp.cos(angle))
        y_coord = float(probability * 3.5)
        z_coord = float(radius * jnp.sin(angle))
        points.append(
            {
                "x": x_coord,
                "y": y_coord,
                "z": z_coord,
                "label": f"action_{idx}",
                "metrics": {
                    "action_index": idx,
                    "probability": float(probability),
                    "threat_level": threat_level,
                },
            }
        )
    return points


def build_payload(output_path: Path) -> Dict[str, Any]:
    cfg = BenchConfig(steps=1, warmup=2, samples=4, steps_per_sample=1, seed=7)

    srsl_key = jax.random.key(cfg.seed)
    srsl_features = gen_srsl_window(srsl_key, 32)
    srsl_algo = StochasticResonanceSignalLifter(
        signal_window_size=32, n_beta_steps=4, key=srsl_key
    )
    srsl_amplified, srsl_forward_kpis = srsl_algo.forward(
        srsl_key, srsl_features, cfg.schedule()
    )
    srsl_kpis = {**srsl_forward_kpis, **srsl_algo.get_kpis()}

    taps_key = jax.random.split(srsl_key, 2)[1]
    threat_scores, sensor_costs = gen_taps_inputs(taps_key)
    taps_algo = ThermodynamicActivePerceptionScheduling(
        n_sensors=threat_scores.shape[0],
        n_bitrate_levels=sensor_costs.shape[1],
        sensor_costs=sensor_costs,
        key=taps_key,
    )
    taps_actions, taps_forward_kpis = taps_algo.forward(
        taps_key, threat_scores, cfg.schedule()
    )
    taps_kpis = {**taps_forward_kpis, **taps_algo.get_kpis()}

    bpp_key = jax.random.split(taps_key, 2)[1]
    threat_level, tactic_scores = gen_bpp_inputs(bpp_key)
    bpp_algo = BoltzmannPolicyPlanner(key=bpp_key)
    bpp_probs, bpp_forward_kpis = bpp_algo.forward(
        bpp_key, threat_level, tactic_scores, cfg.schedule()
    )
    bpp_kpis = {**bpp_forward_kpis, **bpp_algo.get_kpis()}

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bench_config": asdict(cfg),
        "algorithms": [
            {
                "id": "srsl",
                "name": "Stochastic-Resonance Signal Lifter",
                "summary": "Amplifies sub-threshold signals using engineered thermal noise sweeps.",
                "kpis": srsl_kpis,
                "points": _vector_to_points(srsl_amplified, base_z=0.0, label="srsl", scale=3.0),
            },
            {
                "id": "taps",
                "name": "Thermodynamic Active Perception Scheduling",
                "summary": "Landauer-aware sensor scheduling that balances threat coverage with joule budgets.",
                "kpis": taps_kpis,
                "points": _taps_points(taps_actions),
            },
            {
                "id": "bpp",
                "name": "Boltzmann Policy Planner",
                "summary": "Thermal policy sampler that fuses detection signals into calibrated escalations.",
                "kpis": bpp_kpis,
                "points": _bpp_points(bpp_probs, threat_level),
            },
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for algorithm_metrics.json",
    )
    args = parser.parse_args()

    payload = build_payload(args.output)
    print(
        f"Wrote {len(payload['algorithms'])} algorithm payloads to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
