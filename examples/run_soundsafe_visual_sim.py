#!/usr/bin/env python3
"""SoundSafe thermodynamic probability visualization.

This script builds a SoundSafe.ai-inspired visual simulation that shows how
thermodynamic Gibbs sampling fuses multi-modal audio/video/sensor intelligence
into superior threat probabilities versus a naive residual sum, while tracking
correlations, resilience, power usage, and Unified Threat Engine (UTE) case
matches that feed directly into COA/ROE protocols.

Usage:
    python examples/run_soundsafe_visual_sim.py --seed 7 --zones 9 --steps 48
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required for the SoundSafe visual simulation. "
        "Install it with `pip install matplotlib`."
    ) from exc
import numpy as np

from thrml import SamplingSchedule
from thrml.algorithms import ThermodynamicCausalFusion
from thrml.algorithms.coa_roe_adapter import build_threat_event
from thrml.algorithms.energy import (
    EnergyConfig,
    estimate_sampling_energy_j,
    summarize_energy,
)
from thrml.coa_roe.scenarios import ComprehensiveCaseScenarios, SectorType


Modalities = Tuple[str, ...]
THREAT_THRESHOLD = 0.4
DOMAIN_ORDER = ("Audio", "Video", "Sensor")
DOMAIN_COLORS = {
    "Audio": "tab:red",
    "Video": "tab:green",
    "Sensor": "tab:purple",
}
MODALITY_DETECTION_CONFIG = {
    "Acoustic Aggression": {
        "detection_type": "gunshot_detection",
        "domain": "Audio",
        "threshold": 0.08,
        "scale": 3.0,
    },
    "Seismic Impact": {
        "detection_type": "firearm_detection",
        "domain": "Sensor",
        "threshold": 0.06,
        "scale": 2.8,
    },
    "RFID Spoof": {
        "detection_type": "rfid_spoof_detection",
        "domain": "Sensor",
        "threshold": 0.05,
        "scale": 2.2,
    },
    "Thermal Motion": {
        "detection_type": "crowd_panic_detection",
        "domain": "Video",
        "threshold": 0.05,
        "scale": 2.5,
    },
}


@dataclass
class SimulationConfig:
    """User-configurable knobs for the visualization."""

    n_zones: int
    n_steps: int
    modalities: Modalities
    event_rate: float = 0.2
    dropout_rate: float = 0.1
    schedule: SamplingSchedule = field(
        default_factory=lambda: SamplingSchedule(
            n_warmup=16, n_samples=64, steps_per_sample=2
        )
    )
    energy_cfg: EnergyConfig = field(
        default_factory=lambda: EnergyConfig(
            energy_per_sample_per_node_j=8e-13,
            energy_per_token_j=6e-10,
            energy_per_alert_overhead_j=1.5e-6,
        )
    )
    baseline_energy_cfg: EnergyConfig = field(
        default_factory=lambda: EnergyConfig(
            energy_per_sample_per_node_j=2.8e-12,
            energy_per_token_j=2.8e-9,
            energy_per_alert_overhead_j=3.2e-6,
        )
    )
    baseline_token_scale: float = 2.2
    ute_sector: SectorType = SectorType.SCHOOLS


@dataclass
class SimulationState:
    sensor_field: np.ndarray  # (steps, zones, modalities)
    threat_scores: np.ndarray  # (steps, zones)
    fused_vectors: np.ndarray  # (steps, zones, modalities)
    dropout_masks: np.ndarray  # bool mask
    alert_counts: np.ndarray  # per step count of alerts across zones
    energy_joules: np.ndarray  # cumulative Extropic ThermoChip joules per step
    baseline_energy_joules: np.ndarray  # cumulative GPU-style joules per step
    event_flags: np.ndarray  # ground-truth threat activations
    naive_scores: np.ndarray  # simple heuristic fusion scores
    detection_metrics: Dict[str, "DetectionReport"]
    joules_per_detection: Dict[str, float]
    energy_savings_pct: float
    ute_summary: "UnifiedThreatSummary"


@dataclass
class UnifiedThreatSummary:
    scenario_type: str
    scenario_label: str
    threat_level: str
    confidence: float
    fused_confidence: float
    correlated_detections: List[str]
    correlated_domains: Dict[str, int]
    domain_timeline: Dict[str, np.ndarray]
    recommended_protocols: List[str]
    response_actions: List[str]
    description: str
    alerts_rationale: str
    threat_event: Dict[str, Any]


@dataclass
class DetectionReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: float
    false_positives: float
    predicted_positives: float


def _zone_grid_shape(n_zones: int) -> Tuple[int, int]:
    side = int(math.ceil(math.sqrt(n_zones)))
    rows = side
    cols = side if side * (side - 1) < n_zones else side
    return rows, cols


def simulate_sensor_field(
    key: jax.Array,
    cfg: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic SoundSafe sensor readings, dropouts, and truth."""

    n_steps = cfg.n_steps
    n_zones = cfg.n_zones
    n_modalities = len(cfg.modalities)

    base_key, event_key, dropout_key = jax.random.split(key, 3)
    base_field = 0.25 + 0.05 * jax.random.normal(
        base_key, (n_steps, n_zones, n_modalities)
    )
    # Randomly pick critical zones/events
    zone_importance = jax.random.uniform(event_key, (n_zones,))
    zone_importance = zone_importance / jnp.max(zone_importance)
    # Inject events with modality-specific signatures
    sensor_field = []
    event_flags: List[np.ndarray] = []
    signature = jnp.array([1.0, 0.7, 0.4, 0.9][:n_modalities])
    for t in range(n_steps):
        decay_val = float(jnp.exp(-0.02 * t))
        noise_key = jax.random.fold_in(base_key, t)
        noise = jax.random.normal(noise_key, (n_zones, n_modalities)) * 0.03
        frame = base_field[t] + noise
        active_zones = jax.random.bernoulli(
            jax.random.fold_in(event_key, t),
            p=min(0.5, cfg.event_rate + 0.1 * decay_val),
            shape=(n_zones,),
        )
        boost = (0.6 + 0.3 * zone_importance) * active_zones.astype(jnp.float32)
        frame = frame + boost[:, None] * signature
        sensor_field.append(frame)
        event_flags.append(np.array(active_zones, dtype=bool))
    sensor_field_arr = jnp.stack(sensor_field, axis=0)
    dropout_masks = jax.random.bernoulli(
        dropout_key, p=cfg.dropout_rate, shape=sensor_field_arr.shape
    )
    return (
        np.array(sensor_field_arr),
        np.array(dropout_masks, dtype=bool),
        np.stack(event_flags, axis=0),
    )


def _naive_multimodal_score(
    readings: jnp.ndarray, baseline: np.ndarray, failed: jnp.ndarray
) -> float:
    """Simple heuristic fusion via normalized residual averaging."""

    adjusted = np.where(np.array(failed, dtype=bool), baseline, np.array(readings))
    delta = adjusted - baseline
    normalized = np.mean(delta)
    score = 1.0 / (1.0 + np.exp(-6.0 * normalized))
    return float(np.clip(score, 0.0, 1.0))


def _safe_divide(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else num / denom


def _detection_report(scores: np.ndarray, truth: np.ndarray) -> DetectionReport:
    preds = scores > THREAT_THRESHOLD
    tp = float(np.sum(preds & truth))
    fp = float(np.sum(preds & (~truth)))
    fn = float(np.sum((~preds) & truth))
    tn = float(np.sum((~preds) & (~truth)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return DetectionReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        predicted_positives=float(np.sum(preds)),
    )


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1, values.shape[-1])
    corr = np.corrcoef(flattened, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def _build_detection_events(
    sensor_field: np.ndarray, modalities: Modalities
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, np.ndarray]]:
    """Convert sensor amplitudes into Unified Threat Engine detection events."""

    n_steps, n_zones, _ = sensor_field.shape
    warmup = max(1, min(6, n_steps))
    baseline = np.mean(sensor_field[:warmup], axis=0, keepdims=False)
    domain_totals = {domain: 0 for domain in DOMAIN_ORDER}
    domain_timeline = {
        domain: np.zeros((n_steps,), dtype=np.float32) for domain in DOMAIN_ORDER
    }
    events: List[Dict[str, Any]] = []
    now = datetime.now()
    for step in range(n_steps):
        timestamp = (now - timedelta(seconds=(n_steps - step))).isoformat()
        for zone_idx in range(n_zones):
            for mod_idx, modality in enumerate(modalities):
                config = MODALITY_DETECTION_CONFIG.get(modality)
                if config is None:
                    continue
                delta = float(sensor_field[step, zone_idx, mod_idx] - baseline[zone_idx, mod_idx])
                if delta <= config["threshold"]:
                    continue
                confidence = float(np.clip(0.55 + delta * config["scale"], 0.0, 1.0))
                event = {
                    "detection_type": config["detection_type"],
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "zone": int(zone_idx),
                    "domain": config["domain"],
                    "step": int(step),
                }
                events.append(event)
                domain_totals[config["domain"]] += 1
                domain_timeline[config["domain"]][step] += 1
    return events, domain_totals, domain_timeline


def _select_case(
    events: List[Dict[str, Any]], sector: SectorType
) -> Tuple[Dict[str, Any], Any]:
    """Return the top Unified Threat Engine case match and its scenario object."""

    scenario_catalog = ComprehensiveCaseScenarios()
    matches = scenario_catalog.match(events, sector)
    selected: Dict[str, Any] = {}
    scenario_obj = None
    if matches:
        selected = max(matches, key=lambda item: item.get("confidence", 0.0))
        scenario_obj = next(
            (
                scenario
                for scenario in scenario_catalog.scenarios.values()
                if scenario.scenario_id == selected.get("scenario_id")
            ),
            None,
        )
    return selected, scenario_obj


def _evaluate_unified_threat_case(
    cfg: SimulationConfig,
    sensor_field: np.ndarray,
    threat_scores: np.ndarray,
    event_flags: np.ndarray,
) -> UnifiedThreatSummary:
    """Project detections into Unified Threat Engine + COA/ROE summary."""

    events, domain_totals, domain_timeline = _build_detection_events(
        sensor_field, cfg.modalities
    )
    match, scenario_obj = _select_case(events, cfg.ute_sector)
    scenario_type = match.get("scenario_type", "no_match") or "no_match"
    threat_level = match.get("threat_level", "low")
    scenario_label = (
        f"{scenario_type.replace('_', ' ').title()} ({threat_level.title()})"
        if match
        else "No Unified Threat Case Matched"
    )
    confidence = float(match.get("confidence", 0.0))
    recommended_protocols = list(match.get("response_protocols", []))
    response_actions: List[str] = []
    description = "SoundSafe telemetry did not align with a catalogued case."
    if scenario_obj is not None:
        description = scenario_obj.description or description
        for protocol in scenario_obj.response_protocols:
            response_actions.extend(protocol.actions)
    if not response_actions:
        response_actions = ["Continue triage with on-site team", "Validate sensor health"]
    fused_conf = float(np.max(threat_scores))
    correlated_detections = sorted({e["detection_type"] for e in events})
    alerts_rationale = (
        "UTE correlated "
        + (", ".join(correlated_detections) if correlated_detections else "no threat detections")
        + f" across audio/video/sensor channels in {len(events)} total events."
    )
    context = {
        "tcf_max_zone": float(np.max(threat_scores)),
        "alerts": int(np.sum(threat_scores > THREAT_THRESHOLD)),
        "truth_events": int(np.sum(event_flags)),
        "domain_totals": domain_totals,
    }
    threat_event = build_threat_event(
        threat_type=scenario_type,
        threat_level=threat_level,
        confidence=max(confidence, fused_conf),
        description=description,
        metadata={
            "scenario_label": scenario_label,
            "detections": correlated_detections,
        },
        context=context,
    )
    return UnifiedThreatSummary(
        scenario_type=scenario_type,
        scenario_label=scenario_label,
        threat_level=threat_level,
        confidence=confidence,
        fused_confidence=fused_conf,
        correlated_detections=correlated_detections,
        correlated_domains=domain_totals,
        domain_timeline=domain_timeline,
        recommended_protocols=recommended_protocols,
        response_actions=response_actions,
        description=description,
        alerts_rationale=alerts_rationale,
        threat_event=threat_event,
    )


def run_visual_simulation(
    key: jax.Array,
    cfg: SimulationConfig,
) -> SimulationState:
    n_modalities = len(cfg.modalities)
    sensor_field, dropout_masks, event_flags = simulate_sensor_field(key, cfg)
    threat_scores = np.zeros((cfg.n_steps, cfg.n_zones), dtype=np.float32)
    fused_vectors = np.zeros((cfg.n_steps, cfg.n_zones, n_modalities), dtype=np.float32)
    alert_counts = np.zeros((cfg.n_steps,), dtype=np.int32)
    energy_consumed = np.zeros((cfg.n_steps,), dtype=np.float64)
    baseline_energy_consumed = np.zeros((cfg.n_steps,), dtype=np.float64)
    naive_scores = np.zeros_like(threat_scores)

    # Build TCF per zone so they can adapt local context
    init_keys = jax.random.split(key, cfg.n_zones + cfg.n_steps * cfg.n_zones)
    tcf_models: List[ThermodynamicCausalFusion] = []
    zone_baselines = np.mean(
        sensor_field[: min(6, cfg.n_steps), :, :], axis=0, keepdims=False
    )
    for zone_idx in range(cfg.n_zones):
        tcf_models.append(
            ThermodynamicCausalFusion(
                n_modalities=n_modalities,
                n_nodes_per_modality=1,
                key=init_keys[zone_idx],
            )
        )
        tcf_models[zone_idx].discover_causal_structure(
            init_keys[zone_idx], jnp.asarray(zone_baselines[zone_idx]), perturbation_strength=0.15
        )

    schedule = cfg.schedule
    sample_energy = estimate_sampling_energy_j(
        n_samples=schedule.n_samples,
        n_nodes=n_modalities,
        cfg=cfg.energy_cfg,
    )
    baseline_sample_energy = estimate_sampling_energy_j(
        n_samples=schedule.n_samples,
        n_nodes=n_modalities,
        cfg=cfg.baseline_energy_cfg,
    )
    n_tokens = 64 * n_modalities
    tokens_per_step = n_tokens * cfg.n_zones
    baseline_tokens_per_step = int(tokens_per_step * cfg.baseline_token_scale)

    cursor = cfg.n_zones
    for step in range(cfg.n_steps):
        for zone_idx in range(cfg.n_zones):
            zone_key = init_keys[cursor]
            cursor += 1
            readings = jnp.asarray(sensor_field[step, zone_idx, :])
            failed = jnp.asarray(dropout_masks[step, zone_idx, :])
            fused, kpis = tcf_models[zone_idx].forward(
                zone_key,
                readings,
                failed_modalities=failed,
                schedule=schedule,
            )
            threat_scores[step, zone_idx] = kpis["threat_score"]
            fused_vectors[step, zone_idx, :] = np.array(fused)
            naive_scores[step, zone_idx] = _naive_multimodal_score(
                readings, zone_baselines[zone_idx], failed
            )
        alerts_this_step = int(np.sum(threat_scores[step] > 0.35))
        alert_counts[step] = alerts_this_step
        energy_summary = summarize_energy(
            tokens=tokens_per_step,
            sampling_j=sample_energy * cfg.n_zones,
            alerts=alerts_this_step,
            cfg=cfg.energy_cfg,
        )
        baseline_energy_summary = summarize_energy(
            tokens=baseline_tokens_per_step,
            sampling_j=baseline_sample_energy * cfg.n_zones,
            alerts=alerts_this_step,
            cfg=cfg.baseline_energy_cfg,
        )
        energy_consumed[step] = energy_summary["total_joules"]
        baseline_energy_consumed[step] = baseline_energy_summary["total_joules"]

    tcf_report = _detection_report(threat_scores, event_flags)
    naive_report = _detection_report(naive_scores, event_flags)
    thermo_total = float(np.sum(energy_consumed))
    baseline_total = float(np.sum(baseline_energy_consumed))
    joules_per_detection = {
        "thermo": _safe_divide(thermo_total, tcf_report.true_positives),
        "baseline": _safe_divide(baseline_total, naive_report.true_positives),
    }
    energy_savings_pct = 0.0
    if baseline_total > 0:
        energy_savings_pct = 100.0 * (1.0 - (thermo_total / baseline_total))

    ute_summary = _evaluate_unified_threat_case(
        cfg, sensor_field, threat_scores, event_flags
    )

    return SimulationState(
        sensor_field=sensor_field,
        threat_scores=threat_scores,
        fused_vectors=fused_vectors,
        dropout_masks=dropout_masks,
        alert_counts=alert_counts,
        energy_joules=np.cumsum(energy_consumed),
        baseline_energy_joules=np.cumsum(baseline_energy_consumed),
        event_flags=event_flags,
        naive_scores=naive_scores,
        detection_metrics={"tcf": tcf_report, "naive": naive_report},
        joules_per_detection=joules_per_detection,
        energy_savings_pct=energy_savings_pct,
        ute_summary=ute_summary,
    )


def _format_zone_scores(threat_scores: np.ndarray) -> List[int]:
    return np.argsort(threat_scores)[::-1].tolist()


def plot_results(
    cfg: SimulationConfig,
    state: SimulationState,
    modalities: Sequence[str],
    out_path: Path,
) -> None:
    rows, cols = _zone_grid_shape(cfg.n_zones)
    latest_scores = state.threat_scores[-1]
    heatmap = np.full((rows * cols,), -1.0)
    heatmap[: cfg.n_zones] = latest_scores
    heatmap = heatmap.reshape(rows, cols)

    top_zones = _format_zone_scores(latest_scores)[: min(4, cfg.n_zones)]
    time_axis = np.arange(cfg.n_steps)
    ute = state.ute_summary

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.35, height_ratios=[1, 1, 1, 0.7])

    ax_heat = fig.add_subplot(gs[0, 0])
    im = ax_heat.imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    for idx in range(cfg.n_zones):
        r, c = divmod(idx, cols)
        ax_heat.text(c, r, f"Z{idx+1}\n{latest_scores[idx]:.2f}",
                     ha="center", va="center", color="white", fontsize=9)
    ax_heat.set_title("SoundSafe Thermodynamic Threat Map", fontweight="bold")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="Threat Probability")

    ax_traces = fig.add_subplot(gs[0, 1])
    for zone_idx in top_zones:
        label = f"Zone {zone_idx + 1}"
        ax_traces.plot(time_axis, state.threat_scores[:, zone_idx], label=label, lw=2)
    ax_traces.set_xlabel("Time Step")
    ax_traces.set_ylabel("Thermodynamic Threat Score")
    ax_traces.set_ylim(0, 1)
    ax_traces.grid(True, alpha=0.3)
    ax_traces.legend(loc="upper left")
    ax_traces.set_title("Probabilistic Relaxation Trajectories")

    ax_compare = fig.add_subplot(gs[0, 2])
    metrics = state.detection_metrics
    categories = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(categories))
    width = 0.35
    naive_values = [getattr(metrics["naive"], cat) for cat in categories]
    tcf_values = [getattr(metrics["tcf"], cat) for cat in categories]
    ax_compare.bar(x - width / 2, naive_values, width, label="Naive sum", alpha=0.6)
    ax_compare.bar(x + width / 2, tcf_values, width, label="Thermodynamic", alpha=0.8)
    ax_compare.set_xticks(x, [c.title() for c in categories])
    ax_compare.set_ylim(0, 1.05)
    ax_compare.set_ylabel("Score")
    ax_compare.set_title("Threat Detection Quality")
    ax_compare.legend()
    for i, value in enumerate(tcf_values):
        ax_compare.text(x[i] + width / 2, value + 0.02, f"{value:.2f}", ha="center")

    ax_power = fig.add_subplot(gs[0, 3])
    energy_points_mj = np.array(
        [state.baseline_energy_joules[-1], state.energy_joules[-1]]
    ) * 1e3
    f1_values = [metrics["naive"].f1, metrics["tcf"].f1]
    power_labels = ["GPU residual", "Extropic ThermoChip"]
    power_colors = ["tab:orange", "tab:blue"]
    for idx, label in enumerate(power_labels):
        ax_power.scatter(
            energy_points_mj[idx],
            f1_values[idx],
            s=120,
            label=label,
            color=power_colors[idx],
            marker="o" if idx == 1 else "X",
        )
        ax_power.text(
            energy_points_mj[idx],
            min(1.02, f1_values[idx] + 0.03),
            f"{f1_values[idx]:.2f} F1",
            ha="center",
            color=power_colors[idx],
            fontweight="bold",
        )
    ax_power.set_xlabel("Total Energy (mJ)")
    ax_power.set_ylabel("F1 Score")
    ax_power.set_ylim(0, 1.05)
    ax_power.set_title("Accuracy vs. Power Budget")
    ax_power.grid(True, alpha=0.3)
    ax_power.legend(loc="lower right")

    ax_energy = fig.add_subplot(gs[1, 0])
    ax_energy.plot(
        time_axis,
        state.energy_joules,
        color="tab:blue",
        lw=2,
        label="Extropic ThermoChip",
    )
    ax_energy.plot(
        time_axis,
        state.baseline_energy_joules,
        color="tab:orange",
        lw=2,
        ls="--",
        label="GPU residual",
    )
    ax_energy.set_xlabel("Time Step")
    ax_energy.set_ylabel("Cumulative Joules")
    ax_energy.set_title("Power Trajectory (TCF vs. GPU)")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.fill_between(
        time_axis,
        state.energy_joules,
        state.baseline_energy_joules,
        where=state.baseline_energy_joules >= state.energy_joules,
        color="tab:blue",
        alpha=0.08,
    )
    ax_energy.legend(loc="upper left")
    ax_energy.text(
        0.02,
        0.85,
        f"{state.energy_savings_pct:.1f}% less Joules",
        transform=ax_energy.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax_energy2 = ax_energy.twinx()
    ax_energy2.bar(
        time_axis,
        state.alert_counts,
        color="tab:red",
        alpha=0.25,
        width=0.5,
        label="Alerts",
    )
    ax_energy2.set_ylabel("Alerts per step")

    ax_modal = fig.add_subplot(gs[1, 1])
    focus_zone = top_zones[0]
    for m_idx, modality in enumerate(modalities):
        ax_modal.plot(
            time_axis,
            state.sensor_field[:, focus_zone, m_idx],
            label=modality,
            lw=1.6,
        )
    ax_modal.set_title(f"Zone {focus_zone + 1}: Sensor Inputs")
    ax_modal.set_xlabel("Time Step")
    ax_modal.set_ylabel("Normalized Amplitude")
    ax_modal.legend(fontsize=8)
    ax_modal.grid(True, alpha=0.3)

    ax_truth = fig.add_subplot(gs[1, 2])
    truth_counts = state.event_flags.sum(axis=1)
    tcf_counts = (state.threat_scores > THREAT_THRESHOLD).sum(axis=1)
    naive_counts = (state.naive_scores > THREAT_THRESHOLD).sum(axis=1)
    ax_truth.plot(time_axis, truth_counts, label="True events", color="black", lw=2)
    ax_truth.plot(time_axis, tcf_counts, label="Thermodynamic", color="tab:blue", lw=2)
    ax_truth.plot(time_axis, naive_counts, label="Naive sum", color="tab:orange", lw=2)
    ax_truth.set_xlabel("Time Step")
    ax_truth.set_ylabel("Count")
    ax_truth.set_title("Event Tracking vs. Truth")
    ax_truth.grid(True, alpha=0.3)
    ax_truth.legend(fontsize=8)

    ax_efficiency = fig.add_subplot(gs[1, 3])
    eff_labels = ["Extropic ThermoChip", "GPU residual"]
    eff_values = [
        state.joules_per_detection["thermo"],
        state.joules_per_detection["baseline"],
    ]
    eff_colors = ["tab:blue", "tab:orange"]
    ax_efficiency.barh(eff_labels, eff_values, color=eff_colors, alpha=0.8)
    for idx, value in enumerate(eff_values):
        ax_efficiency.text(
            value * 1.02 if value > 0 else 0.02,
            idx,
            f"{value:.2e} J/detect",
            va="center",
            color=eff_colors[idx],
            fontweight="bold",
        )
    ax_efficiency.set_xlabel("Joules per true detection")
    ax_efficiency.set_title("Energy Efficiency of Detections")

    ax_sensor_corr = fig.add_subplot(gs[2, 0])
    sensor_corr = _correlation_matrix(state.sensor_field)
    im_sensor = ax_sensor_corr.imshow(sensor_corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax_sensor_corr.set_xticks(range(len(modalities)), modalities, rotation=45, ha="right")
    ax_sensor_corr.set_yticks(range(len(modalities)), modalities)
    ax_sensor_corr.set_title("Raw Sensor Correlations")
    fig.colorbar(im_sensor, ax=ax_sensor_corr, fraction=0.046, pad=0.04)

    ax_fused_corr = fig.add_subplot(gs[2, 1])
    fused_corr = _correlation_matrix(state.fused_vectors)
    im_fused = ax_fused_corr.imshow(fused_corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax_fused_corr.set_xticks(range(len(modalities)), modalities, rotation=45, ha="right")
    ax_fused_corr.set_yticks(range(len(modalities)), modalities)
    ax_fused_corr.set_title("Thermodynamic Fusion Correlations")
    fig.colorbar(im_fused, ax=ax_fused_corr, fraction=0.046, pad=0.04)

    ax_dropout = fig.add_subplot(gs[2, 2])
    dropout_rates = state.dropout_masks.mean(axis=(0, 1))
    ax_dropout.barh(modalities, dropout_rates, color="tab:purple", alpha=0.7)
    ax_dropout.set_xlim(0, 1)
    ax_dropout.set_xlabel("Failure Rate")
    ax_dropout.set_title("Modal Resilience (TCF handles gaps)")

    ax_alerts = fig.add_subplot(gs[2, 3])
    ax_alerts.bar(
        time_axis,
        state.alert_counts,
        color="tab:red",
        alpha=0.6,
        width=0.6,
    )
    ax_alerts.set_xlabel("Time Step")
    ax_alerts.set_ylabel("Alerts")
    ax_alerts.set_title("Alert Cadence (load on responders)")
    ax_alerts.grid(True, axis="y", alpha=0.3)

    case_spec = gs[3, 0:2].subgridspec(1, 2, width_ratios=[1.3, 0.7], wspace=0.25)
    ax_case_text = fig.add_subplot(case_spec[0, 0])
    ax_case_text.axis("off")
    protocol_block = ", ".join(ute.recommended_protocols) or "No catalogued protocol"
    case_lines = [
        f"Unified Threat Engine → {ute.scenario_label}",
        f"Threat Level: {ute.threat_level.title()} | UTE Confidence: {ute.confidence:.2f}",
        f"Thermo Fused Confidence: {ute.fused_confidence:.2f}",
        f"Detections: {', '.join(ute.correlated_detections) if ute.correlated_detections else 'None'}",
        f"Protocols: {protocol_block}",
        f"Rationale: {ute.alerts_rationale}",
    ]
    ax_case_text.text(
        0.01,
        0.98,
        "\n".join(case_lines),
        va="top",
        ha="left",
        fontsize=10,
        fontfamily="monospace",
    )
    ax_case_bar = fig.add_subplot(case_spec[0, 1])
    domains = list(DOMAIN_ORDER)
    domain_values = [ute.correlated_domains.get(domain, 0) for domain in domains]
    ax_case_bar.barh(
        domains,
        domain_values,
        color=[DOMAIN_COLORS[d] for d in domains],
        alpha=0.8,
    )
    ax_case_bar.set_xlabel("Correlated detections")
    ax_case_bar.set_title("Audio/Video/Sensor coverage")
    for idx, value in enumerate(domain_values):
        ax_case_bar.text(value + 0.1, idx, f"{int(value)}", va="center")

    corr_spec = gs[3, 2:4].subgridspec(2, 1, height_ratios=[0.65, 0.35], hspace=0.35)
    ax_domains = fig.add_subplot(corr_spec[0, 0])
    for domain in DOMAIN_ORDER:
        ax_domains.plot(
            time_axis,
            ute.domain_timeline[domain],
            label=f"{domain} correlations",
            color=DOMAIN_COLORS[domain],
            lw=2,
        )
    ax_domains.set_xlabel("Time Step")
    ax_domains.set_ylabel("Correlated hits")
    ax_domains.set_title("UTE correlation timeline")
    ax_domains.grid(True, alpha=0.3)
    ax_domains.legend(fontsize=8)

    ax_actions = fig.add_subplot(corr_spec[1, 0])
    ax_actions.axis("off")
    max_actions = 5
    listed_actions = ute.response_actions[:max_actions]
    action_lines = [
        "COA/ROE Actions:",
        *[f"- {action}" for action in listed_actions],
    ]
    if len(ute.response_actions) > max_actions:
        action_lines.append(f"... +{len(ute.response_actions) - max_actions} more")
    ax_actions.text(
        0.01,
        0.95,
        "\n".join(action_lines),
        va="top",
        ha="left",
        fontsize=10,
    )

    fig.suptitle(
        "SoundSafe.ai Unified Threat Engine caseboard\n"
        "Extropic ThermoChip TCF boosts accuracy, power savings, and COA/ROE readiness",
        fontsize=12,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoundSafe thermodynamic visual sim")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--zones", type=int, default=9, help="Number of SoundSafe zones")
    parser.add_argument("--steps", type=int, default=48, help="Simulation length")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/soundsafe_visual/soundsafe_sim.png"),
        help="Path for the visualization figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modalities = (
        "Acoustic Aggression",
        "Seismic Impact",
        "RFID Spoof",
        "Thermal Motion",
    )
    cfg = SimulationConfig(
        n_zones=args.zones,
        n_steps=args.steps,
        modalities=modalities,
    )
    key = jax.random.key(args.seed)
    state = run_visual_simulation(key, cfg)
    plot_results(cfg, state, modalities, args.output)
    metrics = state.detection_metrics
    print(
        "Saved SoundSafe thermodynamic visual to"
        f" {args.output}. TCF F1={metrics['tcf'].f1:.2f} vs. Naive F1={metrics['naive'].f1:.2f}."
    )
    thermo_total_mj = state.energy_joules[-1] * 1e3
    baseline_total_mj = state.baseline_energy_joules[-1] * 1e3
    print(
        "Extropic ThermoChip energy:",
        f"{thermo_total_mj:.2f} mJ ({state.energy_savings_pct:.1f}% less than GPU {baseline_total_mj:.2f} mJ).",
        "J/detect →",
        f"Thermo {state.joules_per_detection['thermo']:.2e} vs. GPU {state.joules_per_detection['baseline']:.2e}.",
    )
    ute = state.ute_summary
    protocol_str = ", ".join(ute.recommended_protocols) if ute.recommended_protocols else "None"
    print(
        "Unified Threat Engine:",
        f"{ute.scenario_label} (UTE {ute.confidence:.2f} / Thermo {ute.fused_confidence:.2f})",
        f"Protocols {protocol_str}",
    )
    truncated_actions = ute.response_actions[:5]
    action_suffix = " ..." if len(ute.response_actions) > len(truncated_actions) else ""
    print(
        "COA/ROE actions →",
        ", ".join(truncated_actions) + action_suffix,
    )


if __name__ == "__main__":
    main()
