#!/usr/bin/env python3
"""Build a single-page HTML report aggregating discovery and prototype results.

Usage:
  python examples/build_report.py --results results --output results/report.html
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path


def img_to_data_uri(path: Path) -> str:
	if not path.exists():
		return ""
	b = path.read_bytes()
	enc = base64.b64encode(b).decode("ascii")
	mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
	return f"data:{mime};base64,{enc}"


SOUNDSAFE_SECTIONS = [
    ("Deepfake Voice Detection",
     "TVS watermarking + SRSL pre‑proc + LABI gating. Verify provenance; lift weak spoof artifacts; pay kT ln2 only when ΔH justifies.") ,
    ("Audio Watermarking & Content Protection",
     "TVS embeds thermal nonce watermarks; report correlation and bitrate overhead; provenance for courts/ops."),
    ("Anomalous Sound Detection / Behavioral Sound Mapping",
     "EFSM per‑site energy baselines; SRSL lifts faint distress onsets; LABI skips energy‑expensive updates when not needed."),
    ("Weapon/Aggression/Loitering/Zone",
     "REF low‑power features; TAPS schedules sensors; TBRO routes compute; BPP escalates under ROE constraints."),
    ("Environmental + Access Sensors",
     "PPTS probabilistic time/phase sync; TCF causal fusion across A/V/doors/temperature for robust alerts."),
]

ALGO_CAPTIONS = {
    "srsl": "SRSL (Stochastic Resonance): choose β (1/T) maximizing I(X;Y) to surface weak signals without broadband gain.",
    "labi": "LABI (Landauer‑Aware): skip/update frontier—pay energy per bit only when entropy reduction warrants.",
    "tcf": "TCF (Causal Fusion): perturb‑observe stability—robust multimodal edges for SoundSafe fusion under failures.",
    "taps": "TAPS (Thermal Scheduling): activation matrix shows sensor×bitrate; KPIs: coverage and Joules/sec.",
    "bpp": "BPP (Policy Planner): action probabilities with temperature‑controlled risk; ROE‑compatible escalation.",
    "efsm": "EFSM (Energy Fingerprint): anomaly score bars; ΔE spikes indicate out‑of‑manifold audio/windows.",
    "tbro": "TBRO (Thermal Bandit): risk→allocation; SLO compliance with fewer GPU‑hours.",
    "ppts": "PPTS (Phase/Time Sync): polar phases + pairwise diffs; low‑overhead synchronization.",
    "tvs": "TVS (Thermo‑Verifiable Sensing): residual watermark effect, correlation, and bitrate overhead.",
    "ref": "REF (Reservoir‑EBM): stable features at ultra‑low cost; front‑end for detection stacks.",
}


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--results", type=str, default="results")
	p.add_argument("--output", type=str, default="results/report.html")
	args = p.parse_args()

	root = Path(args.results)
	sections = []

	# Prototypes latest dir
	proto_dirs = sorted([d for d in (root).glob("prototypes_*") if d.is_dir()])
	if proto_dirs:
		latest = proto_dirs[-1]
		imgs = sorted(latest.glob("*.png"))
		blocks = []
		for p in imgs:
			stem = p.stem
			cap = ALGO_CAPTIONS.get(stem, p.name)
			blocks.append(
				f"<div style='display:inline-block;margin:8px;vertical-align:top'>"
				f"<img src='{img_to_data_uri(p)}' width='360'><br><b>{stem}</b><br><small>{cap}</small></div>"
			)
		sections.append(f"<h2>Prototypes ({latest.name})</h2><div>{''.join(blocks)}</div>")

	# Discovery
	disc = root / "discovery"
	if disc.exists():
		parts = []
		for sub in ["srsl", "labi", "tcf"]:
			d = disc / sub
			if not d.exists():
				continue
			plots = sorted(d.glob("*.png"))
			links = sorted(d.glob("*.csv")) + sorted(d.glob("*.json"))
			plot_div = "".join(
				f"<div style='display:inline-block;margin:8px'><img src='{img_to_data_uri(p)}' width='360'><br>{p.name}</div>"
				for p in plots
			)
			link_div = "".join(f"<li><a href='{p.as_posix()}'>{p.name}</a></li>" for p in links)
			explain = ALGO_CAPTIONS.get(sub, "")
			# Add axis legend notes
			legend = ""
			if sub == "srsl":
				legend = "<small>Heatmap axes: β (1/T) vs SNR(dB); color = mutual information.</small>"
			if sub == "labi":
				legend = "<small>Heatmap axes: likelihood scale vs energy threshold (log‑log); color = skip_rate.</small>"
			if sub == "tcf":
				legend = "<small>Plot: perturbation strength vs # causal edges (stability curve).</small>"
			parts.append(f"<h3>{sub.upper()}</h3><p><small>{explain}</small></p><div>{plot_div}{legend}</div><ul>{link_div}</ul>")
		sections.append("<h2>Discovery</h2>" + "".join(parts))

	# SoundSafe mapping
	ss_rows = "".join(
		f"<h3>{title}</h3><p><small>{desc}</small></p>" for title, desc in SOUNDSAFE_SECTIONS
	)
	sections.insert(0, f"<h2>SoundSafe Capability Mapping</h2>{ss_rows}")

	html = f"""
<!doctype html>
<html>
<head>
	<meta charset='utf-8'>
	<title>THRML Thermodynamic Report</title>
	<style>body{{font-family:sans-serif}} h2{{border-bottom:1px solid #ccc}}</style>
</head>
<body>
	<h1>THRML Thermodynamic Report</h1>
	<p>This report aggregates prototype figures and discovery outputs (CSV/JSON + plots).</p>
	{''.join(sections)}
</body>
</html>
"""
	out = Path(args.output)
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(html)
	print(f"Report written to {out}")


if __name__ == "__main__":
	main()
