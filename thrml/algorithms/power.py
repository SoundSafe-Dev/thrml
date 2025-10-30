"""Best-effort power sampling for macOS (powermetrics) with fallback.

This is a convenience for local benchmarking. For precise measurement,
use external watt-meters or platform-specific APIs with privileges.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


POWERMETRICS_CMD = [
	"powermetrics",
	"--samplers",
	"cpu_power",
	"-n",
	"1",
]

CPU_POWER_RE = re.compile(r"CPU Average power:\s*([0-9.]+)\s*mW", re.IGNORECASE)


def sample_power_w() -> Optional[float]:
	"""Try to sample instantaneous CPU power (W) via powermetrics; return None on failure."""
	try:
		out = subprocess.check_output(POWERMETRICS_CMD, stderr=subprocess.STDOUT, text=True, timeout=3)
		m = CPU_POWER_RE.search(out)
		if not m:
			return None
		mw = float(m.group(1))
		return mw / 1000.0
	except Exception:
		return None


@dataclass
class PowerSession:
	"""Measure average power over a code region and estimate Joules.

	Use as:
	    ps = PowerSession(); ps.start(); ...work...; energy_j = ps.stop()
	"""
	start_time_s: float = 0.0
	end_time_s: float = 0.0
	start_w: Optional[float] = None
	end_w: Optional[float] = None

	def start(self) -> None:
		self.start_time_s = time.perf_counter()
		self.start_w = sample_power_w()

	def stop(self) -> float:
		self.end_time_s = time.perf_counter()
		self.end_w = sample_power_w()
		dt = max(0.0, self.end_time_s - self.start_time_s)
		# Average of start/end readings; if None, assume unknown (0)
		p0 = self.start_w or 0.0
		p1 = self.end_w or p0
		avg_w = 0.5 * (p0 + p1)
		return avg_w * dt
