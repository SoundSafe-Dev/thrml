"""Adapter utilities to integrate THRML outputs with COA/ROE Engine.

This module builds threat dicts and context payloads compatible with
COAROEEngine.generate_response_with_rule_correlations(threat_event: Dict).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_threat_event(
	*,
	threat_type: str,
	threat_level: str,
	confidence: float,
	description: str = "",
	source: str = "thrml",
	location: Optional[Dict[str, Any]] = None,
	metadata: Optional[Dict[str, Any]] = None,
	context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Construct a threat_event dict accepted by COAROEEngine.

	- threat_type: e.g., "gunshot_detected", "explosion_detected", "physical_security"
	- threat_level: "low" | "medium" | "high"
	- confidence: 0..1
	- metadata/context can include THRML KPIs and suggestions
	"""
	return {
		"event_id": None,
		"threat_type": threat_type,
		"threat_level": threat_level,
		"confidence": float(confidence),
		"source": source,
		"location": location or {},
		"description": description,
		"metadata": metadata or {},
		"context": context or {},
	}


def attach_thrml_context(
	*,
	bpp: Optional[Dict[str, Any]] = None,
	taps: Optional[Dict[str, Any]] = None,
	tcf: Optional[Dict[str, Any]] = None,
	labi: Optional[Dict[str, Any]] = None,
	extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""Create a context payload carrying THRML algorithm KPIs to COA/ROE.

	Fields are optional; engine-side consumers may selectively use them.
	"""
	payload: Dict[str, Any] = {}
	if bpp: payload["bpp"] = bpp
	if taps: payload["taps"] = taps
	if tcf: payload["tcf"] = tcf
	if labi: payload["labi"] = labi
	if extra: payload.update(extra)
	return payload
