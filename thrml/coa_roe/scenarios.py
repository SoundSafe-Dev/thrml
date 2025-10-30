"""Comprehensive Case Scenarios (minimal subset for integration demo).

Provides scenario definitions, sector adaptations, and pattern matching.
This is a lightweight, cleaned version focusing on Active Shooter and Terrorist Attack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class ScenarioType(Enum):
	ACTIVE_SHOOTER = "active_shooter"
	TERRORIST_ATTACK = "terrorist_attack"


class SectorType(Enum):
	AIRPORTS = "airports"
	SCHOOLS = "schools"
	STADIUMS = "stadiums"
	GOVERNMENT_BUILDINGS = "government_buildings"
	UNIVERSITIES = "universities"
	SHOPPING_MALLS = "shopping_malls"
	WORKPLACES = "workplaces"


@dataclass
class DetectionPattern:
	detection_type: str
	confidence_threshold: float
	time_window: int  # seconds
	spatial_radius: float  # meters
	required_count: int = 1
	optional: bool = False
	weight: float = 1.0


@dataclass
class CorrelationRule:
	rule_id: str
	primary_detection: str
	secondary_detection: str
	correlation_type: str  # temporal, spatial, causal, behavioral
	time_delay: int = 0
	spatial_distance: float = 0.0
	strength_threshold: float = 0.7
	description: str = ""


@dataclass
class ResponseProtocol:
	protocol_id: str
	name: str
	priority: str  # immediate, high, medium, low
	actions: List[str]
	escalation_threshold: float = 0.8
	description: str = ""


@dataclass
class ThreatScenario:
	scenario_id: str
	scenario_type: ScenarioType
	sector_types: List[SectorType]
	detection_patterns: List[DetectionPattern]
	correlation_rules: List[CorrelationRule]
	response_protocols: List[ResponseProtocol]
	threat_level: str  # critical, high, medium, low
	confidence_threshold: float = 0.7
	false_positive_patterns: List[List[str]] = field(default_factory=list)
	description: str = ""
	metadata: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveCaseScenarios:
	def __init__(self):
		self.scenarios: Dict[str, ThreatScenario] = {}
		self._load_scenarios()

	def _load_scenarios(self):
		active_shooter = ThreatScenario(
			scenario_id="active_shooter_primary",
			scenario_type=ScenarioType.ACTIVE_SHOOTER,
			sector_types=[
				SectorType.SCHOOLS,
				SectorType.UNIVERSITIES,
				SectorType.SHOPPING_MALLS,
				SectorType.WORKPLACES,
			],
			detection_patterns=[
				DetectionPattern("firearm_detection", 0.8, 60, 50.0, 1, False, 2.0),
				DetectionPattern("gunshot_detection", 0.7, 60, 50.0, 1, False, 2.0),
				DetectionPattern("crowd_panic_detection", 0.7, 120, 100.0, 1, True, 1.8),
			],
			correlation_rules=[
				CorrelationRule(
					"firearm_gunshot_correlation",
					"firearm_detection",
					"gunshot_detection",
					"temporal",
					5,
					20.0,
					0.8,
					"Firearm detection followed by gunshot detection",
				),
			],
			response_protocols=[
				ResponseProtocol(
					"immediate_lockdown",
					"Immediate Lockdown",
					"immediate",
					["lockdown_building", "secure_entrances", "alert_occupants"],
					0.8,
					"Immediate building lockdown",
				),
				ResponseProtocol(
					"law_enforcement_contact",
					"Law Enforcement Contact",
					"immediate",
					["contact_911", "provide_situation_details", "coordinate_response"],
					0.9,
					"Contact law enforcement",
				),
			],
			threat_level="critical",
			confidence_threshold=0.8,
			false_positive_patterns=[["fireworks", "crowd_excitement"]],
			description="Active shooter with weapon + crowd panic",
		)

		terrorist_attack = ThreatScenario(
			scenario_id="terrorist_attack_primary",
			scenario_type=ScenarioType.TERRORIST_ATTACK,
			sector_types=[
				SectorType.AIRPORTS,
				SectorType.STADIUMS,
				SectorType.GOVERNMENT_BUILDINGS,
			],
			detection_patterns=[
				DetectionPattern("unattended_object_detection", 0.8, 300, 50.0, 1, False, 2.0),
				DetectionPattern("explosion_detection", 0.9, 60, 100.0, 1, False, 2.5),
			],
			correlation_rules=[
				CorrelationRule(
					"object_explosion_correlation",
					"unattended_object_detection",
					"explosion_detection",
					"temporal",
					300,
					50.0,
					0.9,
					"Unattended object followed by explosion",
				),
			],
			response_protocols=[
				ResponseProtocol(
					"emergency_response",
					"Emergency Response",
					"immediate",
					["activate_emergency_teams", "secure_perimeter", "assess_damage"],
					0.9,
					"Activate emergency response teams",
				),
			],
			threat_level="critical",
			confidence_threshold=0.8,
			false_positive_patterns=[["construction_explosion", "dust_cloud"]],
			description="Terrorist attack with explosive device",
		)

		self.scenarios = {
			"active_shooter": active_shooter,
			"terrorist_attack": terrorist_attack,
		}

	def get_scenario(self, name: str) -> Optional[ThreatScenario]:
		return self.scenarios.get(name)

	def match(self, detections: List[Dict[str, Any]], sector: Optional[SectorType] = None) -> List[Dict[str, Any]]:
		candidates = list(self.scenarios.values())
		if sector:
			candidates = [s for s in candidates if sector in s.sector_types]
		matches: List[Dict[str, Any]] = []
		for s in candidates:
			res = self._match_one(s, detections)
			if res.get("matched"):
				matches.append(res)
		return matches

	def _match_one(self, scenario: ThreatScenario, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
		required = [p for p in scenario.detection_patterns if not p.optional]
		opt = [p for p in scenario.detection_patterns if p.optional]

		req_ok = True
		all_matches: List[Dict[str, Any]] = []
		for p in required:
			m = self._find_matches(p, detections)
			if len(m) < p.required_count:
				req_ok = False
				break
			all_matches.extend(m)
		if not req_ok:
			return {"scenario_id": scenario.scenario_id, "scenario_type": scenario.scenario_type.value, "matched": False}

		for p in opt:
			all_matches.extend(self._find_matches(p, detections))

		# Weighted confidence
		total_w = sum(p.weight for p in scenario.detection_patterns)
		matched_types = {m["detection_type"] for m in all_matches}
		matched_w = sum(p.weight for p in scenario.detection_patterns if p.detection_type in matched_types)
		conf = matched_w / total_w if total_w > 0 else 0.0
		if conf < scenario.confidence_threshold:
			return {"scenario_id": scenario.scenario_id, "scenario_type": scenario.scenario_type.value, "matched": False}

		return {
			"scenario_id": scenario.scenario_id,
			"scenario_type": scenario.scenario_type.value,
			"matched": True,
			"confidence": conf,
			"threat_level": scenario.threat_level,
			"response_protocols": [p.name for p in scenario.response_protocols],
		}

	def _find_matches(self, pattern: DetectionPattern, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		out: List[Dict[str, Any]] = []
		for d in detections:
			if d.get("detection_type") != pattern.detection_type:
				continue
			if float(d.get("confidence", 0.0)) < pattern.confidence_threshold:
				continue
			# Time window (from now)
			try:
				ts = datetime.fromisoformat(d["timestamp"])  # type: ignore[index]
			except Exception:
				continue
			if abs((datetime.now() - ts).total_seconds()) > pattern.time_window:
				continue
			out.append(d)
		return out
