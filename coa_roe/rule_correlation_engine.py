"""
FLAGSHIP Rule Correlation Engine

Comprehensive engine that maps every ROE rule to its correlations, dependencies,
and relationships for effective Call-Of-Action (COA) generation and execution.
This engine ensures that all rules are properly correlated and can be used
to generate accurate and compliant COAs.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

logger = get_logger(__name__)


# ============================================================================
# RULE CORRELATION DATA STRUCTURES
# ============================================================================


class CorrelationType(Enum):
    """Types of rule correlations."""

    DEPENDENCY = "dependency"  # Rule depends on another rule
    CONFLICT = "conflict"  # Rule conflicts with another rule
    ENHANCEMENT = "enhancement"  # Rule enhances another rule
    PREREQUISITE = "prerequisite"  # Rule is prerequisite for another rule
    SUPPLEMENTARY = "supplementary"  # Rule supplements another rule
    OVERRIDE = "override"  # Rule overrides another rule
    HIERARCHICAL = "hierarchical"  # Rule is hierarchically related
    TEMPORAL = "temporal"  # Rule has temporal relationship
    SPATIAL = "spatial"  # Rule has spatial relationship
    FUNCTIONAL = "functional"  # Rule has functional relationship


class CorrelationStrength(Enum):
    """Strength of rule correlations."""

    CRITICAL = "critical"  # Critical correlation
    HIGH = "high"  # High correlation
    MEDIUM = "medium"  # Medium correlation
    LOW = "low"  # Low correlation
    MINIMAL = "minimal"  # Minimal correlation


class RuleCategory(Enum):
    """Categories of rules for correlation mapping."""

    USE_OF_FORCE = "use_of_force"
    HUMAN_RIGHTS = "human_rights"
    LEGAL_COMPLIANCE = "legal_compliance"
    SAFETY_PROTOCOLS = "safety_protocols"
    EMERGENCY_RESPONSE = "emergency_response"
    COMMUNICATION = "communication"
    DOCUMENTATION = "documentation"
    TRAINING = "training"
    AUDIT = "audit"
    ESCALATION = "escalation"
    NOTIFICATION = "notification"
    INVESTIGATION = "investigation"
    MEDICAL_CARE = "medical_care"
    EQUIPMENT = "equipment"
    COORDINATION = "coordination"
    PRIVACY = "privacy"
    DATA_PROTECTION = "data_protection"
    ENVIRONMENTAL = "environmental"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    ACCOUNTABILITY = "accountability"


@dataclass
class RuleCorrelation:
    """Individual rule correlation."""

    correlation_id: str
    source_rule_id: str
    target_rule_id: str
    correlation_type: CorrelationType
    strength: CorrelationStrength
    description: str
    rationale: str
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RuleDependency:
    """Rule dependency relationship."""

    dependency_id: str
    dependent_rule_id: str
    prerequisite_rule_id: str
    dependency_type: str
    description: str
    critical: bool = False
    conditions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RuleConflict:
    """Rule conflict relationship."""

    conflict_id: str
    rule_1_id: str
    rule_2_id: str
    conflict_type: str
    description: str
    resolution_strategy: str
    priority_rule_id: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RuleHierarchy:
    """Rule hierarchy relationship."""

    hierarchy_id: str
    parent_rule_id: str
    child_rule_id: str
    hierarchy_type: str
    description: str
    inheritance_rules: List[str] = field(default_factory=list)
    override_conditions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RuleTemporalRelationship:
    """Temporal relationship between rules."""

    temporal_id: str
    rule_1_id: str
    rule_2_id: str
    temporal_type: str  # before, after, during, concurrent
    description: str
    time_constraints: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RuleSpatialRelationship:
    """Spatial relationship between rules."""

    spatial_id: str
    rule_1_id: str
    rule_2_id: str
    spatial_type: str  # same_location, adjacent, within_range, etc.
    description: str
    spatial_constraints: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RuleFunctionalRelationship:
    """Functional relationship between rules."""

    functional_id: str
    rule_1_id: str
    rule_2_id: str
    functional_type: str  # input, output, trigger, result, etc.
    description: str
    functional_constraints: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class COARuleMapping:
    """Mapping of rules to Call-Of-Actions."""

    mapping_id: str
    coa_id: str
    rule_ids: List[str]
    rule_sequence: List[str]
    dependencies: List[str]
    conflicts: List[str]
    prerequisites: List[str]
    enhancements: List[str]
    overrides: List[str]
    created_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# RULE CORRELATION ENGINE
# ============================================================================


class RuleCorrelationEngine:
    """Comprehensive engine for managing rule correlations and relationships."""

    def __init__(self):
        """Initialize the rule correlation engine."""
        self.logger = logger

        # Storage for correlations and relationships
        self.rule_correlations: Dict[str, RuleCorrelation] = {}
        self.rule_dependencies: Dict[str, RuleDependency] = {}
        self.rule_conflicts: Dict[str, RuleConflict] = {}
        self.rule_hierarchies: Dict[str, RuleHierarchy] = {}
        self.rule_temporal_relationships: Dict[str, RuleTemporalRelationship] = {}
        self.rule_spatial_relationships: Dict[str, RuleSpatialRelationship] = {}
        self.rule_functional_relationships: Dict[str, RuleFunctionalRelationship] = {}
        self.coa_rule_mappings: Dict[str, COARuleMapping] = {}

        # Rule categorization
        self.rule_categories: Dict[str, RuleCategory] = {}

        # Load comprehensive rule correlations
        self._load_comprehensive_rule_correlations()

        self.logger.info("Rule Correlation Engine initialized")

    def _load_comprehensive_rule_correlations(self):
        """Load comprehensive rule correlations for all ROE standards."""
        self.logger.info("Loading comprehensive rule correlations...")

        # Load international law rule correlations
        self._load_international_law_correlations()

        # Load national law rule correlations
        self._load_national_law_correlations()

        # Load military protocol correlations
        self._load_military_protocol_correlations()

        # Load law enforcement correlations
        self._load_law_enforcement_correlations()

        # Load corporate security correlations
        self._load_corporate_security_correlations()

        # Load emergency response correlations
        self._load_emergency_response_correlations()

        # Load cybersecurity correlations
        self._load_cybersecurity_correlations()

        # Load specialized security correlations
        self._load_specialized_security_correlations()

        # Load cross-category correlations
        self._load_cross_category_correlations()

        self.logger.info(f"Loaded {len(self.rule_correlations)} rule correlations")

    def _load_international_law_correlations(self):
        """Load correlations for international law rules."""
        # UN Code of Conduct correlations
        self._add_rule_correlation(
            "CORR-UN-001-002",
            "UN-001",
            "UN-002",
            CorrelationType.ENHANCEMENT,
            CorrelationStrength.HIGH,
            "UN Code of Conduct enhances UN Basic Principles on Use of Force",
            "The Code of Conduct provides the ethical foundation that the Basic Principles build upon",
        )

        self._add_rule_correlation(
            "CORR-UN-001-003",
            "UN-001",
            "UN-003",
            CorrelationType.PREREQUISITE,
            CorrelationStrength.CRITICAL,
            "UN Code of Conduct is prerequisite for UN Convention against Torture",
            "The Code of Conduct establishes the fundamental principles that the Convention enforces",
        )

        # UN Basic Principles correlations
        self._add_rule_correlation(
            "CORR-UN-002-003",
            "UN-002",
            "UN-003",
            CorrelationType.DEPENDENCY,
            CorrelationStrength.HIGH,
            "UN Basic Principles depend on UN Convention against Torture",
            "Use of force principles must comply with anti-torture requirements",
        )

        # Add rule categories
        self.rule_categories["UN-001"] = RuleCategory.HUMAN_RIGHTS
        self.rule_categories["UN-002"] = RuleCategory.USE_OF_FORCE
        self.rule_categories["UN-003"] = RuleCategory.HUMAN_RIGHTS

    def _load_national_law_correlations(self):
        """Load correlations for national law rules."""
        # U.S. Constitutional correlations
        self._add_rule_correlation(
            "CORR-US-001-002",
            "US-001",
            "US-002",
            CorrelationType.ENHANCEMENT,
            CorrelationStrength.MEDIUM,
            "U.S. Constitutional standards enhance OSHA Emergency Action Plans",
            "Constitutional requirements inform emergency response procedures",
        )

        # OSHA correlations
        self._add_rule_correlation(
            "CORR-US-002-EMERG-001",
            "US-002",
            "EMERG-001",
            CorrelationType.SUPPLEMENTARY,
            CorrelationStrength.HIGH,
            "OSHA Emergency Action Plans supplement NFPA Emergency Management",
            "OSHA requirements complement NFPA emergency management standards",
        )

        # Add rule categories
        self.rule_categories["US-001"] = RuleCategory.LEGAL_COMPLIANCE
        self.rule_categories["US-002"] = RuleCategory.SAFETY_PROTOCOLS

    def _load_military_protocol_correlations(self):
        """Load correlations for military protocol rules."""
        # NATO ROE correlations
        self._add_rule_correlation(
            "CORR-MIL-001-UN-002",
            "MIL-001",
            "UN-002",
            CorrelationType.DEPENDENCY,
            CorrelationStrength.CRITICAL,
            "NATO ROE depend on UN Basic Principles on Use of Force",
            "NATO operations must comply with UN use of force principles",
        )

        self._add_rule_correlation(
            "CORR-MIL-001-UN-003",
            "MIL-001",
            "UN-003",
            CorrelationType.PREREQUISITE,
            CorrelationStrength.CRITICAL,
            "NATO ROE are prerequisite for UN Convention against Torture compliance",
            "NATO operations must prevent torture and cruel treatment",
        )

        # Add rule categories
        self.rule_categories["MIL-001"] = RuleCategory.USE_OF_FORCE

    def _load_law_enforcement_correlations(self):
        """Load correlations for law enforcement rules."""
        # IACP correlations
        self._add_rule_correlation(
            "CORR-LE-001-US-001",
            "LE-001",
            "US-001",
            CorrelationType.DEPENDENCY,
            CorrelationStrength.CRITICAL,
            "IACP Use of Force Policy depends on U.S. Constitutional standards",
            "Law enforcement use of force must comply with constitutional requirements",
        )

        self._add_rule_correlation(
            "CORR-LE-001-UN-002",
            "LE-001",
            "UN-002",
            CorrelationType.ENHANCEMENT,
            CorrelationStrength.HIGH,
            "IACP Use of Force Policy enhances UN Basic Principles",
            "IACP policy provides detailed implementation of UN principles",
        )

        # Add rule categories
        self.rule_categories["LE-001"] = RuleCategory.USE_OF_FORCE

    def _load_corporate_security_correlations(self):
        """Load correlations for corporate security rules."""
        # ASIS correlations
        self._add_rule_correlation(
            "CORR-CORP-001-LE-001",
            "CORP-001",
            "LE-001",
            CorrelationType.SUPPLEMENTARY,
            CorrelationStrength.MEDIUM,
            "ASIS Security Management supplements IACP Use of Force Policy",
            "Corporate security complements law enforcement procedures",
        )

        # Add rule categories
        self.rule_categories["CORP-001"] = RuleCategory.SAFETY_PROTOCOLS

    def _load_emergency_response_correlations(self):
        """Load correlations for emergency response rules."""
        # NFPA correlations
        self._add_rule_correlation(
            "CORR-EMERG-001-US-002",
            "EMERG-001",
            "US-002",
            CorrelationType.ENHANCEMENT,
            CorrelationStrength.HIGH,
            "NFPA Emergency Management enhances OSHA Emergency Action Plans",
            "NFPA provides comprehensive emergency management framework",
        )

        # Add rule categories
        self.rule_categories["EMERG-001"] = RuleCategory.EMERGENCY_RESPONSE

    def _load_cybersecurity_correlations(self):
        """Load correlations for cybersecurity rules."""
        # NIST correlations
        self._add_rule_correlation(
            "CORR-CYBER-001-CORP-001",
            "CYBER-001",
            "CORP-001",
            CorrelationType.SUPPLEMENTARY,
            CorrelationStrength.MEDIUM,
            "NIST Cybersecurity Framework supplements ASIS Security Management",
            "Cybersecurity complements physical security management",
        )

        # Add rule categories
        self.rule_categories["CYBER-001"] = RuleCategory.DATA_PROTECTION

    def _load_specialized_security_correlations(self):
        """Load correlations for specialized security rules."""
        # Maritime security correlations
        self._add_rule_correlation(
            "CORR-SPEC-001-MIL-001",
            "SPEC-001",
            "MIL-001",
            CorrelationType.ENHANCEMENT,
            CorrelationStrength.MEDIUM,
            "Maritime Security ROE enhance NATO ROE for maritime operations",
            "Maritime-specific rules complement general military ROE",
        )

        # Add rule categories
        self.rule_categories["SPEC-001"] = RuleCategory.SAFETY_PROTOCOLS

    def _load_cross_category_correlations(self):
        """Load correlations across different rule categories."""
        # Use of Force correlations
        use_of_force_rules = ["UN-002", "MIL-001", "LE-001"]
        for i, rule1 in enumerate(use_of_force_rules):
            for rule2 in use_of_force_rules[i + 1 :]:
                self._add_rule_correlation(
                    f"CORR-UOF-{rule1}-{rule2}",
                    rule1,
                    rule2,
                    CorrelationType.FUNCTIONAL,
                    CorrelationStrength.HIGH,
                    f"{rule1} and {rule2} are functionally related use of force rules",
                    "All use of force rules must be considered together for comprehensive compliance",
                )

        # Human Rights correlations
        human_rights_rules = ["UN-001", "UN-003"]
        for i, rule1 in enumerate(human_rights_rules):
            for rule2 in human_rights_rules[i + 1 :]:
                self._add_rule_correlation(
                    f"CORR-HR-{rule1}-{rule2}",
                    rule1,
                    rule2,
                    CorrelationType.HIERARCHICAL,
                    CorrelationStrength.CRITICAL,
                    f"{rule1} and {rule2} are hierarchically related human rights rules",
                    "Human rights rules form a comprehensive framework for protection",
                )

        # Safety Protocol correlations
        safety_rules = ["US-002", "CORP-001", "SPEC-001"]
        for i, rule1 in enumerate(safety_rules):
            for rule2 in safety_rules[i + 1 :]:
                self._add_rule_correlation(
                    f"CORR-SAFETY-{rule1}-{rule2}",
                    rule1,
                    rule2,
                    CorrelationType.SUPPLEMENTARY,
                    CorrelationStrength.MEDIUM,
                    f"{rule1} and {rule2} are supplementary safety protocol rules",
                    "Safety protocols work together to ensure comprehensive protection",
                )

    def _add_rule_correlation(
        self,
        correlation_id: str,
        source_rule_id: str,
        target_rule_id: str,
        correlation_type: CorrelationType,
        strength: CorrelationStrength,
        description: str,
        rationale: str,
    ):
        """Add a rule correlation."""
        correlation = RuleCorrelation(
            correlation_id=correlation_id,
            source_rule_id=source_rule_id,
            target_rule_id=target_rule_id,
            correlation_type=correlation_type,
            strength=strength,
            description=description,
            rationale=rationale,
        )

        self.rule_correlations[correlation_id] = correlation

    async def get_rule_correlations(self, rule_id: str) -> List[RuleCorrelation]:
        """Get all correlations for a specific rule."""
        correlations = []

        for correlation in self.rule_correlations.values():
            if (
                correlation.source_rule_id == rule_id
                or correlation.target_rule_id == rule_id
            ):
                correlations.append(correlation)

        return correlations

    async def get_rule_dependencies(self, rule_id: str) -> List[RuleDependency]:
        """Get all dependencies for a specific rule."""
        dependencies = []

        for dependency in self.rule_dependencies.values():
            if (
                dependency.dependent_rule_id == rule_id
                or dependency.prerequisite_rule_id == rule_id
            ):
                dependencies.append(dependency)

        return dependencies

    async def get_rule_conflicts(self, rule_id: str) -> List[RuleConflict]:
        """Get all conflicts for a specific rule."""
        conflicts = []

        for conflict in self.rule_conflicts.values():
            if conflict.rule_1_id == rule_id or conflict.rule_2_id == rule_id:
                conflicts.append(conflict)

        return conflicts

    async def get_related_rules(self, rule_id: str) -> Dict[str, List[str]]:
        """Get all related rules for a specific rule."""
        related_rules = {
            "dependencies": [],
            "prerequisites": [],
            "enhancements": [],
            "supplementary": [],
            "conflicts": [],
            "hierarchical": [],
            "temporal": [],
            "spatial": [],
            "functional": [],
        }

        # Get correlations
        correlations = await self.get_rule_correlations(rule_id)
        for correlation in correlations:
            if correlation.source_rule_id == rule_id:
                target_rule = correlation.target_rule_id
            else:
                target_rule = correlation.source_rule_id

            if correlation.correlation_type == CorrelationType.DEPENDENCY:
                related_rules["dependencies"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.PREREQUISITE:
                related_rules["prerequisites"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.ENHANCEMENT:
                related_rules["enhancements"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.SUPPLEMENTARY:
                related_rules["supplementary"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.CONFLICT:
                related_rules["conflicts"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.HIERARCHICAL:
                related_rules["hierarchical"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.TEMPORAL:
                related_rules["temporal"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.SPATIAL:
                related_rules["spatial"].append(target_rule)
            elif correlation.correlation_type == CorrelationType.FUNCTIONAL:
                related_rules["functional"].append(target_rule)

        return related_rules

    async def get_rules_by_category(self, category: RuleCategory) -> List[str]:
        """Get all rules in a specific category."""
        rules = []

        for rule_id, rule_category in self.rule_categories.items():
            if rule_category == category:
                rules.append(rule_id)

        return rules

    async def get_cross_category_correlations(
        self, category1: RuleCategory, category2: RuleCategory
    ) -> List[RuleCorrelation]:
        """Get correlations between two rule categories."""
        correlations = []

        rules1 = await self.get_rules_by_category(category1)
        rules2 = await self.get_rules_by_category(category2)

        for correlation in self.rule_correlations.values():
            if (
                correlation.source_rule_id in rules1
                and correlation.target_rule_id in rules2
            ) or (
                correlation.source_rule_id in rules2
                and correlation.target_rule_id in rules1
            ):
                correlations.append(correlation)

        return correlations

    async def generate_coa_rule_mapping(
        self, coa_id: str, threat_type: str, threat_level: str
    ) -> COARuleMapping:
        """Generate a COA rule mapping for a specific threat scenario."""
        # Determine applicable rules based on threat type and level
        applicable_rules = await self._get_applicable_rules(threat_type, threat_level)

        # Determine rule sequence based on dependencies
        rule_sequence = await self._determine_rule_sequence(applicable_rules)

        # Identify dependencies, conflicts, and other relationships
        dependencies = await self._identify_dependencies(applicable_rules)
        conflicts = await self._identify_conflicts(applicable_rules)
        prerequisites = await self._identify_prerequisites(applicable_rules)
        enhancements = await self._identify_enhancements(applicable_rules)
        overrides = await self._identify_overrides(applicable_rules)

        # Create COA rule mapping
        mapping = COARuleMapping(
            mapping_id=f"COA-MAP-{coa_id}",
            coa_id=coa_id,
            rule_ids=applicable_rules,
            rule_sequence=rule_sequence,
            dependencies=dependencies,
            conflicts=conflicts,
            prerequisites=prerequisites,
            enhancements=enhancements,
            overrides=overrides,
        )

        self.coa_rule_mappings[coa_id] = mapping
        return mapping

    async def _get_applicable_rules(
        self, threat_type: str, threat_level: str
    ) -> List[str]:
        """Get applicable rules for a threat scenario."""
        applicable_rules = []

        # Base rules that always apply
        base_rules = ["UN-001", "UN-003"]  # Human rights and anti-torture

        # Add rules based on threat type
        if "violence" in threat_type.lower() or "weapon" in threat_type.lower():
            applicable_rules.extend(
                ["UN-002", "MIL-001", "LE-001"]
            )  # Use of force rules

        if "emergency" in threat_type.lower():
            applicable_rules.extend(["US-002", "EMERG-001"])  # Emergency response rules

        if "cyber" in threat_type.lower():
            applicable_rules.extend(["CYBER-001"])  # Cybersecurity rules

        if "maritime" in threat_type.lower():
            applicable_rules.extend(["SPEC-001"])  # Maritime security rules

        # Add rules based on threat level
        if threat_level in ["critical", "high"]:
            applicable_rules.extend(
                ["CORP-001"]
            )  # Corporate security for high-level threats

        # Add base rules
        applicable_rules.extend(base_rules)

        # Remove duplicates
        return list(set(applicable_rules))

    async def _determine_rule_sequence(self, rules: List[str]) -> List[str]:
        """Determine the sequence of rules based on dependencies."""
        # Simple dependency-based sequencing
        # In a real implementation, this would use topological sorting

        # Start with rules that have no dependencies
        sequence = []
        remaining_rules = rules.copy()

        # Add human rights rules first (prerequisites)
        for rule in ["UN-001", "UN-003"]:
            if rule in remaining_rules:
                sequence.append(rule)
                remaining_rules.remove(rule)

        # Add use of force rules
        for rule in ["UN-002", "MIL-001", "LE-001"]:
            if rule in remaining_rules:
                sequence.append(rule)
                remaining_rules.remove(rule)

        # Add remaining rules
        sequence.extend(remaining_rules)

        return sequence

    async def _identify_dependencies(self, rules: List[str]) -> List[str]:
        """Identify dependencies between rules."""
        dependencies = []

        for rule in rules:
            rule_correlations = await self.get_rule_correlations(rule)
            for correlation in rule_correlations:
                if correlation.correlation_type == CorrelationType.DEPENDENCY:
                    if (
                        correlation.source_rule_id == rule
                        and correlation.target_rule_id in rules
                    ):
                        dependencies.append(f"{rule} -> {correlation.target_rule_id}")
                    elif (
                        correlation.target_rule_id == rule
                        and correlation.source_rule_id in rules
                    ):
                        dependencies.append(f"{correlation.source_rule_id} -> {rule}")

        return dependencies

    async def _identify_conflicts(self, rules: List[str]) -> List[str]:
        """Identify conflicts between rules."""
        conflicts = []

        for rule in rules:
            rule_correlations = await self.get_rule_correlations(rule)
            for correlation in rule_correlations:
                if correlation.correlation_type == CorrelationType.CONFLICT:
                    if (
                        correlation.source_rule_id == rule
                        and correlation.target_rule_id in rules
                    ):
                        conflicts.append(f"{rule} <-> {correlation.target_rule_id}")
                    elif (
                        correlation.target_rule_id == rule
                        and correlation.source_rule_id in rules
                    ):
                        conflicts.append(f"{correlation.source_rule_id} <-> {rule}")

        return conflicts

    async def _identify_prerequisites(self, rules: List[str]) -> List[str]:
        """Identify prerequisites for rules."""
        prerequisites = []

        for rule in rules:
            rule_correlations = await self.get_rule_correlations(rule)
            for correlation in rule_correlations:
                if correlation.correlation_type == CorrelationType.PREREQUISITE:
                    if (
                        correlation.source_rule_id == rule
                        and correlation.target_rule_id in rules
                    ):
                        prerequisites.append(
                            f"{correlation.target_rule_id} requires {rule}"
                        )
                    elif (
                        correlation.target_rule_id == rule
                        and correlation.source_rule_id in rules
                    ):
                        prerequisites.append(
                            f"{rule} requires {correlation.source_rule_id}"
                        )

        return prerequisites

    async def _identify_enhancements(self, rules: List[str]) -> List[str]:
        """Identify enhancements between rules."""
        enhancements = []

        for rule in rules:
            rule_correlations = await self.get_rule_correlations(rule)
            for correlation in rule_correlations:
                if correlation.correlation_type == CorrelationType.ENHANCEMENT:
                    if (
                        correlation.source_rule_id == rule
                        and correlation.target_rule_id in rules
                    ):
                        enhancements.append(
                            f"{rule} enhances {correlation.target_rule_id}"
                        )
                    elif (
                        correlation.target_rule_id == rule
                        and correlation.source_rule_id in rules
                    ):
                        enhancements.append(
                            f"{correlation.source_rule_id} enhances {rule}"
                        )

        return enhancements

    async def _identify_overrides(self, rules: List[str]) -> List[str]:
        """Identify overrides between rules."""
        overrides = []

        for rule in rules:
            rule_correlations = await self.get_rule_correlations(rule)
            for correlation in rule_correlations:
                if correlation.correlation_type == CorrelationType.OVERRIDE:
                    if (
                        correlation.source_rule_id == rule
                        and correlation.target_rule_id in rules
                    ):
                        overrides.append(
                            f"{rule} overrides {correlation.target_rule_id}"
                        )
                    elif (
                        correlation.target_rule_id == rule
                        and correlation.source_rule_id in rules
                    ):
                        overrides.append(
                            f"{correlation.source_rule_id} overrides {rule}"
                        )

        return overrides

    async def get_correlation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all rule correlations."""
        summary = {
            "total_correlations": len(self.rule_correlations),
            "total_rules": len(self.rule_categories),
            "by_correlation_type": {},
            "by_strength": {},
            "by_category": {},
            "cross_category_correlations": {},
        }

        # Count by correlation type
        for correlation in self.rule_correlations.values():
            corr_type = correlation.correlation_type.value
            summary["by_correlation_type"][corr_type] = (
                summary["by_correlation_type"].get(corr_type, 0) + 1
            )

        # Count by strength
        for correlation in self.rule_correlations.values():
            strength = correlation.strength.value
            summary["by_strength"][strength] = (
                summary["by_strength"].get(strength, 0) + 1
            )

        # Count by category
        for rule_id, category in self.rule_categories.items():
            cat_name = category.value
            summary["by_category"][cat_name] = (
                summary["by_category"].get(cat_name, 0) + 1
            )

        # Count cross-category correlations
        for correlation in self.rule_correlations.values():
            source_cat = self.rule_categories.get(correlation.source_rule_id)
            target_cat = self.rule_categories.get(correlation.target_rule_id)

            if source_cat and target_cat and source_cat != target_cat:
                cat_pair = f"{source_cat.value} -> {target_cat.value}"
                summary["cross_category_correlations"][cat_pair] = (
                    summary["cross_category_correlations"].get(cat_pair, 0) + 1
                )

        return summary

    async def export_correlation_data(self, format: str = "json") -> str:
        """Export correlation data in specified format."""
        if format.lower() == "json":
            export_data = {
                "rule_correlations": {
                    corr_id: {
                        "correlation_id": corr.correlation_id,
                        "source_rule_id": corr.source_rule_id,
                        "target_rule_id": corr.target_rule_id,
                        "correlation_type": corr.correlation_type.value,
                        "strength": corr.strength.value,
                        "description": corr.description,
                        "rationale": corr.rationale,
                        "conditions": corr.conditions,
                        "exceptions": corr.exceptions,
                        "created_at": corr.created_at.isoformat(),
                        "updated_at": corr.updated_at.isoformat(),
                    }
                    for corr_id, corr in self.rule_correlations.items()
                },
                "rule_categories": {
                    rule_id: category.value
                    for rule_id, category in self.rule_categories.items()
                },
                "coa_rule_mappings": {
                    coa_id: {
                        "mapping_id": mapping.mapping_id,
                        "coa_id": mapping.coa_id,
                        "rule_ids": mapping.rule_ids,
                        "rule_sequence": mapping.rule_sequence,
                        "dependencies": mapping.dependencies,
                        "conflicts": mapping.conflicts,
                        "prerequisites": mapping.prerequisites,
                        "enhancements": mapping.enhancements,
                        "overrides": mapping.overrides,
                        "created_at": mapping.created_at.isoformat(),
                    }
                    for coa_id, mapping in self.coa_rule_mappings.items()
                },
            }

            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


async def main():
    """Main function to demonstrate rule correlation engine."""
    logger.info("Starting FLAGSHIP Rule Correlation Engine")

    try:
        # Initialize rule correlation engine
        correlation_engine = RuleCorrelationEngine()

        # Get correlation summary
        summary = await correlation_engine.get_correlation_summary()

        # Generate example COA rule mapping
        coa_mapping = await correlation_engine.generate_coa_rule_mapping(
            "COA-001", "armed_violence", "critical"
        )

        # Export correlation data
        export_data = await correlation_engine.export_correlation_data()

        # Save data
        with open("rule_correlation_data.json", "w") as f:
            f.write(export_data)

        logger.info("Rule correlation engine demonstration completed")
        logger.info("Correlation data saved to: rule_correlation_data.json")

        # Print summary
        logger.info(f"Correlation summary: {summary}")

        return summary

    except Exception as e:
        logger.error(f"Error during rule correlation engine demonstration: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
