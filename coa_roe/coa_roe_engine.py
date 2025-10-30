"""
FLAGSHIP COA/ROE Engine

Main engine for managing Course-of-Action (COA) and Rules-of-Engagement (ROE)
responses to threat events. Integrates with the Unified Threat Engine (UTE)
to provide actionable security and law enforcement responses.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

from ..__init__ import ThreatCategory, ThreatEvent, ThreatLevel
from .audit_trail import AuditTrailManager
from .escalation_manager import EscalationManager
from .legal_compliance import LegalComplianceManager
from .notification_coordinator import NotificationCoordinator
from .response_protocols import ResponseProtocolManager
from .rule_correlation_engine import COARuleMapping

logger = get_logger(__name__)


# ============================================================================
# COA/ROE DATA STRUCTURES
# ============================================================================


class ResponseType(Enum):
    """Types of response actions."""

    SECURITY_RESPONSE = "security_response"
    LAW_ENFORCEMENT_RESPONSE = "law_enforcement_response"
    EMERGENCY_SERVICES = "emergency_services"
    EVACUATION = "evacuation"
    LOCKDOWN = "lockdown"
    INVESTIGATION = "investigation"
    COORDINATION = "coordination"


class ForceLevel(Enum):
    """Levels of force that may be used."""

    NO_FORCE = "no_force"
    VERBAL_COMMANDS = "verbal_commands"
    NON_LETHAL_FORCE = "non_lethal_force"
    LETHAL_FORCE = "lethal_force"


class ComplianceStandard(Enum):
    """Legal and regulatory compliance standards."""

    UN_CODE_OF_CONDUCT = "un_code_of_conduct"
    OSHA_EMERGENCY_ACTION = "osha_emergency_action"
    US_CONSTITUTIONAL_USE_OF_FORCE = "us_constitutional_use_of_force"
    INTERNATIONAL_HUMAN_RIGHTS = "international_human_rights"


@dataclass
class COAROEProtocol:
    """Course-of-Action and Rules-of-Engagement protocol."""

    protocol_id: str
    event_type: str
    threat_category: ThreatCategory
    security_coa: Dict[str, Any]
    security_roe: Dict[str, Any]
    law_enforcement_coa: Dict[str, Any]
    law_enforcement_roe: Dict[str, Any]
    response_time_seconds: int
    escalation_path: List[str]
    compliance_standards: List[ComplianceStandard]
    force_level: ForceLevel
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResponseAction:
    """Individual response action within a COA/ROE protocol."""

    action_id: str
    protocol_id: str
    action_type: ResponseType
    description: str
    target_entities: List[str]
    estimated_duration: timedelta
    required_resources: List[str]
    success_criteria: List[str]
    force_level: ForceLevel
    compliance_requirements: List[ComplianceStandard]
    priority: int
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseExecution:
    """Execution record of a COA/ROE response."""

    execution_id: str
    threat_event: ThreatEvent
    protocol: COAROEProtocol
    actions: List[ResponseAction]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "initiated"
    executed_by: Optional[str] = None
    compliance_verified: bool = False
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# COA/ROE ENGINE
# ============================================================================


class COAROEEngine:
    """Main COA/ROE engine for generating and executing responses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the COA/ROE engine with injected registry instances."""
        self.config = config or {}
        self.logger = logger

        # Initialize sub-components
        self.protocol_manager = ResponseProtocolManager(
            self.config.get("protocol_config", {})
        )
        self.legal_compliance = LegalComplianceManager(
            self.config.get("legal_config", {})
        )
        self.escalation_manager = EscalationManager(
            self.config.get("escalation_config", {})
        )
        self.notification_coordinator = NotificationCoordinator(
            self.config.get("notification_config", {})
        )
        self.audit_trail = AuditTrailManager(self.config.get("audit_config", {}))

        # Rule correlation engine (injected or created)
        self.rule_correlation_engine = None
        if "rule_correlation_engine" in self.config:
            self.rule_correlation_engine = self.config["rule_correlation_engine"]
            self.logger.info("Using injected rule correlation engine")
        else:
            try:
                from .rule_correlation_engine import RuleCorrelationEngine

                self.rule_correlation_engine = RuleCorrelationEngine()
                self.logger.info("Created new rule correlation engine")
            except ImportError as e:
                self.logger.warning(f"Rule correlation engine not available: {e}")

        # ROE registry (injected or created)
        self.roe_registry = None
        if "roe_registry" in self.config:
            self.roe_registry = self.config["roe_registry"]
            self.logger.info("Using injected ROE registry")
        else:
            try:
                from .detailed_roe_registry import DetailedROERegistry

                self.roe_registry = DetailedROERegistry()
                self.logger.info("Created new ROE registry")
            except ImportError as e:
                self.logger.warning(f"ROE registry not available: {e}")

        # Load initial protocols
        self._load_initial_protocols()

        self.logger.info("COA/ROE Engine initialized with injected registry instances")

    def _load_initial_protocols(self):
        """Load initial protocols with error handling."""
        try:
            # Load protocol specifications
            protocol_specs = self._load_protocol_specifications()

            # Create protocols from specifications
            for spec in protocol_specs:
                try:
                    # Normalize threat_category values to engine enum where possible
                    if isinstance(spec.get("threat_category"), str):
                        try:
                            from ..__init__ import (
                                ThreatCategory as EngineThreatCategory,
                            )

                            raw = spec["threat_category"].lower()
                            # Map known aliases
                            alias_map = {
                                "violence": EngineThreatCategory.ACTIVE_SHOOTER,
                                "emergency": EngineThreatCategory.NATURAL_DISASTER,
                                "security": EngineThreatCategory.PHYSICAL_SECURITY,
                            }
                            spec["threat_category"] = alias_map.get(
                                raw, EngineThreatCategory.PHYSICAL_SECURITY
                            )
                        except Exception:
                            pass

                    protocol = self._create_protocol_from_data(spec)
                    # Add only if manager supports add_protocol; otherwise ignore
                    try:
                        self.protocol_manager.add_protocol(protocol)
                    except Exception:
                        pass
                except Exception as e:
                    self.logger.warning(
                        f"Failed to create protocol from spec {spec.get('protocol_id', 'unknown')}: {e}"
                    )
                    continue

            self.logger.info(f"Loaded {len(protocol_specs)} initial protocols")

        except Exception as e:
            self.logger.error(f"Error loading initial protocols: {e}")

    def _load_protocol_specifications(self) -> List[Dict[str, Any]]:
        """Load COA/ROE protocol specifications."""
        return [
            {
                "protocol_id": "COA-SEC-GUN-002",
                "event_type": "gunshot_detected",
                "threat_category": ThreatCategory.VIOLENCE,
                "security_coa": {
                    "goal": "Protect occupants, contain shooter, aid victims",
                    "phases": [
                        {
                            "phase": "Alarm & Notify",
                            "duration": "0-5s",
                            "actions": [
                                "Trigger audible alarm",
                                "Call law-enforcement dispatch",
                            ],
                        },
                        {
                            "phase": "Lockdown",
                            "duration": "5-20s",
                            "actions": [
                                "Secure exterior doors",
                                "Direct employees to shelter",
                            ],
                        },
                        {
                            "phase": "Contain",
                            "duration": "20-60s",
                            "actions": [
                                "Take cover near entrances",
                                "Provide first aid if safe",
                            ],
                        },
                    ],
                },
                "security_roe": {
                    "code": "ROE-SEC-112-SELFDEF",
                    "description": "Security personnel may employ non-lethal force to protect life and detain the shooter; lethal force is justified only when the shooter poses an imminent threat of death or serious injury.",
                    "force_level": ForceLevel.NON_LETHAL_FORCE,
                    "restrictions": [
                        "Actions must respect human dignity",
                        "Use force strictly when necessary",
                    ],
                },
                "law_enforcement_coa": {
                    "goal": "Neutralize armed suspect and rescue victims",
                    "phases": [
                        {
                            "phase": "Dispatch & Approach",
                            "duration": "0-2min",
                            "actions": [
                                "Armed patrol/SWAT dispatched",
                                "Form outer perimeter",
                            ],
                        },
                        {
                            "phase": "Engagement",
                            "duration": "2-10min",
                            "actions": [
                                "Approach under cover",
                                "Issue clear commands",
                                "Use calibrated lethal force if necessary",
                            ],
                        },
                    ],
                },
                "law_enforcement_roe": {
                    "code": "ROE-LE-112-DEADLY",
                    "description": "Officers may use deadly force when they have probable cause to believe the suspect poses a threat of serious physical harm to officers or others.",
                    "force_level": ForceLevel.LETHAL_FORCE,
                    "restrictions": ["Force must stop once the threat abates"],
                },
                "response_time_seconds": 10,
                "escalation_path": [
                    "Security-Guard",
                    "Security-Super",
                    "LE Dispatch",
                    "Emergency Team",
                ],
                "compliance_standards": [
                    ComplianceStandard.UN_CODE_OF_CONDUCT,
                    ComplianceStandard.US_CONSTITUTIONAL_USE_OF_FORCE,
                ],
                "force_level": ForceLevel.LETHAL_FORCE,
                "description": "Response protocol for gunshot detection events",
            },
            {
                "protocol_id": "COA-SEC-EXP-002",
                "event_type": "explosion_detected",
                "threat_category": ThreatCategory.EMERGENCY,
                "security_coa": {
                    "goal": "Evacuate occupants and preserve life",
                    "phases": [
                        {
                            "phase": "Alarm & Evacuate",
                            "duration": "0-15s",
                            "actions": [
                                "Activate distinctive alarm",
                                "Announce evacuation",
                                "Escort to designated exits",
                            ],
                        },
                        {
                            "phase": "Shut Utilities",
                            "duration": "15-60s",
                            "actions": ["Shut off gas and electricity if safe"],
                        },
                    ],
                },
                "security_roe": {
                    "code": "ROE-SEC-EVAC",
                    "description": "Focus on safe evacuation rather than force; use of force is limited to protecting evacuees from harm.",
                    "force_level": ForceLevel.NO_FORCE,
                    "restrictions": ["OSHA emergency-action standards apply"],
                },
                "law_enforcement_coa": {
                    "goal": "Secure scene, assist victims, and investigate",
                    "phases": [
                        {
                            "phase": "Respond & Secure",
                            "duration": "0-5min",
                            "actions": ["Establish outer perimeter", "Reroute traffic"],
                        }
                    ],
                },
                "law_enforcement_roe": {
                    "code": "ROE-LE-EVAC",
                    "description": "Prioritize life safety; deadly force authorized only if suspect poses imminent lethal threat.",
                    "force_level": ForceLevel.NON_LETHAL_FORCE,
                    "restrictions": ["Use force only when strictly necessary"],
                },
                "response_time_seconds": 5,
                "escalation_path": [
                    "Site-Manager",
                    "Security-Super",
                    "Fire Dept",
                    "Bomb Squad",
                ],
                "compliance_standards": [
                    ComplianceStandard.OSHA_EMERGENCY_ACTION,
                    ComplianceStandard.UN_CODE_OF_CONDUCT,
                ],
                "force_level": ForceLevel.NON_LETHAL_FORCE,
                "description": "Response protocol for explosion detection events",
            },
            # Additional protocols would be added here...
        ]

    def _create_protocol_from_data(self, data: Dict[str, Any]) -> COAROEProtocol:
        """Create a COAROEProtocol from specification data."""
        return COAROEProtocol(
            protocol_id=data["protocol_id"],
            event_type=data["event_type"],
            threat_category=data["threat_category"],
            security_coa=data["security_coa"],
            security_roe=data["security_roe"],
            law_enforcement_coa=data["law_enforcement_coa"],
            law_enforcement_roe=data["law_enforcement_roe"],
            response_time_seconds=data["response_time_seconds"],
            escalation_path=data["escalation_path"],
            compliance_standards=data["compliance_standards"],
            force_level=data["force_level"],
            description=data["description"],
        )

    async def start(self):
        """Start the COA/ROE engine."""
        self.logger.info("Starting COA/ROE Engine")

        try:
            # Start sub-components
            if self.protocol_manager:
                await self.protocol_manager.start()
            if self.legal_compliance:
                await self.legal_compliance.start()
            if self.escalation_manager:
                await self.escalation_manager.start()
            if self.notification_coordinator:
                await self.notification_coordinator.start()
            if self.audit_trail:
                await self.audit_trail.start()

            self.logger.info("COA/ROE Engine started successfully")

        except Exception as e:
            self.logger.error(f"Error starting COA/ROE Engine: {e}")
            raise

    async def stop(self):
        """Stop the COA/ROE engine."""
        self.logger.info("Stopping COA/ROE Engine")

        try:
            # Stop sub-components
            if self.protocol_manager:
                await self.protocol_manager.stop()
            if self.legal_compliance:
                await self.legal_compliance.stop()
            if self.escalation_manager:
                await self.escalation_manager.stop()
            if self.notification_coordinator:
                await self.notification_coordinator.stop()
            if self.audit_trail:
                await self.audit_trail.stop()

            self.logger.info("COA/ROE Engine stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping COA/ROE Engine: {e}")
            raise

    async def generate_response(
        self, threat_event: ThreatEvent, context: Optional[Dict[str, Any]] = None
    ) -> ResponseExecution:
        """Generate response for a threat event."""
        self.logger.info(
            f"Generating response for threat event: {threat_event.event_id}"
        )

        try:
            # Find appropriate protocol
            protocol = await self._find_protocol_for_event(threat_event)
            if not protocol:
                self.logger.warning(
                    f"No protocol found for threat event: {threat_event.event_id}"
                )
                return None

            # Generate response actions
            actions = await self._generate_response_actions(
                protocol, threat_event, context
            )

            # Create execution
            execution = ResponseExecution(
                execution_id=f"exec_{datetime.now().isoformat()}",
                threat_event=threat_event,
                protocol=protocol,
                actions=actions,
                start_time=datetime.now(),
            )

            self.logger.info(f"Generated response with {len(actions)} actions")
            return execution

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    async def generate_response_with_rule_correlations(
        self, threat_event: Dict[str, Any]
    ) -> ResponseExecution:
        """Generate response with comprehensive rule correlations."""
        self.logger.info(
            f"Generating response with rule correlations for threat: {threat_event.get('threat_type', 'unknown')}"
        )

        try:
            # Generate COA rule mapping if correlation engine is available
            coa_rule_mapping = None
            if self.rule_correlation_engine:
                coa_id = f"COA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                threat_type = threat_event.get("threat_type", "unknown")
                threat_level = threat_event.get("threat_level", "medium")

                coa_rule_mapping = (
                    await self.rule_correlation_engine.generate_coa_rule_mapping(
                        coa_id, threat_type, threat_level
                    )
                )

            # Prefer standard response generation for stability; use mapping for summary
            if not coa_rule_mapping:
                # Fallback to standard response generation
                def _normalize_threat_category(value: str) -> ThreatCategory:
                    try:
                        return ThreatCategory(value)
                    except Exception:
                        fallback_map = {
                            "aviation": ThreatCategory.PHYSICAL_SECURITY,
                            "maritime": ThreatCategory.PHYSICAL_SECURITY,
                            "healthcare": ThreatCategory.PHYSICAL_SECURITY,
                            "education": ThreatCategory.PHYSICAL_SECURITY,
                            "transit": ThreatCategory.INFRASTRUCTURE,
                            "energy": ThreatCategory.INFRASTRUCTURE,
                            "finance": ThreatCategory.CYBER_SECURITY,
                            "retail": ThreatCategory.PHYSICAL_SECURITY,
                            "critical_infrastructure": ThreatCategory.INFRASTRUCTURE,
                        }
                        return fallback_map.get(
                            str(value).lower(), ThreatCategory.PHYSICAL_SECURITY
                        )

                threat_event_obj = ThreatEvent(
                    event_id=threat_event.get(
                        "event_id", f"event_{datetime.now().isoformat()}"
                    ),
                    timestamp=datetime.now(),
                    threat_level=ThreatLevel(
                        threat_event.get("threat_level", "medium")
                    ),
                    threat_category=_normalize_threat_category(
                        threat_event.get("threat_type", "physical_security")
                    ),
                    confidence=threat_event.get("confidence", 0.5),
                    source=threat_event.get("source", "unknown"),
                    location=threat_event.get("location"),
                    description=threat_event.get("description"),
                    metadata=threat_event.get("metadata", {}),
                )
                response = await self.generate_response(
                    threat_event_obj, threat_event.get("context")
                )
            else:
                # With mapping, still use standard generator to produce actions
                def _normalize_threat_category(value: str) -> ThreatCategory:
                    try:
                        return ThreatCategory(value)
                    except Exception:
                        fallback_map = {
                            "aviation": ThreatCategory.PHYSICAL_SECURITY,
                            "maritime": ThreatCategory.PHYSICAL_SECURITY,
                            "healthcare": ThreatCategory.PHYSICAL_SECURITY,
                            "education": ThreatCategory.PHYSICAL_SECURITY,
                            "transit": ThreatCategory.INFRASTRUCTURE,
                            "energy": ThreatCategory.INFRASTRUCTURE,
                            "finance": ThreatCategory.CYBER_SECURITY,
                            "retail": ThreatCategory.PHYSICAL_SECURITY,
                            "critical_infrastructure": ThreatCategory.INFRASTRUCTURE,
                        }
                        return fallback_map.get(
                            str(value).lower(), ThreatCategory.PHYSICAL_SECURITY
                        )

                threat_event_obj = ThreatEvent(
                    event_id=threat_event.get(
                        "event_id", f"event_{datetime.now().isoformat()}"
                    ),
                    timestamp=datetime.now(),
                    threat_level=ThreatLevel(
                        threat_event.get("threat_level", "medium")
                    ),
                    threat_category=_normalize_threat_category(
                        threat_event.get("threat_type", "physical_security")
                    ),
                    confidence=threat_event.get("confidence", 0.5),
                    source=threat_event.get("source", "unknown"),
                    location=threat_event.get("location"),
                    description=threat_event.get("description"),
                    metadata=threat_event.get("metadata", {}),
                )
                response = await self.generate_response(
                    threat_event_obj, threat_event.get("context")
                )

            # Log execution with rule correlations
            if self.audit_trail and response:
                # Align with AuditTrailManager.log_execution_start(self, execution)
                try:
                    await self.audit_trail.log_execution_start(response)
                except TypeError:
                    # Fallback: ignore audit logging if signature differs
                    pass

            return response

        except Exception as e:
            self.logger.error(f"Error generating response with rule correlations: {e}")
            # Return a basic response if correlation fails
            return await self._generate_fallback_response(threat_event)

    async def _generate_fallback_response(
        self, threat_event: Dict[str, Any]
    ) -> ResponseExecution:
        """Generate a fallback response when rule correlations fail."""
        self.logger.info("Generating fallback response")

        # Create basic threat event
        def _normalize_threat_category(value: str) -> ThreatCategory:
            try:
                return ThreatCategory(value)
            except Exception:
                fallback_map = {
                    "aviation": ThreatCategory.PHYSICAL_SECURITY,
                    "maritime": ThreatCategory.PHYSICAL_SECURITY,
                    "healthcare": ThreatCategory.PHYSICAL_SECURITY,
                    "education": ThreatCategory.PHYSICAL_SECURITY,
                    "transit": ThreatCategory.INFRASTRUCTURE,
                    "energy": ThreatCategory.INFRASTRUCTURE,
                    "finance": ThreatCategory.CYBER_SECURITY,
                    "retail": ThreatCategory.PHYSICAL_SECURITY,
                    "critical_infrastructure": ThreatCategory.INFRASTRUCTURE,
                }
                return fallback_map.get(
                    str(value).lower(), ThreatCategory.PHYSICAL_SECURITY
                )

        threat_event_obj = ThreatEvent(
            event_id=threat_event.get(
                "event_id", f"fallback_{datetime.now().isoformat()}"
            ),
            timestamp=datetime.now(),
            threat_level=ThreatLevel(threat_event.get("threat_level", "medium")),
            threat_category=_normalize_threat_category(
                threat_event.get("threat_type", "physical_security")
            ),
            confidence=threat_event.get("confidence", 0.5),
            source=threat_event.get("source", "unknown"),
            location=threat_event.get("location"),
            description=threat_event.get("description"),
            metadata=threat_event.get("metadata", {}),
        )

        # Generate basic response
        return await self.generate_response(
            threat_event_obj, threat_event.get("context")
        )

    async def _generate_response_from_rule_mapping(
        self, threat_event: Dict[str, Any], coa_rule_mapping: COARuleMapping
    ) -> ResponseExecution:
        """Generate response from COA rule mapping."""
        # Create execution
        execution_id = f"EXEC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Generate actions based on rule sequence
        actions = []
        for rule_id in coa_rule_mapping.rule_sequence:
            rule_actions = await self._generate_actions_for_rule(rule_id, threat_event)
            actions.extend(rule_actions)

        # Helper to normalize category for this scope
        def _normalize_threat_category(value: str) -> ThreatCategory:
            try:
                return ThreatCategory(value)
            except Exception:
                fallback_map = {
                    "aviation": ThreatCategory.PHYSICAL_SECURITY,
                    "maritime": ThreatCategory.PHYSICAL_SECURITY,
                    "healthcare": ThreatCategory.PHYSICAL_SECURITY,
                    "education": ThreatCategory.PHYSICAL_SECURITY,
                    "transit": ThreatCategory.INFRASTRUCTURE,
                    "energy": ThreatCategory.INFRASTRUCTURE,
                    "finance": ThreatCategory.CYBER_SECURITY,
                    "retail": ThreatCategory.PHYSICAL_SECURITY,
                    "critical_infrastructure": ThreatCategory.INFRASTRUCTURE,
                }
                return fallback_map.get(
                    str(value).lower(), ThreatCategory.PHYSICAL_SECURITY
                )

        # Create response execution
        execution = ResponseExecution(
            execution_id=execution_id,
            threat_event=ThreatEvent(
                event_id=threat_event.get(
                    "event_id", f"event_{datetime.now().isoformat()}"
                ),
                timestamp=datetime.now(),
                threat_level=ThreatLevel(threat_event.get("threat_level", "medium")),
                threat_category=_normalize_threat_category(
                    threat_event.get("threat_type", "physical_security")
                ),
                confidence=threat_event.get("confidence", 0.5),
                source=threat_event.get("source", "unknown"),
                location=threat_event.get("location"),
                description=threat_event.get("description"),
                metadata=threat_event.get("metadata", {}),
            ),
            protocol_id=coa_rule_mapping.coa_id,
            actions=actions,
            rule_mapping=coa_rule_mapping,
            status="in_progress",
            start_time=datetime.now(),
        )

        return execution

    async def _generate_actions_for_rule(
        self, rule_id: str, threat_event: Dict[str, Any]
    ) -> List[ResponseAction]:
        """Generate actions for a specific rule."""
        actions = []

        try:
            # Get rule correlations if available
            correlations = []
            if self.rule_correlation_engine:
                correlations = await self.rule_correlation_engine.get_rule_correlations(
                    rule_id
                )

            # Generate actions based on rule type and correlations
            if "UN-001" in rule_id:  # Human rights
                actions.append(
                    ResponseAction(
                        action_id=f"ACTION-{rule_id}-{len(actions)}",
                        action_type=ResponseType.SECURITY_RESPONSE,
                        description="Ensure human rights protection",
                        target_entities=["security_personnel", "management"],
                        estimated_duration=timedelta(minutes=5),
                        required_resources=[
                            "training_materials",
                            "compliance_documentation",
                        ],
                        success_criteria=[
                            "human_rights_verified",
                            "no_violations_detected",
                        ],
                        force_level=ForceLevel.NO_FORCE,
                        compliance_requirements=[ComplianceStandard.UN_CODE_OF_CONDUCT],
                        priority=1,
                    )
                )

            elif (
                "UN-002" in rule_id or "MIL-001" in rule_id or "LE-001" in rule_id
            ):  # Use of force
                actions.append(
                    ResponseAction(
                        action_id=f"ACTION-{rule_id}-{len(actions)}",
                        action_type=ResponseType.LAW_ENFORCEMENT_RESPONSE,
                        description="Assess use of force requirements",
                        target_entities=["law_enforcement", "security_personnel"],
                        estimated_duration=timedelta(minutes=10),
                        required_resources=[
                            "use_of_force_policy",
                            "training_certification",
                        ],
                        success_criteria=[
                            "force_assessment_completed",
                            "compliance_verified",
                        ],
                        force_level=ForceLevel.NON_LETHAL_FORCE,
                        compliance_requirements=[
                            ComplianceStandard.US_CONSTITUTIONAL_USE_OF_FORCE
                        ],
                        priority=1,
                    )
                )

            elif "US-002" in rule_id or "EMERG-001" in rule_id:  # Emergency response
                actions.append(
                    ResponseAction(
                        action_id=f"ACTION-{rule_id}-{len(actions)}",
                        action_type=ResponseType.EMERGENCY_SERVICES,
                        description="Initiate emergency response procedures",
                        target_entities=["emergency_services", "security_personnel"],
                        estimated_duration=timedelta(minutes=15),
                        required_resources=["emergency_plan", "communication_systems"],
                        success_criteria=[
                            "emergency_response_initiated",
                            "coordination_established",
                        ],
                        force_level=ForceLevel.NO_FORCE,
                        compliance_requirements=[
                            ComplianceStandard.OSHA_EMERGENCY_ACTION
                        ],
                        priority=1,
                    )
                )

            # Add correlation-based actions
            for correlation in correlations:
                if correlation.correlation_type.value == "prerequisite":
                    actions.append(
                        ResponseAction(
                            action_id=f"ACTION-PREREQ-{rule_id}-{len(actions)}",
                            action_type=ResponseType.COORDINATION,
                            description=f"Verify prerequisite: {correlation.description}",
                            target_entities=["compliance_officer", "management"],
                            estimated_duration=timedelta(minutes=5),
                            required_resources=["compliance_documentation"],
                            success_criteria=[
                                "prerequisite_verified",
                                "compliance_confirmed",
                            ],
                            force_level=ForceLevel.NO_FORCE,
                            compliance_requirements=[
                                ComplianceStandard.UN_CODE_OF_CONDUCT
                            ],
                            priority=1,
                        )
                    )

        except Exception as e:
            self.logger.warning(f"Failed to generate actions for rule {rule_id}: {e}")

        return actions

    async def _get_rule_correlations_summary(
        self, coa_rule_mapping: COARuleMapping
    ) -> Dict[str, Any]:
        """Get summary of rule correlations for COA mapping."""
        if not coa_rule_mapping:
            return {}

        return {
            "total_rules": len(coa_rule_mapping.rule_ids),
            "dependencies": len(coa_rule_mapping.dependencies),
            "conflicts": len(coa_rule_mapping.conflicts),
            "prerequisites": len(coa_rule_mapping.prerequisites),
            "enhancements": len(coa_rule_mapping.enhancements),
            "overrides": len(coa_rule_mapping.overrides),
            "rule_sequence": coa_rule_mapping.rule_sequence,
        }

    async def get_rule_correlation_summary(self) -> Dict[str, Any]:
        """Get comprehensive rule correlation summary."""
        if not self.rule_correlation_engine:
            return {
                "status": "not_available",
                "error": "Rule correlation engine not initialized",
            }

        try:
            return await self.rule_correlation_engine.get_correlation_summary()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def export_rule_correlations(self, format: str = "json") -> str:
        """Export rule correlations data."""
        if not self.rule_correlation_engine:
            return json.dumps({"error": "Rule correlation engine not available"})

        try:
            return await self.rule_correlation_engine.export_correlation_data(format)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _find_protocol_for_event(
        self, threat_event: ThreatEvent
    ) -> Optional[COAROEProtocol]:
        """Find appropriate protocol for a threat event."""
        # Fetch protocols (manager exposes async API)
        try:
            protocols = await self.protocol_manager.get_all_protocols()
        except TypeError:
            # Fallback if returned synchronously in this runtime
            protocols = self.protocol_manager.get_all_protocols()  # type: ignore

        # Try exact event type match if available; ThreatEvent may not have event_type
        event_type = None
        try:
            event_type = getattr(threat_event, "event_type", None)
        except Exception:
            event_type = None
        if event_type:
            for protocol in protocols:
                if protocol.event_type == event_type:
                    return protocol
        else:
            # Fallback: derive from metadata or description
            try:
                meta_type = (threat_event.metadata or {}).get("event_type")
            except Exception:
                meta_type = None
            if meta_type:
                for protocol in protocols:
                    if protocol.event_type == meta_type:
                        return protocol
            else:
                desc = (getattr(threat_event, "description", "") or "").lower()
                for protocol in protocols:
                    if protocol.event_type and protocol.event_type in desc:
                        return protocol

        # Fall back to threat category match
        for protocol in protocols:
            if protocol.threat_category == threat_event.threat_category:
                return protocol

        return None

    async def _generate_response_actions(
        self,
        protocol: COAROEProtocol,
        threat_event: ThreatEvent,
        context: Optional[Dict[str, Any]],
    ) -> List[ResponseAction]:
        """Generate response actions based on protocol."""
        actions = []

        # Generate security response actions
        security_actions = await self._generate_security_actions(
            protocol, threat_event, context
        )
        actions.extend(security_actions)

        # Generate law enforcement response actions
        le_actions = await self._generate_law_enforcement_actions(
            protocol, threat_event, context
        )
        actions.extend(le_actions)

        # Generate coordination actions
        coordination_actions = await self._generate_coordination_actions(
            protocol, threat_event, context
        )
        actions.extend(coordination_actions)

        return actions

    async def _generate_security_actions(
        self,
        protocol: Any,
        threat_event: ThreatEvent,
        context: Optional[Dict[str, Any]],
    ) -> List[ResponseAction]:
        """Generate security response actions (aligned to ResponseProtocolManager schema)."""
        actions: List[ResponseAction] = []

        # ResponseProtocol fields: security_phases, security_force_level, compliance_standards
        phases = getattr(protocol, "security_phases", []) or []
        for i, phase in enumerate(phases):
            duration = phase.get("duration", "1min")
            success = phase.get("success_criteria", phase.get("actions", []))
            resources = phase.get(
                "resources", ["security_personnel", "communication_systems"]
            )
            action = ResponseAction(
                action_id=f"sec_{protocol.protocol_id}_{i}",
                protocol_id=protocol.protocol_id,
                action_type=ResponseType.SECURITY_RESPONSE,
                description=f"Security: {phase.get('phase', 'Phase')}",
                target_entities=["security_personnel", "facility_systems"],
                estimated_duration=self._parse_duration(duration),
                required_resources=resources,
                success_criteria=success,
                force_level=ForceLevel[
                    getattr(protocol, "security_force_level", "no_force").upper()
                ],
                compliance_requirements=[
                    ComplianceStandard[x]
                    for x in (getattr(protocol, "compliance_standards", []) or [])
                    if x in ComplianceStandard.__members__
                ]
                or [ComplianceStandard.UN_CODE_OF_CONDUCT],
                priority=i + 1,
            )
            actions.append(action)

        return actions

    async def _generate_law_enforcement_actions(
        self,
        protocol: Any,
        threat_event: ThreatEvent,
        context: Optional[Dict[str, Any]],
    ) -> List[ResponseAction]:
        """Generate law enforcement response actions (aligned to ResponseProtocolManager schema)."""
        actions: List[ResponseAction] = []

        phases = getattr(protocol, "le_phases", []) or []
        for i, phase in enumerate(phases):
            duration = phase.get("duration", "1min")
            success = phase.get("success_criteria", phase.get("actions", []))
            resources = phase.get("resources", ["patrol_units", "swat_teams"])
            action = ResponseAction(
                action_id=f"le_{protocol.protocol_id}_{i}",
                protocol_id=protocol.protocol_id,
                action_type=ResponseType.LAW_ENFORCEMENT_RESPONSE,
                description=f"Law Enforcement: {phase.get('phase','Phase')}",
                target_entities=["law_enforcement", "emergency_services"],
                estimated_duration=self._parse_duration(duration),
                required_resources=resources,
                success_criteria=success,
                force_level=ForceLevel[
                    getattr(protocol, "le_force_level", "non_lethal_force").upper()
                ],
                compliance_requirements=[
                    ComplianceStandard[x]
                    for x in (getattr(protocol, "compliance_standards", []) or [])
                    if x in ComplianceStandard.__members__
                ]
                or [ComplianceStandard.UN_CODE_OF_CONDUCT],
                priority=i + 1,
            )
            actions.append(action)

        return actions

    async def _generate_coordination_actions(
        self,
        protocol: Any,
        threat_event: ThreatEvent,
        context: Optional[Dict[str, Any]],
    ) -> List[ResponseAction]:
        """Generate coordination actions (aligned to ResponseProtocolManager schema)."""
        actions: List[ResponseAction] = []

        for i, escalation_target in enumerate(
            getattr(protocol, "escalation_path", []) or []
        ):
            action = ResponseAction(
                action_id=f"coord_{protocol.protocol_id}_{i}",
                protocol_id=protocol.protocol_id,
                action_type=ResponseType.COORDINATION,
                description=f"Coordinate with {escalation_target}",
                target_entities=[escalation_target],
                estimated_duration=timedelta(minutes=2),
                required_resources=["communication_systems"],
                success_criteria=[
                    f"{escalation_target}_notified",
                    f"{escalation_target}_responding",
                ],
                force_level=ForceLevel.NO_FORCE,
                compliance_requirements=[
                    ComplianceStandard[x]
                    for x in (getattr(protocol, "compliance_standards", []) or [])
                    if x in ComplianceStandard.__members__
                ]
                or [ComplianceStandard.UN_CODE_OF_CONDUCT],
                priority=i + 1,
            )
            actions.append(action)

        return actions

    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parse duration string into timedelta.

        Supports:
        - "Xs", "Xsec", "Xseconds"
        - "Ymin", "Yminutes"
        - Ranges: "A-Bs", "A-Bmin" (uses the upper bound B)
        - Falls back to 1 minute on parse failure
        """
        if not duration_str:
            return timedelta(minutes=1)

        s = str(duration_str).strip().lower()

        # Normalize common unit variants
        unit = None
        if s.endswith(("seconds", "second", "sec", "s")):
            unit = "s"
            for suf in ("seconds", "second", "sec"):
                if s.endswith(suf):
                    s = s[: -len(suf)] + "s"
                    break
        elif s.endswith(("minutes", "minute", "mins", "min")):
            unit = "min"
            for suf in ("minutes", "minute", "mins"):
                if s.endswith(suf):
                    s = s[: -len(suf)] + "min"
                    break

        # Now s ends with either 's' or 'min'
        try:
            if unit == "s":
                num_part = s[:-1].strip()
                # Handle range A-B
                if "-" in num_part:
                    parts = [p.strip() for p in num_part.split("-", 1)]
                    upper = int(parts[1])
                    return timedelta(seconds=upper)
                else:
                    seconds = int(num_part)
                    return timedelta(seconds=seconds)
            if unit == "min":
                num_part = s[:-3].strip()
                if "-" in num_part:
                    parts = [p.strip() for p in num_part.split("-", 1)]
                    upper = int(parts[1])
                    return timedelta(minutes=upper)
                else:
                    minutes = int(num_part)
                    return timedelta(minutes=minutes)
        except Exception:
            pass

        # Best-effort final fallback
        return timedelta(minutes=1)

    async def execute_response(self, execution_id: str, executed_by: str) -> bool:
        """Execute a COA/ROE response."""
        # This method needs to be updated to use the injected audit_trail and legal_compliance
        # For now, it will raise an error as they are not directly accessible here.
        # This is a placeholder for future integration.
        self.logger.error(
            "execute_response is not fully integrated with injected components."
        )
        return False

    async def get_active_executions(self) -> List[ResponseExecution]:
        """Get all active response executions."""
        # This method needs to be updated to use the injected protocol_manager
        # For now, it will raise an error as it's not directly accessible here.
        # This is a placeholder for future integration.
        self.logger.error(
            "get_active_executions is not fully integrated with injected components."
        )
        return []

    async def get_execution_history(self, limit: int = 100) -> List[ResponseExecution]:
        """Get response execution history."""
        # This method needs to be updated to use the injected audit_trail
        # For now, it will raise an error as it's not directly accessible here.
        # This is a placeholder for future integration.
        self.logger.error(
            "get_execution_history is not fully integrated with injected components."
        )
        return []

    async def get_protocols(self) -> List[COAROEProtocol]:
        """Get all available protocols."""
        try:
            return await self.protocol_manager.get_all_protocols()
        except TypeError:
            # Fallback if the manager returns a non-awaitable in this runtime
            return self.protocol_manager.get_all_protocols()  # type: ignore

    async def get_protocol_by_id(self, protocol_id: str) -> Optional[COAROEProtocol]:
        """Get a specific protocol by ID."""
        # This method needs to be updated to use the injected protocol_manager
        # For now, it will raise an error as it's not directly accessible here.
        # This is a placeholder for future integration.
        self.logger.error(
            "get_protocol_by_id is not fully integrated with injected components."
        )
        return None
