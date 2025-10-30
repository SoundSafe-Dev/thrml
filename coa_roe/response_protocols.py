"""
FLAGSHIP Response Protocols Manager

Manages detailed Course-of-Action (COA) and Rules-of-Engagement (ROE) protocols
for security and law enforcement responses to threat events.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

from ..__init__ import ThreatCategory, ThreatEvent

logger = get_logger(__name__)


# ============================================================================
# PROTOCOL DATA STRUCTURES
# ============================================================================


class ProtocolStatus(Enum):
    """Status of response protocols."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    TESTING = "testing"


class ProtocolPriority(Enum):
    """Priority levels for protocols."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ResponseProtocol:
    """Detailed response protocol with phases and actions."""

    protocol_id: str
    name: str
    event_type: str
    threat_category: ThreatCategory
    priority: ProtocolPriority
    status: ProtocolStatus
    description: str

    # Security COA
    security_goal: str
    security_phases: List[Dict[str, Any]]
    security_roe_code: str
    security_roe_description: str
    security_force_level: str
    security_restrictions: List[str]

    # Law Enforcement COA
    le_goal: str
    le_phases: List[Dict[str, Any]]
    le_roe_code: str
    le_roe_description: str
    le_force_level: str
    le_restrictions: List[str]

    # Response parameters
    response_time_seconds: int
    escalation_path: List[str]
    compliance_standards: List[str]
    required_resources: List[str]
    success_criteria: List[str]

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    author: str = "FLAGSHIP System"
    tags: List[str] = field(default_factory=list)


@dataclass
class ProtocolExecution:
    """Execution tracking for a response protocol."""

    execution_id: str
    protocol: ResponseProtocol
    threat_event: ThreatEvent
    start_time: datetime
    current_phase: int = 0
    completed_phases: List[int] = field(default_factory=list)
    status: str = "initiated"
    executed_by: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PROTOCOL MANAGER
# ============================================================================


class ResponseProtocolManager:
    """Manages response protocols and their execution."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the response protocol manager."""
        self.config = config or {}
        self.logger = logger

        # Protocol storage
        self.protocols: Dict[str, ResponseProtocol] = {}
        self.active_executions: Dict[str, ProtocolExecution] = {}
        self.execution_history: List[ProtocolExecution] = []

        # Load protocols
        self._load_protocols()

        self.logger.info("Response Protocol Manager initialized")

    def _load_protocols(self):
        """Load response protocols from specification."""
        protocols_data = self._get_protocol_specifications()

        for protocol_data in protocols_data:
            protocol = self._create_protocol_from_data(protocol_data)
            self.protocols[protocol.protocol_id] = protocol

        self.logger.info(f"Loaded {len(self.protocols)} response protocols")

    def _get_protocol_specifications(self) -> List[Dict[str, Any]]:
        """Get comprehensive protocol specifications."""
        return [
            {
                "protocol_id": "COA-SEC-GUN-002",
                "name": "Security Response to Gunshot",
                "event_type": "gunshot_detected",
                "threat_category": ThreatCategory.ACTIVE_SHOOTER,
                "priority": ProtocolPriority.CRITICAL,
                "status": ProtocolStatus.ACTIVE,
                "description": "Manual security response to gunshot detection events",
                "security_goal": "Protect occupants, contain shooter, aid victims",
                "security_phases": [
                    {
                        "phase": "Alarm & Notify",
                        "duration": "0-5s",
                        "actions": [
                            "Trigger audible alarm",
                            "Call law-enforcement dispatch",
                        ],
                        "resources": ["alarm_system", "communication_system"],
                        "success_criteria": ["alarm_activated", "le_notified"],
                    },
                    {
                        "phase": "Lockdown",
                        "duration": "5-20s",
                        "actions": [
                            "Secure exterior doors",
                            "Direct employees to shelter",
                        ],
                        "resources": ["access_control_system", "security_personnel"],
                        "success_criteria": ["doors_locked", "personnel_sheltered"],
                    },
                    {
                        "phase": "Contain",
                        "duration": "20-60s",
                        "actions": [
                            "Take cover near entrances",
                            "Provide first aid if safe",
                        ],
                        "resources": ["security_personnel", "first_aid_kits"],
                        "success_criteria": ["shooter_contained", "injured_aided"],
                    },
                ],
                "security_roe_code": "ROE-SEC-112-SELFDEF",
                "security_roe_description": "Security personnel may employ non-lethal force to protect life and detain the shooter; lethal force is justified only when the shooter poses an imminent threat of death or serious injury.",
                "security_force_level": "non_lethal_force",
                "security_restrictions": [
                    "Actions must respect human dignity",
                    "Use force strictly when necessary",
                ],
                "le_goal": "Neutralize armed suspect and rescue victims",
                "le_phases": [
                    {
                        "phase": "Dispatch & Approach",
                        "duration": "0-2min",
                        "actions": [
                            "Armed patrol/SWAT dispatched",
                            "Form outer perimeter",
                        ],
                        "resources": ["patrol_units", "swat_teams"],
                        "success_criteria": [
                            "units_dispatched",
                            "perimeter_established",
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
                        "resources": ["armed_officers", "communication_equipment"],
                        "success_criteria": ["suspect_neutralized", "scene_secured"],
                    },
                ],
                "le_roe_code": "ROE-LE-112-DEADLY",
                "le_roe_description": "Officers may use deadly force when they have probable cause to believe the suspect poses a threat of serious physical harm to officers or others.",
                "le_force_level": "lethal_force",
                "le_restrictions": ["Force must stop once the threat abates"],
                "response_time_seconds": 10,
                "escalation_path": [
                    "Security-Guard",
                    "Security-Super",
                    "LE Dispatch",
                    "Emergency Team",
                ],
                "compliance_standards": [
                    "UN_CODE_OF_CONDUCT",
                    "US_CONSTITUTIONAL_USE_OF_FORCE",
                ],
                "required_resources": [
                    "security_personnel",
                    "communication_systems",
                    "access_control_systems",
                    "first_aid_equipment",
                ],
                "success_criteria": [
                    "threat_neutralized",
                    "personnel_safe",
                    "scene_secured",
                    "investigation_initiated",
                ],
                "tags": ["violence", "active_shooter", "critical_response"],
            },
            {
                "protocol_id": "COA-SEC-EXP-002",
                "name": "Security Response to Explosion",
                "event_type": "explosion_detected",
                "threat_category": ThreatCategory.PHYSICAL_SECURITY,
                "priority": ProtocolPriority.HIGH,
                "status": ProtocolStatus.ACTIVE,
                "description": "Emergency response to explosion detection events",
                "security_goal": "Evacuate occupants and preserve life",
                "security_phases": [
                    {
                        "phase": "Alarm & Evacuate",
                        "duration": "0-15s",
                        "actions": [
                            "Activate distinctive alarm",
                            "Announce evacuation",
                            "Escort to designated exits",
                        ],
                        "resources": [
                            "alarm_system",
                            "pa_system",
                            "security_personnel",
                        ],
                        "success_criteria": ["alarm_activated", "evacuation_announced"],
                    },
                    {
                        "phase": "Shut Utilities",
                        "duration": "15-60s",
                        "actions": ["Shut off gas and electricity if safe"],
                        "resources": ["utility_controls", "safety_equipment"],
                        "success_criteria": ["utilities_shut_off"],
                    },
                ],
                "security_roe_code": "ROE-SEC-EVAC",
                "security_roe_description": "Focus on safe evacuation rather than force; use of force is limited to protecting evacuees from harm.",
                "security_force_level": "no_force",
                "security_restrictions": ["OSHA emergency-action standards apply"],
                "le_goal": "Secure scene, assist victims, and investigate",
                "le_phases": [
                    {
                        "phase": "Respond & Secure",
                        "duration": "0-5min",
                        "actions": ["Establish outer perimeter", "Reroute traffic"],
                        "resources": ["patrol_units", "traffic_control"],
                        "success_criteria": [
                            "perimeter_established",
                            "traffic_rerouted",
                        ],
                    }
                ],
                "le_roe_code": "ROE-LE-EVAC",
                "le_roe_description": "Prioritize life safety; deadly force authorized only if suspect poses imminent lethal threat.",
                "le_force_level": "non_lethal_force",
                "le_restrictions": ["Use force only when strictly necessary"],
                "response_time_seconds": 5,
                "escalation_path": [
                    "Site-Manager",
                    "Security-Super",
                    "Fire Dept",
                    "Bomb Squad",
                ],
                "compliance_standards": ["OSHA_EMERGENCY_ACTION", "UN_CODE_OF_CONDUCT"],
                "required_resources": [
                    "alarm_systems",
                    "evacuation_routes",
                    "communication_systems",
                    "emergency_services",
                ],
                "success_criteria": [
                    "personnel_evacuated",
                    "scene_secured",
                    "emergency_services_notified",
                    "investigation_initiated",
                ],
                "tags": ["emergency", "explosion", "evacuation"],
            },
            {
                "protocol_id": "COA-SEC-TRES-002",
                "name": "Security Response to Trespasser",
                "event_type": "trespassing_detection",
                "threat_category": ThreatCategory.PHYSICAL_SECURITY,
                "priority": ProtocolPriority.MEDIUM,
                "status": ProtocolStatus.ACTIVE,
                "description": "Response to unauthorized access detection",
                "security_goal": "Prevent unauthorized access and hand over intruder",
                "security_phases": [
                    {
                        "phase": "Intercept",
                        "duration": "0-15s",
                        "actions": [
                            "Approach intruder",
                            "Issue clear verbal challenge",
                            "Ask for identification",
                        ],
                        "resources": ["security_personnel", "communication_equipment"],
                        "success_criteria": ["intruder_identified", "challenge_issued"],
                    },
                    {
                        "phase": "Contain",
                        "duration": "15-60s",
                        "actions": [
                            "Use non-lethal tools if fleeing",
                            "Detain if necessary",
                        ],
                        "resources": ["security_personnel", "non_lethal_equipment"],
                        "success_criteria": ["intruder_detained"],
                    },
                ],
                "security_roe_code": "ROE-SEC-211-MINIMAL",
                "security_roe_description": "Security officers may use force only when strictly necessary and proportionate. Lethal force is prohibited unless the intruder threatens life.",
                "security_force_level": "minimal_force",
                "security_restrictions": [
                    "Use minimum force necessary",
                    "Lethal force prohibited unless life threatened",
                ],
                "le_goal": "Apprehend intruder and determine intent",
                "le_phases": [
                    {
                        "phase": "Dispatch & Assess",
                        "duration": "0-3min",
                        "actions": [
                            "Patrol unit arrives",
                            "Identify trespasser",
                            "Check for weapons",
                        ],
                        "resources": ["patrol_units", "search_equipment"],
                        "success_criteria": ["unit_arrived", "assessment_complete"],
                    }
                ],
                "le_roe_code": "ROE-LE-211-DETENTION",
                "le_roe_description": "Officers must use the minimum force necessary to effect arrest or detention; deadly force is permissible only where the suspect poses an immediate threat of serious harm.",
                "le_force_level": "minimal_force",
                "le_restrictions": [
                    "Use minimum force necessary",
                    "Deadly force only for immediate threat",
                ],
                "response_time_seconds": 30,
                "escalation_path": ["Security-Guard", "Security-Super", "LE Patrol"],
                "compliance_standards": ["UN_CODE_OF_CONDUCT"],
                "required_resources": [
                    "security_personnel",
                    "communication_systems",
                    "patrol_units",
                ],
                "success_criteria": [
                    "intruder_apprehended",
                    "intent_determined",
                    "charges_filed",
                ],
                "tags": ["intrusion", "trespassing", "detention"],
            },
        ]

    def _create_protocol_from_data(self, data: Dict[str, Any]) -> ResponseProtocol:
        """Create a ResponseProtocol from specification data."""
        return ResponseProtocol(
            protocol_id=data["protocol_id"],
            name=data["name"],
            event_type=data["event_type"],
            threat_category=data["threat_category"],
            priority=data["priority"],
            status=data["status"],
            description=data["description"],
            security_goal=data["security_goal"],
            security_phases=data["security_phases"],
            security_roe_code=data["security_roe_code"],
            security_roe_description=data["security_roe_description"],
            security_force_level=data["security_force_level"],
            security_restrictions=data["security_restrictions"],
            le_goal=data["le_goal"],
            le_phases=data["le_phases"],
            le_roe_code=data["le_roe_code"],
            le_roe_description=data["le_roe_description"],
            le_force_level=data["le_force_level"],
            le_restrictions=data["le_restrictions"],
            response_time_seconds=data["response_time_seconds"],
            escalation_path=data["escalation_path"],
            compliance_standards=data["compliance_standards"],
            required_resources=data["required_resources"],
            success_criteria=data["success_criteria"],
            tags=data.get("tags", []),
        )

    async def start(self):
        """Start the protocol manager."""
        self.logger.info("Starting Response Protocol Manager")

    async def stop(self):
        """Stop the protocol manager."""
        self.logger.info("Stopping Response Protocol Manager")

    async def get_protocol(self, protocol_id: str) -> Optional[ResponseProtocol]:
        """Get a specific protocol by ID."""
        return self.protocols.get(protocol_id)

    async def get_protocols_by_event_type(
        self, event_type: str
    ) -> List[ResponseProtocol]:
        """Get protocols that match an event type."""
        return [
            protocol
            for protocol in self.protocols.values()
            if protocol.event_type == event_type
            and protocol.status == ProtocolStatus.ACTIVE
        ]

    async def get_protocols_by_category(
        self, threat_category: ThreatCategory
    ) -> List[ResponseProtocol]:
        """Get protocols that match a threat category."""
        return [
            protocol
            for protocol in self.protocols.values()
            if protocol.threat_category == threat_category
            and protocol.status == ProtocolStatus.ACTIVE
        ]

    async def get_all_protocols(self) -> List[ResponseProtocol]:
        """Get all active protocols."""
        return [
            protocol
            for protocol in self.protocols.values()
            if protocol.status == ProtocolStatus.ACTIVE
        ]

    async def create_execution(
        self,
        protocol: ResponseProtocol,
        threat_event: ThreatEvent,
        executed_by: Optional[str] = None,
    ) -> ProtocolExecution:
        """Create a new protocol execution."""
        execution_id = (
            f"exec_{protocol.protocol_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        execution = ProtocolExecution(
            execution_id=execution_id,
            protocol=protocol,
            threat_event=threat_event,
            start_time=datetime.now(),
            executed_by=executed_by,
        )

        self.active_executions[execution_id] = execution
        self.logger.info(f"Created protocol execution: {execution_id}")

        return execution

    async def execute_phase(self, execution_id: str, phase_index: int) -> bool:
        """Execute a specific phase of a protocol."""
        if execution_id not in self.active_executions:
            self.logger.error(f"Execution not found: {execution_id}")
            return False

        execution = self.active_executions[execution_id]
        protocol = execution.protocol

        # Check if phase is valid
        total_phases = len(protocol.security_phases) + len(protocol.le_phases)
        if phase_index >= total_phases:
            self.logger.error(f"Invalid phase index: {phase_index}")
            return False

        # Execute phase
        try:
            if phase_index < len(protocol.security_phases):
                await self._execute_security_phase(execution, phase_index)
            else:
                le_phase_index = phase_index - len(protocol.security_phases)
                await self._execute_le_phase(execution, le_phase_index)

            execution.completed_phases.append(phase_index)
            execution.current_phase = phase_index + 1

            # Check if all phases are complete
            if len(execution.completed_phases) == total_phases:
                execution.status = "completed"
                self.execution_history.append(execution)
                del self.active_executions[execution_id]

            self.logger.info(
                f"Executed phase {phase_index} for execution: {execution_id}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error executing phase {phase_index}: {e}")
            execution.status = "failed"
            return False

    async def _execute_security_phase(
        self, execution: ProtocolExecution, phase_index: int
    ):
        """Execute a security phase."""
        protocol = execution.protocol
        phase = protocol.security_phases[phase_index]

        self.logger.info(f"Executing security phase: {phase['phase']}")

        # Simulate phase execution
        await asyncio.sleep(1)

        # Add execution note
        execution.notes.append(
            f"Security phase {phase_index} completed: {phase['phase']}"
        )

    async def _execute_le_phase(self, execution: ProtocolExecution, phase_index: int):
        """Execute a law enforcement phase."""
        protocol = execution.protocol
        phase = protocol.le_phases[phase_index]

        self.logger.info(f"Executing LE phase: {phase['phase']}")

        # Simulate phase execution
        await asyncio.sleep(1)

        # Add execution note
        execution.notes.append(f"LE phase {phase_index} completed: {phase['phase']}")

    async def get_active_executions(self) -> List[ProtocolExecution]:
        """Get all active protocol executions."""
        return list(self.active_executions.values())

    async def get_execution_history(self, limit: int = 100) -> List[ProtocolExecution]:
        """Get protocol execution history."""
        return self.execution_history[-limit:]

    async def add_protocol(self, protocol: ResponseProtocol) -> bool:
        """Add a new protocol."""
        if protocol.protocol_id in self.protocols:
            self.logger.warning(f"Protocol already exists: {protocol.protocol_id}")
            return False

        self.protocols[protocol.protocol_id] = protocol
        self.logger.info(f"Added new protocol: {protocol.protocol_id}")
        return True

    async def update_protocol(self, protocol_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing protocol."""
        if protocol_id not in self.protocols:
            self.logger.error(f"Protocol not found: {protocol_id}")
            return False

        protocol = self.protocols[protocol_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(protocol, key):
                setattr(protocol, key, value)

        protocol.updated_at = datetime.now()

        self.logger.info(f"Updated protocol: {protocol_id}")
        return True

    async def deactivate_protocol(self, protocol_id: str) -> bool:
        """Deactivate a protocol."""
        if protocol_id not in self.protocols:
            self.logger.error(f"Protocol not found: {protocol_id}")
            return False

        protocol = self.protocols[protocol_id]
        protocol.status = ProtocolStatus.INACTIVE
        protocol.updated_at = datetime.now()

        self.logger.info(f"Deactivated protocol: {protocol_id}")
        return True
