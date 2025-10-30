"""
FLAGSHIP Escalation Manager

Manages escalation paths for COA/ROE responses without direct external communications.
All escalations go through the FLAGSHIP system first for validation and control.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

logger = get_logger(__name__)


# ============================================================================
# ESCALATION DATA STRUCTURES
# ============================================================================


class EscalationLevel(Enum):
    """Escalation levels within the FLAGSHIP system."""

    LEVEL_1 = "level_1"  # Initial response
    LEVEL_2 = "level_2"  # Supervisor notification
    LEVEL_3 = "level_3"  # Management escalation
    LEVEL_4 = "level_4"  # Executive escalation
    LEVEL_5 = "level_5"  # External authority (through FLAGSHIP only)


class EscalationStatus(Enum):
    """Status of escalation requests."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EscalationType(Enum):
    """Types of escalation."""

    SECURITY_ESCALATION = "security_escalation"
    LAW_ENFORCEMENT_ESCALATION = "law_enforcement_escalation"
    EMERGENCY_SERVICES_ESCALATION = "emergency_services_escalation"
    MANAGEMENT_ESCALATION = "management_escalation"
    EXECUTIVE_ESCALATION = "executive_escalation"


@dataclass
class EscalationTarget:
    """Target for escalation within the FLAGSHIP system."""

    target_id: str
    name: str
    escalation_type: EscalationType
    escalation_level: EscalationLevel
    contact_method: str  # Internal FLAGSHIP method only
    response_time_seconds: int
    capabilities: List[str]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EscalationRequest:
    """Escalation request within the FLAGSHIP system."""

    request_id: str
    execution_id: str
    escalation_type: EscalationType
    escalation_level: EscalationLevel
    target: EscalationTarget
    reason: str
    urgency: str
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    status: EscalationStatus = EscalationStatus.PENDING
    acknowledged_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response_notes: Optional[str] = None


@dataclass
class EscalationPath:
    """Escalation path configuration."""

    path_id: str
    name: str
    description: str
    levels: List[EscalationLevel]
    targets: List[EscalationTarget]
    auto_escalate: bool = True
    escalation_delay_seconds: int = 30
    max_escalation_level: EscalationLevel = EscalationLevel.LEVEL_5
    created_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# ESCALATION MANAGER
# ============================================================================


class EscalationManager:
    """Manages escalation paths within the FLAGSHIP system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the escalation manager."""
        self.config = config or {}
        self.logger = logger

        # Escalation targets and paths
        self.escalation_targets: Dict[str, EscalationTarget] = {}
        self.escalation_paths: Dict[str, EscalationPath] = {}
        self.active_requests: Dict[str, EscalationRequest] = {}
        self.request_history: List[EscalationRequest] = []

        # Initialize escalation targets and paths
        self._initialize_escalation_targets()
        self._initialize_escalation_paths()

        self.logger.info("Escalation Manager initialized")

    def _initialize_escalation_targets(self):
        """Initialize escalation targets within the FLAGSHIP system."""
        targets_data = self._get_escalation_targets()

        for target_data in targets_data:
            target = self._create_target_from_data(target_data)
            self.escalation_targets[target.target_id] = target

        self.logger.info(f"Loaded {len(self.escalation_targets)} escalation targets")

    def _get_escalation_targets(self) -> List[Dict[str, Any]]:
        """Get escalation targets configuration."""
        return [
            {
                "target_id": "security_guard",
                "name": "Security Guard",
                "escalation_type": EscalationType.SECURITY_ESCALATION,
                "escalation_level": EscalationLevel.LEVEL_1,
                "contact_method": "flagship_internal_alert",
                "response_time_seconds": 10,
                "capabilities": [
                    "immediate_response",
                    "scene_assessment",
                    "basic_containment",
                ],
            },
            {
                "target_id": "security_supervisor",
                "name": "Security Supervisor",
                "escalation_type": EscalationType.SECURITY_ESCALATION,
                "escalation_level": EscalationLevel.LEVEL_2,
                "contact_method": "flagship_internal_alert",
                "response_time_seconds": 30,
                "capabilities": ["supervision", "resource_allocation", "coordination"],
            },
            {
                "target_id": "site_manager",
                "name": "Site Manager",
                "escalation_type": EscalationType.MANAGEMENT_ESCALATION,
                "escalation_level": EscalationLevel.LEVEL_3,
                "contact_method": "flagship_internal_alert",
                "response_time_seconds": 60,
                "capabilities": [
                    "decision_making",
                    "authority",
                    "external_coordination",
                ],
            },
            {
                "target_id": "flagship_le_coordinator",
                "name": "FLAGSHIP Law Enforcement Coordinator",
                "escalation_type": EscalationType.LAW_ENFORCEMENT_ESCALATION,
                "escalation_level": EscalationLevel.LEVEL_4,
                "contact_method": "flagship_internal_alert",
                "response_time_seconds": 120,
                "capabilities": [
                    "le_coordination",
                    "warrant_processing",
                    "investigation_support",
                ],
            },
            {
                "target_id": "flagship_emergency_coordinator",
                "name": "FLAGSHIP Emergency Services Coordinator",
                "escalation_type": EscalationType.EMERGENCY_SERVICES_ESCALATION,
                "escalation_level": EscalationLevel.LEVEL_4,
                "contact_method": "flagship_internal_alert",
                "response_time_seconds": 60,
                "capabilities": [
                    "emergency_coordination",
                    "medical_support",
                    "fire_services",
                ],
            },
            {
                "target_id": "flagship_executive",
                "name": "FLAGSHIP Executive",
                "escalation_type": EscalationType.EXECUTIVE_ESCALATION,
                "escalation_level": EscalationLevel.LEVEL_5,
                "contact_method": "flagship_internal_alert",
                "response_time_seconds": 300,
                "capabilities": [
                    "executive_authority",
                    "external_authority_contact",
                    "policy_decision",
                ],
            },
        ]

    def _create_target_from_data(self, data: Dict[str, Any]) -> EscalationTarget:
        """Create an EscalationTarget from specification data."""
        return EscalationTarget(
            target_id=data["target_id"],
            name=data["name"],
            escalation_type=data["escalation_type"],
            escalation_level=data["escalation_level"],
            contact_method=data["contact_method"],
            response_time_seconds=data["response_time_seconds"],
            capabilities=data["capabilities"],
        )

    def _initialize_escalation_paths(self):
        """Initialize escalation paths."""
        paths_data = self._get_escalation_paths()

        for path_data in paths_data:
            path = self._create_path_from_data(path_data)
            self.escalation_paths[path.path_id] = path

        self.logger.info(f"Loaded {len(self.escalation_paths)} escalation paths")

    def _get_escalation_paths(self) -> List[Dict[str, Any]]:
        """Get escalation paths configuration."""
        return [
            {
                "path_id": "critical_threat_path",
                "name": "Critical Threat Escalation Path",
                "description": "Escalation path for critical threats requiring immediate response",
                "levels": [
                    EscalationLevel.LEVEL_1,
                    EscalationLevel.LEVEL_2,
                    EscalationLevel.LEVEL_3,
                    EscalationLevel.LEVEL_4,
                    EscalationLevel.LEVEL_5,
                ],
                "targets": [
                    self.escalation_targets["security_guard"],
                    self.escalation_targets["security_supervisor"],
                    self.escalation_targets["site_manager"],
                    self.escalation_targets["flagship_le_coordinator"],
                    self.escalation_targets["flagship_executive"],
                ],
                "auto_escalate": True,
                "escalation_delay_seconds": 15,
                "max_escalation_level": EscalationLevel.LEVEL_5,
            },
            {
                "path_id": "emergency_response_path",
                "name": "Emergency Response Escalation Path",
                "description": "Escalation path for emergency situations",
                "levels": [
                    EscalationLevel.LEVEL_1,
                    EscalationLevel.LEVEL_2,
                    EscalationLevel.LEVEL_3,
                    EscalationLevel.LEVEL_4,
                ],
                "targets": [
                    self.escalation_targets["security_guard"],
                    self.escalation_targets["security_supervisor"],
                    self.escalation_targets["flagship_emergency_coordinator"],
                    self.escalation_targets["site_manager"],
                ],
                "auto_escalate": True,
                "escalation_delay_seconds": 30,
                "max_escalation_level": EscalationLevel.LEVEL_4,
            },
            {
                "path_id": "standard_response_path",
                "name": "Standard Response Escalation Path",
                "description": "Standard escalation path for routine threats",
                "levels": [
                    EscalationLevel.LEVEL_1,
                    EscalationLevel.LEVEL_2,
                    EscalationLevel.LEVEL_3,
                ],
                "targets": [
                    self.escalation_targets["security_guard"],
                    self.escalation_targets["security_supervisor"],
                    self.escalation_targets["site_manager"],
                ],
                "auto_escalate": False,
                "escalation_delay_seconds": 60,
                "max_escalation_level": EscalationLevel.LEVEL_3,
            },
        ]

    def _create_path_from_data(self, data: Dict[str, Any]) -> EscalationPath:
        """Create an EscalationPath from specification data."""
        return EscalationPath(
            path_id=data["path_id"],
            name=data["name"],
            description=data["description"],
            levels=data["levels"],
            targets=data["targets"],
            auto_escalate=data["auto_escalate"],
            escalation_delay_seconds=data["escalation_delay_seconds"],
            max_escalation_level=data["max_escalation_level"],
        )

    async def start(self):
        """Start the escalation manager."""
        self.logger.info("Starting Escalation Manager")

    async def stop(self):
        """Stop the escalation manager."""
        self.logger.info("Stopping Escalation Manager")

    async def initiate_escalation(
        self,
        execution_id: str,
        escalation_path_id: str,
        reason: str,
        urgency: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Initiate escalation through the FLAGSHIP system."""
        if escalation_path_id not in self.escalation_paths:
            self.logger.error(f"Escalation path not found: {escalation_path_id}")
            return None

        path = self.escalation_paths[escalation_path_id]
        request_id = (
            f"escalation_{execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create escalation request for the first level
        first_target = path.targets[0]
        request = EscalationRequest(
            request_id=request_id,
            execution_id=execution_id,
            escalation_type=first_target.escalation_type,
            escalation_level=first_target.escalation_level,
            target=first_target,
            reason=reason,
            urgency=urgency,
            context=context or {},
        )

        self.active_requests[request_id] = request

        # Send internal FLAGSHIP alert (no external communication)
        await self._send_internal_escalation_alert(request)

        self.logger.info(f"Initiated escalation: {request_id} to {first_target.name}")

        return request_id

    async def _send_internal_escalation_alert(self, request: EscalationRequest):
        """Send escalation alert through FLAGSHIP internal system only."""
        alert_data = {
            "type": "escalation_alert",
            "request_id": request.request_id,
            "target": request.target.name,
            "escalation_level": request.escalation_level.value,
            "reason": request.reason,
            "urgency": request.urgency,
            "timestamp": datetime.now().isoformat(),
            "context": request.context,
        }

        # Log the alert (in real implementation, this would go to FLAGSHIP's internal notification system)
        self.logger.info(f"FLAGSHIP Internal Escalation Alert: {alert_data}")

        # Simulate internal processing
        await asyncio.sleep(0.1)

    async def acknowledge_escalation(
        self, request_id: str, acknowledged_by: str, notes: Optional[str] = None
    ) -> bool:
        """Acknowledge an escalation request."""
        if request_id not in self.active_requests:
            self.logger.error(f"Escalation request not found: {request_id}")
            return False

        request = self.active_requests[request_id]
        request.status = EscalationStatus.IN_PROGRESS
        request.acknowledged_at = datetime.now()
        request.response_notes = notes

        self.logger.info(f"Escalation acknowledged: {request_id} by {acknowledged_by}")
        return True

    async def complete_escalation(
        self, request_id: str, completion_notes: Optional[str] = None
    ) -> bool:
        """Complete an escalation request."""
        if request_id not in self.active_requests:
            self.logger.error(f"Escalation request not found: {request_id}")
            return False

        request = self.active_requests[request_id]
        request.status = EscalationStatus.COMPLETED
        request.completed_at = datetime.now()
        if completion_notes:
            request.response_notes = completion_notes

        # Move to history
        self.request_history.append(request)
        del self.active_requests[request_id]

        self.logger.info(f"Escalation completed: {request_id}")
        return True

    async def auto_escalate(self, request_id: str) -> Optional[str]:
        """Automatically escalate to the next level if configured."""
        if request_id not in self.active_requests:
            self.logger.error(f"Escalation request not found: {request_id}")
            return None

        current_request = self.active_requests[request_id]

        # Find the escalation path
        path = None
        for p in self.escalation_paths.values():
            if current_request.target in p.targets:
                path = p
                break

        if not path or not path.auto_escalate:
            return None

        # Find next target in the path
        current_index = path.targets.index(current_request.target)
        if current_index + 1 >= len(path.targets):
            self.logger.info(
                f"Maximum escalation level reached for request: {request_id}"
            )
            return None

        next_target = path.targets[current_index + 1]

        # Create new escalation request
        next_request_id = f"escalation_{current_request.execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        next_request = EscalationRequest(
            request_id=next_request_id,
            execution_id=current_request.execution_id,
            escalation_type=next_target.escalation_type,
            escalation_level=next_target.escalation_level,
            target=next_target,
            reason=f"Auto-escalation from {current_request.target.name}",
            urgency=current_request.urgency,
            context=current_request.context,
        )

        self.active_requests[next_request_id] = next_request

        # Send internal alert
        await self._send_internal_escalation_alert(next_request)

        self.logger.info(f"Auto-escalated to {next_target.name}: {next_request_id}")
        return next_request_id

    async def get_active_escalations(self) -> List[EscalationRequest]:
        """Get all active escalation requests."""
        return list(self.active_requests.values())

    async def get_escalation_history(self, limit: int = 100) -> List[EscalationRequest]:
        """Get escalation request history."""
        return self.request_history[-limit:]

    async def get_escalation_paths(self) -> List[EscalationPath]:
        """Get all available escalation paths."""
        return list(self.escalation_paths.values())

    async def get_escalation_targets(self) -> List[EscalationTarget]:
        """Get all available escalation targets."""
        return list(self.escalation_targets.values())

    async def add_escalation_target(self, target: EscalationTarget) -> bool:
        """Add a new escalation target."""
        if target.target_id in self.escalation_targets:
            self.logger.warning(f"Escalation target already exists: {target.target_id}")
            return False

        self.escalation_targets[target.target_id] = target
        self.logger.info(f"Added escalation target: {target.target_id}")
        return True

    async def add_escalation_path(self, path: EscalationPath) -> bool:
        """Add a new escalation path."""
        if path.path_id in self.escalation_paths:
            self.logger.warning(f"Escalation path already exists: {path.path_id}")
            return False

        self.escalation_paths[path.path_id] = path
        self.logger.info(f"Added escalation path: {path.path_id}")
        return True

    async def cancel_escalation(self, request_id: str, reason: str) -> bool:
        """Cancel an escalation request."""
        if request_id not in self.active_requests:
            self.logger.error(f"Escalation request not found: {request_id}")
            return False

        request = self.active_requests[request_id]
        request.status = EscalationStatus.CANCELLED
        request.response_notes = f"Cancelled: {reason}"

        # Move to history
        self.request_history.append(request)
        del self.active_requests[request_id]

        self.logger.info(f"Escalation cancelled: {request_id} - {reason}")
        return True
