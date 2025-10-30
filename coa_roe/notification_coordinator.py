"""
FLAGSHIP Notification Coordinator

Coordinates all notifications and alerts through the FLAGSHIP system first.
Prevents direct external communications to avoid false positive issues and confusion.

CRITICAL SAFETY FEATURE:
- NO direct emails to government agencies
- NO direct external communications during testing
- ALL notifications go through FLAGSHIP internal system first
- External communications only after FLAGSHIP validation and approval
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

logger = get_logger(__name__)


# ============================================================================
# NOTIFICATION DATA STRUCTURES
# ============================================================================


class NotificationType(Enum):
    """Types of notifications within the FLAGSHIP system."""

    INTERNAL_ALERT = "internal_alert"
    ACTION_NOTIFICATION = "action_notification"
    ESCALATION_ALERT = "escalation_alert"
    COMPLIANCE_ALERT = "compliance_alert"
    SYSTEM_STATUS = "system_status"
    TEST_NOTIFICATION = "test_notification"


class NotificationPriority(Enum):
    """Priority levels for notifications."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class NotificationStatus(Enum):
    """Status of notifications."""

    PENDING = "pending"
    SENT_INTERNAL = "sent_internal"
    APPROVED = "approved"
    SENT_EXTERNAL = "sent_external"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class NotificationChannel(Enum):
    """Notification channels within FLAGSHIP system."""

    FLAGSHIP_DASHBOARD = "flagship_dashboard"
    FLAGSHIP_API = "flagship_api"
    FLAGSHIP_WEBSOCKET = "flagship_websocket"
    FLAGSHIP_LOG = "flagship_log"
    FLAGSHIP_AUDIT = "flagship_audit"


@dataclass
class NotificationMessage:
    """Notification message within the FLAGSHIP system."""

    message_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    content: str
    execution_id: Optional[str] = None
    action_id: Optional[str] = None
    target_entities: List[str] = field(default_factory=list)
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: NotificationStatus = NotificationStatus.PENDING
    sent_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    external_sent_at: Optional[datetime] = None


@dataclass
class NotificationTemplate:
    """Template for notifications."""

    template_id: str
    name: str
    notification_type: NotificationType
    title_template: str
    content_template: str
    default_priority: NotificationPriority
    default_channels: List[NotificationChannel]
    requires_approval: bool = True
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# NOTIFICATION COORDINATOR
# ============================================================================


class NotificationCoordinator:
    """Coordinates all notifications through the FLAGSHIP system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the notification coordinator."""
        self.config = config or {}
        self.logger = logger

        # Notification management
        self.notification_templates: Dict[str, NotificationTemplate] = {}
        self.pending_notifications: Dict[str, NotificationMessage] = {}
        self.sent_notifications: List[NotificationMessage] = []

        # Safety controls
        self.external_communications_enabled = self.config.get(
            "external_communications_enabled", False
        )
        self.test_mode = self.config.get("test_mode", True)
        self.require_approval = self.config.get("require_approval", True)

        # Initialize templates
        self._initialize_notification_templates()

        self.logger.info("Notification Coordinator initialized")
        self.logger.warning(
            "EXTERNAL COMMUNICATIONS DISABLED - All notifications go through FLAGSHIP system first"
        )

    def _initialize_notification_templates(self):
        """Initialize notification templates."""
        templates_data = self._get_notification_templates()

        for template_data in templates_data:
            template = self._create_template_from_data(template_data)
            self.notification_templates[template.template_id] = template

        self.logger.info(
            f"Loaded {len(self.notification_templates)} notification templates"
        )

    def _get_notification_templates(self) -> List[Dict[str, Any]]:
        """Get notification templates configuration."""
        return [
            {
                "template_id": "action_execution",
                "name": "Action Execution Notification",
                "notification_type": NotificationType.ACTION_NOTIFICATION,
                "title_template": "COA/ROE Action Executed: {action_description}",
                "content_template": "Action {action_id} has been executed as part of response {execution_id}. Status: {status}",
                "default_priority": NotificationPriority.MEDIUM,
                "default_channels": [
                    NotificationChannel.FLAGSHIP_DASHBOARD,
                    NotificationChannel.FLAGSHIP_LOG,
                ],
                "requires_approval": False,
            },
            {
                "template_id": "escalation_alert",
                "name": "Escalation Alert",
                "notification_type": NotificationType.ESCALATION_ALERT,
                "title_template": "Escalation Required: {escalation_level}",
                "content_template": "Escalation to {target_name} required for {reason}. Urgency: {urgency}",
                "default_priority": NotificationPriority.HIGH,
                "default_channels": [
                    NotificationChannel.FLAGSHIP_DASHBOARD,
                    NotificationChannel.FLAGSHIP_API,
                    NotificationChannel.FLAGSHIP_WEBSOCKET,
                ],
                "requires_approval": True,
            },
            {
                "template_id": "compliance_violation",
                "name": "Compliance Violation Alert",
                "notification_type": NotificationType.COMPLIANCE_ALERT,
                "title_template": "Compliance Violation Detected: {violation_type}",
                "content_template": "Compliance violation detected in action {action_id}. Severity: {severity}. Description: {description}",
                "default_priority": NotificationPriority.CRITICAL,
                "default_channels": [
                    NotificationChannel.FLAGSHIP_DASHBOARD,
                    NotificationChannel.FLAGSHIP_AUDIT,
                    NotificationChannel.FLAGSHIP_LOG,
                ],
                "requires_approval": True,
            },
            {
                "template_id": "critical_threat",
                "name": "Critical Threat Alert",
                "notification_type": NotificationType.INTERNAL_ALERT,
                "title_template": "CRITICAL THREAT DETECTED: {threat_type}",
                "content_template": "Critical threat {threat_id} detected. Immediate response required. Threat level: {threat_level}",
                "default_priority": NotificationPriority.CRITICAL,
                "default_channels": [
                    NotificationChannel.FLAGSHIP_DASHBOARD,
                    NotificationChannel.FLAGSHIP_API,
                    NotificationChannel.FLAGSHIP_WEBSOCKET,
                    NotificationChannel.FLAGSHIP_LOG,
                ],
                "requires_approval": True,
            },
            {
                "template_id": "test_notification",
                "name": "Test Notification",
                "notification_type": NotificationType.TEST_NOTIFICATION,
                "title_template": "TEST: {test_type}",
                "content_template": "This is a test notification for {test_type}. No external communications will be sent.",
                "default_priority": NotificationPriority.INFO,
                "default_channels": [NotificationChannel.FLAGSHIP_LOG],
                "requires_approval": False,
            },
        ]

    def _create_template_from_data(self, data: Dict[str, Any]) -> NotificationTemplate:
        """Create a NotificationTemplate from specification data."""
        return NotificationTemplate(
            template_id=data["template_id"],
            name=data["name"],
            notification_type=data["notification_type"],
            title_template=data["title_template"],
            content_template=data["content_template"],
            default_priority=data["default_priority"],
            default_channels=data["default_channels"],
            requires_approval=data.get("requires_approval", True),
        )

    async def start(self):
        """Start the notification coordinator."""
        self.logger.info("Starting Notification Coordinator")
        if self.test_mode:
            self.logger.warning(
                "TEST MODE ACTIVE - No external communications will be sent"
            )

    async def stop(self):
        """Stop the notification coordinator."""
        self.logger.info("Stopping Notification Coordinator")

    async def send_action_notification(self, action, execution) -> str:
        """Send notification for action execution through FLAGSHIP system."""
        template = self.notification_templates.get("action_execution")
        if not template:
            self.logger.error("Action execution template not found")
            return None

        message_id = (
            f"notif_{action.action_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Format message using template
        title = template.title_template.format(action_description=action.description)
        content = template.content_template.format(
            action_id=action.action_id,
            execution_id=execution.execution_id,
            status="executing",
        )

        message = NotificationMessage(
            message_id=message_id,
            notification_type=template.notification_type,
            priority=template.default_priority,
            title=title,
            content=content,
            execution_id=execution.execution_id,
            action_id=action.action_id,
            target_entities=action.target_entities,
            channels=template.default_channels,
            metadata={
                "action_type": action.action_type.value,
                "force_level": action.force_level.value,
                "protocol_id": execution.protocol.protocol_id,
            },
        )

        # Send through FLAGSHIP system only
        await self._send_internal_notification(message)

        self.logger.info(f"Action notification sent through FLAGSHIP: {message_id}")
        return message_id

    async def send_escalation_notification(self, escalation_request) -> str:
        """Send escalation notification through FLAGSHIP system."""
        template = self.notification_templates.get("escalation_alert")
        if not template:
            self.logger.error("Escalation alert template not found")
            return None

        message_id = f"notif_escalation_{escalation_request.request_id}"

        title = template.title_template.format(
            escalation_level=escalation_request.escalation_level.value
        )
        content = template.content_template.format(
            target_name=escalation_request.target.name,
            reason=escalation_request.reason,
            urgency=escalation_request.urgency,
        )

        message = NotificationMessage(
            message_id=message_id,
            notification_type=template.notification_type,
            priority=template.default_priority,
            title=title,
            content=content,
            execution_id=escalation_request.execution_id,
            target_entities=[escalation_request.target.name],
            channels=template.default_channels,
            metadata={
                "escalation_type": escalation_request.escalation_type.value,
                "escalation_level": escalation_request.escalation_level.value,
                "target_id": escalation_request.target.target_id,
            },
        )

        # Send through FLAGSHIP system only
        await self._send_internal_notification(message)

        self.logger.info(f"Escalation notification sent through FLAGSHIP: {message_id}")
        return message_id

    async def send_compliance_alert(self, violation) -> str:
        """Send compliance violation alert through FLAGSHIP system."""
        template = self.notification_templates.get("compliance_violation")
        if not template:
            self.logger.error("Compliance violation template not found")
            return None

        message_id = f"notif_compliance_{violation.violation_id}"

        title = template.title_template.format(
            violation_type=violation.requirement.standard.value
        )
        content = template.content_template.format(
            action_id=violation.action_id,
            severity=violation.severity.value,
            description=violation.description,
        )

        message = NotificationMessage(
            message_id=message_id,
            notification_type=template.notification_type,
            priority=template.default_priority,
            title=title,
            content=content,
            target_entities=["compliance_officer", "legal_team"],
            channels=template.default_channels,
            metadata={
                "violation_id": violation.violation_id,
                "requirement_id": violation.requirement.requirement_id,
                "severity": violation.severity.value,
            },
        )

        # Send through FLAGSHIP system only
        await self._send_internal_notification(message)

        self.logger.info(f"Compliance alert sent through FLAGSHIP: {message_id}")
        return message_id

    async def send_critical_threat_alert(self, threat_event) -> str:
        """Send critical threat alert through FLAGSHIP system."""
        template = self.notification_templates.get("critical_threat")
        if not template:
            self.logger.error("Critical threat template not found")
            return None

        message_id = f"notif_critical_{threat_event.event_id}"

        title = template.title_template.format(threat_type=threat_event.event_type)
        content = template.content_template.format(
            threat_id=threat_event.event_id,
            threat_level=threat_event.threat_level.value,
        )

        message = NotificationMessage(
            message_id=message_id,
            notification_type=template.notification_type,
            priority=template.default_priority,
            title=title,
            content=content,
            target_entities=["security_team", "management", "emergency_coordinator"],
            channels=template.default_channels,
            metadata={
                "threat_id": threat_event.event_id,
                "threat_category": threat_event.threat_category.value,
                "threat_level": threat_event.threat_level.value,
            },
        )

        # Send through FLAGSHIP system only
        await self._send_internal_notification(message)

        self.logger.info(f"Critical threat alert sent through FLAGSHIP: {message_id}")
        return message_id

    async def send_test_notification(self, test_type: str) -> str:
        """Send test notification through FLAGSHIP system only."""
        template = self.notification_templates.get("test_notification")
        if not template:
            self.logger.error("Test notification template not found")
            return None

        message_id = f"notif_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        title = template.title_template.format(test_type=test_type)
        content = template.content_template.format(test_type=test_type)

        message = NotificationMessage(
            message_id=message_id,
            notification_type=template.notification_type,
            priority=template.default_priority,
            title=title,
            content=content,
            target_entities=["test_team"],
            channels=template.default_channels,
            metadata={"test_type": test_type},
        )

        # Send through FLAGSHIP system only
        await self._send_internal_notification(message)

        self.logger.info(f"Test notification sent through FLAGSHIP: {message_id}")
        return message_id

    async def _send_internal_notification(self, message: NotificationMessage):
        """Send notification through FLAGSHIP internal system only."""
        message.status = NotificationStatus.SENT_INTERNAL
        message.sent_at = datetime.now()

        # Store in pending notifications if approval required
        if (
            message.notification_type
            in [
                NotificationType.ESCALATION_ALERT,
                NotificationType.COMPLIANCE_ALERT,
                NotificationType.INTERNAL_ALERT,
            ]
            and self.require_approval
        ):
            self.pending_notifications[message.message_id] = message
            self.logger.info(f"Notification pending approval: {message.message_id}")
        else:
            # Send immediately to internal channels
            await self._send_to_internal_channels(message)
            self.sent_notifications.append(message)
            self.logger.info(
                f"Notification sent to internal channels: {message.message_id}"
            )

    async def _send_to_internal_channels(self, message: NotificationMessage):
        """Send message to internal FLAGSHIP channels."""
        for channel in message.channels:
            if channel == NotificationChannel.FLAGSHIP_LOG:
                self.logger.info(f"FLAGSHIP LOG: {message.title} - {message.content}")
            elif channel == NotificationChannel.FLAGSHIP_DASHBOARD:
                # In real implementation, this would send to the dashboard
                self.logger.info(f"FLAGSHIP DASHBOARD: {message.title}")
            elif channel == NotificationChannel.FLAGSHIP_API:
                # In real implementation, this would send via API
                self.logger.info(f"FLAGSHIP API: {message.title}")
            elif channel == NotificationChannel.FLAGSHIP_WEBSOCKET:
                # In real implementation, this would send via WebSocket
                self.logger.info(f"FLAGSHIP WEBSOCKET: {message.title}")
            elif channel == NotificationChannel.FLAGSHIP_AUDIT:
                # In real implementation, this would send to audit system
                self.logger.info(f"FLAGSHIP AUDIT: {message.title}")

    async def approve_notification(self, message_id: str, approved_by: str) -> bool:
        """Approve a notification for potential external sending."""
        if message_id not in self.pending_notifications:
            self.logger.error(f"Pending notification not found: {message_id}")
            return False

        message = self.pending_notifications[message_id]
        message.status = NotificationStatus.APPROVED
        message.approved_by = approved_by
        message.approved_at = datetime.now()

        # Send to internal channels
        await self._send_to_internal_channels(message)

        # Move to sent notifications
        self.sent_notifications.append(message)
        del self.pending_notifications[message_id]

        self.logger.info(f"Notification approved by {approved_by}: {message_id}")

        # Check if external communications are enabled and not in test mode
        if self.external_communications_enabled and not self.test_mode:
            await self._send_external_notification(message)
        else:
            self.logger.info(
                f"External communications disabled or in test mode - notification {message_id} remains internal only"
            )

        return True

    async def reject_notification(
        self, message_id: str, rejected_by: str, reason: str
    ) -> bool:
        """Reject a notification."""
        if message_id not in self.pending_notifications:
            self.logger.error(f"Pending notification not found: {message_id}")
            return False

        message = self.pending_notifications[message_id]
        message.status = NotificationStatus.REJECTED
        message.metadata["rejected_by"] = rejected_by
        message.metadata["rejection_reason"] = reason

        # Move to sent notifications (for audit trail)
        self.sent_notifications.append(message)
        del self.pending_notifications[message_id]

        self.logger.info(
            f"Notification rejected by {rejected_by}: {message_id} - {reason}"
        )
        return True

    async def _send_external_notification(self, message: NotificationMessage):
        """Send notification to external systems (only when explicitly enabled and approved)."""
        if not self.external_communications_enabled:
            self.logger.warning(
                "External communications disabled - notification remains internal only"
            )
            return

        if self.test_mode:
            self.logger.warning(
                "Test mode active - external notification would be sent here"
            )
            return

        # In real implementation, this would send to external systems
        # For now, just log that it would be sent
        self.logger.info(f"EXTERNAL NOTIFICATION WOULD BE SENT: {message.title}")
        message.external_sent_at = datetime.now()

    async def get_pending_notifications(self) -> List[NotificationMessage]:
        """Get all pending notifications requiring approval."""
        return list(self.pending_notifications.values())

    async def get_sent_notifications(
        self, limit: int = 100
    ) -> List[NotificationMessage]:
        """Get sent notification history."""
        return self.sent_notifications[-limit:]

    async def get_notification_templates(self) -> List[NotificationTemplate]:
        """Get all notification templates."""
        return list(self.notification_templates.values())

    async def enable_external_communications(self, enabled: bool = True):
        """Enable or disable external communications."""
        self.external_communications_enabled = enabled
        if enabled:
            self.logger.warning(
                "EXTERNAL COMMUNICATIONS ENABLED - Use with extreme caution"
            )
        else:
            self.logger.info("External communications disabled")

    async def set_test_mode(self, test_mode: bool = True):
        """Set test mode to prevent external communications."""
        self.test_mode = test_mode
        if test_mode:
            self.logger.info(
                "TEST MODE ENABLED - No external communications will be sent"
            )
        else:
            self.logger.warning(
                "Test mode disabled - external communications may be sent if enabled"
            )

    async def add_notification_template(self, template: NotificationTemplate) -> bool:
        """Add a new notification template."""
        if template.template_id in self.notification_templates:
            self.logger.warning(f"Template already exists: {template.template_id}")
            return False

        self.notification_templates[template.template_id] = template
        self.logger.info(f"Added notification template: {template.template_id}")
        return True
