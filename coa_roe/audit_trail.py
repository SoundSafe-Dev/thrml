"""
FLAGSHIP Audit Trail Manager

Provides comprehensive audit trail functionality for all COA/ROE activities.
Ensures accountability, compliance tracking, and legal documentation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

logger = get_logger(__name__)


# ============================================================================
# AUDIT DATA STRUCTURES
# ============================================================================


class AuditEventType(Enum):
    """Types of audit events."""

    EXECUTION_START = "execution_start"
    EXECUTION_COMPLETION = "execution_completion"
    EXECUTION_FAILURE = "execution_failure"
    ACTION_EXECUTION = "action_execution"
    ESCALATION_INITIATED = "escalation_initiated"
    ESCALATION_COMPLETED = "escalation_completed"
    COMPLIANCE_VIOLATION = "compliance_violation"
    COMPLIANCE_VERIFICATION = "compliance_verification"
    NOTIFICATION_SENT = "notification_sent"
    NOTIFICATION_APPROVED = "notification_approved"
    NOTIFICATION_REJECTED = "notification_rejected"
    PROTOCOL_ACTIVATED = "protocol_activated"
    PROTOCOL_DEACTIVATED = "protocol_deactivated"
    SYSTEM_CONFIGURATION = "system_configuration"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AuditEvent:
    """Individual audit event record."""

    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    # Non-default fields must come before any fields with defaults
    description: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    execution_id: Optional[str] = None
    action_id: Optional[str] = None
    protocol_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AuditSession:
    """Audit session for tracking user activities."""

    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    events: List[AuditEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Comprehensive audit report."""

    report_id: str
    title: str
    description: str
    start_date: datetime
    end_date: datetime
    generated_at: datetime
    generated_by: str
    events: List[AuditEvent]
    summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    export_format: str = "json"


# ============================================================================
# AUDIT TRAIL MANAGER
# ============================================================================


class AuditTrailManager:
    """Manages comprehensive audit trails for COA/ROE activities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the audit trail manager."""
        self.config = config or {}
        self.logger = logger

        # Audit storage
        self.audit_events: List[AuditEvent] = []
        self.audit_sessions: Dict[str, AuditSession] = {}
        self.audit_reports: List[AuditReport] = []

        # Configuration
        self.max_events = self.config.get("max_events", 10000)
        self.retention_days = self.config.get("retention_days", 365)
        self.enable_detailed_logging = self.config.get("enable_detailed_logging", True)

        self.logger.info("Audit Trail Manager initialized")

    async def start(self):
        """Start the audit trail manager."""
        self.logger.info("Starting Audit Trail Manager")

    async def stop(self):
        """Stop the audit trail manager."""
        self.logger.info("Stopping Audit Trail Manager")

    async def log_execution_start(self, execution) -> str:
        """Log the start of a COA/ROE execution."""
        event_id = f"audit_exec_start_{execution.execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.EXECUTION_START,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            execution_id=execution.execution_id,
            protocol_id=execution.protocol.protocol_id,
            description=f"COA/ROE execution started for threat event {execution.threat_event.event_id}",
            details={
                "threat_event_id": execution.threat_event.event_id,
                "threat_type": execution.threat_event.event_type,
                "threat_category": execution.threat_event.threat_category.value,
                "threat_level": execution.threat_event.threat_level.value,
                "protocol_id": execution.protocol.protocol_id,
                "action_count": len(execution.actions),
                "compliance_standards": [
                    std.value for std in execution.protocol.compliance_standards
                ],
            },
            tags=["execution", "start", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.info(f"Audit: Execution start logged - {event_id}")
        return event_id

    async def log_execution_completion(self, execution) -> str:
        """Log the completion of a COA/ROE execution."""
        event_id = f"audit_exec_complete_{execution.execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        duration = execution.end_time - execution.start_time

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.EXECUTION_COMPLETION,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            execution_id=execution.execution_id,
            protocol_id=execution.protocol.protocol_id,
            description=f"COA/ROE execution completed for threat event {execution.threat_event.event_id}",
            details={
                "threat_event_id": execution.threat_event.event_id,
                "protocol_id": execution.protocol.protocol_id,
                "execution_duration_seconds": duration.total_seconds(),
                "status": execution.status,
                "executed_by": execution.executed_by,
                "compliance_verified": execution.compliance_verified,
                "total_actions": len(execution.actions),
            },
            tags=["execution", "completion", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.info(f"Audit: Execution completion logged - {event_id}")
        return event_id

    async def log_execution_failure(self, execution, error_message: str) -> str:
        """Log the failure of a COA/ROE execution."""
        event_id = f"audit_exec_failure_{execution.execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.EXECUTION_FAILURE,
            severity=AuditSeverity.CRITICAL,
            timestamp=datetime.now(),
            execution_id=execution.execution_id,
            protocol_id=execution.protocol.protocol_id,
            description=f"COA/ROE execution failed for threat event {execution.threat_event.event_id}",
            details={
                "threat_event_id": execution.threat_event.event_id,
                "protocol_id": execution.protocol.protocol_id,
                "error_message": error_message,
                "status": execution.status,
                "executed_by": execution.executed_by,
            },
            tags=["execution", "failure", "coa_roe", "error"],
        )

        await self._store_audit_event(event)

        self.logger.error(f"Audit: Execution failure logged - {event_id}")
        return event_id

    async def log_action_execution(self, action, execution) -> str:
        """Log the execution of a specific action."""
        event_id = f"audit_action_{action.action_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.ACTION_EXECUTION,
            severity=AuditSeverity.MEDIUM,
            timestamp=datetime.now(),
            execution_id=execution.execution_id,
            action_id=action.action_id,
            protocol_id=execution.protocol.protocol_id,
            description=f"Action executed: {action.description}",
            details={
                "action_type": action.action_type.value,
                "force_level": action.force_level.value,
                "priority": action.priority,
                "target_entities": action.target_entities,
                "required_resources": action.required_resources,
                "success_criteria": action.success_criteria,
                "compliance_requirements": [
                    req.value for req in action.compliance_requirements
                ],
            },
            tags=["action", "execution", "coa_roe"],
        )

        await self._store_audit_event(event)

        if self.enable_detailed_logging:
            self.logger.info(f"Audit: Action execution logged - {event_id}")
        return event_id

    async def log_escalation_initiated(self, escalation_request) -> str:
        """Log the initiation of an escalation."""
        event_id = f"audit_escalation_start_{escalation_request.request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.ESCALATION_INITIATED,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            execution_id=escalation_request.execution_id,
            description=f"Escalation initiated to {escalation_request.target.name}",
            details={
                "escalation_type": escalation_request.escalation_type.value,
                "escalation_level": escalation_request.escalation_level.value,
                "target_name": escalation_request.target.name,
                "target_id": escalation_request.target.target_id,
                "reason": escalation_request.reason,
                "urgency": escalation_request.urgency,
            },
            tags=["escalation", "initiated", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.info(f"Audit: Escalation initiated logged - {event_id}")
        return event_id

    async def log_escalation_completed(self, escalation_request) -> str:
        """Log the completion of an escalation."""
        event_id = f"audit_escalation_complete_{escalation_request.request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        duration = escalation_request.completed_at - escalation_request.created_at

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.ESCALATION_COMPLETED,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            execution_id=escalation_request.execution_id,
            description=f"Escalation completed to {escalation_request.target.name}",
            details={
                "escalation_type": escalation_request.escalation_type.value,
                "escalation_level": escalation_request.escalation_level.value,
                "target_name": escalation_request.target.name,
                "target_id": escalation_request.target.target_id,
                "escalation_duration_seconds": duration.total_seconds(),
                "status": escalation_request.status.value,
                "response_notes": escalation_request.response_notes,
            },
            tags=["escalation", "completed", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.info(f"Audit: Escalation completed logged - {event_id}")
        return event_id

    async def log_compliance_violation(self, violation) -> str:
        """Log a compliance violation."""
        event_id = f"audit_compliance_viol_{violation.violation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.COMPLIANCE_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            timestamp=datetime.now(),
            action_id=violation.action_id,
            description=f"Compliance violation detected: {violation.description}",
            details={
                "violation_id": violation.violation_id,
                "requirement_id": violation.requirement.requirement_id,
                "requirement_standard": violation.requirement.standard.value,
                "severity": violation.severity.value,
                "description": violation.description,
                "detected_at": violation.detected_at.isoformat(),
            },
            tags=["compliance", "violation", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.error(f"Audit: Compliance violation logged - {event_id}")
        return event_id

    async def log_compliance_verification(
        self, execution, compliance_status: bool
    ) -> str:
        """Log compliance verification results."""
        event_id = f"audit_compliance_verify_{execution.execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.COMPLIANCE_VERIFICATION,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            execution_id=execution.execution_id,
            protocol_id=execution.protocol.protocol_id,
            description=f"Compliance verification completed for execution {execution.execution_id}",
            details={
                "compliance_status": compliance_status,
                "protocol_id": execution.protocol.protocol_id,
                "compliance_standards": [
                    std.value for std in execution.protocol.compliance_standards
                ],
                "total_actions": len(execution.actions),
            },
            tags=["compliance", "verification", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.info(f"Audit: Compliance verification logged - {event_id}")
        return event_id

    async def log_notification_sent(self, notification_message) -> str:
        """Log a notification being sent."""
        event_id = f"audit_notif_sent_{notification_message.message_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.NOTIFICATION_SENT,
            severity=AuditSeverity.MEDIUM,
            timestamp=datetime.now(),
            execution_id=notification_message.execution_id,
            action_id=notification_message.action_id,
            description=f"Notification sent: {notification_message.title}",
            details={
                "notification_type": notification_message.notification_type.value,
                "priority": notification_message.priority.value,
                "channels": [ch.value for ch in notification_message.channels],
                "target_entities": notification_message.target_entities,
                "status": notification_message.status.value,
            },
            tags=["notification", "sent", "coa_roe"],
        )

        await self._store_audit_event(event)

        if self.enable_detailed_logging:
            self.logger.info(f"Audit: Notification sent logged - {event_id}")
        return event_id

    async def log_notification_approved(
        self, notification_message, approved_by: str
    ) -> str:
        """Log a notification being approved."""
        event_id = f"audit_notif_approved_{notification_message.message_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.NOTIFICATION_APPROVED,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            execution_id=notification_message.execution_id,
            action_id=notification_message.action_id,
            user_id=approved_by,
            description=f"Notification approved by {approved_by}: {notification_message.title}",
            details={
                "notification_type": notification_message.notification_type.value,
                "priority": notification_message.priority.value,
                "approved_by": approved_by,
                "approved_at": (
                    notification_message.approved_at.isoformat()
                    if notification_message.approved_at
                    else None
                ),
            },
            tags=["notification", "approved", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.info(f"Audit: Notification approved logged - {event_id}")
        return event_id

    async def log_notification_rejected(
        self, notification_message, rejected_by: str, reason: str
    ) -> str:
        """Log a notification being rejected."""
        event_id = f"audit_notif_rejected_{notification_message.message_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        event = AuditEvent(
            event_id=event_id,
            event_type=AuditEventType.NOTIFICATION_REJECTED,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            execution_id=notification_message.execution_id,
            action_id=notification_message.action_id,
            user_id=rejected_by,
            description=f"Notification rejected by {rejected_by}: {notification_message.title}",
            details={
                "notification_type": notification_message.notification_type.value,
                "priority": notification_message.priority.value,
                "rejected_by": rejected_by,
                "rejection_reason": reason,
            },
            tags=["notification", "rejected", "coa_roe"],
        )

        await self._store_audit_event(event)

        self.logger.info(f"Audit: Notification rejected logged - {event_id}")
        return event_id

    async def _store_audit_event(self, event: AuditEvent):
        """Store an audit event."""
        self.audit_events.append(event)

        # Maintain event limit
        if len(self.audit_events) > self.max_events:
            # Remove oldest events
            self.audit_events = self.audit_events[-self.max_events :]

        # Clean up old events based on retention policy
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        self.audit_events = [e for e in self.audit_events if e.timestamp > cutoff_date]

    async def get_audit_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        severity_levels: Optional[List[AuditSeverity]] = None,
        execution_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """Get audit events with optional filtering."""
        events = self.audit_events

        # Apply filters
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        if severity_levels:
            events = [e for e in events if e.severity in severity_levels]
        if execution_id:
            events = [e for e in events if e.execution_id == execution_id]

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]

    async def generate_audit_report(
        self,
        title: str,
        description: str,
        start_date: datetime,
        end_date: datetime,
        generated_by: str,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> AuditReport:
        """Generate a comprehensive audit report."""
        report_id = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get events for the report period
        events = await self.get_audit_events(
            start_date=start_date, end_date=end_date, event_types=event_types
        )

        # Generate summary
        summary = {
            "total_events": len(events),
            "event_types": {},
            "severity_distribution": {},
            "executions": {},
            "compliance_violations": 0,
            "escalations": 0,
            "notifications": 0,
        }

        for event in events:
            # Count by event type
            event_type = event.event_type.value
            summary["event_types"][event_type] = (
                summary["event_types"].get(event_type, 0) + 1
            )

            # Count by severity
            severity = event.severity.value
            summary["severity_distribution"][severity] = (
                summary["severity_distribution"].get(severity, 0) + 1
            )

            # Count executions
            if event.execution_id:
                summary["executions"][event.execution_id] = (
                    summary["executions"].get(event.execution_id, 0) + 1
                )

            # Count specific event types
            if event.event_type == AuditEventType.COMPLIANCE_VIOLATION:
                summary["compliance_violations"] += 1
            elif event.event_type in [
                AuditEventType.ESCALATION_INITIATED,
                AuditEventType.ESCALATION_COMPLETED,
            ]:
                summary["escalations"] += 1
            elif event.event_type in [
                AuditEventType.NOTIFICATION_SENT,
                AuditEventType.NOTIFICATION_APPROVED,
                AuditEventType.NOTIFICATION_REJECTED,
            ]:
                summary["notifications"] += 1

        # Generate recommendations
        recommendations = []
        if summary["compliance_violations"] > 0:
            recommendations.append(
                "Review compliance violations and update protocols as needed"
            )
        if summary["escalations"] > 0:
            recommendations.append(
                "Analyze escalation patterns to optimize response times"
            )
        if summary["notifications"] > 0:
            recommendations.append(
                "Review notification patterns and approval workflows"
            )

        report = AuditReport(
            report_id=report_id,
            title=title,
            description=description,
            start_date=start_date,
            end_date=end_date,
            generated_at=datetime.now(),
            generated_by=generated_by,
            events=events,
            summary=summary,
            recommendations=recommendations,
        )

        self.audit_reports.append(report)

        self.logger.info(f"Audit report generated: {report_id}")
        return report

    async def get_audit_reports(self, limit: int = 100) -> List[AuditReport]:
        """Get audit report history."""
        return self.audit_reports[-limit:]

    async def export_audit_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json",
    ) -> str:
        """Export audit data in specified format."""
        events = await self.get_audit_events(start_date=start_date, end_date=end_date)

        if format.lower() == "json":
            # Convert events to JSON-serializable format
            export_data = []
            for event in events:
                event_dict = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "session_id": event.session_id,
                    "execution_id": event.execution_id,
                    "action_id": event.action_id,
                    "protocol_id": event.protocol_id,
                    "description": event.description,
                    "details": event.details,
                    "ip_address": event.ip_address,
                    "user_agent": event.user_agent,
                    "location": event.location,
                    "tags": event.tags,
                }
                export_data.append(event_dict)

            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def start_audit_session(
        self,
        user_id: str,
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditSession:
        """Start a new audit session for a user."""
        session = AuditSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.audit_sessions[session_id] = session

        self.logger.info(f"Audit session started: {session_id} for user {user_id}")
        return session

    async def end_audit_session(self, session_id: str) -> bool:
        """End an audit session."""
        if session_id not in self.audit_sessions:
            return False

        session = self.audit_sessions[session_id]
        session.end_time = datetime.now()

        self.logger.info(f"Audit session ended: {session_id}")
        return True

    async def get_audit_sessions(
        self, user_id: Optional[str] = None, limit: int = 100
    ) -> List[AuditSession]:
        """Get audit sessions, optionally filtered by user."""
        sessions = list(self.audit_sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]

        # Sort by start time (newest first) and limit
        sessions.sort(key=lambda x: x.start_time, reverse=True)
        return sessions[:limit]
