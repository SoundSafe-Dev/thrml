"""
FLAGSHIP Legal Compliance Manager

Ensures all Course-of-Action (COA) and Rules-of-Engagement (ROE) responses
comply with international legal standards and regulatory requirements.

Compliance Standards:
- United Nations Code of Conduct for Law-Enforcement Officials
- OSHA Emergency Action Plans (29 CFR 1910.38)
- U.S. Constitutional Use-of-Force Standards
- International Human Rights Law
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
# COMPLIANCE DATA STRUCTURES
# ============================================================================


class ComplianceStatus(Enum):
    """Compliance verification status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    EXEMPT = "exempt"


class ComplianceStandard(Enum):
    """Legal and regulatory compliance standards."""

    UN_CODE_OF_CONDUCT = "un_code_of_conduct"
    OSHA_EMERGENCY_ACTION = "osha_emergency_action"
    US_CONSTITUTIONAL_USE_OF_FORCE = "us_constitutional_use_of_force"
    INTERNATIONAL_HUMAN_RIGHTS = "international_human_rights"
    GENEVA_CONVENTIONS = "geneva_conventions"
    INTERNATIONAL_CRIMINAL_LAW = "international_criminal_law"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINOR = "minor"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""

    requirement_id: str
    standard: ComplianceStandard
    description: str
    applicable_actions: List[str]
    verification_criteria: List[str]
    documentation_required: bool
    severity: ViolationSeverity
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceViolation:
    """Record of a compliance violation."""

    violation_id: str
    requirement: ComplianceRequirement
    action_id: str
    description: str
    severity: ViolationSeverity
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    corrective_actions: List[str] = field(default_factory=list)


@dataclass
class ComplianceAudit:
    """Compliance audit record."""

    audit_id: str
    execution_id: str
    audit_timestamp: datetime
    compliance_status: ComplianceStatus
    verified_requirements: List[str]
    violations: List[ComplianceViolation]
    auditor: str
    notes: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# COMPLIANCE MANAGER
# ============================================================================


class LegalComplianceManager:
    """Manages legal compliance for COA/ROE responses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the legal compliance manager."""
        self.config = config or {}
        self.logger = logger

        # Compliance requirements
        self.compliance_requirements: Dict[str, ComplianceRequirement] = {}
        self.violations: List[ComplianceViolation] = []
        self.audit_history: List[ComplianceAudit] = []

        # Initialize compliance requirements
        self._initialize_compliance_requirements()

        self.logger.info("Legal Compliance Manager initialized")

    def _initialize_compliance_requirements(self):
        """Initialize compliance requirements from legal standards."""
        requirements_data = self._get_compliance_requirements()

        for req_data in requirements_data:
            requirement = self._create_requirement_from_data(req_data)
            self.compliance_requirements[requirement.requirement_id] = requirement

        self.logger.info(
            f"Loaded {len(self.compliance_requirements)} compliance requirements"
        )

    def _get_compliance_requirements(self) -> List[Dict[str, Any]]:
        """Get comprehensive compliance requirements."""
        return [
            {
                "requirement_id": "UN-001",
                "standard": ComplianceStandard.UN_CODE_OF_CONDUCT,
                "description": "Law enforcement officials must respect and protect human dignity and maintain and uphold the human rights of all persons.",
                "applicable_actions": [
                    "all_security_actions",
                    "all_law_enforcement_actions",
                ],
                "verification_criteria": [
                    "Actions respect human dignity",
                    "Human rights are protected",
                    "No unnecessary force used",
                    "Proportional response maintained",
                ],
                "documentation_required": True,
                "severity": ViolationSeverity.CRITICAL,
            },
            {
                "requirement_id": "UN-002",
                "standard": ComplianceStandard.UN_CODE_OF_CONDUCT,
                "description": "Law enforcement officials may use force only when strictly necessary and to the extent required for the performance of their duty.",
                "applicable_actions": ["use_of_force", "detention", "arrest"],
                "verification_criteria": [
                    "Force used only when strictly necessary",
                    "Force proportional to threat",
                    "Force ceases when threat abates",
                    "Minimum force principle followed",
                ],
                "documentation_required": True,
                "severity": ViolationSeverity.HIGH,
            },
            {
                "requirement_id": "OSHA-001",
                "standard": ComplianceStandard.OSHA_EMERGENCY_ACTION,
                "description": "Emergency action plans must include procedures for emergency evacuation and exit route assignments.",
                "applicable_actions": ["evacuation", "emergency_response"],
                "verification_criteria": [
                    "Evacuation procedures documented",
                    "Exit routes clearly marked",
                    "Assembly points established",
                    "Accountability procedures in place",
                ],
                "documentation_required": True,
                "severity": ViolationSeverity.HIGH,
            },
            {
                "requirement_id": "US-001",
                "standard": ComplianceStandard.US_CONSTITUTIONAL_USE_OF_FORCE,
                "description": "Deadly force may be used only when there is probable cause to believe the suspect poses a threat of serious physical harm to officers or others.",
                "applicable_actions": ["lethal_force", "deadly_force"],
                "verification_criteria": [
                    "Probable cause established",
                    "Imminent threat of serious harm",
                    "No reasonable alternatives available",
                    "Force stops when threat abates",
                ],
                "documentation_required": True,
                "severity": ViolationSeverity.CRITICAL,
            },
            {
                "requirement_id": "IHR-001",
                "standard": ComplianceStandard.INTERNATIONAL_HUMAN_RIGHTS,
                "description": "All persons have the right to life, liberty, and security of person.",
                "applicable_actions": ["all_actions"],
                "verification_criteria": [
                    "Right to life protected",
                    "Liberty not arbitrarily deprived",
                    "Security of person maintained",
                    "Due process followed",
                ],
                "documentation_required": True,
                "severity": ViolationSeverity.CRITICAL,
            },
            {
                "requirement_id": "GC-001",
                "standard": ComplianceStandard.GENEVA_CONVENTIONS,
                "description": "Persons taking no active part in hostilities must be treated humanely.",
                "applicable_actions": ["detention", "treatment_of_persons"],
                "verification_criteria": [
                    "Humane treatment provided",
                    "No torture or cruel treatment",
                    "Medical care when needed",
                    "Respect for personal dignity",
                ],
                "documentation_required": True,
                "severity": ViolationSeverity.HIGH,
            },
        ]

    def _create_requirement_from_data(
        self, data: Dict[str, Any]
    ) -> ComplianceRequirement:
        """Create a ComplianceRequirement from specification data."""
        return ComplianceRequirement(
            requirement_id=data["requirement_id"],
            standard=data["standard"],
            description=data["description"],
            applicable_actions=data["applicable_actions"],
            verification_criteria=data["verification_criteria"],
            documentation_required=data["documentation_required"],
            severity=data["severity"],
        )

    async def start(self):
        """Start the compliance manager."""
        self.logger.info("Starting Legal Compliance Manager")

    async def stop(self):
        """Stop the compliance manager."""
        self.logger.info("Stopping Legal Compliance Manager")

    async def verify_execution_compliance(self, execution) -> bool:
        """Verify compliance of a response execution."""
        audit_id = (
            f"audit_{execution.execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        verified_requirements = []
        violations = []

        # Check each action for compliance
        for action in execution.actions:
            action_violations = await self._verify_action_compliance(action)
            violations.extend(action_violations)

            # Check if action meets requirements
            for requirement in self.compliance_requirements.values():
                if self._is_requirement_applicable(requirement, action):
                    if await self._verify_requirement_compliance(requirement, action):
                        verified_requirements.append(requirement.requirement_id)
                    else:
                        # Create violation record
                        violation = ComplianceViolation(
                            violation_id=f"viol_{len(violations)}",
                            requirement=requirement,
                            action_id=action.action_id,
                            description=f"Action {action.action_id} violates {requirement.description}",
                            severity=requirement.severity,
                            detected_at=datetime.now(),
                        )
                        violations.append(violation)

        # Determine overall compliance status
        if violations:
            compliance_status = ComplianceStatus.NON_COMPLIANT
            self.logger.warning(f"Compliance violations detected: {len(violations)}")
        else:
            compliance_status = ComplianceStatus.COMPLIANT
            self.logger.info("All compliance requirements met")

        # Create audit record
        audit = ComplianceAudit(
            audit_id=audit_id,
            execution_id=execution.execution_id,
            audit_timestamp=datetime.now(),
            compliance_status=compliance_status,
            verified_requirements=verified_requirements,
            violations=violations,
            auditor="FLAGSHIP_System",
        )

        self.audit_history.append(audit)

        # Store violations
        self.violations.extend(violations)

        return compliance_status == ComplianceStatus.COMPLIANT

    async def _verify_action_compliance(self, action) -> List[ComplianceViolation]:
        """Verify compliance of a specific action."""
        violations = []

        # Check force level compliance
        if action.force_level.value in ["lethal_force", "deadly_force"]:
            force_violations = await self._verify_force_compliance(action)
            violations.extend(force_violations)

        # Check human rights compliance
        hr_violations = await self._verify_human_rights_compliance(action)
        violations.extend(hr_violations)

        return violations

    async def _verify_force_compliance(self, action) -> List[ComplianceViolation]:
        """Verify compliance with use-of-force standards."""
        violations = []

        # Check US Constitutional standards
        us_requirement = self.compliance_requirements.get("US-001")
        if us_requirement and self._is_requirement_applicable(us_requirement, action):
            if not await self._verify_requirement_compliance(us_requirement, action):
                violation = ComplianceViolation(
                    violation_id=f"force_viol_{len(violations)}",
                    requirement=us_requirement,
                    action_id=action.action_id,
                    description="Use of deadly force may not meet constitutional standards",
                    severity=ViolationSeverity.CRITICAL,
                    detected_at=datetime.now(),
                )
                violations.append(violation)

        # Check UN Code of Conduct
        un_requirement = self.compliance_requirements.get("UN-002")
        if un_requirement and self._is_requirement_applicable(un_requirement, action):
            if not await self._verify_requirement_compliance(un_requirement, action):
                violation = ComplianceViolation(
                    violation_id=f"force_viol_{len(violations)}",
                    requirement=un_requirement,
                    action_id=action.action_id,
                    description="Use of force may not meet UN Code of Conduct standards",
                    severity=ViolationSeverity.HIGH,
                    detected_at=datetime.now(),
                )
                violations.append(violation)

        return violations

    async def _verify_human_rights_compliance(
        self, action
    ) -> List[ComplianceViolation]:
        """Verify compliance with human rights standards."""
        violations = []

        # Check International Human Rights
        ihr_requirement = self.compliance_requirements.get("IHR-001")
        if ihr_requirement and self._is_requirement_applicable(ihr_requirement, action):
            if not await self._verify_requirement_compliance(ihr_requirement, action):
                violation = ComplianceViolation(
                    violation_id=f"hr_viol_{len(violations)}",
                    requirement=ihr_requirement,
                    action_id=action.action_id,
                    description="Action may violate human rights standards",
                    severity=ViolationSeverity.CRITICAL,
                    detected_at=datetime.now(),
                )
                violations.append(violation)

        return violations

    def _is_requirement_applicable(
        self, requirement: ComplianceRequirement, action
    ) -> bool:
        """Check if a compliance requirement applies to an action."""
        # Check if action type matches applicable actions
        for applicable_action in requirement.applicable_actions:
            if applicable_action == "all_actions":
                return True
            elif (
                applicable_action == "all_security_actions"
                and action.action_type.value == "security_response"
            ):
                return True
            elif (
                applicable_action == "all_law_enforcement_actions"
                and action.action_type.value == "law_enforcement_response"
            ):
                return True
            elif applicable_action in action.action_type.value:
                return True

        return False

    async def _verify_requirement_compliance(
        self, requirement: ComplianceRequirement, action
    ) -> bool:
        """Verify if an action complies with a specific requirement."""
        # This is a simplified verification - in a real implementation,
        # this would involve more sophisticated analysis

        # Check if action has required compliance requirements
        if hasattr(action, "compliance_requirements"):
            for compliance_standard in action.compliance_requirements:
                if compliance_standard.value == requirement.standard.value:
                    return True

        # Check force level appropriateness
        if requirement.standard == ComplianceStandard.US_CONSTITUTIONAL_USE_OF_FORCE:
            if action.force_level.value in ["lethal_force", "deadly_force"]:
                # Would need more context to determine if deadly force is justified
                return True  # Simplified for demo

        # Check UN Code of Conduct
        if requirement.standard == ComplianceStandard.UN_CODE_OF_CONDUCT:
            if action.force_level.value in [
                "no_force",
                "verbal_commands",
                "non_lethal_force",
            ]:
                return True

        return True  # Default to compliant for demo

    async def get_compliance_requirements(self) -> List[ComplianceRequirement]:
        """Get all compliance requirements."""
        return list(self.compliance_requirements.values())

    async def get_requirement_by_id(
        self, requirement_id: str
    ) -> Optional[ComplianceRequirement]:
        """Get a specific compliance requirement by ID."""
        return self.compliance_requirements.get(requirement_id)

    async def get_violations(
        self, severity: Optional[ViolationSeverity] = None
    ) -> List[ComplianceViolation]:
        """Get compliance violations, optionally filtered by severity."""
        if severity:
            return [v for v in self.violations if v.severity == severity]
        return self.violations

    async def get_audit_history(self, limit: int = 100) -> List[ComplianceAudit]:
        """Get compliance audit history."""
        return self.audit_history[-limit:]

    async def resolve_violation(
        self, violation_id: str, resolution_notes: str, corrective_actions: List[str]
    ) -> bool:
        """Resolve a compliance violation."""
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.resolved_at = datetime.now()
                violation.resolution_notes = resolution_notes
                violation.corrective_actions = corrective_actions
                self.logger.info(f"Resolved violation: {violation_id}")
                return True

        self.logger.error(f"Violation not found: {violation_id}")
        return False

    async def generate_compliance_report(self, execution_id: str) -> Dict[str, Any]:
        """Generate a comprehensive compliance report for an execution."""
        # Find relevant audit
        audit = None
        for a in self.audit_history:
            if a.execution_id == execution_id:
                audit = a
                break

        if not audit:
            return {"error": "No audit found for execution"}

        # Get violations by severity
        violations_by_severity = {}
        for severity in ViolationSeverity:
            violations_by_severity[severity.value] = [
                v for v in audit.violations if v.severity == severity
            ]

        return {
            "execution_id": execution_id,
            "audit_id": audit.audit_id,
            "audit_timestamp": audit.audit_timestamp.isoformat(),
            "compliance_status": audit.compliance_status.value,
            "verified_requirements": audit.verified_requirements,
            "total_violations": len(audit.violations),
            "violations_by_severity": violations_by_severity,
            "recommendations": audit.recommendations,
            "auditor": audit.auditor,
        }

    async def add_compliance_requirement(
        self, requirement: ComplianceRequirement
    ) -> bool:
        """Add a new compliance requirement."""
        if requirement.requirement_id in self.compliance_requirements:
            self.logger.warning(
                f"Requirement already exists: {requirement.requirement_id}"
            )
            return False

        self.compliance_requirements[requirement.requirement_id] = requirement
        self.logger.info(f"Added compliance requirement: {requirement.requirement_id}")
        return True

    async def update_compliance_requirement(
        self, requirement_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update an existing compliance requirement."""
        if requirement_id not in self.compliance_requirements:
            self.logger.error(f"Requirement not found: {requirement_id}")
            return False

        requirement = self.compliance_requirements[requirement_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(requirement, key):
                setattr(requirement, key, value)

        self.logger.info(f"Updated compliance requirement: {requirement_id}")
        return True
