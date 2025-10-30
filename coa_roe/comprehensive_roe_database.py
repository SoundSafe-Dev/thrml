"""
FLAGSHIP Comprehensive ROE Database

Comprehensive database of all pertinent Rules of Engagement (ROEs) from:
- International law and treaties
- National legal frameworks
- Industry standards and best practices
- Military and law enforcement protocols
- Corporate security standards
- Emergency response guidelines
"""

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
# ROE CATEGORIES AND STANDARDS
# ============================================================================


class ROECategory(Enum):
    """Categories of Rules of Engagement."""

    INTERNATIONAL_LAW = "international_law"
    NATIONAL_LAW = "national_law"
    MILITARY_PROTOCOLS = "military_protocols"
    LAW_ENFORCEMENT = "law_enforcement"
    CORPORATE_SECURITY = "corporate_security"
    EMERGENCY_RESPONSE = "emergency_response"
    CYBERSECURITY = "cybersecurity"
    MARITIME_SECURITY = "maritime_security"
    AVIATION_SECURITY = "aviation_security"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    HEALTHCARE_SECURITY = "healthcare_security"
    EDUCATIONAL_SECURITY = "educational_security"
    TRANSPORTATION_SECURITY = "transportation_security"
    FINANCIAL_SECURITY = "financial_security"
    RETAIL_SECURITY = "retail_security"


class ROEAuthority(Enum):
    """Authorities that establish ROEs."""

    UNITED_NATIONS = "united_nations"
    NATO = "nato"
    US_DEPARTMENT_OF_DEFENSE = "us_department_of_defense"
    US_DEPARTMENT_OF_HOMELAND_SECURITY = "us_department_of_homeland_security"
    FBI = "fbi"
    INTERPOL = "interpol"
    EUROPOL = "europol"
    OSHA = "osha"
    NFPA = "nfpa"
    ISO = "iso"
    NIST = "nist"
    SANS = "sans"
    ASIS = "asis"
    IACP = "iacp"
    IADLEST = "iadlest"


class ROESeverity(Enum):
    """Severity levels for ROE violations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINOR = "minor"


@dataclass
class ROEStandard:
    """Individual ROE standard."""

    roe_id: str
    title: str
    category: ROECategory
    authority: ROEAuthority
    jurisdiction: str
    description: str
    legal_basis: str
    applicability: List[str]
    force_levels: List[str]
    restrictions: List[str]
    requirements: List[str]
    documentation_required: bool
    severity: ROESeverity
    effective_date: datetime
    last_updated: datetime
    source_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ROEImplementation:
    """ROE implementation guidelines."""

    implementation_id: str
    roe_standard: ROEStandard
    implementation_guidelines: List[str]
    training_requirements: List[str]
    equipment_requirements: List[str]
    certification_requirements: List[str]
    audit_requirements: List[str]
    compliance_metrics: List[str]


# ============================================================================
# COMPREHENSIVE ROE DATABASE
# ============================================================================


class ComprehensiveROEDatabase:
    """Comprehensive database of all pertinent ROEs."""

    def __init__(self):
        """Initialize the comprehensive ROE database."""
        self.logger = logger
        self.roe_standards: Dict[str, ROEStandard] = {}
        self.roe_implementations: Dict[str, ROEImplementation] = {}

        # Load comprehensive ROE data
        self._load_international_roes()
        self._load_national_roes()
        self._load_military_roes()
        self._load_law_enforcement_roes()
        self._load_corporate_security_roes()
        self._load_emergency_response_roes()
        self._load_cybersecurity_roes()
        self._load_specialized_roes()

        self.logger.info(f"Loaded {len(self.roe_standards)} ROE standards")

    def _load_international_roes(self):
        """Load international ROE standards."""
        international_roes = [
            {
                "roe_id": "UN-001",
                "title": "United Nations Code of Conduct for Law-Enforcement Officials",
                "category": ROECategory.INTERNATIONAL_LAW,
                "authority": ROEAuthority.UNITED_NATIONS,
                "jurisdiction": "International",
                "description": "Fundamental principles for law enforcement conduct",
                "legal_basis": "UN General Assembly Resolution 34/169",
                "applicability": ["all_law_enforcement", "all_security_personnel"],
                "force_levels": [
                    "no_force",
                    "verbal_commands",
                    "non_lethal_force",
                    "lethal_force",
                ],
                "restrictions": [
                    "Respect and protect human dignity",
                    "Use force only when strictly necessary",
                    "Maintain and uphold human rights",
                    "Proportional response required",
                ],
                "requirements": [
                    "Human rights training",
                    "Use-of-force training",
                    "Accountability mechanisms",
                    "Regular review and assessment",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(1979, 12, 17),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["human_rights", "law_enforcement", "use_of_force"],
            },
            {
                "roe_id": "UN-002",
                "title": "United Nations Basic Principles on the Use of Force and Firearms",
                "category": ROECategory.INTERNATIONAL_LAW,
                "authority": ROEAuthority.UNITED_NATIONS,
                "jurisdiction": "International",
                "description": "Guidelines for use of force and firearms by law enforcement",
                "legal_basis": "UN General Assembly Resolution 45/166",
                "applicability": ["law_enforcement", "security_personnel"],
                "force_levels": ["non_lethal_force", "lethal_force"],
                "restrictions": [
                    "Use force only when strictly necessary",
                    "Proportional to legitimate objective",
                    "Minimize damage and injury",
                    "Force ceases when objective achieved",
                ],
                "requirements": [
                    "Specialized training",
                    "Equipment and weapons training",
                    "Medical assistance for injured",
                    "Reporting and review procedures",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(1990, 12, 18),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["use_of_force", "firearms", "law_enforcement"],
            },
            {
                "roe_id": "UN-003",
                "title": "United Nations Convention against Torture",
                "category": ROECategory.INTERNATIONAL_LAW,
                "authority": ROEAuthority.UNITED_NATIONS,
                "jurisdiction": "International",
                "description": "Prohibition of torture and cruel treatment",
                "legal_basis": "UN Convention against Torture",
                "applicability": ["all_personnel", "all_operations"],
                "force_levels": ["no_force", "verbal_commands", "non_lethal_force"],
                "restrictions": [
                    "No torture or cruel treatment",
                    "No degrading treatment",
                    "Humane treatment required",
                    "Medical care when needed",
                ],
                "requirements": [
                    "Training on prohibition of torture",
                    "Reporting mechanisms",
                    "Investigation procedures",
                    "Accountability measures",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(1984, 12, 10),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["torture", "human_rights", "prohibition"],
            },
            {
                "roe_id": "GENEVA-001",
                "title": "Geneva Conventions - Treatment of Persons",
                "category": ROECategory.INTERNATIONAL_LAW,
                "authority": ROEAuthority.UNITED_NATIONS,
                "jurisdiction": "International",
                "description": "Humane treatment of persons in custody",
                "legal_basis": "Geneva Conventions",
                "applicability": ["all_personnel", "detention_operations"],
                "force_levels": ["no_force", "verbal_commands", "non_lethal_force"],
                "restrictions": [
                    "Humane treatment required",
                    "No torture or cruel treatment",
                    "Medical care when needed",
                    "Respect for personal dignity",
                ],
                "requirements": [
                    "Training on Geneva Conventions",
                    "Medical support capabilities",
                    "Detention facility standards",
                    "Regular inspections",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(1949, 8, 12),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["geneva_conventions", "detention", "humane_treatment"],
            },
        ]

        for roe_data in international_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _load_national_roes(self):
        """Load national ROE standards."""
        national_roes = [
            {
                "roe_id": "US-001",
                "title": "U.S. Constitutional Use-of-Force Standards",
                "category": ROECategory.NATIONAL_LAW,
                "authority": ROEAuthority.US_DEPARTMENT_OF_DEFENSE,
                "jurisdiction": "United States",
                "description": "Constitutional standards for use of force",
                "legal_basis": "U.S. Constitution - 4th Amendment",
                "applicability": ["law_enforcement", "security_personnel"],
                "force_levels": ["non_lethal_force", "lethal_force"],
                "restrictions": [
                    "Probable cause required",
                    "Imminent threat of serious harm",
                    "No reasonable alternatives",
                    "Force stops when threat abates",
                ],
                "requirements": [
                    "Constitutional law training",
                    "Use-of-force training",
                    "Legal review procedures",
                    "Accountability mechanisms",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(1791, 12, 15),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["constitutional", "use_of_force", "4th_amendment"],
            },
            {
                "roe_id": "US-002",
                "title": "OSHA Emergency Action Plans",
                "category": ROECategory.NATIONAL_LAW,
                "authority": ROEAuthority.OSHA,
                "jurisdiction": "United States",
                "description": "Emergency action plan requirements",
                "legal_basis": "29 CFR 1910.38",
                "applicability": ["all_workplaces", "emergency_response"],
                "force_levels": ["no_force", "verbal_commands"],
                "restrictions": [
                    "Safe evacuation procedures",
                    "Clear exit routes",
                    "Assembly point procedures",
                    "Accountability procedures",
                ],
                "requirements": [
                    "Emergency action plan",
                    "Employee training",
                    "Regular drills",
                    "Plan review and updates",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(2002, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["osha", "emergency", "evacuation"],
            },
            {
                "roe_id": "US-003",
                "title": "NFPA 3000 Active Shooter/Hostile Event Response",
                "category": ROECategory.NATIONAL_LAW,
                "authority": ROEAuthority.NFPA,
                "jurisdiction": "United States",
                "description": "Standard for active shooter response",
                "legal_basis": "NFPA 3000 Standard",
                "applicability": ["all_facilities", "emergency_response"],
                "force_levels": [
                    "no_force",
                    "verbal_commands",
                    "non_lethal_force",
                    "lethal_force",
                ],
                "restrictions": [
                    "Life safety priority",
                    "Coordinated response",
                    "Communication protocols",
                    "Medical support",
                ],
                "requirements": [
                    "Response plan development",
                    "Multi-agency coordination",
                    "Training and exercises",
                    "After-action reviews",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(2018, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["active_shooter", "emergency_response", "nfpa"],
            },
        ]

        for roe_data in national_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _load_military_roes(self):
        """Load military ROE standards."""
        military_roes = [
            {
                "roe_id": "MIL-001",
                "title": "NATO Rules of Engagement",
                "category": ROECategory.MILITARY_PROTOCOLS,
                "authority": ROEAuthority.NATO,
                "jurisdiction": "NATO Member States",
                "description": "NATO standard ROE for military operations",
                "legal_basis": "NATO Standardization Agreement",
                "applicability": ["military_operations", "peacekeeping"],
                "force_levels": ["non_lethal_force", "lethal_force"],
                "restrictions": [
                    "Proportional response",
                    "Minimum force necessary",
                    "Civilian protection",
                    "International law compliance",
                ],
                "requirements": [
                    "ROE training",
                    "Legal review",
                    "Command approval",
                    "Regular updates",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(2000, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["nato", "military", "peacekeeping"],
            },
            {
                "roe_id": "MIL-002",
                "title": "U.S. Department of Defense ROE",
                "category": ROECategory.MILITARY_PROTOCOLS,
                "authority": ROEAuthority.US_DEPARTMENT_OF_DEFENSE,
                "jurisdiction": "United States",
                "description": "DoD standard ROE for military operations",
                "legal_basis": "DoD Directive 3121.02",
                "applicability": ["military_operations", "defense_operations"],
                "force_levels": ["non_lethal_force", "lethal_force"],
                "restrictions": [
                    "Self-defense authorized",
                    "Proportional response",
                    "Civilian protection",
                    "International law compliance",
                ],
                "requirements": [
                    "ROE training",
                    "Legal review",
                    "Command approval",
                    "Regular updates",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(2005, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["dod", "military", "self_defense"],
            },
        ]

        for roe_data in military_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _load_law_enforcement_roes(self):
        """Load law enforcement ROE standards."""
        law_enforcement_roes = [
            {
                "roe_id": "LE-001",
                "title": "IACP Model Policy on Use of Force",
                "category": ROECategory.LAW_ENFORCEMENT,
                "authority": ROEAuthority.IACP,
                "jurisdiction": "United States",
                "description": "Model policy for law enforcement use of force",
                "legal_basis": "IACP Model Policy",
                "applicability": ["law_enforcement", "police_operations"],
                "force_levels": ["verbal_commands", "non_lethal_force", "lethal_force"],
                "restrictions": [
                    "Reasonable force only",
                    "Proportional response",
                    "De-escalation when possible",
                    "Medical care for injured",
                ],
                "requirements": [
                    "Use-of-force training",
                    "De-escalation training",
                    "Reporting requirements",
                    "Review procedures",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(2017, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["iacp", "law_enforcement", "use_of_force"],
            },
            {
                "roe_id": "LE-002",
                "title": "FBI Use of Force Policy",
                "category": ROECategory.LAW_ENFORCEMENT,
                "authority": ROEAuthority.FBI,
                "jurisdiction": "United States",
                "description": "FBI standard use of force policy",
                "legal_basis": "FBI Policy Manual",
                "applicability": ["fbi_operations", "federal_law_enforcement"],
                "force_levels": ["verbal_commands", "non_lethal_force", "lethal_force"],
                "restrictions": [
                    "Imminent threat requirement",
                    "Proportional response",
                    "De-escalation when possible",
                    "Medical care for injured",
                ],
                "requirements": [
                    "FBI training",
                    "Legal review",
                    "Reporting requirements",
                    "Review procedures",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(2016, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["fbi", "federal", "use_of_force"],
            },
        ]

        for roe_data in law_enforcement_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _load_corporate_security_roes(self):
        """Load corporate security ROE standards."""
        corporate_security_roes = [
            {
                "roe_id": "CORP-001",
                "title": "ASIS International Security Management Standard",
                "category": ROECategory.CORPORATE_SECURITY,
                "authority": ROEAuthority.ASIS,
                "jurisdiction": "International",
                "description": "ASIS standard for corporate security management",
                "legal_basis": "ASIS International Standard",
                "applicability": ["corporate_security", "private_security"],
                "force_levels": ["no_force", "verbal_commands", "non_lethal_force"],
                "restrictions": [
                    "Legal compliance required",
                    "Proportional response",
                    "Documentation required",
                    "Regular review",
                ],
                "requirements": [
                    "Security training",
                    "Legal compliance",
                    "Documentation",
                    "Regular review",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(2015, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["asis", "corporate", "security_management"],
            },
            {
                "roe_id": "CORP-002",
                "title": "ISO 28000 Security Management Systems",
                "category": ROECategory.CORPORATE_SECURITY,
                "authority": ROEAuthority.ISO,
                "jurisdiction": "International",
                "description": "ISO standard for security management systems",
                "legal_basis": "ISO 28000 Standard",
                "applicability": ["corporate_security", "supply_chain_security"],
                "force_levels": ["no_force", "verbal_commands", "non_lethal_force"],
                "restrictions": [
                    "Risk-based approach",
                    "Legal compliance",
                    "Continuous improvement",
                    "Documentation required",
                ],
                "requirements": [
                    "Risk assessment",
                    "Security planning",
                    "Training and awareness",
                    "Performance monitoring",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(2007, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["iso", "security_management", "risk_based"],
            },
        ]

        for roe_data in corporate_security_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _load_emergency_response_roes(self):
        """Load emergency response ROE standards."""
        emergency_response_roes = [
            {
                "roe_id": "EMERG-001",
                "title": "NFPA 1600 Emergency Management",
                "category": ROECategory.EMERGENCY_RESPONSE,
                "authority": ROEAuthority.NFPA,
                "jurisdiction": "United States",
                "description": "NFPA standard for emergency management",
                "legal_basis": "NFPA 1600 Standard",
                "applicability": ["emergency_management", "disaster_response"],
                "force_levels": ["no_force", "verbal_commands"],
                "restrictions": [
                    "Life safety priority",
                    "Coordinated response",
                    "Communication protocols",
                    "Resource management",
                ],
                "requirements": [
                    "Emergency planning",
                    "Training and exercises",
                    "Communication systems",
                    "Resource management",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(2016, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["nfpa", "emergency_management", "disaster_response"],
            }
        ]

        for roe_data in emergency_response_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _load_cybersecurity_roes(self):
        """Load cybersecurity ROE standards."""
        cybersecurity_roes = [
            {
                "roe_id": "CYBER-001",
                "title": "NIST Cybersecurity Framework",
                "category": ROECategory.CYBERSECURITY,
                "authority": ROEAuthority.NIST,
                "jurisdiction": "United States",
                "description": "NIST framework for cybersecurity",
                "legal_basis": "NIST Cybersecurity Framework",
                "applicability": ["cybersecurity", "information_security"],
                "force_levels": ["no_force", "verbal_commands"],
                "restrictions": [
                    "Legal compliance",
                    "Privacy protection",
                    "Data protection",
                    "Incident reporting",
                ],
                "requirements": [
                    "Risk assessment",
                    "Security controls",
                    "Incident response",
                    "Continuous monitoring",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(2014, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["nist", "cybersecurity", "framework"],
            },
            {
                "roe_id": "CYBER-002",
                "title": "SANS Incident Response",
                "category": ROECategory.CYBERSECURITY,
                "authority": ROEAuthority.SANS,
                "jurisdiction": "International",
                "description": "SANS standard for incident response",
                "legal_basis": "SANS Incident Response Standard",
                "applicability": ["cybersecurity", "incident_response"],
                "force_levels": ["no_force", "verbal_commands"],
                "restrictions": [
                    "Legal compliance",
                    "Evidence preservation",
                    "Communication protocols",
                    "Documentation required",
                ],
                "requirements": [
                    "Incident response plan",
                    "Team training",
                    "Communication procedures",
                    "Documentation procedures",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(2010, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["sans", "incident_response", "cybersecurity"],
            },
        ]

        for roe_data in cybersecurity_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _load_specialized_roes(self):
        """Load specialized ROE standards."""
        specialized_roes = [
            {
                "roe_id": "SPEC-001",
                "title": "Maritime Security ROE",
                "category": ROECategory.MARITIME_SECURITY,
                "authority": ROEAuthority.US_DEPARTMENT_OF_HOMELAND_SECURITY,
                "jurisdiction": "United States",
                "description": "Maritime security rules of engagement",
                "legal_basis": "Maritime Security Act",
                "applicability": ["maritime_security", "port_security"],
                "force_levels": [
                    "no_force",
                    "verbal_commands",
                    "non_lethal_force",
                    "lethal_force",
                ],
                "restrictions": [
                    "Maritime law compliance",
                    "International waters considerations",
                    "Vessel safety",
                    "Environmental protection",
                ],
                "requirements": [
                    "Maritime training",
                    "Legal compliance",
                    "Communication protocols",
                    "Coordination procedures",
                ],
                "documentation_required": True,
                "severity": ROESeverity.HIGH,
                "effective_date": datetime(2002, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["maritime", "port_security", "homeland_security"],
            },
            {
                "roe_id": "SPEC-002",
                "title": "Aviation Security ROE",
                "category": ROECategory.AVIATION_SECURITY,
                "authority": ROEAuthority.US_DEPARTMENT_OF_HOMELAND_SECURITY,
                "jurisdiction": "United States",
                "description": "Aviation security rules of engagement",
                "legal_basis": "Aviation and Transportation Security Act",
                "applicability": ["aviation_security", "airport_security"],
                "force_levels": [
                    "no_force",
                    "verbal_commands",
                    "non_lethal_force",
                    "lethal_force",
                ],
                "restrictions": [
                    "Aviation safety priority",
                    "Passenger safety",
                    "Aircraft security",
                    "Coordination with authorities",
                ],
                "requirements": [
                    "Aviation training",
                    "Security protocols",
                    "Communication systems",
                    "Emergency procedures",
                ],
                "documentation_required": True,
                "severity": ROESeverity.CRITICAL,
                "effective_date": datetime(2001, 1, 1),
                "last_updated": datetime(2023, 1, 1),
                "tags": ["aviation", "airport_security", "homeland_security"],
            },
        ]

        for roe_data in specialized_roes:
            roe = self._create_roe_from_data(roe_data)
            self.roe_standards[roe.roe_id] = roe

    def _create_roe_from_data(self, data: Dict[str, Any]) -> ROEStandard:
        """Create an ROEStandard from specification data."""
        return ROEStandard(
            roe_id=data["roe_id"],
            title=data["title"],
            category=data["category"],
            authority=data["authority"],
            jurisdiction=data["jurisdiction"],
            description=data["description"],
            legal_basis=data["legal_basis"],
            applicability=data["applicability"],
            force_levels=data["force_levels"],
            restrictions=data["restrictions"],
            requirements=data["requirements"],
            documentation_required=data["documentation_required"],
            severity=data["severity"],
            effective_date=data["effective_date"],
            last_updated=data["last_updated"],
            source_url=data.get("source_url"),
            tags=data.get("tags", []),
        )

    async def get_roe_by_id(self, roe_id: str) -> Optional[ROEStandard]:
        """Get a specific ROE by ID."""
        return self.roe_standards.get(roe_id)

    async def get_roes_by_category(self, category: ROECategory) -> List[ROEStandard]:
        """Get ROEs by category."""
        return [roe for roe in self.roe_standards.values() if roe.category == category]

    async def get_roes_by_authority(self, authority: ROEAuthority) -> List[ROEStandard]:
        """Get ROEs by authority."""
        return [
            roe for roe in self.roe_standards.values() if roe.authority == authority
        ]

    async def get_roes_by_jurisdiction(self, jurisdiction: str) -> List[ROEStandard]:
        """Get ROEs by jurisdiction."""
        return [
            roe
            for roe in self.roe_standards.values()
            if roe.jurisdiction == jurisdiction
        ]

    async def get_roes_by_severity(self, severity: ROESeverity) -> List[ROEStandard]:
        """Get ROEs by severity level."""
        return [roe for roe in self.roe_standards.values() if roe.severity == severity]

    async def search_roes(self, query: str) -> List[ROEStandard]:
        """Search ROEs by title, description, or tags."""
        query_lower = query.lower()
        results = []

        for roe in self.roe_standards.values():
            if (
                query_lower in roe.title.lower()
                or query_lower in roe.description.lower()
                or any(query_lower in tag.lower() for tag in roe.tags)
            ):
                results.append(roe)

        return results

    async def get_all_roes(self) -> List[ROEStandard]:
        """Get all ROE standards."""
        return list(self.roe_standards.values())

    async def get_roe_summary(self) -> Dict[str, Any]:
        """Get a summary of all ROEs."""
        summary = {
            "total_roes": len(self.roe_standards),
            "by_category": {},
            "by_authority": {},
            "by_severity": {},
            "by_jurisdiction": {},
        }

        for roe in self.roe_standards.values():
            # Count by category
            category = roe.category.value
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )

            # Count by authority
            authority = roe.authority.value
            summary["by_authority"][authority] = (
                summary["by_authority"].get(authority, 0) + 1
            )

            # Count by severity
            severity = roe.severity.value
            summary["by_severity"][severity] = (
                summary["by_severity"].get(severity, 0) + 1
            )

            # Count by jurisdiction
            jurisdiction = roe.jurisdiction
            summary["by_jurisdiction"][jurisdiction] = (
                summary["by_jurisdiction"].get(jurisdiction, 0) + 1
            )

        return summary

    async def export_roe_data(self, format: str = "json") -> str:
        """Export ROE data in specified format."""
        if format.lower() == "json":
            export_data = []
            for roe in self.roe_standards.values():
                roe_dict = {
                    "roe_id": roe.roe_id,
                    "title": roe.title,
                    "category": roe.category.value,
                    "authority": roe.authority.value,
                    "jurisdiction": roe.jurisdiction,
                    "description": roe.description,
                    "legal_basis": roe.legal_basis,
                    "applicability": roe.applicability,
                    "force_levels": roe.force_levels,
                    "restrictions": roe.restrictions,
                    "requirements": roe.requirements,
                    "documentation_required": roe.documentation_required,
                    "severity": roe.severity.value,
                    "effective_date": roe.effective_date.isoformat(),
                    "last_updated": roe.last_updated.isoformat(),
                    "source_url": roe.source_url,
                    "tags": roe.tags,
                }
                export_data.append(roe_dict)

            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
