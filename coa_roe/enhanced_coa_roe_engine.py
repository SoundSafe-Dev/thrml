"""
FLAGSHIP Enhanced COA/ROE Engine

Advanced Course-of-Action (COA) and Rules-of-Engagement (ROE) engine with:
- Intelligent decision making and real-time adaptation
- Comprehensive rule correlation and conflict resolution
- Advanced compliance monitoring and validation
- Dynamic response generation and execution
- Multi-dimensional threat assessment integration
- Predictive analytics and trend analysis
- Automated escalation and notification systems
- Comprehensive audit trails and reporting
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

from ..__init__ import ThreatCategory, ThreatEvent, ThreatLevel
from .audit_trail import AuditTrailManager
from .detailed_roe_registry import DetailedROERegistry
from .escalation_manager import EscalationManager
from .legal_compliance import LegalComplianceManager
from .notification_coordinator import NotificationCoordinator
from .response_protocols import ResponseProtocolManager
from .rule_correlation_engine import RuleCorrelationEngine

logger = get_logger(__name__)


# ============================================================================
# ENHANCED COA/ROE DATA STRUCTURES
# ============================================================================


class IntelligentResponseType(Enum):
    """Enhanced response types with intelligent capabilities."""

    ADAPTIVE_SECURITY_RESPONSE = "adaptive_security_response"
    INTELLIGENT_LAW_ENFORCEMENT = "intelligent_law_enforcement"
    PREDICTIVE_EMERGENCY_SERVICES = "predictive_emergency_services"
    CONTEXT_AWARE_EVACUATION = "context_aware_evacuation"
    DYNAMIC_LOCKDOWN = "dynamic_lockdown"
    AI_ENHANCED_INVESTIGATION = "ai_enhanced_investigation"
    SMART_COORDINATION = "smart_coordination"
    BEHAVIORAL_RESPONSE = "behavioral_response"
    ENVIRONMENTAL_ADAPTATION = "environmental_adaptation"
    TEMPORAL_OPTIMIZATION = "temporal_optimization"


class ResponseIntelligenceLevel(Enum):
    """Intelligence levels for response generation."""

    BASIC = "basic"
    ENHANCED = "enhanced"
    INTELLIGENT = "intelligent"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    AUTONOMOUS = "autonomous"


class ComplianceValidationStatus(Enum):
    """Enhanced compliance validation status."""

    VALIDATED = "validated"
    PENDING_VALIDATION = "pending_validation"
    VALIDATION_FAILED = "validation_failed"
    EXEMPT = "exempt"
    CONDITIONAL = "conditional"
    UNDER_REVIEW = "under_review"
    AUTO_VALIDATED = "auto_validated"


@dataclass
class IntelligentCOAROEProtocol:
    """Enhanced COA/ROE protocol with intelligent capabilities."""

    protocol_id: str
    name: str
    event_type: str
    threat_category: ThreatCategory
    intelligence_level: ResponseIntelligenceLevel
    adaptive_capabilities: List[str]
    predictive_models: List[str]
    context_awareness: Dict[str, Any]
    behavioral_analysis: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    temporal_optimization: Dict[str, Any]

    # Enhanced response capabilities
    security_coa: Dict[str, Any]
    security_roe: Dict[str, Any]
    law_enforcement_coa: Dict[str, Any]
    law_enforcement_roe: Dict[str, Any]

    # Intelligent parameters
    response_time_seconds: int
    escalation_path: List[str]
    compliance_standards: List[str]
    force_level: str
    success_criteria: List[str]
    risk_assessment: Dict[str, float]
    confidence_indicators: Dict[str, float]

    # Metadata
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "2.0"
    author: str = "FLAGSHIP Intelligent System"
    tags: List[str] = field(default_factory=list)


@dataclass
class IntelligentResponseAction:
    """Enhanced response action with intelligent capabilities."""

    action_id: str
    protocol_id: str
    action_type: IntelligentResponseType
    intelligence_level: ResponseIntelligenceLevel
    description: str
    target_entities: List[str]
    estimated_duration: timedelta
    required_resources: List[str]
    success_criteria: List[str]
    force_level: str
    compliance_requirements: List[str]
    priority: int

    # Intelligent features
    adaptive_parameters: Dict[str, Any]
    predictive_models: List[str]
    context_awareness: Dict[str, Any]
    behavioral_analysis: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    temporal_optimization: Dict[str, Any]

    # Execution tracking
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    enhancements: List[str] = field(default_factory=list)
    overrides: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligentResponseExecution:
    """Enhanced response execution with intelligent tracking."""

    execution_id: str
    threat_event: ThreatEvent
    protocol: IntelligentCOAROEProtocol
    actions: List[IntelligentResponseAction]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "initiated"
    executed_by: Optional[str] = None

    # Intelligent tracking
    intelligence_level: ResponseIntelligenceLevel
    adaptive_decisions: List[Dict[str, Any]]
    predictive_insights: List[Dict[str, Any]]
    context_analysis: Dict[str, Any]
    behavioral_analysis: Dict[str, Any]
    environmental_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]

    # Compliance and validation
    compliance_verified: bool = False
    validation_status: ComplianceValidationStatus = (
        ComplianceValidationStatus.PENDING_VALIDATION
    )
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    success_indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligentDecisionContext:
    """Context for intelligent decision making."""

    threat_context: Dict[str, Any]
    environmental_context: Dict[str, Any]
    behavioral_context: Dict[str, Any]
    temporal_context: Dict[str, Any]
    spatial_context: Dict[str, Any]
    historical_context: Dict[str, Any]
    compliance_context: Dict[str, Any]
    resource_context: Dict[str, Any]
    risk_context: Dict[str, Any]
    confidence_context: Dict[str, Any]


class EnhancedCOAROEEngine:
    """Enhanced COA/ROE engine with intelligent decision making capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced COA/ROE engine."""
        self.config = config or {}
        self.logger = logger

        # Core components
        self.roe_registry: Optional[DetailedROERegistry] = None
        self.rule_correlation_engine: Optional[RuleCorrelationEngine] = None
        self.response_protocol_manager: Optional[ResponseProtocolManager] = None
        self.legal_compliance_manager: Optional[LegalComplianceManager] = None
        self.escalation_manager: Optional[EscalationManager] = None
        self.notification_coordinator: Optional[NotificationCoordinator] = None
        self.audit_trail_manager: Optional[AuditTrailManager] = None

        # Intelligent features
        self.intelligent_protocols: Dict[str, IntelligentCOAROEProtocol] = {}
        self.adaptive_decisions: List[Dict[str, Any]] = []
        self.predictive_models: Dict[str, Any] = {}
        self.context_analyzer: Dict[str, Any] = {}
        self.behavioral_analyzer: Dict[str, Any] = {}
        self.environmental_analyzer: Dict[str, Any] = {}
        self.temporal_analyzer: Dict[str, Any] = {}

        # State management
        self.is_running = False
        self.active_executions: Dict[str, IntelligentResponseExecution] = {}
        self.execution_history: List[IntelligentResponseExecution] = []

        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.success_rates: Dict[str, float] = {}
        self.response_times: Dict[str, List[float]] = {}

        # Initialize components
        self._initialize_components()
        self._initialize_intelligent_features()

        self.logger.info("Enhanced COA/ROE Engine initialized successfully")

    def _initialize_components(self):
        """Initialize core components."""
        try:
            # Initialize ROE registry
            self.roe_registry = DetailedROERegistry(self.config.get("roe_config", {}))

            # Initialize rule correlation engine
            self.rule_correlation_engine = RuleCorrelationEngine()

            # Initialize response protocol manager
            self.response_protocol_manager = ResponseProtocolManager(
                self.config.get("protocol_config", {})
            )

            # Initialize legal compliance manager
            self.legal_compliance_manager = LegalComplianceManager(
                self.config.get("compliance_config", {})
            )

            # Initialize escalation manager
            self.escalation_manager = EscalationManager(
                self.config.get("escalation_config", {})
            )

            # Initialize notification coordinator
            self.notification_coordinator = NotificationCoordinator(
                self.config.get("notification_config", {})
            )

            # Initialize audit trail manager
            self.audit_trail_manager = AuditTrailManager(
                self.config.get("audit_config", {})
            )

            self.logger.info("Core components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing core components: {e}")

    def _initialize_intelligent_features(self):
        """Initialize intelligent features."""
        try:
            # Initialize context analyzer
            self.context_analyzer = {
                "threat_analysis": {},
                "environmental_analysis": {},
                "behavioral_analysis": {},
                "temporal_analysis": {},
                "spatial_analysis": {},
                "historical_analysis": {},
                "compliance_analysis": {},
                "resource_analysis": {},
                "risk_analysis": {},
                "confidence_analysis": {},
            }

            # Initialize behavioral analyzer
            self.behavioral_analyzer = {
                "user_profiles": {},
                "group_patterns": {},
                "anomaly_detection": {},
                "risk_assessment": {},
                "predictive_models": {},
            }

            # Initialize environmental analyzer
            self.environmental_analyzer = {
                "sensor_data": {},
                "environmental_factors": {},
                "risk_assessment": {},
                "pattern_detection": {},
                "predictive_models": {},
            }

            # Initialize temporal analyzer
            self.temporal_analyzer = {
                "time_patterns": {},
                "seasonal_analysis": {},
                "trend_analysis": {},
                "predictive_models": {},
                "optimization_models": {},
            }

            # Initialize predictive models
            self.predictive_models = {
                "threat_prediction": {},
                "response_prediction": {},
                "escalation_prediction": {},
                "compliance_prediction": {},
                "resource_prediction": {},
            }

            self.logger.info("Intelligent features initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing intelligent features: {e}")

    async def start(self):
        """Start the enhanced COA/ROE engine."""
        if self.is_running:
            return

        self.is_running = True

        # Start core components
        if self.response_protocol_manager:
            await self.response_protocol_manager.start()

        if self.legal_compliance_manager:
            await self.legal_compliance_manager.start()

        if self.escalation_manager:
            await self.escalation_manager.start()

        if self.notification_coordinator:
            await self.notification_coordinator.start()

        if self.audit_trail_manager:
            await self.audit_trail_manager.start()

        self.logger.info("Enhanced COA/ROE Engine started successfully")

    async def stop(self):
        """Stop the enhanced COA/ROE engine."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop core components
        if self.response_protocol_manager:
            await self.response_protocol_manager.stop()

        if self.legal_compliance_manager:
            await self.legal_compliance_manager.stop()

        if self.escalation_manager:
            await self.escalation_manager.stop()

        if self.notification_coordinator:
            await self.notification_coordinator.stop()

        if self.audit_trail_manager:
            await self.audit_trail_manager.stop()

        self.logger.info("Enhanced COA/ROE Engine stopped")

    async def generate_intelligent_response(
        self, threat_event: ThreatEvent, context: Optional[Dict[str, Any]] = None
    ) -> IntelligentResponseExecution:
        """
        Generate intelligent response using enhanced COA/ROE capabilities.

        This method provides:
        - Context-aware response generation
        - Intelligent decision making
        - Predictive analytics integration
        - Adaptive response optimization
        - Comprehensive compliance validation
        - Real-time risk assessment
        """
        try:
            # Analyze threat context
            decision_context = await self._analyze_decision_context(
                threat_event, context
            )

            # Generate intelligent protocol
            protocol = await self._generate_intelligent_protocol(
                threat_event, decision_context
            )

            # Generate intelligent actions
            actions = await self._generate_intelligent_actions(
                protocol, threat_event, decision_context
            )

            # Create intelligent execution
            execution = IntelligentResponseExecution(
                execution_id=str(uuid4()),
                threat_event=threat_event,
                protocol=protocol,
                actions=actions,
                start_time=datetime.now(),
                intelligence_level=protocol.intelligence_level,
                adaptive_decisions=[],
                predictive_insights=[],
                context_analysis=decision_context.threat_context,
                behavioral_analysis=decision_context.behavioral_context,
                environmental_analysis=decision_context.environmental_context,
                temporal_analysis=decision_context.temporal_context,
            )

            # Store execution
            self.active_executions[execution.execution_id] = execution

            # Log execution
            await self._log_execution(execution)

            self.logger.info(
                f"Intelligent response generated: {execution.execution_id}"
            )
            return execution

        except Exception as e:
            self.logger.error(f"Error generating intelligent response: {e}")
            raise

    async def _analyze_decision_context(
        self, threat_event: ThreatEvent, context: Optional[Dict[str, Any]]
    ) -> IntelligentDecisionContext:
        """Analyze decision context for intelligent response generation."""
        try:
            # Threat context analysis
            threat_context = await self._analyze_threat_context(threat_event)

            # Environmental context analysis
            environmental_context = await self._analyze_environmental_context(
                threat_event, context
            )

            # Behavioral context analysis
            behavioral_context = await self._analyze_behavioral_context(
                threat_event, context
            )

            # Temporal context analysis
            temporal_context = await self._analyze_temporal_context(
                threat_event, context
            )

            # Spatial context analysis
            spatial_context = await self._analyze_spatial_context(threat_event, context)

            # Historical context analysis
            historical_context = await self._analyze_historical_context(
                threat_event, context
            )

            # Compliance context analysis
            compliance_context = await self._analyze_compliance_context(
                threat_event, context
            )

            # Resource context analysis
            resource_context = await self._analyze_resource_context(
                threat_event, context
            )

            # Risk context analysis
            risk_context = await self._analyze_risk_context(threat_event, context)

            # Confidence context analysis
            confidence_context = await self._analyze_confidence_context(
                threat_event, context
            )

            return IntelligentDecisionContext(
                threat_context=threat_context,
                environmental_context=environmental_context,
                behavioral_context=behavioral_context,
                temporal_context=temporal_context,
                spatial_context=spatial_context,
                historical_context=historical_context,
                compliance_context=compliance_context,
                resource_context=resource_context,
                risk_context=risk_context,
                confidence_context=confidence_context,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing decision context: {e}")
            raise

    async def _analyze_threat_context(
        self, threat_event: ThreatEvent
    ) -> Dict[str, Any]:
        """Analyze threat context for intelligent decision making."""
        try:
            threat_context = {
                "threat_level": threat_event.threat_level.value,
                "threat_category": threat_event.threat_category.value,
                "confidence": threat_event.confidence,
                "source": threat_event.source,
                "location": threat_event.location,
                "timestamp": threat_event.timestamp.isoformat(),
                "metadata": threat_event.metadata,
                "related_events": threat_event.related_events,
                "response_actions": threat_event.response_actions,
            }

            # Add threat analysis
            threat_context["threat_analysis"] = {
                "severity_assessment": self._assess_threat_severity(threat_event),
                "risk_factors": self._identify_risk_factors(threat_event),
                "escalation_potential": self._assess_escalation_potential(threat_event),
                "response_urgency": self._assess_response_urgency(threat_event),
            }

            return threat_context

        except Exception as e:
            self.logger.error(f"Error analyzing threat context: {e}")
            return {}

    async def _analyze_environmental_context(
        self, threat_event: ThreatEvent, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze environmental context for intelligent decision making."""
        try:
            environmental_context = {
                "sensor_data": {},
                "environmental_factors": {},
                "risk_assessment": {},
                "pattern_detection": {},
                "predictive_models": {},
            }

            # Extract environmental data from context
            if context and "environmental" in context:
                environmental_context.update(context["environmental"])

            # Analyze environmental factors
            if threat_event.location:
                environmental_context["environmental_factors"] = {
                    "location_type": self._classify_location_type(
                        threat_event.location
                    ),
                    "environmental_risk": self._assess_environmental_risk(
                        threat_event.location
                    ),
                    "accessibility": self._assess_accessibility(threat_event.location),
                    "vulnerability": self._assess_location_vulnerability(
                        threat_event.location
                    ),
                }

            return environmental_context

        except Exception as e:
            self.logger.error(f"Error analyzing environmental context: {e}")
            return {}

    async def _analyze_behavioral_context(
        self, threat_event: ThreatEvent, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze behavioral context for intelligent decision making."""
        try:
            behavioral_context = {
                "user_profiles": {},
                "group_patterns": {},
                "anomaly_detection": {},
                "risk_assessment": {},
                "predictive_models": {},
            }

            # Extract behavioral data from context
            if context and "behavioral" in context:
                behavioral_context.update(context["behavioral"])

            # Analyze behavioral patterns
            behavioral_context["behavioral_analysis"] = {
                "activity_patterns": self._analyze_activity_patterns(threat_event),
                "interaction_patterns": self._analyze_interaction_patterns(
                    threat_event
                ),
                "movement_patterns": self._analyze_movement_patterns(threat_event),
                "anomaly_detection": self._detect_behavioral_anomalies(threat_event),
            }

            return behavioral_context

        except Exception as e:
            self.logger.error(f"Error analyzing behavioral context: {e}")
            return {}

    async def _analyze_temporal_context(
        self, threat_event: ThreatEvent, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal context for intelligent decision making."""
        try:
            temporal_context = {
                "time_patterns": {},
                "seasonal_analysis": {},
                "trend_analysis": {},
                "predictive_models": {},
                "optimization_models": {},
            }

            # Extract temporal data from context
            if context and "temporal" in context:
                temporal_context.update(context["temporal"])

            # Analyze temporal patterns
            temporal_context["temporal_analysis"] = {
                "time_of_day": threat_event.timestamp.hour,
                "day_of_week": threat_event.timestamp.weekday(),
                "is_weekend": threat_event.timestamp.weekday() >= 5,
                "is_business_hours": 8 <= threat_event.timestamp.hour <= 18,
                "season": (threat_event.timestamp.month % 12 + 3) // 3,
                "temporal_risk": self._assess_temporal_risk(threat_event.timestamp),
            }

            return temporal_context

        except Exception as e:
            self.logger.error(f"Error analyzing temporal context: {e}")
            return {}

    async def _generate_intelligent_protocol(
        self, threat_event: ThreatEvent, decision_context: IntelligentDecisionContext
    ) -> IntelligentCOAROEProtocol:
        """Generate intelligent protocol based on threat event and decision context."""
        try:
            # Determine intelligence level
            intelligence_level = self._determine_intelligence_level(
                threat_event, decision_context
            )

            # Generate protocol ID
            protocol_id = f"intelligent_protocol_{threat_event.event_id}_{intelligence_level.value}"

            # Create intelligent protocol
            protocol = IntelligentCOAROEProtocol(
                protocol_id=protocol_id,
                name=f"Intelligent {threat_event.threat_category.value} Response",
                event_type=threat_event.event_id,
                threat_category=threat_event.threat_category,
                intelligence_level=intelligence_level,
                adaptive_capabilities=self._identify_adaptive_capabilities(
                    threat_event, decision_context
                ),
                predictive_models=self._identify_predictive_models(
                    threat_event, decision_context
                ),
                context_awareness=decision_context.threat_context,
                behavioral_analysis=decision_context.behavioral_context,
                environmental_factors=decision_context.environmental_context,
                temporal_optimization=decision_context.temporal_context,
                # Enhanced response capabilities
                security_coa=self._generate_security_coa(
                    threat_event, decision_context
                ),
                security_roe=self._generate_security_roe(
                    threat_event, decision_context
                ),
                law_enforcement_coa=self._generate_law_enforcement_coa(
                    threat_event, decision_context
                ),
                law_enforcement_roe=self._generate_law_enforcement_roe(
                    threat_event, decision_context
                ),
                # Intelligent parameters
                response_time_seconds=self._calculate_response_time(
                    threat_event, decision_context
                ),
                escalation_path=self._determine_escalation_path(
                    threat_event, decision_context
                ),
                compliance_standards=self._identify_compliance_standards(
                    threat_event, decision_context
                ),
                force_level=self._determine_force_level(threat_event, decision_context),
                success_criteria=self._define_success_criteria(
                    threat_event, decision_context
                ),
                risk_assessment=self._assess_overall_risk(
                    threat_event, decision_context
                ),
                confidence_indicators=self._calculate_confidence_indicators(
                    threat_event, decision_context
                ),
                description=f"Intelligent response protocol for {threat_event.threat_category.value} threat",
            )

            # Store protocol
            self.intelligent_protocols[protocol_id] = protocol

            return protocol

        except Exception as e:
            self.logger.error(f"Error generating intelligent protocol: {e}")
            raise

    async def _generate_intelligent_actions(
        self,
        protocol: IntelligentCOAROEProtocol,
        threat_event: ThreatEvent,
        decision_context: IntelligentDecisionContext,
    ) -> List[IntelligentResponseAction]:
        """Generate intelligent actions for the protocol."""
        try:
            actions = []

            # Generate security actions
            security_actions = await self._generate_intelligent_security_actions(
                protocol, threat_event, decision_context
            )
            actions.extend(security_actions)

            # Generate law enforcement actions
            le_actions = await self._generate_intelligent_law_enforcement_actions(
                protocol, threat_event, decision_context
            )
            actions.extend(le_actions)

            # Generate coordination actions
            coordination_actions = (
                await self._generate_intelligent_coordination_actions(
                    protocol, threat_event, decision_context
                )
            )
            actions.extend(coordination_actions)

            # Sort actions by priority
            actions.sort(key=lambda x: x.priority)

            return actions

        except Exception as e:
            self.logger.error(f"Error generating intelligent actions: {e}")
            return []

    def _determine_intelligence_level(
        self, threat_event: ThreatEvent, decision_context: IntelligentDecisionContext
    ) -> ResponseIntelligenceLevel:
        """Determine the intelligence level for response generation."""
        try:
            # Base intelligence level
            base_level = ResponseIntelligenceLevel.ENHANCED

            # Upgrade based on threat level
            if threat_event.threat_level == ThreatLevel.CRITICAL:
                base_level = ResponseIntelligenceLevel.ADAPTIVE
            elif threat_event.threat_level == ThreatLevel.HIGH:
                base_level = ResponseIntelligenceLevel.INTELLIGENT

            # Upgrade based on context complexity
            if (
                decision_context.behavioral_context
                and decision_context.environmental_context
            ):
                if base_level.value < ResponseIntelligenceLevel.PREDICTIVE.value:
                    base_level = ResponseIntelligenceLevel.PREDICTIVE

            # Upgrade based on historical patterns
            if (
                decision_context.historical_context
                and len(decision_context.historical_context) > 10
            ):
                if base_level.value < ResponseIntelligenceLevel.ADAPTIVE.value:
                    base_level = ResponseIntelligenceLevel.ADAPTIVE

            return base_level

        except Exception as e:
            self.logger.error(f"Error determining intelligence level: {e}")
            return ResponseIntelligenceLevel.BASIC

    def _assess_threat_severity(self, threat_event: ThreatEvent) -> float:
        """Assess threat severity based on multiple factors."""
        try:
            severity = 0.0

            # Base severity from threat level
            if threat_event.threat_level == ThreatLevel.CRITICAL:
                severity += 0.8
            elif threat_event.threat_level == ThreatLevel.HIGH:
                severity += 0.6
            elif threat_event.threat_level == ThreatLevel.MEDIUM:
                severity += 0.4
            elif threat_event.threat_level == ThreatLevel.LOW:
                severity += 0.2

            # Adjust based on confidence
            severity *= threat_event.confidence

            # Adjust based on metadata
            if threat_event.metadata:
                if "casualty_potential" in threat_event.metadata:
                    severity += threat_event.metadata["casualty_potential"] * 0.2
                if "infrastructure_impact" in threat_event.metadata:
                    severity += threat_event.metadata["infrastructure_impact"] * 0.2

            return min(1.0, severity)

        except Exception as e:
            self.logger.error(f"Error assessing threat severity: {e}")
            return 0.5

    def _identify_risk_factors(self, threat_event: ThreatEvent) -> List[str]:
        """Identify risk factors for the threat event."""
        try:
            risk_factors = []

            # Threat level risk factors
            if threat_event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                risk_factors.append("high_threat_level")

            # Category-specific risk factors
            if threat_event.threat_category == ThreatCategory.ACTIVE_SHOOTER:
                risk_factors.extend(
                    ["immediate_danger", "casualty_potential", "panic_risk"]
                )
            elif threat_event.threat_category == ThreatCategory.CYBER_SECURITY:
                risk_factors.extend(
                    ["data_breach", "system_compromise", "privacy_violation"]
                )
            elif threat_event.threat_category == ThreatCategory.TERRORISM:
                risk_factors.extend(
                    ["mass_casualty", "infrastructure_damage", "psychological_impact"]
                )

            # Location-based risk factors
            if threat_event.location:
                if "crowded_area" in threat_event.location:
                    risk_factors.append("crowd_risk")
                if "critical_infrastructure" in threat_event.location:
                    risk_factors.append("infrastructure_risk")

            return risk_factors

        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
            return []

    async def _log_execution(self, execution: IntelligentResponseExecution):
        """Log execution for audit and analysis."""
        try:
            # Add to audit trail
            if self.audit_trail_manager:
                await self.audit_trail_manager.log_execution(execution)

            # Update performance metrics
            self._update_performance_metrics(execution)

            # Store in history
            self.execution_history.append(execution)

            # Limit history size
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error logging execution: {e}")

    def _update_performance_metrics(self, execution: IntelligentResponseExecution):
        """Update performance metrics for the execution."""
        try:
            # Calculate response time
            if execution.end_time:
                response_time = (
                    execution.end_time - execution.start_time
                ).total_seconds()

                # Update response times
                if execution.protocol.event_type not in self.response_times:
                    self.response_times[execution.protocol.event_type] = []
                self.response_times[execution.protocol.event_type].append(response_time)

                # Keep only last 100 response times
                if len(self.response_times[execution.protocol.event_type]) > 100:
                    self.response_times[execution.protocol.event_type] = (
                        self.response_times[execution.protocol.event_type][-100:]
                    )

            # Update success rates
            if execution.status == "completed":
                if execution.protocol.event_type not in self.success_rates:
                    self.success_rates[execution.protocol.event_type] = 0.0

                # Calculate success rate
                total_executions = len(
                    [
                        e
                        for e in self.execution_history
                        if e.protocol.event_type == execution.protocol.event_type
                    ]
                )
                successful_executions = len(
                    [
                        e
                        for e in self.execution_history
                        if e.protocol.event_type == execution.protocol.event_type
                        and e.status == "completed"
                    ]
                )

                if total_executions > 0:
                    self.success_rates[execution.protocol.event_type] = (
                        successful_executions / total_executions
                    )

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def get_intelligent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the enhanced COA/ROE engine."""
        try:
            status = {
                "system_status": "intelligent" if self.is_running else "stopped",
                "intelligent_protocols": len(self.intelligent_protocols),
                "active_executions": len(self.active_executions),
                "execution_history": len(self.execution_history),
                "performance_metrics": {
                    "success_rates": self.success_rates,
                    "average_response_times": {
                        event_type: np.mean(times) if times else 0.0
                        for event_type, times in self.response_times.items()
                    },
                },
                "intelligent_features": {
                    "context_analyzer": len(self.context_analyzer),
                    "behavioral_analyzer": len(self.behavioral_analyzer),
                    "environmental_analyzer": len(self.environmental_analyzer),
                    "temporal_analyzer": len(self.temporal_analyzer),
                    "predictive_models": len(self.predictive_models),
                },
                "component_status": {
                    "roe_registry": (
                        "available" if self.roe_registry else "not_available"
                    ),
                    "rule_correlation_engine": (
                        "available" if self.rule_correlation_engine else "not_available"
                    ),
                    "response_protocol_manager": (
                        "available"
                        if self.response_protocol_manager
                        else "not_available"
                    ),
                    "legal_compliance_manager": (
                        "available"
                        if self.legal_compliance_manager
                        else "not_available"
                    ),
                    "escalation_manager": (
                        "available" if self.escalation_manager else "not_available"
                    ),
                    "notification_coordinator": (
                        "available"
                        if self.notification_coordinator
                        else "not_available"
                    ),
                    "audit_trail_manager": (
                        "available" if self.audit_trail_manager else "not_available"
                    ),
                },
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting intelligent status: {e}")
            return {"error": str(e)}

    async def get_execution_insights(self, execution_id: str) -> Dict[str, Any]:
        """Get detailed insights for a specific execution."""
        try:
            if execution_id not in self.active_executions:
                return {"error": "Execution not found"}

            execution = self.active_executions[execution_id]

            insights = {
                "execution_id": execution_id,
                "threat_event": {
                    "event_id": execution.threat_event.event_id,
                    "threat_level": execution.threat_event.threat_level.value,
                    "threat_category": execution.threat_event.threat_category.value,
                    "confidence": execution.threat_event.confidence,
                },
                "protocol": {
                    "protocol_id": execution.protocol.protocol_id,
                    "intelligence_level": execution.protocol.intelligence_level.value,
                    "adaptive_capabilities": execution.protocol.adaptive_capabilities,
                    "predictive_models": execution.protocol.predictive_models,
                },
                "actions": [
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type.value,
                        "intelligence_level": action.intelligence_level.value,
                        "priority": action.priority,
                        "status": "pending",
                    }
                    for action in execution.actions
                ],
                "context_analysis": execution.context_analysis,
                "behavioral_analysis": execution.behavioral_analysis,
                "environmental_analysis": execution.environmental_analysis,
                "temporal_analysis": execution.temporal_analysis,
                "performance_metrics": execution.performance_metrics,
                "success_indicators": execution.success_indicators,
            }

            return insights

        except Exception as e:
            self.logger.error(f"Error getting execution insights: {e}")
            return {"error": str(e)}
