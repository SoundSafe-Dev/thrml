"""
FLAGSHIP Call-Of-Action & Rules-Of-Engagement System

Provides comprehensive Course-of-Action (COA) and Rules-of-Engagement (ROE)
management for security and law enforcement responses to threat events.

This system integrates directly with the Unified Threat Engine (UTE) to:
- Generate appropriate security responses based on threat events
- Enforce legal and ethical rules of engagement
- Coordinate between security and law enforcement
- Ensure compliance with international standards
- Provide escalation paths and response coordination

Key Features:
- Human-only response protocols (no drones/robots)
- Legal compliance with UN Code of Conduct
- OSHA emergency action requirements
- U.S. constitutional use-of-force standards
- Multi-tier escalation management
- Real-time response coordination
- Comprehensive audit trails

Advanced Features:
- Redis-based distributed caching with cache warming and invalidation
- Real-time performance monitoring and analytics
- Machine learning-based rule optimization
- Advanced external integrations and compliance reporting
- Comprehensive dashboard integration
- Advanced testing and validation capabilities
"""

from .audit_trail import AuditTrailManager
from .coa_roe_engine import COAROEEngine
from .escalation_manager import EscalationManager
from .legal_compliance import LegalComplianceManager
from .notification_coordinator import NotificationCoordinator
from .response_protocols import ResponseProtocol, ResponseProtocolManager

# Advanced features
try:
    from .cache_manager import AdvancedCacheManager, CacheConfig, CacheTier
    from .external_integrations import (
        ComplianceReport,
        ExternalIntegrationsManager,
        IntegrationConfig,
        IntegrationResult,
        IntegrationType,
        ROEStandard,
    )
    from .performance_monitor import (
        AlertThreshold,
        PerformanceMetric,
        PerformanceMonitor,
    )
    from .rule_optimizer import (
        OptimizationResult,
        RuleFeature,
        RuleOptimizer,
        RulePrediction,
    )
    from .test_advanced_features import (
        AdvancedTestingManager,
        LoadTestScenario,
        TestConfig,
        TestResult,
        TestStatus,
        TestType,
    )
except Exception:  # optional advanced imports may fail in minimal envs
    AdvancedCacheManager = CacheConfig = CacheTier = None  # type: ignore
    PerformanceMonitor = PerformanceMetric = AlertThreshold = None  # type: ignore
    RuleOptimizer = RuleFeature = RulePrediction = OptimizationResult = None  # type: ignore
    ExternalIntegrationsManager = IntegrationConfig = IntegrationResult = ComplianceReport = None  # type: ignore
    IntegrationType = ROEStandard = None  # type: ignore
    AdvancedTestingManager = TestConfig = TestResult = LoadTestScenario = TestType = TestStatus = None  # type: ignore

__all__ = [
    # Core components
    "COAROEEngine",
    "ResponseProtocol",
    "ResponseProtocolManager",
    "LegalComplianceManager",
    "EscalationManager",
    "NotificationCoordinator",
    "AuditTrailManager",
    # Advanced caching
    "AdvancedCacheManager",
    "CacheConfig",
    "CacheTier",
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceMetric",
    "AlertThreshold",
    # Rule optimization
    "RuleOptimizer",
    "RuleFeature",
    "RulePrediction",
    "OptimizationResult",
    # External integrations
    "ExternalIntegrationsManager",
    "IntegrationConfig",
    "IntegrationResult",
    "ComplianceReport",
    "IntegrationType",
    "ROEStandard",
    # Advanced testing
    "AdvancedTestingManager",
    "TestConfig",
    "TestResult",
    "LoadTestScenario",
    "TestType",
    "TestStatus",
]
