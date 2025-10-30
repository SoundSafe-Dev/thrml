"""
FLAGSHIP COA/ROE System Integration Test

Demonstrates the complete COA/ROE system integration with the Unified Threat Engine (UTE).
Shows how all notifications go through FLAGSHIP first with no direct external communications.
Tests the streamlined integration with injected registry instances and error handling.
"""

import asyncio
from datetime import datetime

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

from ..__init__ import UnifiedThreatEngine

logger = get_logger(__name__)


class COAROEIntegrationTest:
    """Integration test for the COA/ROE system with streamlined integration."""

    def __init__(self):
        """Initialize the integration test."""
        self.logger = logger
        self.ute = None
        self.coa_roe_engine = None
        self.roe_registry = None
        self.rule_correlation_engine = None
        self.test_results = []

    async def setup(self):
        """Set up the COA/ROE system for testing with streamlined integration."""
        self.logger.info(
            "Setting up COA/ROE Integration Test with streamlined integration"
        )

        # Test configuration with injected registry instances
        config = {
            "test_mode": True,
            "external_communications_enabled": False,
            "require_approval": True,
            "coa_roe_config": {
                "test_mode": True,
                "external_communications_enabled": False,
                "require_approval": True,
                "lazy_load": True,
                "enable_caching": True,
            },
            "uts_config": {"test_mode": True},
            "threat_config": {"test_mode": True},
            "decision_config": {"test_mode": True},
            "risk_config": {"test_mode": True},
            "history_config": {"test_mode": True},
            "scenario_config": {"test_mode": True},
        }

        # Initialize UTE with injected registry instances
        self.ute = UnifiedThreatEngine(config)
        await self.ute.start()

        # Verify registry instances are properly injected
        self.roe_registry = self.ute.roe_registry
        self.rule_correlation_engine = self.ute.rule_correlation_engine
        self.coa_roe_engine = self.ute.coa_roe_engine

        self.logger.info("COA/ROE system initialized with streamlined integration")
        self.logger.warning(
            "EXTERNAL COMMUNICATIONS DISABLED - All notifications will go through FLAGSHIP only"
        )

    async def test_registry_injection(self):
        """Test that registry instances are properly injected."""
        self.logger.info("Testing registry injection...")

        try:
            # Verify registry instances are injected
            assert self.roe_registry is not None, "ROE registry should be injected"
            assert (
                self.rule_correlation_engine is not None
            ), "Rule correlation engine should be injected"
            assert self.coa_roe_engine is not None, "COA/ROE engine should be injected"

            # Verify registry instances are shared
            assert (
                self.coa_roe_engine.roe_registry is self.roe_registry
            ), "ROE registry should be shared"
            assert (
                self.coa_roe_engine.rule_correlation_engine
                is self.rule_correlation_engine
            ), "Rule correlation engine should be shared"

            self.logger.info("✅ Registry injection test passed")
            self.test_results.append({"test": "registry_injection", "status": "passed"})

        except Exception as e:
            self.logger.error(f"❌ Registry injection test failed: {e}")
            self.test_results.append(
                {"test": "registry_injection", "status": "failed", "error": str(e)}
            )

    async def test_lazy_loading(self):
        """Test lazy loading functionality of the registry."""
        self.logger.info("Testing lazy loading...")

        try:
            # Test that registry is not loaded initially if lazy loading is enabled
            if self.roe_registry.lazy_load:
                assert (
                    not self.roe_registry._is_loaded
                ), "Registry should not be loaded initially with lazy loading"

                # Trigger loading by accessing data
                summary = await self.roe_registry.get_roe_summary()
                assert (
                    self.roe_registry._is_loaded
                ), "Registry should be loaded after first access"
                assert summary["total_roes"] > 0, "Registry should contain ROEs"

            self.logger.info("✅ Lazy loading test passed")
            self.test_results.append({"test": "lazy_loading", "status": "passed"})

        except Exception as e:
            self.logger.error(f"❌ Lazy loading test failed: {e}")
            self.test_results.append(
                {"test": "lazy_loading", "status": "failed", "error": str(e)}
            )

    async def test_caching_functionality(self):
        """Test caching functionality of the registry."""
        self.logger.info("Testing caching functionality...")

        try:
            if self.roe_registry.enable_caching:
                # First access should populate cache
                roes1 = await self.roe_registry.get_all_roes()
                assert len(roes1) > 0, "Should return ROEs"

                # Second access should use cache
                roes2 = await self.roe_registry.get_all_roes()
                assert roes1 == roes2, "Cached results should be identical"

                # Test cache invalidation
                await self.roe_registry.refresh_cache()
                roes3 = await self.roe_registry.get_all_roes()
                assert len(roes3) > 0, "Should return ROEs after cache refresh"

            self.logger.info("✅ Caching functionality test passed")
            self.test_results.append(
                {"test": "caching_functionality", "status": "passed"}
            )

        except Exception as e:
            self.logger.error(f"❌ Caching functionality test failed: {e}")
            self.test_results.append(
                {"test": "caching_functionality", "status": "failed", "error": str(e)}
            )

    async def test_error_handling(self):
        """Test error handling and fallback mechanisms."""
        self.logger.info("Testing error handling...")

        try:
            # Test with invalid ROE ID
            invalid_roe = await self.roe_registry.get_roe_by_id("INVALID-ROE-ID")
            assert invalid_roe is None, "Should return None for invalid ROE ID"

            # Test rule correlations with invalid rule
            correlations = await self.roe_registry.get_rule_correlations(
                "INVALID-RULE-ID"
            )
            assert isinstance(
                correlations, list
            ), "Should return empty list for invalid rule"

            # Test COA generation with invalid threat event
            invalid_response = (
                await self.coa_roe_engine.generate_response_with_rule_correlations({})
            )
            assert (
                invalid_response is not None
            ), "Should return fallback response for invalid threat event"

            self.logger.info("✅ Error handling test passed")
            self.test_results.append({"test": "error_handling", "status": "passed"})

        except Exception as e:
            self.logger.error(f"❌ Error handling test failed: {e}")
            self.test_results.append(
                {"test": "error_handling", "status": "failed", "error": str(e)}
            )

    async def test_gunshot_detection_response(self):
        """Test gunshot detection response with rule correlations."""
        self.logger.info("Testing gunshot detection response...")

        try:
            threat_event = {
                "threat_type": "active_shooter",
                "threat_level": "critical",
                "location": {"building": "Main Building", "floor": "1st Floor"},
                "description": "Gunshot detected in main building",
                "confidence": 0.95,
                "source": "audio_sensor",
                "metadata": {
                    "sensor_id": "AUDIO_001",
                    "timestamp": datetime.now().isoformat(),
                },
            }

            response = (
                await self.coa_roe_engine.generate_response_with_rule_correlations(
                    threat_event
                )
            )

            assert (
                response is not None
            ), "Should generate response for gunshot detection"
            assert hasattr(
                response, "execution_id"
            ), "Response should have execution ID"
            assert hasattr(response, "actions"), "Response should have actions"
            assert len(response.actions) > 0, "Response should have at least one action"

            self.logger.info(
                f"✅ Gunshot detection response test passed - Generated {len(response.actions)} actions"
            )
            self.test_results.append(
                {
                    "test": "gunshot_detection_response",
                    "status": "passed",
                    "actions_count": len(response.actions),
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Gunshot detection response test failed: {e}")
            self.test_results.append(
                {
                    "test": "gunshot_detection_response",
                    "status": "failed",
                    "error": str(e),
                }
            )

    async def test_explosion_detection_response(self):
        """Test explosion detection response with rule correlations."""
        self.logger.info("Testing explosion detection response...")

        try:
            threat_event = {
                "threat_type": "explosive",
                "threat_level": "critical",
                "location": {"building": "Warehouse", "area": "Loading Dock"},
                "description": "Explosion detected in warehouse",
                "confidence": 0.90,
                "source": "vibration_sensor",
                "metadata": {
                    "sensor_id": "VIB_001",
                    "timestamp": datetime.now().isoformat(),
                },
            }

            response = (
                await self.coa_roe_engine.generate_response_with_rule_correlations(
                    threat_event
                )
            )

            assert (
                response is not None
            ), "Should generate response for explosion detection"
            assert hasattr(
                response, "execution_id"
            ), "Response should have execution ID"
            assert hasattr(response, "actions"), "Response should have actions"
            assert len(response.actions) > 0, "Response should have at least one action"

            self.logger.info(
                f"✅ Explosion detection response test passed - Generated {len(response.actions)} actions"
            )
            self.test_results.append(
                {
                    "test": "explosion_detection_response",
                    "status": "passed",
                    "actions_count": len(response.actions),
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Explosion detection response test failed: {e}")
            self.test_results.append(
                {
                    "test": "explosion_detection_response",
                    "status": "failed",
                    "error": str(e),
                }
            )

    async def test_trespassing_detection_response(self):
        """Test trespassing detection response with rule correlations."""
        self.logger.info("Testing trespassing detection response...")

        try:
            threat_event = {
                "threat_type": "physical_security",
                "threat_level": "medium",
                "location": {"building": "Perimeter", "area": "North Gate"},
                "description": "Unauthorized access detected at north gate",
                "confidence": 0.85,
                "source": "motion_sensor",
                "metadata": {
                    "sensor_id": "MOTION_001",
                    "timestamp": datetime.now().isoformat(),
                },
            }

            response = (
                await self.coa_roe_engine.generate_response_with_rule_correlations(
                    threat_event
                )
            )

            assert (
                response is not None
            ), "Should generate response for trespassing detection"
            assert hasattr(
                response, "execution_id"
            ), "Response should have execution ID"
            assert hasattr(response, "actions"), "Response should have actions"
            assert len(response.actions) > 0, "Response should have at least one action"

            self.logger.info(
                f"✅ Trespassing detection response test passed - Generated {len(response.actions)} actions"
            )
            self.test_results.append(
                {
                    "test": "trespassing_detection_response",
                    "status": "passed",
                    "actions_count": len(response.actions),
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Trespassing detection response test failed: {e}")
            self.test_results.append(
                {
                    "test": "trespassing_detection_response",
                    "status": "failed",
                    "error": str(e),
                }
            )

    async def test_notification_safety(self):
        """Test that notifications go through FLAGSHIP system only."""
        self.logger.info("Testing notification safety...")

        try:
            # Verify test mode is enabled
            assert (
                self.coa_roe_engine.notification_coordinator.test_mode
            ), "Test mode should be enabled"
            assert (
                not self.coa_roe_engine.notification_coordinator.external_communications_enabled
            ), "External communications should be disabled"

            # Test notification sending
            notification_result = await self.coa_roe_engine.notification_coordinator.send_test_notification(
                "Test notification for safety verification"
            )

            assert notification_result, "Notification should be sent successfully"

            self.logger.info("✅ Notification safety test passed")
            self.test_results.append(
                {"test": "notification_safety", "status": "passed"}
            )

        except Exception as e:
            self.logger.error(f"❌ Notification safety test failed: {e}")
            self.test_results.append(
                {"test": "notification_safety", "status": "failed", "error": str(e)}
            )

    async def test_compliance_verification(self):
        """Test compliance verification functionality."""
        self.logger.info("Testing compliance verification...")

        try:
            # Test compliance verification for a known ROE
            compliance_result = await self.roe_registry.validate_compliance("UN-001")

            assert (
                compliance_result is not None
            ), "Should return compliance validation result"
            assert (
                "valid" in compliance_result
            ), "Compliance result should have valid field"
            assert (
                "roe_id" in compliance_result
            ), "Compliance result should have roe_id field"

            self.logger.info("✅ Compliance verification test passed")
            self.test_results.append(
                {"test": "compliance_verification", "status": "passed"}
            )

        except Exception as e:
            self.logger.error(f"❌ Compliance verification test failed: {e}")
            self.test_results.append(
                {"test": "compliance_verification", "status": "failed", "error": str(e)}
            )

    async def test_escalation_management(self):
        """Test escalation management functionality."""
        self.logger.info("Testing escalation management...")

        try:
            # Test escalation initiation
            escalation_id = (
                await self.coa_roe_engine.escalation_manager.initiate_escalation(
                    execution_id="TEST-EXEC-001",
                    escalation_path_id="SECURITY_ESCALATION",
                    reason="Test escalation for integration verification",
                    urgency="normal",
                )
            )

            assert escalation_id is not None, "Should return escalation ID"

            # Test getting active escalations
            active_escalations = (
                await self.coa_roe_engine.escalation_manager.get_active_escalations()
            )
            assert isinstance(
                active_escalations, list
            ), "Should return list of active escalations"

            self.logger.info("✅ Escalation management test passed")
            self.test_results.append(
                {"test": "escalation_management", "status": "passed"}
            )

        except Exception as e:
            self.logger.error(f"❌ Escalation management test failed: {e}")
            self.test_results.append(
                {"test": "escalation_management", "status": "failed", "error": str(e)}
            )

    async def test_audit_trail(self):
        """Test audit trail functionality."""
        self.logger.info("Testing audit trail...")

        try:
            # Test audit trail logging
            await self.coa_roe_engine.audit_trail.log_execution_start(
                "TEST-EXEC-001",
                {
                    "test": "audit_trail_verification",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Test getting audit events
            audit_events = await self.coa_roe_engine.audit_trail.get_audit_events()
            assert isinstance(audit_events, list), "Should return list of audit events"

            self.logger.info("✅ Audit trail test passed")
            self.test_results.append({"test": "audit_trail", "status": "passed"})

        except Exception as e:
            self.logger.error(f"❌ Audit trail test failed: {e}")
            self.test_results.append(
                {"test": "audit_trail", "status": "failed", "error": str(e)}
            )

    async def test_ute_integration(self):
        """Test full UTE integration with COA/ROE system."""
        self.logger.info("Testing full UTE integration...")

        try:
            # Create test sensor data
            sensor_data = {
                "audio": {"amplitude": 0.8, "frequency": "high"},
                "video": {"motion_detected": True, "object_class": "person"},
                "motion": {"detected": True, "location": "north_gate"},
            }

            # Perform threat assessment
            assessment = await self.ute.assess_threat(sensor_data, {"test": True})

            assert assessment is not None, "Should return threat assessment"
            assert hasattr(
                assessment, "threat_events"
            ), "Assessment should have threat events"
            assert hasattr(assessment, "metadata"), "Assessment should have metadata"
            assert (
                "coa_responses" in assessment.metadata
            ), "Assessment should include COA responses"

            self.logger.info(
                f"✅ UTE integration test passed - Generated {len(assessment.threat_events)} threat events"
            )
            self.test_results.append(
                {
                    "test": "ute_integration",
                    "status": "passed",
                    "threat_events_count": len(assessment.threat_events),
                }
            )

        except Exception as e:
            self.logger.error(f"❌ UTE integration test failed: {e}")
            self.test_results.append(
                {"test": "ute_integration", "status": "failed", "error": str(e)}
            )

    async def run_all_tests(self):
        """Run all integration tests."""
        self.logger.info("Starting comprehensive COA/ROE integration tests...")

        try:
            await self.setup()

            # Run all tests
            await self.test_registry_injection()
            await self.test_lazy_loading()
            await self.test_caching_functionality()
            await self.test_error_handling()
            await self.test_gunshot_detection_response()
            await self.test_explosion_detection_response()
            await self.test_trespassing_detection_response()
            await self.test_notification_safety()
            await self.test_compliance_verification()
            await self.test_escalation_management()
            await self.test_audit_trail()
            await self.test_ute_integration()

            # Generate test summary
            passed_tests = [t for t in self.test_results if t["status"] == "passed"]
            failed_tests = [t for t in self.test_results if t["status"] == "failed"]

            self.logger.info(
                f"Integration tests completed: {len(passed_tests)} passed, {len(failed_tests)} failed"
            )

            if failed_tests:
                self.logger.error("Failed tests:")
                for test in failed_tests:
                    self.logger.error(
                        f"  - {test['test']}: {test.get('error', 'Unknown error')}"
                    )

            return len(failed_tests) == 0

        except Exception as e:
            self.logger.error(f"Integration test suite failed: {e}")
            return False

        finally:
            if self.ute:
                await self.ute.stop()

    async def cleanup(self):
        """Clean up test resources."""
        if self.ute:
            await self.ute.stop()
        self.logger.info("Test cleanup completed")


async def main():
    """Main test execution function."""
    test = COAROEIntegrationTest()

    try:
        success = await test.run_all_tests()

        if success:
            logger.info("🎉 All integration tests passed!")
        else:
            logger.error("❌ Some integration tests failed!")

        return success

    except Exception as e:
        logger.error(f"Integration test suite failed: {e}")
        return False

    finally:
        await test.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
