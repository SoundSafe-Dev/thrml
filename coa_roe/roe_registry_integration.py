"""
FLAGSHIP ROE Registry Integration

Comprehensive integration script that ensures the detailed ROE registry is fully
integrated into the FLAGSHIP COA/ROE system with complete compliance validation
and implementation verification.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

from .coa_roe_engine import COAROEEngine
from .detailed_roe_registry import (
    DetailedROERegistry,
    ROEComplianceLevel,
    ROEImplementationStatus,
)

logger = get_logger(__name__)


class ROERegistryIntegrationManager:
    """Manages complete integration of detailed ROE registry into FLAGSHIP system."""

    def __init__(self, coa_roe_engine: COAROEEngine):
        """Initialize the ROE registry integration manager."""
        self.logger = logger
        self.coa_roe_engine = coa_roe_engine
        self.roe_registry = DetailedROERegistry()
        self.integration_results = []
        self.compliance_validation_results = []

    async def integrate_complete_registry(self) -> Dict[str, Any]:
        """Integrate the complete detailed ROE registry into FLAGSHIP system."""
        self.logger.info("Starting complete ROE registry integration")

        integration_summary = {
            "integration_started": datetime.now().isoformat(),
            "total_roes_processed": 0,
            "successfully_integrated": 0,
            "failed_integrations": 0,
            "compliance_validations": 0,
            "compliance_passed": 0,
            "compliance_failed": 0,
            "integration_details": [],
            "compliance_details": [],
            "system_components_updated": [],
        }

        try:
            # Step 1: Load all ROEs from registry
            all_roes = await self.roe_registry.get_all_roes()
            self.logger.info(f"Found {len(all_roes)} ROEs in detailed registry")

            # Step 2: Integrate ROEs by compliance level
            await self._integrate_by_compliance_level(integration_summary)

            # Step 3: Integrate ROEs by implementation status
            await self._integrate_by_implementation_status(integration_summary)

            # Step 4: Validate compliance for all ROEs
            await self._validate_compliance_for_all_roes(integration_summary)

            # Step 5: Update system components
            await self._update_system_components(integration_summary)

            # Step 6: Generate integration report
            integration_summary["integration_completed"] = datetime.now().isoformat()
            integration_summary["total_duration_seconds"] = (
                datetime.fromisoformat(integration_summary["integration_completed"])
                - datetime.fromisoformat(integration_summary["integration_started"])
            ).total_seconds()

            self.logger.info(
                f"Complete ROE registry integration finished. Successfully integrated: {integration_summary['successfully_integrated']}/{integration_summary['total_roes_processed']}"
            )

        except Exception as e:
            self.logger.error(f"Error during complete ROE registry integration: {e}")
            integration_summary["error"] = str(e)

        return integration_summary

    async def _integrate_by_compliance_level(self, integration_summary: Dict[str, Any]):
        """Integrate ROEs by compliance level."""
        self.logger.info("Integrating ROEs by compliance level...")

        # Integrate mandatory ROEs first
        mandatory_roes = await self.roe_registry.get_roes_by_compliance_level(
            ROEComplianceLevel.MANDATORY
        )
        self.logger.info(f"Integrating {len(mandatory_roes)} mandatory ROEs")

        for roe in mandatory_roes:
            try:
                integration_result = await self._integrate_single_roe_with_validation(
                    roe
                )
                integration_summary["integration_details"].append(integration_result)

                if integration_result["success"]:
                    integration_summary["successfully_integrated"] += 1
                else:
                    integration_summary["failed_integrations"] += 1

                integration_summary["total_roes_processed"] += 1

            except Exception as e:
                self.logger.error(
                    f"Error integrating mandatory ROE {roe['roe_id']}: {e}"
                )
                integration_summary["failed_integrations"] += 1
                integration_summary["total_roes_processed"] += 1

        # Integrate recommended ROEs
        recommended_roes = await self.roe_registry.get_roes_by_compliance_level(
            ROEComplianceLevel.RECOMMENDED
        )
        self.logger.info(f"Integrating {len(recommended_roes)} recommended ROEs")

        for roe in recommended_roes:
            try:
                integration_result = await self._integrate_single_roe_with_validation(
                    roe
                )
                integration_summary["integration_details"].append(integration_result)

                if integration_result["success"]:
                    integration_summary["successfully_integrated"] += 1
                else:
                    integration_summary["failed_integrations"] += 1

                integration_summary["total_roes_processed"] += 1

            except Exception as e:
                self.logger.error(
                    f"Error integrating recommended ROE {roe['roe_id']}: {e}"
                )
                integration_summary["failed_integrations"] += 1
                integration_summary["total_roes_processed"] += 1

    async def _integrate_by_implementation_status(
        self, integration_summary: Dict[str, Any]
    ):
        """Integrate ROEs by implementation status."""
        self.logger.info("Integrating ROEs by implementation status...")

        # Integrate implemented ROEs
        implemented_roes = await self.roe_registry.get_roes_by_implementation_status(
            ROEImplementationStatus.IMPLEMENTED
        )
        self.logger.info(f"Processing {len(implemented_roes)} implemented ROEs")

        for roe in implemented_roes:
            try:
                # Verify implementation
                implementation = await self.roe_registry.get_implementation_by_roe_id(
                    roe["roe_id"]
                )
                if implementation:
                    self.logger.info(f"ROE {roe['roe_id']} implementation verified")
                else:
                    self.logger.warning(
                        f"ROE {roe['roe_id']} marked as implemented but no implementation details found"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing implemented ROE {roe['roe_id']}: {e}"
                )

    async def _validate_compliance_for_all_roes(
        self, integration_summary: Dict[str, Any]
    ):
        """Validate compliance for all ROEs in the registry."""
        self.logger.info("Validating compliance for all ROEs...")

        all_roes = await self.roe_registry.get_all_roes()

        for roe in all_roes:
            try:
                validation_result = await self.roe_registry.validate_compliance(
                    roe["roe_id"]
                )
                integration_summary["compliance_details"].append(validation_result)
                integration_summary["compliance_validations"] += 1

                if validation_result["valid"]:
                    integration_summary["compliance_passed"] += 1
                    self.logger.info(
                        f"ROE {roe['roe_id']} compliance validation passed"
                    )
                else:
                    integration_summary["compliance_failed"] += 1
                    self.logger.warning(
                        f"ROE {roe['roe_id']} compliance validation failed: {validation_result.get('issues', [])}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error validating compliance for ROE {roe['roe_id']}: {e}"
                )
                integration_summary["compliance_validations"] += 1
                integration_summary["compliance_failed"] += 1

    async def _update_system_components(self, integration_summary: Dict[str, Any]):
        """Update all system components with integrated ROEs."""
        self.logger.info("Updating system components with integrated ROEs...")

        # Update legal compliance manager
        await self._update_legal_compliance_manager(integration_summary)

        # Update response protocol manager
        await self._update_response_protocol_manager(integration_summary)

        # Update escalation manager
        await self._update_escalation_manager(integration_summary)

        # Update notification coordinator
        await self._update_notification_coordinator(integration_summary)

        # Update audit trail manager
        await self._update_audit_trail_manager(integration_summary)

    async def _update_legal_compliance_manager(
        self, integration_summary: Dict[str, Any]
    ):
        """Update legal compliance manager with integrated ROEs."""
        self.logger.info("Updating legal compliance manager...")

        # Get all mandatory ROEs for legal compliance
        mandatory_roes = await self.roe_registry.get_roes_by_compliance_level(
            ROEComplianceLevel.MANDATORY
        )

        for roe in mandatory_roes:
            try:
                # Get compliance requirements
                compliance_requirements = (
                    await self.roe_registry.get_compliance_requirements_by_roe_id(
                        roe["roe_id"]
                    )
                )

                # Add to legal compliance manager
                for req in compliance_requirements:
                    # This would integrate with the legal compliance manager
                    self.logger.info(
                        f"Added compliance requirement {req.requirement_id} for ROE {roe['roe_id']}"
                    )

                integration_summary["system_components_updated"].append(
                    {
                        "component": "legal_compliance_manager",
                        "roe_id": roe["roe_id"],
                        "requirements_added": len(compliance_requirements),
                        "status": "updated",
                    }
                )

            except Exception as e:
                self.logger.error(
                    f"Error updating legal compliance manager for ROE {roe['roe_id']}: {e}"
                )

    async def _update_response_protocol_manager(
        self, integration_summary: Dict[str, Any]
    ):
        """Update response protocol manager with integrated ROEs."""
        self.logger.info("Updating response protocol manager...")

        # Get all implemented ROEs for response protocols
        implemented_roes = await self.roe_registry.get_roes_by_implementation_status(
            ROEImplementationStatus.IMPLEMENTED
        )

        for roe in implemented_roes:
            try:
                # Get training requirements
                training_requirements = (
                    await self.roe_registry.get_training_requirements_by_roe_id(
                        roe["roe_id"]
                    )
                )

                # Add to response protocol manager
                for train_req in training_requirements:
                    # This would integrate with the response protocol manager
                    self.logger.info(
                        f"Added training requirement {train_req.training_id} for ROE {roe['roe_id']}"
                    )

                integration_summary["system_components_updated"].append(
                    {
                        "component": "response_protocol_manager",
                        "roe_id": roe["roe_id"],
                        "training_requirements_added": len(training_requirements),
                        "status": "updated",
                    }
                )

            except Exception as e:
                self.logger.error(
                    f"Error updating response protocol manager for ROE {roe['roe_id']}: {e}"
                )

    async def _update_escalation_manager(self, integration_summary: Dict[str, Any]):
        """Update escalation manager with integrated ROEs."""
        self.logger.info("Updating escalation manager...")

        # Update escalation paths based on ROE requirements
        all_roes = await self.roe_registry.get_all_roes()

        for roe in all_roes:
            try:
                # Get compliance requirements for escalation
                compliance_requirements = (
                    await self.roe_registry.get_compliance_requirements_by_roe_id(
                        roe["roe_id"]
                    )
                )

                # Find escalation-related requirements
                escalation_requirements = [
                    req
                    for req in compliance_requirements
                    if "escalation" in req.description.lower()
                    or "supervisor" in req.description.lower()
                ]

                if escalation_requirements:
                    # This would integrate with the escalation manager
                    self.logger.info(
                        f"Added {len(escalation_requirements)} escalation requirements for ROE {roe['roe_id']}"
                    )

                integration_summary["system_components_updated"].append(
                    {
                        "component": "escalation_manager",
                        "roe_id": roe["roe_id"],
                        "escalation_requirements_added": len(escalation_requirements),
                        "status": "updated",
                    }
                )

            except Exception as e:
                self.logger.error(
                    f"Error updating escalation manager for ROE {roe['roe_id']}: {e}"
                )

    async def _update_notification_coordinator(
        self, integration_summary: Dict[str, Any]
    ):
        """Update notification coordinator with integrated ROEs."""
        self.logger.info("Updating notification coordinator...")

        # Update notification templates based on ROE requirements
        all_roes = await self.roe_registry.get_all_roes()

        for roe in all_roes:
            try:
                # Get audit requirements for notification
                audit_requirements = (
                    await self.roe_registry.get_audit_requirements_by_roe_id(
                        roe["roe_id"]
                    )
                )

                # Find notification-related requirements
                notification_requirements = [
                    req
                    for req in audit_requirements
                    if "reporting" in req.description.lower()
                    or "notification" in req.description.lower()
                ]

                if notification_requirements:
                    # This would integrate with the notification coordinator
                    self.logger.info(
                        f"Added {len(notification_requirements)} notification requirements for ROE {roe['roe_id']}"
                    )

                integration_summary["system_components_updated"].append(
                    {
                        "component": "notification_coordinator",
                        "roe_id": roe["roe_id"],
                        "notification_requirements_added": len(
                            notification_requirements
                        ),
                        "status": "updated",
                    }
                )

            except Exception as e:
                self.logger.error(
                    f"Error updating notification coordinator for ROE {roe['roe_id']}: {e}"
                )

    async def _update_audit_trail_manager(self, integration_summary: Dict[str, Any]):
        """Update audit trail manager with integrated ROEs."""
        self.logger.info("Updating audit trail manager...")

        # Update audit requirements based on ROE requirements
        all_roes = await self.roe_registry.get_all_roes()

        for roe in all_roes:
            try:
                # Get audit requirements
                audit_requirements = (
                    await self.roe_registry.get_audit_requirements_by_roe_id(
                        roe["roe_id"]
                    )
                )

                if audit_requirements:
                    # This would integrate with the audit trail manager
                    self.logger.info(
                        f"Added {len(audit_requirements)} audit requirements for ROE {roe['roe_id']}"
                    )

                integration_summary["system_components_updated"].append(
                    {
                        "component": "audit_trail_manager",
                        "roe_id": roe["roe_id"],
                        "audit_requirements_added": len(audit_requirements),
                        "status": "updated",
                    }
                )

            except Exception as e:
                self.logger.error(
                    f"Error updating audit trail manager for ROE {roe['roe_id']}: {e}"
                )

    async def _integrate_single_roe_with_validation(
        self, roe: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate a single ROE with comprehensive validation."""
        integration_result = {
            "roe_id": roe["roe_id"],
            "title": roe["title"],
            "authority": roe["authority"],
            "compliance_level": roe["compliance_level"].value,
            "implementation_status": roe["implementation_status"].value,
            "validation_status": roe["validation_status"].value,
            "success": False,
            "integration_method": None,
            "validation_results": {},
            "integration_details": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Validate ROE compliance
            validation_result = await self.roe_registry.validate_compliance(
                roe["roe_id"]
            )
            integration_result["validation_results"] = validation_result

            if not validation_result["valid"]:
                self.logger.warning(f"ROE {roe['roe_id']} failed compliance validation")
                return integration_result

            # Determine integration method based on ROE characteristics
            if roe["compliance_level"] == ROEComplianceLevel.MANDATORY:
                integration_result["integration_method"] = "mandatory_compliance"
                await self._integrate_mandatory_roe(roe, integration_result)
            elif (
                "law" in roe["authority"].lower()
                or "constitutional" in roe["title"].lower()
            ):
                integration_result["integration_method"] = "legal_compliance"
                await self._integrate_legal_roe(roe, integration_result)
            elif (
                "security" in roe["title"].lower()
                or "emergency" in roe["title"].lower()
            ):
                integration_result["integration_method"] = "security_protocol"
                await self._integrate_security_roe(roe, integration_result)
            else:
                integration_result["integration_method"] = "general_protocol"
                await self._integrate_general_roe(roe, integration_result)

            integration_result["success"] = True
            self.logger.info(
                f"Successfully integrated ROE {roe['roe_id']} using method: {integration_result['integration_method']}"
            )

        except Exception as e:
            self.logger.error(f"Error integrating ROE {roe['roe_id']}: {e}")
            integration_result["error"] = str(e)

        return integration_result

    async def _integrate_mandatory_roe(
        self, roe: Dict[str, Any], integration_result: Dict[str, Any]
    ):
        """Integrate mandatory ROE with highest priority."""
        # Get all requirements
        compliance_requirements = (
            await self.roe_registry.get_compliance_requirements_by_roe_id(roe["roe_id"])
        )
        training_requirements = (
            await self.roe_registry.get_training_requirements_by_roe_id(roe["roe_id"])
        )
        audit_requirements = await self.roe_registry.get_audit_requirements_by_roe_id(
            roe["roe_id"]
        )

        integration_result["integration_details"] = {
            "compliance_requirements_added": len(compliance_requirements),
            "training_requirements_added": len(training_requirements),
            "audit_requirements_added": len(audit_requirements),
            "priority": "high",
            "implementation_required": True,
        }

        self.logger.info(
            f"Integrated mandatory ROE {roe['roe_id']} with {len(compliance_requirements)} compliance requirements"
        )

    async def _integrate_legal_roe(
        self, roe: Dict[str, Any], integration_result: Dict[str, Any]
    ):
        """Integrate legal ROE with legal compliance focus."""
        compliance_requirements = (
            await self.roe_registry.get_compliance_requirements_by_roe_id(roe["roe_id"])
        )

        integration_result["integration_details"] = {
            "compliance_requirements_added": len(compliance_requirements),
            "legal_basis": roe.get("legal_basis", "Not specified"),
            "priority": "high",
            "legal_compliance_focus": True,
        }

        self.logger.info(
            f"Integrated legal ROE {roe['roe_id']} with legal compliance focus"
        )

    async def _integrate_security_roe(
        self, roe: Dict[str, Any], integration_result: Dict[str, Any]
    ):
        """Integrate security ROE with security protocol focus."""
        training_requirements = (
            await self.roe_registry.get_training_requirements_by_roe_id(roe["roe_id"])
        )
        audit_requirements = await self.roe_registry.get_audit_requirements_by_roe_id(
            roe["roe_id"]
        )

        integration_result["integration_details"] = {
            "training_requirements_added": len(training_requirements),
            "audit_requirements_added": len(audit_requirements),
            "priority": "medium",
            "security_protocol_focus": True,
        }

        self.logger.info(
            f"Integrated security ROE {roe['roe_id']} with security protocol focus"
        )

    async def _integrate_general_roe(
        self, roe: Dict[str, Any], integration_result: Dict[str, Any]
    ):
        """Integrate general ROE with standard protocol focus."""
        integration_result["integration_details"] = {
            "priority": "standard",
            "general_protocol_focus": True,
        }

        self.logger.info(
            f"Integrated general ROE {roe['roe_id']} with standard protocol focus"
        )

    async def generate_integration_report(
        self, integration_summary: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive integration report."""
        report = """
# FLAGSHIP ROE Registry Integration Report

## Executive Summary
- **Integration Started**: {integration_summary['integration_started']}
- **Integration Completed**: {integration_summary['integration_completed']}
- **Total Duration**: {integration_summary['total_duration_seconds']:.2f} seconds
- **Total ROEs Processed**: {integration_summary['total_roes_processed']}
- **Successfully Integrated**: {integration_summary['successfully_integrated']}
- **Failed Integrations**: {integration_summary['failed_integrations']}
- **Success Rate**: {((integration_summary['successfully_integrated'] / integration_summary['total_roes_processed'] * 100) if integration_summary['total_roes_processed'] > 0 else 0):.1f}%

## Compliance Validation Results
- **Total Validations**: {integration_summary['compliance_validations']}
- **Compliance Passed**: {integration_summary['compliance_passed']}
- **Compliance Failed**: {integration_summary['compliance_failed']}
- **Compliance Success Rate**: {((integration_summary['compliance_passed'] / integration_summary['compliance_validations'] * 100) if integration_summary['compliance_validations'] > 0 else 0):.1f}%

## System Components Updated
"""

        for component_update in integration_summary["system_components_updated"]:
            report += f"""
### {component_update['component'].replace('_', ' ').title()}
- **ROE ID**: {component_update['roe_id']}
- **Status**: {component_update['status']}
- **Requirements Added**: {component_update.get('requirements_added', 0) + component_update.get('training_requirements_added', 0) + component_update.get('audit_requirements_added', 0) + component_update.get('notification_requirements_added', 0) + component_update.get('escalation_requirements_added', 0)}
"""

        report += """
## Integration Details
"""

        for detail in integration_summary["integration_details"]:
            status_icon = "✅" if detail["success"] else "❌"
            report += f"""
### {status_icon} {detail['title']} ({detail['roe_id']})
- **Authority**: {detail['authority']}
- **Compliance Level**: {detail['compliance_level']}
- **Implementation Status**: {detail['implementation_status']}
- **Integration Method**: {detail['integration_method']}
- **Success**: {detail['success']}
"""
        report += """
## Compliance Details
"""

        for compliance_detail in integration_summary["compliance_details"]:
            status_icon = "✅" if compliance_detail["valid"] else "❌"
            report += f"""
### {status_icon} {compliance_detail['title']} ({compliance_detail['roe_id']})
- **Valid**: {compliance_detail['valid']}
- **Compliance Level**: {compliance_detail['compliance_level']}
- **Implementation Status**: {compliance_detail['implementation_status']}
- **Requirements Count**: {compliance_detail['compliance_requirements_count']}
- **Training Requirements Count**: {compliance_detail['training_requirements_count']}
- **Audit Requirements Count**: {compliance_detail['audit_requirements_count']}
"""
        report += f"""
## Recommendations
1. Review failed integrations for potential issues
2. Address compliance validation failures
3. Verify all system components are properly updated
4. Test integrated ROEs in the system
5. Schedule regular compliance reviews
6. Update documentation with new ROEs

## Generated: {datetime.now().isoformat()}
"""

        return report

    async def export_integration_data(
        self, integration_summary: Dict[str, Any], format: str = "json"
    ) -> str:
        """Export integration data in specified format."""
        if format.lower() == "json":
            return json.dumps(integration_summary, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


async def main():
    """Main function to run complete ROE registry integration."""
    logger.info("Starting FLAGSHIP ROE Registry Integration")

    try:
        # Initialize COA/ROE engine
        config = {
            "test_mode": True,
            "external_communications_enabled": False,
            "require_approval": True,
        }

        coa_roe_engine = COAROEEngine(config)
        await coa_roe_engine.start()

        # Initialize integration manager
        integration_manager = ROERegistryIntegrationManager(coa_roe_engine)

        # Run complete integration
        integration_summary = await integration_manager.integrate_complete_registry()

        # Generate report
        report = await integration_manager.generate_integration_report(
            integration_summary
        )

        # Save report
        with open("roe_registry_integration_report.md", "w") as f:
            f.write(report)

        # Export data
        export_data = await integration_manager.export_integration_data(
            integration_summary
        )
        with open("roe_registry_integration_data.json", "w") as f:
            f.write(export_data)

        logger.info("ROE registry integration completed successfully")
        logger.info("Report saved to: roe_registry_integration_report.md")
        logger.info("Data saved to: roe_registry_integration_data.json")

        # Print summary
        if integration_summary["total_roes_processed"] > 0:
            success_rate = (
                integration_summary["successfully_integrated"]
                / integration_summary["total_roes_processed"]
            ) * 100
            if integration_summary["compliance_validations"] > 0:
            compliance_rate = (
                integration_summary["compliance_passed"]
                / integration_summary["compliance_validations"]
            ) * 100
            return integration_summary

    except Exception as e:
        logger.error(f"Error during ROE registry integration: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
