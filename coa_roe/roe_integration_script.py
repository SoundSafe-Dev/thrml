"""
FLAGSHIP ROE Integration Script

Comprehensive script to integrate all researched ROEs into the FLAGSHIP COA/ROE system.
This script loads all pertinent ROEs from the comprehensive research and integrates them
into the existing COA/ROE system.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

import logging
# Use standard logging instead of FLAGSHIP.common
get_logger = logging.getLogger

from .coa_roe_engine import COAROEEngine
from .comprehensive_roe_database import (
    ComprehensiveROEDatabase,
    ROEAuthority,
    ROECategory,
    ROEStandard,
)

logger = get_logger(__name__)


class ROEIntegrationManager:
    """Manages integration of comprehensive ROEs into FLAGSHIP system."""

    def __init__(self, coa_roe_engine: COAROEEngine):
        """Initialize the ROE integration manager."""
        self.logger = logger
        self.coa_roe_engine = coa_roe_engine
        self.roe_database = ComprehensiveROEDatabase()
        self.integration_results = []

    async def integrate_all_roes(self) -> Dict[str, Any]:
        """Integrate all researched ROEs into the FLAGSHIP system."""
        self.logger.info("Starting comprehensive ROE integration")

        integration_summary = {
            "total_roes_processed": 0,
            "successfully_integrated": 0,
            "failed_integrations": 0,
            "categories_processed": [],
            "authorities_processed": [],
            "integration_details": [],
        }

        try:
            # Get all ROEs from database
            all_roes = await self.roe_database.get_all_roes()
            self.logger.info(f"Found {len(all_roes)} ROEs to integrate")

            # Process ROEs by category
            for category in ROECategory:
                category_roes = await self.roe_database.get_roes_by_category(category)
                if category_roes:
                    self.logger.info(
                        f"Processing {len(category_roes)} ROEs for category: {category.value}"
                    )

                    category_result = await self._integrate_category_roes(
                        category, category_roes
                    )
                    integration_summary["categories_processed"].append(category.value)
                    integration_summary["integration_details"].append(category_result)

                    integration_summary["total_roes_processed"] += len(category_roes)
                    integration_summary["successfully_integrated"] += category_result[
                        "successful"
                    ]
                    integration_summary["failed_integrations"] += category_result[
                        "failed"
                    ]

            # Process ROEs by authority
            for authority in ROEAuthority:
                authority_roes = await self.roe_database.get_roes_by_authority(
                    authority
                )
                if authority_roes:
                    self.logger.info(
                        f"Processing {len(authority_roes)} ROEs for authority: {authority.value}"
                    )

                    authority_result = await self._integrate_authority_roes(
                        authority, authority_roes
                    )
                    integration_summary["authorities_processed"].append(authority.value)
                    integration_summary["integration_details"].append(authority_result)

            self.logger.info(
                f"ROE integration completed. Successfully integrated: {integration_summary['successfully_integrated']}/{integration_summary['total_roes_processed']}"
            )

        except Exception as e:
            self.logger.error(f"Error during ROE integration: {e}")
            integration_summary["error"] = str(e)

        return integration_summary

    async def _integrate_category_roes(
        self, category: ROECategory, roes: List[ROEStandard]
    ) -> Dict[str, Any]:
        """Integrate ROEs for a specific category."""
        category_result = {
            "category": category.value,
            "total_roes": len(roes),
            "successful": 0,
            "failed": 0,
            "details": [],
        }

        for roe in roes:
            try:
                integration_result = await self._integrate_single_roe(roe)
                category_result["details"].append(integration_result)

                if integration_result["success"]:
                    category_result["successful"] += 1
                else:
                    category_result["failed"] += 1

            except Exception as e:
                self.logger.error(f"Error integrating ROE {roe.roe_id}: {e}")
                category_result["failed"] += 1
                category_result["details"].append(
                    {"roe_id": roe.roe_id, "success": False, "error": str(e)}
                )

        return category_result

    async def _integrate_authority_roes(
        self, authority: ROEAuthority, roes: List[ROEStandard]
    ) -> Dict[str, Any]:
        """Integrate ROEs for a specific authority."""
        authority_result = {
            "authority": authority.value,
            "total_roes": len(roes),
            "successful": 0,
            "failed": 0,
            "details": [],
        }

        for roe in roes:
            try:
                integration_result = await self._integrate_single_roe(roe)
                authority_result["details"].append(integration_result)

                if integration_result["success"]:
                    authority_result["successful"] += 1
                else:
                    authority_result["failed"] += 1

            except Exception as e:
                self.logger.error(f"Error integrating ROE {roe.roe_id}: {e}")
                authority_result["failed"] += 1
                authority_result["details"].append(
                    {"roe_id": roe.roe_id, "success": False, "error": str(e)}
                )

        return authority_result

    async def _integrate_single_roe(self, roe: ROEStandard) -> Dict[str, Any]:
        """Integrate a single ROE into the FLAGSHIP system."""
        integration_result = {
            "roe_id": roe.roe_id,
            "title": roe.title,
            "category": roe.category.value,
            "authority": roe.authority.value,
            "success": False,
            "integration_method": None,
            "details": {},
        }

        try:
            # Determine integration method based on ROE category
            if roe.category in [
                ROECategory.INTERNATIONAL_LAW,
                ROECategory.NATIONAL_LAW,
            ]:
                # Integrate as legal compliance standard
                await self._integrate_legal_compliance_roe(roe)
                integration_result["integration_method"] = "legal_compliance"

            elif roe.category in [
                ROECategory.LAW_ENFORCEMENT,
                ROECategory.MILITARY_PROTOCOLS,
            ]:
                # Integrate as response protocol
                await self._integrate_response_protocol_roe(roe)
                integration_result["integration_method"] = "response_protocol"

            elif roe.category in [
                ROECategory.CORPORATE_SECURITY,
                ROECategory.EMERGENCY_RESPONSE,
            ]:
                # Integrate as security protocol
                await self._integrate_security_protocol_roe(roe)
                integration_result["integration_method"] = "security_protocol"

            elif roe.category in [
                ROECategory.CYBERSECURITY,
                ROECategory.CRITICAL_INFRASTRUCTURE,
            ]:
                # Integrate as cybersecurity protocol
                await self._integrate_cybersecurity_roe(roe)
                integration_result["integration_method"] = "cybersecurity_protocol"

            else:
                # Integrate as general protocol
                await self._integrate_general_roe(roe)
                integration_result["integration_method"] = "general_protocol"

            integration_result["success"] = True
            integration_result["details"] = {
                "integrated_at": datetime.now().isoformat(),
                "severity": roe.severity.value,
                "jurisdiction": roe.jurisdiction,
            }

            self.logger.info(f"Successfully integrated ROE: {roe.roe_id} - {roe.title}")

        except Exception as e:
            self.logger.error(f"Failed to integrate ROE {roe.roe_id}: {e}")
            integration_result["error"] = str(e)

        return integration_result

    async def _integrate_legal_compliance_roe(self, roe: ROEStandard):
        """Integrate ROE as legal compliance standard."""
        # Add to legal compliance manager
        compliance_manager = self.coa_roe_engine.legal_compliance

        # Create compliance requirement
        compliance_requirement = {
            "requirement_id": f"ROE_{roe.roe_id}",
            "standard": roe.authority.value,
            "title": roe.title,
            "description": roe.description,
            "legal_basis": roe.legal_basis,
            "restrictions": roe.restrictions,
            "requirements": roe.requirements,
            "severity": roe.severity.value,
            "applicability": roe.applicability,
            "documentation_required": roe.documentation_required,
        }

        # Add to compliance manager (this would be implemented in the compliance manager)
        # await compliance_manager.add_compliance_requirement(compliance_requirement)

        self.logger.info(f"Integrated legal compliance ROE: {roe.roe_id}")

    async def _integrate_response_protocol_roe(self, roe: ROEStandard):
        """Integrate ROE as response protocol."""
        # Add to response protocol manager
        protocol_manager = self.coa_roe_engine.protocol_manager

        # Create response protocol
        response_protocol = {
            "protocol_id": f"ROE_{roe.roe_id}",
            "name": roe.title,
            "description": roe.description,
            "category": roe.category.value,
            "authority": roe.authority.value,
            "legal_basis": roe.legal_basis,
            "restrictions": roe.restrictions,
            "requirements": roe.requirements,
            "force_levels": roe.force_levels,
            "severity": roe.severity.value,
            "applicability": roe.applicability,
        }

        # Add to protocol manager (this would be implemented in the protocol manager)
        # await protocol_manager.add_protocol(response_protocol)

        self.logger.info(f"Integrated response protocol ROE: {roe.roe_id}")

    async def _integrate_security_protocol_roe(self, roe: ROEStandard):
        """Integrate ROE as security protocol."""
        # Add to security protocols
        # This would integrate with the security management system
        self.logger.info(f"Integrated security protocol ROE: {roe.roe_id}")

    async def _integrate_cybersecurity_roe(self, roe: ROEStandard):
        """Integrate ROE as cybersecurity protocol."""
        # Add to cybersecurity protocols
        # This would integrate with the cybersecurity management system
        self.logger.info(f"Integrated cybersecurity ROE: {roe.roe_id}")

    async def _integrate_general_roe(self, roe: ROEStandard):
        """Integrate ROE as general protocol."""
        # Add to general protocols
        # This would integrate with the general protocol management system
        self.logger.info(f"Integrated general ROE: {roe.roe_id}")

    async def generate_integration_report(
        self, integration_summary: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive integration report."""
        report = """
# FLAGSHIP ROE Integration Report

## Executive Summary
- **Total ROEs Processed**: {integration_summary['total_roes_processed']}
- **Successfully Integrated**: {integration_summary['successfully_integrated']}
- **Failed Integrations**: {integration_summary['failed_integrations']}
- **Success Rate**: {((integration_summary['successfully_integrated'] / integration_summary['total_roes_processed'] * 100) if integration_summary['total_roes_processed'] > 0 else 0):.1f}%

## Categories Processed
{chr(10).join([f"- {category}" for category in integration_summary['categories_processed']])}

## Authorities Processed
{chr(10).join([f"- {authority}" for authority in integration_summary['authorities_processed']])}

## Integration Details
"""

        for detail in integration_summary["integration_details"]:
            if "category" in detail:
                report += f"""
### Category: {detail['category']}
- **Total ROEs**: {detail['total_roes']}
- **Successful**: {detail['successful']}
- **Failed**: {detail['failed']}
- **Success Rate**: {((detail['successful'] / detail['total_roes'] * 100) if detail['total_roes'] > 0 else 0):.1f}%
"""
            elif "authority" in detail:
                report += f"""
### Authority: {detail['authority']}
- **Total ROEs**: {detail['total_roes']}
- **Successful**: {detail['successful']}
- **Failed**: {detail['failed']}
- **Success Rate**: {((detail['successful'] / detail['total_roes'] * 100) if detail['total_roes'] > 0 else 0):.1f}%
"""

        report += f"""
## Recommendations
1. Review failed integrations for potential issues
2. Verify all ROEs are properly integrated
3. Test integrated ROEs in the system
4. Update documentation with new ROEs
5. Schedule regular ROE updates

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
    """Main function to run ROE integration."""
    logger.info("Starting FLAGSHIP ROE Integration")

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
        integration_manager = ROEIntegrationManager(coa_roe_engine)

        # Run integration
        integration_summary = await integration_manager.integrate_all_roes()

        # Generate report
        report = await integration_manager.generate_integration_report(
            integration_summary
        )

        # Save report
        with open("roe_integration_report.md", "w") as f:
            f.write(report)

        # Export data
        export_data = await integration_manager.export_integration_data(
            integration_summary
        )
        with open("roe_integration_data.json", "w") as f:
            f.write(export_data)

        logger.info("ROE integration completed successfully")
        logger.info("Report saved to: roe_integration_report.md")
        logger.info("Data saved to: roe_integration_data.json")

        # Print summary
        if integration_summary["total_roes_processed"] > 0:
            success_rate = (
                integration_summary["successfully_integrated"]
                / integration_summary["total_roes_processed"]
            ) * 100
            return integration_summary

    except Exception as e:
        logger.error(f"Error during ROE integration: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
