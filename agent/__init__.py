"""GCP Solutions Advisor Agent package built on Google Agent Development Kit (ADK)."""

from agent.constitution import AGENT_CONSTITUTION, SYSTEM_INSTRUCTION
from agent.coordinator import GCPSolutionsCoordinator, create_advisor_coordinator

root_agent = create_advisor_coordinator()

__all__ = [
    "AGENT_CONSTITUTION",
    "SYSTEM_INSTRUCTION",
    "GCPSolutionsCoordinator",
    "create_advisor_coordinator",
    "root_agent",
]
