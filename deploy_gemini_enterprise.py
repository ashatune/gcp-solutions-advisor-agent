"""Deploy or update GCP Solutions Advisor Agent on Vertex AI Agent Engine and Gemini Enterprise."""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

import requests
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.coordinator import GCPSolutionsAdkApp, create_advisor_coordinator

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
PROJECT_NUMBER = os.getenv("GCP_PROJECT_NUMBER", "000000000000")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
STAGING_BUCKET = f"gs://{PROJECT_ID}-adk-staging"
GEMINI_ENTERPRISE_ENGINE_ID = os.getenv("GEMINI_ENTERPRISE_ENGINE_ID", "your-gemini-enterprise-engine-id")
EXISTING_REASONING_ENGINE_NAME = os.getenv("EXISTING_REASONING_ENGINE_NAME", "")

AGENT_ENGINE_REQUIREMENTS = [
    "google-cloud-aiplatform[adk,agent_engines]>=2.6.0",
    "google-adk==2.12.0",
    "google-genai>=1.0.0",
    "google-cloud-secret-manager>=2.20.0",
    "google-cloud-logging>=3.10.0",
    "pydantic==2.14.1",
    "python-dotenv>=1.0.1",
    "opentelemetry-api>=1.25.0",
    "opentelemetry-sdk>=1.25.0",
    "cloudpickle==3.1.2",
]


def get_access_token() -> str:
    """Retrieve an OAuth2 access token from gcloud CLI."""
    out = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True)
    return out.strip()


def deploy_or_update_agent_engine(
    existing_resource_name: Optional[str] = EXISTING_REASONING_ENGINE_NAME,
) -> str:
    """Update the existing Vertex AI Agent Engine in-place (or create a new one if none exists)."""
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["FAST_MODEL_NAME"] = "gemini-3.5-flash"
    os.environ["PRO_MODEL_NAME"] = "gemini-3.1-pro-preview"

    print(f"[1/2] Initializing Vertex AI ({PROJECT_ID} / {LOCATION}, staging={STAGING_BUCKET})...")
    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    adk_app = GCPSolutionsAdkApp(
        agent=create_advisor_coordinator(),
        enable_tracing=True,
        env_vars={
            "GOOGLE_CLOUD_LOCATION": "global",
            "FAST_MODEL_NAME": "gemini-3.5-flash",
            "PRO_MODEL_NAME": "gemini-3.1-pro-preview",
        },
    )

    if existing_resource_name:
        print(f"[1/2] Updating existing Reasoning Engine in-place: {existing_resource_name}...")
        existing_engine = agent_engines.get(existing_resource_name)
        existing_engine.update(
            agent_engine=adk_app,
            requirements=AGENT_ENGINE_REQUIREMENTS,
            extra_packages=["./agent"],
        )
        print(f"[1/2] Updated Reasoning Engine: {existing_engine.resource_name}")
        return existing_engine.resource_name

    print("[1/2] Creating new Vertex AI Agent Engine...")
    remote_agent = agent_engines.create(
        agent_engine=adk_app,
        display_name="GCP Solutions Advisor Agent",
        description=(
            "Senior Google Cloud Principal Solutions Architect agent providing grounded GCP "
            "documentation research and deep architectural trade-off analysis."
        ),
        requirements=AGENT_ENGINE_REQUIREMENTS,
        extra_packages=["./agent"],
    )
    print(f"[1/2] Created Reasoning Engine: {remote_agent.resource_name}")
    return remote_agent.resource_name


if __name__ == "__main__":
    re_name = deploy_or_update_agent_engine()
    print(f"\n=== Reasoning Engine Ready: {re_name} ===")
