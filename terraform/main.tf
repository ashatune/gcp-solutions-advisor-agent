terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

variable "project_id" {
  description = "Target Google Cloud Project ID for deploying the GCP Solutions Advisor Agent."
  type        = string
}

variable "region" {
  description = "Primary GCP region for Cloud Run deployment."
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "Artifact Registry container image URI for the advisor agent service."
  type        = string
  default     = "us-central1-docker.pkg.dev/example-project/cloud-run-source-deploy/gcp-solutions-advisor-agent:latest"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Least-Privilege Dedicated Service Account for the Agent Runtime
resource "google_service_account" "advisor_agent_sa" {
  account_id   = "gcp-solutions-advisor-sa"
  display_name = "GCP Solutions Advisor ADK Agent Runtime Service Account"
  description  = "Least-privilege identity for Cloud Run, Secret Manager, Vertex AI, and Cloud Logging."
}

# 2. Google Cloud Secret Manager Resources for API Keys & Datastore Config
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"

  replication {
    auto {}
  }

  labels = {
    app       = "gcp-solutions-advisor-agent"
    managedBy = "terraform"
  }
}

resource "google_secret_manager_secret" "search_engine_id" {
  secret_id = "gcp-docs-search-engine-id"

  replication {
    auto {}
  }

  labels = {
    app       = "gcp-solutions-advisor-agent"
    managedBy = "terraform"
  }
}

# 3. Least-Privilege IAM Bindings (Secret Manager Accessor, Vertex AI User, Log Writer, Trace Agent)
resource "google_secret_manager_secret_iam_member" "gemini_key_accessor" {
  secret_id = google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.advisor_agent_sa.email}"
}

resource "google_secret_manager_secret_iam_member" "search_engine_accessor" {
  secret_id = google_secret_manager_secret.search_engine_id.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.advisor_agent_sa.email}"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.advisor_agent_sa.email}"
}

resource "google_project_iam_member" "cloud_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.advisor_agent_sa.email}"
}

resource "google_project_iam_member" "cloud_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.advisor_agent_sa.email}"
}

# 4. Cloud Logging Metric for Guardrail & HITL Audit Observability
resource "google_logging_metric" "guardrail_block_metric" {
  name        = "gcp_advisor_guardrail_blocks"
  description = "Counts candidate responses blocked by GroundingPolicyEvaluator"
  filter      = "resource.type=\"cloud_run_revision\" AND jsonPayload.component=\"guardrails.policy\" AND severity>=WARNING"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

# 5. Cloud Run v2 Service Hosting the GCP Solutions Advisor Agent
resource "google_cloud_run_v2_service" "advisor_agent_service" {
  name     = "gcp-solutions-advisor-agent"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.advisor_agent_sa.email

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GCP_LOCATION"
        value = var.region
      }

      env {
        name  = "FAST_MODEL_NAME"
        value = "gemini-2.5-flash"
      }

      env {
        name  = "PRO_MODEL_NAME"
        value = "gemini-2.5-pro"
      }

      env {
        name  = "ENABLE_LIVE_SECRET_MANAGER"
        value = "true"
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "GCP_SEARCH_ENGINE_ID"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.search_engine_id.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.gemini_key_accessor,
    google_secret_manager_secret_iam_member.search_engine_accessor,
  ]
}

output "cloud_run_service_uri" {
  description = "Deployed URL of the GCP Solutions Advisor Agent Cloud Run service."
  value       = google_cloud_run_v2_service.advisor_agent_service.uri
}
