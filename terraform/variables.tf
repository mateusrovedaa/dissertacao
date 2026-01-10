# =============================================================================
# Variables for VISPAC Infrastructure
# =============================================================================

# -----------------------------------------------------------------------------
# AWS Configuration
# -----------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for Edge and Fog resources"
  type        = string
  default     = "us-east-1"
}

variable "cloud_region" {
  description = "AWS region for Cloud resources (separate from Edge/Fog)"
  type        = string
  default     = "us-west-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "vispac"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# -----------------------------------------------------------------------------
# Network Configuration
# -----------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for Edge/Fog VPC (us-east-1)"
  type        = string
  default     = "10.0.0.0/16"
}

variable "edge_subnet_cidr" {
  description = "CIDR block for Edge subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "fog_subnet_cidr" {
  description = "CIDR block for Fog subnet (separate from Edge)"
  type        = string
  default     = "10.0.3.0/24"
}

variable "cloud_vpc_cidr" {
  description = "CIDR block for Cloud VPC (us-west-1)"
  type        = string
  default     = "10.1.0.0/16"
}

variable "cloud_subnet_cidr" {
  description = "CIDR block for Cloud subnet"
  type        = string
  default     = "10.1.1.0/24"
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Restrict this in production!
}

# -----------------------------------------------------------------------------
# SSH Configuration
# -----------------------------------------------------------------------------

variable "ssh_public_key" {
  description = "SSH public key for EC2 access"
  type        = string
}

# -----------------------------------------------------------------------------
# Instance Configuration
# -----------------------------------------------------------------------------

variable "edge_configs" {
  description = "Configuration for each edge device with dataset selection"
  type = list(object({
    id                = string
    high_patients     = number
    low_patients      = number
    specific_patients = optional(string)
  }))

  validation {
    condition     = length(var.edge_configs) > 0
    error_message = "At least one edge must be configured in edge_configs"
  }

  validation {
    condition     = alltrue([for e in var.edge_configs : e.high_patients >= 0 && e.low_patients >= 0])
    error_message = "Patient counts must be non-negative"
  }

  validation {
    condition     = alltrue([for e in var.edge_configs : (e.high_patients + e.low_patients > 0) || (e.specific_patients != null)])
    error_message = "Each edge must have at least 1 patient (high_patients + low_patients > 0) or defined specific_patients"
  }
}

variable "edge_instance_type" {
  description = "EC2 instance type for Edge (simulating constrained IoT device)"
  type        = string
  default     = "t3.micro" # 2 vCPU, 1GB RAM - simulates edge device
}

variable "fog_instance_type" {
  description = "EC2 instance type for Fog"
  type        = string
  default     = "t3.small" # 2 vCPU, 2GB RAM
}

variable "cloud_instance_type" {
  description = "EC2 instance type for Cloud"
  type        = string
  default     = "t3.small" # 2 vCPU, 2GB RAM
}

# -----------------------------------------------------------------------------
# Edge Resource Limits (simulating constrained devices)
# -----------------------------------------------------------------------------

variable "edge_memory_limit_mb" {
  description = "Memory limit for edge process in MB (cgroup limit)"
  type        = number
  default     = 256 # 256MB - simulates constrained device
}

variable "edge_cpu_limit_percent" {
  description = "CPU limit for edge process as percentage (cgroup limit)"
  type        = number
  default     = 50 # 50% of 1 vCPU
}

# -----------------------------------------------------------------------------
# Application Configuration
# -----------------------------------------------------------------------------

variable "git_repo_url" {
  description = "Git repository URL for VISPAC code"
  type        = string
  default     = "https://github.com/YOUR_USER/dissertacao.git"
}

variable "git_branch" {
  description = "Git branch to deploy"
  type        = string
  default     = "main"
}

variable "scenario" {
  description = "Test scenario: scenario1_baseline (raw), scenario2_static (compression only), scenario3_vispac (full)"
  type        = string
  default     = "scenario3_vispac"

  validation {
    condition     = contains(["scenario1_baseline", "scenario2_static", "scenario3_vispac"], var.scenario)
    error_message = "Scenario must be one of: scenario1_baseline, scenario2_static, scenario3_vispac"
  }
}

variable "experiment_duration_hours" {
  description = "Duration of experiment in hours. Services stop and logs are archived after this time."
  type        = number
  default     = 8

  validation {
    condition     = var.experiment_duration_hours > 0 && var.experiment_duration_hours <= 168
    error_message = "Experiment duration must be between 1 and 168 hours (1 week max)"
  }
}

variable "db_password" {
  description = "PostgreSQL password for cloud database"
  type        = string
  sensitive   = true
}
