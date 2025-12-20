# =============================================================================
# VISPAC AWS Infrastructure
# Edge-Fog-Cloud Architecture for Vital Signs Processing
# =============================================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# =============================================================================
# VPC Configuration
# =============================================================================

resource "aws_vpc" "vispac" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-vpc"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.vispac.id

  tags = {
    Name    = "${var.project_name}-igw"
    Project = var.project_name
  }
}

# Private subnet for Edge and Fog (same network)
resource "aws_subnet" "private" {
  vpc_id                  = aws_vpc.vispac.id
  cidr_block              = var.private_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true # For initial setup, can be changed later

  tags = {
    Name    = "${var.project_name}-private-subnet"
    Project = var.project_name
    Tier    = "edge-fog"
  }
}

# Public subnet for Cloud (separate network)
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.vispac.id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.project_name}-public-subnet"
    Project = var.project_name
    Tier    = "cloud"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.vispac.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name    = "${var.project_name}-public-rt"
    Project = var.project_name
  }
}

resource "aws_route_table_association" "private" {
  subnet_id      = aws_subnet.private.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# =============================================================================
# Security Groups
# =============================================================================

# Security Group for Edge instances
resource "aws_security_group" "edge" {
  name        = "${var.project_name}-edge-sg"
  description = "Security group for Edge instances"
  vpc_id      = aws_vpc.vispac.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-edge-sg"
    Project = var.project_name
  }
}

# Security Group for Fog instance
resource "aws_security_group" "fog" {
  name        = "${var.project_name}-fog-sg"
  description = "Security group for Fog instance"
  vpc_id      = aws_vpc.vispac.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # MQTT from Edge instances
  ingress {
    from_port       = 1883
    to_port         = 1883
    protocol        = "tcp"
    security_groups = [aws_security_group.edge.id]
  }

  # HTTP API from Edge instances (alternative to MQTT)
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.edge.id]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-fog-sg"
    Project = var.project_name
  }
}

# Security Group for Cloud instance
resource "aws_security_group" "cloud" {
  name        = "${var.project_name}-cloud-sg"
  description = "Security group for Cloud instance"
  vpc_id      = aws_vpc.vispac.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Cloud API from Fog
  ingress {
    from_port       = 9000
    to_port         = 9000
    protocol        = "tcp"
    security_groups = [aws_security_group.fog.id]
  }

  # PostgreSQL from Cloud itself
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    self        = true
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-cloud-sg"
    Project = var.project_name
  }
}

# =============================================================================
# SSH Key Pair
# =============================================================================

resource "aws_key_pair" "vispac" {
  key_name   = "${var.project_name}-key"
  public_key = var.ssh_public_key

  tags = {
    Name    = "${var.project_name}-key"
    Project = var.project_name
  }
}

# =============================================================================
# EC2 Instances - Edge (simulated IoT devices with resource limits)
# =============================================================================

resource "aws_instance" "edge" {
  count = var.edge_count

  ami           = data.aws_ami.ubuntu.id
  instance_type = var.edge_instance_type
  subnet_id     = aws_subnet.private.id
  key_name      = aws_key_pair.vispac.key_name

  vpc_security_group_ids = [aws_security_group.edge.id]

  # CPU credits for t3.micro - standard mode limits burst
  credit_specification {
    cpu_credits = "standard"
  }

  root_block_device {
    volume_size = 8
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/scripts/edge_setup.sh", {
    edge_id        = "edge-${format("%02d", count.index + 1)}"
    patient_range  = local.patient_ranges[count.index]
    mqtt_broker    = aws_instance.fog.private_ip
    dataset_type   = var.dataset_type
    git_repo       = var.git_repo_url
    git_branch     = var.git_branch
    memory_limit   = var.edge_memory_limit_mb
    cpu_limit      = var.edge_cpu_limit_percent
  })

  tags = {
    Name         = "${var.project_name}-edge-${format("%02d", count.index + 1)}"
    Project      = var.project_name
    Role         = "edge"
    PatientRange = local.patient_ranges[count.index]
  }

  depends_on = [aws_instance.fog]
}

# =============================================================================
# EC2 Instance - Fog
# =============================================================================

resource "aws_instance" "fog" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.fog_instance_type
  subnet_id     = aws_subnet.private.id
  key_name      = aws_key_pair.vispac.key_name

  vpc_security_group_ids = [aws_security_group.fog.id]

  root_block_device {
    volume_size = 16
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/scripts/fog_setup.sh", {
    cloud_url    = "http://${aws_instance.cloud.private_ip}:9000"
    git_repo     = var.git_repo_url
    git_branch   = var.git_branch
  })

  tags = {
    Name    = "${var.project_name}-fog"
    Project = var.project_name
    Role    = "fog"
  }

  depends_on = [aws_instance.cloud]
}

# =============================================================================
# EC2 Instance - Cloud
# =============================================================================

resource "aws_instance" "cloud" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.cloud_instance_type
  subnet_id     = aws_subnet.public.id
  key_name      = aws_key_pair.vispac.key_name

  vpc_security_group_ids = [aws_security_group.cloud.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/scripts/cloud_setup.sh", {
    db_password  = var.db_password
    git_repo     = var.git_repo_url
    git_branch   = var.git_branch
  })

  tags = {
    Name    = "${var.project_name}-cloud"
    Project = var.project_name
    Role    = "cloud"
  }
}

# =============================================================================
# Local Values - Patient Range Distribution
# =============================================================================

locals {
  # Calculate patient ranges for each edge
  # Total 53 patients in high_risk dataset
  total_patients     = var.total_patients
  patients_per_edge  = ceil(local.total_patients / var.edge_count)
  
  patient_ranges = [
    for i in range(var.edge_count) : 
    "${(i * local.patients_per_edge) + 1}-${min((i + 1) * local.patients_per_edge, local.total_patients)}"
  ]
}
