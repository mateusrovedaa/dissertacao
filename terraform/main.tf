# =============================================================================
# VISPAC AWS Infrastructure
# Edge-Fog-Cloud Architecture for Vital Signs Processing
# Multi-Region: Edge+Fog in us-east-1, Cloud in us-west-1
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

# =============================================================================
# Providers - Multi-Region
# =============================================================================

provider "aws" {
  region = var.aws_region
}

provider "aws" {
  alias  = "cloud"
  region = var.cloud_region
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_availability_zones" "cloud" {
  provider = aws.cloud
  state    = "available"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_ami" "ubuntu_cloud" {
  provider    = aws.cloud
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# =============================================================================
# VPC Configuration - Edge/Fog (us-east-1)
# =============================================================================

resource "aws_vpc" "edge_fog" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-edge-fog-vpc"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "edge_fog" {
  vpc_id = aws_vpc.edge_fog.id

  tags = {
    Name    = "${var.project_name}-edge-fog-igw"
    Project = var.project_name
  }
}

# Edge subnet
resource "aws_subnet" "edge" {
  vpc_id                  = aws_vpc.edge_fog.id
  cidr_block              = var.edge_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.project_name}-edge-subnet"
    Project = var.project_name
    Tier    = "edge"
  }
}

# Fog subnet (separate from Edge)
resource "aws_subnet" "fog" {
  vpc_id                  = aws_vpc.edge_fog.id
  cidr_block              = var.fog_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.project_name}-fog-subnet"
    Project = var.project_name
    Tier    = "fog"
  }
}

resource "aws_route_table" "edge_fog" {
  vpc_id = aws_vpc.edge_fog.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.edge_fog.id
  }

  # Route to Cloud VPC via peering
  route {
    cidr_block                = var.cloud_vpc_cidr
    vpc_peering_connection_id = aws_vpc_peering_connection.edge_fog_to_cloud.id
  }

  tags = {
    Name    = "${var.project_name}-edge-fog-rt"
    Project = var.project_name
  }
}

resource "aws_route_table_association" "edge" {
  subnet_id      = aws_subnet.edge.id
  route_table_id = aws_route_table.edge_fog.id
}

resource "aws_route_table_association" "fog" {
  subnet_id      = aws_subnet.fog.id
  route_table_id = aws_route_table.edge_fog.id
}

# =============================================================================
# VPC Configuration - Cloud (us-west-1)
# =============================================================================

resource "aws_vpc" "cloud" {
  provider             = aws.cloud
  cidr_block           = var.cloud_vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "${var.project_name}-cloud-vpc"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "cloud" {
  provider = aws.cloud
  vpc_id   = aws_vpc.cloud.id

  tags = {
    Name    = "${var.project_name}-cloud-igw"
    Project = var.project_name
  }
}

resource "aws_subnet" "cloud" {
  provider                = aws.cloud
  vpc_id                  = aws_vpc.cloud.id
  cidr_block              = var.cloud_subnet_cidr
  availability_zone       = data.aws_availability_zones.cloud.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.project_name}-cloud-subnet"
    Project = var.project_name
    Tier    = "cloud"
  }
}

resource "aws_route_table" "cloud" {
  provider = aws.cloud
  vpc_id   = aws_vpc.cloud.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.cloud.id
  }

  # Route to Edge/Fog VPC via peering
  route {
    cidr_block                = var.vpc_cidr
    vpc_peering_connection_id = aws_vpc_peering_connection.edge_fog_to_cloud.id
  }

  tags = {
    Name    = "${var.project_name}-cloud-rt"
    Project = var.project_name
  }
}

resource "aws_route_table_association" "cloud" {
  provider       = aws.cloud
  subnet_id      = aws_subnet.cloud.id
  route_table_id = aws_route_table.cloud.id
}

# =============================================================================
# VPC Peering - Cross-Region
# =============================================================================

resource "aws_vpc_peering_connection" "edge_fog_to_cloud" {
  vpc_id      = aws_vpc.edge_fog.id
  peer_vpc_id = aws_vpc.cloud.id
  peer_region = var.cloud_region
  auto_accept = false

  tags = {
    Name    = "${var.project_name}-peering-edge-fog-to-cloud"
    Project = var.project_name
  }
}

resource "aws_vpc_peering_connection_accepter" "cloud" {
  provider                  = aws.cloud
  vpc_peering_connection_id = aws_vpc_peering_connection.edge_fog_to_cloud.id
  auto_accept               = true

  tags = {
    Name    = "${var.project_name}-peering-accepter"
    Project = var.project_name
  }
}

# =============================================================================
# Security Groups - Edge/Fog (us-east-1)
# =============================================================================

resource "aws_security_group" "edge" {
  name        = "${var.project_name}-edge-sg"
  description = "Security group for Edge instances"
  vpc_id      = aws_vpc.edge_fog.id

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

resource "aws_security_group" "fog" {
  name        = "${var.project_name}-fog-sg"
  description = "Security group for Fog instance"
  vpc_id      = aws_vpc.edge_fog.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # MQTT from Edge subnet
  ingress {
    from_port   = 1883
    to_port     = 1883
    protocol    = "tcp"
    cidr_blocks = [var.edge_subnet_cidr]
  }

  # HTTP API from Edge subnet
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.edge_subnet_cidr]
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

# =============================================================================
# Security Groups - Cloud (us-west-1)
# =============================================================================

resource "aws_security_group" "cloud" {
  provider    = aws.cloud
  name        = "${var.project_name}-cloud-sg"
  description = "Security group for Cloud instance"
  vpc_id      = aws_vpc.cloud.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Cloud API from Fog subnet (via VPC peering)
  ingress {
    from_port   = 9000
    to_port     = 9000
    protocol    = "tcp"
    cidr_blocks = [var.fog_subnet_cidr]
  }

  # PostgreSQL (localhost only via security group self)
  ingress {
    from_port = 5432
    to_port   = 5432
    protocol  = "tcp"
    self      = true
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
# SSH Key Pairs
# =============================================================================

resource "aws_key_pair" "edge_fog" {
  key_name   = "${var.project_name}-key"
  public_key = var.ssh_public_key

  tags = {
    Name    = "${var.project_name}-key"
    Project = var.project_name
  }
}

resource "aws_key_pair" "cloud" {
  provider   = aws.cloud
  key_name   = "${var.project_name}-key"
  public_key = var.ssh_public_key

  tags = {
    Name    = "${var.project_name}-key"
    Project = var.project_name
  }
}

# =============================================================================
# EC2 Instances - Edge (us-east-1)
# =============================================================================

resource "aws_instance" "edge" {
  count = var.edge_count

  ami           = data.aws_ami.ubuntu.id
  instance_type = var.edge_instance_type
  subnet_id     = aws_subnet.edge.id
  key_name      = aws_key_pair.edge_fog.key_name

  vpc_security_group_ids = [aws_security_group.edge.id]

  credit_specification {
    cpu_credits = "standard"
  }

  root_block_device {
    volume_size = 8
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/scripts/edge_setup.sh", {
    edge_id      = "edge-${format("%02d", count.index + 1)}"
    patient_range = local.patient_ranges[count.index]
    mqtt_broker  = aws_instance.fog.private_ip
    dataset_type = var.dataset_type
    git_repo     = var.git_repo_url
    git_branch   = var.git_branch
    memory_limit = var.edge_memory_limit_mb
    cpu_limit    = var.edge_cpu_limit_percent
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
# EC2 Instance - Fog (us-east-1)
# =============================================================================

resource "aws_instance" "fog" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.fog_instance_type
  subnet_id     = aws_subnet.fog.id
  key_name      = aws_key_pair.edge_fog.key_name

  vpc_security_group_ids = [aws_security_group.fog.id]

  root_block_device {
    volume_size = 16
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/scripts/fog_setup.sh", {
    cloud_url  = "http://${aws_instance.cloud.private_ip}:9000"
    git_repo   = var.git_repo_url
    git_branch = var.git_branch
  })

  tags = {
    Name    = "${var.project_name}-fog"
    Project = var.project_name
    Role    = "fog"
  }

  depends_on = [aws_instance.cloud, aws_vpc_peering_connection_accepter.cloud]
}

# =============================================================================
# EC2 Instance - Cloud (us-west-1)
# =============================================================================

resource "aws_instance" "cloud" {
  provider      = aws.cloud
  ami           = data.aws_ami.ubuntu_cloud.id
  instance_type = var.cloud_instance_type
  subnet_id     = aws_subnet.cloud.id
  key_name      = aws_key_pair.cloud.key_name

  vpc_security_group_ids = [aws_security_group.cloud.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/scripts/cloud_setup.sh", {
    db_password = var.db_password
    git_repo    = var.git_repo_url
    git_branch  = var.git_branch
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
  total_patients    = var.total_patients
  patients_per_edge = ceil(local.total_patients / var.edge_count)

  patient_ranges = [
    for i in range(var.edge_count) :
    "${(i * local.patients_per_edge) + 1}-${min((i + 1) * local.patients_per_edge, local.total_patients)}"
  ]
}
