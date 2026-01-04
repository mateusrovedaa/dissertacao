# =============================================================================
# Outputs for VISPAC Infrastructure (Multi-Region)
# =============================================================================

# VPC Outputs
output "edge_fog_vpc_id" {
  description = "Edge/Fog VPC ID (us-east-1)"
  value       = aws_vpc.edge_fog.id
}

output "cloud_vpc_id" {
  description = "Cloud VPC ID (us-west-1)"
  value       = aws_vpc.cloud.id
}

output "vpc_peering_id" {
  description = "VPC Peering Connection ID"
  value       = aws_vpc_peering_connection.edge_fog_to_cloud.id
}

# Instance IPs
output "cloud_public_ip" {
  description = "Public IP of Cloud instance (us-west-1)"
  value       = aws_instance.cloud.public_ip
}

output "cloud_private_ip" {
  description = "Private IP of Cloud instance"
  value       = aws_instance.cloud.private_ip
}

output "fog_public_ip" {
  description = "Public IP of Fog instance (us-east-1)"
  value       = aws_instance.fog.public_ip
}

output "fog_private_ip" {
  description = "Private IP of Fog instance"
  value       = aws_instance.fog.private_ip
}

output "edge_public_ips" {
  description = "Public IPs of Edge instances (us-east-1)"
  value       = aws_instance.edge[*].public_ip
}

output "edge_private_ips" {
  description = "Private IPs of Edge instances"
  value       = aws_instance.edge[*].private_ip
}

output "edge_patient_ranges" {
  description = "Patient ranges assigned to each edge"
  value       = local.patient_ranges
}

# SSH Commands
output "ssh_commands" {
  description = "SSH commands for each instance"
  value = {
    cloud = "ssh -i ~/.ssh/vispac ubuntu@${aws_instance.cloud.public_ip}"
    fog   = "ssh -i ~/.ssh/vispac ubuntu@${aws_instance.fog.public_ip}"
    edges = [for i, ip in aws_instance.edge[*].public_ip :
    "ssh -i ~/.ssh/vispac ubuntu@${ip}  # edge-${format("%02d", i + 1)} (patients ${local.patient_ranges[i]})"]
  }
}

# Architecture Summary
output "architecture_summary" {
  description = "Multi-region architecture summary"
  value       = <<-EOT

    ========================================
    VISPAC Multi-Region Architecture
    ========================================

    REGION: ${var.aws_region} (Edge + Fog)
    ----------------------------------------
    | Layer | Subnet        | IP              |
    |-------|---------------|-----------------|
    | Edge  | ${var.edge_subnet_cidr} | ${join(", ", aws_instance.edge[*].private_ip)} |
    | Fog   | ${var.fog_subnet_cidr} | ${aws_instance.fog.private_ip}     |

    Edge Instances:
    %{for i in range(var.edge_count)~}
      - edge-${format("%02d", i + 1)}: patients ${local.patient_ranges[i]} (${aws_instance.edge[i].private_ip})
    %{endfor~}

    Fog Services:
      - MQTT Broker: ${aws_instance.fog.private_ip}:1883
      - NEWS2 API:   ${aws_instance.fog.private_ip}:8000

                    |
               VPC PEERING
                    |
                    v

    REGION: ${var.cloud_region} (Cloud)
    ----------------------------------------
    | Layer | Subnet        | IP              |
    |-------|---------------|-----------------|
    | Cloud | ${var.cloud_subnet_cidr} | ${aws_instance.cloud.private_ip}    |

    Cloud Services:
      - Cloud API:  ${aws_instance.cloud.private_ip}:9000
      - PostgreSQL: ${aws_instance.cloud.private_ip}:5432

    ========================================

  EOT
}
