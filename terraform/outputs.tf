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

output "edge_instances" {
  description = "Edge instance details"
  value = {
    for id, instance in aws_instance.edge : id => {
      public_ip     = instance.public_ip
      private_ip    = instance.private_ip
      high_patients = var.edge_configs[index(var.edge_configs[*].id, id)].high_patients
      low_patients  = var.edge_configs[index(var.edge_configs[*].id, id)].low_patients
    }
  }
}

# SSH Commands
output "ssh_commands" {
  description = "SSH commands for each instance"
  value = {
    cloud = "ssh -i ~/.ssh/vispac ubuntu@${aws_instance.cloud.public_ip}"
    fog   = "ssh -i ~/.ssh/vispac ubuntu@${aws_instance.fog.public_ip}"
    edges = {
      for id, instance in aws_instance.edge :
      id => "ssh -i ~/.ssh/vispac ubuntu@${instance.public_ip}"
    }
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
    
    Edge Instances:
    %{for cfg in var.edge_configs~}
      - ${cfg.id}: ${cfg.high_patients} high + ${cfg.low_patients} low patients
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
    Cloud Services:
      - Cloud API:  ${aws_instance.cloud.private_ip}:9000
      - PostgreSQL: ${aws_instance.cloud.private_ip}:5432

    ========================================

  EOT
}
