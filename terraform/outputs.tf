# =============================================================================
# Outputs for VISPAC Infrastructure
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.vispac.id
}

output "cloud_public_ip" {
  description = "Public IP of Cloud instance"
  value       = aws_instance.cloud.public_ip
}

output "cloud_private_ip" {
  description = "Private IP of Cloud instance"
  value       = aws_instance.cloud.private_ip
}

output "fog_public_ip" {
  description = "Public IP of Fog instance"
  value       = aws_instance.fog.public_ip
}

output "fog_private_ip" {
  description = "Private IP of Fog instance"
  value       = aws_instance.fog.private_ip
}

output "edge_public_ips" {
  description = "Public IPs of Edge instances"
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

output "ssh_commands" {
  description = "SSH commands for each instance"
  value = {
    cloud = "ssh -i ~/.ssh/vispac ubuntu@${aws_instance.cloud.public_ip}"
    fog   = "ssh -i ~/.ssh/vispac ubuntu@${aws_instance.fog.public_ip}"
    edges = [for i, ip in aws_instance.edge[*].public_ip : 
             "ssh -i ~/.ssh/vispac ubuntu@${ip}  # edge-${format("%02d", i + 1)} (patients ${local.patient_ranges[i]})"]
  }
}

output "architecture_summary" {
  description = "Architecture summary"
  value = <<-EOT
    
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    VISPAC Architecture Deployed                   ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  Edge Instances (${var.edge_count}x ${var.edge_instance_type}):                              ║
    ║    ${join("\n    ║    ", [for i in range(var.edge_count) : "edge-${format("%02d", i + 1)}: patients ${local.patient_ranges[i]}"])}
    ║                                                                   ║
    ║  Fog Instance (${var.fog_instance_type}):                                       ║
    ║    MQTT Broker: ${aws_instance.fog.private_ip}:1883                     ║
    ║    NEWS2 API:   ${aws_instance.fog.private_ip}:8000                     ║
    ║                                                                   ║
    ║  Cloud Instance (${var.cloud_instance_type}):                                     ║
    ║    Cloud API:   ${aws_instance.cloud.private_ip}:9000                   ║
    ║    PostgreSQL:  ${aws_instance.cloud.private_ip}:5432                   ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    
  EOT
}
