output "instance_ids" {
  description = "EC2 instance IDs"
  value       = aws_instance.gpu[*].id
}

output "public_ips" {
  description = "Public IPs of the GPU instance(s)"
  value       = aws_instance.gpu[*].public_ip
}

output "ssh_commands" {
  description = "Ready-to-run SSH commands (swap in your actual .pem path)"
  value       = [for ip in aws_instance.gpu[*].public_ip : "ssh -i /path/to/${var.key_pair_name}.pem ubuntu@${ip}"]
}

output "ssm_session_commands" {
  description = "Alternative: connect via SSM (no open SSH port needed)"
  value       = [for id in aws_instance.gpu[*].id : "aws ssm start-session --target ${id}"]
}
