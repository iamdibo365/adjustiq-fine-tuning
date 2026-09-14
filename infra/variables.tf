variable "aws_region" {
  description = "AWS region to provision the GPU instance(s) in"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Used to tag/name all resources"
  type        = string
  default     = "adjustiq-fine-tuning"
}

variable "instance_type" {
  description = "GPU instance type. g5.2xlarge (1x A10G, 24GB VRAM) comfortably runs QLoRA on a 7B model. Use g5.12xlarge (4x A10G) or p4d.24xlarge (8x A100) for multi-GPU."
  type        = string
  default     = "g5.2xlarge"
}

variable "instance_count" {
  description = "Number of GPU instances to provision. 1 is enough for QLoRA on a single 7B model; raise this only if you're doing multi-node distributed training."
  type        = number
  default     = 1
}

variable "use_spot" {
  description = "Provision as EC2 Spot instances (60-70% cheaper, can be interrupted). Recommended for this workload since checkpoints save every 50 steps."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Max hourly price per instance in USD for Spot requests. Leave empty to default to on-demand price as the cap."
  type        = string
  default     = ""
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GB. Model weights + checkpoints + HF cache need headroom -- 200GB is comfortable for a 7B model."
  type        = number
  default     = 200
}

variable "key_pair_name" {
  description = "Name of an EXISTING EC2 key pair in this region, used for SSH access. Create one first: aws ec2 create-key-pair --key-name adjustiq-key --query 'KeyMaterial' --output text > adjustiq-key.pem && chmod 400 adjustiq-key.pem"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into the instance(s). Set to your own IP/32, not 0.0.0.0/0."
  type        = string
}

variable "vpc_id" {
  description = "VPC to launch into. Leave empty to use the account's default VPC."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet to launch into. Leave empty to use the default subnet in the first AZ of the chosen VPC."
  type        = string
  default     = ""
}

variable "artifacts_bucket_name" {
  description = "S3 bucket the instance's IAM role gets read/write access to, for pushing trained LoRA adapters and datasets. Bucket must already exist or be created separately."
  type        = string
}

variable "github_repo_url" {
  description = "Repo to clone on first boot so the training scripts are ready to run."
  type        = string
  default     = "https://github.com/<your-username>/adjustiq-fine-tuning.git"
}

variable "hf_token" {
  description = "Hugging Face access token, used at boot to pre-authenticate the Hugging Face CLI for gated models (Mistral/Llama). Leave empty to log in manually over SSH instead."
  type        = string
  default     = ""
  sensitive   = true
}
