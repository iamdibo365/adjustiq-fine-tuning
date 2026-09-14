data "aws_vpc" "selected" {
  id      = var.vpc_id != "" ? var.vpc_id : null
  default = var.vpc_id == "" ? true : null
}

# g5 (and other GPU families) aren't available in every AZ within a
# region -- this finds which AZs actually have capacity for the
# instance type you asked for, so we don't land in an unsupported one
# (e.g. us-east-1e frequently lacks g5 capacity).
data "aws_ec2_instance_type_offerings" "gpu" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
  location_type = "availability-zone"
}

data "aws_subnets" "selected" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.selected.id]
  }

  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.gpu.locations
  }
}

locals {
  subnet_id = var.subnet_id != "" ? var.subnet_id : data.aws_subnets.selected.ids[0]
}

# AWS Deep Learning AMI (GPU PyTorch) -- ships with NVIDIA drivers, CUDA,
# and conda envs pre-installed so we don't have to fight driver setup.
data "aws_ssm_parameter" "dlami" {
  # AWS publishes the latest DLAMI ID per framework/OS combo as an SSM
  # public parameter -- this avoids hardcoding an AMI name that AWS
  # periodically renames (it was "Deep Learning AMI GPU PyTorch ..." and
  # is now "Deep Learning OSS Nvidia Driver AMI GPU PyTorch ..."; the
  # SSM parameter path below is stable across those renames).
  name = "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id"
}

# --------------------------------------------------------------------------
# Security group: SSH only from the CIDR you specify. No inbound app ports
# needed -- this is a training box, not a service.
# --------------------------------------------------------------------------
resource "aws_security_group" "gpu_sg" {
  name        = "${var.project_name}-gpu-sg"
  description = "SSH access for ${var.project_name} GPU training instance(s)"
  vpc_id      = data.aws_vpc.selected.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-gpu-sg"
    Project = var.project_name
  }
}

# --------------------------------------------------------------------------
# IAM: least-privilege role scoped to one S3 bucket for adapters/datasets,
# plus SSM so you can connect without opening SSH if you'd rather use
# `aws ssm start-session` instead of a key pair.
# --------------------------------------------------------------------------
resource "aws_iam_role" "gpu_role" {
  name = "${var.project_name}-gpu-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_role_policy" "s3_artifacts" {
  name = "${var.project_name}-s3-artifacts"
  role = aws_iam_role.gpu_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
      ]
      Resource = [
        "arn:aws:s3:::${var.artifacts_bucket_name}",
        "arn:aws:s3:::${var.artifacts_bucket_name}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.gpu_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "gpu_profile" {
  name = "${var.project_name}-gpu-profile"
  role = aws_iam_role.gpu_role.name
}

# --------------------------------------------------------------------------
# GPU instance(s). count=1 by default (single-GPU QLoRA). Raise
# instance_count only for multi-node distributed training.
# --------------------------------------------------------------------------
resource "aws_instance" "gpu" {
  count = var.instance_count

  ami                    = data.aws_ssm_parameter.dlami.value
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  subnet_id              = local.subnet_id
  vpc_security_group_ids = [aws_security_group.gpu_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu_profile.name

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type            = "gp3"
    delete_on_termination  = true
    encrypted               = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    github_repo_url = var.github_repo_url
    hf_token         = var.hf_token
  })

  tags = {
    Name    = "${var.project_name}-gpu-${count.index}"
    Project = var.project_name
  }
}
