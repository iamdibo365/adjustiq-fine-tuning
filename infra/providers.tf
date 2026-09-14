terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Optional: uncomment and configure for remote state instead of local
  # backend "s3" {
  #   bucket = "your-tfstate-bucket"
  #   key    = "adjustiq-fine-tuning/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}
