variable "region" {
  type    = string
  default = "us-east-2"
}

variable "instance_type" {
  type    = string
  default = "t3.medium"  # 2 vCPUs, 4GB RAM
}

variable "ssh_key_name" {
  type    = string
  default = "mlops-ec2-key"
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/mlops_ec2.pub"
}

variable "allowed_ssh_cidr" {
  type    = string
  default = "0.0.0.0/0" # Overridden in tfvars for better security
}

variable "ecr_repo_name" {
  type    = string
  default = "mlops-comment"
}

variable "artifact_bucket_name" {
  type    = string
  default = "mlops-comment-artifacts"
}