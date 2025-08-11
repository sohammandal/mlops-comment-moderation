#!/bin/bash
set -eux

# Install prerequisites
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release awscli

# Add Docker repo
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

# Install Docker and Compose plugin
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin git

# Enable Docker for ubuntu user
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker

# Prepare app directory
mkdir -p /opt/app
chown -R ubuntu:ubuntu /opt/app

# Clone repo and set .env with ECR url (substituted by Terraform)
sudo -u ubuntu bash -c "
  cd /opt/app
  git clone https://github.com/sohammandal/mlops-comment-moderation.git .
  echo \"ECR_REPOSITORY_URL=$${ecr_url}\" >> .env
"

# Login to ECR using instance role
REGISTRY="$(echo "$${ecr_url}" | cut -d/ -f1)"
aws ecr get-login-password --region $${AWS_DEFAULT_REGION:-us-east-2} \
  | docker login --username AWS --password-stdin "$REGISTRY"

# Pull image with retries
cd /opt/app
for i in {1..20}; do
  if docker compose -f docker/docker-compose.yml pull; then
    break
  fi
  echo "Image not available yet - retrying in 30s..."
  sleep 30
done

# Start containers
docker compose -f docker/docker-compose.yml up -d
docker image prune -af
echo "Container status:" && docker ps
