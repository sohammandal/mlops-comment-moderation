#!/bin/bash
set -eux

# Install prerequisites
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's GPG key and repo
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker and Compose plugin
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin git curl

# Enable Docker for the ubuntu user
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker

# Prepare app directory
mkdir -p /opt/app
chown -R ubuntu:ubuntu /opt/app

# Clone repo, create .env, run Docker Compose, then prune build cache
sudo -u ubuntu bash -c '
  cd /opt/app
  git clone https://github.com/sohammandal/mlops-comment-moderation.git .
  # Create placeholder .env so docker compose does not fail
  echo "# Auto-generated placeholder .env" > .env
  echo "DEBUG=true" >> .env
  docker compose -f docker/docker-compose.yml up -d --build
  # Free disk space by removing unused images/layers
  docker builder prune -af
'