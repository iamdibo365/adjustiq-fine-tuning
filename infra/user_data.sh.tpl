#!/bin/bash
set -eux

# This DLAMI generation (Deep Learning OSS Nvidia Driver AMI) uses a
# plain venv at /opt/pytorch rather than conda -- activate that instead.
source /opt/pytorch/bin/activate

pip install -U transformers peft bitsandbytes trl datasets accelerate nvidia-ml-py

# Clone the repo so scripts + data generator are ready to go.
cd /home/ubuntu
git clone ${github_repo_url} adjustiq-fine-tuning || true
chown -R ubuntu:ubuntu adjustiq-fine-tuning

%{ if hf_token != "" ~}
sudo -u ubuntu bash -c "source /opt/pytorch/bin/activate && hf auth login --token ${hf_token}"
%{ endif ~}

echo "AdjustIQ GPU box ready. SSH in, run: source /opt/pytorch/bin/activate && cd adjustiq-fine-tuning/scripts" > /etc/motd
