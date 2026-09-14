# AdjustIQ Fine-Tuning

Fine-tunes a 7B open-weight LLM (default: Mistral-7B) with **QLoRA**
(4-bit quantization + LoRA adapters) to triage insurance customer-service
messages into `{category, priority, draft reply}`. Training data is
synthetically generated — no external dataset or API key required.
Includes Terraform to provision the GPU box the training runs on.

```
adjustiq-fine-tuning/
├── infra/
│   ├── providers.tf                   # AWS provider + backend
│   ├── variables.tf                   # all configurable inputs
│   ├── main.tf                        # VPC lookup, SG, IAM, EC2 GPU instance(s)
│   ├── user_data.sh.tpl               # first-boot setup script
│   ├── outputs.tf                     # SSH/SSM connection info
│   └── terraform.tfvars.example       # copy to terraform.tfvars and fill in
├── data/
│   └── insurance_tickets.jsonl        # generated dataset (created by you)
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train_qlora.py
│   ├── infer_qlora.py
│   └── eval_harness.py
├── outputs/                           # adapter checkpoints land here
└── requirements.txt
```

## Step 1 — Provision the GPU box with Terraform

Prereqs: Terraform ≥1.5, an AWS account with EC2/IAM/S3 permissions, and
the AWS CLI configured (`aws configure`).

**1a. Create an S3 bucket for artifacts** (adapters, datasets) if you
don't already have one:

```bash
aws s3 mb s3://adjustiq-fine-tuning-artifacts --region us-east-1
```

**1b. Create an EC2 key pair** for SSH access:

```bash
cd infra
aws ec2 create-key-pair --key-name adjustiq-key \
  --query 'KeyMaterial' --output text > adjustiq-key.pem
chmod 400 adjustiq-key.pem
```

**1c. Configure your variables:**

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:
- `key_pair_name` — the key pair you just created
- `allowed_ssh_cidr` — your public IP + `/32` (check with `curl ifconfig.me`)
- `artifacts_bucket_name` — the bucket from step 1a
- `github_repo_url` — this repo's URL, once pushed to your GitHub
- `instance_type` — `g5.2xlarge` (1x A10G, 24GB VRAM) is enough for QLoRA on a 7B model; bump to `g5.12xlarge` (4x A10G) only if you want multi-GPU
- `use_spot` — leave `true` unless you need guaranteed availability; checkpoints save every 50 steps so a Spot interruption loses little

**1d. Provision:**

```bash
terraform init
terraform plan
terraform apply
```

`terraform output` gives you the public IP and a ready-to-run SSH command.
The instance boots with the Deep Learning AMI (drivers + CUDA
pre-installed), clones this repo, and installs the Python dependencies
automatically via `user_data.sh.tpl` — by the time SSH is up, it's ready
to train.

**1e. When you're done, tear it down** to stop paying for the GPU:

```bash
terraform destroy
```

(Push your trained adapter to S3 — see Step 7 — *before* destroying, or you'll lose it.)

## Step 2 — SSH in and activate the environment

```bash
ssh -i adjustiq-key.pem ubuntu@<public-ip>   # from terraform output
source /opt/pytorch/bin/activate
cd adjustiq-fine-tuning/scripts
```

## Step 3 — Authenticate with Hugging Face

Mistral/Llama weights are gated. Visit the model page while logged in
(e.g. `huggingface.co/mistralai/Mistral-7B-v0.1`) and click **Agree and
access repository**. If you didn't set `hf_token` in `terraform.tfvars`,
log in manually:

```bash
hf auth login
```

## Step 4 — Generate the synthetic insurance dataset

```bash
python generate_synthetic_data.py --n 800 --output ../data/insurance_tickets.jsonl --seed 42
```

This produces 800 unique, templated customer messages spanning 11
insurance categories (claims status/denial, billing disputes, premium
increases, cancellations, coverage questions, policy changes,
escalations, fraud reports, health-claim delays), each paired with a
structured triage + draft-reply response. Bump `--n` to 2000+ for a
stronger fine-tune; it runs in seconds since it's pure templating, no
API calls.

Inspect a few rows:

```bash
head -n 3 ../data/insurance_tickets.jsonl
```

## Step 5 — Run a smoke test (small, fast, cheap)

Confirm the pipeline runs end-to-end before committing real compute:

```bash
python train_qlora.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --dataset_path ../data/insurance_tickets.jsonl \
  --output_dir ../outputs/smoke-test \
  --epochs 1 --eval_split 0.2 --batch_size 1 --grad_accum 4
```

Watch for:
- `print_trainable_parameters()` showing a small fraction of total params (LoRA is working)
- Train/eval loss printing every 10 steps without OOM errors

If you hit OOM: drop `--batch_size` to 1, raise `--grad_accum`, or drop `--max_seq_len` to 256.

## Step 6 — Run the full training job

```bash
python train_qlora.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --dataset_path ../data/insurance_tickets.jsonl \
  --output_dir ../outputs/adjustiq-lora \
  --epochs 3 --eval_split 0.1 --batch_size 2 --grad_accum 8 --lora_r 16
```

The adapter (a few MB, not the full model) saves to
`../outputs/adjustiq-lora/final_adapter`.

## Step 7 — Push the adapter to S3

Do this before running `terraform destroy`:

```bash
aws s3 cp ../outputs/adjustiq-lora/final_adapter \
  s3://adjustiq-fine-tuning-artifacts/adjustiq-lora/final_adapter --recursive
```

The instance's IAM role already has read/write access to this bucket
(set up in `infra/main.tf`), so no extra credentials needed.

## Step 8 — Try it on a new message

```bash
python infer_qlora.py \
  --base_model mistralai/Mistral-7B-v0.1 \
  --adapter_dir ../outputs/adjustiq-lora/final_adapter \
  --prompt "Hi, this is Maria Garcia. My claim CLM-4471203 for storm damage has been stuck in review for 3 weeks."
```

## Step 9 — Evaluate base vs. fine-tuned

```bash
python eval_harness.py \
  --base_model mistralai/Mistral-7B-v0.1 \
  --adapter_dir ../outputs/adjustiq-lora/final_adapter \
  --dataset_path ../data/insurance_tickets.jsonl \
  --n_samples 15
```

This loads the model once, toggles the adapter on/off per example
(`model.disable_adapter()`), and checks whether each response follows
the `Triage: ... | Priority: ... | Draft reply: ...` structure. The
base model — never trained on this format — will mostly fail to
produce it; the fine-tuned adapter should hit it consistently. This
checks *format compliance*, not draft-reply quality — read a handful
of full generations manually to judge tone and triage accuracy.

## Tuning knobs worth trying next

| Change | Effect |
|---|---|
| `--target_modules q_proj v_proj k_proj o_proj` | More adapter capacity, slightly slower |
| `--lora_r 8` vs `32` | Smaller/larger adapter, quality vs. size tradeoff |
| `--n 2000+` in step 4 | More data diversity, usually the highest-leverage change |
| `instance_count = 2+` in `terraform.tfvars` | Multi-node setup for distributed training (requires code changes to `train_qlora.py` for `accelerate`/DDP — not wired up by default) |
| Merge adapter (`merge_and_unload()`, commented at bottom of `train_qlora.py`) | Standalone deployable model instead of base+adapter |

## Deployment tie-in

Once trained, load the adapter directly from S3 in a SageMaker
real-time endpoint (`PeftModel.from_pretrained` at container startup) —
one base model, swap adapters per line of business (auto/home/health)
without redeploying the base weights. Good story for a multi-tenant or
multi-vertical serving setup.

## Cost note

`g5.2xlarge` on Spot runs roughly $0.30-0.45/hr depending on region (vs.
~$1.00-1.20/hr on-demand); a full training run at the settings above
typically finishes in under an hour on 800-2000 examples. Always run
`terraform destroy` when you're done for the day — the instance is not
free while idle.
