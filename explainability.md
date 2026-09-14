# Explainability — AdjustIQ Fine-Tuning

This document walks through every file in this repo and the exact order
things execute in, from `terraform apply` to a trained adapter answering
a prompt. Read it top to bottom the first time; after that, use it as a
reference for what a specific file/function does.

---

## Execution order, end to end

```
1. infra/*.tf            terraform apply provisions the GPU box
2. infra/user_data.sh.tpl runs once, automatically, on first boot
3. scripts/generate_synthetic_data.py   you run this first, by hand
4. scripts/train_qlora.py               then this
5. scripts/infer_qlora.py               then this, as many times as you want
6. scripts/eval_harness.py              and/or this
```

Everything below follows that order.

---

## 1. Terraform (`infra/`) — provisioning the GPU box

Terraform resources don't run "in file order" — Terraform builds a
dependency graph and creates resources in the order the graph requires.
Here's the actual order it resolves to for this config:

### 1a. `providers.tf`
Declares the `aws` provider and pins the Terraform/provider versions.
This is read first, before anything else, so Terraform knows which API
to talk to and which region (`var.aws_region`, default `us-east-1`).

### 1b. `variables.tf`
Pure declarations — no resources are created here. Every `var.xxx`
referenced elsewhere in `main.tf` is defined in this file with a type,
description, and (usually) a default. Terraform loads these before
evaluating anything else so it can substitute values. Your
`terraform.tfvars` (copied from `terraform.tfvars.example`) overrides
the defaults here.

### 1c. `main.tf` — resolution order
Terraform figures out dependencies automatically from references
(e.g. a security group referencing a VPC ID means the VPC lookup
happens first). The effective order is:

1. **`data "aws_vpc" "selected"`** — looks up your default VPC (or the
   one you specified in `var.vpc_id`). This is a *read*, not a create.
2. **`data "aws_ec2_instance_type_offerings" "gpu"`** — asks AWS which
   Availability Zones in this region actually have capacity for
   `var.instance_type`. GPU families like g5 aren't available in every
   AZ, and picking an unsupported one (e.g. `us-east-1e` often lacks g5
   capacity) fails at `apply` time with a confusing error otherwise.
3. **`data "aws_subnets" "selected"`** — lists subnets inside that VPC,
   filtered down to only the AZs the previous lookup confirmed support
   your instance type; `locals.subnet_id` picks the first one unless you
   overrode it.
4. **`data "aws_ssm_parameter" "dlami"`** — reads AWS's officially
   published SSM parameter for "the latest Deep Learning AMI, GPU
   PyTorch 2.7, Ubuntu 22.04." AWS maintains this parameter path
   specifically so infrastructure code doesn't have to hardcode AMI
   names (which AWS periodically renames — the family used to be called
   "Deep Learning AMI GPU PyTorch..." and is now "Deep Learning OSS
   Nvidia Driver AMI GPU PyTorch..."). This AMI already has NVIDIA
   drivers, CUDA, and a pre-built Python environment baked in, which is
   why the boot script (section 2) doesn't need to install drivers
   itself.
5. **`aws_security_group.gpu_sg`** — one inbound rule (SSH, port 22,
   restricted to `var.allowed_ssh_cidr`), one outbound rule (allow all).
   No other inbound ports are opened — this box only serves your SSH
   session, not any application traffic.
6. **`aws_iam_role.gpu_role`** — the role the EC2 instance assumes.
   The `assume_role_policy` says "only the EC2 service can assume this
   role," which is what lets the *instance itself* (not a human) use
   the permissions below.
7. **`aws_iam_role_policy.s3_artifacts`** — an inline policy scoped to
   exactly one bucket (`var.artifacts_bucket_name`): `GetObject`,
   `PutObject`, `ListBucket`. This is what lets Step 7 in the README
   (`aws s3 cp` from inside the instance) work without you configuring
   separate AWS credentials on the box.
8. **`aws_iam_role_policy_attachment.ssm`** — attaches AWS's managed
   `AmazonSSMManagedInstanceCore` policy, enabling `aws ssm
   start-session` as an alternative to SSH (useful if you'd rather not
   open port 22 at all — you'd remove the ingress rule in that case).
9. **`aws_iam_instance_profile.gpu_profile`** — EC2 instances can't
   assume an IAM role directly; they need an instance *profile* wrapping
   the role. This is that wrapper, and it's what actually gets attached
   to the instance in step 9.
10. **`aws_instance.gpu`** (the `count = var.instance_count` resource) —
   created last because it references everything above (AMI, security
   group, instance profile, subnet). Key parts:
   - `root_block_device` — a `gp3` volume (default 200GB, encrypted)
     sized for model weights + checkpoints + the Hugging Face cache.
   - `dynamic "instance_market_options"` — only added if
     `var.use_spot = true`; this is what makes it a Spot request instead
     of on-demand. The `for_each = var.use_spot ? [1] : []` trick is a
     standard Terraform pattern for "conditionally include this block."
   - `user_data = templatefile(...)` — renders `user_data.sh.tpl` with
     `github_repo_url` and `hf_token` substituted in, and hands the
     result to EC2 as the cloud-init script (see section 2).

### 1d. `outputs.tf`
Evaluated after all resources exist. Prints instance IDs, public IPs,
and pre-built SSH/SSM commands so you don't have to hand-assemble them
from `terraform show`.

### 1e. `terraform.tfvars.example`
Not executed at all — it's a template you copy to `terraform.tfvars`
(gitignored) and fill in with real values (your key pair name, your IP,
your bucket name). Terraform automatically loads `terraform.tfvars` if
present.

---

## 2. `infra/user_data.sh.tpl` — first boot

EC2 runs this once, automatically, as root, the first time the instance
boots — you don't invoke it yourself. By the time you SSH in, it's
already finished (usually 1-3 minutes after `terraform apply` reports
success). Line by line:

1. `source /opt/pytorch/bin/activate` — this DLAMI generation (the
   "Deep Learning OSS Nvidia Driver AMI" line) ships a plain Python venv
   at `/opt/pytorch` with PyTorch + CUDA already wired together
   correctly, rather than a conda environment like older DLAMI releases
   used. This activates it.
2. `pip install -U transformers peft bitsandbytes trl datasets accelerate nvidia-ml-py`
   — the libraries this project needs on top of the base AMI.
   `nvidia-ml-py` replaces the deprecated `pynvml` that `torch` warns
   about otherwise.
3. `git clone ${github_repo_url} adjustiq-fine-tuning` — pulls this repo
   onto the box. The `|| true` means a second boot (if you ever
   re-trigger user_data) won't fail just because the directory already
   exists.
4. `chown -R ubuntu:ubuntu adjustiq-fine-tuning` — the clone happens as
   root; this hands ownership to the `ubuntu` user you'll actually SSH
   in as.
5. The `%{ if hf_token != "" ~}` block — Terraform's templating syntax
   for a conditional. If you set `hf_token` in `terraform.tfvars`, this
   runs `hf auth login` (the current Hugging Face CLI — it replaced the
   deprecated `huggingface-cli`) non-interactively as the `ubuntu` user
   so gated models (Mistral, Llama) are usable immediately. If you left
   it blank, you log in manually over SSH instead (README Step 3).
6. Writes a one-line reminder to `/etc/motd` so it's the first thing you
   see when you SSH in.

---

## 3. `scripts/generate_synthetic_data.py` — you run this first

No model, no GPU needed for this one — pure Python string templating.
Structure:

- **Entity pools** (`FIRST_NAMES`, `LAST_NAMES`, `CITIES`, `VEHICLES`,
  `CONDITIONS`) and **randomizers** (`rand_name`, `rand_policy_number`,
  `rand_claim_number`, `rand_amount`, `rand_date`) — these get mixed
  into each generated ticket so no two examples are identical even
  though they come from the same template.
- **11 `gen_*` functions** (`gen_claim_status`, `gen_claim_denial`,
  `gen_billing_dispute`, etc.) — each one builds one insurance scenario:
  it picks random entities, formats a customer complaint string, and
  returns `(instruction, category, priority, reply_body)`. Priority is
  usually rule-based inside the function (e.g. `gen_claim_status` marks
  anything over 20 days as `High`).
- **`GENERATORS`** — a list of all 11 functions. `build_example()` calls
  `random.choice(GENERATORS)` to pick one, then assembles the final
  training pair:
  ```
  {"instruction": <customer message>,
   "response": "Triage: <category> | Priority: <priority> | Draft reply: <reply_body>"}
  ```
- **`main()`** — parses `--n`, `--output`, `--seed`; loops calling
  `build_example()` until it has `n` *unique* instructions (a `seen` set
  prevents duplicates), writes one JSON object per line to the output
  path, then prints a category breakdown so you can eyeball the class
  balance.

**Output**: `data/insurance_tickets.jsonl` — this is the only artifact
this step produces, and it's the direct input to step 4.

---

## 4. `scripts/train_qlora.py` — the actual fine-tuning

This is where the GPU gets used. Execution order inside `main()`:

1. **Argument parsing** — `--model_id`, `--dataset_path`, `--output_dir`,
   plus every LoRA/training hyperparameter (`--lora_r`, `--epochs`,
   `--batch_size`, etc.) with sensible defaults tuned for a 7B model on
   a single 24GB GPU.
2. **`load_dataset("json", data_files=...)`** — loads the JSONL from
   step 3 into a Hugging Face `Dataset`.
3. **`dataset.map(format_example, ...)`** — `format_example()` takes
   each `{instruction, response}` row and renders it through
   `PROMPT_TEMPLATE` ("### Customer Message: ... ### Triage Response:
   ...") into a single `text` field. This is the exact string the model
   will be trained to predict.
4. **`train_test_split`** — carves off `--eval_split` (default 10%) as
   held-out data so you get a genuine eval loss during training, not
   just train loss.
5. **`load_model_and_tokenizer(args.model_id)`**:
   - **`build_bnb_config()`** — the actual "Q" in QLoRA: loads the base
     model in 4-bit NF4 quantization with double quantization and
     bfloat16 compute dtype. This is what lets a 7B model fit in ~5-6GB
     instead of ~28GB.
   - Tokenizer is loaded; if there's no pad token (common for Mistral),
     it's set to the eos token.
   - `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, device_map="auto")`
     — downloads and loads the base weights, quantized, spread
     automatically across available GPU memory.
   - `model.config.use_cache = False` — required because gradient
     checkpointing (enabled later) and KV-caching are incompatible
     during training.
   - **`prepare_model_for_kbit_training(model)`** — a PEFT helper that
     casts norm layers to fp32 for stability, enables gradient
     checkpointing, and preps the quantized model to accept gradients
     at all (quantized weights are frozen and non-differentiable by
     default).
6. **`build_lora_config(...)`** — defines the LoRA adapter: rank `r`
   (default 16), scaling `alpha` (default 32), `dropout` (0.05), and
   which weight matrices get adapters injected (`target_modules`,
   default `["q_proj", "v_proj"]` — the attention query/value
   projections, the highest-leverage layers to adapt cheaply).
7. **`get_peft_model(model, lora_config)`** — this is the step that
   actually inserts small trainable LoRA matrices alongside the frozen
   base weights. `model.print_trainable_parameters()` right after
   confirms only a small fraction of total parameters (the adapter) are
   trainable — everything else stays frozen.
8. **`SFTConfig(...)`** — training hyperparameters: cosine LR schedule
   with warmup, `paged_adamw_8bit` optimizer (memory-efficient, designed
   for quantized training), eval/save every 50 steps, `bf16=True`,
   `packing=False` (each example trains independently rather than being
   concatenated with others).
9. **`SFTTrainer(...)`** (from `trl`) — wraps the model, args, and both
   dataset splits into a standard Hugging Face-style trainer purpose
   built for supervised fine-tuning on a text field.
10. **`trainer.train()`** — the actual training loop runs here. This is
    the long-running step; progress prints every `logging_steps`.
11. **`trainer.model.save_pretrained(adapter_dir)`** +
    `tokenizer.save_pretrained(adapter_dir)` — saves *only* the LoRA
    adapter weights (a few MB) plus tokenizer config to
    `outputs/adjustiq-lora/final_adapter/`. The frozen base model is
    never re-saved — that's the point of LoRA.
12. The commented-out `merge_and_unload()` line, if uncommented, would
    fuse the adapter into the base weights and save a full standalone
    model instead of base+adapter — useful for deployment simplicity at
    the cost of losing the "swap adapters on one base model" flexibility.

**Output**: `outputs/adjustiq-lora/final_adapter/` — this directory is
what steps 5 and 6 (and README Step 7's S3 push) consume.

---

## 5. `scripts/infer_qlora.py` — try it on a new message

Run manually, as many times as you like, after training finishes.

1. Parses `--base_model`, `--adapter_dir`, `--prompt`.
2. Loads the tokenizer **from the adapter directory** (not the base
   model repo) — this matters because the adapter dir has the exact
   tokenizer state used during training.
3. Loads the base model in 4-bit (same `BitsAndBytesConfig` pattern as
   training).
4. **`PeftModel.from_pretrained(base_model, args.adapter_dir)`** — this
   is the key line: it takes the frozen base model and re-attaches the
   trained LoRA adapter on top of it, reconstructing the fine-tuned
   behavior without ever having saved a full fine-tuned model.
5. Formats your `--prompt` through the same `PROMPT_TEMPLATE` used in
   training (customer message in, triage response section left open for
   the model to fill).
6. `model.generate(...)` with sampling (`temperature=0.7, top_p=0.9`) —
   decodes and prints the model's completion.

---

## 6. `scripts/eval_harness.py` — base vs. fine-tuned, side by side

Run manually, after training, to sanity-check that fine-tuning actually
changed behavior.

1. Loads `--n_samples` random rows from the dataset (held-out by virtue
   of just being a random sample, not a strict eval split — good enough
   for a quick structural check).
2. Loads the base model **once**, then wraps it in a single `PeftModel`
   (`tuned_model`) — this is more efficient than loading two separate
   models.
3. For each sampled ticket:
   - **`with tuned_model.disable_adapter():`** — a PEFT context manager
     that temporarily turns the LoRA adapter off, so a `generate()` call
     inside this block behaves exactly like the untouched base model,
     without needing a second model in memory.
   - Outside that block, `generate()` runs normally with the adapter
     active — the fine-tuned behavior.
4. **`STRUCT_RE`** — a regex checking for the literal
   `Triage: ... | Priority: ... | Draft reply:` structure. Since the
   base model was never trained on this format, it should rarely
   produce it; the adapter, having seen thousands of examples in that
   exact shape, should hit it consistently.
5. Prints a running per-example comparison, then a final tally:
   `Base model: X/N correctly structured` vs. `Tuned model: Y/N`.
6. A closing note reminds you this only checks *format compliance* —
   whether the triage category and draft reply are actually *correct*
   for the ticket needs a manual read of a few full generations.

---

## Quick mental model

- **Terraform** exists once, spins up compute, tears down when you're
  done — it doesn't know anything about insurance or LoRA.
- **`generate_synthetic_data.py`** exists to solve "I have no insurance
  dataset" — it's a data problem, solved with plain Python, zero ML.
- **`train_qlora.py`** is the only file that touches a GPU seriously —
  everything before it is prep, everything after it consumes its output.
- **`infer_qlora.py`** and **`eval_harness.py`** are two views of the
  same idea (load base + reattach adapter) — one for a single prompt,
  one for a batch comparison against the untouched base model.
