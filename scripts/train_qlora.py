"""
QLoRA Fine-Tuning -- Insurance Support Ticket Triage
======================================================
Fine-tunes a causal LM (default: Mistral-7B) with 4-bit quantization +
LoRA adapters to triage insurance customer messages and draft a reply.

Install:
    pip install -U transformers peft bitsandbytes trl datasets accelerate

Run (after generating data with generate_synthetic_data.py):
    python train_qlora.py \
        --model_id mistralai/Mistral-7B-v0.1 \
        --dataset_path ../data/insurance_tickets.jsonl \
        --output_dir ../outputs/adjustiq-lora
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


PROMPT_TEMPLATE = """### Customer Message:
{instruction}

### Triage Response:
{response}"""


def build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization -- the 'Q' in QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_lora_config(target_modules, r, alpha, dropout) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(model_id: str):
    bnb_config = build_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def format_example(example: dict) -> dict:
    text = PROMPT_TEMPLATE.format(
        instruction=example["instruction"].strip(),
        response=example["response"].strip(),
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset_path", default="../data/insurance_tickets.jsonl")
    parser.add_argument("--output_dir", default="../outputs/adjustiq-lora")
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
    train_ds, eval_ds = dataset["train"], dataset["test"]
    print(f"Train examples: {len(train_ds)} | Eval examples: {len(eval_ds)}")

    # ---- Model + tokenizer ----
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # ---- LoRA ----
    lora_config = build_lora_config(args.target_modules, args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training config ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        max_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to: {adapter_dir}")

    # Optional: merge for a standalone deployable model
    # merged = trainer.model.merge_and_unload()
    # merged.save_pretrained(os.path.join(args.output_dir, "merged_model"))


if __name__ == "__main__":
    main()
