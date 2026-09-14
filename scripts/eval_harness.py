"""
Compare base model vs LoRA-adapted model on a held-out sample of tickets.
Checks whether the fine-tuned model reliably emits the "Triage: ... | Priority: ... |
Draft reply: ..." structure the base model was never trained to produce.

Run:
    python eval_harness.py \
        --base_model mistralai/Mistral-7B-v0.1 \
        --adapter_dir ../outputs/adjustiq-lora/final_adapter \
        --dataset_path ../data/insurance_tickets.jsonl \
        --n_samples 15
"""

import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROMPT_TEMPLATE = """### Customer Message:
{instruction}

### Triage Response:
"""

STRUCT_RE = re.compile(r"Triage:.*\|\s*Priority:.*\|\s*Draft reply:", re.IGNORECASE)


def generate(model, tokenizer, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducible eval
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--dataset_path", default="../data/insurance_tickets.jsonl")
    parser.add_argument("--n_samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = [json.loads(l) for l in open(args.dataset_path)]
    sample = random.sample(rows, args.n_samples)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    tuned_model.eval()

    base_hits, tuned_hits = 0, 0
    for i, row in enumerate(sample, 1):
        prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])

        # Adapter temporarily disabled -> pure base-model behavior on the
        # same loaded weights (avoids loading the model twice).
        with tuned_model.disable_adapter():
            base_out = generate(tuned_model, tokenizer, prompt)
        base_ok = bool(STRUCT_RE.search(base_out))
        base_hits += base_ok

        # Adapter active -> fine-tuned behavior.
        tuned_out = generate(tuned_model, tokenizer, prompt)
        tuned_ok = bool(STRUCT_RE.search(tuned_out))
        tuned_hits += tuned_ok

        print(f"\n[{i}/{len(sample)}] Customer: {row['instruction'][:90]}...")
        print(f"  Base structured-output match:  {base_ok}")
        print(f"  Tuned structured-output match: {tuned_ok}")

    print("\n" + "=" * 50)
    print(f"Base model:  {base_hits}/{len(sample)} correctly structured")
    print(f"Tuned model: {tuned_hits}/{len(sample)} correctly structured")
    print("=" * 50)
    print(
        "Note: this checks output *format* compliance (triage/priority/reply structure), "
        "not factual quality. Read a few full generations manually to judge the draft-reply "
        "content and triage-category accuracy against the ground truth in the dataset."
    )


if __name__ == "__main__":
    main()
