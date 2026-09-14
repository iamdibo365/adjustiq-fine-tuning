"""
Run a single customer message through the fine-tuned insurance adapter.

Run:
    python infer_qlora.py \
        --base_model mistralai/Mistral-7B-v0.1 \
        --adapter_dir ../outputs/adjustiq-lora/final_adapter \
        --prompt "Hi, this is Maria Garcia. My claim CLM-4471203 for storm damage has been stuck in review for 3 weeks."
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROMPT_TEMPLATE = """### Customer Message:
{instruction}

### Triage Response:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    text = PROMPT_TEMPLATE.format(instruction=args.prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
