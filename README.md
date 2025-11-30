## QLoRA Fine-Tuning on Mistral-7B-Instruct (Dolly-15k Dataset)

This repository contains a QLoRA fine-tuned model based on **Mistral-7B-Instruct-v0.2**, trained on the **Dolly-15k instruction dataset**.  
The repository stores **LoRA adapter weights**, not a full model.  
To use the model for inference, you must first load the Mistral base model, then apply the adapter.

---

## 🚀 Model Repository
HuggingFace Hub: **Darialjx2001/qlora-mistral7b-dolly15k**

Contents:
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja
special_tokens_map.json

---

# Quick Start -> Load QLoRA Adapter (HuggingFace)

```python
!pip install -U bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

adapter_repo = "Darialjx2001/qlora-mistral7b-dolly15k"
original_model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(original_model)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    original_model,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    adapter_repo
)

prompt = "Explain QLoRA in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

---

## 📊 Training Summary

Method: QLoRA (Quantized Low-Rank Adaptation)
Base Model: Mistral-7B-Instruct-v0.2
Dataset: Databricks Dolly-15k
Quantization: 4-bit NF4
PEFT Library: HuggingFace PEFT
Hardware: GPU: NVIDIA GeForce RTX 3090
Objective: Instruction tuning and conversational improvements
