"""
Lab 7 — local runner (no Colab dependency).
Loads dpo_pairs.jsonl from the same directory and runs the full lab.
"""
import os, json, random, sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import DPOTrainer, DPOConfig

# ─── setup ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

PAIRS_FILE = Path(__file__).parent / "dpo_pairs.jsonl"

# ─── load data ────────────────────────────────────────────────────────────────
def load_dpo_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if all(k in r for k in ["prompt", "chosen", "rejected"]):
                rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    return rows

pairs = load_dpo_jsonl(PAIRS_FILE)
print(f"Loaded {len(pairs)} preference pairs")
print("\nExample pair keys:", list(pairs[0].keys()))
print("Prompt:", pairs[0]["prompt"])

# ─── split ────────────────────────────────────────────────────────────────────
random.seed(0)
random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
test_pairs  = pairs[split:] if split < len(pairs) else pairs[:2]

train_ds = Dataset.from_list(train_pairs)
test_ds  = Dataset.from_list(test_pairs)
print(f"\nTrain: {len(train_ds)}  Test: {len(test_ds)}")

# ─── load GPT-2 ───────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"
print(f"\nLoading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
ref    = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)
print("Models loaded.")

# ─── helper functions ─────────────────────────────────────────────────────────
def tokenize_prompt_and_response(prompt: str, response: str, max_len=256):
    full = prompt + "\n" + response
    enc_full   = tokenizer(full,        return_tensors="pt", truncation=True, max_length=max_len, padding=False)
    enc_prompt = tokenizer(prompt + "\n", return_tensors="pt", truncation=True, max_length=max_len, padding=False)
    return enc_full, enc_prompt["input_ids"].shape[1]

def logprobs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def kl_tokenwise(policy_logits, ref_logits):
    p    = F.softmax(policy_logits, dim=-1)
    logp = F.log_softmax(policy_logits, dim=-1)
    logr = F.log_softmax(ref_logits,    dim=-1)
    return torch.sum(p * (logp - logr), dim=-1)

# ─── KL-regularised warm-up ───────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 — KL-regularised behaviour cloning")
print("="*60)

beta  = 0.05
lr    = 5e-5
steps = min(30, len(train_pairs) * 3)
opt   = torch.optim.AdamW(policy.parameters(), lr=lr)
policy.train()
set_seed(0)

def one_kl_step(example):
    prompt, chosen = example["prompt"], example["chosen"]
    enc_full, prompt_len = tokenize_prompt_and_response(prompt, chosen, max_len=256)
    input_ids = enc_full["input_ids"].to(device)
    attn      = enc_full["attention_mask"].to(device)

    labels      = input_ids[:, 1:].contiguous()
    input_ids_in = input_ids[:, :-1].contiguous()
    attn_in     = attn[:, :-1].contiguous()

    logits_p = policy(input_ids=input_ids_in, attention_mask=attn_in).logits
    logits_r = ref(   input_ids=input_ids_in, attention_mask=attn_in).logits

    start    = max(prompt_len - 1, 0)
    tok_logp = logprobs_from_logits(logits_p, labels)
    nll      = -tok_logp[:, start:].mean()
    kl       = kl_tokenwise(logits_p, logits_r)[:, start:].mean()
    loss     = nll + beta * kl
    return loss, nll.detach(), kl.detach()

kl_losses = []
for i in range(steps):
    ex = train_pairs[i % len(train_pairs)]
    opt.zero_grad()
    loss, nll, kl = one_kl_step(ex)
    loss.backward()
    opt.step()
    kl_losses.append((loss.item(), nll.item(), kl.item()))
    if (i + 1) % 10 == 0:
        print(f"  step {i+1:03d} | loss={loss.item():.4f}  nll={nll.item():.4f}  kl={kl.item():.4f}")

policy_kl = policy

# ─── DPO ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 — Direct Preference Optimisation (DPO)")
print("="*60)

dpo_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
dpo_ref   = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
dpo_ref.eval()
for p in dpo_ref.parameters():
    p.requires_grad_(False)
print("DPO models ready.")

train_ds_dpo = train_ds
eval_ds_dpo  = test_ds

config = DPOConfig(
    output_dir="dpo_out",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-6,
    num_train_epochs=1,
    max_length=256,
    truncation_mode="keep_start",
    logging_steps=5,
    save_strategy="no",
    eval_strategy="no",
    bf16=False,
    fp16=False,   # MPS/CPU: disable fp16
    beta=0.1,
    report_to="none",
)

trainer = DPOTrainer(
    model=dpo_model,
    ref_model=dpo_ref,
    args=config,
    train_dataset=train_ds_dpo,
    eval_dataset=eval_ds_dpo,
)

trainer.train()
print("DPO training complete.")

# ─── generation + evaluation ──────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 3 — Evaluation on held-out prompts")
print("="*60)

heldout_prompts = [ex["prompt"] for ex in test_pairs[:min(6, len(test_pairs))]]

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

def generate(model, prompt, max_new_tokens=80, temperature=0.8, top_p=0.95, seed=0):
    model.eval()
    set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

print("\nSample outputs on held-out prompts:\n")
for p in heldout_prompts:
    a = generate(base_model, p, seed=1)
    b = generate(policy_kl,  p, seed=1)
    c = generate(dpo_model,  p, seed=1)
    print("=" * 90)
    print("PROMPT:", p)
    print("\n[BASE]\n", a)
    print("\n[KL-ONLY]\n", b)
    print("\n[DPO]\n", c)

# ─── metrics ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 4 — Automatic metrics")
print("="*60)

def repetition_score(text: str) -> float:
    toks = text.split()
    if len(toks) < 10:
        return 0.0
    return 1.0 - (len(set(toks)) / len(toks))

REFUSAL_PHRASES = [
    "i'm not able to", "i cannot", "i can't", "i am not able to",
    "unable to", "i won't", "i will not", "i don't think i can",
    "i'm unable to", "that's not something i can", "i must decline",
    "i have to decline", "i'd prefer not to", "i'm not going to",
    "i am not going to",
]

def refusal_flag(text: str) -> int:
    t = text.lower()
    return int(any(ph in t for ph in REFUSAL_PHRASES))

rows = []
for p in heldout_prompts:
    for name, model in [("base", base_model), ("kl", policy_kl), ("dpo", dpo_model)]:
        out = generate(model, p, seed=2)
        rows.append({
            "model":      name,
            "prompt":     p[:60] + ("..." if len(p) > 60 else ""),
            "len_words":  len(out.split()),
            "repetition": repetition_score(out),
            "refusal":    refusal_flag(out),
        })

df = pd.DataFrame(rows)
print("\nMean metrics by model:")
print(df.groupby("model")[["len_words", "repetition", "refusal"]].mean().to_string())
print("\nDone.")
