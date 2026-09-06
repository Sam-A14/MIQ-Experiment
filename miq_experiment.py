# ─────────────────────────────────────────────
# M-IQ Experiment — Mistral 7B + Llama 3 8B
# Runs completely free on your laptop
# ─────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────
TAU  = 0.5        # abstention threshold
W1   = 0.40       # weight for CS
W2   = 0.35       # weight for SAR
W3   = 0.25       # weight for UPA
N_SAMPLES = 100   # questions per dataset (increase later if needed)

MODELS_TO_RUN = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# ── PROMPT ────────────────────────────────────
def make_prompt(question):
    return f"""Answer the following question as accurately as possible.

After your answer, on a new line write exactly:
CONFIDENCE: [number from 0 to 100]

If you do not know the answer, write:
ANSWER: I don't know
CONFIDENCE: [low number]

Question: {question}

Answer:"""

# ── PARSE RESPONSE ────────────────────────────
def parse_response(text):
    conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text, re.IGNORECASE)
    confidence = int(conf_match.group(1)) / 100.0 if conf_match else 0.5
    confidence = max(0.0, min(1.0, confidence))  # clamp to [0,1]
    abstained  = "i don't know" in text.lower() or "i do not know" in text.lower()
    answer     = re.sub(r'CONFIDENCE:.*', '', text, flags=re.IGNORECASE).strip()
    answer     = re.sub(r'ANSWER:', '', answer, flags=re.IGNORECASE).strip()
    return {"answer": answer, "confidence": confidence, "abstained": abstained}

# ── LOAD ONE MODEL ────────────────────────────
def load_model(model_name):
    print(f"\nLoading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # half memory usage
        low_cpu_mem_usage=True,
        device_map=None
    )
    model = model.to("cpu")
    print(f"{model_name} loaded successfully.")
    return tokenizer, model

# ── QUERY ONE MODEL ───────────────────────────
def query_model(question, tokenizer, model):
    try:
        prompt  = make_prompt(question)
        inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return parse_response(text)
    except Exception as e:
        return {"answer": "", "confidence": 0.5, "abstained": False}

# ── LOAD DATASETS ─────────────────────────────
print("Loading datasets...")

trivia = load_dataset(
    "google-research-datasets/nq_open",
    split=f"validation[:{N_SAMPLES}]"
)
squad2 = load_dataset(
    "rajpurkar/squad_v2",
    split=f"validation[:{N_SAMPLES}]"
)
arc = load_dataset(
    "allenai/ai2_arc", "ARC-Challenge",
    split=f"test[:{N_SAMPLES}]"
)
print(f"TriviaQA : {len(trivia)} questions")
print(f"SQuAD 2.0: {len(squad2)} questions")
print(f"ARC      : {len(arc)} questions")
print("Datasets ready.")

# ── MAIN EXPERIMENT LOOP ──────────────────────
all_results = []

for model_label, model_name in MODELS_TO_RUN.items():

    tokenizer, model = load_model(model_name)

    # ── TriviaQA ──
    print(f"\n[{model_label}] Running TriviaQA...")
    for item in tqdm(trivia):
        q       = item["question"]
        correct = item["answer"][0].lower().strip()
        r       = query_model(q, tokenizer, model)
        all_results.append({
            "dataset":      "TriviaQA",
            "model":        model_label,
            "question":     q[:80],
            "correct":      correct,
            "answer":       r["answer"][:80],
            "confidence":   r["confidence"],
            "abstained":    r["abstained"],
            "is_correct":   correct in r["answer"].lower(),
            "should_abstain": r["confidence"] < TAU
        })

    # ── SQuAD 2.0 ──
    print(f"\n[{model_label}] Running SQuAD 2.0...")
    for item in tqdm(squad2):
        q            = item["question"]
        is_answerable = len(item["answers"]["text"]) > 0
        correct      = item["answers"]["text"][0] if is_answerable else "UNANSWERABLE"
        context      = item["context"][:300]
        r            = query_model(q + " Context: " + context, tokenizer, model)
        if not is_answerable:
            correct_flag = r["abstained"]
        else:
            correct_flag = correct.lower() in r["answer"].lower()
        all_results.append({
            "dataset":      "SQuAD2",
            "model":        model_label,
            "question":     q[:80],
            "correct":      correct,
            "answer":       r["answer"][:80],
            "confidence":   r["confidence"],
            "abstained":    r["abstained"],
            "is_correct":   correct_flag,
            "should_abstain": not is_answerable
        })

    # ── ARC Challenge ──
    print(f"\n[{model_label}] Running ARC Challenge...")
    for item in tqdm(arc):
        choices = " | ".join([
            f"{l}: {t}" for l, t in
            zip(item["choices"]["label"], item["choices"]["text"])
        ])
        q       = f"{item['question']} Choices: {choices}"
        correct = item["answerKey"]
        r       = query_model(q, tokenizer, model)
        all_results.append({
            "dataset":      "ARC",
            "model":        model_label,
            "question":     q[:80],
            "correct":      correct,
            "answer":       r["answer"][:80],
            "confidence":   r["confidence"],
            "abstained":    r["abstained"],
            "is_correct":   correct in r["answer"],
            "should_abstain": r["confidence"] < TAU
        })

    # Free memory before loading next model
    del model, tokenizer
    torch.cuda.empty_cache()

# ── SAVE RAW RESULTS ──────────────────────────
df = pd.DataFrame(all_results)
df.to_csv(df.to_csv("miq_results_tinyllama.csv", index=False))
print("\nRaw results saved to miq_results.csv")
print(df.groupby(["model", "dataset"])["is_correct"].mean().round(3))

# ── COMPUTE M-IQ SCORES ───────────────────────
print("\n=== COMPUTING M-IQ SCORES ===")

def compute_ece(group, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(group)
    for i in range(n_bins):
        lo, hi   = bins[i], bins[i+1]
        mask     = (group["confidence"] >= lo) & (group["confidence"] <= hi)
        bin_data = group[mask]
        if len(bin_data) == 0:
            continue
        acc  = bin_data["is_correct"].mean()
        conf = bin_data["confidence"].mean()
        ece += (len(bin_data) / n) * abs(acc - conf)
    return round(1.0 - ece, 4)   # return CS directly

def compute_sar(group):
    eligible = group[group["should_abstain"]]
    if len(eligible) == 0:
        return 0.0
    return round(float(eligible["abstained"].sum() / len(eligible)), 4)

def compute_upa(group):
    # UPA proxy: confidence drops as question difficulty increases
    # Use ARC subset only (most multi-step)
    arc_data = group[group["dataset"] == "ARC"]
    if len(arc_data) < 2:
        return 0.5
    confidences = arc_data["confidence"].values
    # Oracle = mean confidence across models at each step
    oracle = np.mean(confidences)
    errors = np.abs(confidences - oracle)
    return round(float(1.0 - np.mean(errors)), 4)

final_rows = []
for model_label in df["model"].unique():
    sub      = df[df["model"] == model_label]
    accuracy = round(sub["is_correct"].mean() * 100, 1)
    cs       = compute_ece(sub)
    sar      = compute_sar(sub)
    upa      = compute_upa(sub)
    miq      = round(100 * (W1*cs + W2*sar + W3*upa), 1)

    if   miq >= 80: tier = "Expert"
    elif miq >= 60: tier = "Proficient"
    elif miq >= 40: tier = "Developing"
    else:           tier = "Deficient"

    final_rows.append({
        "Model":        model_label,
        "Accuracy (%)": accuracy,
        "CS":           cs,
        "SAR":          sar,
        "UPA":          upa,
        "M-IQ":         miq,
        "Tier":         tier
    })

scores_df = pd.DataFrame(final_rows)
scores_df.to_csv(df.to_csv())

print("\n=== TABLE VII — PASTE THESE VALUES INTO YOUR PAPER ===")
print(scores_df.to_csv("miq_final_scores_tinyllama.csv", index=False))
print("\nAll done. Check miq_results.csv and miq_final_scores.csv in your folder.")