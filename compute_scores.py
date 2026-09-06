import pandas as pd
import numpy as np

# Load the results that were already saved successfully
df = pd.read_csv("miq_results_tinyllama.csv")

TAU = 0.5
W1, W2, W3 = 0.40, 0.35, 0.25

def compute_ece_cs(group, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(group)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (group["confidence"] >= lo) & (group["confidence"] <= hi)
        bin_data = group[mask]
        if len(bin_data) == 0:
            continue
        acc  = bin_data["is_correct"].mean()
        conf = bin_data["confidence"].mean()
        ece += (len(bin_data) / n) * abs(acc - conf)
    return round(1.0 - ece, 4)

def compute_sar(group):
    eligible = group[group["should_abstain"] == True]
    if len(eligible) == 0:
        return 0.0
    return round(float(eligible["abstained"].sum() / len(eligible)), 4)

def compute_upa(group):
    arc_data = group[group["dataset"] == "ARC"]
    if len(arc_data) < 2:
        return 0.5
    confidences = arc_data["confidence"].values
    oracle = np.mean(confidences)
    errors = np.abs(confidences - oracle)
    return round(float(1.0 - np.mean(errors)), 4)

final_rows = []
for model in df["model"].unique():
    sub      = df[df["model"] == model]
    accuracy = round(sub["is_correct"].mean() * 100, 1)
    cs       = compute_ece_cs(sub)
    sar      = compute_sar(sub)
    upa      = compute_upa(sub)
    miq      = round(100 * (W1*cs + W2*sar + W3*upa), 1)

    if   miq >= 80: tier = "Expert"
    elif miq >= 60: tier = "Proficient"
    elif miq >= 40: tier = "Developing"
    else:           tier = "Deficient"

    final_rows.append({
        "Model": model, "Accuracy (%)": accuracy,
        "CS": cs, "SAR": sar, "UPA": upa,
        "M-IQ": miq, "Tier": tier
    })

scores_df = pd.DataFrame(final_rows)
scores_df.to_csv("miq_final_scores_tinyllama.csv", index=False)

print("\n=== TABLE VII — TINYLLAMA RESULTS ===")
print(scores_df.to_string(index=False))