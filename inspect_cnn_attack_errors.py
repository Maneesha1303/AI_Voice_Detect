from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS = (
    ROOT / "results" / "predictions"
    / "cnn_eval_predictions_with_attacks.csv"
)

OUT_DIR = ROOT / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS)

# Inspect the actual column names first.
print("Columns:", df.columns.tolist())

# Adapt this mapping if your CSV uses different names.
attack_col = "attack_id"
actual_col = "actual_label"
pred_col = "predicted_label"
prob_col = "spoof_probability"

a17 = df[df[attack_col].astype(str) == "A17"].copy()

# Label 1 = spoof, according to your current pipeline.
a17["error_type"] = "correct"
a17.loc[
    (a17[actual_col] == 1) & (a17[pred_col] == 0),
    "error_type"
] = "false_negative"
a17.loc[
    (a17[actual_col] == 0) & (a17[pred_col] == 1),
    "error_type"
] = "false_positive"

print("\nA17 summary")
print("Samples:", len(a17))
print(a17["error_type"].value_counts())

print("\nProbability statistics by error type")
print(
    a17.groupby("error_type")[prob_col]
    .agg(["count", "mean", "median", "min", "max"])
    .round(4)
)

# Show spoof samples the model missed, lowest scores first.
missed = a17[
    (a17[actual_col] == 1) & (a17[pred_col] == 0)
].sort_values(prob_col)

output = OUT_DIR / "cnn_a17_missed_samples.csv"
missed.to_csv(output, index=False)

print(f"\nSaved missed A17 samples to: {output}")
print(missed.head(20).to_string(index=False))