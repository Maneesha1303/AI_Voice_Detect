from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "results" / "predictions"
    / "cnn_eval_predictions_with_attacks.csv"
)

OUTPUT = ROOT / "results" / "metrics"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Filter A17 spoof samples only.
a17 = df[
    (df["attack_id"].astype(str) == "A17")
    & (df["actual_label"] == 1)
].copy()

a17["result"] = a17["predicted_label"].map(
    {0: "Missed spoof", 1: "Detected spoof"}
)

print("=" * 65)
print("A17 SPOOF PROBABILITY ANALYSIS")
print("=" * 65)

print("Total A17 spoof samples:", len(a17))
print("\nPrediction counts:")
print(a17["result"].value_counts())

print("\nProbability statistics:")
print(
    a17.groupby("result")["spoof_probability"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(6)
)

# Save summary.
summary = (
    a17.groupby("result")["spoof_probability"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .reset_index()
)

summary.to_csv(
    OUTPUT / "cnn_a17_probability_summary.csv",
    index=False
)

# Count predictions across probability ranges.
bins = [-0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.001]

a17["probability_range"] = pd.cut(
    a17["spoof_probability"],
    bins=bins,
    labels=[
        "0-0.01", "0.01-0.1", "0.1-0.3",
        "0.3-0.5", "0.5-0.7", "0.7-0.9",
        "0.9-1.0"
    ]
)

distribution = pd.crosstab(
    a17["probability_range"],
    a17["result"]
)

print("\nProbability distribution:")
print(distribution)

distribution.to_csv(
    OUTPUT / "cnn_a17_probability_distribution.csv"
)

print("\nAnalysis saved in:", OUTPUT)