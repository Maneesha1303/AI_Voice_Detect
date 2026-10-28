from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent

PROTOCOL_FILE = (
    BASE_DIR
    / "dataset"
    / "downloads"
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.eval.trl.txt"
)

PREDICTIONS_FILE = (
    BASE_DIR
    / "results"
    / "predictions"
    / "cnn_eval_predictions.csv"
)

METRICS_DIR = BASE_DIR / "results" / "metrics"
GRAPHS_DIR = BASE_DIR / "results" / "graphs"

OUTPUT_FILE = METRICS_DIR / "cnn_attack_analysis.csv"
MERGED_FILE = (
    BASE_DIR / "results" / "predictions"
    / "cnn_eval_predictions_with_attacks.csv"
)
GRAPH_FILE = GRAPHS_DIR / "cnn_attack_performance.png"


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [PROTOCOL_FILE, PREDICTIONS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # Read official ASVspoof protocol.
    protocol = pd.read_csv(
        PROTOCOL_FILE,
        sep=r"\s+",
        header=None,
        names=[
            "speaker_id",
            "file_id",
            "placeholder",
            "attack_id",
            "protocol_label",
        ],
        dtype=str,
    )

    # Read model predictions.
    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        dtype={"file_id": str},
    )

    required = {
        "file_id",
        "actual_label",
        "predicted_label",
        "spoof_probability",
    }

    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    # Ensure sample IDs are consistent.
    protocol["file_id"] = protocol["file_id"].str.strip()
    predictions["file_id"] = predictions["file_id"].str.strip()

    # Confirm there are no duplicate protocol IDs.
    duplicate_ids = protocol["file_id"].duplicated().sum()
    if duplicate_ids:
        raise ValueError(
            f"Protocol contains {duplicate_ids} duplicate file IDs."
        )

    # Match every prediction with its official protocol metadata.
    merged = predictions.merge(
        protocol,
        on="file_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())

    if unmatched:
        examples = merged.loc[
            merged["_merge"] != "both", "file_id"
        ].head(10).tolist()

        raise ValueError(
            f"{unmatched} predictions did not match the protocol. "
            f"Example IDs: {examples}"
        )

    merged = merged.drop(columns=["_merge"])

    # Verify that prediction labels agree with official labels.
    expected_labels = merged["protocol_label"].map(
        {"bonafide": 0, "spoof": 1}
    )

    if expected_labels.isna().any():
        raise ValueError("Unexpected labels found in the protocol.")

    merged["actual_label"] = merged["actual_label"].astype(int)
    merged["predicted_label"] = merged["predicted_label"].astype(int)
    merged["spoof_probability"] = pd.to_numeric(
        merged["spoof_probability"], errors="raise"
    )

    label_mismatches = int(
        (merged["actual_label"] != expected_labels.astype(int)).sum()
    )

    if label_mismatches:
        raise ValueError(
            f"{label_mismatches} labels disagree with the official protocol."
        )

    merged["correct"] = (
        merged["actual_label"] == merged["predicted_label"]
    )

    # Save joined predictions for reproducibility.
    merged.to_csv(MERGED_FILE, index=False)

    # Analyze each spoof attack separately.
    spoof = merged.loc[
        merged["protocol_label"] == "spoof"
    ].copy()

    spoof["attack_id"] = spoof["attack_id"].fillna("-")

    rows = []

    for attack_id, group in spoof.groupby("attack_id"):
        y_true = group["actual_label"]
        y_pred = group["predicted_label"]

        false_negatives = int(
            ((y_true == 1) & (y_pred == 0)).sum()
        )

        spoof_count = len(group)

        rows.append({
            "attack_id": attack_id,
            "samples": spoof_count,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "false_negatives": false_negatives,
            "false_negative_rate": (
                false_negatives / spoof_count
                if spoof_count else 0
            ),
            "mean_spoof_probability": group[
                "spoof_probability"
            ].mean(),
        })

    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError(
            "No spoof attack groups were found in the matched EVAL data."
        )

    results = results.sort_values(
        "false_negative_rate",
        ascending=False,
    )

    results.to_csv(OUTPUT_FILE, index=False)

    # Plot attack-wise recall.
    plot_data = results.sort_values(
        "attack_id"
    )

    plt.figure(figsize=(12, 6))
    plt.bar(
        plot_data["attack_id"],
        plot_data["recall"] * 100,
        color="steelblue",
    )
    plt.ylabel("Spoof Detection Recall (%)")
    plt.xlabel("Official Attack ID")
    plt.title("CNN EVAL: Spoof Detection Recall by Attack Type")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=200)
    plt.close()

    print("=" * 65)
    print("CNN ATTACK-TYPE ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"Protocol rows: {len(protocol)}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Matched rows: {len(merged)}")
    print(f"Unmatched predictions: {unmatched}")
    print(f"Label mismatches: {label_mismatches}")
    print(f"Spoof samples analyzed: {len(spoof)}")
    print(f"Attack types found: {len(results)}")
    print()
    print("Attack results, lowest recall first:")
    print(
        results[
            [
                "attack_id",
                "samples",
                "recall",
                "false_negatives",
                "false_negative_rate",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"CSV saved: {OUTPUT_FILE}")
    print(f"Joined predictions saved: {MERGED_FILE}")
    print(f"Graph saved: {GRAPH_FILE}")


if __name__ == "__main__":
    main()