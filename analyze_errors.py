from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "predictions" / "cnn_eval_predictions.csv"
OUTPUT_DIR = ROOT / "results" / "reports"
PREDICTIONS_DIR = ROOT / "results" / "predictions"

ERRORS_FILE = PREDICTIONS_DIR / "cnn_eval_misclassified.csv"
SUMMARY_FILE = OUTPUT_DIR / "cnn_eval_error_summary.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "file_id",
        "actual_label",
        "predicted_label",
        "spoof_probability",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["actual_label"] = df["actual_label"].astype(int)
    df["predicted_label"] = df["predicted_label"].astype(int)

    df["actual_class"] = df["actual_label"].map({
        0: "bonafide",
        1: "spoof",
    })

    df["predicted_class"] = df["predicted_label"].map({
        0: "bonafide",
        1: "spoof",
    })

    df["correct"] = (
        df["actual_label"] == df["predicted_label"]
    )

    # Positive class = spoof
    false_positives = df[
        (df["actual_label"] == 0)
        & (df["predicted_label"] == 1)
    ]

    false_negatives = df[
        (df["actual_label"] == 1)
        & (df["predicted_label"] == 0)
    ]

    errors = df[~df["correct"]].copy()
    errors["error_type"] = errors.apply(
        lambda row: (
            "false_positive"
            if row["actual_label"] == 0
            else "false_negative"
        ),
        axis=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    errors.to_csv(ERRORS_FILE, index=False)

    total = len(df)
    correct = int(df["correct"].sum())

    summary = pd.DataFrame([
        {
            "dataset": "ASVspoof 2019 LA EVAL",
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": len(errors),
            "accuracy": correct / total if total else 0,
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "bonafide_samples": int((df["actual_label"] == 0).sum()),
            "spoof_samples": int((df["actual_label"] == 1).sum()),
        }
    ])

    summary.to_csv(SUMMARY_FILE, index=False)

    print("=" * 60)
    print("CNN EVAL ERROR ANALYSIS")
    print("=" * 60)
    print(f"Total samples       : {total}")
    print(f"Correct predictions : {correct}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"False positives     : {len(false_positives)}")
    print(f"False negatives     : {len(false_negatives)}")

    print("\nError rate by actual class:")
    for label, name in [(0, "bonafide"), (1, "spoof")]:
        subset = df[df["actual_label"] == label]
        count = int((~subset["correct"]).sum())
        rate = count / len(subset) if len(subset) else 0
        print(f"{name:10s}: {count}/{len(subset)} ({rate:.2%})")

    print(f"\nMisclassified samples saved to:\n{ERRORS_FILE}")
    print(f"Summary saved to:\n{SUMMARY_FILE}")


if __name__ == "__main__":
    main()