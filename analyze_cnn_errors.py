from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "results" / "predictions"
METRICS_DIR = BASE_DIR / "results" / "metrics"
GRAPHS_DIR = BASE_DIR / "results" / "graphs"

PRED_FILE = PRED_DIR / "cnn_eval_predictions.csv"
ERROR_FILE = PRED_DIR / "cnn_eval_misclassified.csv"


def main():
    for folder in [PRED_DIR, METRICS_DIR, GRAPHS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    if not PRED_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PRED_FILE}"
        )

    df = pd.read_csv(PRED_FILE)

    required = {
        "actual_label",
        "predicted_label",
        "spoof_probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["actual_label"] = df["actual_label"].astype(int)
    df["predicted_label"] = df["predicted_label"].astype(int)
    df["spoof_probability"] = pd.to_numeric(
        df["spoof_probability"], errors="raise"
    )

    df["actual_class"] = df["actual_label"].map(
        {0: "bonafide", 1: "spoof"}
    )
    df["predicted_class"] = df["predicted_label"].map(
        {0: "bonafide", 1: "spoof"}
    )

    df["correct"] = (
        df["actual_label"] == df["predicted_label"]
    )

    df["error_type"] = "correct"
    df.loc[
        (df["actual_label"] == 0)
        & (df["predicted_label"] == 1),
        "error_type",
    ] = "false_positive"

    df.loc[
        (df["actual_label"] == 1)
        & (df["predicted_label"] == 0),
        "error_type",
    ] = "false_negative"

    errors = df.loc[~df["correct"]].copy()

    errors.to_csv(ERROR_FILE, index=False)

    total = len(df)
    correct = int(df["correct"].sum())
    fp = int((df["error_type"] == "false_positive").sum())
    fn = int((df["error_type"] == "false_negative").sum())

    summary = pd.DataFrame([
        {
            "split": "eval",
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": len(errors),
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": correct / total if total else 0,
            "false_positive_rate": (
                fp / int((df["actual_label"] == 0).sum())
                if (df["actual_label"] == 0).any()
                else 0
            ),
            "false_negative_rate": (
                fn / int((df["actual_label"] == 1).sum())
                if (df["actual_label"] == 1).any()
                else 0
            ),
        }
    ])

    summary_file = METRICS_DIR / "cnn_eval_error_summary.csv"
    summary.to_csv(summary_file, index=False)

    print("=" * 60)
    print("CNN EVAL ERROR ANALYSIS")
    print("=" * 60)
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Incorrect     : {len(errors)}")
    print(f"False positive: {fp}")
    print(f"False negative: {fn}")
    print()
    print("Error summary saved to:")
    print(summary_file)
    print()
    print("Misclassified samples saved to:")
    print(ERROR_FILE)


if __name__ == "__main__":
    main()