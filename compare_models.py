from pathlib import Path

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"

BASELINE_FILE = METRICS_DIR / "baseline_results.csv"
CNN_FILE = METRICS_DIR / "cnn_results.csv"
CNN_EVAL_FILE = METRICS_DIR / "cnn_eval_results.csv"

OUTPUT_FILE = METRICS_DIR / "model_comparison.csv"


def load_baseline():
    """Load baseline results and identify its model name."""
    df = pd.read_csv(BASELINE_FILE)

    required = {
        "split", "accuracy", "precision",
        "recall", "f1", "roc_auc"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing baseline columns: {sorted(missing)}"
        )

    df = df.copy()
    df["model"] = "Baseline"
    return df


def load_cnn(path, expected_split):
    """Load CNN metrics and validate the split label."""
    df = pd.read_csv(path)

    required = {
        "model", "split", "accuracy",
        "precision", "recall", "f1", "roc_auc"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {sorted(missing)}"
        )

    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower()

    if not (df["split"] == expected_split).all():
        raise ValueError(
            f"Expected split '{expected_split}' in {path.name}"
        )

    return df


def main():
    baseline = load_baseline()
    cnn_dev = load_cnn(CNN_FILE, "dev")
    cnn_eval = load_cnn(CNN_EVAL_FILE, "eval")

    # Compare models on the same Dev split.
    baseline_dev = baseline[
        baseline["split"].astype(str).str.lower() == "dev"
    ].copy()

    if baseline_dev.empty:
        raise ValueError(
            "No Dev results found in baseline_results.csv"
        )

    baseline_dev["model"] = "Baseline"
    cnn_dev["model"] = "CNN"

    dev_comparison = pd.concat(
        [baseline_dev, cnn_dev],
        ignore_index=True,
        sort=False
    )

    # Keep CNN EVAL results separate from the Dev comparison.
    cnn_eval["comparison_group"] = "CNN EVAL - separate evaluation"
    dev_comparison["comparison_group"] = "Dev - direct comparison"

    comparison = pd.concat(
        [dev_comparison, cnn_eval],
        ignore_index=True,
        sort=False
    )

    columns = [
        "model", "split", "comparison_group",
        "accuracy", "precision", "recall",
        "f1", "roc_auc"
    ]

    comparison = comparison[columns]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_FILE, index=False)

    print("=" * 65)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 65)

    print("\nDirect comparison: Dev split")
    print(
        dev_comparison[
            ["model", "split", "accuracy",
             "precision", "recall", "f1", "roc_auc"]
        ].to_string(index=False)
    )

    print("\nCNN EVAL results (reported separately)")
    print(
        cnn_eval[
            ["model", "split", "accuracy",
             "precision", "recall", "f1", "roc_auc"]
        ].to_string(index=False)
    )

    print(f"\nComparison saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()