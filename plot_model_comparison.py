from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
GRAPHS_DIR = ROOT / "results" / "graphs"

INPUT_FILE = METRICS_DIR / "model_comparison.csv"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {
        "model", "split", "accuracy",
        "precision", "recall", "f1", "roc_auc"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Compare baseline and CNN on the same Dev split only.
    dev = df[df["split"].str.lower() == "dev"].copy()

    if dev.empty:
        raise ValueError("No Dev results available to plot.")

    metric_columns = [
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ]

    # --------------------------------------------------
    # Graph 1: Dev metric comparison
    # --------------------------------------------------
    ax = dev.set_index("model")[metric_columns].plot(
        kind="bar",
        figsize=(11, 6),
        ylim=(0, 1.05),
        width=0.75
    )

    ax.set_title("Baseline vs CNN — Development Set")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        loc="lower right"
    )

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / "dev_model_comparison.png",
        dpi=300
    )
    plt.close()

   # --------------------------------------------------
    # Graph 2: CNN Dev vs EVAL
    # --------------------------------------------------
    cnn = df[df["model"].str.lower() == "cnn"].copy()
    cnn["split"] = cnn["split"].str.lower()

    cnn = cnn[cnn["split"].isin(["dev", "eval"])]

    available_splits = set(cnn["split"])

    if {"dev", "eval"}.issubset(available_splits):
        cnn_plot = cnn.set_index("split").loc[["dev", "eval"]]

        ax = cnn_plot[metric_columns].plot(
            kind="bar",
            figsize=(11, 6),
            ylim=(0, 1.05),
            width=0.75
        )

        ax.set_title("CNN Performance — Dev vs EVAL")
        ax.set_ylabel("Score")
        ax.set_xlabel("Dataset split")
        ax.set_xticklabels(["Dev", "EVAL"], rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(
            ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            loc="lower right"
        )

        plt.tight_layout()
        plt.savefig(
            GRAPHS_DIR / "cnn_dev_vs_eval.png",
            dpi=300
        )
        plt.close()

        print(" - cnn_dev_vs_eval.png")
    else:
        print("CNN Dev/EVAL comparison skipped: both splits are required.")

    print("MODEL COMPARISON GRAPHS CREATED")
    print(f"Output directory: {GRAPHS_DIR}")


if __name__ == "__main__":
    main()