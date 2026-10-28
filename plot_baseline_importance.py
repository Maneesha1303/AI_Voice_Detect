from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path(
    "results/metrics/baseline_feature_importance.csv"
)

OUTPUT_FILE = Path(
    "results/graphs/baseline_feature_importance.png"
)


def main():

    print("=" * 70)
    print("BASELINE MODEL - FEATURE IMPORTANCE")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    # Select top 15 features
    top_features = df.head(15).sort_values(
        "importance",
        ascending=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.title(
        "Random Forest Baseline - Top 15 MFCC Features"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Graph saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()