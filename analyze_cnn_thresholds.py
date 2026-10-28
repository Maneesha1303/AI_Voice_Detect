from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evaluate_cnn import (
    AudioFeatureDataset,
    VoiceCloneCNN,
    DEVICE,
    MODEL_FILE,
)

DEV_DIR = Path("dataset/processed/cnn/dev")
OUTPUT_FILE = Path("results/metrics/cnn_threshold_analysis.csv")
BATCH_SIZE = 64

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def main():
    print("Loading development dataset...")

    dataset = AudioFeatureDataset(DEV_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Loading original CNN...")
    model = VoiceCloneCNN().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_FILE, map_location=DEVICE)
    )
    model.eval()

    labels_all = []
    probabilities_all = []

    with torch.no_grad():
        for batch_index, (features, labels) in enumerate(loader):
            features = features.to(DEVICE)

            logits = model(features)
            probabilities = torch.sigmoid(logits)

            labels_all.extend(labels.numpy().tolist())
            probabilities_all.extend(
                probabilities.cpu().numpy().tolist()
            )

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed {batch_index + 1}/{len(loader)} batches"
                )

    y_true = np.asarray(labels_all, dtype=int)
    y_prob = np.asarray(probabilities_all)

    rows = []

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "spoof_recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "false_positive_rate": fp / (fp + tn),
            "false_negative_rate": fn / (fn + tp),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    results = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    display_columns = [
        "threshold",
        "accuracy",
        "precision",
        "spoof_recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
    ]

    print("\nTHRESHOLD COMPARISON — DEVELOPMENT SET")
    print(
        results[display_columns].round(4).to_string(index=False)
    )
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()