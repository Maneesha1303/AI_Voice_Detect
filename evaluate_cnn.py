from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
# ============================================================
# ASVspoof 2019 LA - CNN EVALUATION
# ============================================================

EVAL_DIR = Path("dataset/processed/cnn/eval")
MODEL_FILE = Path("models/cnn/cnn_baseline.pth")

#MODEL_FILE = Path("models/cnn/cnn_a17_retrained.pth")

RESULTS_DIR = Path("results/metrics")
CM_DIR = Path("results/confusion_matrices")

METRICS_FILE = RESULTS_DIR / "cnn_eval_results.csv"
CM_FILE = CM_DIR / "cnn_eval_confusion_matrix.csv"
PREDICTIONS_DIR = Path("results/predictions")
PREDICTIONS_FILE = PREDICTIONS_DIR / "cnn_eval_predictions.csv"

BATCH_SIZE = 64
THRESHOLD = 0.8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# ============================================================
# Dataset
# ============================================================

class AudioFeatureDataset(Dataset):

    def __init__(self, feature_dir):

        self.feature_dir = Path(feature_dir)

        self.labels_file = self.feature_dir / "labels.csv"

        self.labels_df = pd.read_csv(
            self.labels_file
        )

        self.files = self.labels_df["file_id"].tolist()

        self.labels = (
            self.labels_df["label"]
            .astype(int)
            .tolist()
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        file_id = self.files[index]

        feature_path = (
            self.feature_dir /
            f"{file_id}.npy"
        )

        feature = np.load(
            feature_path
        )

        feature = torch.tensor(
            feature,
            dtype=torch.float32
        )

        feature = feature.unsqueeze(0)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return feature, label
# ============================================================
# CNN Model
# ============================================================

class VoiceCloneCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)
# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for batch_index, (features, labels) in enumerate(loader):

            features = features.to(DEVICE)

            labels = labels.to(DEVICE)

            logits = model(features)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= THRESHOLD
            ).int()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Evaluating batch "
                    f"{batch_index + 1}/{len(loader)}"
                )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ASVSPOOF 2019 LA - CNN EVALUATION")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    print()
    print("Checking evaluation dataset...")

    if not EVAL_DIR.exists():

        print()
        print("ERROR: Evaluation feature directory not found:")
        print(EVAL_DIR)
        print()
        print("The EVAL CNN features must be prepared first.")
        return

    if not MODEL_FILE.exists():

        print()
        print("ERROR: CNN model not found:")
        print(MODEL_FILE)
        return

    # --------------------------------------------------------
    # Create result directories
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CM_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load evaluation dataset
    # --------------------------------------------------------

    print()
    print("Loading evaluation dataset...")

    eval_dataset = AudioFeatureDataset(
        EVAL_DIR
    )

    print(
        "Evaluation samples:",
        len(eval_dataset)
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        "Evaluation batches:",
        len(eval_loader)
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading CNN model...")

    model = VoiceCloneCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=DEVICE
        )
    )

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)

    results = evaluate(
        model,
        eval_loader
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL EVAL RESULTS")
    print("=" * 70)

    print()
    print(
        f"Accuracy : {results['accuracy']:.4f}"
    )

    print(
        f"Precision: {results['precision']:.4f}"
    )

    print(
        f"Recall   : {results['recall']:.4f}"
    )

    print(
        f"F1-score : {results['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {results['roc_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        results["confusion_matrix"]
    )

    print()
    print("Classification Report:")

    print(
        classification_report(
            results["labels"],
            results["predictions"],
            target_names=[
                "bonafide",
                "spoof"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "model": "CNN",
            "split": "eval",
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "roc_auc": results["roc_auc"]
        }
    ])

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        results["confusion_matrix"],
        index=[
            "actual_bonafide",
            "actual_spoof"
        ],
        columns=[
            "predicted_bonafide",
            "predicted_spoof"
        ]
    )

    cm_df.to_csv(
        CM_FILE
    )
# --------------------------------------------------------
    # Save individual predictions for error analysis
    # --------------------------------------------------------

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame({
        "file_id": eval_dataset.files,
        "actual_label": results["labels"],
        "predicted_label": results["predictions"],
        "spoof_probability": results["probabilities"],
    })

    prediction_df["actual_class"] = prediction_df[
        "actual_label"
    ].map({
        0: "bonafide",
        1: "spoof"
    })

    prediction_df["predicted_class"] = prediction_df[
        "predicted_label"
    ].map({
        0: "bonafide",
        1: "spoof"
    })

    prediction_df["correct"] = (
        prediction_df["actual_label"]
        == prediction_df["predicted_label"]
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print(
        f"Individual predictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )
    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CNN EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Metrics saved to:")
    print(METRICS_FILE)

    print()
    print("Confusion matrix saved to:")
    print(CM_FILE)


if __name__ == "__main__":

    main()