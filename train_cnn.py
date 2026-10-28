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
# ASVspoof 2019 LA - CNN Training
# ============================================================

TRAIN_DIR = Path("dataset/processed/cnn/train")
DEV_DIR = Path("dataset/processed/cnn/dev")

MODEL_DIR = Path("models/cnn")
RESULTS_DIR = Path("results/metrics")
CM_DIR = Path("results/confusion_matrices")

MODEL_FILE = MODEL_DIR / "cnn_a17_retrained.pth"
METRICS_FILE = RESULTS_DIR / "cnn_results.csv"
CM_FILE = CM_DIR / "cnn_confusion_matrix.csv"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

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

        self.labels_df = pd.read_csv(self.labels_file)

        self.files = self.labels_df["file_id"].tolist()
        self.labels = self.labels_df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_id = self.files[index]

        feature_path = self.feature_dir / f"{file_id}.npy"

        feature = np.load(feature_path)

        feature = torch.tensor(
            feature,
            dtype=torch.float32
        )

        # CNN expects:
        # [channels, height, width]
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
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
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

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
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
    print("ASVSPOOF 2019 LA - CNN TRAINING")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CM_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print()
    print("Loading training dataset...")

    train_dataset = AudioFeatureDataset(
        TRAIN_DIR
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print()
    print("Loading development dataset...")

    dev_dataset = AudioFeatureDataset(
        DEV_DIR
    )

    print(
        "Development samples:",
        len(dev_dataset)
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = VoiceCloneCNN().to(DEVICE)

    print()
    print("Model:")
    print(model)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_f1 = 0.0

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_index, (features, labels) in enumerate(
            train_loader
        ):

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(features)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if (
                batch_index + 1
            ) % 100 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Batch {batch_index + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        average_loss = (
            running_loss /
            len(train_loader)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        results = evaluate(
            model,
            dev_loader
        )

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} complete"
        )

        print(
            f"Training Loss: {average_loss:.4f}"
        )

        print(
            f"Dev Accuracy: {results['accuracy']:.4f}"
        )

        print(
            f"Dev Precision: {results['precision']:.4f}"
        )

        print(
            f"Dev Recall: {results['recall']:.4f}"
        )

        print(
            f"Dev F1: {results['f1']:.4f}"
        )

        print(
            f"Dev ROC-AUC: {results['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if results["f1"] > best_f1:

            best_f1 = results["f1"]

            torch.save(
                model.state_dict(),
                MODEL_FILE
            )

            print(
                "Best model saved."
            )

        print("-" * 70)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DEV RESULTS")
    print("=" * 70)

    # Load best model

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=DEVICE
        )
    )

    final_results = evaluate(
        model,
        dev_loader
    )

    print()
    print(
        f"Accuracy : {final_results['accuracy']:.4f}"
    )

    print(
        f"Precision: {final_results['precision']:.4f}"
    )

    print(
        f"Recall   : {final_results['recall']:.4f}"
    )

    print(
        f"F1-score : {final_results['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {final_results['roc_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        final_results["confusion_matrix"]
    )

    print()
    print("Classification Report:")

    print(
        classification_report(
            final_results["labels"],
            final_results["predictions"],
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
            "split": "dev",
            "accuracy": final_results["accuracy"],
            "precision": final_results["precision"],
            "recall": final_results["recall"],
            "f1": final_results["f1"],
            "roc_auc": final_results["roc_auc"]
        }
    ])

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm = final_results["confusion_matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=["actual_bonafide", "actual_spoof"],
        columns=["predicted_bonafide", "predicted_spoof"]
    )

    cm_df.to_csv(
        CM_FILE
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CNN TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("Model saved to:")
    print(MODEL_FILE)

    print()
    print("Metrics saved to:")
    print(METRICS_FILE)

    print()
    print("Confusion matrix saved to:")
    print(CM_FILE)


if __name__ == "__main__":
    main()