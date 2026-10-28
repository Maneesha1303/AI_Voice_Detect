from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib


# ============================================================
# ASVspoof 2019 LA - Random Forest Baseline
# ============================================================

METADATA_FILE = Path("dataset/metadata.csv")

MODEL_DIR = Path("models/baseline")
RESULTS_DIR = Path("results/metrics")
CM_DIR = Path("results/confusion_matrices")

SAMPLE_RATE = 16000
DURATION = 4.0

N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 256


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

def extract_features(audio_path):
    """
    Load audio and convert MFCC matrix into a fixed-length
    feature vector.

    40 MFCC coefficients x 4 statistics = 160 features.
    """

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Fixed 4-second length
    target_samples = int(SAMPLE_RATE * DURATION)

    if len(audio) < target_samples:
        audio = np.pad(
            audio,
            (0, target_samples - len(audio)),
            mode="constant"
        )
    else:
        audio = audio[:target_samples]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Statistical summary
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.min(mfcc, axis=1),
        np.max(mfcc, axis=1)
    ])

    return features.astype(np.float32)


# ------------------------------------------------------------
# Load split
# ------------------------------------------------------------

def load_split(df, split_name):

    split_df = df[df["split"] == split_name].copy()

    X = []
    y = []

    total = len(split_df)

    print()
    print("-" * 70)
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print("-" * 70)

    for index, (_, row) in enumerate(split_df.iterrows(), start=1):

        if index % 500 == 0 or index == 1:
            print(f"Processing {index}/{total}")

        try:
            features = extract_features(row["audio_path"])

            X.append(features)

            # bonafide = 0
            # spoof = 1
            label = 0 if row["label"] == "bonafide" else 1

            y.append(label)

        except Exception as e:
            print(
                f"WARNING: Failed to process "
                f"{row['file_id']}: {e}"
            )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print()
    print(f"{split_name.upper()} complete")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape:         {y.shape}")

    return X, y


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_model(model, X, y, split_name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print()
    print("=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["bonafide", "spoof"],
            zero_division=0
        )
    )

    return {
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1]
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("ASVSPOOF 2019 LA - RANDOM FOREST BASELINE")
    print("=" * 70)

    # Create directories
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
    # Metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA_FILE
    )

    print()
    print(f"Total metadata records: {len(df)}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    X_train, y_train = load_split(
        df,
        "train"
    )

    # --------------------------------------------------------
    # Dev
    # --------------------------------------------------------

    X_dev, y_dev = load_split(
        df,
        "dev"
    )

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print("Training complete.")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    train_results = evaluate_model(
        model,
        X_train,
        y_train,
        "train"
    )

    dev_results = evaluate_model(
        model,
        X_dev,
        y_dev,
        "dev"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR /
        "random_forest_baseline.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = pd.DataFrame([
        train_results,
        dev_results
    ])

    results_file = (
        RESULTS_DIR /
        "baseline_results.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = model.feature_importances_

    feature_names = []

    for statistic in [
        "mean",
        "std",
        "min",
        "max"
    ]:
        for i in range(N_MFCC):
            feature_names.append(
                f"mfcc_{i + 1}_{statistic}"
            )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_file = (
        RESULTS_DIR /
        "baseline_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE MODEL COMPLETE")
    print("=" * 70)

    print()
    print("Model saved to:")
    print(model_file)

    print()
    print("Metrics saved to:")
    print(results_file)

    print()
    print("Feature importance saved to:")
    print(importance_file)


if __name__ == "__main__":
    main()