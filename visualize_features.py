from pathlib import Path

import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

from preprocessing import (
    load_audio,
    extract_mfcc,
    extract_mel_spectrogram
)


# ============================================================
# ASVspoof 2019 LA - Feature Visualization
# ============================================================

METADATA_FILE = Path("dataset/metadata.csv")

OUTPUT_FILE = Path(
    "results/graphs/sample_feature_visualization.png"
)

SAMPLE_RATE = 16000
DURATION = 4.0


def main():

    print("=" * 70)
    print("ASVSPOOF 2019 LA - FEATURE VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_FILE)

    # First bonafide training example
    row = df[
        (df["split"] == "train") &
        (df["label"] == "bonafide")
    ].iloc[0]

    print()
    print("File:", row["file_id"])
    print("Label:", row["label"])
    print("Audio:", row["audio_path"])

    # --------------------------------------------------------
    # Load and preprocess audio
    # --------------------------------------------------------

    audio = load_audio(
        row["audio_path"],
        sample_rate=SAMPLE_RATE,
        duration=DURATION
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    mfcc = extract_mfcc(
        audio,
        sample_rate=SAMPLE_RATE
    )

    mel = extract_mel_spectrogram(
        audio,
        sample_rate=SAMPLE_RATE
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 12)
    )

    # --------------------------------------------------------
    # Waveform
    # --------------------------------------------------------

    librosa.display.waveshow(
        audio,
        sr=SAMPLE_RATE,
        ax=axes[0]
    )

    axes[0].set_title(
        f"Waveform - {row['label'].upper()}"
    )

    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    img1 = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=SAMPLE_RATE,
        hop_length=256,
        ax=axes[1]
    )

    axes[1].set_title("MFCC")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("MFCC Coefficient")

    fig.colorbar(
        img1,
        ax=axes[1],
        format="%+2.0f"
    )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    img2 = librosa.display.specshow(
        mel,
        x_axis="time",
        y_axis="mel",
        sr=SAMPLE_RATE,
        hop_length=256,
        ax=axes[2]
    )

    axes[2].set_title(
        "Log-Mel Spectrogram"
    )

    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Frequency (Hz)")

    fig.colorbar(
        img2,
        ax=axes[2],
        format="%+2.0f dB"
    )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        "ASVspoof 2019 LA - Audio Feature Analysis",
        fontsize=16
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = OUTPUT_FILE.resolve()

    print()
    print("Saving visualization to:")
    print(output_path)

    fig.savefig(
        str(output_path),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print()
    print("Saved to:")
    print(output_path)


if __name__ == "__main__":
    main()