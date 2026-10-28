from pathlib import Path

import numpy as np
import pandas as pd
import librosa


# ============================================================
# ASVspoof 2019 LA - Prepare EVAL CNN Features
# ============================================================

METADATA_FILE = Path("dataset/metadata.csv")

OUTPUT_DIR = Path(
    "dataset/processed/cnn/eval"
)

SAMPLE_RATE = 16000
DURATION = 4.0

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# Audio loading
# ============================================================

def load_audio(
    audio_path,
    sample_rate=16000,
    duration=4.0
):

    target_samples = int(
        sample_rate * duration
    )

    audio, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True
    )

    audio = audio - np.mean(audio)

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = (
            audio / max_value
        )

    if len(audio) < target_samples:

        padding = (
            target_samples -
            len(audio)
        )

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    else:

        audio = audio[
            :target_samples
        ]

    return audio.astype(
        np.float32
    )


# ============================================================
# Mel spectrogram extraction
# ============================================================

def extract_mel_spectrogram(
    audio,
    sample_rate=16000
):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(
        np.float32
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ASVSPOOF 2019 LA - PREPARE EVAL CNN FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print()
    print("Loading metadata...")

    df = pd.read_csv(
        METADATA_FILE
    )

    eval_df = df[
        df["split"] == "eval"
    ].copy()

    print(
        "Evaluation records:",
        len(eval_df)
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING EVAL SPLIT")
    print("=" * 70)

    total = len(eval_df)

    successful = 0
    failed = 0

    labels = []

    for index, row in enumerate(
        eval_df.itertuples(index=False),
        start=1
    ):

        try:

            audio = load_audio(
                row.audio_path,
                sample_rate=SAMPLE_RATE,
                duration=DURATION
            )

            mel = extract_mel_spectrogram(
                audio,
                sample_rate=SAMPLE_RATE
            )

            output_file = (
                OUTPUT_DIR /
                f"{row.file_id}.npy"
            )

            np.save(
                output_file,
                mel
            )

            labels.append({
                "file_id": row.file_id,
                "label": 1 if row.label == "spoof" else 0
            })

            successful += 1

        except Exception as error:

            failed += 1

            print()
            print(
                f"ERROR: {row.file_id}"
            )

            print(
                error
            )

        if (
            index % 500 == 0
            or index == total
        ):

            print(
                f"Processing "
                f"{index}/{total}"
            )

    # --------------------------------------------------------
    # Save labels
    # --------------------------------------------------------

    labels_df = pd.DataFrame(
        labels
    )

    labels_file = (
        OUTPUT_DIR /
        "labels.csv"
    )

    labels_df.to_csv(
        labels_file,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVAL COMPLETE")
    print("=" * 70)

    print(
        "Successfully processed:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print()
    print("Features saved to:")
    print(OUTPUT_DIR)

    print()
    print("Labels saved to:")
    print(labels_file)

    print()
    print("=" * 70)
    print("EVAL CNN FEATURE PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()