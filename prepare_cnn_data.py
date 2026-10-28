from pathlib import Path

import numpy as np
import pandas as pd
import librosa


# ============================================================
# ASVspoof 2019 LA - CNN Dataset Preparation
# ============================================================

METADATA_FILE = Path("dataset/metadata.csv")

OUTPUT_DIR = Path("dataset/processed/cnn")

SAMPLE_RATE = 16000
DURATION = 4.0

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ------------------------------------------------------------
# Audio + Mel extraction
# ------------------------------------------------------------

def extract_mel(audio_path):
    """
    Load audio and convert it into a fixed-size log-Mel
    spectrogram for CNN training.
    """

    # Load audio
    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize safely
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Fixed 4-second length
    target_samples = int(
        SAMPLE_RATE * DURATION
    )

    if len(audio) < target_samples:

        audio = np.pad(
            audio,
            (0, target_samples - len(audio)),
            mode="constant"
        )

    else:

        audio = audio[:target_samples]

    # Log-Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(np.float32)


# ------------------------------------------------------------
# Process one split
# ------------------------------------------------------------

def process_split(df, split_name):

    split_df = df[
        df["split"] == split_name
    ].copy()

    output_dir = OUTPUT_DIR / split_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print("=" * 70)

    print(
        f"Total files: {len(split_df)}"
    )

    successful = 0
    failed = 0

    labels = []
    file_ids = []

    for index, (_, row) in enumerate(
        split_df.iterrows(),
        start=1
    ):

        if (
            index == 1
            or index % 500 == 0
            or index == len(split_df)
        ):
            print(
                f"Processing "
                f"{index}/{len(split_df)}"
            )

        try:

            mel = extract_mel(
                row["audio_path"]
            )

            output_file = (
                output_dir /
                f"{row['file_id']}.npy"
            )

            np.save(
                output_file,
                mel
            )

            label = (
                0
                if row["label"] == "bonafide"
                else 1
            )

            labels.append(label)
            file_ids.append(row["file_id"])

            successful += 1

        except Exception as e:

            print(
                f"WARNING: Failed "
                f"{row['file_id']}: {e}"
            )

            failed += 1

    # Save labels metadata
    label_df = pd.DataFrame({
        "file_id": file_ids,
        "label": labels
    })

    label_file = (
        output_dir /
        "labels.csv"
    )

    label_df.to_csv(
        label_file,
        index=False
    )

    print()
    print(
        f"{split_name.upper()} COMPLETE"
    )

    print(
        f"Successfully processed: "
        f"{successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Features saved to: "
        f"{output_dir}"
    )

    print(
        f"Labels saved to: "
        f"{label_file}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("ASVSPOOF 2019 LA - CNN DATA PREPARATION")
    print("=" * 70)

    # Load metadata
    df = pd.read_csv(
        METADATA_FILE
    )

    print()
    print(
        f"Total metadata records: "
        f"{len(df)}"
    )

    print()
    print(
        "CNN input configuration:"
    )

    print(
        f"Sample rate : {SAMPLE_RATE}"
    )

    print(
        f"Duration    : {DURATION} seconds"
    )

    print(
        f"Mel bands   : {N_MELS}"
    )

    print(
        f"FFT size    : {N_FFT}"
    )

    print(
        f"Hop length  : {HOP_LENGTH}"
    )

    print()
    print(
        "Expected feature shape:"
    )

    print(
        "(64, approximately 251)"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Prepare only train and dev for now.
    # Eval remains untouched.
    # --------------------------------------------------------

    process_split(
        df,
        "train"
    )

    process_split(
        df,
        "dev"
    )

    print()
    print("=" * 70)
    print("CNN DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()