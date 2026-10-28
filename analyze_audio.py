from pathlib import Path
import pandas as pd
import librosa
import numpy as np


# ============================================================
# ASVspoof 2019 LA - Audio Analysis
# ============================================================

METADATA_FILE = Path("dataset/metadata.csv")

SAMPLES_PER_GROUP = 5


def analyze_file(audio_path):
    """
    Load one audio file and return basic audio properties.
    """

    try:
        audio, sample_rate = librosa.load(
            audio_path,
            sr=None,
            mono=True
        )

        duration = len(audio) / sample_rate

        return {
            "success": True,
            "sample_rate": sample_rate,
            "samples": len(audio),
            "duration": duration,
            "min_amplitude": float(np.min(audio)),
            "max_amplitude": float(np.max(audio)),
            "rms": float(np.sqrt(np.mean(audio ** 2))),
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


def main():

    print("=" * 70)
    print("ASVSPOOF 2019 LA - AUDIO ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_FILE)

    print()
    print(f"Total metadata records: {len(df)}")

    # --------------------------------------------------------
    # Select representative samples
    # --------------------------------------------------------

    samples = (
        df.groupby(
            ["split", "label"],
            group_keys=False
        )
        .head(SAMPLES_PER_GROUP)
    )

    print()
    print(
        f"Analyzing {len(samples)} representative audio files..."
    )

    results = []

    # --------------------------------------------------------
    # Analyze files
    # --------------------------------------------------------

    for index, row in samples.iterrows():

        print(
            f"[{len(results) + 1}/{len(samples)}] "
            f"{row['split']} | "
            f"{row['label']} | "
            f"{row['file_id']}"
        )

        info = analyze_file(row["audio_path"])

        result = {
            "file_id": row["file_id"],
            "split": row["split"],
            "label": row["label"],
            "speaker_id": row["speaker_id"],
        }

        result.update(info)

        results.append(result)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    successful = results_df["success"].sum()
    failed = len(results_df) - successful

    print()
    print(f"Successfully loaded: {successful}")
    print(f"Failed to load      : {failed}")

    if successful > 0:

        successful_df = results_df[
            results_df["success"] == True
        ]

        print()
        print("Sample rate:")
        print(
            successful_df["sample_rate"]
            .value_counts()
            .sort_index()
        )

        print()
        print("Duration statistics (seconds):")

        print(
            successful_df["duration"]
            .describe()
        )

        print()
        print("Amplitude statistics:")

        print(
            successful_df[
                [
                    "min_amplitude",
                    "max_amplitude",
                    "rms"
                ]
            ].describe()
        )

    if failed > 0:

        print()
        print("FAILED FILES:")

        failed_df = results_df[
            results_df["success"] == False
        ]

        for _, row in failed_df.iterrows():

            print(
                row["file_id"],
                "->",
                row["error"]
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = Path(
        "results/metrics/audio_sample_analysis.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 70)
    print("AUDIO ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(f"Results saved to:")
    print(output_file)


if __name__ == "__main__":
    main()