from pathlib import Path
import csv


# ============================================================
# ASVspoof 2019 LA - Metadata Generator
# ============================================================

DATASET_ROOT = Path("dataset/downloads/LA")

PROTOCOL_DIR = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols"

OUTPUT_FILE = Path("dataset/metadata.csv")


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

DATASETS = {
    "train": {
        "protocol": PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_train" / "flac",
    },

    "dev": {
        "protocol": PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac",
    },

    "eval": {
        "protocol": PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_eval" / "flac",
    },
}


def read_protocol(protocol_path, audio_dir, split):

    records = []

    if not protocol_path.exists():
        print(f"ERROR: Protocol file not found:")
        print(protocol_path)
        return records

    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found:")
        print(audio_dir)
        return records

    with protocol_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Expected format:
            # speaker_id file_id source attack_id label
            #
            # Example:
            # LA_0079 LA_T_1138215 - - bonafide

            if len(parts) != 5:
                print(
                    f"WARNING: Unexpected format in "
                    f"{protocol_path.name}, "
                    f"line {line_number}"
                )
                continue

            speaker_id = parts[0]
            file_id = parts[1]
            source = parts[2]
            attack_id = parts[3]
            label = parts[4]

            audio_path = audio_dir / f"{file_id}.flac"

            records.append({
                "file_id": file_id,
                "audio_path": str(audio_path),
                "speaker_id": speaker_id,
                "source": source,
                "attack_id": attack_id,
                "label": label,
                "split": split,
                "exists": audio_path.exists(),
            })

    return records


def main():

    print("=" * 70)
    print("ASVSPOOF 2019 LA - METADATA GENERATION")
    print("=" * 70)

    all_records = []

    for split, config in DATASETS.items():

        print()
        print("-" * 70)
        print(f"PROCESSING: {split.upper()}")
        print("-" * 70)

        records = read_protocol(
            config["protocol"],
            config["audio_dir"],
            split,
        )

        total = len(records)

        existing = sum(
            record["exists"]
            for record in records
        )

        missing = total - existing

        print(f"Protocol records   : {total}")
        print(f"Audio files found  : {existing}")
        print(f"Audio files missing: {missing}")

        all_records.extend(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "file_id",
        "audio_path",
        "speaker_id",
        "source",
        "attack_id",
        "label",
        "split",
        "exists",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)

    print()
    print("=" * 70)
    print("METADATA GENERATION COMPLETE")
    print("=" * 70)

    print()
    print(f"Total records: {len(all_records)}")

    print()
    print("Label distribution:")

    label_counts = {}

    for record in all_records:

        label = record["label"]

        label_counts[label] = (
            label_counts.get(label, 0) + 1
        )

    for label, count in sorted(label_counts.items()):

        print(f"  {label}: {count}")

    print()
    print("Metadata file:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()