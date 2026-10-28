from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PROTOCOL_DIR = (
    ROOT / "dataset" / "downloads" / "LA"
    / "ASVspoof2019_LA_cm_protocols"
)

FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

for split, filename in FILES.items():
    path = PROTOCOL_DIR / filename

    print("\n" + "=" * 60)
    print(f"{split.upper()} ATTACK COVERAGE")
    print("=" * 60)

    if not path.exists():
        print("Protocol not found:", path)
        continue

    # Protocol format:
    # speaker_id file_id placeholder attack_id bonafide/spoof
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=[
            "speaker_id",
            "file_id",
            "placeholder",
            "attack_id",
            "protocol_label",
        ],
    )

    spoof = df[df["protocol_label"] == "spoof"].copy()

    # Genuine samples have attack ID '-'.
    counts = Counter(spoof["attack_id"])

    print("Total samples:", len(df))
    print("Bona fide samples:",
          (df["protocol_label"] == "bonafide").sum())
    print("Spoof samples:", len(spoof))
    print("Attack types:", len(counts))
    print("A17 samples:", counts.get("A17", 0))

    print("\nSpoof samples by attack:")
    print(
        spoof["attack_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )