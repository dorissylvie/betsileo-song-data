from pathlib import Path

import pandas as pd

# Les fichiers parquet sont dans le dossier parent (song-bible/)
DOSSIER = Path(__file__).resolve().parent.parent

for parquet in sorted(DOSSIER.glob("*.parquet")):
    df = pd.read_parquet(parquet)
    sortie = parquet.with_suffix(".json")
    df.to_json(sortie, orient="records", force_ascii=False, indent=2)
    print(f"{parquet.name} -> {sortie.name} ({len(df)} lignes)")
