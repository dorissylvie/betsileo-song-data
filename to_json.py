from pathlib import Path

import pandas as pd

# Les fichiers parquet sont dans le même dossier que ce script
DOSSIER = Path(__file__).resolve().parent

# Colonnes inutiles : "book" est déjà dans "ref", "identical" vaut toujours false
COLONNES_A_SUPPRIMER = ["book", "identical"]

for parquet in sorted(DOSSIER.glob("*.parquet")):
    df = pd.read_parquet(parquet).drop(columns=COLONNES_A_SUPPRIMER)
    sortie = parquet.with_suffix(".json")
    df.to_json(sortie, orient="records", force_ascii=False, indent=2)
    print(f"{parquet.name} -> {sortie.name} ({len(df)} lignes)")
