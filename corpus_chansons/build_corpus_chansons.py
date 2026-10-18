"""
Construit le corpus parallele betsileo -> malgache officiel des chansons.

Sources
-------
  */*.json (Majeur 7, oladad, tsivahiny, ventsonFianar)
      {"bts", "mlg"} : un "mlg" deja rempli est TA traduction, il est
      garde tel quel (traducteur = "utilisateur").
  corpus_chansons/traductions_claude.tsv
      traductions proposees pour les vers vides et pour autres.txt
      (fichier, ligne, verifier, bts, mlg). A relire : corrige la colonne
      mlg et remets verifier a 0 une fois verifie. Un mlg vide = vers ignore
      (interjection).

Sorties (dans corpus_chansons/)
-------------------------------
  corpus_chansons.json
      [{"bts", "mlg", "chanson", "groupe", "ref", "traducteur", "verifier"}]
      sans doublons (refrains), sans interjections.

Le champ "chanson" sert a separer train / test PAR CHANSON : un refrain
present dans les deux fausserait l'evaluation.

Usage
-----
  python corpus_chansons/build_corpus_chansons.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TSV = os.path.join(HERE, "traductions_claude.tsv")
OUT = os.path.join(HERE, "corpus_chansons.json")
SKIP_DIRS = {"bible", "data", "kabary", "corpus_chansons"}


def main():
    claude = {}
    with open(TSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            claude[(r["fichier"], int(r["ligne"]))] = r

    pairs, stale = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))):
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        group = rel.split("/")[0]
        if group in SKIP_DIRS:
            continue
        song = os.path.splitext(os.path.basename(rel))[0]
        for i, x in enumerate(json.load(open(p, encoding="utf-8"))):
            bts = x.get("bts", "").strip()
            if x.get("mlg", "").strip():
                mlg, who, check = x["mlg"].strip(), "utilisateur", 0
            elif (rel, i) in claude:
                r = claude[(rel, i)]
                if r["bts"].strip() != bts:
                    stale.append(f"{rel}:{i}")  # le vers a change depuis la traduction
                    continue
                mlg, who, check = r["mlg"].strip(), "claude", int(r["verifier"])
            else:
                continue
            pairs.append({"bts": bts, "mlg": mlg, "chanson": song, "groupe": group,
                          "ref": f"{rel}:{i}", "traducteur": who, "verifier": check})

    for (rel, i), r in sorted(claude.items()):
        if rel == "autres.txt":
            pairs.append({"bts": r["bts"].strip(), "mlg": r["mlg"].strip(),
                          "chanson": "autres", "groupe": "dialogue",
                          "ref": f"autres.txt:{i}", "traducteur": "claude",
                          "verifier": int(r["verifier"])})

    seen, corpus = set(), []
    for x in pairs:
        # meme apostrophe que la Bible (’), sinon deux orthographes de amin’ny
        x["bts"], x["mlg"] = x["bts"].replace("'", "’"), x["mlg"].replace("'", "’")
        key = (x["bts"].lower(), x["mlg"].lower())
        if not x["bts"] or not x["mlg"] or key in seen:
            continue
        seen.add(key)
        corpus.append(x)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("=" * 62)
    print("CORPUS CHANSONS betsileo -> malgache officiel")
    print("=" * 62)
    print(f"  vers lus        : {len(pairs)}")
    print(f"  paires uniques  : {len(corpus)}  (doublons/interjections retires)")
    for who, n in Counter(x["traducteur"] for x in corpus).items():
        print(f"    {who:<12} {n:>4}")
    print(f"  a verifier      : {sum(x['verifier'] for x in corpus)}")
    print("\n  Par groupe :")
    for g, n in sorted(Counter(x["groupe"] for x in corpus).items()):
        print(f"    {g:<14} {n:>4} paires, "
              f"{len({x['chanson'] for x in corpus if x['groupe'] == g})} chanson(s)")
    if stale:
        print(f"\n  ATTENTION : {len(stale)} vers modifies depuis la traduction "
              f"(ignores) : {', '.join(stale[:10])}")
    print(f"\n  Ecrit : {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
