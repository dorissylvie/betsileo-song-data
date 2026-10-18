"""
Construit un lexique malgache <-> betsileo a partir de data/, le confronte
au corpus (Bible + chansons) et produit des exemples d'entrainement au meme
format que bible/baibolybets.json.

Sources retenues
----------------
  data/swadesh_mg_betsileo_pairs.json   paires malagasy / betsileo
  data/data.txt                          entrees qui ont un champ malagasy
                                         (lu en mode tolerant : le fichier
                                         n'est pas du JSON valide)

Sources ignorees
----------------
  asjp_* et serva_* : anglais <-> betsileo uniquement, et transcription en
  -tse (efatse, hoditse) que ni la Bible (0 mot en -tse, 3733 en -tsa) ni
  les chansons n'utilisent.

Statut de chaque paire
----------------------
  coherent  le mot malgache est dans la Bible et la forme betsileo du
            lexique apparait dans au moins 30 % des versets correspondants
  nouveau   le mot malgache n'apparait pas dans la Bible : vocabulaire neuf
  conflit   le mot est dans la Bible mais celle-ci utilise une autre forme
            (ex. izy -> iy et non i). Exclu par defaut : a trancher a la main
  a_verifier  entree suspecte dans data.txt (meme mot malgache pour des
            concepts differents, cle en double). Toujours exclue

Sorties (dans data/)
--------------------
  lexique_mg_bts.tsv       a relire : statut, frequences, forme suggeree
  lexique_examples.json    exemples a ajouter au TRAIN uniquement

Usage
-----
  python data/build_lexicon.py
  python data/build_lexicon.py --keep-conflicts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
import difflib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "bible", "baibolybets.json")
SWADESH = os.path.join(ROOT, "data", "swadesh_mg_betsileo_pairs.json")
DATA_TXT = os.path.join(ROOT, "data", "data.txt")
OUT_TSV = os.path.join(ROOT, "data", "lexique_mg_bts.tsv")
OUT_EXAMPLES = os.path.join(ROOT, "data", "lexique_examples.json")

MG2BTS = "Malagasy->Betsileo"
BTS2MG = "Betsileo->Malagasy"
COHERENCE_MIN = 0.30

WORD_RE = re.compile(r"[\w’'-]+", re.UNICODE)
KV_RE = re.compile(r'"(\w+)"\s*[:,]\s*"([^"]*)"')


def words(s: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in WORD_RE.findall(s)]


def contains(seq: list[str], phrase: str) -> bool:
    p = words(phrase)
    return any(seq[i:i + len(p)] == p for i in range(len(seq) - len(p) + 1))


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower().replace("’", "'")


# ---------------------------------------------------------------------------
# Lecture des sources
# ---------------------------------------------------------------------------
def load_swadesh() -> list[dict]:
    rows = json.load(open(SWADESH, encoding="utf-8"))
    return [{"mg": clean(r["malagasy"]), "bts": clean(r["betsileo"]),
             "source": "swadesh", "en": ""} for r in rows]


def load_data_txt() -> tuple[list[dict], list[str]]:
    """Lecture tolerante : extrait les cles/valeurs de chaque {...}."""
    entries, problems = [], []
    text = open(DATA_TXT, encoding="utf-8").read()
    for n_obj, obj in enumerate(re.findall(r"\{[^{}]*\}", text), 1):
        pairs = KV_RE.findall(obj)
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            problems.append(f"cle en double : {obj.strip()}")
            continue
        d = dict(pairs)
        if d.get("malagasy") and d.get("betsileo"):
            entries.append({"mg": clean(d["malagasy"]), "bts": clean(d["betsileo"]),
                            "source": "data.txt", "en": d.get("en", "")})

    # Meme mot malgache pour des concepts differents avec des formes
    # betsileo differentes : souvent un copier-coller (BITE -> ankizy) ou un
    # mot provisoire (eto pour LIVER, ROUND...).
    by_mg = defaultdict(set)
    for e in entries:
        by_mg[e["mg"]].add((e["en"], e["bts"]))
    for e in entries:
        if len({bts for _, bts in by_mg[e["mg"]]}) > 1:
            e["suspect"] = True
            problems.append(f"'{e['mg']}' utilise pour {e['en']} -> {e['bts']} "
                            f"(et {len(by_mg[e['mg']]) - 1} autre(s) concept(s))")
    return entries, problems


def load_corpus():
    rows = json.load(open(CORPUS, encoding="utf-8"))
    prefix = {r["direction"]: r["source"].split(": ", 1)[0] + ": " for r in rows}
    bible = [(words(r["source"].split(": ", 1)[1]), words(r["target"]))
             for r in rows if r["direction"] == MG2BTS]
    apostrophe = ("’" if sum(r["target"].count("’") for r in rows)
                  >= sum(r["target"].count("'") for r in rows) else "'")
    songs = []
    for p in glob.glob(os.path.join(ROOT, "*", "*.json")):
        if os.path.basename(os.path.dirname(p)) in ("bible", "data", "corpus_chansons"):
            continue
        songs += [words(x["bts"]) for x in json.load(open(p, encoding="utf-8"))
                  if x.get("bts")]
    return bible, songs, prefix, apostrophe


def aligned_forms(bible) -> dict[str, Counter]:
    """Mot malgache -> formes betsileo alignees (substitutions 1-1 et mots
    inchanges). Sert a suggerer la forme que la Bible utilise."""
    forms: dict[str, Counter] = defaultdict(Counter)
    for a, b in bible:
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace") and (i2 - i1) == (j2 - j1):
                for k in range(i2 - i1):
                    forms[a[i1 + k]][b[j1 + k]] += 1
    return forms


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-conflicts", action="store_true",
                    help="garde aussi les paires en conflit avec la Bible")
    args = ap.parse_args()

    bible, songs, prefix, apostrophe = load_corpus()
    bts_bible = Counter(w for _, b in bible for w in b)
    bts_songs = Counter(w for s in songs for w in s)
    forms = aligned_forms(bible)

    data_entries, problems = load_data_txt()
    seen, lexicon = set(), []
    for e in load_swadesh() + data_entries:
        key = (e["mg"], e["bts"])
        if key in seen:
            continue
        seen.add(key)
        lexicon.append(e)

    rows, keep = [], []
    for e in lexicon:
        mg, bts = e["mg"], e["bts"]
        ctx = [b for m, b in bible if contains(m, mg)]
        rate = sum(contains(b, bts) for b in ctx) / len(ctx) if ctx else 0.0
        if e.get("suspect"):
            status = "a_verifier"
        elif not ctx:
            status = "nouveau"
        elif rate >= COHERENCE_MIN:
            status = "coherent"
        else:
            status = "conflit"
        suggestion = ""
        if " " not in mg and forms.get(mg):
            suggestion = ", ".join(f"{w} ({n})" for w, n in forms[mg].most_common(3))
        freq = lambda c: sum(c[w] for w in words(bts)) // max(1, len(words(bts)))
        rows.append([status, mg, bts, e["source"], e["en"], len(ctx),
                     f"{100 * rate:.0f}", freq(bts_bible), freq(bts_songs), suggestion])
        if status in ("coherent", "nouveau") or (status == "conflit" and args.keep_conflicts):
            keep.append(e)

    order = {"conflit": 0, "a_verifier": 1, "nouveau": 2, "coherent": 3}
    rows.sort(key=lambda r: (order[r[0]], -r[5]))
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("statut\tmalagasy\tbetsileo\tsource\ten\tversets_bible\t"
                "trouve_pct\tfreq_bts_bible\tfreq_bts_chansons\tforme_bible_suggeree\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    # Meme prefixe et meme apostrophe que le corpus, sinon le modele voit
    # une tache differente et deux orthographes pour amin'ny.
    examples = []
    for e in keep:
        mg, bts = e["mg"].replace("'", apostrophe), e["bts"].replace("'", apostrophe)
        ref = f"LEX {e['source']}:{e['mg']}"
        examples.append({"source": prefix[MG2BTS] + mg, "target": bts,
                         "direction": MG2BTS, "ref": ref})
        examples.append({"source": prefix[BTS2MG] + bts, "target": mg,
                         "direction": BTS2MG, "ref": ref})
    with open(OUT_EXAMPLES, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    counts = Counter(r[0] for r in rows)
    print("=" * 62)
    print("LEXIQUE malgache <-> betsileo")
    print("=" * 62)
    print(f"  paires uniques : {len(lexicon)}")
    for s in ("coherent", "nouveau", "conflit", "a_verifier"):
        print(f"    {s:<11} {counts[s]:>4}")
    print(f"\n  exemples ecrits : {len(examples)} "
          f"({100 * len(examples) / (2 * len(bible)):.1f}% du corpus de "
          f"{2 * len(bible)} exemples)")
    print(f"  prefixes repris du corpus : {sorted(prefix.values())}")
    if problems:
        print(f"\n  data.txt - {len(problems)} entree(s) a corriger :")
        for p in problems:
            print(f"    - {p}")
    print("\n  Conflits les plus frequents (lexique vs Bible) :")
    for r in [r for r in rows if r[0] == "conflit"][:12]:
        print(f"    {r[1]:<14} lexique={r[2]:<14} Bible -> {r[9] or '?'}")
    print(f"\n  Ecrit : {os.path.relpath(OUT_TSV, ROOT)}")
    print(f"  Ecrit : {os.path.relpath(OUT_EXAMPLES, ROOT)}")


if __name__ == "__main__":
    main()
