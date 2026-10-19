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

  corpus_chansons/exclusions.tsv
      vers retires a la main (ref, raison) : lieux, noms propres, argot,
      mots rares, mots du nord... Ajoute ou retire des lignes librement.

Filtres (dans cet ordre)
------------------------
  religieux    chansons tirees des Psaumes (CHANSONS_EXCLUES)
  a_verifier   traduction incertaine : revient des que verifier = 0
  identique    bts = mlg une fois retires accents, ponctuation et
               interjections (e, eh, a, an, o...) : rien a apprendre
  manuel       ref presente dans exclusions.tsv
  doublon      meme paire deja gardee (refrains)

Sorties (dans corpus_chansons/)
-------------------------------
  corpus_chansons.json
      [{"bts", "mlg", "chanson", "groupe", "ref", "traducteur", "verifier"}]
      paires gardees : vie quotidienne, vocabulaire courant.
  corpus_chansons_exclus.tsv
      paires retirees a corriger ou a controler (raison, ref, bts, mlg,
      valide), une seule fois par vers. Les identiques et les doublons de
      refrain n'y figurent pas : il n'y a rien a y corriger.

Corriger dans corpus_chansons_exclus.tsv
----------------------------------------
  Pour une traduction de Claude (raison a_verifier, mais aussi les autres) :
    - corrige la colonne mlg, ou
    - mets 1 dans la colonne valide si la traduction est deja bonne.
  Au lancement suivant, la correction est reportee dans
  traductions_claude.tsv (verifier = 0, et pour toutes les repetitions du
  vers dans la chanson) : un vers a_verifier passe alors dans le corpus.
  Tes propres traductions se corrigent dans les fichiers de chansons.

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
import re
import sys
import unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TSV = os.path.join(HERE, "traductions_claude.tsv")
EXCLUSIONS = os.path.join(HERE, "exclusions.tsv")
OUT = os.path.join(HERE, "corpus_chansons.json")
OUT_EXCLUS = os.path.join(HERE, "corpus_chansons_exclus.tsv")
SKIP_DIRS = {"bible", "data", "kabary", "corpus_chansons"}

CHANSONS_EXCLUES = {
    "Aia e ro tany tsa ahitanao ahy": "religieux (Psaume 139)",
    "Arakandron'Andriana": "religieux (Psaume 23)",
}
INTERJECTIONS = {"e", "eh", "ee", "eee", "a", "an", "o", "oh", "ohh", "ie", "iei",
                 "he", "aaa", "aaaa", "ann", "annh", "hann", "liaka"}


APOSTROPHES = str.maketrans({"'": "’", "‘": "’", "`": "’", "´": "’", "ʼ": "’"})


def clean_text(s: str) -> str:
    """Forme uniforme des deux cotes : minuscules, apostrophe ’ (comme la
    Bible), pas de guillemets, ponctuation collee au mot et suivie d'un
    espace, pas de ponctuation repetee ni de point/virgule en fin de vers."""
    s = unicodedata.normalize("NFC", s).translate(APOSTROPHES).lower()
    s = re.sub(r"[\"“”«»]", "", s).replace("…", "...")
    s = re.sub(r"\([^)]*\)", " ", s)                          # didascalies, gloses
    s = re.sub(r",\s*\.+", ",", s)                            # "haria,..ka"
    # "!!!" -> "!", "an..!" -> "an!", ".." -> "." (garde "...")
    s = re.sub(r"[.!?]{2,}", lambda m: next((c for c in m.group() if c in "!?"),
               "..." if len(m.group()) >= 3 else "."), s)
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)                    # "mitomany ?"
    s = re.sub(r"([,;:!?])(?=[^\s\d,.!?;:)])", r"\1 ", s)    # "mitombo,fa"
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^[,;:.\s]+|[,;:\s]+$|(?<!\.)\.$", "", s).strip()


def align_end(bts: str, mlg: str) -> tuple[str, str]:
    """Meme ponctuation finale (! ou ?) des deux cotes."""
    eb = bts[-1] if bts[-1:] in ("!", "?") else ""
    em = mlg[-1] if mlg[-1:] in ("!", "?") else ""
    if eb and not em:
        mlg += eb
    elif em and not eb:
        bts += em
    return bts, mlg


def key(s: str) -> tuple[str, ...]:
    """Forme de comparaison : sans accents, ponctuation ni interjections."""
    s = unicodedata.normalize("NFD", s.lower().replace("’", "'"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return tuple(w for w in re.findall(r"[a-z'-]+", s) if w not in INTERJECTIONS)


def import_corrections(claude: dict) -> tuple[int, int]:
    """Reporte dans traductions_claude.tsv les corrections faites dans
    corpus_chansons_exclus.tsv : mlg modifie ou valide = 1 -> verifier = 0.
    Retourne (vers corriges, corrections ignorees car hors traductions_claude)."""
    if not os.path.exists(OUT_EXCLUS):
        return 0, 0
    done, ignored = 0, 0
    with open(OUT_EXCLUS, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rel, _, i = r["ref"].rpartition(":")
            t = claude.get((rel, int(i))) if i.isdigit() else None
            new = (r.get("mlg") or "").strip()
            changed = bool(new) and t is not None and clean_text(new) != clean_text(t["mlg"])
            valid = (r.get("valide") or "").strip().lower() in ("1", "x", "ok", "oui")
            if not (changed or valid):
                continue
            if t is None:
                ignored += 1
                continue
            # meme correction pour les repetitions du vers (refrains)
            for (f2, _), t2 in claude.items():
                if f2 == rel and t2["bts"].strip() == t["bts"].strip():
                    if changed:
                        t2["mlg"] = new
                    t2["verifier"] = "0"
            done += 1
    if done:
        with open(TSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fichier", "ligne", "verifier", "bts", "mlg"],
                               delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(claude.values())
    return done, ignored


def main():
    claude = {}
    with open(TSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            claude[(r["fichier"], int(r["ligne"]))] = r
    imported, ignored = import_corrections(claude)

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

    manual = {}
    with open(EXCLUSIONS, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            manual[r["ref"].strip()] = r["raison"].strip()

    # Une exclusion manuelle vaut pour toutes les repetitions du meme vers
    # (refrains) : sinon la ligne suivante du refrain reviendrait.
    banned: dict[tuple, str] = {}
    seen, corpus, removed = set(), [], []
    for x in pairs:
        if not x["bts"] or not x["mlg"]:
            continue  # interjection seule
        x["bts"], x["mlg"] = align_end(clean_text(x["bts"]), clean_text(x["mlg"]))
        if not x["bts"] or not x["mlg"]:
            continue
        kb, km = key(x["bts"]), key(x["mlg"])
        if x["chanson"] in CHANSONS_EXCLUES:
            reason = CHANSONS_EXCLUES[x["chanson"]]
        elif x["verifier"]:
            reason = "a_verifier"
        elif kb == km:
            reason = "identique"
        elif x["ref"] in manual:
            reason = banned[(kb, km)] = manual[x["ref"]]
        elif (kb, km) in banned:
            reason = banned[(kb, km)]
        elif (kb, km) in seen:
            reason = "doublon"
        else:
            seen.add((kb, km))
            corpus.append(x)
            continue
        removed.append({**x, "raison": reason})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    # Fichier de correction : sans identiques ni doublons, un vers une fois
    to_fix, listed = [], set()
    for x in removed:
        k = (x["raison"], key(x["bts"]), key(x["mlg"]))
        if x["raison"] in ("identique", "doublon") or k in listed:
            continue
        listed.add(k)
        to_fix.append(x)
    with open(OUT_EXCLUS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["raison", "ref", "bts", "mlg", "valide"])
        for x in to_fix:
            w.writerow([x["raison"], x["ref"], x["bts"], x["mlg"], ""])

    def bucket(r: str) -> str:
        return r if r in ("a_verifier", "identique", "doublon") or r.startswith("religieux") else "manuel"

    print("=" * 62)
    print("CORPUS CHANSONS betsileo -> malgache officiel")
    print("=" * 62)
    if imported or ignored:
        print(f"  corrections reprises de {os.path.basename(OUT_EXCLUS)} : {imported}"
              + (f"  ({ignored} ignorees : tes traductions se corrigent dans les "
                 "fichiers de chansons)" if ignored else ""))
    print(f"  vers lus        : {len(pairs)}")
    print(f"  paires gardees  : {len(corpus)}")
    for who, n in Counter(x["traducteur"] for x in corpus).items():
        print(f"    {who:<12} {n:>4}")
    print(f"  paires retirees : {len(removed)}")
    for r, n in Counter(bucket(x["raison"]) for x in removed).most_common():
        print(f"    {r:<24} {n:>4}")
    unused = set(manual) - {x["ref"] for x in removed}
    if unused:
        print(f"  (exclusions.tsv : {len(unused)} ref deja retirees par une autre regle "
              "ou introuvables)")
    print("\n  Par groupe :")
    for g, n in sorted(Counter(x["groupe"] for x in corpus).items()):
        print(f"    {g:<14} {n:>4} paires, "
              f"{len({x['chanson'] for x in corpus if x['groupe'] == g})} chanson(s)")
    if stale:
        print(f"\n  ATTENTION : {len(stale)} vers modifies depuis la traduction "
              f"(ignores) : {', '.join(stale[:10])}")
    print(f"\n  Ecrit : {os.path.relpath(OUT, ROOT)}")
    print(f"  Ecrit : {os.path.relpath(OUT_EXCLUS, ROOT)}  ({len(to_fix)} lignes a corriger)")


if __name__ == "__main__":
    main()
