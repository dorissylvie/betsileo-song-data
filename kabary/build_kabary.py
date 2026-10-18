"""
Transforme la transcription du livre « I Fandriana sy ny kabariny »
(RASOANJANAHARY Honorine, Famoaham-boky HARAL) en phrases betsileo
utilisables pour l'entrainement.

Entree
------
  kabary/kabary_fandriana_ocr.txt
      transcription page par page :
        === p. N ===     debut de la page N du livre
        # ...            commentaire / titre (ignore)
        [LOCUTEUR]       changement d'orateur (MPAMOAKA, MPANGATAKA...)
        (...)            didascalie en malgache officiel (retiree)
        [?]              lecture incertaine (retiree, phrase signalee)

Sorties (dans kabary/)
----------------------
  kabary_fandriana_bts.json
      meme format que les chansons ({"bts", "mlg"}), "mlg" vide a
      remplir (a la main ou par retro-traduction). Seulement les
      repliques d'un orateur (dialogue) classees betsileo : la preface,
      les descriptions et la postface sont exclues, comme les phrases
      mixtes, neutres ou officielles.
  kabary_fandriana_phrases.tsv
      toutes les phrases avec page, orateur, marqueurs, classement et
      colonne "garde", a relire.

Classement dialectal (par phrase)
---------------------------------
  betsileo  plus de marqueurs betsileo (tsa, agny, -gn-, atsika, ro...)
            que de marqueurs officiels (tsy, isika, olona, ianareo...)
  mixte     les deux, mais autant ou plus de marqueurs officiels
  officiel  uniquement des marqueurs officiels
  neutre    aucun marqueur : la phrase s'ecrit comme en malgache officiel

Usage
-----
  python kabary/build_kabary.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252 : ỳ

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "kabary_fandriana_ocr.txt")
OUT_JSON = os.path.join(HERE, "kabary_fandriana_bts.json")
OUT_TSV = os.path.join(HERE, "kabary_fandriana_phrases.tsv")

APOSTROPHE = "’"      # dominante dans la Bible (10309) et les chansons (59)
MAX_CHARS = 250       # au-dela, la phrase est recoupee (p99 du corpus ~260 octets)
MIN_WORDS = 4         # fragment plus court -> fusionne avec la phrase suivante

PAGE_RE = re.compile(r"^=== p\. (\d+) ===$")
SPEAKER_RE = re.compile(r"^\[([^\]?]+)\]$")
WORD_RE = re.compile(r"[\wỳ’'-]+")
SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“«A-ZÀ-Ý])")

# Marqueurs lexicaux, verifies sur les paires de la Bible (cote betsileo /
# cote officiel) : tsa 2079/0, ro 1475/0, ko 982/0, aby 682/0, ona 863/0, ampela 88/0
# contre tsy 87/2089, isika 2/168, olona 1/577, ianareo 0/552.
# hoe (755/687), raha (488/720) et anefa (28/35) servent dans les deux
# varietes : ce ne sont pas des marqueurs.
# Le digramme "gn" (nasale velaire) est le plus fiable : agnareo, tragno,
# managna... n'existent pas en malgache officiel.
BTS_WORDS = {
    "tsa", "atsika", "antsika", "agnareo", "agnay", "agnao", "agny", "tagny",
    "igny", "iraigny", "ahatagny", "atỳ", "etỳ", "tarỳ", "tatỳ", "indrỳ",
    "reha", "hoa", "ro", "ko", "aby", "ngeza", "ona", "aia", "naky",
    "fianakaviena", "ampela", "olombelogna", "niany",
}
OFF_WORDS = {
    "tsy", "isika", "any", "olona", "trano", "ianareo", "fianakaviana", "manana",
}


def norm_word(w: str) -> str:
    return w.lower().replace("'", APOSTROPHE)


def markers(sentence: str) -> tuple[list[str], list[str]]:
    words = [norm_word(w) for w in WORD_RE.findall(sentence)]
    bts = [w for w in words if w in BTS_WORDS or ("gn" in w and w not in OFF_WORDS)]
    off = [w for w in words if w in OFF_WORDS]
    return bts, off


def classify(bts: list[str], off: list[str]) -> str:
    if bts and len(bts) > len(off):
        return "betsileo"
    if bts:
        return "mixte"
    if off:
        return "officiel"
    return "neutre"


# ---------------------------------------------------------------------------
# 1. Lecture : paragraphes avec page et orateur, recolles entre deux pages
# ---------------------------------------------------------------------------
def read_paragraphs(path: str) -> list[dict]:
    paras: list[dict] = []
    page, speaker, new_page = 0, "", False
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if line.startswith("##"):
            # Titre de partie : l'orateur precedent ne parle plus (sinon la
            # postface FEHINTENY serait attribuee au dernier orateur).
            speaker = ""
            paras.append({"page": page, "speaker": "", "text": "", "break": True})
            continue
        if not line or line.startswith("#"):
            continue
        m = PAGE_RE.match(line)
        if m:
            page, new_page = int(m.group(1)), True
            continue
        m = SPEAKER_RE.match(line)
        if m:
            speaker, new_page = m.group(1).title(), False
            paras.append({"page": page, "speaker": speaker, "text": "", "break": True})
            continue
        if line.startswith("(") and line.endswith(")"):
            continue  # didascalie seule sur sa ligne

        prev = paras[-1] if paras else None
        # Un paragraphe coupe en bas de page se poursuit en haut de la
        # page suivante : pas de ponctuation finale dans la page precedente.
        if (new_page and prev and prev["text"] and not prev.get("break")
                and not re.search(r"[.!?…:\"”»)]$", prev["text"])):
            sep = "" if prev["text"].endswith("-") else " "
            prev["text"] += sep + line
        else:
            paras.append({"page": page, "speaker": speaker, "text": line})
        new_page = False
    return [p for p in paras if p["text"]]


# ---------------------------------------------------------------------------
# 2. Nettoyage et decoupage en phrases
# ---------------------------------------------------------------------------
def clean(text: str) -> str:
    """Les marques [?] sont gardees ici et retirees phrase par phrase."""
    text = re.sub(r"\([^)]*\)", " ", text)            # didascalies en ligne
    text = re.sub(r"[\"“”«»]", "", text)               # guillemets de citation
    text = text.replace("'", APOSTROPHE)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)       # "teny ." -> "teny."
    text = re.sub(r"([.!?])\1+(?!\.)", r"\1", text)    # ".." -> "." (garde "...")
    return re.sub(r"\s+", " ", text).strip(" ,;-–")


def split_long(s: str) -> list[str]:
    """Recoupe une phrase trop longue sur ; ou : puis sur la virgule la plus
    proche du milieu, recursivement."""
    if len(s) <= MAX_CHARS:
        return [s]
    for sep in (";", ":"):
        parts = [p.strip() for p in s.split(sep) if p.strip()]
        if len(parts) > 1:
            return [x for p in parts for x in split_long(p)]
    commas = [m.end() for m in re.finditer(r",\s", s)]
    if not commas:
        return [s]
    cut = min(commas, key=lambda i: abs(i - len(s) / 2))
    return split_long(s[:cut].strip(" ,")) + split_long(s[cut:].strip())


def sentences(text: str) -> list[str]:
    out, buf = [], ""
    for s in SENT_END_RE.split(text):
        buf = f"{buf} {s}".strip() if buf else s
        if len(WORD_RE.findall(buf)) >= MIN_WORDS:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(WORD_RE.findall(buf)) < MIN_WORDS:
            out[-1] += " " + buf
        else:
            out.append(buf)
    return [x for s in out for x in split_long(s)]


# ---------------------------------------------------------------------------
def main():
    rows, seen = [], set()
    for p in read_paragraphs(SRC):
        for s in sentences(clean(p["text"])):
            uncertain = "[?]" in s
            s = re.sub(r"\s+", " ", s.replace("[?]", "")).strip()
            key = s.lower()
            if key in seen or len(WORD_RE.findall(s)) < 2:
                continue
            seen.add(key)
            bts, off = markers(s)
            dialect = classify(bts, off)
            rows.append({
                "ref": f"KABARY p.{p['page']}",
                "page": p["page"],
                "locuteur": p["speaker"],
                "dialecte": dialect,
                "garde": int(bool(p["speaker"]) and dialect == "betsileo"),
                "marqueurs_bts": " ".join(sorted(set(bts))),
                "marqueurs_off": " ".join(sorted(set(off))),
                "incertain": int(uncertain),
                "phrase": s,
            })

    keep = [r for r in rows if r["garde"]]
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump([{"bts": r["phrase"], "mlg": "", "ref": r["ref"],
                    "dialecte": r["dialecte"]} for r in keep],
                  f, ensure_ascii=False, indent=2)
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["dialecte"] for r in rows)
    lens = sorted(len(r["phrase"].encode("utf-8")) for r in keep)
    bts_vocab = Counter(norm_word(w) for r in keep for w in WORD_RE.findall(r["phrase"]))
    print("=" * 62)
    print("KABARY BETSILEO (Fandriana)")
    print("=" * 62)
    print(f"  phrases       : {len(rows)}  (dont {sum(bool(r['locuteur']) for r in rows)}"
          " dites par un orateur)")
    for d in ("betsileo", "mixte", "neutre", "officiel"):
        print(f"    {d:<10} {counts[d]:>4}")
    print(f"  gardees (JSON): {len(keep)} repliques betsileo  ({sum(lens)} octets, "
          f"{len(bts_vocab)} formes distinctes)")
    print(f"  longueur octets : mediane {lens[len(lens) // 2]}, max {lens[-1]}")
    print(f"  incertaines   : {sum(r['incertain'] for r in rows)}")
    print("\n  Marqueurs betsileo les plus frequents :")
    m = Counter(w for r in keep for w in r["marqueurs_bts"].split())
    print("    " + ", ".join(f"{w} ({n})" for w, n in m.most_common(15)))
    print(f"\n  Ecrit : {os.path.relpath(OUT_JSON, os.path.dirname(HERE))}")
    print(f"  Ecrit : {os.path.relpath(OUT_TSV, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
