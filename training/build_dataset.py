"""
Construit le dataset Betsileo <-> malgache officiel (v2) pour l'entrainement.

Sources (4 fichiers)
--------------------
  bible/baibolybets.json                -> train
  data/lexique_examples.json            -> train (15 % des mots mis de cote pour le test)
  bible/validation-00000-of-00001.json  -> validation
  bible/test-00000-of-00001.json        -> test

Les fichiers sources ne sont JAMAIS modifies : tout le nettoyage se fait ici,
de facon reproductible. Relance le script apres chaque correction a la main.

Etapes (les memes regles pour train, validation et test, sinon le score du
test ne veut rien dire)
----------------------------------------------------------------------------
  1. Normalisation : apostrophes ’ -> ', guillemets -> ", renvois de notes
     (« denaria b zato »), asterisques, crochets, references bibliques
     « (Isa 56.7) », espaces avant la ponctuation.
  2. Les remplacements deja appliques au train (commit 9d5e487 « remplacement
     mot ») sont appliques aussi a la validation et au test : pronoms divins
     en minuscule, finoana -> fahatokiana, mandrakizay -> mandrakariva,
     an-danitra -> ambony, noms bibliques -> noms malgaches, etc.
  3. Suppression des versets encore trop religieux (Dieu, Jesus, Esprit,
     Satan, temple, pretres, resurrection, noms bibliques sans equivalent...).
  4. Suppression des paires mal alignees : une cible tronquee apprend au
     modele a couper des morceaux ou a en inventer (= hallucination).
  5. Train uniquement : retrait des fuites (phrase deja en validation/test)
     et des doublons.
  6. Train uniquement : ajout de segments courts (propositions alignees des
     versets) pour que le modele sache aussi traduire des phrases courtes.

Sorties (training/dataset/)
---------------------------
  train.jsonl, validation.jsonl, test.jsonl   le dataset
  README.md                                   carte du dataset (Hugging Face)
  rapport_nettoyage.md                        chiffres + decisions
  retires.tsv                                 chaque verset retire et pourquoi
  modifications.tsv                           chaque texte modifie (avant/apres)

Usage
-----
  python training/build_dataset.py
  python training/build_dataset.py --push nirina-doris-sylvie/betsileo-mg-v2
      (necessite HF_TOKEN avec droit d'ecriture, depot prive par defaut)
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import os
import random
import re
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "training", "dataset")

SOURCES = {
    "train": os.path.join(ROOT, "bible", "baibolybets.json"),
    "lexique": os.path.join(ROOT, "data", "lexique_examples.json"),
    "validation": os.path.join(ROOT, "bible", "validation-00000-of-00001.json"),
    "test": os.path.join(ROOT, "bible", "test-00000-of-00001.json"),
}

MG2BTS = "Malagasy->Betsileo"
BTS2MG = "Betsileo->Malagasy"
PREFIX = {MG2BTS: "translate Malagasy to Betsileo: ",
          BTS2MG: "translate Betsileo to Malagasy: "}
SEED = 42
# Livres du fichier test deplaces en validation (validation de 37 versets trop
# petite pour regler l'arret anticipe et les garde-fous). Le test garde MRK.
TEST_TO_VALIDATION_BOOKS = {"PHP", "JUD"}
LEXIQUE_TEST_RATIO = 0.15

# Une paire est mal alignee si une langue fait plus du double de l'autre
# (en caracteres). Mesure sur le train : ces paires sont presque toutes des
# cibles tronquees (« Nanambady an-drampela ny faharoa, dia maty... » -> « Ny
# faharoa »).
LEN_RATIO_MIN, LEN_RATIO_MAX = 0.5, 2.0
LEN_RATIO_MIN_CHARS = 20

# Paraphrases trop libres (train uniquement) : similarite caractere entre le
# malgache et le betsileo. Les deux varietes etant proches, une similarite
# tres basse veut dire que la traduction a ajoute ou retire une idee.
PARAPHRASE_MIN_SIM = 0.25

# Versets retires a la main. Train : artefacts d'anciens remplacements.
# Test : relecture un par un des versets qui passaient les filtres
# automatiques mais restent religieux par le contexte.
MANUAL_REMOVE = {
    "JHN 8:54": "Dieu remplace par « izyreo » : phrase cassee et religieuse",
    "HEB 10:7": "citation de psaume, Dieu remplace par « azy ô »",
}
for _ref in ("ACT 1:24", "ACT 4:29", "ACT 7:59", "ACT 7:60", "REV 15:4", "ROM 11:3",
             "ACT 26:15", "HEB 1:10"):
    MANUAL_REMOVE[_ref] = "priere au Seigneur (« tompo ô »)"
# « tompo » seul ou « tompontsika » restant apres les remplacements = le
# Seigneur -> verset retire. « manompo tompo roa » (deux maitres) reste ;
# tompoko, tompony, tompon'ny... ne sont pas concernes.
TOMPO_RE = re.compile(r"(?<![\w'-])tompo(?:ntsika|tsika)?(?![\w'-])(?! roa)", re.IGNORECASE)
_TEST_REVIEW = {
    "exorcisme (« aretina » = esprit impur qui parle ou obeit)":
        "MRK 1:26, MRK 1:27, MRK 3:30, MRK 5:12, MRK 9:17, MRK 9:18, MRK 9:22, MRK 9:26",
    "voix venue du ciel / transfiguration":
        "MRK 1:11, MRK 9:3, MRK 9:7",
    "fin des temps, enfer, salut":
        "MRK 8:38, MRK 9:43, MRK 9:44, MRK 9:45, MRK 9:46, MRK 10:26, MRK 10:30, "
        "MRK 13:13, MRK 13:20, MRK 13:24, MRK 13:26, MRK 13:27, MRK 13:31, MRK 13:32, "
        "PHP 3:21, JUD 1:6, JUD 1:23",
    "Dieu (le Pere « any ambony », le Seigneur, doxologie)":
        "MRK 11:25, MRK 11:26, MRK 11:30, MRK 11:31, MRK 12:33, MRK 14:36, JUD 1:5, "
        "JUD 1:24, PHP 4:13",
    "theologie (rancon, Ecritures accomplies, peche pardonne)":
        "MRK 1:7, MRK 3:28, MRK 9:13, MRK 9:31, MRK 10:45, MRK 14:21",
    "Cene / Gethsemani":
        "MRK 14:23, MRK 14:24, MRK 14:35",
    "ange au tombeau / signes miraculeux des croyants":
        "MRK 16:5, MRK 16:18",
}
_TEST_REVIEW["mot arameen glose (« Efata, izany hoe: Maladia! », « Boanerjesy »)"] = \
    "MRK 7:34, MRK 3:17"
for _reason, _refs in _TEST_REVIEW.items():
    for _ref in _refs.split(", "):
        MANUAL_REMOVE[_ref] = "relecture : " + _reason

# Corrections ponctuelles, appliquees apres la normalisation :
# (ref, langue, avant, apres).
MANUAL_FIXES = [
    # Le betsileo ajoute « choisie par Dieu », absent du malgache.
    ("2JN 1:1", "bts", "voafidin'Andriamagnitsa", "voafidy"),
    # Chansons, traductions de Claude : erreurs nettes.
    # « mandihana » = « mandehana » (tu l'as corrige ainsi dans tsiahyMisareba:32 et :60).
    ("tsivahiny/Asabotsy mifankahita.json:18", "mg", "mandihiza", "mandehana"),
    # « roso agn'fianarana » = parti etudier, pas « parti a Fianarantsoa ».
    ("tsivahiny/tsiahyMisareba.json:9", "mg", "lasa any fianarantsoa", "lasa any am-pianarana"),
    # Chansons, tes traductions : fautes de frappe et mots betsileo restes cote malgache.
    ("Majeur 7/Ataovy serieux.json:25", "mg", "hidivianako", "hividianako"),
    ("Majeur 7/Tagnitagny madigna.json:16", "mg", "sentosntonao", "sentosentonao"),
    ("Majeur 7/galeoka.json:1", "mg", "manontany hoa", "manontany hoe"),
    ("tsivahiny/velo-po.json:7", "mg", "salama aby ve tsika ao", "salama daholo ve isika ao"),
    # Ta relecture a coupe « ; tsa hoandry ve » cote betsileo : meme coupe cote malgache.
    ("tsivahiny/tsiahyMisareba.json:14", "mg", "aminay a; tsy handry ve", "aminay a"),
]

# ---------------------------------------------------------------------------
# Corpus des chansons (corpus_chansons/corpus_chansons.json)
# ---------------------------------------------------------------------------
CHANSONS_DIR = os.path.join(ROOT, "corpus_chansons")
# Decoupage PAR CHANSON : un refrain present a la fois en train et en test
# fausserait le score. Validation et test ne gardent que TES traductions :
# celles de Claude (traductions_claude.tsv) n'ont jamais ete relues.
CHANSONS_VALIDATION = {"galeoka", "velo-po"}
CHANSONS_TEST = {"Tagnitagny madigna", "tsofyrano"}
# Ta relecture des traductions de Claude (copie de chansons_claude_a_relire.tsv
# que tu as corrigee). Ligne presente = textes repris tels quels ; ligne
# supprimee = vers retire du corpus. Si le fichier n'existe pas, les
# traductions de Claude sont gardees telles quelles en train.
CHANSONS_RELUES = os.path.join(ROOT, "training", "chansons_relues.tsv")
# Interjections chantees retirees des deux cotes (« a », « an! », « oh », « e! »...).
INTERJECTION_RE = re.compile(
    r"(?<![\w'-])(?:a|an|ann|annh|hann|aaa|aaaa|o|oh|ohh|e|è|eh|ee|eee|eeh|ie|ié|iei|he)(?![\w'-])",
    re.IGNORECASE)


def strip_interjections(t: str) -> str:
    t = INTERJECTION_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    # « ! ! » ou « , ! » laisses par un mot retire : on garde ! ou ? s'il y en a un
    t = re.sub(r"[,.;:!?](?:\s*[,.;:!?])+",
               lambda m: next((c for c in m.group() if c in "!?"), m.group()[0]), t)
    t = re.sub(r"^[\s,.;:!?]+", "", t)
    return re.sub(r"[\s,;:]+$", "", t).strip()


# Pas de filtre « quasi identique » : une paire qui ne differe que par
# tsa/tsy, ro/no ou agnao/anao est justement un bon exemple minimal.
# Traductions de Claude gardees en train mais douteuses : a relire.
CHANSON_DOUTES = {
    "oladad/agnao.json:1": "« le » rendu par « satria » : a confirmer",
    "oladad/ziva.json:15": "« na dia akory aza » : tournure douteuse",
    "tsivahiny/Asabotsy mifankahita.json:37": "« aba » laisse tel quel (daholo ? dada ?)",
    "tsivahiny/tsiahyMisareba.json:35": "« aba » laisse tel quel",
    "tsivahiny/tsahavelako.json:8": "« raha » rendu par « zavatra » ; peut-etre « raha » (si)",
    "tsivahiny/zipety.json:6": "« bokaka » rendu par « zazavavy » : a confirmer",
    "tsivahiny/Mihinana soa.json:11": "« ravin-mangahazo » : forme a verifier",
    "autres.txt:4": "« raha tsa ho afake » rendu par « no zavatra tsy afaka »",
}
# Tes traductions : points a verifier, laisses tels quels.
CHANSON_A_VERIFIER = {
    "Majeur 7/galeoka.json:19": "« misotrotro » : « misotro » ?",
    "tsivahiny/velo-po.json:2": "« apela » (betsileo) reste cote malgache",
    "Majeur 7/Ataovy serieux.json:27": "« fahatanitany » : mot a confirmer (aussi ligne 31)",
}

# ---------------------------------------------------------------------------
# 1. Normalisation du texte
# ---------------------------------------------------------------------------
APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'",
                      "“": '"', "”": '"', "«": '"', "»": '"',
                      " ": " "})
# Renvois de notes de la Bible betsileo : une lettre a-d isolee
# (« denaria b zato », « orana b ro », « zava-poagna izay.” b »).
FOOTNOTE_RE = re.compile(r"(?<=[\w,.;:!?\"'])\s+[a-d](?=\s|[,.;:!?\"']|$)")
SCRIPTURE_REF_RE = re.compile(
    r"\(\s*(?:[1-3]\s?)?[A-Z][a-z]{1,4}\.?\s*\d+[\d.,:\-– ]*\)?")


def normalize(text: str, lang: str = "mg", bible: bool = True) -> tuple[str, bool]:
    """Retourne (texte normalise, contenait une reference biblique).
    Les renvois de notes n'existent que dans la Bible betsileo ; dans les
    chansons, un « a » isole est une interjection et reste."""
    t = unicodedata.normalize("NFC", text).translate(APOS)
    had_ref = bool(SCRIPTURE_REF_RE.search(t))
    t = SCRIPTURE_REF_RE.sub(" ", t)
    t = t.replace("*", "")
    t = re.sub(r"[\[\]]", "", t)
    if lang == "bts":
        # Une seule orthographe pour la nasale : la Bible ecrit toujours « gn »
        # (19 489 fois, jamais « ñ ») ; deux chansons ecrivent « agñao », « aiña ».
        t = t.replace("gñ", "gn").replace("ñ", "gn").replace("Ñ", "Gn")
        if bible:
            t = FOOTNOTE_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,;:])(?=[^\s\d\"'])", r"\1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, had_ref


# ---------------------------------------------------------------------------
# 2. Remplacements (repris du commit « remplacement mot » sur le train)
# ---------------------------------------------------------------------------
# (motif, remplacement malgache, remplacement betsileo). None = ne pas toucher
# a cette langue. Motifs insensibles a la casse ; la majuscule initiale est
# conservee.
REPLACEMENTS: list[tuple[str, str | None, str | None]] = [
    # vocabulaire religieux -> quotidien
    ("ny Zanak'Olona", "ilay lehilahy", None),
    ("Zanak'Olona", "ilay lehilahy", None),
    ("ny Zanak'Ona", None, "ilay lihilahy"),
    ("Zanak'Ona", None, "ilay lihilahy"),
    ("fitoeran'ny maty", "fasana", "fasagna"),
    ("fanahy ratsy", "aretina", None),
    ("fanahy maloto", "aretina", None),
    ("fagnahy ratsy", None, "aretigna"),
    ("fagnahy maloto", None, "aretigna"),
    ("mpanompo sampy", "mpanao ody", "mpanao ody"),
    ("fanompoam-pivavahana", "fanompoana", "fanompoana"),
    ("fanomezam-pahasoavana", "fanomezam-pitiavana", None),
    ("fagnomezam-pahasoavana", None, "fagnomezam-pitiavana"),
    ("Vaovao Mahafaly", "vaovao tsara", "vaovao soa"),
    ("Vaovao Mahafalin'ny", "vaovao tsaran'ny", "vaovao soan'ny"),
    ("natao batisa", "nosasana", "nosasana"),
    ("an-danitra", "ambony", None),
    ("an-dagnitsa", None, "ambony"),
    ("an-dagnitra", None, "ambony"),
    ("manam-pinoana", "manam-pahatokiana", "manam-pahatokiana"),
    ("am-pinoana", "am-pahatokiana", "am-pahatokiana"),
    ("finoana", "fahatokiana", "fahatokiana"),
    ("finoanareo", "fahatokianareo", "fahatokianareo"),
    ("finoanao", "fahatokianao", "fahatokianao"),
    ("finoany", "fahatokiany", "fahatokiany"),
    ("finoantsika", "fahatokiantsika", "fahatokiantsika"),
    ("finoan'ny", "fahatokian'ny", "fahatokian'ny"),
    ("finoan'ireo", "fahatokian'ireo", "fahatokian'ireo"),
    ("finoako", "fahatokiako", "fahatokiako"),
    ("finoanay", "fahatokianay", "fahatokianay"),
    ("finoan'i", "fahatokian'i", "fahatokian'i"),
    ("finoan'izy", "fahatokian'izy", "fahatokian'izy"),
    ("mino", "matoky", "matoky"),
    ("nino", "natoky", "natoky"),
    ("hino", "hatoky", "hatoky"),
    ("minoa", "matokia", "matokia"),
    ("mpino", "mpiara-dia", "mpiara-dia"),
    ("ninoany", "natokiany", "natokiany"),
    ("ninoanao", "natokianao", "natokianao"),
    ("hinoanareo", "hatokianareo", "hatokianareo"),
    ("inoanareo", "itokianareo", "itokianareo"),
    ("nofo", "vatana", "vatagna"),
    ("nofoko", "vatako", "vatagnako"),
    ("nofony", "vatany", "vatagnany"),
    ("nofonareo", "vatanareo", "vatagnareo"),
    ("nofon'ny", "vatan'ny", "vatagnan'ny"),
    ("ara-nofo", "ara-batana", "ara-batagna"),
    ("masina", "madio", "madio"),
    ("fahamasinana", "fahadiovana", "fahadiovana"),
    ("mandrakizay", "mandrakariva", "mandrakariva"),
    ("fahotana", "fahadisoana", "fahadisoana"),
    ("fahotany", "fahadisoany", "fahadisoany"),
    ("fahotanareo", "fahadisoanareo", "fahadisoanareo"),
    ("fahotantsika", "fahadisoantsika", "fahadisoantsika"),
    ("helokao", "fahadisoanao", "fahadisoanao"),
    ("ota", "fahadisoana", "fahadisoana"),
    ("mpanota", "mpanao ratsy", "mpanao ratsy"),
    ("manota", "manao ratsy", "manao ratsy"),
    ("nanota", "nanao ratsy", "nanao ratsy"),
    ("hanota", "hanao ratsy", "hanao ratsy"),
    ("anjely", "iraka", "iraka"),
    ("anjeliny", "irakany", "irakany"),
    ("fahasoavana", "fitiavana", "fitiavana"),
    ("fahasoavany", "fitiavany", "fitiavany"),
    ("fahasoavan'", "fitiavan'", "fitiavan'"),
    ("mpaminany", "olon-kendry", "ona hendry"),
    ("faminaniana", "toro-hevitra", "toro-hevitsa"),
    ("maminany", "manoro hevitra", "manoro hevitsa"),
    ("haminany", "hanoro hevitra", "hanoro hevitsa"),
    ("naminany", "nanoro hevitra", "nanoro hevitsa"),
    ("Fariseo", "loholona", "lohologna"),
    ("hazofijaliana", "hazo", "hazo"),
    ("hazofijaleana", "hazo", "hazo"),
    ("hazofijaliany", "hazony", "hazony"),
    ("fanatitra", "fanomezana", "fanomezana"),
    ("fanatitrao", "fanomezanao", "fanomezanao"),
    ("fagnatitsa", None, "fagnomezana"),
    ("epistily", "taratasy", "taratasy"),
    ("nitory", "nilaza", "nitarogna"),
    ("mitory", "milaza", "mitarogna"),
    ("hitory", "hilaza", "hitarogna"),
    ("notoriko", "nolazaiko", None),
    ("mibebaka", "miova fo", "miova fo"),
    ("hibebaka", "hiova fo", "hiova fo"),
    ("nibebaka", "niova fo", "niova fo"),
    ("mibebaha", "miovà fo", "miovà fo"),
    ("fibebahana", "fiovam-po", "fiovam-po"),
    ("Paska", "fety", "fety"),
    ("fetin'ny Paka", None, "fety"),
    ("Paka", None, "fety"),
    ("Sinedriona", "filan-kevitry ny loholona", "filan-kevitsy ny lohologna"),
    ("dragona", "voay", None),
    ("dragogna", None, "voay"),
    ("alitara", "latabatra", "latabatsa"),
    ("devoly", "fahavalo", "fahavalo"),
    ("Samaritana", "vahiny", "vahiny"),
    ("Grika", "vahiny", "vahiny"),
    ("Galiliana", "avy any Lalangina", "avy agny Lalangina"),
    # « tompo » = le Seigneur : tournures qu'on peut neutraliser. Le reste des
    # « tompo » seuls / « tompontsika » fait retirer le verset (TOMPO_RE).
    ("ao amin'ny tompo", "", ""), ("ao amin'izy tompo", "", ""), ("ao amin'iy tompo", "", ""),
    ("tao aminy tompontsika", "", ""), ("ao aminy tompontsika", "", ""),
    ("hoy ny tompo", "hoy izy", "hoy iy"),
    ("tompo ô", "tompoko ô", "tompoko ô"), ("ry tompo", "ry tompoko", "ry tompoko"),
    # noms bibliques -> noms malgaches (memes choix que pour le train)
    ("Paoly", "Fidy", "Fidy"), ("Saoly", "Fidy", "Fidy"),
    ("Davida", "Ravelo", "Ravelo"), ("Galilia", "Lalangina", "Lalangina"),
    ("Lazarosy", "Solo", "Solo"), ("Mosesy", "Rabe", "Rabe"),
    ("Simona", "Tojo", "Tojo"), ("Marta", "Lalao", "Lalao"),
    ("Maria", "Vola", "Vola"), ("Timoty", "Tahina", "Tahina"),
    ("Filipo", "Hery", "Hery"), ("Kapernaomy", "Alakamisy", "Alakamisy"),
    # « oliva » seul = l'olivier (hazo oliva) : seul le mont des Oliviers change.
    ("Elizabeta", "Noro", "Noro"),
    ("tendrombohitra Oliva", "tendrombohitra Andringitra", "tendrombohitra Andringitra"),
    ("tendrombohitsa Oliva", "tendrombohitsa Andringitra", "tendrombohitsa Andringitra"),
    ("Stefana", "Mamy", "Mamy"), ("Zakaiosy", "Rabary", "Rabary"),
    ("Nikodemosy", "Ranaivo", "Ranaivo"), ("Barabasy", "Bezaza", "Bezaza"),
    ("Jodasy", "Bema", "Bema"), ("Jopa", "Manakara", "Manakara"),
    ("Titosy", "Tiana", "Tiana"), ("Heroda", "Andrianaivo", "Andrianaivo"),
    ("Tikiko", "Tsiry", "Tsiry"), ("Espaina", "Maorisy", "Maorisy"),
    ("Ananiasy", "Nirina", "Nirina"), ("Agripa", "Andrianjafy", "Andrianjafy"),
    ("Kaisaria", "Mananjary", "Mananjary"), ("Levy", "Doda", "Doda"),
    ("Saraha", "Hanta", "Hanta"), ("Efesosy", "Mahajanga", "Mahajanga"),
    ("Lota", "Rasolo", "Rasolo"), ("Noa", "Rainibe", "Rainibe"),
    ("Tomasy", "Rija", "Rija"), ("Jona", "Zo", "Zo"),
    ("Solomona", "Andriana", "Andriana"), ("Ikonioma", "Ambohimahasoa", "Ambohimahasoa"),
    ("Betania", "Sahambavy", "Sahambavy"), ("Demetrio", "Feno", "Feno"),
    ("Jerosalema", "Fianarantsoa", "Fianarantsoa"), ("Nazareta", "Ambositra", "Ambositra"),
    ("Jodia", "Antsirabe", "Antsirabe"), ("Egypta", "Ambalavao", "Ambalavao"),
    ("Jiosy", "Gasy", "Gasy"), ("Abrahama", "Besa", "Besa"),
    ("Elia", "Rado", "Rado"), ("Jakoba", "Rivo", "Rivo"),
    ("Zakaria", "Koto", "Koto"), ("Jairo", "Fetra", "Fetra"),
    ("Josefa", "Rakoto", "Rakoto"), ("Barnabasy", "Lova", "Lova"),
    ("Aleksandro", "Rasamy", "Rasamy"), ("Zebedio", "Rabenja", "Rabenja"),
    ("Isaka", "Haja", "Haja"), ("Jaona", "Lita", "Lita"),
    ("Adama", "Rabeony", "Rabeony"), ("Epafra", "Mahery", "Mahery"),
    ("Gerasena", "Ambatofinandrahana", "Ambatofinandrahana"),
    ("Korinto", "Antsiranana", "Antsiranana"), ("Roma", "Antananarivo", "Antananarivo"),
    ("Antiokia", "Toamasina", "Toamasina"), ("Samaria", "Ihosy", "Ihosy"),
    ("Makedonia", "Amoron'i Mania", "Amoron'i Mania"),
]

# Pronoms et titres ecrits avec une majuscule au milieu d'une phrase
# parce qu'ils designent Dieu ou Jesus. Le train les a mis en minuscule.
DIVINE_CAPS = """Izy Azy Aho Ahy Izaho Iy Anao Ianao Agnao Aminy Taminy Agnaminy Tagnaminy
Aminao Taminao Agnaminao Amiko Tamiko Agnamiko Anahy Aminahy Taminahy Aminazy
Taminazy Ilay Izay Ray Raiko Rainao Rainareo Raintsika Rainy Zanaka Zanako Zanany
Zanaky Zanakao Tompo Tompoko Tompontsika Tompony Tompon'ny Agnarako Anarako
Azoko Ahitanao Ity Lehilahy""".split()

# Termes qui font retirer le verset entier (cherches APRES normalisation et
# AVANT les remplacements, pour attraper « Fanahy Masina » avant que
# « masina » devienne « madio »).
HARD_PATTERNS = [
    # Dieu, Jesus, Esprit, Satan
    r"andriamanitr", r"andriamagnit", r"\bjesosy\b", r"\bjesoa\b", r"\bkristy\b",
    r"\bmesia\b", r"\bTOMPO\b", r"\bJehovah\b",
    r"\bfa(?:nahy|gnahy) masina\b", r"(?<![.!?:\"] )(?<!^)\bFa(?:nahy|gnahy)\b",
    r"\bsatana\b", r"\bbelzebo(?:la|ba|ly)\b", r"\bdemonia\b", r"\bdemoniaka\b",
    r"\bzanak'andriama", r"fanjakan'(?:ny )?la(?:nitra|gnitsa)",
    # culte et institutions bibliques
    r"\btempoly\b", r"\bsinagoga", r"\bsabata\b", r"\bmpisoron", r"\bmpisorogn",
    r"\bsoronabe", r"\bsadoseo\b", r"\bfilazantsara",
    r"\bapostoly\b", r"\bamena\b", r"\bhosana\b", r"\bhaleloia\b", r"\bbatisa\w*",
    r"\braby\b", r"\brabony\b", r"\bsoratra masina\b", r"\bsoratsa masina\b",
    # resurrection : (ni|hi|tafa|na|ha)tsangana / fitsanganana (t)amin'ny maty
    r"\b\w*tsanga\w*(?: t?amin)?'ny maty\b",
    r"(?<![.!?:\"] )(?<!^)\bMasina\b", r"\bmpamonjy\b", r"\bfamonjena\b",
    # (gardes volontairement, comme dans le train : fiangonana, vavaka,
    #  fivavahana, tompo, lanitra, sampy, fahagagana, mpanoradalana, Dada)
    # artefacts d'anciens remplacements de Dieu
    r"\bizyreo\b", r"\bazy ô\b",
    # noms bibliques sans equivalent malgache dans les tableaux de remplacement
    r"\bpetera\b", r"\bpilato\b", r"\bkaisara\b", r"\bandrea\b", r"\bmagdalena\b",
    r"\bjordana\b", r"\bherodiasy\b", r"\bherodiana\b", r"\btiro\b", r"\bsidona\b",
    r"\bjosesy\b", r"\bepafrodito\b", r"\bsalome\b", r"\bbetsaida\b", r"\balfeo\b",
    r"\bdekapolisy\b", r"\bjeriko\b", r"\beodia\b", r"\bsintike\b", r"\bgolgota\b",
    r"\bisi?raely\b", r"\bisraely\b", r"\barimatia\b", r"\biskariota\b",
    r"\bbenjamina\b", r"\bhebreo\b", r"\bedoma\b", r"\babiatara\b", r"\bfilipiana\b",
    r"\bjoda\b", r"\bkaina\b", r"\bbalama\b", r"\bKora\b",
    r"\bbartolomeo\b", r"\bmatio\b", r"\btadeo\b", r"\bkananita\b", r"\bbartimea\b",
    r"\btimea\b", r"\bkirenana\b", r"\bdalmanota\b", r"\bjairosy\b", r"\bnazarena\b",
    r"\bkesaria\b", r"\bfilipisy\b", r"\bklemanta\b", r"\bsyntyke\b", r"\bmikaela\b",
    r"\benoka\b", r"\bkoraha\b", r"\bsodoma\b", r"\bgomora\b", r"\begypta\b(?=.*\bfarao\b)",
]
# Sensibles a la casse : TOMPO (= YHWH), Fanahy en majuscule (= l'Esprit),
# Jehovah, Kora (nom propre ; « kora » est aussi une mesure de grain).
CASE_SENSITIVE = {r"\bTOMPO\b", r"\bJehovah\b", r"\bKora\b",
                  r"(?<![.!?:\"] )(?<!^)\bFa(?:nahy|gnahy)\b",
                  r"(?<![.!?:\"] )(?<!^)\bMasina\b"}
HARD_RE = [re.compile(p) if p in CASE_SENSITIVE else re.compile(p, re.IGNORECASE)
           for p in HARD_PATTERNS]

WORD_BOUND = r"(?<![\w'-]){}(?![\w'-])"


def _case_like(m: re.Match, repl: str) -> str:
    """Nom propre (Fidy) : tel quel. Mot commun : majuscule seulement en debut
    de phrase (« ny Fariseo » -> « ny loholona », « Finoana no... » ->
    « Fahatokiana no... »)."""
    if repl[:1].isupper():
        return repl
    before = m.string[:m.start()].rstrip()
    sentence_start = not before or before[-1] in '.!?:"'
    if m.group(0)[:1].isupper() and sentence_start:
        return repl[:1].upper() + repl[1:]
    return repl


def _compile_replacements():
    out = []
    # Les motifs les plus longs d'abord (« ny Zanak'Olona » avant « Zanak'Olona »).
    for pat, mg, bts in sorted(REPLACEMENTS, key=lambda x: -len(x[0])):
        rx = re.compile(WORD_BOUND.format(re.escape(pat)), re.IGNORECASE)
        out.append((rx, mg, bts))
    return out


_REPL = _compile_replacements()
_DIVINE_RE = re.compile(
    r"(?<=[^\s.!?:;\"])(\s+)(" + "|".join(map(re.escape, DIVINE_CAPS)) + r")(?![\w'-])")
_ANAZY_RE = re.compile(r"(?<![\w'-])an'azy(?![\w'-])")
_GENITIVE_AZY_RE = re.compile(r"(?<![\w'-])(\w+n)'azy(?![\w'-])")


def apply_replacements(text: str, lang: str) -> str:
    """lang : 'mg' ou 'bts'."""
    for rx, mg, bts in _REPL:
        repl = mg if lang == "mg" else bts
        if repl is None:
            continue
        text = rx.sub(lambda m, r=repl: _case_like(m, r), text)
    # pronoms divins en minuscule (sauf en debut de phrase / de citation)
    text = _DIVINE_RE.sub(lambda m: m.group(1) + m.group(2).lower(), text)
    if lang == "mg":
        # « an'azy », « tenin'azy » : traces de noms effaces, pas du malgache
        # officiel. -> « azy », « teniny ».
        text = _ANAZY_RE.sub("azy", text)
        text = _GENITIVE_AZY_RE.sub(lambda m: m.group(1) + "y", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()   # apres une suppression


def hard_terms(text: str) -> list[str]:
    found = []
    for rx in HARD_RE:
        m = rx.search(text)
        if m:
            found.append(m.group(0))
    return found


# ---------------------------------------------------------------------------
# Similarite (sans dependance externe)
# ---------------------------------------------------------------------------
def char_sim(a: str, b: str) -> float:
    a = re.sub(r"[^\w\s]", "", a.lower())
    b = re.sub(r"[^\w\s]", "", b.lower())
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def key(text: str) -> str:
    """Cle de comparaison pour doublons/fuites : minuscules, sans ponctuation."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_rows(path: str) -> list[dict]:
    rows = json.load(open(path, encoding="utf-8"))
    out = []
    for r in rows:
        src = r["source"].split(": ", 1)[1] if ": " in r["source"] else r["source"]
        if r["direction"] == MG2BTS:
            mg, bts = src, r["target"]
        else:
            bts, mg = src, r["target"]
        out.append({"ref": r["ref"], "direction": r["direction"],
                    "mg": mg, "bts": bts})
    return out


# ---------------------------------------------------------------------------
# Segments courts (train uniquement)
# ---------------------------------------------------------------------------
SEG_SPLIT_RE = re.compile(r"(?<=[,;:.!?])\s+")


def segments(mg: str, bts: str) -> list[tuple[str, str]]:
    """Coupe aux virgules/points-virgules si les deux cotes ont le meme
    nombre de propositions et que chaque paire de propositions se ressemble.
    Retourne [] si l'alignement n'est pas sur."""
    a = [s.strip() for s in SEG_SPLIT_RE.split(mg) if s.strip()]
    b = [s.strip() for s in SEG_SPLIT_RE.split(bts) if s.strip()]
    if len(a) < 2 or len(a) != len(b):
        return []
    out = []
    for x, y in zip(a, b):
        if len(x.split()) < 3 or len(y.split()) < 3:
            return []
        r = len(y) / max(1, len(x))
        if not (0.6 <= r <= 1.7) or char_sim(x, y) < 0.35:
            return []
        out.append((x.rstrip(",;:"), y.rstrip(",;:")))
    return out


# ---------------------------------------------------------------------------
# Chansons
# ---------------------------------------------------------------------------
def _corpus_module():
    """Reutilise le nettoyage de corpus_chansons/build_corpus_chansons.py."""
    import importlib.util
    import sys
    path = os.path.join(CHANSONS_DIR, "build_corpus_chansons.py")
    spec = importlib.util.spec_from_file_location("build_corpus_chansons", path)
    mod = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # pas de __pycache__ chez toi
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = previous
    return mod


def pending_song_corrections(bcc, pairs: list[dict]) -> list[dict]:
    """Corrections faites dans corpus_chansons_exclus.tsv mais absentes de
    corpus_chansons.json. Ce fichier a perdu sa ligne d'en-tete : depuis,
    build_corpus_chansons.py ne lit plus tes corrections. On les lit ici avec
    les memes regles (mlg corrige ou valide = 1)."""
    exclus = os.path.join(CHANSONS_DIR, "corpus_chansons_exclus.tsv")
    tsv = os.path.join(CHANSONS_DIR, "traductions_claude.tsv")
    if not (os.path.exists(exclus) and os.path.exists(tsv)):
        return []
    claude = {(r["fichier"], int(r["ligne"])): r
              for r in csv.DictReader(open(tsv, encoding="utf-8", newline=""), delimiter="\t")}
    manual = {r["ref"].strip() for r in csv.DictReader(
        open(os.path.join(CHANSONS_DIR, "exclusions.tsv"), encoding="utf-8", newline=""),
        delimiter="\t")}
    known = {p["ref"] for p in pairs}
    seen = {(bcc.key(p["bts"]), bcc.key(p["mlg"])) for p in pairs}
    out = []
    for row in csv.reader(open(exclus, encoding="utf-8", newline=""), delimiter="\t"):
        _, ref, _, mlg, valide = ([c.strip() for c in row] + [""] * 5)[:5]
        rel, _, idx = ref.rpartition(":")
        if not idx.isdigit() or ref in known or ref in manual:
            continue
        t = claude.get((rel, int(idx)))
        song = "autres" if rel == "autres.txt" else os.path.splitext(os.path.basename(rel))[0]
        if t is None or song in bcc.CHANSONS_EXCLUES:
            continue
        changed = bool(mlg) and bcc.clean_text(mlg) != bcc.clean_text(t["mlg"])
        if not (changed or valide.lower() in ("1", "x", "ok", "oui")):
            continue
        if rel.endswith(".json"):
            items = json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))
            if items[int(idx)].get("bts", "").strip() != t["bts"].strip():
                continue  # vers modifie depuis la traduction
        bts, mg = bcc.align_end(bcc.clean_text(t["bts"]),
                                bcc.clean_text(mlg if changed else t["mlg"]))
        k = (bcc.key(bts), bcc.key(mg))
        if not bts or not mg or k[0] == k[1] or k in seen:
            continue
        seen.add(k)
        out.append({"bts": bts, "mlg": mg, "chanson": song, "groupe": rel.split("/")[0],
                    "ref": ref, "traducteur": "utilisateur"})
    return out


def load_chansons(removed: list, imported: list) -> tuple[dict[str, list[dict]], int]:
    """Paires des chansons nettoyees et reparties PAR CHANSON, et nombre de
    paires lues. Pas de remplacements « religieux » ici : ce ne sont pas des
    versets (« tsara toa anjely » est un compliment, pas un ange)."""
    out = {"train": [], "validation": [], "test": []}
    path = os.path.join(CHANSONS_DIR, "corpus_chansons.json")
    if not os.path.exists(path):
        return out, 0
    bcc = _corpus_module()
    pairs = json.load(open(path, encoding="utf-8"))
    extra = pending_song_corrections(bcc, pairs)
    imported.extend(extra)
    reviewed = None
    if os.path.exists(CHANSONS_RELUES):
        reviewed = {r["ref"].strip(): r for r in csv.DictReader(
            open(CHANSONS_RELUES, encoding="utf-8", newline=""), delimiter="\t") if r.get("ref")}
    seen = {s: set() for s in out}
    for p in pairs + extra:
        ref, song = p["ref"], p["chanson"]
        split = ("test" if song in CHANSONS_TEST else
                 "validation" if song in CHANSONS_VALIDATION else "train")
        human = p["traducteur"] == "utilisateur"
        label = "humain" if human else "claude"
        if not human and reviewed is not None and split == "train":
            r = reviewed.get(ref)
            if r is None:
                removed.append((split, ref, "chanson : retiree a ta relecture", p["mlg"], p["bts"]))
                continue
            p = {**p, "bts": r["betsileo"], "mlg": r["malagasy_claude"]}
            label = "claude_relu"
        texts, mods, hard = {}, [], []
        for lang, field in (("mg", "mlg"), ("bts", "bts")):
            t = normalize(p[field], lang, bible=False)[0]
            t = re.sub(r"(?<=\w) ?- (?=\w)|(?<=\w) -(?=\w)", "-", t)  # « madignan - droky »
            before = t
            for fref, flang, old, new in MANUAL_FIXES:
                if fref == ref and flang == lang and old in t:
                    t = t.replace(old, new)
            t = strip_interjections(t)
            if t != before:
                mods.append((split, ref, lang, before, t))
            hard += hard_terms(t)
            texts[lang] = t
        mg, bts = texts["mg"], texts["bts"]
        if not re.search(r"\w", mg) or not re.search(r"\w", bts):
            removed.append((split, ref, "chanson : vide apres retrait des interjections", mg, bts))
            continue
        if split != "train" and label == "claude":
            removed.append((split, ref, "chanson : traduction de Claude non relue, "
                            "exclue de validation/test", mg, bts))
            continue
        if hard:
            removed.append((split, ref, "religieux : " + ", ".join(sorted(set(hard))), mg, bts))
            continue
        k = (key(mg), key(bts))
        if k in seen[split]:
            removed.append((split, ref, "doublon", mg, bts))
            continue
        seen[split].add(k)
        out[split].append({"ref": ref, "mg": mg, "bts": bts, "chanson": song,
                           "traducteur": label, "_mods": mods})
    return out, len(pairs) + len(extra)


# ---------------------------------------------------------------------------
def build(args) -> None:
    rnd = random.Random(SEED)
    removed: list[tuple] = []      # (split, ref, raison, mg, bts)
    modified: list[tuple] = []     # (split, ref, langue, avant, apres)
    stats = collections.OrderedDict()

    def clean_row(split: str, r: dict) -> dict | None:
        ref = r["ref"]
        if ref in MANUAL_REMOVE:
            removed.append((split, ref, "manuel : " + MANUAL_REMOVE[ref], r["mg"], r["bts"]))
            return None
        out = {}
        hard = []
        mods = []
        for lang in ("mg", "bts"):
            t, had_ref = normalize(r[lang], lang)
            normalized = t
            for fref, flang, old, new in MANUAL_FIXES:
                if fref == ref and flang == lang and old in t:
                    t = t.replace(old, new)
            if had_ref:
                hard.append("reference biblique")
            hard += hard_terms(t)
            t2 = apply_replacements(t, lang)
            # Verification aussi apres remplacement : « natsangan'azy tamin'ny maty »
            # ne devient detectable qu'une fois corrige en « natsangany tamin'ny maty ».
            hard += hard_terms(t2)
            m = TOMPO_RE.search(t2)
            if m:
                hard.append(m.group(0) + " (le Seigneur)")
            out[lang] = t2
            # On ne journalise que les vrais changements de mots, pas la
            # normalisation (apostrophes, espaces) qui touche presque tout.
            if t2 != normalized:
                mods.append((split, ref, lang, normalized, t2))
        if hard:
            removed.append((split, ref, "religieux : " + ", ".join(sorted(set(hard))),
                            out["mg"], out["bts"]))
            return {"_hard": True, "ref": ref}
        if not out["mg"] or not out["bts"]:
            removed.append((split, ref, "vide", out["mg"], out["bts"]))
            return None
        a, b = len(out["mg"]), len(out["bts"])
        if min(a, b) >= LEN_RATIO_MIN_CHARS or max(a, b) >= 2 * LEN_RATIO_MIN_CHARS:
            ratio = b / max(1, a)
            if not (LEN_RATIO_MIN <= ratio <= LEN_RATIO_MAX):
                removed.append((split, ref, f"mal aligne (longueur bts/mg = {ratio:.2f})",
                                out["mg"], out["bts"]))
                return None
        return {"ref": ref, "direction": r["direction"], "mg": out["mg"],
                "bts": out["bts"], "_mods": mods}

    def clean_split(split: str, rows: list[dict]) -> list[dict]:
        cleaned = [clean_row(split, r) for r in rows]
        # Un verset religieux dans un sens est retire dans les deux sens.
        hard_refs = {c["ref"] for c in cleaned if c and c.get("_hard")}
        kept = [c for c in cleaned if c and not c.get("_hard") and c["ref"] not in hard_refs]
        for c in cleaned:
            if c and not c.get("_hard") and c["ref"] in hard_refs:
                removed.append((split, c["ref"], "religieux (dans l'autre sens)", c["mg"], c["bts"]))
        return kept

    # ---- bible ----
    bible = {s: clean_split(s, load_rows(SOURCES[s])) for s in ("train", "validation", "test")}
    # Validation agrandie : des livres entiers du test y passent (jamais un livre
    # a cheval sur deux splits).
    moved = [r for r in bible["test"] if r["ref"].split(" ")[0] in TEST_TO_VALIDATION_BOOKS]
    bible["test"] = [r for r in bible["test"] if r["ref"].split(" ")[0] not in TEST_TO_VALIDATION_BOOKS]
    bible["validation"] += moved
    raw_counts = {s: len(load_rows(SOURCES[s])) for s in ("train", "validation", "test")}

    # ---- train : paraphrases trop libres ----
    kept = []
    for r in bible["train"]:
        sim = char_sim(r["mg"], r["bts"])
        if sim < PARAPHRASE_MIN_SIM:
            removed.append(("train", r["ref"], f"paraphrase trop libre (similarite {sim:.2f})",
                            r["mg"], r["bts"]))
        else:
            kept.append(r)
    bible["train"] = kept

    # ---- chansons (decoupage par chanson) ----
    imported: list[dict] = []
    songs, n_song_pairs = load_chansons(removed, imported)

    # ---- train : fuites et doublons ----
    heldout = {key(r[l]) for s in ("validation", "test") for r in bible[s] for l in ("mg", "bts")}
    heldout |= {key(p[l]) for s in ("validation", "test") for p in songs[s] for l in ("mg", "bts")}
    kept = []
    for p in songs["train"]:
        if key(p["mg"]) in heldout or key(p["bts"]) in heldout:
            removed.append(("train", p["ref"], "fuite : phrase presente en validation/test",
                            p["mg"], p["bts"]))
        else:
            kept.append(p)
    songs["train"] = kept
    seen, kept = set(), []
    for r in bible["train"]:
        if key(r["mg"]) in heldout or key(r["bts"]) in heldout:
            removed.append(("train", r["ref"], "fuite : phrase presente en validation/test",
                            r["mg"], r["bts"]))
            continue
        k = (r["direction"], key(r["mg"]), key(r["bts"]))
        if k in seen:
            removed.append(("train", r["ref"], "doublon", r["mg"], r["bts"]))
            continue
        seen.add(k)
        kept.append(r)
    bible["train"] = kept

    # ---- lexique : 15 % des mots mis de cote pour le test ----
    lex = load_rows(SOURCES["lexique"])
    for r in lex:
        r["mg"] = normalize(r["mg"], "mg")[0]
        r["bts"] = normalize(r["bts"], "bts")[0]
    lex_refs = sorted({r["ref"] for r in lex})
    rnd.shuffle(lex_refs)
    n_test = round(LEXIQUE_TEST_RATIO * len(lex_refs))
    lex_test_refs = set(lex_refs[:n_test])
    # Un meme mot malgache ne doit pas etre dans les deux splits.
    test_words = {key(r["mg"]) for r in lex if r["ref"] in lex_test_refs} | \
                 {key(r["bts"]) for r in lex if r["ref"] in lex_test_refs}
    lex_train = [r for r in lex if r["ref"] not in lex_test_refs
                 and key(r["mg"]) not in test_words and key(r["bts"]) not in test_words]
    lex_test = [r for r in lex if r["ref"] in lex_test_refs]

    # ---- segments courts (train) ----
    seg_rows, seg_seen = [], set()
    if not args.no_segments:
        for r in bible["train"]:
            for x, y in segments(r["mg"], r["bts"]):
                k = (r["direction"], key(x), key(y))
                if k in seg_seen or key(x) in heldout or key(y) in heldout:
                    continue
                seg_seen.add(k)
                seg_rows.append({"ref": r["ref"], "direction": r["direction"], "mg": x, "bts": y})

    # Journal des modifications : seulement pour les lignes gardees.
    for s in ("train", "validation", "test"):
        for r in bible[s] + songs[s]:
            modified.extend(r.get("_mods", []))

    # ---- assemblage ----
    def to_example(r: dict, origin: str) -> dict:
        d = r["direction"]
        src, tgt = (r["mg"], r["bts"]) if d == MG2BTS else (r["bts"], r["mg"])
        return {
            "id": f"{origin}|{r['ref']}|{'mg2bts' if d == MG2BTS else 'bts2mg'}",
            "source": PREFIX[d] + src,      # format historique (T5 / ByT5)
            "target": tgt,
            "direction": d,
            "src_lang": "mg" if d == MG2BTS else "bts",
            "tgt_lang": "bts" if d == MG2BTS else "mg",
            "src_text": src,
            "ref": r["ref"],
            "origin": origin,               # bible | lexique | segment | chanson
            "traducteur": r.get("traducteur", "humain"),   # humain | claude (non relu)
        }

    def both_directions(pairs: list[dict], origin: str) -> list[dict]:
        return [to_example({**p, "direction": d}, origin) for p in pairs for d in (MG2BTS, BTS2MG)]

    splits = {
        "train": [to_example(r, "bible") for r in bible["train"]]
                 + [to_example(r, "lexique") for r in lex_train]
                 + [to_example(r, "segment") for r in seg_rows]
                 + both_directions(songs["train"], "chanson"),
        "validation": [to_example(r, "bible") for r in bible["validation"]]
                      + both_directions(songs["validation"], "chanson"),
        "test": [to_example(r, "bible") for r in bible["test"]]
                + both_directions(songs["test"], "chanson")
                + [to_example(r, "lexique") for r in lex_test],
    }
    for s in ("train",):
        rnd.shuffle(splits[s])

    # ---- ecriture ----
    # 4 champs seulement : le reste se deduit (langues <- direction, texte sans
    # prefixe <- source, origine <- ref : « LEX ... », « ....json:N » ou verset).
    os.makedirs(OUT_DIR, exist_ok=True)
    for s, rows in splits.items():
        with open(os.path.join(OUT_DIR, f"{s}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                slim = {k: row[k] for k in ("source", "target", "direction", "ref")}
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    # Les versets contiennent des guillemets : csv les echappe pour qu'Excel,
    # LibreOffice ou pandas relisent bien une ligne par verset.
    to_review = [(p["ref"], p["bts"], p["mg"], CHANSON_DOUTES.get(p["ref"], ""))
                 for p in songs["train"] if p["traducteur"] == "claude"]
    for name, header, rows in (("retires.tsv", ["split", "ref", "raison", "malagasy", "betsileo"], removed),
                               ("modifications.tsv", ["split", "ref", "langue", "avant", "apres"], modified),
                               ("chansons_claude_a_relire.tsv", ["ref", "betsileo", "malagasy_claude", "doute"],
                                to_review)):
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    # ---- statistiques ----
    def count(rows, origin=None):
        rows = [r for r in rows if origin is None or r["origin"] == origin]
        return len(rows), len({r["ref"] for r in rows})

    reasons = collections.Counter()
    for split, ref, reason, *_ in removed:
        reasons[(split, reason.split(" :")[0].split(" (")[0])] += 1

    stats["sources"] = {"train (bible)": raw_counts["train"], "validation": raw_counts["validation"],
                        "test": raw_counts["test"], "lexique": len(lex),
                        "chansons (paires)": n_song_pairs}
    stats["sortie"] = {
        "train bible": count(splits["train"], "bible"),
        "train lexique": count(splits["train"], "lexique"),
        "train segments": count(splits["train"], "segment"),
        "train chansons": count(splits["train"], "chanson"),
        "validation bible": count(splits["validation"], "bible"),
        "validation chansons": count(splits["validation"], "chanson"),
        "test bible": count(splits["test"], "bible"),
        "test chansons": count(splits["test"], "chanson"),
        "test lexique": count(splits["test"], "lexique"),
    }
    stats["retraits"] = reasons
    stats["chansons"] = songs
    stats["imported"] = imported
    write_report(splits, stats, removed, modified)
    write_card(splits)

    print("=" * 66)
    print("DATASET Betsileo <-> malgache officiel (v2)")
    print("=" * 66)
    print("Lignes en entree :", stats["sources"])
    print("\nLignes (versets) en sortie :")
    for k, (n, refs) in stats["sortie"].items():
        print(f"  {k:<16} {n:>6} lignes  ({refs} versets/mots)")
    print("\nRetraits (lignes) :")
    for (split, reason), n in sorted(reasons.items()):
        print(f"  {split:<11} {reason:<45} {n:>5}")
    print(f"\nTextes modifies : {len(modified)} (voir modifications.tsv)")
    if imported:
        print(f"Chansons : {len(imported)} corrections reprises de corpus_chansons_exclus.tsv "
              "(fichier sans ligne d'en-tete, ignorees par build_corpus_chansons.py)")
    print(f"Chansons : {len(to_review)} traductions de Claude non relues, en train seulement "
          "(chansons_claude_a_relire.tsv)")
    print(f"Ecrit dans : {os.path.relpath(OUT_DIR, ROOT)}/")

    if args.push:
        push(args.push, args.public)


# ---------------------------------------------------------------------------
def write_report(splits, stats, removed, modified) -> None:
    lines = ["# Rapport de nettoyage du dataset v2", ""]
    lines += ["## Lignes en sortie", "", "| Partie | Lignes | Versets / mots |", "|---|---|---|"]
    for k, (n, refs) in stats["sortie"].items():
        lines.append(f"| {k} | {n} | {refs} |")
    lines += ["", "## Retraits", "", "| Split | Raison | Lignes |", "|---|---|---|"]
    for (split, reason), n in sorted(stats["retraits"].items()):
        lines.append(f"| {split} | {reason} | {n} |")
    term_count = collections.Counter()
    for split, ref, reason, *_ in removed:
        if reason.startswith("religieux :"):
            for t in reason.split(": ", 1)[1].split(", "):
                term_count[(split, t.lower())] += 1
    for split in ("validation", "test", "train"):
        top = [(t, n) for (s, t), n in term_count.most_common() if s == split][:30]
        if top:
            lines += ["", f"### Termes religieux trouves ({split})", "",
                      ", ".join(f"{t} ({n})" for t, n in top)]
    lines += ["", "## Validation : versets retires ou modifies", ""]
    seen = set()
    for split, ref, reason, mg, bts in removed:
        if split == "validation" and (ref, reason) not in seen:
            seen.add((ref, reason))
            lines.append(f"- **{ref}** retire : {reason}")
    for split, ref, lang, before, after in modified:
        if split == "validation" and (ref, lang, after) not in seen:
            seen.add((ref, lang, after))
            lines.append(f"- **{ref}** ({lang}) : `{before}` -> `{after}`")
    lines += ["", "## Test : versets retires apres relecture", ""]
    by_reason = collections.defaultdict(list)
    for split, ref, reason, *_ in removed:
        if split == "test" and reason.startswith("relecture") and ref not in by_reason[reason]:
            by_reason[reason].append(ref)
    for reason, refs in by_reason.items():
        lines.append(f"- {reason.split(': ', 1)[1]} : {', '.join(refs)}")
    lines += ["", "## Garde volontairement", "",
              "Mots religieux devenus courants, gardes comme dans le train (decision de",
              "`bible/remplacements_proposes.md`) : fiangonana, vavaka / mivavaka,",
              "fivavahana, lanitra (le ciel), Dada, sampy, fahagagana, mpanoradalana.",
              "A retirer en les ajoutant a `HARD_PATTERNS` si besoin.", "",
              "« tompo » (le Seigneur) : « ao amin'ny tompo » supprime, « hoy ny tompo » -> « hoy izy »,",
              "« Tompo ô » adresse a quelqu'un -> « Tompoko ô » ; tout autre « tompo » seul ou",
              "« tompontsika » fait retirer le verset. Restent : tompoko, tompony, tompon'ny...,",
              "« manompo tompo roa » (deux maitres)."]

    songs = stats["chansons"]
    lines += ["", "## Chansons", "",
              "Decoupage PAR CHANSON (un refrain ne doit pas etre a la fois en train et en test).",
              "Validation et test : uniquement tes traductions. Les traductions de Claude",
              "(`traductions_claude.tsv`, jamais relues depuis leur creation) ne vont qu'en train,",
              "et sont listees dans `chansons_claude_a_relire.tsv`.", "",
              "| Split | Chanson | Paires | dont traduites par Claude |", "|---|---|---|---|"]
    for split in ("train", "validation", "test"):
        per_song = collections.Counter(p["chanson"] for p in songs[split])
        for song, n in sorted(per_song.items()):
            n_claude = sum(1 for p in songs[split] if p["chanson"] == song and p["traducteur"] == "claude")
            lines.append(f"| {split} | {song} | {n} | {n_claude} |")
    if stats["imported"]:
        lines += ["", "### Tes corrections reprises de `corpus_chansons_exclus.tsv`", "",
                  "Ce fichier a perdu sa ligne d'en-tete (`raison ref bts mlg valide`) : "
                  "`build_corpus_chansons.py` ne lisait plus tes corrections. Elles sont reprises ici :", ""]
        lines += [f"- {p['ref']} : `{p['bts']}` -> `{p['mlg']}`" for p in stats["imported"]]
    unreviewed = {p["ref"] for p in songs["train"] if p["traducteur"] == "claude"}
    n_relu = sum(1 for p in songs["train"] if p["traducteur"] == "claude_relu")
    lines += ["", f"### Traductions de Claude : {n_relu} relues par toi "
              f"(`training/chansons_relues.tsv`), {len(unreviewed)} pas encore relues", ""]
    lines += [f"- {ref} : {note}" for ref, note in CHANSON_DOUTES.items() if ref in unreviewed]
    lines += ["", "### Tes traductions : points a verifier (laisses tels quels)", ""]
    lines += [f"- {ref} : {note}" for ref, note in CHANSON_A_VERIFIER.items()]
    lines += ["", "Les corrections appliquees (erreurs nettes de Claude, fautes de frappe) sont dans",
              "`modifications.tsv` et `MANUAL_FIXES` en haut du script."]
    with open(os.path.join(OUT_DIR, "rapport_nettoyage.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_card(splits) -> None:
    n = {s: len(rows) for s, rows in splits.items()}
    card = f"""---
language:
- mg
license: other
task_categories:
- translation
pretty_name: Betsileo <-> malgache officiel (v2)
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Betsileo <-> malgache officiel (v2)

Paires paralleles betsileo / malgache officiel, dans les deux sens.
Construit par `training/build_dataset.py` (depot betsileo-song-data).

| Split | Lignes |
|---|---|
| train | {n['train']} |
| validation | {n['validation']} |
| test | {n['test']} |

## Colonnes

- `source` : entree avec le prefixe T5 (`translate Malagasy to Betsileo: ...`)
- `target` : traduction attendue
- `direction` : `Malagasy->Betsileo` ou `Betsileo->Malagasy`
- `ref` : verset biblique (`MRK 1:31`), entree du lexique (`LEX ...`) ou vers
  de chanson (`Majeur 7/galeoka.json:6`, `autres.txt:3`). Les segments courts
  du train portent la ref du verset dont ils sont tires.

## Decoupage

- train : livres du Nouveau Testament hors test/validation + lexique
  (85 %) + segments courts + chansons hors validation/test.
- validation : TIT, PHM, 2JN, 3JN, PHP, JUD + chansons {', '.join(sorted(CHANSONS_VALIDATION))}.
- test : MRK + chansons {', '.join(sorted(CHANSONS_TEST))}
  + 15 % du lexique (`origin = lexique`, a evaluer a part : precision mot a mot).

Les chansons sont reparties par chanson ; en validation et test, seules les
traductions humaines sont gardees.

Les versets trop religieux ont ete retires et le vocabulaire religieux
restant remplace par des mots du quotidien, avec les memes regles pour les
trois splits.

Licence : les versets viennent de traductions bibliques dont les droits
sont a verifier avant toute publication (garder le depot prive).
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def push(repo_id: str, public: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=OUT_DIR,
                      allow_patterns=["*.jsonl", "README.md"],
                      commit_message="dataset v2 (build_dataset.py)")
    print(f"\nPousse : https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="REPO_ID", help="pousse le dataset sur le Hub")
    ap.add_argument("--public", action="store_true", help="depot public (prive par defaut)")
    ap.add_argument("--no-segments", action="store_true", help="sans segments courts")
    build(ap.parse_args())
