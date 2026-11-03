# Entraînement betsileo ↔ malgache officiel (v2)

## 1. Construire le dataset

```bash
python training/build_dataset.py
```

Regroupe les sources sans les modifier :

| Source | Va dans |
|---|---|
| `bible/baibolybets.json` | train |
| `data/lexique_examples.json` | train (85 %) et test (15 %, pour mesurer la traduction de mots jamais vus) |
| `bible/validation-00000-of-00001.json` | validation |
| `bible/test-00000-of-00001.json` | test |
| `corpus_chansons/corpus_chansons.json` | par chanson : validation = galeoka, velo-po ; test = Tagnitagny madigna, tsofyrano ; le reste en train |

Les chansons de validation et de test ne gardent que tes traductions. Les vers traduits par Claude
(`corpus_chansons/traductions_claude.tsv`, jamais relus) ne vont qu'en train ; ils sont listés dans
`chansons_claude_a_relire.tsv`. Pour les corriger, modifie la colonne `mlg` de `traductions_claude.tsv`
puis relance `corpus_chansons/build_corpus_chansons.py` et ce script.

Sorties dans `training/dataset/` :

- `train.jsonl`, `validation.jsonl`, `test.jsonl` : le dataset
- `rapport_nettoyage.md` : chiffres et décisions
- `retires.tsv` : chaque verset retiré et pourquoi
- `modifications.tsv` : chaque texte modifié (avant / après)
- `chansons_claude_a_relire.tsv` : traductions de Claude utilisées en train, avec mes doutes

Relance le script après chaque correction à la main dans les fichiers sources. Pour garder ou retirer un mot
religieux, modifie `REPLACEMENTS`, `HARD_PATTERNS` ou `MANUAL_REMOVE` en haut du script.

## 2. Mettre les données en ligne

Recommandé : un dataset Hugging Face privé. Il est versionné, et le même dataset sert sur Kaggle, Colab ou en local.

```powershell
pip install huggingface_hub
$env:HF_TOKEN = "hf_..."          # token avec droit d'écriture
python training/build_dataset.py --push nirina-doris-sylvie/betsileo-mg-v2
```

Autre possibilité, sans Hugging Face : sur Kaggle, *Datasets → New Dataset*, dépose les 3 fichiers `.jsonl`, puis
*Add Input* dans le notebook. Un dataset Kaggle attaché est prioritaire sur celui du Hub.

## 3. Entraîner sur Kaggle

1. *Create → Notebook → File → Import Notebook* : `training/betsileo_mg_nllb_kaggle.ipynb`.
2. *Settings* : Accelerator **GPU T4 x2**, Internet **On**.
3. *Add-ons → Secrets* : `HF_TOKEN` (droit d'écriture), activé pour le notebook.
4. Pour un premier essai rapide, mets `SMOKE_TEST = True` dans le bloc 2 (20 pas, quelques minutes).
5. *Save Version → Save & Run All (Commit)* : tourne en arrière-plan, 45 à 60 min sur une T4.

Le modèle est envoyé sur `nirina-doris-sylvie/nllb-betsileo-mg` (privé), avec `inference_config.json`
(codes de langue, réglages des garde-fous, lexique). Les scores sont dans `resultats.json` et les traductions
du test dans `predictions_test.tsv` (onglet *Output* de Kaggle).
