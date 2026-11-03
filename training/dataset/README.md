---
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
| train | 13394 |
| validation | 122 |
| test | 652 |

## Colonnes

- `source` : entree avec le prefixe T5 (`translate Malagasy to Betsileo: ...`)
- `target` : traduction attendue
- `src_text` : entree sans prefixe ; `src_lang` / `tgt_lang` : `mg` ou `bts`
- `direction` : `Malagasy->Betsileo` ou `Betsileo->Malagasy`
- `ref` : verset biblique (`MRK 1:31`), entree du lexique (`LEX ...`) ou vers
  de chanson (`Majeur 7/galeoka.json:6`)
- `origin` : `bible`, `lexique`, `chanson` ou `segment` (proposition courte
  decoupee d'un verset du train, train uniquement)
- `traducteur` : `humain`, ou `claude` pour les vers de chansons traduits par
  Claude et pas encore relus (train uniquement)

## Decoupage

- train : livres du Nouveau Testament hors test/validation + lexique
  (85 %) + segments courts + chansons hors validation/test.
- validation : TIT, PHM, 2JN, 3JN + chansons galeoka, velo-po.
- test : MRK, PHP, JUD + chansons Tagnitagny madigna, tsofyrano
  + 15 % du lexique (`origin = lexique`, a evaluer a part : precision mot a mot).

Les chansons sont reparties par chanson ; en validation et test, seules les
traductions humaines sont gardees.

Les versets trop religieux ont ete retires et le vocabulaire religieux
restant remplace par des mots du quotidien, avec les memes regles pour les
trois splits.

Licence : les versets viennent de traductions bibliques dont les droits
sont a verifier avant toute publication (garder le depot prive).
