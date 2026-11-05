---
title: Traduction Betsileo ByT5
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
models:
- nirina-doris-sylvie/byt5-small-betsileo-mg
short_description: Traduction betsileo <-> malgache officiel (ByT5-small)
---

# Traduction betsileo ↔ malgache officiel (ByT5-small)

Modèle : [`nirina-doris-sylvie/byt5-small-betsileo-mg`](https://huggingface.co/nirina-doris-sylvie/byt5-small-betsileo-mg)
(`google/byt5-small` fine-tuné). Une ligne = un vers : une chanson entière peut être collée d'un bloc.

Quand le modèle n'est pas sûr de lui (mot ou phrase incertain, boucle de répétitions, longueur anormale), il recopie
la phrase source au lieu d'inventer. Les réglages viennent de `inference_config.json`, à côté du modèle.

## Réglages du Space

- Matériel : **ZeroGPU** (sur CPU, ByT5 génère lettre par lettre et reste lent).
- *Settings → Variables and secrets* :
  - `HF_TOKEN` (secret) : token du compte propriétaire du modèle, qui est privé ;
  - `MODEL_ID` (facultatif) : autre dépôt du modèle, par exemple `...-final`.

## API

```python
from gradio_client import Client

client = Client("<compte>/<space>", token=HF_TOKEN)
traduction, details = client.predict("Aiza ianao no mipetraka?", "Malagasy->Betsileo", True,
                                     api_name="/translate")
lot = client.predict(["vers 1", "vers 2"], "Betsileo->Malagasy", True, api_name="/translate_batch")
# lot = {"translations": [...], "details": [...]}
```

Directions acceptées : `Malagasy->Betsileo`, `Betsileo->Malagasy`, `mg2bts`, `bts2mg`. Le troisième argument active
les garde-fous anti-hallucination.
