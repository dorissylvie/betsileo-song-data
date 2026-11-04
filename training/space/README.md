---
title: Traduction Betsileo
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.31.0"
app_file: app.py
pinned: false
---

# Traduction betsileo <-> malgache officiel

NLLB-200 600M fine-tune (`nirina-doris-sylvie/nllb-betsileo-mg`), avec lexique et garde-fous anti-hallucination,
sur ZeroGPU.

API (Python, `pip install gradio_client`) :

```python
from gradio_client import Client
client = Client("<compte>/<space>", token=HF_TOKEN)
traduction, details = client.predict("Tsy fantatro izay nolazainy.", "Malagasy->Betsileo", True, api_name="/translate")
lot = client.predict(["vers 1", "vers 2"], "Betsileo->Malagasy", True, api_name="/translate_batch")
```

Secret requis : `HF_TOKEN` du compte proprietaire du modele.
