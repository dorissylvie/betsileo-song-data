"""Traduction betsileo <-> malgache officiel (NLLB-200 600M fine-tune) : Space Hugging Face Gradio + ZeroGPU.

Reproduit translate() du notebook betsileo_mg_nllb_kaggle.ipynb (lexique, bonus de recopie, garde-fous
mot et phrase) avec les reglages enregistres dans inference_config.json a cote du modele.
Sur ZeroGPU la traduction tourne sur GPU ; ailleurs (en local : python app.py) sur CPU, en int8.

API (gradio_client) :
  Client("<compte>/<space>", token=HF_TOKEN).predict(texte, "Malagasy->Betsileo", True, api_name="/translate")
      -> (traduction, details)
  Client(...).predict(["vers 1", "vers 2"], "Betsileo->Malagasy", True, api_name="/translate_batch")
      -> {"translations": [...], "details": [...]}

Variables d'environnement (Settings -> Variables and secrets du Space) :
  HF_TOKEN    secret, token du compte proprietaire du modele (depot prive)
  MODEL_ID    depot du modele (defaut : nirina-doris-sylvie/nllb-betsileo-mg)
  NUM_BEAMS   nombre de faisceaux (defaut : celui du notebook)
  QUANTIZE    CPU seulement : 1 = poids en int8, ~2x plus rapide (defaut) ; 0 = fp32
  THREADS     CPU seulement : threads de calcul (defaut 2)
"""
import os

# A fixer avant d'importer torch. Sur CPU partage, os.cpu_count() voit toute la machine hote : sans
# limite, torch lance des dizaines de threads qui se battent pour quelques coeurs.
THREADS = int(os.environ.get("THREADS", "2"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = str(THREADS)

ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
if ZERO_GPU:
    import spaces   # avant torch, comme le demande ZeroGPU

import contextlib
import difflib
import json
import math
import re
import threading
import time
import traceback
import unicodedata
import warnings
from collections import OrderedDict

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

MODEL_ID = os.environ.get("MODEL_ID", "nirina-doris-sylvie/nllb-betsileo-mg")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
DEVICE = "cuda" if ZERO_GPU or torch.cuda.is_available() else "cpu"
QUANTIZE = DEVICE == "cpu" and os.environ.get("QUANTIZE", "1") == "1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16" if DEVICE == "cuda" else "8"))   # phrases traduites ensemble
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "5000"))    # traductions gardees en memoire
MAX_CHARS = int(os.environ.get("MAX_CHARS", "20000"))     # par requete
torch.set_num_threads(THREADS)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def model_file(name):
    if os.path.isdir(MODEL_ID):
        return os.path.join(MODEL_ID, name)
    return hf_hub_download(MODEL_ID, name, token=HF_TOKEN)


t_load = time.time()
try:
    with open(model_file("inference_config.json"), encoding="utf-8") as f:
        CFG = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=HF_TOKEN, dtype=torch.float32).eval()
except Exception as e:
    raise RuntimeError(f"Chargement de {MODEL_ID} impossible ({type(e).__name__}: {e}). Depot prive : le secret "
                       "HF_TOKEN du Space doit venir du compte proprietaire du modele.") from e
if CFG.get("family") != "nllb":
    raise RuntimeError(f"Cette app est ecrite pour NLLB, pas pour family={CFG.get('family')!r}")

MAX_LEN = CFG["max_len"]
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", CFG["num_beams"]))
GUARD = CFG["guard"]
LEXICON = CFG["lexicon"]
LANG_CODES = CFG["lang_codes"]
LANG_ID = {k: tokenizer.convert_tokens_to_ids(code) for k, code in LANG_CODES.items()}
if tokenizer.unk_token_id in LANG_ID.values():
    raise RuntimeError(f"Codes de langue absents du tokenizer : {LANG_CODES}")
SPECIAL_SET = set(tokenizer.all_special_ids) | set(LANG_ID.values())
SPECIAL_IDS = sorted(SPECIAL_SET)
DIR_LANGS = {"mg2bts": ("mg", "bts"), "bts2mg": ("bts", "mg")}

if QUANTIZE:
    # int8 dynamique sur toutes les couches lineaires, y compris la couche de sortie de 256 000 mots :
    # a chaque token genere le CPU relit ~4x moins de donnees. Les embeddings restent en fp32.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # torch.ao.quantization est annonce obsolete, mais marche encore
            from torch.ao.quantization import quantize_dynamic
            # inplace : sinon torch copie d'abord tout le modele (2 x 2,5 Go en RAM)
            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    except Exception as e:
        QUANTIZE = False
        print("Quantification int8 impossible, modele garde en fp32 :", type(e).__name__, e)
# Sur ZeroGPU, deplacer le modele sur cuda ici est prevu : il n'arrive sur le GPU qu'a la premiere requete.
model.to(DEVICE)


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------
# Meme normalisation que build_dataset.py (sans le nettoyage propre a la Bible).
_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "“": '"',
                       "”": '"', "«": '"', "»": '"', " ": " "})


def normalize_text(text, lang):
    t = unicodedata.normalize("NFC", str(text)).translate(_APOS)
    t = re.sub(r"[\[\]*]", "", t)
    if lang == "bts":   # une seule orthographe pour la nasale, comme dans le dataset
        t = t.replace("gñ", "gn").replace("ñ", "gn").replace("Ñ", "Gn")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,;:])(?=[^\s\d\"'])", r"\1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def wnorm(word):
    return re.sub(r"[^\w'-]", "", word.lower())


def words(text):
    return [w for w in (wnorm(x) for x in text.split()) if w]


def lex_key(text):
    return " ".join(words(text))


def clean_decoded(text):
    for code in LANG_CODES.values():
        text = text.replace(code, " ")
    return re.sub(r"\s+", " ", text).strip()


def n_tokens(text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_long(line):
    # Le modele a appris sur des vers et versets de moins de MAX_LEN tokens : une ligne plus longue serait
    # coupee. On la decoupe en phrases, regroupees tant qu'elles tiennent.
    limit = MAX_LEN - 2
    if n_tokens(line) <= limit:
        return [line]
    out = []
    for part in (p for p in re.split(r"(?<=[.!?;:])\s+", line) if p):
        if out and n_tokens(out[-1] + " " + part) <= limit:
            out[-1] += " " + part
        else:
            out.append(part)
    return out


# ---------------------------------------------------------------------------
# Generation (identique au notebook, bloc 8)
# ---------------------------------------------------------------------------
class SourceCopyBias(LogitsProcessor):
    # Ajoute `bonus` au score des tokens presents dans la phrase source (hors tokens speciaux).
    def __init__(self, source_ids, bonus):
        self.source_ids = source_ids
        self.bonus = bonus
        self.add = None

    def __call__(self, input_ids, scores):
        if self.add is None or self.add.shape != scores.shape:
            mask = torch.zeros((self.source_ids.shape[0], scores.shape[-1]), device=scores.device)
            mask.scatter_(1, self.source_ids.to(scores.device), 1.0)
            mask[:, [i for i in SPECIAL_IDS if i < scores.shape[-1]]] = 0.0
            reps = scores.shape[0] // mask.shape[0]          # = nombre de faisceaux
            self.add = (mask * self.bonus).repeat_interleave(reps, dim=0)
        return scores + self.add.to(scores.dtype)


def word_probs(tok_ids, tok_lp):
    # Probabilite de chaque mot = celle de son token le moins sur.
    pieces = tokenizer.convert_ids_to_tokens(tok_ids)
    groups, new_word, pending = [], True, None
    for piece, lp in zip(pieces, tok_lp):
        if piece == " " or piece == "▁":
            new_word, pending = True, lp if piece == "▁" else None
            continue
        if piece.startswith("▁"):
            new_word = True
        if pending is not None:
            lp, pending = min(lp, pending), None
        if new_word or not groups:
            groups.append(lp)
            new_word = False
        else:
            groups[-1] = min(groups[-1], lp)
    return [math.exp(v) for v in groups]


def encode_sources(texts, src_lang):
    ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [[LANG_ID[src_lang]] + x[: MAX_LEN - 2] + [tokenizer.eos_token_id] for x in ids]


def pad_batch(seqs):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return {"input_ids": ids.to(DEVICE), "attention_mask": mask.to(DEVICE)}


@torch.inference_mode()
def generate_raw(texts, direction, copy_bonus, need_probs):
    src_lang, tgt_lang = DIR_LANGS[direction]
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = list(texts[i:i + BATCH_SIZE])
        batch = pad_batch(encode_sources(chunk, src_lang))
        gen = dict(num_beams=NUM_BEAMS, early_stopping=NUM_BEAMS > 1, forced_bos_token_id=LANG_ID[tgt_lang],
                   max_length=min(MAX_LEN, int(batch["input_ids"].shape[1] * 1.6) + 10) + 2)
        if copy_bonus:
            gen["logits_processor"] = LogitsProcessorList([SourceCopyBias(batch["input_ids"], copy_bonus)])
        # fp16 sur GPU, comme pendant l'entrainement et le reglage des garde-fous (bloc 9).
        amp = torch.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else contextlib.nullcontext()
        with amp:
            # L'encodeur ne tourne qu'une fois : sa sortie sert a la generation puis au calcul des probabilites.
            enc = model.get_encoder()(**batch)
            # generate etend la sortie de l'encodeur sur place (x faisceaux) : on lui en passe une copie.
            seqs = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
                                  attention_mask=batch["attention_mask"], **gen)
            if need_probs:
                # Probabilites du modele seul (sans le bonus), recalculees sur la sortie choisie.
                logits = model(encoder_outputs=enc, attention_mask=batch["attention_mask"],
                               decoder_input_ids=seqs[:, :-1]).logits
        hyps = [clean_decoded(t) for t in tokenizer.batch_decode(seqs, skip_special_tokens=True)]
        if not need_probs:
            results += [{"src": s, "hyp": h, "conf": None, "word_probs": []} for s, h in zip(chunk, hyps)]
            continue
        targets = seqs[:, 1:]
        lp = torch.stack([torch.log_softmax(logits[j].float(), dim=-1)
                          .gather(-1, targets[j].unsqueeze(-1)).squeeze(-1) for j in range(len(chunk))])
        for row, (ids, lps) in enumerate(zip(targets.tolist(), lp.tolist())):
            tok_ids, tok_lp = [], []
            for t, l in zip(ids, lps):
                if t == tokenizer.eos_token_id:
                    break
                if t in SPECIAL_SET:
                    continue
                tok_ids.append(t)
                tok_lp.append(l)
            results.append({"src": chunk[row], "hyp": hyps[row],
                            "conf": math.exp(sum(tok_lp) / max(1, len(tok_lp))),
                            "word_probs": word_probs(tok_ids, tok_lp)})
    return results


def gpu_duration(texts, direction, copy_bonus, need_probs):
    # Temps GPU reserve sur ZeroGPU (en s). Une reservation plus courte passe plus vite dans la file et doit
    # tenir dans le quota restant de l'utilisateur : on l'ajuste au nombre de phrases.
    return int(min(120, 10 + 0.15 * len(texts)))


if ZERO_GPU:
    # La fonction tourne dans un processus a part qui a le GPU ; elle renvoie des listes et des nombres.
    generate = spaces.GPU(duration=gpu_duration)(generate_raw)
else:
    generate = generate_raw


# ---------------------------------------------------------------------------
# Garde-fous (identiques au notebook, bloc 8)
# ---------------------------------------------------------------------------
def _transplant(src_word, hyp_word):
    # Mot source, avec la ponctuation et la majuscule du mot traduit.
    core = re.sub(r"^[^\w']+|[^\w']+$", "", src_word) or src_word
    head = re.match(r"^[^\w']*", hyp_word).group(0)
    tail = re.search(r"[^\w']*$", hyp_word).group(0)
    if hyp_word[len(head):len(head) + 1].isupper():
        core = core[:1].upper() + core[1:]
    return head + core + tail


def _similar(a, b):
    return difflib.SequenceMatcher(None, wnorm(a), wnorm(b), autojunk=False).ratio()


def word_guard(src, hyp, probs, threshold):
    # Un mot traduit incertain redevient le mot source, seulement quand la correspondance est sure.
    hw, sw = hyp.split(), src.split()
    if threshold <= 0 or len(hw) != len(probs):
        return hyp, []
    out, copied = [], []
    ops = difflib.SequenceMatcher(None, [wnorm(w) for w in sw], [wnorm(w) for w in hw], autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag != "replace":
            out.extend(hw[j1:j2])
        elif all(p < threshold for p in probs[j1:j2]):
            out.extend(sw[i1:i2])
            copied.append((" ".join(hw[j1:j2]), " ".join(sw[i1:i2])))
        elif (i2 - i1) == (j2 - j1):
            anchored = (j2 - j1) == 1
            for k in range(j2 - j1):
                s, h = sw[i1 + k], hw[j1 + k]
                if probs[j1 + k] < threshold and (anchored or _similar(s, h) >= 0.5):
                    out.append(_transplant(s, h))
                    copied.append((h, s))
                else:
                    out.append(h)
        else:
            out.extend(hw[j1:j2])
    return " ".join(out), copied


LOOP_RE = re.compile(r"(\b\S+(?:\s+\S+){0,3}?)(?:\s+\1){3,}")


def sentence_guard(src, hyp, conf, threshold):
    if not hyp.strip():
        return src, "sortie vide"
    if LOOP_RE.search(hyp):
        return src, "boucle de repetitions"
    ratio = len(hyp) / max(1, len(src))
    if len(src) >= 15 and not 0.4 <= ratio <= 2.5:
        return src, f"longueur anormale ({ratio:.2f})"
    if conf is not None and conf < threshold:
        return src, f"phrase incertaine (proba {conf:.2f})"
    return hyp, None


def apply_guards(r, word_min, sent_min):
    hyp, copied = word_guard(r["src"], r["hyp"], r["word_probs"], word_min)
    hyp, reason = sentence_guard(r["src"], hyp, r["conf"], sent_min)
    return hyp, copied, reason


# ---------------------------------------------------------------------------
# Traduction : lexique, cache, lots
# ---------------------------------------------------------------------------
_cache = OrderedDict()
_cache_lock = threading.Lock()
# Sur CPU, une generation a la fois : deux en parallele sur 2 vCPU vont chacune 2x moins vite.
# Sur ZeroGPU, chaque requete a son propre GPU : pas de verrou.
_model_lock = threading.Lock() if DEVICE == "cpu" else contextlib.nullcontext()


def translate_segments(texts, direction, guard=True):
    out = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        hit = LEXICON[direction].get(lex_key(t))
        if hit:
            hit = hit[:1].upper() + hit[1:] if t[:1].isupper() else hit
            out[i] = {"traduction": hit, "methode": "lexique", "raison": None, "mots_recopies": [], "confiance": None}
            continue
        with _cache_lock:
            cached = _cache.get((direction, guard, t))
            if cached is not None:
                _cache.move_to_end((direction, guard, t))
        if cached is not None:
            out[i] = cached
        else:
            todo.append(i)
    # Un refrain repete n'est traduit qu'une fois ; tri par longueur pour des lots sans remplissage inutile.
    unique = sorted({texts[i] for i in todo}, key=len)
    if unique:
        need_probs = guard and (GUARD["word_min"] > 0 or GUARD["sent_min"] > 0)
        with _model_lock:
            raw = generate(unique, direction, GUARD["copy_bonus"] if guard else 0.0, need_probs)
        done = {}
        for t, r in zip(unique, raw):
            if guard:
                hyp, copied, reason = apply_guards(r, GUARD["word_min"], GUARD["sent_min"])
            else:
                hyp, copied, reason = r["hyp"], [], None
            done[t] = {"traduction": hyp, "methode": "recopie" if reason else "modele", "raison": reason,
                       "mots_recopies": copied, "confiance": None if r["conf"] is None else round(r["conf"], 3)}
        with _cache_lock:
            for t, res in done.items():
                _cache[(direction, guard, t)] = res
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
        for i in todo:
            out[i] = done[texts[i]]
    return out


def translate_texts(texts, direction, guard=True):
    # Une chanson collee d'un bloc : chaque ligne est traduite a part (le modele a appris vers par vers),
    # toutes les lignes de tous les textes dans les memes lots. Les lignes vides sont gardees.
    src_lang = DIR_LANGS[direction][0]
    segs, owner, n_lines = [], [], []
    for n, text in enumerate(texts):
        lines = text.split("\n")
        n_lines.append(len(lines))
        for k, line in enumerate(lines):
            line = normalize_text(line, src_lang)
            for seg in (split_long(line) if line else []):
                segs.append(seg)
                owner.append((n, k))
    res = translate_segments(segs, direction, guard) if segs else []
    parts = [[[] for _ in range(m)] for m in n_lines]
    details = [[] for _ in texts]
    for (n, k), r in zip(owner, res):
        parts[n][k].append(r["traduction"])
        details[n].append(r)
    return ["\n".join(" ".join(p) for p in lines) for lines in parts], details


_DIR_RE = re.compile(r"(malagasy|mg|betsileo|bts)\s*(?:->|→|>|2|\bto\b|\bvers\b)\s*(malagasy|mg|betsileo|bts)")


def parse_direction(value):
    # Accepte "Malagasy->Betsileo" (format du dataset et de l'ancienne API), "mg2bts", "translate Malagasy to Betsileo"...
    m = _DIR_RE.search(str(value).strip().lower())
    if m:
        src, tgt = ("bts" if x in ("betsileo", "bts") else "mg" for x in m.groups())
        if src != tgt:
            return f"{src}2{tgt}"
    raise gr.Error(f"direction inconnue : {value!r}. Valeurs acceptees : "
                   "'Malagasy->Betsileo', 'Betsileo->Malagasy', 'mg2bts', 'bts2mg'.")


def run(texts, direction, guard):
    direction = parse_direction(direction)
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise gr.Error("texts doit etre une liste de textes")
    if sum(len(t) for t in texts) > MAX_CHARS:
        raise gr.Error(f"texte trop long (max {MAX_CHARS} caracteres par requete)")
    t0 = time.time()
    try:
        out = translate_texts(texts, direction, bool(guard))
    except gr.Error:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise gr.Error(f"{type(e).__name__}: {e}")
    print(f"[{direction}] {sum(len(d) for d in out[1])} segment(s) en {time.time() - t0:.2f} s", flush=True)
    return out


def translate(text, direction, guard=True):
    translations, details = run([text or ""], direction, guard)
    return translations[0], details[0]


def translate_batch(texts, direction, guard=True):
    translations, details = run(texts, direction, guard)
    return {"translations": translations, "details": details}


if DEVICE == "cpu":
    # Premier passage a vide : les noyaux de calcul s'initialisent maintenant, pas a la premiere vraie requete.
    # (Sur ZeroGPU il n'y a pas de GPU au demarrage, et ce passage consommerait du quota.)
    generate_raw(["Manao ahoana ianao?"], "mg2bts", GUARD["copy_bonus"], True)
STATUS = (f"`{MODEL_ID}` | {'ZeroGPU' if ZERO_GPU else DEVICE}{' int8' if QUANTIZE else ''} | {NUM_BEAMS} faisceaux | "
          f"garde-fous {GUARD} | lexique {', '.join(f'{d} {len(v)}' for d, v in LEXICON.items())}")
print(f"Pret en {time.time() - t_load:.0f} s : {STATUS}", flush=True)


# ---------------------------------------------------------------------------
# Interface et API
# ---------------------------------------------------------------------------
DIRECTIONS = ["Malagasy->Betsileo", "Betsileo->Malagasy"]

with gr.Blocks(title="Traduction betsileo") as demo:
    gr.Markdown("# Traduction betsileo ↔ malgache officiel\nUne ligne = un vers : une chanson entière peut être "
                f"collée d'un bloc.\n\n<small>{STATUS}</small>")
    with gr.Row():
        # allow_custom_value : l'API accepte aussi "mg2bts", "bts2mg"...
        direction = gr.Dropdown(DIRECTIONS, value=DIRECTIONS[0], allow_custom_value=True, label="Direction")
        guard = gr.Checkbox(value=True, label="Garde-fous anti-hallucination")
    with gr.Row():
        text = gr.Textbox(lines=10, label="Texte")
        translation = gr.Textbox(lines=10, label="Traduction")
    button = gr.Button("Traduire", variant="primary")
    with gr.Accordion("Détails par ligne", open=False):
        details = gr.JSON(label="Détails")
    gr.Examples([["Tsy fantatro izay nolazainy.", DIRECTIONS[0]], ["Aiza ianao no mipetraka?", DIRECTIONS[0]],
                 ["Tsa fantatsiko izay notarogniny.", DIRECTIONS[1]]], inputs=[text, direction])
    button.click(translate, [text, direction, guard], [translation, details], api_name="translate")
    text.submit(translate, [text, direction, guard], [translation, details], api_name=False)

    # API seule : plusieurs textes d'un coup (ex. tous les vers d'une chanson), dans les memes lots.
    batch_texts = gr.JSON(visible=False)
    batch_out = gr.JSON(visible=False)
    gr.Button(visible=False).click(translate_batch, [batch_texts, direction, guard], batch_out,
                                   api_name="translate_batch")

if __name__ == "__main__":
    demo.launch()
