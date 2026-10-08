# pictoweb/clip_local.py
# ------------------------------------------------------------
# Envoltura LOCAL del notebook de CLIP para Flask.
# - Lógica de scoring/selección DEL NOTEBOOK igual.
# - Añadidos:
#     * ensure_loaded(): carga modelos una sola vez y toma PICTOS_DIR de Flask.
#     * assemble_one(oracion, tokens, candmap): procesa UNA oración con candidatos.
# - NO lee JSON_PATH ni escribe OUT_JSON (la web no lo necesita).
# ------------------------------------------------------------
from __future__ import annotations
import os, re, json, math, unicodedata, csv, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from unidecode import unidecode
from flask import current_app

# --- Rutas (el PICTOS_DIR REAL se fija en ensure_loaded() desde app.config)
JSON_PATH  = "/dev/null"               # ignorado en web
PICTOS_DIR = "/tmp/pictos"             # será reemplazado
OUT_JSON   = "/dev/null"               # ignorado en web

# --- Modelos CLIP (texto/imágenes)
TEXT_MODEL_NAME  = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # M-CLIP (texto)
IMAGE_MODEL_NAME = "clip-ViT-B-32"                                        # CLIP visión

# --- Heurísticas (idénticas al notebook)
BONUS_MATCH_EXACTO =  0.12
BONUS_SUBCADENA    =  0.03
BONUS_UNIPALABRA   =  0.08
BONUS_FRASE_NGRAM  =  0.18
BONUS_SUFFIX_TOKEN =  0.02
PENALIZA_PREGUNTAS = -0.08
PENALIZA_PROHIBIDO = -0.12
PENALIZA_COMPUESTO_LARGO = -0.06

PENALIZA_CATEGORIA = {
    "fuerte": [("caja_fuerte", -0.40)]
}

STRICT_NO_DUP = True
FALLBACK_IF_ALL_DUP = True

INCLUIR_FUNCIONALES = True
FUNCIONALES = set("de del la el los las un una y o u a en con por para sin al del lo le les me te se que".split())

# =========================
# 2) Utilidades de texto/labels (idénticas)
# =========================
def norm(s: str) -> str:
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('¿','').replace('¡','')
    s = s.replace('_',' ').replace('-',' ')
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fname_to_label(fname: str) -> str:
    stem = Path(fname).stem
    stem = re.sub(r'^\d+-', '', stem)
    return stem.replace('_',' ').replace('-',' ').strip()

def canonical_label_key_from_fname(fname: str) -> str:
    stem = Path(fname).stem
    stem = re.sub(r'^\d+-', '', stem)
    s = stem.replace('_',' ').replace('-',' ').lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s)
    return s

def tokens_norm(tokens):
    return [norm(t) for t in tokens]

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def label_norm(label):
    return norm(label)

def contiene_frase(label_n: str, frase_n: str) -> bool:
    return f" {frase_n} " in f" {label_n} "

# =========================
# 3) Carga modelos (idéntica API, pero sin leer JSON)
# =========================
text_model: SentenceTransformer | None = None
image_model: SentenceTransformer | None = None

def emb_text(texts):
    return text_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64
    )

def emb_images(pil_list):
    return image_model.encode(
        pil_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64
    )

# --- Caches (idénticos)
img_vec_cache  = {}
text_vec_cache = {}
label_cache    = {}


def clear_runtime_cache():
    """Limpia caches de embeddings (texto/imagen) y labels."""
    img_vec_cache.clear()
    text_vec_cache.clear()
    label_cache.clear()

def get_text_vec(s: str):
    # cache por-llamada (se vacía en assemble_with_clip)
    if s in text_vec_cache:
        return text_vec_cache[s]
    v = emb_text([s])[0]
    text_vec_cache[s] = v
    return v

def get_label(fname: str):
    if fname not in label_cache:
        label_cache[fname] = fname_to_label(fname)
    return label_cache[fname]

def get_image_vec(fname: str):
    if fname in img_vec_cache:
        return img_vec_cache[fname]
    path = os.path.join(PICTOS_DIR, fname)
    img  = Image.open(path).convert("RGB")
    v    = emb_images([img])[0]
    img_vec_cache[fname] = v
    return v


# =========================
# 4) Heurísticas multimodales (idénticas)
# =========================
def frase_bonus_desde_contexto(label_n: str, toks_n: list) -> float:
    bonus = 0.0
    bigr  = ngrams(toks_n, 2)
    trigr = ngrams(toks_n, 3)
    bigr_de = []
    for b in bigr:
        w = b.split()
        if len(w) == 2:
            bigr_de.append(f"{w[0]} de {w[1]}")
    for frase in trigr + bigr + bigr_de:
        if contiene_frase(label_n, frase):
            bonus = max(bonus, BONUS_FRASE_NGRAM)
    return bonus

def reglas_heuristicas(token_raw, token_n, cand_fname, cand_label_raw, cand_label_n, toks_n):
    score = 0.0
    if token_n == cand_label_n:
        score += BONUS_MATCH_EXACTO
        if len(cand_label_raw.split()) == 1:
            score += BONUS_UNIPALABRA
    if re.search(rf"(?:^|\s){re.escape(token_n)}(?:\s|$)", cand_label_n):
        score += BONUS_SUBCADENA
    score += frase_bonus_desde_contexto(cand_label_n, toks_n)
    if cand_label_n.endswith(f" {token_n}"):
        score += BONUS_SUFFIX_TOKEN
    lname = cand_label_n
    has_no = re.search(r"\bno\b", lname) is not None
    if (("prohibido" in lname) or has_no) and token_n not in {"no","prohibido"}:
        score += PENALIZA_PREGUNTAS  # ojo: el notebook usaba penalizaciones separadas; mantenemos mismas constantes
    if "?" in cand_label_raw or "¿" in cand_label_raw:
        score += PENALIZA_PREGUNTAS
    for clave, pares in PENALIZA_CATEGORIA.items():
        if token_n == clave:
            for fragmento, pena in pares:
                if fragmento in lname:
                    score += pena
    if token_n.endswith("a") and lname.endswith("o") and any(w in token_n for w in ["enfermer","medic"]):
        score -= 0.03
    if token_n.endswith("o") and lname.endswith("a") and any(w in token_n for w in ["enfermer","medic"]):
        score -= 0.03
    if len(cand_label_raw.split()) >= 3:
        score += PENALIZA_COMPUESTO_LARGO
    return score

# =========================
# 5) Parámetros/Utilidades (idénticos)
# =========================
W_SENT   = 0.45
W_PROMPT = 0.35
W_TOK2   = 0.20
W_PHRASE = 0.12

EPS_SELECT = 0.02

def extraer_ancla_semantica(tokens_raw, i):
    """
    Devuelve una palabra de contenido cercana al token i (o None).
    Prioriza derecha hasta 3 posiciones; si no, izquierda hasta 3.
    Ignora funcionales definidas en FUNCIONALES.
    """
    def es_contenido(t):
        tn = norm(t)
        return tn and (tn not in FUNCIONALES) and len(tn) > 2

    # 1) primero a la derecha
    for j in range(i+1, min(i+4, len(tokens_raw))):
        if es_contenido(tokens_raw[j]):
            return tokens_raw[j]

    # 2) si no hubo, probar a la izquierda
    for j in range(i-1, max(-1, i-4), -1):
        if es_contenido(tokens_raw[j]):
            return tokens_raw[j]

    return None


def pick_determinista_por_oracion(oracion_raw, candidatos, eps=EPS_SELECT):
    if not candidatos:
        return None
    candidatos = sorted(candidatos, key=lambda x: x[0], reverse=True)
    best = candidatos[0][0]
    pool = [c for c in candidatos if c[0] >= best - eps]
    if len(pool) == 1:
        return pool[0][1]
    h = hashlib.md5(oracion_raw.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(pool)
    return pool[idx][1]

# =========================
# 6) Selector por token (idéntico)
# =========================
def escoger_candidato(token_raw, oracion_raw, candidatos_fnames, toks_n,
                      elegidos_file_set, elegidos_label_set,
                      tokens_raw_list, token_index):
    token_n = norm(token_raw)
    if (not INCLUIR_FUNCIONALES) and (token_n in FUNCIONALES):
        return None, {"razon": "token_funcional_descartado"}
    if not candidatos_fnames:
        return None, {"razon": "sin_candidatos"}

    sent_vec   = get_text_vec(oracion_raw)
    prompt_vec = get_text_vec(f"Pictograma de {token_raw} en el contexto: {oracion_raw}")
    tok_vec    = get_text_vec(token_raw)

    anchor = extraer_ancla_semantica(tokens_raw_list, token_index)
    anchor_vec = get_text_vec(f"{token_raw} {anchor}") if anchor else None

    cand_list = []
    for fname in candidatos_fnames:
        if fname in elegidos_file_set:
            continue
        try:
            c_vec = get_image_vec(fname)
        except FileNotFoundError:
            continue
        label_raw = get_label(fname)
        label_n   = label_norm(label_raw)
        label_key = canonical_label_key_from_fname(fname)
        if STRICT_NO_DUP and (label_key in elegidos_label_set):
            continue
        cand_list.append((fname, c_vec, label_raw, label_n, label_key))

    if not cand_list and FALLBACK_IF_ALL_DUP:
        for fname in candidatos_fnames:
            if fname in elegidos_file_set:
                continue
            try:
                c_vec = get_image_vec(fname)
            except FileNotFoundError:
                continue
            label_raw = get_label(fname)
            label_n   = label_norm(label_raw)
            label_key = canonical_label_key_from_fname(fname)
            cand_list.append((fname, c_vec, label_raw, label_n, label_key))
        if not cand_list:
            return None, {"razon": "sin_candidatos_validos"}

    scored = []
    for fname, c_vec, label_raw, label_n, label_key in cand_list:
        s_sent   = float(np.dot(sent_vec,   c_vec))
        s_prompt = float(np.dot(prompt_vec, c_vec))
        s_tok2   = float(np.dot(tok_vec,    c_vec))
        s_phrase = float(np.dot(anchor_vec, c_vec)) if anchor_vec is not None else 0.0

        score = (W_SENT * s_sent) + (W_PROMPT * s_prompt) + (W_TOK2 * s_tok2)
        if anchor_vec is not None:
            score += W_PHRASE * s_phrase

        score += reglas_heuristicas(token_raw, token_n, fname, label_raw, label_n, toks_n)
        scored.append((score, (fname, label_raw, label_key)))

    ganador = pick_determinista_por_oracion(oracion_raw, scored, eps=EPS_SELECT)
    if ganador is None:
        return None, {"razon": "sin_candidatos_validos"}

    fname, label_raw, label_key = ganador
    elegidos_file_set.add(fname)
    elegidos_label_set.add(label_key)
    return {"archivo": fname, "label": label_raw}, None

# =========================
# 7) Envolturas para Flask
# =========================
_STATE = {"loaded": False}

def ensure_loaded():
    """Carga modelos y fija PICTOS_DIR desde Flask solo una vez."""
    if _STATE["loaded"]:
        return
    cfg = current_app.config
    pictos = cfg.get("PICTOS_DIR")
    if not pictos or not os.path.isdir(pictos):
        raise RuntimeError("PICTOS_DIR no configurado o inexistente.")
    globals()["PICTOS_DIR"] = pictos

    global text_model, image_model
    text_model  = SentenceTransformer(TEXT_MODEL_NAME)
    image_model = SentenceTransformer(IMAGE_MODEL_NAME)

    _STATE["loaded"] = True
    print(f"[CLIP] Modelos cargados. PICTOS_DIR={PICTOS_DIR}")

def assemble_one(oracion: str, tokens: list[str], candmap: dict[str, list[str]]) -> dict:
    """
    Entrada (desde la web):
      - oracion: str
      - tokens:  lista de tokens visibles (ordenados)
      - candmap: dict token -> [fname1, fname2, ...] (candidatos de BETO)
    Salida (misma estructura que tu notebook):
      {
        "oracion": oracion,
        "tokens": tokens,
        "elecciones": [...],
        "secuencia_final": [...]
      }
    """
    ensure_loaded()
    toks = list(tokens or [])
    toks_n = tokens_norm(toks)
    elecciones = []
    elegidos_file_set  = set()
    elegidos_label_set = set()

    for i, tok in enumerate(toks):
        cands = (candmap or {}).get(tok, [])
        valid_cands = [c for c in cands if c and isinstance(c, str)]
        elegido, _meta = escoger_candidato(
            tok, oracion, valid_cands, toks_n,
            elegidos_file_set, elegidos_label_set,
            tokens_raw_list=toks, token_index=i
        )
        elecciones.append(elegido["archivo"] if elegido else None)

    return {
        "oracion": oracion,
        "tokens": toks,
        "elecciones": elecciones,
        "secuencia_final": list(elecciones)
    }

