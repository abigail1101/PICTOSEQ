# pictoweb/beto_local.py
# ------------------------------------------------------------
# Envoltura para usar el notebook donde esta BETO en local dentro de Flask.
# - NO se la lógica del pipeline: mismas constantes y funciones.
# - Solo se añadio:
#     * ensure_loaded(): carga modelo/índices/embeddings una vez
#     * infer_sentence(oracion, dominio): ejecuta pipeline 1-oración
# - Usa PICTOS_DIR y BETO_CACHE_EMB desde app.config
# ------------------------------------------------------------
from __future__ import annotations
import os, re, json, numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from flask import current_app
from unidecode import unidecode
from rapidfuzz import fuzz

# ===========================
# 1) CONSTANTES DEL NOTEBOOK
# ===========================
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

# Parámetros de ranking
K_POR_TOKEN   = 6
M_POOL        = 60
UMBRAL_SCORE  = 0.52

# Reglas para tokens cortos / fuzzy
LONG_MIN_FUZZY      = 6
TOKEN_CORTO_EXACTO  = 3

# Stopwords y puntuación
OMITIR_STOPWORDS_EN_SALIDA = False
STOP_TOKENS = set("""
 el las un para
, . ; : ! ? ¿ ¡ ( ) - ' " _
""".split())

# Allowlist
ALLOWLIST_STOPWORDS_RAW = {
    'a','ahí','al','algunos','ante','antes','aquel','aquellos','aquí','aunque',
    'ayer','bajo','como','con','contra','cuando','cuándo','cómo','de','del',
    'demasiado','desde','después','durante','dónde','e','ella','ellas','ellos',
    'en','entre','esos','estar','este','estos','haber','hacia','hasta','hoy',
    'la','los','mañana','más','menos','mis','mucho','muchos','muy','ni','ninguno',
    'ningún','no','nosotras','nosotros','nuestro','nuestros','nunca','o','otro',
    'otros','para','pero','poco','pocos','por','porque','que','qué','quién',
    'quiénes','según','ser','seria','si','sí','siempre','sin','sobre','su','sus',
    'tantos','todo','todos','tras','tu','tus','tú','una','unas','uno','unos',
    'vía','vosotras','vosotros','vuestro','vuestros','y','ya','yo'
}
ALLOWLIST_STOPWORDS = None
FORZAR_INCLUSION_ALLOWLIST_VACIOS = True

# Sinónimos base
SINONIMOS_BASE = {
    "baño": {"banio","bano","wc","aseo","lavabo","servicio","sanitario","vater"},
    "dolor": {"doler","molestia"},
    "maestro": {"profesor","profe","docente"},
    "tarea": {"deberes","trabajo_escolar"},
    "comida": {"alimentacion","alimento"},
    "escuela": {"colegio"},
    "hablar": {"platicar"},
    "enfermera": {"enfermero"},
    "doctor": {"medico"},
    # empujes cortos
    "si": {"sí"},
    "no": {"prohibido"},
    "hoy": {"dia_de_hoy"},
    "manana": {"mañana"},
    "ayer": {"dia_de_ayer"},
    "aqui": {"aqui","aquí"},
    "ahi": {"ahi","ahí"},
    "donde": {"donde","dónde"},
    "cuando": {"cuando","cuándo"},
    "como": {"como","cómo"},
    "que": {"que","qué"},
    "quien": {"quien","quién"},
}

# Semillas de n-gramas
SEMILLAS_NGRAM = {
    "por favor","buenos dias","buenas tardes","buenas noches",
    "cuarto de baño","lavar las manos","tomar medicina","me duele",
    "no entiendo","necesito ayuda","quiero ir","quiero agua",
}

# Filtro sensible
BLOCKLIST_GLOBAL = {"genital","porn","violac","violación","estupro","mastur","erec","orgas","coito","sexo"}
BLOCKLIST_POR_DOMINIO = {
    "educacion": BLOCKLIST_GLOBAL | {"alcohol","droga"},
    "cultura":   BLOCKLIST_GLOBAL,
    "salud":     BLOCKLIST_GLOBAL,
}

# Boost por dominio
BOOST_DOMINIO = {
    "educacion": {"escuela","clase","colegio","maestro","profesor","tarea","recreo","examen","libro","cuaderno"},
    "cultura":   {"baile","mariachi","comida","tortilla","fiesta","tradicion","museo","danza","musica","artesania"},
    "salud":     {"dolor","medico","enfermera","hospital","curita","termometro","inyeccion","medicina","sangre","vendaje"},
}

# Bonos / penalizaciones
W_COOCC = 0.06
W_BIGRAM = 0.08
W_BOOST_DOM = 0.03

EXACT_UNIGRAM_BONUS      = 3.5
UNIGRAM_PREFER_BONUS     = 0.8
HEAD_POS_BONUS           = 0.25
LONG_COMPOUND_PENALTY    = 0.40
ADJ_VERB_NOUN_PAIR_BONUS = 0.15

# ==================================
# 2) FUNCIONES DEL NOTEBOOK 
# ==================================
USE_SPACY = True
nlp = None  # se setea en ensure_loaded()

def norm_text(s: str) -> str:
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = unidecode(s)
    return s

# construir allowlist normalizada
ALLOWLIST_STOPWORDS = { norm_text(w) for w in ALLOWLIST_STOPWORDS_RAW }

def es_puntuacion(token: str) -> bool:
    return bool(re.fullmatch(r"[\W_]+", token))

def es_stop(token: str) -> bool:
    t = norm_text(token)
    return (t in STOP_TOKENS) and (t not in ALLOWLIST_STOPWORDS)

def es_stop_or_punt(token: str) -> bool:
    return es_puntuacion(token) or es_stop(token)

def tokens_de_etiqueta(et_norm: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", et_norm)

def lematiza_tokens(tokens: List[str]) -> List[str]:
    if nlp is None:
        return tokens
    doc = nlp(" ".join(tokens))
    out = []
    for tok in doc:
        if tok.text.strip():
            out.append(tok.lemma_)
    return out

# --------- Embeddings (usa tokenizer/model globales) ----------
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cpu"       # se setea en ensure_loaded()
tokenizer = None     # se setea en ensure_loaded()
model = None         # se setea en ensure_loaded()

@torch.no_grad()
def embed_sentence_cls(textos: List[str] | str) -> np.ndarray:
    single = False
    if isinstance(textos, str):
        textos = [textos]
        single = True
    vs = []
    bs = 64
    for i in range(0, len(textos), bs):
        batch = textos[i:i+bs]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
        out = model(**enc)
        hs = out.hidden_states
        cls = torch.stack([hs[-1][:,0], hs[-2][:,0], hs[-3][:,0], hs[-4][:,0]], dim=0).mean(dim=0)
        v = torch.nn.functional.normalize(cls, p=2, dim=1)
        vs.append(v.detach().cpu().numpy())
    V = np.vstack(vs) if vs else np.zeros((0,768), dtype=np.float32)
    return V[0] if (single and V.shape[0] == 1) else V

@torch.no_grad()
def embed_tokens_en_context(oracion: str, palabras: List[str]) -> List[np.ndarray]:
    enc = tokenizer(oracion, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=True, truncation=True, max_length=256)
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(**enc)
    hs = out.hidden_states

    spans_pal, idx = [], 0
    for w in palabras:
        j = oracion.lower().find(w.lower(), idx)
        if j == -1: j = idx
        spans_pal.append((j, j+len(w)))
        idx = j + len(w)

    token_vecs = []
    for (a, b) in spans_pal:
        sub_idxs = [i for i, (x, y) in enumerate(offs) if not (x == 0 and y == 0) and (max(a, x) < min(b, y))]
        if not sub_idxs:
            v = embed_sentence_cls(oracion)
            token_vecs.append(v)
            continue
        sub_stack = []
        for layer in (-1, -2, -3, -4):
            layer_out = hs[layer].squeeze(0)
            sel = layer_out[sub_idxs, :].mean(dim=0)
            sub_stack.append(sel)
        v = torch.stack(sub_stack, dim=0).mean(dim=0)
        v = torch.nn.functional.normalize(v, p=2, dim=0)
        token_vecs.append(v.detach().cpu().numpy())
    return token_vecs

# --------- Indexado de pictos + embeddings ----------
def _parse_nombre_picto(fname: str):
    """
    Acepta dos esquemas:
      1) '1234-nombre.jpg'   (el del notebook)
      2) '1234.nombre.jpg'   (flexibilización ligera; no cambia lógica del ranking)
    Devuelve (pid:str, etiqueta_sin_ext:str) o None si no coincide.
    """
    base = fname
    if "-" in base:
        try:
            pid, name = base.split("-", 1)
            int(pid)
            return pid, name.rsplit(".",1)[0]
        except Exception:
            return None
    if "." in base:
        # fallback: 1234.nombre.jpg -> pid='1234', etiqueta='nombre'
        parts = base.split(".")
        if len(parts) >= 3 and parts[0].isdigit():
            pid = parts[0]
            etiqueta = ".".join(parts[1:-1])
            return pid, etiqueta
    return None

def cargar_pictos_y_indexar(dir_pictos: str):
    etiquetas_txt: Dict[str, str] = {}
    etiquetas_norm: Dict[str, str] = {}
    idx_token: Dict[str, Set[str]] = defaultdict(set)
    idx_frase: Dict[str, Set[str]] = defaultdict(set)
    exacto_unigrama_map: Dict[str, Set[str]] = defaultdict(set)

    files = [f for f in os.listdir(dir_pictos) if f.lower().endswith((".jpg",".jpeg",".png"))]
    for fname in files:
        parsed = _parse_nombre_picto(fname)
        if not parsed:
            continue
        pid, etiqueta = parsed
        et_norm = norm_text(etiqueta)

        etiquetas_txt[pid]  = etiqueta
        etiquetas_norm[pid] = et_norm

        toks = tokens_de_etiqueta(et_norm)
        for t in toks:
            idx_token[t].add(pid)

        if " " in et_norm:
            idx_frase[et_norm].add(pid)

        if len(toks) == 1 and toks[0] == et_norm:
            exacto_unigrama_map[et_norm].add(pid)

    return etiquetas_txt, etiquetas_norm, idx_token, idx_frase, exacto_unigrama_map

def construir_embeddings_etiquetas(etiquetas_txt: Dict[str,str], cache_path: str|None=None) -> Dict[str, np.ndarray]:
    emb: Dict[str, np.ndarray] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True).item()
            emb = {k: np.array(v) for k, v in loaded.items()}
            print(f"[BETO] Cache embeddings cargado: {cache_path} (#{len(emb)})")
        except Exception as e:
            print("[BETO] Cache inválido, se reconstruirá:", e)

    faltan = [pid for pid in etiquetas_txt if pid not in emb]
    if faltan:
        textos = [etiquetas_txt[pid].replace("_"," ") for pid in faltan]
        V = embed_sentence_cls(textos)
        for pid, v in zip(faltan, V):
            emb[pid] = v
        if cache_path:
            np.save(cache_path, emb, allow_pickle=True)
            print(f"[BETO] Cache embeddings guardado: {cache_path} (#{len(emb)})")
    return emb

# --------- N-gramas / filtros / pool léxico ----------
def detectar_ngramas(oracion_norm_toks: List[str], idx_frase: Dict[str,Set[str]]) -> List[Tuple[int,int,str,List[str]]]:
    n = len(oracion_norm_toks)
    spans = []
    ocupado = [False]*n
    frases_bd = set(idx_frase.keys())

    for L in [4,3,2]:
        i = 0
        while i <= n - L:
            if any(ocupado[i:i+L]):
                i += 1; continue
            frag_norm = " ".join(oracion_norm_toks[i:i+L])
            ids = None
            if frag_norm in frases_bd:
                ids = list(idx_frase[frag_norm])
            elif frag_norm in SEMILLAS_NGRAM:
                mejor, best = None, 0
                for f in frases_bd:
                    r = fuzz.ratio(frag_norm, f)
                    if r > best:
                        best, mejor = r, f
                if best >= 90 and mejor is not None:
                    ids = list(idx_frase[mejor])
                    frag_norm = mejor
            if ids:
                spans.append((i, i+L, frag_norm, ids))
                for k in range(i, i+L): ocupado[k] = True
                i += L
            else:
                i += 1
    spans.sort(key=lambda x: x[0])
    return spans

def _token_pasa_blocklist(etiqueta_norm: str, dominio: str|None) -> bool:
    bl = set(BLOCKLIST_GLOBAL)
    if dominio and dominio in BLOCKLIST_POR_DOMINIO:
        bl |= BLOCKLIST_POR_DOMINIO[dominio]
    return not any(bad in etiqueta_norm for bad in bl)

def pool_lexico(token: str, idx_token: Dict[str,Set[str]], etiquetas_norm: Dict[str,str],
                M: int=M_POOL, dominio: str|None=None) -> List[str]:
    t = norm_text(token)
    cand: Set[str] = set()

    if t in idx_token:
        cand |= idx_token[t]
    for s in SINONIMOS_BASE.get(t, set()):
        if s in idx_token:
            cand |= idx_token[s]

    if len(t) >= LONG_MIN_FUZZY and len(cand) < M:
        scored = []
        for pid, en in etiquetas_norm.items():
            if not _token_pasa_blocklist(en, dominio): continue
            toks = tokens_de_etiqueta(en)
            best = max((fuzz.ratio(t, tok) for tok in toks), default=0)
            if best >= 92 and abs(len(t) - max((len(tok) for tok in toks), default=len(t))) <= 1:
                scored.append((best, pid))
        scored.sort(reverse=True)
        for _, pid in scored:
            cand.add(pid)
            if len(cand) >= M: break

    cand_filtrado = [pid for pid in cand if _token_pasa_blocklist(etiquetas_norm[pid], dominio)]
    return cand_filtrado[:max(M, len(cand_filtrado))]

# --------- Scoring / bonos ----------
def score_lexico_base(token_norm: str, etiqueta_norm: str) -> float:
    t = token_norm
    toks = tokens_de_etiqueta(etiqueta_norm)
    if t in toks: return 1.0
    if t in SINONIMOS_BASE and any(s in toks for s in SINONIMOS_BASE[t]): return 0.9
    best = max((fuzz.ratio(t, tok) for tok in toks), default=0) / 100.0
    return min(best, 0.70)

def penalty_pos(token_norm: str, etiqueta_norm: str) -> float:
    e0 = tokens_de_etiqueta(etiqueta_norm)[:1]
    e0 = e0[0] if e0 else ""
    es_verbo_t = token_norm.endswith(("ar","er","ir")) or token_norm in {"ser","estar","haber","ir"}
    es_verbo_e = e0.endswith(("ar","er","ir"))
    return 1.0 if es_verbo_t and not es_verbo_e else 0.0

def construir_bigrama_set(tokens_norm: List[str]) -> Set[Tuple[str,str]]:
    toks = [t for t in tokens_norm if t and ((t not in STOP_TOKENS) or (t in ALLOWLIST_STOPWORDS))]
    return set(zip(toks, toks[1:])) if len(toks) >= 2 else set()

def bonus_coocurrencia(etiqueta_norm: str, tokens_oracion_set: Set[str], token_actual_norm: str) -> float:
    etoks = set(tokens_de_etiqueta(etiqueta_norm))
    otros = tokens_oracion_set - {token_actual_norm}
    overlap = len(etoks & otros)
    return min(2, overlap) * W_COOCC

def bonus_bigrama(etiqueta_norm: str, bigramas_oracion: Set[Tuple[str,str]]) -> float:
    etoks = tokens_de_etiqueta(etiqueta_norm)
    et_bi = set(zip(etoks, etoks[1:])) if len(etoks) >= 2 else set()
    return W_BIGRAM if len(et_bi & bigramas_oracion) > 0 else 0.0

def bonus_dominio(etiqueta_norm: str, dominio: str|None) -> float:
    if not dominio or dominio not in BOOST_DOMINIO: return 0.0
    etoks = set(tokens_de_etiqueta(etiqueta_norm))
    bd = set(BOOST_DOMINIO[dominio])
    overlap = len(etoks & bd)
    return min(2, overlap) * W_BOOST_DOM

def bonus_unigrama_exact(token_norm: str, etiqueta_norm: str, toks: List[str], tokens_oracion_set: Set[str]) -> float:
    bonus = 0.0
    if len(toks) == 1 and toks[0] == etiqueta_norm and etiqueta_norm == token_norm:
        bonus += EXACT_UNIGRAM_BONUS
        if token_norm in {"agua","leche","jugo","pan","comida","sandia"} and ({"beber","tomar","querer","comer"} & tokens_oracion_set):
            bonus += ADJ_VERB_NOUN_PAIR_BONUS
        if token_norm in {"baño"} and ({"ir","querer"} & tokens_oracion_set):
            bonus += ADJ_VERB_NOUN_PAIR_BONUS
    if len(toks) == 1:
        bonus += UNIGRAM_PREFER_BONUS
    elif len(toks) >= 3:
        bonus -= LONG_COMPOUND_PENALTY
    if toks and (toks[0] == token_norm or toks[-1] == token_norm):
        bonus += HEAD_POS_BONUS
    return bonus

def combinar_score(sim_emb, sim_ctx, lex_base, pen, b_cooc, b_bi, b_dom, b_uni):
    return (0.50*sim_emb + 0.25*sim_ctx + 0.15*lex_base - 0.04*pen) + b_cooc + b_bi + b_dom + b_uni

# --------- Top-K por token + pipeline de oración ----------
def topk_candidatos_para_token(token_vis: str, token_norm: str, vec_token: np.ndarray, vec_oracion_cls: np.ndarray,
                               pool_ids: List[str], emb_etq: Dict[str,np.ndarray], etiquetas_norm: Dict[str,str],
                               etiquetas_txt: Dict[str,str], tokens_oracion_set: Set[str], bigramas_oracion: Set[tuple],
                               dominio: str|None, exacto_unigrama_map: Dict[str,Set[str]],
                               token_norm_visible: str = None, K: int=K_POR_TOKEN) -> List[str]:

    exactos = set(exacto_unigrama_map.get(token_norm, set()))
    if token_norm_visible:
        exactos |= set(exacto_unigrama_map.get(token_norm_visible, set()))
    pool_union = set(pool_ids) | exactos
    if not pool_union:
        return [""] if not OMITIR_STOPWORDS_EN_SALIDA else []

    scores = []
    for pid in pool_union:
        en = etiquetas_norm[pid]
        if not _token_pasa_blocklist(en, dominio): continue
        ve = emb_etq[pid]
        sim_emb = float(np.dot(vec_token, ve))
        sim_ctx = float(np.dot(vec_oracion_cls, ve))
        lex_base = score_lexico_base(token_norm, en)
        pen = penalty_pos(token_norm, en)
        b_cooc = bonus_coocurrencia(en, tokens_oracion_set, token_norm)
        b_bi   = bonus_bigrama(en, bigramas_oracion)
        b_dom  = bonus_dominio(en, dominio)
        toks_en = tokens_de_etiqueta(en)
        b_uni  = bonus_unigrama_exact(token_norm, en, toks_en, tokens_oracion_set)
        s = combinar_score(sim_emb, sim_ctx, lex_base, pen, b_cooc, b_bi, b_dom, b_uni)
        scores.append((s, pid))

    if not scores:
        return [""] if not OMITIR_STOPWORDS_EN_SALIDA else []
    scores.sort(reverse=True, key=lambda x: x[0])
    if scores[0][0] < UMBRAL_SCORE:
        return [""] if not OMITIR_STOPWORDS_EN_SALIDA else []

    out = []
    for _, pid in scores[:K]:
        out.append(f"{pid}-{etiquetas_txt[pid].replace(' ', '_')}.jpg")
    return out or ([""] if not OMITIR_STOPWORDS_EN_SALIDA else [])

def candidatos_para_oracion(oracion_str: str,
                            etiquetas_txt, etiquetas_norm, idx_token, idx_frase, emb_etq,
                            exacto_unigrama_map, dominio: str|None,
                            k_por_token=K_POR_TOKEN, m_pool=M_POOL):
    import re as _re
    toks_vis = [t for t in _re.split(r"\s+", oracion_str.strip()) if t]

    toks_base = lematiza_tokens(toks_vis) if nlp is not None else toks_vis
    toks_norm_lemma = [norm_text(t) for t in toks_base]
    toks_norm_vis   = [norm_text(t) for t in toks_vis]
    vec_cls = embed_sentence_cls(oracion_str)
    vecs_tok = embed_tokens_en_context(oracion_str, toks_vis)

    tokens_oracion_set = set([t for t in toks_norm_lemma if t and ((t not in STOP_TOKENS) or (t in ALLOWLIST_STOPWORDS))])
    bigramas_oracion = construir_bigrama_set(toks_norm_lemma)

    n = len(toks_vis)
    candidatos_por_token: Dict[str, List[str]] = {}
    spans = detectar_ngramas(toks_norm_lemma, idx_frase)
    ocupado = [False]*n
    preasignados: Dict[int, List[str]] = {}

    # n-gramas primero
    for (i0, i1, _frase_norm, ids) in spans:
        vtok = vecs_tok[i0]
        pool = [pid for pid in ids if _token_pasa_blocklist(etiquetas_norm[pid], dominio)]
        topk = topk_candidatos_para_token(" ".join(toks_vis[i0:i1]), toks_norm_lemma[i0], vtok, vec_cls,
                                          pool, emb_etq, etiquetas_norm, etiquetas_txt,
                                          tokens_oracion_set, bigramas_oracion, dominio,
                                          exacto_unigrama_map, token_norm_visible=toks_norm_vis[i0],
                                          K=k_por_token)
        preasignados[i0] = topk
        for k in range(i0, i1):
            ocupado[k] = True
            if k != i0:
                candidatos_por_token[toks_vis[k]] = [] if OMITIR_STOPWORDS_EN_SALIDA else [""]

    # tokens simples
    for i, (tok_vis, tok_norm_lem, tok_norm_vis_) in enumerate(zip(toks_vis, toks_norm_lemma, toks_norm_vis)):
        if i in preasignados:
            candidatos_por_token[tok_vis] = preasignados[i]; continue

        if ocupado[i]:
            if not OMITIR_STOPWORDS_EN_SALIDA and (norm_text(tok_vis) in ALLOWLIST_STOPWORDS):
                candidatos_por_token[tok_vis] = [""]
            continue

        if es_puntuacion(tok_vis) or es_stop(tok_vis):
            if (norm_text(tok_vis) in ALLOWLIST_STOPWORDS):
                pass
            else:
                if not OMITIR_STOPWORDS_EN_SALIDA:
                    candidatos_por_token[tok_vis] = [""]
                continue

        if len(tok_norm_lem) <= TOKEN_CORTO_EXACTO:
            pool = []
            for key in {tok_norm_lem, tok_norm_vis_}:
                if key in idx_token: pool += list(idx_token[key])
                for s in SINONIMOS_BASE.get(key, set()):
                    if s in idx_token: pool += list(idx_token[s])
            pool = list({
                *pool,
                *list(exacto_unigrama_map.get(tok_norm_lem, [])),
                *list(exacto_unigrama_map.get(tok_norm_vis_, [])),
            })
            pool = [pid for pid in pool if _token_pasa_blocklist(etiquetas_norm[pid], dominio)][:m_pool]
        else:
            pool = pool_lexico(tok_norm_lem, idx_token, etiquetas_norm, M=m_pool, dominio=dominio)
            pool = list(set(pool)
                        | set(exacto_unigrama_map.get(tok_norm_lem, set()))
                        | set(exacto_unigrama_map.get(tok_norm_vis_, set())))

        if not pool:
            if (norm_text(tok_vis) in ALLOWLIST_STOPWORDS) and FORZAR_INCLUSION_ALLOWLIST_VACIOS:
                candidatos_por_token[tok_vis] = [""]
            else:
                if not OMITIR_STOPWORDS_EN_SALIDA:
                    candidatos_por_token[tok_vis] = [""]
            continue

        topk = topk_candidatos_para_token(tok_vis, tok_norm_lem, vecs_tok[i], vec_cls,
                                          pool, emb_etq, etiquetas_norm, etiquetas_txt,
                                          tokens_oracion_set, bigramas_oracion, dominio,
                                          exacto_unigrama_map, token_norm_visible=tok_norm_vis_,
                                          K=k_por_token)
        if topk == [""] and OMITIR_STOPWORDS_EN_SALIDA:
            continue
        candidatos_por_token[tok_vis] = topk

    return toks_vis, candidatos_por_token

# =========================================
# 3) CARGA lenta + inferencia 1-oración
# =========================================
_STATE = {
    "loaded": False,
    "DEVICE": "cpu",
    "tokenizer": None,
    "model": None,
    "nlp": None,
    "etiquetas_txt": None,
    "etiquetas_norm": None,
    "idx_token": None,
    "idx_frase": None,
    "exacto_unigrama_map": None,
    "emb_etq": None,
}

def ensure_loaded():
    """Carga modelo BETO, spaCy e índices/embeddings de pictos SOLO una vez."""
    if _STATE["loaded"]:
        return

    cfg = current_app.config
    pictos_dir = cfg["PICTOS_DIR"]
    cache_emb  = cfg.get("BETO_CACHE_EMB", os.path.join(pictos_dir, "..", "emb_pictos_cache.npy"))
    model_name = cfg.get("BETO_MODEL_NAME", MODEL_NAME)
    force_cpu  = cfg.get("BETO_FORCE_CPU", False)

    # spaCy
    global nlp
    try:
        import spacy
        nlp = spacy.load("es_core_news_sm") if USE_SPACY else None
    except Exception:
        nlp = None

    # Modelo
    global tokenizer, model, DEVICE
    import torch
    DEVICE = "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(DEVICE).eval()

    # Indexado + embeddings
    etiquetas_txt, etiquetas_norm, idx_token, idx_frase, exacto_unigrama_map = cargar_pictos_y_indexar(pictos_dir)
    emb_etq = construir_embeddings_etiquetas(etiquetas_txt, cache_path=cache_emb)

    # Guardar en estado
    _STATE.update({
        "loaded": True, "DEVICE": DEVICE, "tokenizer": tokenizer, "model": model, "nlp": nlp,
        "etiquetas_txt": etiquetas_txt, "etiquetas_norm": etiquetas_norm,
        "idx_token": idx_token, "idx_frase": idx_frase, "exacto_unigrama_map": exacto_unigrama_map,
        "emb_etq": emb_etq
    })
    print(f"[BETO] Cargado. device={DEVICE}, pictos={len(etiquetas_txt)}, cache={os.path.abspath(cache_emb)}")

def infer_sentence(oracion: str, dominio: str | None = None) -> Dict:
    """
    Ejecuta el pipeline del notebook para UNA oración y devuelve:
      {
        "tokens": [...],
        "candidatos_por_token": { token_visible: [img1, img2, ...], ... },
        "sugerida": [top1_token_0, top1_token_1, ...]
      }
    """
    ensure_loaded()
    tokens_vis, cand_map = candidatos_para_oracion(
        oracion,
        _STATE["etiquetas_txt"], _STATE["etiquetas_norm"],
        _STATE["idx_token"], _STATE["idx_frase"], _STATE["emb_etq"],
        _STATE["exacto_unigrama_map"],
        dominio=dominio, k_por_token=K_POR_TOKEN, m_pool=M_POOL
    )
    sugerida = []
    for t in tokens_vis:
        lst = cand_map.get(t, [])
        sugerida.append(lst[0] if lst else None)
    return {
        "tokens": tokens_vis,
        "candidatos_por_token": cand_map,
        "sugerida": sugerida
    }

