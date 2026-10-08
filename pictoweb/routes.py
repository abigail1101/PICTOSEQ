import os, io, zipfile, json, re, unicodedata
from flask import (
    Blueprint, render_template, request, send_file, abort,
    current_app, jsonify
)
from .data_access import get_selecciones, map_candidatos_por_oracion
from .beto_service import infer_pictos_for_text
from .clip_service import assemble_with_clip

bp = Blueprint("main", __name__)

# ---------- util ----------
def slugify(s: str) -> str:
    """Convierte un texto en un slug seguro para nombres de carpetas/archivos."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9\-\_\. ]+", "", s).strip().lower()
    return re.sub(r"[\s]+", "-", s)[:64] or "item"

# ---------- páginas básicas ----------
@bp.route("/")
def home():
    return render_template("home.html")

@bp.route("/acerca-de")
def acerca_de():
    return render_template("acerca_de.html")

@bp.route("/manual-usuario")
def manual_usuario():
    return render_template("manual_usuario.html")

# ---------- hub de categorías ----------
@bp.route("/categorias")
def categorias_hub():
    counts = {
        "salud":     len(get_selecciones("salud") or []),
        "educacion": len(get_selecciones("educacion") or []) if current_app.config["SELECCIONES"].get("educacion") else 0,
        "cultura":   len(get_selecciones("cultura") or [])   if current_app.config["SELECCIONES"].get("cultura")   else 0,
    }
    return render_template("categoria.html", counts=counts)

# ---------- detalle de categoría (carrusel + lista + descargas) ----------
@bp.route("/categoria/<cat>", methods=["GET", "POST"])
def categoria_detalle(cat):
    if cat not in {"salud", "educacion", "cultura"}:
        abort(404)

    selecciones = get_selecciones(cat)
    if not selecciones:
        return render_template("categoria_vacia.html", cat=cat)

    # GET: UI siempre con originales
    if request.method == "GET":
        cand_map = map_candidatos_por_oracion(cat)
        return render_template(
            "categoria_detalle.html",
            cat=cat,
            selecciones=selecciones,
            effective=None,            # siempre None en pantalla
            candidatos_map=cand_map
        )

    # POST: descarga (aplicando overrides temporales que vinieron del form)
    # 1) qué filas descargar (si no hay, se asume todas)
    indices = [int(i) for i in request.form.getlist("indices") if i.isdigit()]
    if not indices:
        indices = None

    # 2) overrides de IMÁGENES (objeto dict {"0":[...], "3":[...]})
    edits_json = request.form.get("edits_json") or "{}"
    try:
        client_edits_imgs = json.loads(edits_json)
    except Exception:
        client_edits_imgs = {}

    # Normaliza a lista alineada con selecciones (None donde no hay override)
    effective_for_zip = None
    if isinstance(client_edits_imgs, dict) and client_edits_imgs:
        effective_for_zip = [None] * len(selecciones)
        for k, v in client_edits_imgs.items():
            try:
                i = int(k)
                if 0 <= i < len(selecciones) and isinstance(v, list):
                    effective_for_zip[i] = [n for n in v if n]  # filtra vacíos
            except Exception:
                continue

    # 3) overrides de TOKENS/ORACIÓN (dict {"0":[...]} o {"0":"texto"})
    edits_tokens_raw = request.form.get("edits_tokens_json") or "{}"
    try:
        client_edits_tokens = json.loads(edits_tokens_raw)
    except Exception:
        client_edits_tokens = {}

    # 4) empaqueta y envía ZIP
    return _zip_categoria(
        cat,
        selecciones,
        effective_for_zip,          # imágenes override por índice
        indices,
        token_overrides=client_edits_tokens  # tokens/oración override por índice
    )

# ---------- descarga completa vía link (sin cambios temporales) ----------
@bp.route("/descargar/categoria/<cat>")
def descargar_categoria(cat):
    selecciones = get_selecciones(cat)
    if not selecciones:
        abort(404)
    return _zip_categoria(cat, selecciones, effective=None, indices=None)

# ---------- estáticos de pictogramas ----------
@bp.route("/pictos/<path:filename>")
def pictos(filename):
    from flask import send_from_directory
    # Seguridad: sólo sirve archivos por basename dentro de PICTOS_DIR
    return send_from_directory(current_app.config["PICTOS_DIR"], os.path.basename(filename))

# ---------- util de empaquetado ----------
def _zip_categoria(cat, selecciones, effective, indices, token_overrides=None):
    """
    Empaqueta la categoría en ZIP.

    - selecciones: lista de dicts con claves: 'oracion', 'tokens', 'secuencia_final'/'elecciones'
    - effective:   lista de listas de imágenes (overrides de imágenes) o None
    - indices:     subconjunto de índices a incluir o None (todas)
    - token_overrides: dict { "idx_str": list[str] | str } con tokens/oración editados
    """
    token_overrides = token_overrides or {}

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        pictos_dir = current_app.config["PICTOS_DIR"]

        for idx, item in enumerate(selecciones):
            if indices is not None and idx not in indices:
                continue

            # oracion/tokens originales
            oracion_orig = item.get("oracion", "secuencia")
            tokens_orig  = item.get("tokens", []) or oracion_orig.split()

            # aplica override de tokens/oración si existe
            idx_str = str(idx)
            ovr_tokens = token_overrides.get(idx_str)
            if isinstance(ovr_tokens, list):
                oracion_final = " ".join(ovr_tokens)
                tokens_final  = ovr_tokens
            elif isinstance(ovr_tokens, str):
                oracion_final = ovr_tokens
                tokens_final  = oracion_final.split()
            else:
                oracion_final = oracion_orig
                tokens_final  = tokens_orig

            # imágenes base (original) o overrides si existen
            base_imgs = (item.get("secuencia_final") or item.get("elecciones") or [])[:]
            if effective and idx < len(effective) and effective[idx]:
                base_imgs = effective[idx]

            # carpeta con slug de la oración final
            basefolder = f"{cat}/{idx:03d}-{slugify(oracion_final)}"

            # manifest por secuencia
            manifest = {
                "categoria": cat,
                "oracion": oracion_final,
                "tokens": tokens_final,
                "imagenes": base_imgs,
                "fuente": "selecciones_originales"
                          if (effective is None and not token_overrides)
                          else "selecciones + cambios_temporales"
            }
            z.writestr(
                f"{basefolder}/manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2)
            )

            # adjunta imágenes
            for name in (n for n in base_imgs if n):
                path = os.path.join(pictos_dir, os.path.basename(name))
                if os.path.isfile(path):
                    z.write(path, arcname=f"{basefolder}/imagenes/{os.path.basename(name)}")

    mem.seek(0)
    return send_file(
        mem,
        as_attachment=True,
        download_name=f"{cat}_secuencias.zip",
        mimetype="application/zip"
    )

# --- NUEVA SECCIÓN: Crear Picto Secuencia ---
@bp.route("/crear", methods=["GET", "POST"])
def crear_picto_secuencia():
    raw_basket = request.form.get("basket_json") or request.args.get("basket_json") or ""
    try:
        saved_list = json.loads(raw_basket) if raw_basket else []
    except Exception:
        saved_list = []

    if request.method == "POST":
        frase = (request.form.get("frase") or "").strip()
        if not frase:
            return render_template("crear_picto.html", error="Escribe una frase.", saved_list=saved_list)
        try:
            result   = infer_pictos_for_text(frase, dominio=None)  # modo local
            tokens   = result.get("tokens") or []
            sugerida = result.get("sugerida") or [None]*len(tokens)
            cand_map = result.get("candidatos_por_token") or {}
            return render_template(
                "crear_picto.html",
                frase=frase,
                tokens=tokens, sugerida=sugerida,
                candidatos_map=cand_map,
                saved_list=saved_list
            )
        except Exception as e:
            return render_template("crear_picto.html", error=f"No se pudo inferir: {e}", saved_list=saved_list)

    return render_template("crear_picto.html", saved_list=saved_list)

# --- DESCARGA DE TODAS LAS SECUENCIAS DEL CARRITO ---
@bp.route("/descargar/creadas_lote", methods=["POST"])
def descargar_creadas_lote():
    raw = request.form.get("basket_json") or "[]"
    try:
        items = json.loads(raw)
    except Exception:
        items = []
    if not items:
        abort(400)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        pictos_dir = current_app.config["PICTOS_DIR"]
        for idx, it in enumerate(items):
            oracion = it.get("oracion") or f"oracion_{idx+1}"
            tokens = it.get("tokens") or oracion.split()
            imagenes = it.get("imagenes") or []
            basefolder = f"creadas/{idx:03d}-{slugify(oracion)}"
            manifest = {
                "categoria": "creada",
                "oracion": oracion,
                "tokens": tokens,
                "imagenes": imagenes,
                "fuente": "crear_picto_secuencia"
            }
            z.writestr(f"{basefolder}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for name in (n for n in imagenes if n):
                path = os.path.join(pictos_dir, os.path.basename(name))
                if os.path.isfile(path):
                    z.write(path, arcname=f"{basefolder}/imagenes/{os.path.basename(name)}")
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="pictoseq_creadas.zip", mimetype="application/zip")

# --- API: re-inferir candidatos para una oración editada ---
@bp.post("/api/reinfer")
def api_reinfer():
    data = request.get_json(silent=True) or {}
    frase = (data.get("frase") or "").strip()
    dominio = (data.get("dominio") or "").strip() or None
    if not frase:
        return jsonify({"error": "frase vacía"}), 400
    try:
        result = infer_pictos_for_text(frase, dominio=dominio)
        # esperado: { tokens: [...], sugerida: [...], candidatos_por_token: {token: [img,..], ...} }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- API: re-inferir SOLO un token de una oración editada ---
@bp.post("/api/reinfer_token")
def api_reinfer_token():
    data    = request.get_json(silent=True) or {}
    frase   = (data.get("frase") or "").strip()
    pos     = data.get("pos")  # índice (int) del token editado según la UI
    raw_tok = (data.get("token") or "").strip()
    dominio = (data.get("dominio") or "").strip() or None
    if not frase or pos is None:
        return jsonify({"error": "faltan parámetros (frase, pos)"}), 400

    # normalizador simple sin tildes
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return s.lower().strip()

    try:
        result = infer_pictos_for_text(frase, dominio=dominio)
        tokens = result.get("tokens") or []
        cmap   = result.get("candidatos_por_token") or {}
        suger  = result.get("sugerida") or [None]*len(tokens)

        # Mapea el índice: intenta usar 'pos' si existe; si no, busca por texto normalizado
        idx = None
        if isinstance(pos, int) and 0 <= pos < len(tokens):
            idx = pos
        else:
            tnorm = _norm(raw_tok)
            for i, tk in enumerate(tokens):
                if _norm(tk) == tnorm:
                    idx = i
                    break
        if idx is None:
            return jsonify({"tokens": tokens, "index": None, "token": None,
                            "candidatos": [], "sugerido": None})

        tk   = tokens[idx]
        cnds = cmap.get(tk, []) or []
        sug  = suger[idx] if idx < len(suger) else (cnds[0] if cnds else None)

        return jsonify({"tokens": tokens, "index": idx, "token": tk,
                        "candidatos": cnds, "sugerido": sug})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@bp.post("/api/clip/assemble")
def api_clip_assemble():
    try:
        payload = request.get_json(force=True)
        oracion   = (payload or {}).get("oracion", "").strip()
        tokens    = (payload or {}).get("tokens") or []
        candidatos= (payload or {}).get("candidatos") or {}

        if not oracion or not tokens or not isinstance(candidatos, dict):
            return jsonify({"ok": False, "error": "payload inválido"}), 400

        from .clip_service import assemble_clip
        result = assemble_clip(oracion, tokens, candidatos)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

