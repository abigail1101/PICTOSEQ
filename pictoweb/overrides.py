import os, json
from flask import current_app

def _path_for(cat: str):
    # Un archivo por categoría, p.ej.: overrides/salud.overrides.json
    base = current_app.config.setdefault("OVERRIDES_DIR",
              os.path.join(os.path.dirname(__file__), "overrides"))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{cat}.overrides.json")

def load_overrides(cat: str) -> dict:
    p = _path_for(cat)
    if not os.path.isfile(p): return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)  # esperado: {"0":[...], "1":[...]}
    
def save_overrides(cat: str, edits: dict):
    # Merge: lo nuevo pisa lo anterior; nunca se borra el original
    p = _path_for(cat)
    data = load_overrides(cat)
    for k, v in edits.items():
        # normaliza key a str (guardamos por índice de oración)
        data[str(k)] = v
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def apply_overrides(selecciones: list, overrides: dict) -> list:
    """Devuelve lista de listas de imágenes (final + overrides)."""
    out = []
    for idx, item in enumerate(selecciones):
        base = (item.get("secuencia_final") or item.get("elecciones") or [])[:]
        if str(idx) in overrides:
            ov = overrides[str(idx)]
            # Asegura misma longitud que tokens (rellena con None)
            tokens = item.get("tokens", [])
            base = (ov + [None]*len(tokens))[:len(tokens)]
        out.append(base)
    return out
