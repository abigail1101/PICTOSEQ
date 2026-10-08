# pictoweb/clip_service.py
from typing import List, Dict
from flask import current_app

def assemble_clip(oracion: str, tokens: List[str], candmap: Dict) -> Dict:
    mode = (current_app.config.get("CLIP_INFERENCE_MODE") or "local").lower()
    if mode == "local":
        from . import clip_local
        return clip_local.assemble_one(oracion, tokens, candmap)
    # futuro implementar modo remoto aquí
    raise RuntimeError("Modo remoto no implementado para CLIP.")

# Alias para compatibilidad con routes.py (evita tocar routes)


def assemble_with_clip(oracion: str, tokens: List[str], candmap: Dict) -> Dict:
    """
    Ensambla una secuencia con CLIP forzando ejecución limpia
    (sin reutilizar caches de embeddings entre oraciones).
    """
    # Carga lenta de modelos si existe
    try:
        ensure_loaded()  
    except NameError:
        pass

    # Limpia caches de embeddings/labels si existe el helper
    try:
        _clear_runtime_cache()  # debe limpiar img_vec_cache, text_vec_cache, label_cache
    except NameError:
        # Fallback: intenta limpiar dicts comunes si están definidos
        for _name in ("img_vec_cache", "text_vec_cache", "label_cache"):
            try:
                globals()[_name].clear()
            except Exception:
                pass

    # Ejecuta el ensamblado con CLIP
    return assemble_clip(oracion, tokens, candmap)


__all__ = ["assemble_clip", "assemble_with_clip"]

