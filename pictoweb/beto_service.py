# pictoweb/beto_service.py
from flask import current_app

def infer_remote(text: str) -> dict:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("El modo remoto requiere instalar 'requests' (pip install requests).") from e

    url = current_app.config.get("BETO_REMOTE_URL")
    if not url:
        raise RuntimeError("BETO_REMOTE_URL no configurado.")
    r = requests.post(url, json={"text": text}, timeout=120)
    r.raise_for_status()
    return r.json()

def infer_local(text: str, dominio: str | None = None) -> dict:
    from . import beto_local
    return beto_local.infer_sentence(text, dominio=dominio)

def infer_pictos_for_text(text: str, dominio: str | None = None) -> dict:
    mode = (current_app.config.get("BETO_INFERENCE_MODE") or "local").lower()
    if mode == "local":
        return infer_local(text, dominio=dominio)
    return infer_remote(text)
