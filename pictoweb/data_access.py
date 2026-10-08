import json
from functools import lru_cache
from flask import current_app

def _load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)
@lru_cache(maxsize=16)
def get_selecciones(cat):
    path=current_app.config['SELECCIONES'].get(cat)
    if not path: return None
    return _load_json(path)
@lru_cache(maxsize=16)
def get_candidatos(cat):
    path=current_app.config['CANDIDATOS'].get(cat)
    if not path: return None
    return _load_json(path)

def map_candidatos_por_oracion(cat):
    candidatos=get_candidatos(cat)
    if not candidatos: return {}
    salida={}
    lista=candidatos.get('oraciones') if isinstance(candidatos,dict) else candidatos
    if not isinstance(lista,list): return {}
    for item in lista:
        key=item.get('oracion')
        if not key: continue
        salida[key]={ 'tokens': item.get('tokens',[]), 'candidatos_por_token': item.get('candidatos',{}) }
    return salida