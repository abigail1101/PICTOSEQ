import os
class Config:

    BETO_INFERENCE_MODE = "local"   # forzamos local
    BETO_MODEL_NAME     = "dccuchile/bert-base-spanish-wwm-cased"
    BETO_FORCE_CPU      = False     # True si queremos obligar CPU
    # cache de embeddings - se crea la primera vez)
    BETO_CACHE_EMB      = r"C:\Users\Abigail\Desktop\NOVENO SEMESTRE\PICTOGRAMAS\PICTOGRAMAS\cache\emb_pictos_cache.npy"

    PERSIST_OVERRIDES = False  # No aplicar cambios guardados al recargar

    SECRET_KEY = os.environ.get('SECRET_KEY','dev-key-change-me')
    BOOTSTRAP_DIR  = r"C:\Users\Abigail\Desktop\NOVENO SEMESTRE\PICTOGRAMAS\PICTOGRAMAS\bootstrap-5.3.8-dist"
    PICTOS_DIR     = r"C:\Users\Abigail\Desktop\NOVENO SEMESTRE\PICTOGRAMAS\PICTOGRAMAS\pictogramas"
    SECUENCIAS_DIR = r"C:\Users\Abigail\Desktop\NOVENO SEMESTRE\PICTOGRAMAS\PICTOGRAMAS\Secuencias"
    CANDIDATOS_DIR = r"C:\Users\Abigail\Desktop\NOVENO SEMESTRE\PICTOGRAMAS\PICTOGRAMAS\candidatos"
    SELECCIONES={
        'salud':os.path.join(SECUENCIAS_DIR,'selecciones_salud_final.json'),
        'educacion':os.path.join(SECUENCIAS_DIR,'selecciones_educacion_final.json'),
        'cultura': os.path.join(SECUENCIAS_DIR,'selecciones_cultura_final.json')
    }
    CANDIDATOS={
        'salud':os.path.join(CANDIDATOS_DIR,'candidatos_salud_final.json'),
        'educacion':os.path.join(CANDIDATOS_DIR,'candidatos_educacion_final.json'),
        'cultura':os.path.join(CANDIDATOS_DIR,'candidatos_cultura_final.json'),
    }
