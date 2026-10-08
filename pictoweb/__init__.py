from flask import Flask
from .config import Config
from .routes import bp as main_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(main_bp)
    app.config.setdefault("PICTOS_DIR", r"C:\Users\Abigail\Desktop\NOVENO SEMESTRE\PICTOGRAMAS\PICTOGRAMAS\pictogramas")
    # CLIP
    app.config.setdefault("CLIP_TEXT_MODEL",  "sentence-transformers/clip-ViT-B-32-multilingual-v1")
    app.config.setdefault("CLIP_IMAGE_MODEL", "clip-ViT-B-32")
    app.config.setdefault("CLIP_FORCE_CPU",   False)
    # Si algún día se expone servicio remoto:
    app.config.setdefault("CLIP_REMOTE_URL",  None)
    return app
