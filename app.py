"""
Aplicação Flask.

Sprint 0 (HT03): entidades Medico e Pesquisador (db/models.py).
Sprint 1: endpoint de classificação (api/routes.py).
Sprint 2: persistência das avaliações e registro dos modelos treinados.
Sprint 3: API REST completa + servir o frontend React (frontend/dist).

Configuração por variáveis de ambiente (ou arquivo .env, ver .env.example):
  DATABASE_URL             ex.: postgresql://usuario:senha@localhost:5432/tcc_cancer_mama
                           (padrão: SQLite local, só para desenvolvimento)
  PACIENTE_ENCRYPTION_KEY  chave AES-256 em base64 (python -m services.crypto)
"""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from api.routes import api_bp
from db.models import db
from services.modelos import sincronizar_modelos

load_dotenv()

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")


def create_app(config=None):
    app = Flask(__name__, static_folder=None)

    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///tcc_cancer_mama.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    if config:
        app.config.update(config)

    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": os.environ.get("CORS_ORIGINS", "*").split(",")}})
    app.register_blueprint(api_bp)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    # Frontend compilado (npm run build em frontend/), quando existir.
    @app.route("/", defaults={"caminho": ""})
    @app.route("/<path:caminho>")
    def frontend(caminho):
        if caminho == "api" or caminho.startswith("api/"):
            return jsonify({"erro": "Rota não encontrada.", "codigo": "ROTA_NAO_ENCONTRADA"}), 404
        if not os.path.isdir(FRONTEND_DIST):
            return jsonify({"erro": "Frontend não compilado. Rode 'npm run build' em frontend/.",
                            "codigo": "FRONTEND_AUSENTE"}), 404
        if caminho and os.path.isfile(os.path.join(FRONTEND_DIST, caminho)):
            return send_from_directory(FRONTEND_DIST, caminho)
        return send_from_directory(FRONTEND_DIST, "index.html")

    with app.app_context():
        db.create_all()  # medicos, pesquisadores, modelos_treinados, avaliacoes
        if not app.config.get("TESTING"):
            sincronizar_modelos()

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
