"""
Aplicação Flask.

Sprint 0 (HT03): perfis de médico e pesquisador, tabela Usuario (db/models.py).
Sprint 1: endpoint de classificação (api/routes.py).
Sprint 2: persistência das avaliações e registro dos modelos treinados.
Sprint 3: API REST completa + servir o frontend React (frontend/dist).

Configuração por variáveis de ambiente (ou arquivo .env, ver .env.example):
  SECRET_KEY               chave que assina o cookie de sessão (obrigatória em produção;
                           sem ela, uma chave temporária é gerada e as sessões caem a cada reinício)
  SESSION_COOKIE_SECURE    "true" para enviar o cookie só por HTTPS (use em produção)
  CORS_ORIGINS             origens permitidas, separadas por vírgula (padrão: http://localhost:5173)
  DATABASE_URL             SQL Server (mssql+pyodbc), ver .env.example
                           (padrão: SQL Server Express local, autenticação do Windows)
"""
import os
import secrets
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from api.routes import api_bp
from db.models import db
from services.modelos import sincronizar_modelos

load_dotenv()

DATABASE_URL_PADRAO = (
    r"mssql+pyodbc://@localhost\SQLEXPRESS/tcc_cancer_mama"
    "?driver=ODBC+Driver+18+for+SQL+Server&Trusted_Connection=yes&TrustServerCertificate=yes"
)

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")


def create_app(config=None):
    app = Flask(__name__, static_folder=None)

    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", DATABASE_URL_PADRAO)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    chave = os.environ.get("SECRET_KEY")
    if not chave:
        chave = secrets.token_hex(32)
        app.logger.warning("SECRET_KEY não definida: usando chave temporária (as sessões caem ao reiniciar).")
    app.config["SECRET_KEY"] = chave
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)
    if config:
        app.config.update(config)

    db.init_app(app)
    # Com cookie, a origem precisa ser explícita (nunca "*").
    origens = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origens}}, supports_credentials=True)
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
        db.create_all()  # usuarios, modelos_treinados, avaliacoes
        if not app.config.get("TESTING"):
            sincronizar_modelos()

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
