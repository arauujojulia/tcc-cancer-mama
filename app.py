"""
Aplicação Flask.

Expõe a API REST em /api/v1 (api/routes.py), persiste usuários, avaliações e
modelos treinados (db/models.py) e serve o frontend React compilado
(frontend/dist).

Configuração por variáveis de ambiente (ou arquivo .env, ver .env.example):
  DATABASE_URL           ex.: postgresql+psycopg://usuario:senha@localhost:5432/tcc_cancer_mama
                         (se ausente: SQLite local, só para desenvolvimento)
  CORS_ORIGINS           origens do frontend separadas por vírgula; precisa ser
                         a lista explícita, pois o cookie de sessão viaja com
                         credenciais (padrão: http://localhost:5173)
  SECRET_KEY             assinatura do cookie de sessão (sem ela, uma chave
                         temporária é gerada em memória e um aviso é logado)
  SESSION_COOKIE_SECURE  true para restringir o cookie a HTTPS (padrão: false)
"""
import os
import secrets

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from api.routes import api_bp
from db.models import db
from services.modelos import sincronizar_modelos

load_dotenv()

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
CORS_ORIGINS_PADRAO = "http://localhost:5173"


def _origens_cors():
    """
    Lista explícita de origens. Com `supports_credentials`, o navegador recusa
    a origem curinga, então '*' não é aceito aqui.
    """
    bruto = os.environ.get("CORS_ORIGINS") or CORS_ORIGINS_PADRAO
    origens = [o.strip() for o in bruto.split(",") if o.strip() and o.strip() != "*"]
    return origens or [CORS_ORIGINS_PADRAO]


def _booleano(nome, padrao=False):
    bruto = os.environ.get(nome)
    if bruto is None or bruto == "":
        return padrao
    return bruto.strip().lower() in ("1", "true", "yes", "on")


def _erro_app(mensagem, code, status):
    """Mesmo envelope de erro da API, para não haver duas convenções no serviço."""
    return jsonify({"error": mensagem, "code": code}), status


def create_app(config=None):
    app = Flask(__name__, static_folder=None)

    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///tcc_cancer_mama.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    chave = os.environ.get("SECRET_KEY")
    if not chave:
        chave = secrets.token_hex(32)
        app.logger.warning(
            "SECRET_KEY não configurada: usando chave temporária em memória. "
            "As sessões caem a cada reinício; defina SECRET_KEY no .env."
        )
    app.config["SECRET_KEY"] = chave
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = _booleano("SESSION_COOKIE_SECURE", False)

    if config:
        app.config.update(config)

    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": _origens_cors()}}, supports_credentials=True)
    app.register_blueprint(api_bp)

    def _tratar_erro_interno(e):
        """
        Falha inesperada → 500 `INTERNAL_ERROR` no mesmo envelope das demais
        (contrato do README e da especificação consolidada). O stack trace vai
        para o log do servidor (`logger.exception`) e **não** para a resposta:
        o corpo devolvido traz só a mensagem genérica e o código.
        """
        app.logger.exception("INTERNAL_ERROR: falha não tratada em %s", request.path)
        return _erro_app(
            "Erro interno do servidor. Tente novamente ou contate o suporte técnico.",
            "INTERNAL_ERROR", 500,
        )

    @app.errorhandler(Exception)
    def _tratar_excecao(e):
        # Os erros HTTP (404, 405, …) continuam com o tratamento próprio: este
        # handler cobre apenas exceção não tratada, que antes virava HTML.
        if isinstance(e, HTTPException):
            return e
        return _tratar_erro_interno(e)

    # `abort(500)` levanta InternalServerError, que é HTTPException e passaria
    # pelo guarda acima; por isso o código 500 tem registro próprio.
    app.register_error_handler(500, _tratar_erro_interno)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    # Frontend compilado (npm run build em frontend/), quando existir.
    @app.route("/", defaults={"caminho": ""})
    @app.route("/<path:caminho>")
    def frontend(caminho):
        if caminho == "api" or caminho.startswith("api/"):
            return _erro_app("Rota não encontrada.", "NOT_FOUND", 404)
        if not os.path.isdir(FRONTEND_DIST):
            return _erro_app(
                "Frontend não compilado. Rode 'npm run build' em frontend/.", "NOT_FOUND", 404
            )
        if caminho and os.path.isfile(os.path.join(FRONTEND_DIST, caminho)):
            return send_from_directory(FRONTEND_DIST, caminho)
        return send_from_directory(FRONTEND_DIST, "index.html")

    with app.app_context():
        db.create_all()  # users, evaluations, trained_models, training_runs
        if not app.config.get("TESTING"):
            sincronizar_modelos()

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
