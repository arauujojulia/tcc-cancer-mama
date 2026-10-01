"""
Aplicação Flask.

Sprint 0 (HT03): entidades Medico e Pesquisador (db/models.py).
Sprint 1: endpoint de classificação registrado (api/routes.py).
Próximo passo (Sprint 2): persistência das avaliações (US07) e endpoint
de métricas.
"""

from flask import Flask
from flask_cors import CORS

from api.routes import api_bp
from db.models import db


def create_app():
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:791016@localhost:5432/tcc_cancer_mama"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    CORS(app)
    app.register_blueprint(api_bp)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        db.create_all()  # cria as tabelas (medicos, pesquisadores) se não existirem
    app.run(debug=True, port=5000)