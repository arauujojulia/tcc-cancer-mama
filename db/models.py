"""
Sprint 0 — HT03: configuração dos perfis de médico e pesquisador.

Só as entidades por enquanto, sem login/autenticação — isso fica pra
depois, quando o sistema precisar de fato distinguir quem está usando
cada endpoint.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Medico(db.Model):
    __tablename__ = "medicos"

    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def to_dict(self):
        return {"id": self.id, "nome": self.nome, "email": self.email}


class Pesquisador(db.Model):
    __tablename__ = "pesquisadores"

    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def to_dict(self):
        return {"id": self.id, "nome": self.nome, "email": self.email}