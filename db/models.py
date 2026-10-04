"""
Modelo de dados.

Sprint 0 (HT03): Medico e Pesquisador.
Sprint 2 (US06, US07): ModeloTreinado (registro dos treinamentos para
reprodutibilidade) e Avaliacao (histórico de classificações).

Sem login/autenticação por enquanto — Avaliacao.medico_id é opcional.
"""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event

db = SQLAlchemy()


def _agora():
    return datetime.now(timezone.utc)


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


class ModeloTreinado(db.Model):
    """Um classificador treinado, com tudo que é preciso para reproduzi-lo."""

    __tablename__ = "modelos_treinados"

    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(60), nullable=False, index=True)       # regressao_logistica | svm
    hash_sha256 = db.Column(db.String(64), nullable=False, unique=True)  # hash do artefato .joblib
    hiperparametros = db.Column(db.JSON, nullable=False)
    semente = db.Column(db.Integer, nullable=False)
    test_size = db.Column(db.Float, nullable=False)
    n_treino = db.Column(db.Integer, nullable=False)
    n_teste = db.Column(db.Integer, nullable=False)
    limiar_seguranca = db.Column(db.Float, nullable=False)
    metricas = db.Column(db.JSON, nullable=False)    # teste_padrao / teste_operacao / validacao_cruzada
    versoes = db.Column(db.JSON)
    commit_git = db.Column(db.String(40))
    em_producao = db.Column(db.Boolean, nullable=False, default=False)
    treinado_em = db.Column(db.DateTime(timezone=True))
    registrado_em = db.Column(db.DateTime(timezone=True), nullable=False, default=_agora)

    def to_dict(self):
        return {
            "id": self.id,
            "nome": self.nome,
            "hash_sha256": self.hash_sha256,
            "hiperparametros": self.hiperparametros,
            "semente": self.semente,
            "test_size": self.test_size,
            "n_treino": self.n_treino,
            "n_teste": self.n_teste,
            "limiar_seguranca": self.limiar_seguranca,
            "metricas": self.metricas,
            "versoes": self.versoes,
            "commit_git": self.commit_git,
            "em_producao": self.em_producao,
            "treinado_em": self.treinado_em.isoformat() if self.treinado_em else None,
        }


class Avaliacao(db.Model):
    """
    Registro de uma classificação (UC07). Imutável após gravado: a
    rastreabilidade dos casos exige que o histórico não seja reescrito.
    """

    __tablename__ = "avaliacoes"

    id = db.Column(db.Integer, primary_key=True)
    criado_em = db.Column(db.DateTime(timezone=True), nullable=False, default=_agora)
    medico_id = db.Column(db.Integer, db.ForeignKey("medicos.id"), nullable=True)
    modelo_id = db.Column(db.Integer, db.ForeignKey("modelos_treinados.id"), nullable=True)

    caracteristicas = db.Column(db.JSON, nullable=False)  # as 30 características informadas
    classificacao = db.Column(db.String(10), nullable=False)  # benigno | maligno
    probabilidade_maligno = db.Column(db.Float, nullable=False)
    confianca = db.Column(db.Float, nullable=False)
    faixa_confianca = db.Column(db.String(10), nullable=False)
    limiar_utilizado = db.Column(db.Float, nullable=False)
    tempo_resposta_ms = db.Column(db.Float, nullable=False)

    # Dados do paciente: somente no modo identificado, sempre cifrados (AES-256-GCM).
    identificado = db.Column(db.Boolean, nullable=False, default=False)
    paciente_nome_cifrado = db.Column(db.Text)
    paciente_prontuario_cifrado = db.Column(db.Text)

    medico = db.relationship("Medico")
    modelo = db.relationship("ModeloTreinado")

    __table_args__ = (
        # filtros combinados do histórico: período + classe, classe + confiança
        db.Index("ix_avaliacoes_criado_classe", "criado_em", "classificacao"),
        db.Index("ix_avaliacoes_classe_confianca", "classificacao", "confianca"),
    )

    def to_dict(self, paciente=None):
        d = {
            "id": self.id,
            "criado_em": self.criado_em.isoformat() if self.criado_em else None,
            "medico_id": self.medico_id,
            "modelo_id": self.modelo_id,
            "modelo_utilizado": self.modelo.nome if self.modelo else None,
            "classificacao": self.classificacao,
            "probabilidade_maligno": self.probabilidade_maligno,
            "confianca": self.confianca,
            "faixa_confianca": self.faixa_confianca,
            "limiar_utilizado": self.limiar_utilizado,
            "tempo_resposta_ms": self.tempo_resposta_ms,
            "identificado": self.identificado,
            "caracteristicas": self.caracteristicas,
        }
        if paciente is not None:
            d["paciente"] = paciente
        return d


@event.listens_for(Avaliacao, "before_update")
def _bloquear_update(mapper, connection, target):
    raise RuntimeError("Avaliações são imutáveis: não podem ser alteradas após o registro.")


@event.listens_for(Avaliacao, "before_delete")
def _bloquear_delete(mapper, connection, target):
    raise RuntimeError("Avaliações são imutáveis: não podem ser removidas.")
