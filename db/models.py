"""
Modelo de dados.

Quatro tabelas: `users` (médico ou pesquisador, perfil declarado no cadastro,
sem trava de autorização), `evaluations` (histórico imutável de classificações),
`trained_models` (classificadores treinados, com métricas e ponto de operação) e
`training_runs` (cada disparo de treino, para reprodutibilidade).

Em `trained_models`, as métricas escalares e a matriz de confusão são sempre do
CONJUNTO DE TESTE no limiar vigente da linha; o portão de promoção do RNF04 fica
em `train_oof_sensitivity`, calculado nas predições out-of-fold do TREINO (D11).
A unicidade é `(training_run_id, name)` — uma linha por execução (D12).

Os nomes de tabela e de coluna são em inglês (contrato da API e do banco); os
comentários e as mensagens, em português.

Tipos escolhidos para funcionar igual em PostgreSQL e SQLite: `db.JSON`,
`db.DateTime(timezone=True)` e defaults no nível Python (nunca `server_default`
com SQL de dialeto).
"""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event

db = SQLAlchemy()


def _agora():
    return datetime.now(timezone.utc)


def _iso(valor):
    return valor.isoformat() if valor else None


class User(db.Model):
    """
    Usuário da ferramenta. `profile` é apenas declarado no cadastro e serve
    para o frontend escolher o menu — não há controle de acesso por perfil.
    """

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)  # pbkdf2 (werkzeug)
    profile = db.Column(db.String(20), nullable=False)         # medico | pesquisador
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_agora)

    def to_dict(self):
        """Nunca expõe `password_hash`."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "profile": self.profile,
            "created_at": _iso(self.created_at),
        }


class Evaluation(db.Model):
    """
    Registro de uma classificação (UC07). Imutável após gravado: a
    rastreabilidade dos casos exige que o histórico não seja reescrito.

    No modo identificado, `patient_name` e `patient_record` são gravados em
    TEXTO PURO — a ferramenta não deve ser exposta fora de ambiente controlado.
    """

    __tablename__ = "evaluations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    features = db.Column(db.JSON, nullable=False)              # as 30 características informadas
    label = db.Column(db.String(10), nullable=False)           # benigno | maligno
    probability = db.Column(db.Float, nullable=False)          # P(maligno)
    # Confiança na classe prevista: P se maligno, 1-P se benigno. Coluna
    # numérica que sustenta os filtros confidence_min / confidence_max.
    confidence = db.Column(db.Float, nullable=False)
    confidence_band = db.Column(db.String(10), nullable=False)  # alta | media | baixa
    model_name = db.Column(db.String(60), nullable=False)
    threshold_used = db.Column(db.Float, nullable=False)
    mode = db.Column(db.String(15), nullable=False)            # anonimo | identificado
    patient_name = db.Column(db.Text, nullable=True)
    patient_record = db.Column(db.Text, nullable=True)
    elapsed_ms = db.Column(db.Integer, nullable=False)         # RNF02
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_agora)

    user = db.relationship("User")

    __table_args__ = (
        db.Index("ix_evaluations_created_at", "created_at"),                 # ordenação padrão
        db.Index("ix_evaluations_label_created", "label", "created_at"),     # classe + período
        db.Index("ix_evaluations_mode_created", "mode", "created_at"),       # modo + período
        db.Index("ix_evaluations_label_confidence", "label", "confidence"),  # classe + confiança
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "label": self.label,
            "probability": self.probability,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band,
            "model_name": self.model_name,
            "threshold_used": self.threshold_used,
            "mode": self.mode,
            "patient_name": self.patient_name,
            "patient_record": self.patient_record,
            "elapsed_ms": self.elapsed_ms,
            "features": self.features,
            "created_at": _iso(self.created_at),
        }


@event.listens_for(Evaluation, "before_update")
def _bloquear_update(mapper, connection, target):
    raise RuntimeError("Avaliações são imutáveis: não podem ser alteradas após o registro.")


@event.listens_for(Evaluation, "before_delete")
def _bloquear_delete(mapper, connection, target):
    raise RuntimeError("Avaliações são imutáveis: não podem ser removidas.")


class TrainingRun(db.Model):
    """
    Um disparo de treino. O treino inicial versionado em ml/artifacts/ é
    representado por um registro sintético com status 'success'.
    """

    __tablename__ = "training_runs"

    id = db.Column(db.Integer, primary_key=True)
    models_requested = db.Column(db.JSON, nullable=False)   # ex.: ["regressao_logistica", "svm"]
    test_size = db.Column(db.Float, nullable=False)
    random_seed = db.Column(db.Integer, nullable=False)
    grid_search = db.Column(db.Boolean, nullable=False, default=False)
    dataset_sha256 = db.Column(db.String(64))
    split = db.Column(db.JSON)              # índices das partições (split.json)
    library_versions = db.Column(db.JSON)   # python, scikit_learn, numpy, joblib
    git_commit = db.Column(db.String(40))
    status = db.Column(db.String(10), nullable=False)  # running | success | failed
    error_message = db.Column(db.Text, nullable=True)
    started_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_agora)
    finished_at = db.Column(db.DateTime(timezone=True), nullable=True)

    models = db.relationship("TrainedModel", back_populates="training_run")

    def to_dict(self):
        return {
            "id": self.id,
            "models_requested": self.models_requested,
            "test_size": self.test_size,
            "random_seed": self.random_seed,
            "grid_search": self.grid_search,
            "dataset_sha256": self.dataset_sha256,
            "split": self.split,
            "library_versions": self.library_versions,
            "git_commit": self.git_commit,
            "status": self.status,
            "error_message": self.error_message,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
        }


class TrainedModel(db.Model):
    """
    Um classificador treinado, com tudo que é preciso para reproduzi-lo.

    `threshold` é o ponto de operação vigente (ajustável pelo pesquisador,
    sempre validado pela meta de sensibilidade de 0,97); `safety_threshold` é o
    limiar calculado no treino.

    D11 — duas grandezas que não devem ser confundidas:

    - `accuracy`, `sensitivity`, `specificity`, `false_negative_rate` e
      `confusion_matrix` são **sempre** do CONJUNTO DE TESTE da execução deste
      modelo, recalculadas em conjunto no `threshold` vigente. Vale a invariante
      `sensitivity == vp/(vp+fn)` e `false_negative_rate == fn/(vp+fn)` sobre a
      própria `confusion_matrix`. `auc` é do teste e independe do limiar.
    - `train_oof_sensitivity` é o portão de promoção do RNF04: sensibilidade das
      predições out-of-fold do TREINO avaliada no `threshold` desta linha. Nunca
      é métrica reportada e nunca entra em `sensitivity`.

    D12 — uma linha por (execução, modelo): a unicidade é
    `(training_run_id, name)` e `artifact_path` aponta sempre para o artefato da
    própria execução. `artifact_sha256` fica como procedência (RNF03), sem
    unicidade: dois treinos deterministas com a mesma semente e os mesmos dados
    produzem legitimamente os mesmos bytes em execuções distintas.
    """

    __tablename__ = "trained_models"

    id = db.Column(db.Integer, primary_key=True)
    training_run_id = db.Column(db.Integer, db.ForeignKey("training_runs.id"), nullable=False)
    name = db.Column(db.String(60), nullable=False, index=True)  # regressao_logistica | svm

    # Métricas do CONJUNTO DE TESTE no `threshold` vigente (D11).
    accuracy = db.Column(db.Float, nullable=False)
    sensitivity = db.Column(db.Float, nullable=False)
    specificity = db.Column(db.Float, nullable=False)
    auc = db.Column(db.Float, nullable=False)
    false_negative_rate = db.Column(db.Float, nullable=False)  # RNF04

    threshold = db.Column(db.Float, nullable=False)
    safety_threshold = db.Column(db.Float, nullable=False)
    # Portão do RNF04: sensibilidade das predições out-of-fold do TREINO
    # avaliada no `threshold` desta linha. Nullable porque o `{modelo}_oof.json`
    # da execução pode não estar legível quando a linha é criada; a promoção,
    # que é onde o portão decide, nunca aceita ausência (devolve 503 antes de
    # gravar). NÃO é métrica reportada.
    train_oof_sensitivity = db.Column(db.Float, nullable=True)

    hyperparameters = db.Column(db.JSON, nullable=False)
    random_seed = db.Column(db.Integer, nullable=False)
    artifact_path = db.Column(db.String(255), nullable=False)
    # Procedência dos bytes (RNF03), sem unicidade — ver D12 na docstring.
    artifact_sha256 = db.Column(db.String(64), nullable=False, index=True)
    is_production = db.Column(db.Boolean, nullable=False, default=False)
    roc_points = db.Column(db.JSON)        # {"fpr": [...], "tpr": [...]}
    confusion_matrix = db.Column(db.JSON)  # {"vn", "fp", "fn", "vp"}
    trained_at = db.Column(db.DateTime(timezone=True))

    training_run = db.relationship("TrainingRun", back_populates="models")

    __table_args__ = (
        # D12: uma linha por (execução, modelo). Restrição de tabela declarada
        # aqui é criada igual por create_all() em PostgreSQL e em SQLite.
        db.UniqueConstraint("training_run_id", "name", name="uq_trained_models_run_name"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "training_run_id": self.training_run_id,
            "name": self.name,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "auc": self.auc,
            "false_negative_rate": self.false_negative_rate,
            "train_oof_sensitivity": self.train_oof_sensitivity,
            "threshold": self.threshold,
            "safety_threshold": self.safety_threshold,
            "hyperparameters": self.hyperparameters,
            "random_seed": self.random_seed,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "is_production": self.is_production,
            "roc_points": self.roc_points,
            "confusion_matrix": self.confusion_matrix,
            "trained_at": _iso(self.trained_at),
        }
