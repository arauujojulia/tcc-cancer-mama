"""
Registro e consulta do histórico de avaliações (US07 / UC07).

Os dados de paciente do modo identificado são cifrados com AES-256-GCM
(services/crypto.py). O médico vê os dados em claro; o pesquisador recebe `null`
(minimização de dados — LGPD).
"""
from cryptography.exceptions import InvalidTag
from flask import current_app
from sqlalchemy import update

from db.models import Evaluation, db
from services import crypto


def _decifrar(av: Evaluation, valor: str | None, campo: bytes) -> str | None:
    """Valor ilegível vira `null` e vai para o log: uma linha não derruba o histórico."""
    if valor is None:
        return None
    try:
        return crypto.decrypt(valor, crypto.ad_linha(campo, av.id))
    except (InvalidTag, ValueError, RuntimeError) as e:
        current_app.logger.error(
            "Avaliação %s: %s ilegível (%s).", av.id, campo.decode(), type(e).__name__
        )
        return None


def _av_dict(av: Evaluation, profile: str | None) -> dict:
    """
    Converte uma avaliação em dict. Só o médico recebe os dados identificáveis;
    para o pesquisador nem são decifrados (minimização de dados).
    """
    d = av.to_dict()
    if profile == "medico":
        d["patient_name"] = _decifrar(av, av.patient_name, crypto._AD_EVAL_NAME)
        d["patient_record"] = _decifrar(av, av.patient_record, crypto._AD_EVAL_RECORD)
    return d


def registrar(*, features: dict, resultado: dict, tempo_ms: float,
              mode: str, patient_name: str | None, patient_record: str | None,
              user_id: int | None) -> Evaluation:
    """Grava a avaliação em uma única transação, já com o resultado da inferência."""
    av = Evaluation(
        user_id=user_id,
        features=features,
        label=resultado["classificacao"],
        probability=resultado["probabilidade_maligno"],
        confidence=resultado["confianca"],
        confidence_band=resultado["faixa_confianca"],
        model_name=resultado["modelo_utilizado"],
        threshold_used=resultado["limiar_utilizado"],
        mode=mode,
        elapsed_ms=round(tempo_ms),
    )
    db.session.add(av)
    if mode == "identificado":
        db.session.flush()  # o id da linha entra no dado associado da cifra
        db.session.execute(update(Evaluation).where(Evaluation.id == av.id).values(
            patient_name=crypto.encrypt(patient_name, crypto.ad_linha(crypto._AD_EVAL_NAME, av.id)),
            patient_record=crypto.encrypt(patient_record, crypto.ad_linha(crypto._AD_EVAL_RECORD, av.id)),
        ))
    db.session.commit()
    return av


def listar(filtros: dict, profile: str | None = None) -> dict:
    """Envelope paginado {items, page, limit, total}, mais recentes primeiro."""
    q = Evaluation.query
    if filtros["label"]:
        q = q.filter(Evaluation.label == filtros["label"])
    if filtros["mode"]:
        q = q.filter(Evaluation.mode == filtros["mode"])
    if filtros["date_from"]:
        q = q.filter(Evaluation.created_at >= filtros["date_from"])
    if filtros["date_to_exclusive"]:
        q = q.filter(Evaluation.created_at < filtros["date_to_exclusive"])
    if filtros["confidence_min"] is not None:
        q = q.filter(Evaluation.confidence >= filtros["confidence_min"])
    if filtros["confidence_max"] is not None:
        q = q.filter(Evaluation.confidence <= filtros["confidence_max"])

    q = q.order_by(Evaluation.created_at.desc(), Evaluation.id.desc())
    total = q.count()
    limit, page = filtros["limit"], filtros["page"]
    itens = q.offset((page - 1) * limit).limit(limit).all()

    return {
        "items": [_av_dict(a, profile) for a in itens],
        "page": page,
        "limit": limit,
        "total": total,
    }


def obter(evaluation_id: int, profile: str | None = None):
    av = db.session.get(Evaluation, evaluation_id)
    if av is None:
        return None
    return _av_dict(av, profile)