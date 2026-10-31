"""
Registro e consulta do histórico de avaliações (US07 / UC07).

Os dados de paciente do modo identificado são cifrados com AES-256-GCM
(TypeDecorator `EncryptedText` em db/models.py). O médico vê os dados em claro;
o pesquisador recebe `null` (minimização de dados — LGPD).
"""
from db.models import Evaluation, db


def _av_dict(av: Evaluation, profile: str | None) -> dict:
    """
    Converte uma avaliação em dict, ocultando dados identificáveis para
    pesquisadores (minimização de dados). O médico vê em claro.
    """
    d = av.to_dict()
    if profile != "medico":
        d["patient_name"] = None
        d["patient_record"] = None
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
        patient_name=patient_name,
        patient_record=patient_record,
        elapsed_ms=round(tempo_ms),
    )
    db.session.add(av)
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