"""
Validação das entradas da API (UC01 FA01 — dados inválidos).
Funções puras: não dependem de Flask nem do banco.

Os nomes dos parâmetros seguem o contrato em inglês de /api/v1; as mensagens
de erro são em português.
"""
import math
import re
from datetime import date, datetime, timedelta, timezone

from ml.data import FEATURE_COLUMNS, feature_schema

LIMITE_PADRAO = 25
LIMITE_MAXIMO = 100

PERFIS = ("medico", "pesquisador")
SENHA_MINIMA = 8
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ErroValidacao(ValueError):
    """Entrada inválida. `detalhes` é serializável em JSON."""

    def __init__(self, mensagem, detalhes=None, codigo="VALIDATION_ERROR"):
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.detalhes = detalhes or {}
        self.codigo = codigo


def validar_caracteristicas(payload: dict) -> list[float]:
    """
    Devolve as 30 características como floats, na ordem de FEATURE_COLUMNS.
    Rejeita campos ausentes, não numéricos, não finitos ou fora da faixa
    aceita (faixa observada no WDBC ± margem).
    """
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")

    faltando = [n for n in FEATURE_COLUMNS if n not in payload]
    if faltando:
        raise ErroValidacao("Características faltando no formulário", {"campos_faltando": faltando})

    faixas = {c["nome"]: c for c in feature_schema()["campos"]}
    valores, nao_numericos, fora_da_faixa = [], [], []
    for nome in FEATURE_COLUMNS:
        bruto = payload[nome]
        if isinstance(bruto, bool) or bruto is None or bruto == "":
            nao_numericos.append(nome)
            valores.append(None)
            continue
        try:
            v = float(bruto)
        except (TypeError, ValueError):
            nao_numericos.append(nome)
            valores.append(None)
            continue
        if not math.isfinite(v):
            nao_numericos.append(nome)
            valores.append(None)
            continue
        f = faixas[nome]
        if not (f["min_aceito"] <= v <= f["max_aceito"]):
            fora_da_faixa.append({
                "campo": nome, "valor": v,
                "min_aceito": round(f["min_aceito"], 6), "max_aceito": round(f["max_aceito"], 6),
            })
        valores.append(v)

    if nao_numericos:
        raise ErroValidacao("Todas as características devem ser numéricas", {"campos_invalidos": nao_numericos})
    if fora_da_faixa:
        raise ErroValidacao("Valores fora dos limites esperados", {"campos_fora_da_faixa": fora_da_faixa})
    return valores


def validar_paciente(mode: str, patient_name, patient_record) -> tuple[bool, str | None, str | None]:
    """
    Retorna (identificado, patient_name, patient_record). Modo anônimo descarta
    qualquer dado de paciente enviado (minimização de dados).
    """
    if mode not in ("anonimo", "identificado"):
        raise ErroValidacao("Campo 'mode' deve ser 'anonimo' ou 'identificado'.")
    if mode == "anonimo":
        return False, None, None

    nome = str(patient_name or "").strip()
    prontuario = str(patient_record or "").strip()
    if not nome or not prontuario:
        raise ErroValidacao("Modo identificado exige 'patient_name' e 'patient_record'.")
    if len(nome) > 200 or len(prontuario) > 100:
        raise ErroValidacao("Nome ou prontuário excede o tamanho máximo.")
    return True, nome, prontuario


def _int_param(args, nome, padrao, minimo, maximo):
    bruto = args.get(nome)
    if bruto in (None, ""):
        return padrao
    try:
        v = int(bruto)
    except ValueError:
        raise ErroValidacao(f"Parâmetro '{nome}' deve ser inteiro.") from None
    if not (minimo <= v <= maximo):
        raise ErroValidacao(f"Parâmetro '{nome}' deve estar entre {minimo} e {maximo}.")
    return v


def _float_param(args, nome):
    bruto = args.get(nome)
    if bruto in (None, ""):
        return None
    try:
        v = float(bruto)
    except ValueError:
        raise ErroValidacao(f"Parâmetro '{nome}' deve ser numérico.") from None
    if not (0.0 <= v <= 1.0):
        raise ErroValidacao(f"Parâmetro '{nome}' deve estar entre 0 e 1.")
    return v


def _data_param(args, nome):
    bruto = args.get(nome)
    if bruto in (None, ""):
        return None
    try:
        return date.fromisoformat(bruto)
    except ValueError:
        raise ErroValidacao(f"Parâmetro '{nome}' deve estar no formato AAAA-MM-DD.") from None


def validar_filtros_historico(args) -> dict:
    """
    Parâmetros de GET /api/v1/predictions (UC07): paginação e filtros por
    período, classe, faixa de confiança e modo de registro.
    """
    label = args.get("label") or None
    if label not in (None, "benigno", "maligno"):
        raise ErroValidacao("Parâmetro 'label' deve ser 'benigno' ou 'maligno'.")

    mode = args.get("mode") or None
    if mode not in (None, "anonimo", "identificado"):
        raise ErroValidacao("Parâmetro 'mode' deve ser 'anonimo' ou 'identificado'.")

    inicio, fim = _data_param(args, "date_from"), _data_param(args, "date_to")
    if inicio and fim and inicio > fim:
        raise ErroValidacao("'date_from' não pode ser posterior a 'date_to'.")
    cmin, cmax = _float_param(args, "confidence_min"), _float_param(args, "confidence_max")
    if cmin is not None and cmax is not None and cmin > cmax:
        raise ErroValidacao("'confidence_min' não pode ser maior que 'confidence_max'.")

    utc = timezone.utc
    return {
        "page": _int_param(args, "page", 1, 1, 10**9),
        "limit": _int_param(args, "limit", LIMITE_PADRAO, 1, LIMITE_MAXIMO),
        "label": label,
        "mode": mode,
        "date_from": datetime.combine(inicio, datetime.min.time(), utc) if inicio else None,
        # date_to é inclusiva: usa o início do dia seguinte como limite aberto
        "date_to_exclusive": datetime.combine(fim + timedelta(days=1), datetime.min.time(), utc) if fim else None,
        "confidence_min": cmin,
        "confidence_max": cmax,
    }


def validar_cadastro(payload) -> tuple[str, str, str, str]:
    """
    Corpo de POST /api/v1/auth/register. Retorna (name, email, password, profile).
    """
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")

    name = str(payload.get("name") or "").strip()
    email = str(payload.get("email") or "").strip().lower()
    password = payload.get("password")
    profile = str(payload.get("profile") or "").strip()

    if not name or len(name) > 120:
        raise ErroValidacao("Campo 'name' é obrigatório e deve ter até 120 caracteres.")
    if not email or len(email) > 120 or not _EMAIL.match(email):
        raise ErroValidacao("Campo 'email' deve ser um e-mail válido.")
    if not isinstance(password, str) or len(password) < SENHA_MINIMA:
        raise ErroValidacao(f"Campo 'password' deve ter no mínimo {SENHA_MINIMA} caracteres.")
    if profile not in PERFIS:
        raise ErroValidacao("Campo 'profile' deve ser 'medico' ou 'pesquisador'.")

    return name, email, password, profile


def validar_login(payload) -> tuple[str, str]:
    """Corpo de POST /api/v1/auth/login. Retorna (email, password)."""
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")
    email = str(payload.get("email") or "").strip().lower()
    password = payload.get("password")
    if not email or not isinstance(password, str) or not password:
        raise ErroValidacao("Campos 'email' e 'password' são obrigatórios.")
    return email, password


MODELOS_VALIDOS = ("regressao_logistica", "svm")


def validar_training_run(payload) -> tuple[list[str], float, bool, dict | None]:
    """
    Corpo de POST /api/v1/training-runs (design.md §5.1). Retorna
    (models, test_size, grid_search, hyperparameters).
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")

    models = payload.get("models", list(MODELOS_VALIDOS))
    if not isinstance(models, list) or not models:
        raise ErroValidacao("Campo 'models' deve ser uma lista não vazia.")
    desconhecidos = [m for m in models if m not in MODELOS_VALIDOS]
    if desconhecidos:
        raise ErroValidacao("Campo 'models' contém modelo desconhecido.", {"desconhecidos": desconhecidos})

    test_size = payload.get("test_size", 0.2)
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise ErroValidacao("Campo 'test_size' deve ser numérico.")
    test_size = float(test_size)
    if not (0.0 < test_size <= 0.5):
        raise ErroValidacao("Campo 'test_size' deve estar no intervalo (0, 0.5].")

    grid_search = payload.get("grid_search", True)
    if not isinstance(grid_search, bool):
        raise ErroValidacao("Campo 'grid_search' deve ser booleano.")

    hyperparameters = payload.get("hyperparameters")
    if hyperparameters is not None and not isinstance(hyperparameters, dict):
        raise ErroValidacao("Campo 'hyperparameters' deve ser um objeto.")

    return list(models), test_size, grid_search, hyperparameters


def validar_threshold(payload) -> float:
    """Corpo de PUT /api/v1/models/{id}/threshold. Retorna threshold."""
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")
    threshold = payload.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ErroValidacao("Campo 'threshold' deve ser numérico.")
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ErroValidacao("Campo 'threshold' deve estar entre 0 e 1.")
    return threshold
