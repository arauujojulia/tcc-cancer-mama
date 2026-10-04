"""
Validação das entradas da API (UC01 FA01 — dados inválidos).
Funções puras: não dependem de Flask nem do banco.
"""
import math
from datetime import date, datetime, timedelta, timezone

from ml.data import FEATURE_COLUMNS, feature_schema

LIMITE_PADRAO = 25
LIMITE_MAXIMO = 100


class ErroValidacao(ValueError):
    """Entrada inválida. `detalhes` é serializável em JSON."""

    def __init__(self, mensagem, detalhes=None, codigo="DADOS_INVALIDOS"):
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


def validar_paciente(modo: str, paciente) -> tuple[bool, str | None, str | None]:
    """
    Retorna (identificado, nome, prontuario). Modo anônimo descarta qualquer
    dado de paciente enviado (minimização de dados — LGPD).
    """
    if modo not in ("anonimo", "identificado"):
        raise ErroValidacao("Campo 'modo' deve ser 'anonimo' ou 'identificado'.")
    if modo == "anonimo":
        return False, None, None

    if not isinstance(paciente, dict):
        raise ErroValidacao("Modo identificado exige o objeto 'paciente' com 'nome' e 'prontuario'.")
    nome = str(paciente.get("nome") or "").strip()
    prontuario = str(paciente.get("prontuario") or "").strip()
    if not nome or not prontuario:
        raise ErroValidacao("Modo identificado exige 'paciente.nome' e 'paciente.prontuario'.")
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
    Parâmetros de GET /api/avaliacoes (UC07): paginação e filtros por
    período, classe, faixa de confiança e modo de registro.
    """
    classificacao = args.get("classificacao") or None
    if classificacao not in (None, "benigno", "maligno"):
        raise ErroValidacao("Parâmetro 'classificacao' deve ser 'benigno' ou 'maligno'.")

    identificado = args.get("identificado") or None
    if identificado not in (None, "true", "false"):
        raise ErroValidacao("Parâmetro 'identificado' deve ser 'true' ou 'false'.")

    inicio, fim = _data_param(args, "data_inicio"), _data_param(args, "data_fim")
    if inicio and fim and inicio > fim:
        raise ErroValidacao("'data_inicio' não pode ser posterior a 'data_fim'.")
    cmin, cmax = _float_param(args, "confianca_min"), _float_param(args, "confianca_max")
    if cmin is not None and cmax is not None and cmin > cmax:
        raise ErroValidacao("'confianca_min' não pode ser maior que 'confianca_max'.")

    utc = timezone.utc
    return {
        "pagina": _int_param(args, "pagina", 1, 1, 10**9),
        "limite": _int_param(args, "limite", LIMITE_PADRAO, 1, LIMITE_MAXIMO),
        "classificacao": classificacao,
        "identificado": None if identificado is None else identificado == "true",
        "inicio": datetime.combine(inicio, datetime.min.time(), utc) if inicio else None,
        # data_fim é inclusiva: usa o início do dia seguinte como limite aberto
        "fim_exclusivo": datetime.combine(fim + timedelta(days=1), datetime.min.time(), utc) if fim else None,
        "confianca_min": cmin,
        "confianca_max": cmax,
    }
