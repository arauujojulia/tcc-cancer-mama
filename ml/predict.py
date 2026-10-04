"""
Classificação de um novo caso (US02, US03).

Os artefatos (scaler + modelos + metadados) são carregados uma única vez
e mantidos em memória, o que mantém a inferência bem abaixo do limite de
2 segundos (RNF02).
"""
import json
import os
from functools import lru_cache

import joblib

from ml.data import FEATURE_COLUMNS
from ml.train import ARTIFACTS_DIR, LIMIAR_PADRAO

# Faixas de confiança (probabilidade da classe prevista).
FAIXA_ALTA = 0.90
FAIXA_MEDIA = 0.70


class ModeloIndisponivel(FileNotFoundError):
    """Artefatos não encontrados — rode 'python -m ml.train'."""


def _ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def load_bundle(artifacts_dir=ARTIFACTS_DIR):
    """Carrega scaler, modelos e metadados uma vez e reaproveita."""
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    if not os.path.exists(metrics_path) or not os.path.exists(scaler_path):
        raise ModeloIndisponivel(
            f"Artefatos não encontrados em {artifacts_dir}. Rode 'python -m ml.train' primeiro."
        )
    metrics = _ler_json(metrics_path)
    modelos = {}
    for nome in metrics["modelos"]:
        caminho = os.path.join(artifacts_dir, f"{nome}.joblib")
        if not os.path.exists(caminho):
            raise ModeloIndisponivel(f"Modelo '{nome}' não encontrado em {artifacts_dir}.")
        modelos[nome] = joblib.load(caminho)
    return {
        "scaler": joblib.load(scaler_path),
        "modelos": modelos,
        "metrics": metrics,
        "melhor_modelo": metrics["melhor_modelo"],
    }


def clear_cache():
    load_bundle.cache_clear()


def faixa_confianca(confianca: float) -> str:
    if confianca >= FAIXA_ALTA:
        return "alta"
    if confianca >= FAIXA_MEDIA:
        return "media"
    return "baixa"


def predict(features: list[float], model_name: str | None = None,
            artifacts_dir: str = ARTIFACTS_DIR) -> dict:
    """
    features: lista com as 30 características, na ordem de
    ml.data.FEATURE_COLUMNS.

    O modelo padrão é o melhor da comparação (metrics.json). A decisão usa
    o limiar de segurança do modelo (RNF04): o caso é classificado como
    maligno quando P(maligno) >= limiar, que pode ser menor que 0,5 para
    reduzir falsos negativos.
    """
    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(f"Esperadas {len(FEATURE_COLUMNS)} características, recebidas {len(features)}.")

    bundle = load_bundle(artifacts_dir)
    nome = model_name or bundle["melhor_modelo"]
    if nome not in bundle["modelos"]:
        raise KeyError(f"Modelo desconhecido: {nome}")

    limiar = bundle["metrics"]["modelos"][nome]["limiar_seguranca"]
    X = bundle["scaler"].transform([features])
    p_maligno = float(bundle["modelos"][nome].predict_proba(X)[0][1])

    maligno = p_maligno >= limiar
    confianca = p_maligno if maligno else 1.0 - p_maligno

    return {
        "classificacao": "maligno" if maligno else "benigno",
        "probabilidade_maligno": round(p_maligno, 4),
        "confianca": round(confianca, 4),
        "faixa_confianca": faixa_confianca(confianca),
        "limiar_utilizado": limiar,
        # maligno decidido abaixo de 50%: efeito do limiar de segurança
        "acionou_limiar_seguranca": bool(maligno and p_maligno < LIMIAR_PADRAO),
        "modelo_utilizado": nome,
    }
