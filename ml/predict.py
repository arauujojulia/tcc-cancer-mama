import os

import joblib

from ml.train import ARTIFACTS_DIR

MODELO_PADRAO = "regressao_logistica" #atualizar após comparar as metricas no US04 e US05

def load_artifacts(model_name=MODELO_PADRAO, artifacts_dir=ARTIFACTS_DIR):
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    model_path = os.path.join(artifacts_dir, f"{model_name}.joblib")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Artefatos do modelo '{model_name}' não encontrados em {artifacts_dir}."
            "Rode 'python -m ml.train' primeiro."
        )
    scaler = joblib.load(scaler_path)
    modelo = joblib.load(model_path)
    return scaler, modelo

def predict(features: list[float],model_name: str = MODELO_PADRAO) -> dict:
    """
    features: lista com as 30 características, na ordem de
    ml.data.FEATURE_COLUMNS.
    """
    scaler, modelo = load_artifacts(model_name)

    X = scaler.transform([features])
    probabilidades = modelo.predict_proba(X)[0] # [P(benigno), P(maligno)]
    predicao = modelo.predict(X)[0]

    classificacao = "maligno" if predicao == 1 else "benigno"
    confianca = float(max(probabilidades))

    return {
        "classificacao": classificacao,
        "confianca": round(confianca, 4),
        "modelo_utilizado": model_name,
    }

