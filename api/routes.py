"""
Sprint 1 — fluxo principal da classificação:
  US01: receber as 30 características quantitativas da biópsia
  US02: classificar entre tumor benigno e maligno
  US03: retornar o grau de confiança da classificação

Ainda NÃO persiste no banco (isso é a US07, da Sprint 2 — histórico das
avaliações). Por enquanto o endpoint só classifica e responde.
"""

from flask import Blueprint, jsonify, request

from ml.data import FEATURE_COLUMNS
from ml.predict import predict

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/biopsias/classificar", methods=["POST"])
def classificar():
    payload = request.get_json(silent=True) or {}

    faltando = [nome for nome in FEATURE_COLUMNS if nome not in payload]
    if faltando:
        return jsonify({
            "erro": "Características faltando no formulário",
            "campos_faltando": faltando,
        }), 400

    try:
        features = [float(payload[nome]) for nome in FEATURE_COLUMNS]
    except (TypeError, ValueError):
        return jsonify({"erro": "Todas as características devem ser numéricas"}), 400

    try:
        resultado = predict(features)
    except FileNotFoundError as e:
        # Modelos ainda não treinados — rode 'python -m ml.train' primeiro.
        return jsonify({"erro": str(e)}), 503

    resultado["aviso"] = (
        "Esta classificação é uma ferramenta de apoio e não substitui "
        "o laudo histopatológico e a avaliação médica."
    )
    return jsonify(resultado), 200