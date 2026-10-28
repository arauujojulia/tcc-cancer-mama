"""
Testes unitários de ml/predict.py (sem Flask/banco).

predict() decide a classe por P(maligno) >= limiar — nunca por
sklearn.predict() — e calcula a faixa de confiança nas três faixas
(alta >= 0,90, media 0,70-0,90, baixa < 0,70).
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ml import predict as predict_mod
from ml.data import FEATURE_COLUMNS


class _ModeloFalso:
    """Simula um classificador scikit-learn: predict_proba controlado nos testes,
    predict() devolvendo deliberadamente o OPOSTO do que P(maligno)>=limiar daria,
    para comprovar que predict() nunca é chamado nessa decisão."""

    def __init__(self, p_maligno):
        self._p = p_maligno

    def predict_proba(self, X):
        return np.array([[1 - self._p, self._p]])

    def predict(self, X):  # pragma: no cover - não deve ser chamado por ml.predict.predict()
        raise AssertionError("predict() do estimador não deve ser chamado por ml.predict.predict().")


def _bundle_falso(p_maligno, limiar):
    return {
        "scaler": MagicMock(transform=lambda X: X),
        "modelos": {"modelo_teste": _ModeloFalso(p_maligno)},
        "metrics": {"modelos": {"modelo_teste": {"limiar_seguranca": limiar}}},
        "melhor_modelo": "modelo_teste",
    }


class TestPredict(unittest.TestCase):
    def setUp(self):
        predict_mod.clear_cache()
        self.features = [0.0] * len(FEATURE_COLUMNS)

    def tearDown(self):
        predict_mod.clear_cache()

    def test_decide_maligno_quando_proba_maior_igual_limiar(self):
        with patch.object(predict_mod, "load_bundle", return_value=_bundle_falso(0.5, 0.5)):
            resultado = predict_mod.predict(self.features, model_name="modelo_teste")
        self.assertEqual(resultado["classificacao"], "maligno")

    def test_decide_benigno_quando_proba_menor_que_limiar(self):
        with patch.object(predict_mod, "load_bundle", return_value=_bundle_falso(0.49, 0.5)):
            resultado = predict_mod.predict(self.features, model_name="modelo_teste")
        self.assertEqual(resultado["classificacao"], "benigno")

    def test_limiar_de_seguranca_abaixo_de_0_5_pode_classificar_maligno(self):
        # p_maligno=0.35 < 0,5 mas >= limiar de segurança 0,31 => maligno.
        with patch.object(predict_mod, "load_bundle", return_value=_bundle_falso(0.35, 0.31)):
            resultado = predict_mod.predict(self.features, model_name="modelo_teste")
        self.assertEqual(resultado["classificacao"], "maligno")
        self.assertTrue(resultado["acionou_limiar_seguranca"])

    def test_faixa_confianca_alta(self):
        self.assertEqual(predict_mod.faixa_confianca(0.90), "alta")
        self.assertEqual(predict_mod.faixa_confianca(0.99), "alta")

    def test_faixa_confianca_media_sem_acento(self):
        self.assertEqual(predict_mod.faixa_confianca(0.70), "media")
        self.assertEqual(predict_mod.faixa_confianca(0.89), "media")

    def test_faixa_confianca_baixa(self):
        self.assertEqual(predict_mod.faixa_confianca(0.0), "baixa")
        self.assertEqual(predict_mod.faixa_confianca(0.69), "baixa")

    def test_numero_de_caracteristicas_incorreto_levanta_erro(self):
        with patch.object(predict_mod, "load_bundle", return_value=_bundle_falso(0.5, 0.5)):
            with self.assertRaises(ValueError):
                predict_mod.predict([0.0, 1.0], model_name="modelo_teste")


if __name__ == "__main__":
    unittest.main()
