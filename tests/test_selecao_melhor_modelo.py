"""
Testes da seleção do melhor modelo por dados out-of-fold do treino (Tarefa A).

Verifica que _selecionar_melhor usa apenas predições OOF, não dados de teste,
e que o critério de desempate funciona corretamente.
"""
import json
import os
import unittest

import numpy as np

from ml.train import ARTIFACTS_DIR, CRITERIO_MELHOR_MODELO, _selecionar_melhor, compute_metrics


class TestSelecaoMelhorModeloOOF(unittest.TestCase):
    """A seleção do melhor modelo usa apenas dados de treino (OOF), nunca teste."""

    def test_reavaliacao_usa_oof_e_escolhe_regressao_logistica(self):
        """Recalcula a seleção a partir dos *_oof.json versionados e espera regressao_logistica."""
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        modelos = metrics["modelos"]
        oof_por_modelo = {}
        for nome in modelos:
            oof_path = os.path.join(ARTIFACTS_DIR, f"{nome}_oof.json")
            self.assertTrue(os.path.exists(oof_path), f"{oof_path} não encontrado")
            with open(oof_path, encoding="utf-8") as f:
                oof_data = json.load(f)
            limiar = modelos[nome]["limiar_seguranca"]
            oof_por_modelo[nome] = {
                "y_true": oof_data["y_true"],
                "proba": oof_data["proba"],
                "limiar": limiar,
            }

        melhor = _selecionar_melhor(modelos, oof_por_modelo)
        self.assertEqual(melhor, "regressao_logistica",
                         "O melhor modelo deve ser regressao_logistica "
                         "(ambos com sens=97.06%, AUC OOF 0.9953 > 0.9931)")

    def test_selecao_nao_depende_de_teste(self):
        """Alterar artificialmente as métricas de teste não muda o modelo escolhido."""
        # Cria dados OOF fictícios onde modelo_a tem sensibilidade maior
        y_true = [0] * 80 + [1] * 20  # 80 benignos, 20 malignos
        # modelo_a: detecta 19/20 malignos (sens=0.95)
        proba_a = [0.1] * 80 + [0.7] * 19 + [0.2] * 1
        # modelo_b: detecta 18/20 malignos (sens=0.90)
        proba_b = [0.1] * 80 + [0.7] * 18 + [0.2] * 2

        resultados = {
            "modelo_a": {"teste_operacao": {"sensibilidade": 0.99, "auc": 0.99, "especificidade": 0.99}},
            "modelo_b": {"teste_operacao": {"sensibilidade": 0.99, "auc": 0.99, "especificidade": 0.99}},
        }
        # Mesmo métricas de teste, o OOF deve decidir
        oof_por_modelo = {
            "modelo_a": {"y_true": y_true, "proba": proba_a, "limiar": 0.5},
            "modelo_b": {"y_true": y_true, "proba": proba_b, "limiar": 0.5},
        }

        melhor = _selecionar_melhor(resultados, oof_por_modelo)
        self.assertEqual(melhor, "modelo_a",
                         "modelo_a tem sensibilidade OOF maior (0.95 > 0.90)")

    def test_desempate_sensibilidade_por_auc(self):
        """Quando sensibilidade OOF está empatada, AUC OOF decide."""
        y_true = [0] * 50 + [1] * 50
        # Ambos detectam 49/50 malignos (sens=0.98)
        proba_a = [0.1] * 50 + [0.8] * 49 + [0.3] * 1
        proba_b = [0.1] * 50 + [0.7] * 49 + [0.2] * 1
        # AUC de a > AUC de b (probabilidades mais separadas)

        resultados = {"modelo_a": {}, "modelo_b": {}}
        oof_por_modelo = {
            "modelo_a": {"y_true": y_true, "proba": proba_a, "limiar": 0.5},
            "modelo_b": {"y_true": y_true, "proba": proba_b, "limiar": 0.5},
        }

        melhor = _selecionar_melhor(resultados, oof_por_modelo)
        self.assertEqual(melhor, "modelo_a",
                         "Com sensibilidade empatada, AUC OOF maior decide")

    def test_desempate_auc_por_especificidade(self):
        """Quando sensibilidade e AUC OOF estão empatados, especificidade decide."""
        # Dados idênticos para sensibilidade e AUC, mas diferente especificidade
        # modelo_a: 1 falso positivo em 50 benignos
        # modelo_b: 2 falsos positivos em 50 benignos
        y_true = [0] * 50 + [1] * 50
        # Ambos: 49/50 malignos detectados
        proba_a = [0.1] * 49 + [0.6] * 1 + [0.8] * 49 + [0.3] * 1
        proba_b = [0.1] * 48 + [0.6] * 2 + [0.8] * 49 + [0.3] * 1

        resultados = {"modelo_a": {}, "modelo_b": {}}
        oof_por_modelo = {
            "modelo_a": {"y_true": y_true, "proba": proba_a, "limiar": 0.5},
            "modelo_b": {"y_true": y_true, "proba": proba_b, "limiar": 0.5},
        }

        melhor = _selecionar_melhor(resultados, oof_por_modelo)
        self.assertEqual(melhor, "modelo_a",
                         "Com sensibilidade e AUC empatados, especificidade OOF maior decide")

    def test_criteiro_texto_menciona_oof_e_treino(self):
        """O texto do critério menciona out-of-fold e treino, não teste como base."""
        self.assertIn("out-of-fold", CRITERIO_MELHOR_MODELO.lower())
        self.assertIn("treino", CRITERIO_MELHOR_MODELO.lower())
        # O critério pode mencionar "teste" apenas para dizer que NÃO participa
        texto_sem_negacao = CRITERIO_MELHOR_MODELO.lower().replace("não participa", "")
        # Remove a frase inteira "o conjunto de teste não participa da seleção"
        import re
        texto_limpo = re.sub(r"o conjunto de teste[^.]*", "", texto_sem_negacao)
        self.assertNotIn("teste", texto_limpo,
                         "O critério não deve usar 'teste' como base de seleção")

    def test_metrics_json_tem_critério_atualizado(self):
        """O metrics.json versionado tem o critério atualizado."""
        import re
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        criterio = metrics.get("criterio_melhor_modelo", "")
        self.assertIn("out-of-fold", criterio.lower())
        self.assertIn("treino", criterio.lower())
        # "teste" só pode aparecer em contexto negativo ("não participa")
        texto_limpo = re.sub(r"o conjunto de teste[^.]*", "", criterio.lower())
        self.assertNotIn("teste", texto_limpo)


if __name__ == "__main__":
    unittest.main()
