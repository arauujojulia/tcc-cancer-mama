"""
Testes da invariante de coerência interna de `trained_models` (defeito A) e do
artefato `{modelo}_test_proba.json` que a sustenta (D11).

A invariante: em qualquer linha da tabela, as métricas escalares e a matriz de
confusão descrevem o MESMO conjunto de teste no MESMO limiar, então

    sensitivity          == vp / (vp + fn)
    false_negative_rate  == fn / (vp + fn)
    specificity          == vn / (vn + fp)
    accuracy             == (vn + vp) / (vn + fp + fn + vp)

Antes da correção isso falhava: `PUT /models/{id}/threshold` gravava em
`sensitivity` a sensibilidade recalculada nas predições out-of-fold do TREINO e
deixava `specificity`, `accuracy`, `false_negative_rate` e `confusion_matrix`
com os valores do CONJUNTO DE TESTE no limiar antigo — uma linha misturando dois
conjuntos de avaliação (ver `.agents/tasks/correcoes-metricas-documentacao/
diagnostico-antes.txt`, modelos id=2 e id=6).
"""
import json
import os
import unittest
from unittest.mock import patch

from db.models import TrainedModel
from ml.train import ARTIFACTS_DIR
from services import modelos as svc_modelos
from tests.base import ApiAutenticadoTestCase

CHAVES_METRICAS = (
    "acuracia", "sensibilidade", "especificidade", "auc", "taxa_falsos_negativos",
)


def _ler_json(*partes):
    with open(os.path.join(*partes), encoding="utf-8") as f:
        return json.load(f)


class TestInvarianteMetricas(ApiAutenticadoTestCase):
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def assert_linha_coerente(self, modelo: TrainedModel):
        """Toda métrica gravada tem de sair da própria `confusion_matrix`."""
        contexto = (
            f"id={modelo.id} {modelo.name} limiar={modelo.threshold} "
            f"matriz={modelo.confusion_matrix}"
        )
        self.assertIsNotNone(modelo.confusion_matrix, contexto)
        cm = modelo.confusion_matrix
        vn, fp, fn, vp = (cm["vn"], cm["fp"], cm["fn"], cm["vp"])
        malignos, benignos, total = vp + fn, vn + fp, vn + fp + fn + vp

        self.assertGreater(malignos, 0, contexto)
        self.assertAlmostEqual(modelo.sensitivity, vp / malignos, places=6, msg=contexto)
        self.assertAlmostEqual(modelo.false_negative_rate, fn / malignos, places=6, msg=contexto)
        self.assertAlmostEqual(modelo.specificity, vn / benignos, places=6, msg=contexto)
        self.assertAlmostEqual(modelo.accuracy, (vn + vp) / total, places=6, msg=contexto)
        self.assertAlmostEqual(
            modelo.sensitivity + modelo.false_negative_rate, 1.0, places=6, msg=contexto
        )

    def assert_tabela_coerente(self):
        modelos = TrainedModel.query.all()
        self.assertTrue(modelos, "nenhuma linha de trained_models para verificar")
        for modelo in modelos:
            self.assert_linha_coerente(modelo)

    def _sincronizar_referencia(self):
        modelos = svc_modelos.sincronizar_modelos()
        self.assertTrue(modelos, "artefatos de referência ausentes: rode python -m ml.train")
        return modelos

    def _treinar(self, test_size=0.2):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": test_size, "grid_search": False,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        self.assertEqual(resp.get_json()["status"], "success", resp.get_json())
        return resp.get_json()["training_run_id"]

    # ------------------------------------------------------------------
    # (a) referência sincronizada
    # ------------------------------------------------------------------
    def test_referencia_sincronizada_grava_linhas_coerentes(self):
        self._sincronizar_referencia()
        self.assert_tabela_coerente()

    # ------------------------------------------------------------------
    # (b) treino sob demanda
    # ------------------------------------------------------------------
    def test_treino_sob_demanda_grava_linhas_coerentes(self):
        for test_size in (0.2, 0.3):
            with self.subTest(test_size=test_size):
                self._treinar(test_size)
                self.assert_tabela_coerente()

    # ------------------------------------------------------------------
    # (c) PUT /threshold varrendo a tabela inteira
    # ------------------------------------------------------------------
    def test_put_threshold_mantem_a_tabela_inteira_coerente(self):
        self._sincronizar_referencia()
        self._treinar(0.3)
        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")

        for threshold in (0.01, 0.20, 0.45):
            with self.subTest(threshold=threshold):
                resp = self.client.put(
                    f"/api/v1/models/{alvo.id}/threshold", json={"threshold": threshold}
                )
                # 200 (promovido) ou 400 (portão abaixo da meta, nada gravado):
                # nos dois casos nenhuma linha pode ficar meio atualizada.
                self.assertIn(resp.status_code, (200, 400), resp.get_json())
                self.assert_tabela_coerente()

    # ------------------------------------------------------------------
    # (d) regressão direta do caso real id=6 (limiar 0,01 com esp. 0,9907)
    # ------------------------------------------------------------------
    def test_limiar_baixo_derruba_a_especificidade_gravada(self):
        self._sincronizar_referencia()
        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        especificidade_antes = alvo.specificity
        fp_antes = alvo.confusion_matrix["fp"]

        resp = self.client.put(f"/api/v1/models/{alvo.id}/threshold", json={"threshold": 0.01})
        self.assertEqual(resp.status_code, 200, resp.get_json())
        corpo = resp.get_json()

        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        # Era aqui que o defeito aparecia: a especificidade continuava a do
        # limiar antigo (0,9907 num limiar de 0,01, impossível).
        self.assertLess(alvo.specificity, especificidade_antes)
        self.assertGreater(alvo.confusion_matrix["fp"], fp_antes)
        self.assert_linha_coerente(alvo)

        # O portão é do treino out-of-fold; a sensibilidade gravada é do teste.
        self.assertIsNotNone(alvo.train_oof_sensitivity)
        self.assertAlmostEqual(
            alvo.train_oof_sensitivity, corpo["train_oof_sensitivity"], places=4
        )
        self.assertAlmostEqual(alvo.sensitivity, corpo["test_metrics"]["sensitivity"], places=9)
        esperado = svc_modelos.metricas_teste_no_limiar("regressao_logistica", 0.01)
        self.assertAlmostEqual(alvo.sensitivity, esperado["sensibilidade"], places=9)
        self.assertEqual(alvo.confusion_matrix, esperado["matriz_confusao"])
        # A resposta não tem mais a chave ambígua `sensitivity` de topo.
        self.assertNotIn("sensitivity", corpo)

    def test_sensibilidade_de_teste_nao_e_o_portao_quando_os_dois_diferem(self):
        self._sincronizar_referencia()
        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")

        resp = self.client.put(f"/api/v1/models/{alvo.id}/threshold", json={"threshold": 0.01})
        self.assertEqual(resp.status_code, 200, resp.get_json())

        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        portao = svc_modelos.sensibilidade_oof("regressao_logistica", 0.01)
        teste = svc_modelos.metricas_teste_no_limiar("regressao_logistica", 0.01)["sensibilidade"]
        if abs(portao - teste) > 1e-9:
            self.assertNotAlmostEqual(alvo.sensitivity, alvo.train_oof_sensitivity, places=6)
        self.assertAlmostEqual(alvo.sensitivity, teste, places=9)
        self.assertAlmostEqual(alvo.train_oof_sensitivity, portao, places=9)

    # ------------------------------------------------------------------
    # (e) AUC independe do limiar
    # ------------------------------------------------------------------
    def test_auc_nao_muda_com_o_limiar(self):
        self._sincronizar_referencia()
        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        auc_antes = alvo.auc

        resp = self.client.put(f"/api/v1/models/{alvo.id}/threshold", json={"threshold": 0.01})
        self.assertEqual(resp.status_code, 200, resp.get_json())

        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        self.assertAlmostEqual(alvo.auc, auc_antes, places=9)

    # ------------------------------------------------------------------
    # (f) D-A2: as probabilidades persistidas reproduzem a referência
    # ------------------------------------------------------------------
    def test_regeneracao_reproduz_as_metricas_da_referencia(self):
        referencia = _ler_json(ARTIFACTS_DIR, "metrics.json")
        svc_modelos.regenerar_test_proba_ausentes(referencia)

        for nome, r in referencia["modelos"].items():
            with self.subTest(modelo=nome):
                calculado = svc_modelos.metricas_teste_no_limiar(nome, r["limiar_seguranca"])
                esperado = r["teste_operacao"]
                for chave in CHAVES_METRICAS:
                    self.assertAlmostEqual(calculado[chave], esperado[chave], delta=1e-9, msg=chave)
                self.assertEqual(calculado["matriz_confusao"], esperado["matriz_confusao"])

    def test_test_proba_ausente_devolve_503_sem_alterar_a_linha(self):
        self._sincronizar_referencia()
        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        antes = alvo.to_dict()

        with patch.object(
            svc_modelos, "metricas_teste_no_limiar",
            side_effect=svc_modelos.TesteProbaIndisponivel("sem test_proba"),
        ):
            resp = self.client.put(f"/api/v1/models/{alvo.id}/threshold", json={"threshold": 0.01})

        self.assertEqual(resp.status_code, 503, resp.get_json())
        self.assertEqual(resp.get_json()["code"], "MODEL_UNAVAILABLE")
        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        self.assertEqual(alvo.to_dict(), antes)

    # ------------------------------------------------------------------
    # Item 11 — o artefato de probabilidades do teste por execução
    # ------------------------------------------------------------------
    def test_treino_grava_test_proba_da_execucao(self):
        run_id = self._treinar(0.3)
        dir_run = os.path.join(ARTIFACTS_DIR, "runs", str(run_id))
        config = _ler_json(dir_run, "train_config.json")

        dados = _ler_json(dir_run, "regressao_logistica_test_proba.json")
        self.assertEqual(len(dados["y_true"]), config["n_teste"])
        self.assertEqual(len(dados["proba"]), config["n_teste"])
        self.assertEqual(sum(dados["y_true"]), config["distribuicao_teste"]["maligno"])


if __name__ == "__main__":
    unittest.main()
