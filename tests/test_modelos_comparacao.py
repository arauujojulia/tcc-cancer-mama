"""
Testes de `GET /api/v1/models/comparison` (defeitos A e C).

Defeito C: a tela do Laboratório mostrava o texto fixo "Conjunto de teste (114
casos, 42 malignos)", derivado da referência 80/20, ao lado de linhas e matrizes
de confusão que podiam vir de execuções com outra partição (171 casos, 64
malignos). O tamanho e a composição do teste passam a sair dos dados, por
execução, e a mistura de partições fica explícita na resposta.

Defeito A: `train_oof_sensitivity` (portão out-of-fold do TREINO) é um campo
separado de `operating_point.sensitivity` (conjunto de teste no limiar vigente).
"""
import unittest

from db.models import TrainedModel, db
from services import modelos as svc_modelos
from tests.base import ApiAutenticadoTestCase


class TestComparacaoDeModelos(ApiAutenticadoTestCase):
    def _sincronizar_referencia(self):
        modelos = svc_modelos.sincronizar_modelos()
        self.assertTrue(modelos, "artefatos de referência ausentes: rode python -m ml.train")
        return modelos

    def _comparar(self):
        resp = self.client.get("/api/v1/models/comparison")
        self.assertEqual(resp.status_code, 200, resp.get_json())
        return resp.get_json()

    def test_composicao_do_teste_vem_da_matriz_de_confusao_de_cada_linha(self):
        self._sincronizar_referencia()
        corpo = self._comparar()

        self.assertTrue(corpo["models"])
        for item in corpo["models"]:
            cm = item["operating_point"]["confusion_matrix"]
            self.assertEqual(
                item["test_set_size"], cm["vn"] + cm["fp"] + cm["fn"] + cm["vp"], item["name"]
            )
            self.assertEqual(item["test_set_malignant"], cm["vp"] + cm["fn"], item["name"])
            self.assertEqual(item["test_size"], 0.2, item["name"])
            # A referência versionada é a partição 80/20: 114 casos, 42 malignos.
            self.assertEqual(item["test_set_size"], 114, item["name"])
            self.assertEqual(item["test_set_malignant"], 42, item["name"])

    def test_referencia_sozinha_nao_mistura_particoes(self):
        self._sincronizar_referencia()
        corpo = self._comparar()
        self.assertFalse(corpo["test_partitions_mixed"])
        self.assertEqual(len({p["test_set_size"] for p in corpo["test_partitions"]}), 1)

    def test_execucao_com_outra_particao_sinaliza_mistura(self):
        self._sincronizar_referencia()
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": 0.3, "grid_search": False,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        run_id = resp.get_json()["training_run_id"]

        corpo = self._comparar()
        self.assertTrue(corpo["test_partitions_mixed"])
        self.assertGreater(len({p["test_set_size"] for p in corpo["test_partitions"]}), 1)

        por_nome = {m["name"]: m for m in corpo["models"]}
        # O modelo retreinado é o da execução nova (linha mais recente).
        self.assertEqual(por_nome["regressao_logistica"]["training_run_id"], run_id)
        self.assertEqual(por_nome["regressao_logistica"]["test_size"], 0.3)
        self.assertEqual(por_nome["regressao_logistica"]["test_set_size"], 171)
        # O SVM continua na referência 80/20 — é justamente o que torna a
        # comparação entre as duas linhas inválida.
        self.assertEqual(por_nome["svm"]["test_size"], 0.2)
        self.assertEqual(por_nome["svm"]["test_set_size"], 114)

    def test_default_point_vem_da_execucao_de_cada_modelo(self):
        self._sincronizar_referencia()
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": 0.3, "grid_search": False,
        })
        run_id = resp.get_json()["training_run_id"]

        item = {m["name"]: m for m in self._comparar()["models"]}["regressao_logistica"]
        self.assertIsNotNone(item["default_point"])
        self.assertEqual(item["default_point"]["threshold"], 0.5)
        cm = item["default_point"]["confusion_matrix"]
        # 171 casos: o limiar padrão é avaliado na partição DESTA execução, não
        # nos 114 casos da referência.
        self.assertEqual(cm["vn"] + cm["fp"] + cm["fn"] + cm["vp"], 171)
        self.assertIsNotNone(item["cross_validation"])
        self.assertEqual(item["training_run_id"], run_id)

    def test_execucao_descartada_devolve_default_point_nulo_sem_quebrar(self):
        self._sincronizar_referencia()
        registro = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        registro.artifact_path = "ml/artifacts/runs/999999/regressao_logistica.joblib"
        db.session.commit()

        item = {m["name"]: m for m in self._comparar()["models"]}["regressao_logistica"]
        self.assertIsNone(item["default_point"])
        self.assertIsNone(item["cross_validation"])
        # A composição do teste continua vindo da matriz gravada na linha.
        self.assertEqual(item["test_set_size"], 114)

    def test_portao_oof_e_campo_separado_da_sensibilidade_de_teste(self):
        self._sincronizar_referencia()
        alvo = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        resp = self.client.put(f"/api/v1/models/{alvo.id}/threshold", json={"threshold": 0.01})
        self.assertEqual(resp.status_code, 200, resp.get_json())

        item = {m["name"]: m for m in self._comparar()["models"]}["regressao_logistica"]
        self.assertIsNotNone(item["train_oof_sensitivity"])
        self.assertNotAlmostEqual(
            item["train_oof_sensitivity"], item["operating_point"]["sensitivity"], places=6
        )
        self.assertTrue(item["gate_meets_sensitivity_target"])

    def test_item_sem_matriz_de_confusao_nao_quebra_a_rota(self):
        self._sincronizar_referencia()
        registro = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        registro.confusion_matrix = None
        db.session.commit()

        item = {m["name"]: m for m in self._comparar()["models"]}["regressao_logistica"]
        self.assertIsNone(item["test_set_size"])
        self.assertIsNone(item["test_set_malignant"])

    def test_uma_linha_por_nome_a_mais_recente(self):
        self._sincronizar_referencia()
        for _ in range(2):
            self.client.post("/api/v1/training-runs", json={
                "models": ["regressao_logistica"], "grid_search": False,
            })

        self.assertGreater(TrainedModel.query.filter_by(name="regressao_logistica").count(), 1)
        nomes = [m["name"] for m in self._comparar()["models"]]
        self.assertEqual(len(nomes), len(set(nomes)))
        mais_recente = svc_modelos.modelo_registrado_mais_recente("regressao_logistica")
        item = {m["name"]: m for m in self._comparar()["models"]}["regressao_logistica"]
        self.assertEqual(item["id"], mais_recente.id)


if __name__ == "__main__":
    unittest.main()
