"""
Testes de integração: treino sob demanda e promoção de modelo
(POST /training-runs, GET /training-runs/{id}, PUT /models/{id}/threshold —
design.md §5, RNF04).

O treino real (com grid_search=True) é lento mesmo em CPU para os dois
modelos; para manter a suíte rápida, a maioria dos testes usa
grid_search=False (hiperparâmetros padrão) e só um teste dedicado confirma
que grid_search=True de fato executa sem erro.
"""
import hashlib
import json
import os
import unittest
from unittest.mock import patch

from sqlalchemy.exc import IntegrityError

from db.models import TrainedModel, TrainingRun, db
from ml.train import ARTIFACTS_DIR
from services import modelos as svc_modelos
from tests.base import ApiAutenticadoTestCase

# Artefatos de referência (ml/artifacts/) que o treino sob demanda não pode
# tocar: são a partição 80/20 e os hiperparâmetros citados no artigo (RNF03).
ARTEFATOS_REFERENCIA = (
    "metrics.json", "train_config.json", "split.json", "scaler.joblib",
    "regressao_logistica.joblib", "svm.joblib",
    "regressao_logistica_oof.json", "svm_oof.json",
    "regressao_logistica_test_proba.json", "svm_test_proba.json",
)


def _hashes_referencia() -> dict:
    """SHA-256 dos artefatos de referência presentes em ml/artifacts/."""
    hashes = {}
    for nome in ARTEFATOS_REFERENCIA:
        caminho = os.path.join(ARTIFACTS_DIR, nome)
        if os.path.exists(caminho):
            with open(caminho, "rb") as f:
                hashes[nome] = hashlib.sha256(f.read()).hexdigest()
    return hashes


def _ler_json(*partes):
    with open(os.path.join(*partes), encoding="utf-8") as f:
        return json.load(f)


class TestPostTrainingRuns(ApiAutenticadoTestCase):
    def test_post_training_runs_grava_run_e_trained_models_com_status_final(self):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": 0.2, "grid_search": False,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        corpo = resp.get_json()
        self.assertIn("training_run_id", corpo)
        self.assertEqual(corpo["status"], "success")

        run = db.session.get(TrainingRun, corpo["training_run_id"])
        self.assertIsNotNone(run)
        self.assertEqual(run.status, "success")
        self.assertIsNotNone(run.dataset_sha256)
        self.assertIsNotNone(run.split)
        self.assertIsNotNone(run.library_versions)

        modelos = TrainedModel.query.filter_by(training_run_id=run.id).all()
        self.assertEqual(len(modelos), 1)
        m = modelos[0]
        self.assertEqual(m.name, "regressao_logistica")
        for atributo in ("accuracy", "sensitivity", "specificity", "auc", "false_negative_rate"):
            self.assertIsInstance(getattr(m, atributo), float)
        self.assertIsNotNone(m.roc_points)
        self.assertIn("fpr", m.roc_points)
        self.assertIn("tpr", m.roc_points)
        self.assertTrue(m.is_production)  # único modelo do run => melhor do run

    def test_post_training_runs_com_grid_search_executa_sem_erro(self):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "grid_search": True,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        self.assertEqual(resp.get_json()["status"], "success")

    def test_get_training_run_inexistente_devolve_404(self):
        resp = self.client.get("/api/v1/training-runs/999999")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["code"], "NOT_FOUND")

    def test_get_training_run_traz_os_trained_models_aninhados(self):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "grid_search": False,
        })
        run_id = resp.get_json()["training_run_id"]
        resp_get = self.client.get(f"/api/v1/training-runs/{run_id}")
        self.assertEqual(resp_get.status_code, 200)
        corpo = resp_get.get_json()
        self.assertEqual(corpo["status"], "success")
        self.assertEqual(len(corpo["models"]), 1)
        self.assertEqual(corpo["models"][0]["name"], "regressao_logistica")

    def test_body_invalido_devolve_400_sem_criar_run(self):
        total_antes = TrainingRun.query.count()
        resp = self.client.post("/api/v1/training-runs", json={"models": []})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["code"], "VALIDATION_ERROR")
        self.assertEqual(TrainingRun.query.count(), total_antes)

    def test_falha_durante_o_treino_marca_run_como_failed_e_devolve_201(self):
        with patch("api.routes.train_and_evaluate", side_effect=RuntimeError("falha simulada")):
            resp = self.client.post("/api/v1/training-runs", json={
                "models": ["regressao_logistica"], "grid_search": False,
            })
        self.assertEqual(resp.status_code, 201)
        corpo = resp.get_json()
        self.assertEqual(corpo["status"], "failed")
        run = db.session.get(TrainingRun, corpo["training_run_id"])
        self.assertEqual(run.status, "failed")
        self.assertIn("falha simulada", run.error_message)

    # ------------------------------------------------------------------
    # D8 / RNF03 — isolamento dos artefatos de referência
    # ------------------------------------------------------------------
    def test_treino_sob_demanda_nao_altera_os_artefatos_de_referencia(self):
        """Reproduz o defeito relatado: treino com test_size=0.3 pela API."""
        antes = _hashes_referencia()
        self.assertTrue(antes, "artefatos de referência ausentes: rode python -m ml.train")

        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": 0.3, "grid_search": False,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        self.assertEqual(resp.get_json()["status"], "success")

        self.assertEqual(_hashes_referencia(), antes)

    def test_artefatos_do_run_ficam_no_diretorio_da_execucao(self):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": 0.3, "grid_search": False,
        })
        run_id = resp.get_json()["training_run_id"]
        dir_run = os.path.join(ARTIFACTS_DIR, "runs", str(run_id))

        self.assertTrue(os.path.exists(os.path.join(dir_run, "metrics.json")))
        self.assertEqual(_ler_json(dir_run, "train_config.json")["test_size"], 0.3)
        self.assertEqual(_ler_json(ARTIFACTS_DIR, "train_config.json")["test_size"], 0.2)

        modelo = TrainedModel.query.filter_by(training_run_id=run_id).first()
        self.assertTrue(
            modelo.artifact_path.startswith(f"ml/artifacts/runs/{run_id}/"),
            modelo.artifact_path,
        )

    def test_resposta_informa_o_diretorio_da_execucao(self):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "grid_search": False,
        })
        corpo = resp.get_json()
        self.assertEqual(
            corpo["artifacts_dir"], f"ml/artifacts/runs/{corpo['training_run_id']}"
        )

    def test_figuras_do_run_nao_sobrescrevem_o_report_de_referencia(self):
        csv_referencia = os.path.join(ARTIFACTS_DIR, "report", "metricas.csv")
        conteudo_antes = (
            open(csv_referencia, encoding="utf-8").read()
            if os.path.exists(csv_referencia) else None
        )

        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": 0.3, "grid_search": False,
        })
        run_id = resp.get_json()["training_run_id"]

        self.assertTrue(os.path.exists(
            os.path.join(ARTIFACTS_DIR, "runs", str(run_id), "report", "metricas.csv")
        ))
        if conteudo_antes is not None:
            self.assertEqual(open(csv_referencia, encoding="utf-8").read(), conteudo_antes)


class TestPutThreshold(ApiAutenticadoTestCase):
    def _criar_modelo_treinado(self):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "grid_search": False,
        })
        run_id = resp.get_json()["training_run_id"]
        modelo = TrainedModel.query.filter_by(training_run_id=run_id).first()
        return modelo

    def test_sensibilidade_abaixo_de_0_97_devolve_400_sem_alterar_nada(self):
        modelo = self._criar_modelo_treinado()
        threshold_antes = modelo.threshold
        is_production_antes = modelo.is_production

        with patch.object(svc_modelos, "sensibilidade_oof", return_value=0.80):
            resp = self.client.put(f"/api/v1/models/{modelo.id}/threshold", json={"threshold": 0.9})

        self.assertEqual(resp.status_code, 400)
        corpo = resp.get_json()
        self.assertEqual(corpo["code"], "SENSITIVITY_BELOW_TARGET")
        self.assertEqual(corpo["obtained"], 0.8)
        self.assertEqual(corpo["required"], 0.97)

        db.session.refresh(modelo)
        self.assertEqual(modelo.threshold, threshold_antes)
        self.assertEqual(modelo.is_production, is_production_antes)

    def test_sensibilidade_igual_ou_acima_de_0_97_promove(self):
        modelo = self._criar_modelo_treinado()

        # O mock cobre só o PORTÃO (out-of-fold do treino); as métricas
        # reportadas continuam vindo do conjunto de teste (D11).
        with patch.object(svc_modelos, "sensibilidade_oof", return_value=0.98):
            resp = self.client.put(f"/api/v1/models/{modelo.id}/threshold", json={"threshold": 0.2})

        self.assertEqual(resp.status_code, 200, resp.get_json())
        corpo = resp.get_json()
        self.assertEqual(corpo["train_oof_sensitivity"], 0.98)
        self.assertEqual(corpo["threshold"], 0.2)
        self.assertNotIn("sensitivity", corpo)  # a chave ambígua de topo saiu
        for chave in ("accuracy", "sensitivity", "specificity", "auc",
                      "false_negative_rate", "confusion_matrix"):
            self.assertIn(chave, corpo["test_metrics"])

        db.session.refresh(modelo)
        self.assertEqual(modelo.threshold, 0.2)
        self.assertTrue(modelo.is_production)
        self.assertEqual(modelo.train_oof_sensitivity, 0.98)
        # O portão não pode ter vazado para a coluna das métricas de teste.
        self.assertEqual(modelo.sensitivity, corpo["test_metrics"]["sensitivity"])

    def test_modelo_inexistente_devolve_404(self):
        resp = self.client.put("/api/v1/models/999999/threshold", json={"threshold": 0.5})
        self.assertEqual(resp.status_code, 404)

    def test_oof_ausente_devolve_503(self):
        modelo = self._criar_modelo_treinado()
        with patch.object(svc_modelos, "sensibilidade_oof",
                           side_effect=svc_modelos.OofIndisponivel("sem oof")):
            resp = self.client.put(f"/api/v1/models/{modelo.id}/threshold", json={"threshold": 0.5})
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["code"], "MODEL_UNAVAILABLE")

    def test_body_invalido_devolve_400(self):
        modelo = self._criar_modelo_treinado()
        resp = self.client.put(f"/api/v1/models/{modelo.id}/threshold", json={"threshold": "alto"})
        self.assertEqual(resp.status_code, 400)

    def test_promocao_le_o_oof_do_diretorio_da_execucao(self):
        """Sem resolver runs/{id}/, o OOF não seria encontrado e viria 503 (D8)."""
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": 0.3, "grid_search": False,
        })
        run_id = resp.get_json()["training_run_id"]
        modelo = TrainedModel.query.filter_by(training_run_id=run_id).first()

        resp_put = self.client.put(
            f"/api/v1/models/{modelo.id}/threshold", json={"threshold": 0.01}
        )
        # 200 só é possível se o OOF e as probabilidades do teste de
        # runs/{run_id}/ foram encontrados; com o diretório errado, a rota
        # responderia 503 MODEL_UNAVAILABLE.
        self.assertEqual(resp_put.status_code, 200, resp_put.get_json())
        self.assertGreaterEqual(resp_put.get_json()["train_oof_sensitivity"], 0.97)


class TestUmaLinhaPorExecucao(ApiAutenticadoTestCase):
    """
    D12 / defeito B: a identidade de `trained_models` é (execução, modelo), não
    o hash do artefato. Antes, a unicidade global de `artifact_sha256` fazia uma
    execução nova reaproveitar o registro de outra e herdar o `artifact_path`
    antigo — foi assim que modelos nascidos de execuções 70/30 ficaram
    apontando para `ml/artifacts/regressao_logistica.joblib`, a referência 80/20.
    """

    def _treinar(self, test_size=0.2):
        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "test_size": test_size, "grid_search": False,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        self.assertEqual(resp.get_json()["status"], "success", resp.get_json())
        return resp.get_json()["training_run_id"]

    def test_duas_execucoes_identicas_criam_duas_linhas_com_o_proprio_artefato(self):
        run_a, run_b = self._treinar(), self._treinar()
        self.assertNotEqual(run_a, run_b)

        linhas = TrainedModel.query.filter_by(name="regressao_logistica").all()
        self.assertEqual(len(linhas), 2)
        por_run = {m.training_run_id: m for m in linhas}
        self.assertEqual(set(por_run), {run_a, run_b})

        # Treino determinista com a mesma semente e os mesmos dados: os bytes
        # coincidem, e isso é legítimo — não pode mais colapsar as duas linhas.
        self.assertEqual(por_run[run_a].artifact_sha256, por_run[run_b].artifact_sha256)
        for run_id, modelo in por_run.items():
            self.assertEqual(
                modelo.artifact_path,
                f"ml/artifacts/runs/{run_id}/regressao_logistica.joblib",
            )

    def test_execucao_com_outra_particao_nao_aponta_para_a_referencia(self):
        run_id = self._treinar(0.3)
        modelo = TrainedModel.query.filter_by(training_run_id=run_id).first()
        self.assertEqual(
            modelo.artifact_path, f"ml/artifacts/runs/{run_id}/regressao_logistica.joblib"
        )
        self.assertNotIn("ml/artifacts/regressao_logistica.joblib", modelo.artifact_path)

    def test_sincronizar_duas_vezes_nao_duplica_linhas_nem_run_sintetico(self):
        primeira = svc_modelos.sincronizar_modelos()
        self.assertTrue(primeira, "artefatos de referência ausentes: rode python -m ml.train")
        total_modelos = TrainedModel.query.count()
        total_runs = TrainingRun.query.count()

        svc_modelos.sincronizar_modelos()

        self.assertEqual(TrainedModel.query.count(), total_modelos)
        self.assertEqual(TrainingRun.query.count(), total_runs)

    def test_duas_linhas_com_a_mesma_execucao_e_nome_violam_a_unicidade(self):
        run_id = self._treinar()
        existente = TrainedModel.query.filter_by(training_run_id=run_id).first()

        duplicata = TrainedModel(
            training_run_id=run_id,
            name=existente.name,  # mesmo (training_run_id, name)
            accuracy=existente.accuracy,
            sensitivity=existente.sensitivity,
            specificity=existente.specificity,
            auc=existente.auc,
            false_negative_rate=existente.false_negative_rate,
            threshold=existente.threshold,
            safety_threshold=existente.safety_threshold,
            hyperparameters=existente.hyperparameters,
            random_seed=existente.random_seed,
            artifact_path=existente.artifact_path,
            artifact_sha256=existente.artifact_sha256,
        )
        db.session.add(duplicata)
        with self.assertRaises(IntegrityError):
            db.session.commit()
        db.session.rollback()


if __name__ == "__main__":
    unittest.main()
