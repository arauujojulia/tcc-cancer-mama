"""
Testes de integração: POST/GET /api/v1/predictions (US02, US03, US07, RNF02).

Usa os artefatos reais de ml/artifacts/ (gerados por `python -m ml.train`)
para o caminho feliz, e substitui ml.predict.predict (import usado em
api.routes) para simular timeout (504) e modelo indisponível (503) sem
re-treinar nada.
"""
import unittest
from unittest.mock import patch

from db.models import Evaluation, db
from ml.predict import ModeloIndisponivel
from tests.base import ApiAutenticadoTestCase, ApiTestCase


def _payload_valido(mode="anonimo", patient_name=None, patient_record=None):
    from ml.data import feature_schema
    return {
        "features": dict(feature_schema()["exemplos"]["benigno"]),
        "mode": mode,
        "patient_name": patient_name,
        "patient_record": patient_record,
    }


class TestPredictions(ApiTestCase):
    """Testes de POST/GET /predictions usam perfil médico (único autorizado)."""

    def setUp(self):
        super().setUp()
        self.user = self.registrar_e_logar(email="medico@exemplo.com", profile="medico")

    def test_post_predictions_200_grava_evaluation_com_campos_em_ingles(self):
        resp = self.client.post("/api/v1/predictions", json=_payload_valido())
        self.assertEqual(resp.status_code, 200, resp.get_json())
        corpo = resp.get_json()
        for campo in ("evaluation_id", "label", "probability", "confidence_band",
                      "model_name", "threshold_used", "elapsed_ms", "disclaimer"):
            self.assertIn(campo, corpo)
        self.assertIn(corpo["label"], ("benigno", "maligno"))

        av = db.session.get(Evaluation, corpo["evaluation_id"])
        self.assertIsNotNone(av)
        self.assertEqual(av.mode, "anonimo")
        self.assertIsNone(av.patient_name)
        # Sessão obrigatória (D10) ⇒ a autoria passa a ser sempre registrada.
        self.assertEqual(av.user_id, self.user["id"])

    def test_modo_identificado_grava_patient_name_e_record_cifrados(self):
        """No modo identificado, nome/prontuário são cifrados no banco e descriptografados ao ler."""
        resp = self.client.post("/api/v1/predictions", json=_payload_valido(
            mode="identificado", patient_name="Fulana de Tal", patient_record="PRT-0042",
        ))
        self.assertEqual(resp.status_code, 200, resp.get_json())
        av = db.session.get(Evaluation, resp.get_json()["evaluation_id"])
        self.assertEqual(av.mode, "identificado")
        # TypeDecorator descriptografa transparentemente
        self.assertEqual(av.patient_name, "Fulana de Tal")
        self.assertEqual(av.patient_record, "PRT-0042")

    def test_dados_cifrados_nao_aparecem_em_claro_no_banco(self):
        """Lendo a coluna crua, o valor começa com o prefixo de versão e não contém o texto em claro."""
        from sqlalchemy import text
        self.client.post("/api/v1/predictions", json=_payload_valido(
            mode="identificado", patient_name="Fulana de Tal", patient_record="PRT-0042",
        ))
        rows = db.session.execute(text(
            "SELECT patient_name, patient_record FROM evaluations WHERE patient_name IS NOT NULL"
        )).fetchall()
        self.assertTrue(len(rows) > 0)
        for row in rows:
            name_raw, record_raw = row
            if name_raw is not None:
                self.assertTrue(name_raw.startswith("enc:v1:"),
                                f"Valor cru não cifrado: {name_raw[:30]}")
                self.assertNotIn("Fulana", name_raw)
            if record_raw is not None:
                self.assertTrue(record_raw.startswith("enc:v1:"),
                                f"Valor cru não cifrado: {record_raw[:30]}")
                self.assertNotIn("PRT-0042", record_raw)

    def test_rnf02_elapsed_ms_presente_e_abaixo_de_2000_no_caminho_feliz(self):
        resp = self.client.post("/api/v1/predictions", json=_payload_valido())
        corpo = resp.get_json()
        self.assertLess(corpo["elapsed_ms"], 2000)

    def test_modelo_lento_devolve_504_timeout_e_nada_e_gravado(self):
        total_antes = Evaluation.query.count()

        def _lento(features, model_name=None, artifacts_dir=None):
            import time
            time.sleep(2.1)
            return {
                "classificacao": "benigno", "probabilidade_maligno": 0.1,
                "confianca": 0.9, "faixa_confianca": "alta",
                "limiar_utilizado": 0.5, "acionou_limiar_seguranca": False,
                "modelo_utilizado": "modelo_teste",
            }

        with patch("api.routes.predict", side_effect=_lento):
            resp = self.client.post("/api/v1/predictions", json=_payload_valido())
        self.assertEqual(resp.status_code, 504)
        self.assertEqual(resp.get_json()["code"], "TIMEOUT")
        self.assertEqual(Evaluation.query.count(), total_antes)

    def test_modelo_ausente_devolve_503(self):
        with patch("api.routes.predict", side_effect=ModeloIndisponivel("sem artefatos")):
            resp = self.client.post("/api/v1/predictions", json=_payload_valido())
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["code"], "MODEL_UNAVAILABLE")

    def test_get_predictions_paginado_com_envelope_items_page_limit_total(self):
        for _ in range(3):
            self.client.post("/api/v1/predictions", json=_payload_valido())
        resp = self.client.get("/api/v1/predictions?limit=2&page=1")
        corpo = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(corpo["items"]), 2)
        self.assertEqual(corpo["page"], 1)
        self.assertEqual(corpo["limit"], 2)
        self.assertEqual(corpo["total"], 3)

    def test_get_predictions_filtro_por_label(self):
        self.client.post("/api/v1/predictions", json=_payload_valido())
        resp = self.client.get("/api/v1/predictions?label=maligno")
        self.assertEqual(resp.status_code, 200)
        for item in resp.get_json()["items"]:
            self.assertEqual(item["label"], "maligno")

    def test_imutabilidade_update_levanta_erro(self):
        resp = self.client.post("/api/v1/predictions", json=_payload_valido())
        av = db.session.get(Evaluation, resp.get_json()["evaluation_id"])
        av.label = "maligno"
        with self.assertRaises(RuntimeError):
            db.session.commit()
        db.session.rollback()

    def test_imutabilidade_delete_levanta_erro(self):
        resp = self.client.post("/api/v1/predictions", json=_payload_valido())
        av = db.session.get(Evaluation, resp.get_json()["evaluation_id"])
        db.session.delete(av)
        with self.assertRaises(RuntimeError):
            db.session.commit()
        db.session.rollback()


if __name__ == "__main__":
    unittest.main()
