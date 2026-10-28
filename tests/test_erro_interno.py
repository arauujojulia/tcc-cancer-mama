"""
Teste de integração do 500 `INTERNAL_ERROR`.

O envelope de erro `{"error", "code"}` é contratual para toda falha, inclusive a
inesperada (README e `.agents/ESPECIFICACAO-CONSOLIDADA.md`). Antes do handler de
`app.py` uma exceção não tratada devolvia a página HTML padrão do Flask, então o
caminho é exercitado de verdade aqui: uma rota real é forçada a falhar.
"""
from unittest.mock import patch

from tests.base import ApiAutenticadoTestCase


class TestErroInterno(ApiAutenticadoTestCase):
    def test_excecao_nao_tratada_devolve_envelope_json_e_nao_html(self):
        with patch("api.routes.feature_schema", side_effect=RuntimeError("falha simulada")):
            with self.assertLogs(self.app.logger, level="ERROR") as registro:
                resp = self.client.get("/api/v1/features/schema")

        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.mimetype, "application/json")
        corpo = resp.get_json()
        self.assertEqual(corpo["code"], "INTERNAL_ERROR")
        self.assertIsInstance(corpo["error"], str)
        self.assertTrue(corpo["error"])
        # Nada de HTML do Flask nem de stack trace vazando na resposta.
        self.assertNotIn("<html", resp.get_data(as_text=True).lower())
        self.assertNotIn("Traceback", resp.get_data(as_text=True))
        self.assertNotIn("falha simulada", resp.get_data(as_text=True))
        # O erro tem de ficar registrado no log do servidor, com o traceback.
        self.assertTrue(any("INTERNAL_ERROR" in linha for linha in registro.output))
        self.assertTrue(any("falha simulada" in linha for linha in registro.output))

    def test_handler_nao_sequestra_os_erros_http_normais(self):
        """O handler cobre exceção não tratada, não 404 — que segue `NOT_FOUND`."""
        resp = self.client.get("/api/v1/predictions/999999")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["code"], "NOT_FOUND")


if __name__ == "__main__":
    import unittest
    unittest.main()
