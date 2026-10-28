"""
Testes de integração da trava de sessão (D10).

É **autenticação, não autorização**: toda rota de dados exige sessão válida
(401 `UNAUTHENTICATED` sem ela), `/auth/*` e `/health` seguem públicas, e
nenhuma rota verifica o perfil do usuário — médico autenticado chama as rotas
do pesquisador e vice-versa, exatamente como D3 determina.
"""
import re
import unittest

from tests.base import ApiAutenticadoTestCase, ApiTestCase

# Converte a regra do Flask em uma URL concreta: <int:id> → 1, <path:x> → x.
_CONVERSOR = re.compile(r"<(?:int|float):[^>]+>")
_OUTROS = re.compile(r"<[^>]+>")


def _url_concreta(regra: str) -> str:
    return _OUTROS.sub("x", _CONVERSOR.sub("1", regra))


class TestSessaoObrigatoria(ApiTestCase):
    def test_todas_as_rotas_de_dados_devolvem_401_sem_sessao(self):
        """
        Varre o url_map: qualquer rota /api/v1 fora de /api/v1/auth precisa
        responder 401 sem sessão. É a rede de segurança contra uma rota nova
        nascer sem o decorador `exigir_sessao`.
        """
        regras = [
            r for r in self.app.url_map.iter_rules()
            if str(r.rule).startswith("/api/v1") and not str(r.rule).startswith("/api/v1/auth")
        ]
        self.assertTrue(regras, "nenhuma rota de dados encontrada no url_map")

        for regra in regras:
            url = _url_concreta(str(regra.rule))
            for metodo in sorted(regra.methods - {"HEAD", "OPTIONS"}):
                with self.subTest(rota=url, metodo=metodo):
                    resp = self.client.open(url, method=metodo, json={})
                    self.assertEqual(resp.status_code, 401, f"{metodo} {url}")
                    self.assertEqual(resp.get_json()["code"], "UNAUTHENTICATED")

    def test_rotas_de_auth_continuam_publicas(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Fulana", "email": "publica@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())

        self.client.post("/api/v1/auth/logout")
        resp_login = self.client.post("/api/v1/auth/login", json={
            "email": "publica@exemplo.com", "password": "senha-errada",
        })
        # 401 vindo do handler (credenciais inválidas), não da trava de sessão.
        self.assertEqual(resp_login.status_code, 401)
        self.assertEqual(resp_login.get_json()["code"], "UNAUTHENTICATED")

        self.client.post("/api/v1/auth/logout")
        self.assertEqual(self.client.post("/api/v1/auth/logout").status_code, 204)
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 200)


class TestRotasDeDadosComSessao(ApiAutenticadoTestCase):
    def test_com_sessao_as_rotas_de_dados_respondem_normalmente(self):
        resp_schema = self.client.get("/api/v1/features/schema")
        self.assertEqual(resp_schema.status_code, 200)
        self.assertEqual(len(resp_schema.get_json()["campos"]), 30)

        self.assertEqual(self.client.get("/api/v1/predictions").status_code, 200)

        resp_modelos = self.client.get("/api/v1/models")
        self.assertEqual(resp_modelos.status_code, 200)
        self.assertIn("items", resp_modelos.get_json())


class TestSemAutorizacaoPorPerfil(ApiTestCase):
    """D3 removeu a segregação por perfil, não a autenticação: qualquer usuário
    autenticado alcança qualquer rota de dados. Isto é intencional."""

    def test_medico_acessa_rotas_de_pesquisador(self):
        self.registrar_e_logar(email="medico@exemplo.com", profile="medico")

        resp = self.client.post("/api/v1/training-runs", json={
            "models": ["regressao_logistica"], "grid_search": False,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        self.assertEqual(self.client.get("/api/v1/models").status_code, 200)

    def test_pesquisador_acessa_rota_de_classificacao(self):
        from ml.data import feature_schema

        self.registrar_e_logar(email="pesquisador@exemplo.com", profile="pesquisador")
        resp = self.client.post("/api/v1/predictions", json={
            "features": dict(feature_schema()["exemplos"]["benigno"]),
            "mode": "anonimo",
        })
        self.assertEqual(resp.status_code, 200, resp.get_json())


if __name__ == "__main__":
    unittest.main()
