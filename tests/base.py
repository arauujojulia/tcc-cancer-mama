"""
Base comum dos testes de integração: app Flask de teste com SQLite em
memória (TESTING=True), sem sincronizar os modelos pré-treinados do
repositório automaticamente (create_app já pula sincronizar_modelos() em
modo TESTING — ver app.py).

O `tearDown` apaga as pastas de `ml/artifacts/runs/` criadas durante o teste:
o treino sob demanda grava por execução (D8), então a referência versionada em
`ml/artifacts/` nunca é tocada e o repositório fica limpo depois da suíte.
"""
import os
import shutil
import unittest

from app import create_app
from db.models import db
from ml.train import RUNS_DIR


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app({
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        })
        self.ctx = self.app.app_context()
        self.ctx.push()
        self.client = self.app.test_client()
        self._runs_preexistentes = self._entradas_de_runs()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()
        for entrada in self._entradas_de_runs() - self._runs_preexistentes:
            shutil.rmtree(os.path.join(RUNS_DIR, entrada), ignore_errors=True)

    @staticmethod
    def _entradas_de_runs() -> set[str]:
        return set(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else set()

    def registrar_e_logar(self, email="pesquisadora@exemplo.com", profile="pesquisador"):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Usuária de Teste", "email": email,
            "password": "senha1234", "profile": profile,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        return resp.get_json()["user"]


class ApiAutenticadoTestCase(ApiTestCase):
    """
    Variante com sessão aberta: todas as rotas de dados exigem sessão válida
    (D10), então os testes que as exercitam precisam estar autenticados. O
    perfil usado é irrelevante — não existe trava por perfil (D3).
    """

    def setUp(self):
        super().setUp()
        self.user = self.registrar_e_logar()  # o register já abre a sessão
