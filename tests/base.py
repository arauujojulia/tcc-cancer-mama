"""
Base comum dos testes de integração: app Flask de teste com SQLite em
memória (TESTING=True), sem sincronizar os modelos pré-treinados do
repositório automaticamente (create_app já pula sincronizar_modelos() em
modo TESTING — ver app.py).

O `tearDown` apaga as pastas de `ml/artifacts/runs/` criadas durante o teste:
o treino sob demanda grava por execução (D8), então a referência versionada em
`ml/artifacts/` nunca é tocada e o repositório fica limpo depois da suíte.

Banco de dados: lê `TEST_DATABASE_URL` do ambiente; se ausente, usa SQLite
em memória. Para PostgreSQL, o nome do banco deve terminar em `_test` para
evitar apagar dados reais.

Criptografia: define `PATIENT_DATA_KEY` fixa para os testes.
"""
import base64
import os
import shutil
import unittest

from sqlalchemy.engine import make_url

from app import create_app
from db.models import db
from ml.train import RUNS_DIR

# Chave fixa de teste (32 bytes em base64) — nunca usada em produção.
os.environ["PATIENT_DATA_KEY"] = base64.b64encode(b"\xab" * 32).decode("ascii")


def _db_url():
    """URL do banco de dados para testes. SQLite em memória se não configurado."""
    url_str = os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")
    url = make_url(url_str)

    # Proteção: recusa PostgreSQL cujo nome do banco não termina em _test.
    if url.drivername.startswith("postgresql"):
        db_name = url.database
        if db_name and not db_name.endswith("_test"):
            raise ValueError(
                f"TEST_DATABASE_URL deve apontar para um banco cujo nome "
                f"termine em '_test' (encontrado: '{db_name}'). "
                f"Isto evita apagar dados reais com db.drop_all()."
            )
    return url_str


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app({
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": _db_url(),
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


class TestDbUrl(unittest.TestCase):
    """Testes unitários da função _db_url() para validação do banco de testes."""

    def test_postgresql_com_sufixo_test_aceita(self):
        """URL PostgreSQL com nome terminando em _test é aceita."""
        import os
        from unittest.mock import patch
        env = dict(os.environ)
        env["TEST_DATABASE_URL"] = "postgresql+psycopg://postgres:postgres@localhost:5432/meu_banco_test"
        with patch.dict(os.environ, env, clear=True):
            url = _db_url()
            self.assertIn("meu_banco_test", url)

    def test_postgresql_sem_sufixo_test_recusa(self):
        """URL PostgreSQL sem _test levanta ValueError."""
        import os
        from unittest.mock import patch
        env = dict(os.environ)
        env["TEST_DATABASE_URL"] = "postgresql+psycopg://postgres:postgres@localhost:5432/producao"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                _db_url()

    def test_sqlite_sempre_aceita(self):
        """URL SQLite em memória é sempre aceita."""
        import os
        from unittest.mock import patch
        env = dict(os.environ)
        env.pop("TEST_DATABASE_URL", None)
        with patch.dict(os.environ, env, clear=True):
            url = _db_url()
            self.assertEqual(url, "sqlite:///:memory:")


class ApiAutenticadoTestCase(ApiTestCase):
    """
    Variante com sessão aberta: todas as rotas de dados exigem sessão válida
    e autorização por perfil. O perfil padrão é pesquisador.
    """

    def setUp(self):
        super().setUp()
        self.user = self.registrar_e_logar()  # o register já abre a sessão
