"""
Testes de cadastro e autenticação de usuários.

Cobre registro via API (perfil médico/pesquisador), hash de senha PBKDF2,
rejeição de e-mail duplicado e perfil inválido, e autenticação com senha
correta/errada. Usa a base de teste comum (tests/base.py) que configura
SQLite em memória ou PostgreSQL via TEST_DATABASE_URL.
"""
import unittest

from werkzeug.security import check_password_hash

from db.models import User, db
from tests.base import ApiTestCase


class TestCadastroUsuario(ApiTestCase):

    def test_registra_medico_e_abre_sessao(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "ANA@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        data = resp.get_json()
        self.assertEqual(data["user"]["profile"], "medico")
        self.assertNotIn("password_hash", data["user"])

        # Verifica no banco que o hash foi gravado com PBKDF2.
        user = db.session.get(User, data["user"]["id"])
        self.assertIsNotNone(user)
        self.assertTrue(user.password_hash.startswith("pbkdf2:"))
        self.assertTrue(check_password_hash(user.password_hash, "senha1234"))

    def test_registra_pesquisador(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Bia", "email": "bia@exemplo.com",
            "password": "senha1234", "profile": "pesquisador",
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        self.assertEqual(resp.get_json()["user"]["profile"], "pesquisador")

    def test_to_dict_nao_expoe_hash(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "ana@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        user_dict = resp.get_json()["user"]
        self.assertNotIn("password_hash", user_dict)
        self.assertIn("id", user_dict)
        self.assertIn("name", user_dict)
        self.assertIn("email", user_dict)
        self.assertIn("profile", user_dict)

    def test_rejeita_perfil_invalido(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "ana@exemplo.com",
            "password": "senha1234", "profile": "admin",
        })
        self.assertEqual(resp.status_code, 400, resp.get_json())
        self.assertEqual(User.query.count(), 0)

    def test_rejeita_senha_curta(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "ana@exemplo.com",
            "password": "curta", "profile": "medico",
        })
        self.assertEqual(resp.status_code, 400, resp.get_json())
        self.assertEqual(User.query.count(), 0)

    def test_rejeita_email_invalido(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "sem-arroba",
            "password": "senha1234", "profile": "medico",
        })
        self.assertEqual(resp.status_code, 400, resp.get_json())
        self.assertEqual(User.query.count(), 0)

    def test_rejeita_email_duplicado(self):
        self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "ana@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Outra", "email": "ANA@exemplo.com",
            "password": "outra1234", "profile": "pesquisador",
        })
        self.assertEqual(resp.status_code, 400, resp.get_json())
        self.assertEqual(User.query.count(), 1)

    def test_perfil_nao_alteravel_pelo_frontend(self):
        """O perfil vem do corpo do register; a autorização é verificada
        no servidor via sessão, não pelo corpo da requisição."""
        self.registrar_e_logar(email="medico@exemplo.com", profile="medico")
        # Tentar classificar como médico funciona.
        features = {f"worst_{c}": 0.1 for c in (
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
        )}
        features.update({f"mean_{c}": 0.1 for c in (
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
        )})
        resp = self.client.post("/api/v1/predictions", json={"features": features})
        # Pode ser 200 ou 503 (modelo não disponível), mas não 403.
        self.assertNotEqual(resp.status_code, 403)


class TestAutenticacao(ApiTestCase):

    def test_login_senha_correta(self):
        self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "ana@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        # Limpa a sessão do register para testar login independente.
        with self.client.session_transaction() as sess:
            sess.clear()
        resp = self.client.post("/api/v1/auth/login", json={
            "email": "ana@exemplo.com", "password": "senha1234",
        })
        self.assertEqual(resp.status_code, 200, resp.get_json())
        self.assertEqual(resp.get_json()["user"]["email"], "ana@exemplo.com")

    def test_login_senha_errada(self):
        self.client.post("/api/v1/auth/register", json={
            "name": "Ana", "email": "ana@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        with self.client.session_transaction() as sess:
            sess.clear()
        resp = self.client.post("/api/v1/auth/login", json={
            "email": "ana@exemplo.com", "password": "errada",
        })
        self.assertEqual(resp.status_code, 401)

    def test_login_email_inexistente(self):
        resp = self.client.post("/api/v1/auth/login", json={
            "email": "nada@exemplo.com", "password": "senha1234",
        })
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main()
