"""
Testes de integração: autenticação simples (D3, D4).
"""
from werkzeug.security import generate_password_hash

from db.models import User, db
from services.auth import METODO_HASH
from tests.base import ApiTestCase


class TestAuth(ApiTestCase):
    def test_register_cria_usuario_e_devolve_201(self):
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Dra. Fulana", "email": "fulana@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        self.assertEqual(resp.status_code, 201)
        corpo = resp.get_json()
        self.assertEqual(corpo["user"]["email"], "fulana@exemplo.com")
        self.assertEqual(corpo["user"]["profile"], "medico")
        self.assertNotIn("password_hash", corpo["user"])

    def test_register_email_duplicado_devolve_400(self):
        self.registrar_e_logar(email="dup@exemplo.com")
        resp = self.client.post("/api/v1/auth/register", json={
            "name": "Outra Pessoa", "email": "dup@exemplo.com",
            "password": "outrasenha", "profile": "pesquisador",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["code"], "VALIDATION_ERROR")

    def test_login_com_credenciais_corretas_devolve_200(self):
        self.client.post("/api/v1/auth/logout")
        self.registrar_e_logar(email="login@exemplo.com")
        self.client.post("/api/v1/auth/logout")
        resp = self.client.post("/api/v1/auth/login", json={
            "email": "login@exemplo.com", "password": "senha1234",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["user"]["email"], "login@exemplo.com")

    def test_login_com_senha_errada_devolve_401_generico(self):
        self.registrar_e_logar(email="senha@exemplo.com")
        resp = self.client.post("/api/v1/auth/login", json={
            "email": "senha@exemplo.com", "password": "errada123",
        })
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["code"], "UNAUTHENTICATED")

    def test_login_com_email_inexistente_devolve_401_generico(self):
        resp = self.client.post("/api/v1/auth/login", json={
            "email": "nao-existe@exemplo.com", "password": "qualquercoisa",
        })
        self.assertEqual(resp.status_code, 401)

    def test_me_sem_sessao_devolve_401(self):
        resp = self.client.get("/api/v1/auth/me")
        self.assertEqual(resp.status_code, 401)

    def test_me_com_sessao_devolve_200(self):
        self.registrar_e_logar(email="me@exemplo.com")
        resp = self.client.get("/api/v1/auth/me")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["user"]["email"], "me@exemplo.com")

    def test_hash_gravado_usa_pbkdf2_sha256_como_d4_documenta(self):
        """
        D4 fixa o algoritmo: o hash não pode depender do padrão da versão do
        Werkzeug instalada (que hoje seria scrypt).
        """
        self.assertEqual(METODO_HASH, "pbkdf2:sha256")
        self.registrar_e_logar(email="hash@exemplo.com")
        user = User.query.filter_by(email="hash@exemplo.com").first()
        self.assertTrue(user.password_hash.startswith("pbkdf2:sha256:"))
        self.assertNotIn("senha1234", user.password_hash)

    def test_login_de_conta_antiga_com_hash_scrypt_continua_funcionando(self):
        """
        Os usuários do banco de demonstração foram gravados com o padrão
        anterior (`scrypt:32768:8:1$`). `check_password_hash` escolhe o
        algoritmo pelo prefixo do hash armazenado, então fixar pbkdf2 no
        cadastro não invalida essas contas.
        """
        antigo = generate_password_hash("demo12345", method="scrypt")
        self.assertTrue(antigo.startswith("scrypt:"))
        db.session.add(User(
            name="Conta Antiga", email="antiga@exemplo.com",
            password_hash=antigo, profile="medico",
        ))
        db.session.commit()

        resp = self.client.post("/api/v1/auth/login", json={
            "email": "antiga@exemplo.com", "password": "demo12345",
        })
        self.assertEqual(resp.status_code, 200, resp.get_json())
        self.assertEqual(resp.get_json()["user"]["email"], "antiga@exemplo.com")

    def test_logout_encerra_sessao(self):
        self.registrar_e_logar(email="logout@exemplo.com")
        resp = self.client.post("/api/v1/auth/logout")
        self.assertEqual(resp.status_code, 204)
        resp_me = self.client.get("/api/v1/auth/me")
        self.assertEqual(resp_me.status_code, 401)


if __name__ == "__main__":
    import unittest
    unittest.main()
