import unittest

from app import create_app
from db.models import Avaliacao, Usuario, db
from ml.data import FEATURE_COLUMNS
from services.usuarios import criar_usuario

SENHA = "senha-forte-1"
# Primeira linha da base WDBC (842302)
LINHA = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
         0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119,
         0.2654, 0.4601, 0.1189]


class AuthTest(unittest.TestCase):
    def setUp(self):
        # SQLite em memória: nunca toca o banco de desenvolvimento.
        self.app = create_app({"TESTING": True, "SECRET_KEY": "teste", "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
        self.ctx = self.app.app_context()
        self.ctx.push()
        self.medico = criar_usuario(nome="Ana", email="ana@exemplo.com", senha=SENHA, papel="medico")
        self.pesq = criar_usuario(nome="Bia", email="bia@exemplo.com", senha=SENHA, papel="pesquisador")
        self.c = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def login(self, email, senha=SENHA, client=None):
        return (client or self.c).post("/api/auth/login", json={"email": email, "senha": senha})

    # ---------- registro e login ----------
    def test_registro_aberto_aos_dois_papeis(self):
        for papel in ("medico", "pesquisador"):
            r = self.c.post("/api/auth/registrar", json={
                "nome": "Novo", "email": f"{papel}@novo.com", "senha": SENHA, "papel": papel})
            self.assertEqual(r.status_code, 201)
            self.assertEqual(r.json["papel"], papel)
            self.assertNotIn("senha_hash", r.json)

    def test_registro_email_duplicado_e_dados_invalidos(self):
        base = {"nome": "X", "email": "ana@exemplo.com", "senha": SENHA, "papel": "medico"}
        self.assertEqual(self.c.post("/api/auth/registrar", json=base).status_code, 409)
        self.assertEqual(self.c.post("/api/auth/registrar", json={**base, "email": "n@n.com", "senha": "curta"}).status_code, 400)
        self.assertEqual(self.c.post("/api/auth/registrar", json={**base, "email": "n@n.com", "papel": "admin"}).status_code, 400)

    def test_login_define_cookie_httponly_samesite(self):
        r = self.login("ana@exemplo.com")
        self.assertEqual(r.status_code, 200)
        cookie = r.headers["Set-Cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertEqual(self.c.get("/api/auth/me").json["email"], "ana@exemplo.com")

    def test_login_invalido_tem_mensagem_unica(self):
        a = self.login("ana@exemplo.com", "errada")
        b = self.login("naoexiste@exemplo.com")
        self.assertEqual(a.status_code, 401)
        self.assertEqual(a.json, b.json)

    def test_conta_inativa_nao_entra_e_cai_na_hora(self):
        self.login("ana@exemplo.com")
        self.medico.ativo = False
        db.session.commit()
        self.assertEqual(self.c.get("/api/auth/me").status_code, 401)
        self.assertEqual(self.login("ana@exemplo.com").status_code, 401)

    def test_logout_encerra_a_sessao(self):
        self.login("ana@exemplo.com")
        self.assertEqual(self.c.post("/api/auth/logout", json={}).status_code, 200)
        self.assertEqual(self.c.get("/api/auth/me").status_code, 401)

    # ---------- proteção das rotas ----------
    def test_sem_login_todas_as_rotas_dao_401(self):
        for rota in ("/api/auth/me", "/api/features/schema", "/api/avaliacoes", "/api/avaliacoes/1",
                     "/api/metricas", "/api/metricas/features", "/api/metricas/export", "/api/modelos"):
            self.assertEqual(self.c.get(rota).status_code, 401, rota)
        r = self.c.post("/api/biopsias/classificar", json={})
        self.assertEqual((r.status_code, r.json["codigo"]), (401, "NAO_AUTENTICADO"))

    def test_medico_nao_acessa_area_do_pesquisador(self):
        self.login("ana@exemplo.com")
        for rota in ("/api/metricas", "/api/metricas/features", "/api/metricas/export", "/api/modelos"):
            r = self.c.get(rota)
            self.assertEqual((r.status_code, r.json["codigo"]), (403, "SEM_PERMISSAO"), rota)
        self.assertEqual(self.c.get("/api/avaliacoes").status_code, 200)

    def test_pesquisador_nao_classifica(self):
        self.login("bia@exemplo.com")
        r = self.c.post("/api/biopsias/classificar", json={"caracteristicas": dict(zip(FEATURE_COLUMNS, LINHA))})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.c.get("/api/modelos").status_code, 200)
        self.assertEqual(self.c.get("/api/avaliacoes").status_code, 200)

    def test_classificacao_usa_o_medico_da_sessao(self):
        self.login("ana@exemplo.com")
        r = self.c.post("/api/biopsias/classificar", json={
            "caracteristicas": dict(zip(FEATURE_COLUMNS, LINHA)), "medico_id": self.pesq.id})
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(db.session.get(Avaliacao, r.json["avaliacao_id"]).medico_id, self.medico.id)

    # ---------- CSRF ----------
    def test_escrita_exige_json(self):
        r = self.c.post("/api/auth/login", data="email=ana@exemplo.com&senha=x",
                        content_type="application/x-www-form-urlencoded")
        self.assertEqual((r.status_code, r.json["codigo"]), (415, "CONTENT_TYPE_INVALIDO"))

    def test_usuario_inexistente_na_sessao_e_descartado(self):
        self.login("ana@exemplo.com")
        db.session.delete(self.medico)
        db.session.commit()
        self.assertEqual(self.c.get("/api/auth/me").status_code, 401)


if __name__ == "__main__":
    unittest.main()
