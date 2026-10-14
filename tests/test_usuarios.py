import unittest

from app import create_app
from db.models import Papel, Usuario, db
from services.avaliacoes import medico_existe
from services.usuarios import ErroUsuario, criar_usuario


class UsuarioTest(unittest.TestCase):
    def setUp(self):
        # SQLite em memória: nunca toca o banco de desenvolvimento.
        self.app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        db.session.remove()
        self.ctx.pop()

    def test_cria_com_papel_e_senha_com_hash(self):
        u = criar_usuario(nome="Ana", email="ANA@exemplo.com ", senha="senha-forte-1", papel="medico")
        self.assertEqual(u.email, "ana@exemplo.com")
        self.assertIs(u.papel, Papel.MEDICO)
        self.assertNotIn("senha-forte-1", u.senha_hash)
        self.assertTrue(u.senha_confere("senha-forte-1"))
        self.assertFalse(u.senha_confere("outra"))

    def test_papel_persistido_como_texto(self):
        criar_usuario(nome="Bia", email="bia@exemplo.com", senha="senha-forte-1", papel=Papel.PESQUISADOR)
        db.session.expire_all()
        self.assertEqual(Usuario.query.one().papel, Papel.PESQUISADOR)

    def test_to_dict_nao_expoe_hash(self):
        u = criar_usuario(nome="Ana", email="ana@exemplo.com", senha="senha-forte-1", papel="medico")
        self.assertEqual(set(u.to_dict()), {"id", "nome", "email", "papel"})

    def test_rejeita_dados_invalidos(self):
        casos = [
            dict(nome="", email="a@b.com", senha="senha-forte-1", papel="medico"),
            dict(nome="Ana", email="sem-arroba", senha="senha-forte-1", papel="medico"),
            dict(nome="Ana", email="a@b.com", senha="curta", papel="medico"),
            dict(nome="Ana", email="a@b.com", senha="senha-forte-1", papel="admin"),
        ]
        for caso in casos:
            with self.subTest(caso=caso), self.assertRaises(ErroUsuario):
                criar_usuario(**caso)
        self.assertEqual(Usuario.query.count(), 0)

    def test_medico_existe_so_para_medico_ativo(self):
        medico = criar_usuario(nome="Ana", email="ana@exemplo.com", senha="senha-forte-1", papel="medico")
        pesq = criar_usuario(nome="Bia", email="bia@exemplo.com", senha="senha-forte-1", papel="pesquisador")
        inativo = criar_usuario(nome="Caio", email="caio@exemplo.com", senha="senha-forte-1", papel="medico")
        inativo.ativo = False
        db.session.commit()
        self.assertTrue(medico_existe(medico.id))
        self.assertFalse(medico_existe(pesq.id))
        self.assertFalse(medico_existe(inativo.id))
        self.assertFalse(medico_existe(9999))

    def test_rejeita_email_duplicado(self):
        criar_usuario(nome="Ana", email="ana@exemplo.com", senha="senha-forte-1", papel="medico")
        with self.assertRaises(ErroUsuario):
            criar_usuario(nome="Outra", email="ANA@exemplo.com", senha="senha-forte-2", papel="pesquisador")


if __name__ == "__main__":
    unittest.main()
