"""
Testes unitários de services/validation.py (funções puras, sem Flask/banco).
"""
import unittest
from datetime import date

from ml.data import FEATURE_COLUMNS, feature_schema
from services.validation import (
    ErroValidacao,
    validar_caracteristicas,
    validar_cadastro,
    validar_filtros_historico,
    validar_login,
    validar_paciente,
    validar_threshold,
    validar_training_run,
)


def _payload_valido():
    exemplos = feature_schema()["exemplos"]["benigno"]
    return dict(exemplos)


class _Args(dict):
    """Simula werkzeug.datastructures.MultiDict o suficiente para os testes (.get)."""


class TestValidarCaracteristicas(unittest.TestCase):
    def test_aceita_as_30_caracteristicas_validas(self):
        valores = validar_caracteristicas(_payload_valido())
        self.assertEqual(len(valores), len(FEATURE_COLUMNS))

    def test_rejeita_campo_faltando(self):
        payload = _payload_valido()
        del payload["mean_radius"]
        with self.assertRaises(ErroValidacao) as ctx:
            validar_caracteristicas(payload)
        self.assertIn("mean_radius", ctx.exception.detalhes["campos_faltando"])

    def test_rejeita_valor_nao_numerico(self):
        payload = _payload_valido()
        payload["mean_radius"] = "abc"
        with self.assertRaises(ErroValidacao) as ctx:
            validar_caracteristicas(payload)
        self.assertIn("mean_radius", ctx.exception.detalhes["campos_invalidos"])

    def test_rejeita_valor_fora_da_faixa_aceita(self):
        payload = _payload_valido()
        payload["mean_radius"] = 10_000.0
        with self.assertRaises(ErroValidacao) as ctx:
            validar_caracteristicas(payload)
        campos = [c["campo"] for c in ctx.exception.detalhes["campos_fora_da_faixa"]]
        self.assertIn("mean_radius", campos)

    def test_corpo_nao_dict_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_caracteristicas(["nao", "é", "dict"])


class TestValidarPaciente(unittest.TestCase):
    def test_modo_anonimo_descarta_dados_de_paciente(self):
        identificado, nome, prontuario = validar_paciente("anonimo", "Fulano", "123")
        self.assertFalse(identificado)
        self.assertIsNone(nome)
        self.assertIsNone(prontuario)

    def test_modo_identificado_exige_nome_e_prontuario(self):
        with self.assertRaises(ErroValidacao):
            validar_paciente("identificado", "", "123")
        with self.assertRaises(ErroValidacao):
            validar_paciente("identificado", "Fulano", "")

    def test_modo_identificado_aceita_campos_planos(self):
        identificado, nome, prontuario = validar_paciente("identificado", "Fulana de Tal", "PRT-1")
        self.assertTrue(identificado)
        self.assertEqual(nome, "Fulana de Tal")
        self.assertEqual(prontuario, "PRT-1")

    def test_modo_invalido_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_paciente("outro", "Fulano", "123")

    def test_nome_acima_do_limite_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_paciente("identificado", "x" * 201, "123")


class TestValidarFiltrosHistorico(unittest.TestCase):
    def test_defaults(self):
        filtros = validar_filtros_historico(_Args())
        self.assertEqual(filtros["page"], 1)
        self.assertEqual(filtros["limit"], 25)
        self.assertIsNone(filtros["label"])
        self.assertIsNone(filtros["mode"])

    def test_limit_acima_do_maximo_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_filtros_historico(_Args({"limit": "101"}))

    def test_label_invalido_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_filtros_historico(_Args({"label": "indefinido"}))

    def test_mode_invalido_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_filtros_historico(_Args({"mode": "x"}))

    def test_datas_em_ingles_sao_aceitas(self):
        filtros = validar_filtros_historico(_Args({"date_from": "2026-01-01", "date_to": "2026-01-31"}))
        self.assertEqual(filtros["date_from"].date(), date(2026, 1, 1))
        self.assertEqual(filtros["date_to_exclusive"].date(), date(2026, 2, 1))

    def test_date_from_posterior_a_date_to_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_filtros_historico(_Args({"date_from": "2026-02-01", "date_to": "2026-01-01"}))

    def test_confianca_invertida_rejeitada(self):
        with self.assertRaises(ErroValidacao):
            validar_filtros_historico(_Args({"confidence_min": "0.9", "confidence_max": "0.1"}))


class TestValidarCadastroELogin(unittest.TestCase):
    def test_cadastro_valido(self):
        name, email, password, profile = validar_cadastro({
            "name": "Fulano", "email": "fulano@exemplo.com",
            "password": "senha1234", "profile": "medico",
        })
        self.assertEqual(email, "fulano@exemplo.com")
        self.assertEqual(profile, "medico")

    def test_cadastro_rejeita_email_invalido(self):
        with self.assertRaises(ErroValidacao):
            validar_cadastro({"name": "F", "email": "invalido", "password": "senha1234", "profile": "medico"})

    def test_cadastro_rejeita_senha_curta(self):
        with self.assertRaises(ErroValidacao):
            validar_cadastro({"name": "F", "email": "f@x.com", "password": "123", "profile": "medico"})

    def test_cadastro_rejeita_perfil_invalido(self):
        with self.assertRaises(ErroValidacao):
            validar_cadastro({"name": "F", "email": "f@x.com", "password": "senha1234", "profile": "admin"})

    def test_login_exige_email_e_senha(self):
        with self.assertRaises(ErroValidacao):
            validar_login({"email": "f@x.com"})
        email, password = validar_login({"email": "f@x.com", "password": "senha1234"})
        self.assertEqual(email, "f@x.com")


class TestValidarTrainingRunEThreshold(unittest.TestCase):
    def test_defaults(self):
        models, test_size, grid_search, hyperparameters = validar_training_run({})
        self.assertEqual(set(models), {"regressao_logistica", "svm"})
        self.assertEqual(test_size, 0.2)
        self.assertTrue(grid_search)
        self.assertIsNone(hyperparameters)

    def test_models_vazio_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_training_run({"models": []})

    def test_models_desconhecido_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_training_run({"models": ["arvore_de_decisao"]})

    def test_test_size_fora_da_faixa_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_training_run({"test_size": 0.9})

    def test_grid_search_nao_booleano_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_training_run({"grid_search": "sim"})

    def test_threshold_valido(self):
        self.assertEqual(validar_threshold({"threshold": 0.42}), 0.42)

    def test_threshold_fora_da_faixa_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_threshold({"threshold": 1.5})

    def test_threshold_nao_numerico_rejeitado(self):
        with self.assertRaises(ErroValidacao):
            validar_threshold({"threshold": "alto"})


if __name__ == "__main__":
    unittest.main()
