"""
Testes da criptografia AES-256-GCM de dados do paciente (Tarefa B2).

Cobre:
- (a) Ida e volta (cifra → decifra) igual ao original, com acentos.
- (b) Duas cifragens do mesmo texto geram saídas diferentes (nonce aleatório).
- (c) Adulteração do texto cifrado ou chave errada → falha (não devolve lixo).
- (d) PATIENT_DATA_KEY ausente/inválida → erro explícito.
- (e) Migração cifra linhas antigas em texto puro e é idempotente.
- (f) Associated data impede troca de campos entre patient_name e patient_record.
"""
import base64
import os
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from services.crypto import (
    ad_linha, encrypt, decrypt, _get_key, migrar_dados_existentes,
    _ENCRYPT_PREFIX, _AD_EVAL_NAME, _AD_EVAL_RECORD,
)
from db.models import db
from tests.base import ApiTestCase


class TestEncryptDecrypt(unittest.TestCase):
    """Testes unitários das funções encrypt/decrypt de services/crypto.py."""

    def test_ida_e_volta_texto_simples(self):
        ct = encrypt("Fulana de Tal", _AD_EVAL_NAME)
        self.assertTrue(ct.startswith(_ENCRYPT_PREFIX))
        pt = decrypt(ct, _AD_EVAL_NAME)
        self.assertEqual(pt, "Fulana de Tal")

    def test_ida_e_volta_com_acentos(self):
        texto = "João da Silva Ñãõ Êê Íí Ôô Üü"
        ct = encrypt(texto, _AD_EVAL_NAME)
        pt = decrypt(ct, _AD_EVAL_NAME)
        self.assertEqual(pt, texto)

    def test_ida_e_volta_patient_record(self):
        ct = encrypt("PRT-0042", _AD_EVAL_RECORD)
        pt = decrypt(ct, _AD_EVAL_RECORD)
        self.assertEqual(pt, "PRT-0042")

    def test_dois_nonces_diferentes(self):
        """Duas cifragens do mesmo texto geram saídas diferentes (nonce aleatório)."""
        ct1 = encrypt("mesmo texto", _AD_EVAL_NAME)
        ct2 = encrypt("mesmo texto", _AD_EVAL_NAME)
        self.assertNotEqual(ct1, ct2)
        # Ambas decifram para o mesmo valor.
        self.assertEqual(decrypt(ct1, _AD_EVAL_NAME), "mesmo texto")
        self.assertEqual(decrypt(ct2, _AD_EVAL_NAME), "mesmo texto")

    def test_adulteracao_falha(self):
        """Adulterar o texto cifrado faz decrypt levantar exceção."""
        ct = encrypt("segredo", _AD_EVAL_NAME)
        # Corrompe alguns bytes do payload.
        raw = base64.b64decode(ct[len(_ENCRYPT_PREFIX):])
        corrupted = raw[:10] + b"\x00\x00\x00" + raw[13:]
        bad_ct = f"{_ENCRYPT_PREFIX}{base64.b64encode(corrupted).decode('ascii')}"
        with self.assertRaises(Exception):
            decrypt(bad_ct, _AD_EVAL_NAME)

    def test_chave_errada_falha(self):
        """Usar chave diferente da de cifragem faz decrypt levantar exceção."""
        ct = encrypt("segredo", _AD_EVAL_NAME)
        wrong_key = base64.b64encode(b"\xff" * 32).decode("ascii")
        with patch.dict(os.environ, {"PATIENT_DATA_KEY": wrong_key}):
            with self.assertRaises(Exception):
                decrypt(ct, _AD_EVAL_NAME)

    def test_chave_ausente_falha_ao_cifrar(self):
        """PATIENT_DATA_KEY ausente → encrypt levanta RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                encrypt("teste", _AD_EVAL_NAME)

    def test_chave_invalida_tamanho_errado(self):
        """PATIENT_DATA_KEY com tamanho ≠ 32 bytes → encrypt levanta RuntimeError."""
        bad_key = base64.b64encode(b"\xab" * 16).decode("ascii")  # 16 bytes
        with patch.dict(os.environ, {"PATIENT_DATA_KEY": bad_key}):
            with self.assertRaises(RuntimeError):
                encrypt("teste", _AD_EVAL_NAME)

    def test_associated_data_impede_troca_de_campos(self):
        """Cifrar com AD de patient_name e decifrar com AD de patient_record falha."""
        ct = encrypt("Fulana", _AD_EVAL_NAME)
        with self.assertRaises(Exception):
            decrypt(ct, _AD_EVAL_RECORD)

    def test_associated_data_impede_troca_entre_linhas(self):
        ct = encrypt("Fulana", ad_linha(_AD_EVAL_NAME, 1))
        with self.assertRaises(Exception):
            decrypt(ct, ad_linha(_AD_EVAL_NAME, 2))

    def test_decrypt_recusa_valor_sem_prefixo(self):
        """Texto puro no banco não pode passar pela autenticação do GCM."""
        for valor in ("texto puro", ""):
            with self.assertRaises(ValueError):
                decrypt(valor, _AD_EVAL_NAME)


class TestMigracao(ApiTestCase):
    """Testes do script de migração in-place de texto puro para cifrado."""

    def setUp(self):
        super().setUp()
        # Cria uma avaliação com texto puro simulado (gravação direta na tabela).
        from sqlalchemy import text
        from db.models import Evaluation
        # Injeta diretamente no banco, simulando dados antigos.
        db.session.execute(text(
            "INSERT INTO evaluations "
            "(user_id, features, label, probability, confidence, confidence_band, "
             "model_name, threshold_used, mode, patient_name, patient_record, "
             "elapsed_ms, created_at) "
            "VALUES (NULL, '{}', 'benigno', 0.1, 0.9, 'alta', "
             "'modelo_teste', 0.5, 'identificado', 'Nome Antigo', 'REC-001', "
             "100, CURRENT_TIMESTAMP)"
        ))
        db.session.commit()

    def test_migracao_cifra_linhas_texto_puro(self):
        """Após migração, a coluna crua contém o prefixo de versão cifrado."""
        from sqlalchemy import text
        migrar_dados_existentes()
        row = db.session.execute(text(
            "SELECT patient_name, patient_record FROM evaluations WHERE id = 1"
        )).fetchone()
        self.assertTrue(row[0].startswith(_ENCRYPT_PREFIX))
        self.assertTrue(row[1].startswith(_ENCRYPT_PREFIX))
        self.assertNotIn("Nome Antigo", row[0])
        self.assertNotIn("REC-001", row[1])
        self.assertEqual(decrypt(row[0], ad_linha(_AD_EVAL_NAME, 1)), "Nome Antigo")

    def test_migracao_converte_formato_v1(self):
        from sqlalchemy import text
        v1 = "enc:v1:" + encrypt("Nome V1", _AD_EVAL_NAME)[len(_ENCRYPT_PREFIX):]
        db.session.execute(text("UPDATE evaluations SET patient_name = :v WHERE id = 1"), {"v": v1})
        db.session.commit()
        migrar_dados_existentes()
        cifrado = db.session.execute(text(
            "SELECT patient_name FROM evaluations WHERE id = 1"
        )).scalar()
        self.assertEqual(decrypt(cifrado, ad_linha(_AD_EVAL_NAME, 1)), "Nome V1")

    def test_migracao_idempotente(self):
        """Rodar migração duas vezes não quebra nem duplica cifragem."""
        from sqlalchemy import text
        migrar_dados_existentes()
        # Segunda execução não deve alterar nada.
        migrar_dados_existentes()
        row = db.session.execute(text(
            "SELECT patient_name, patient_record FROM evaluations WHERE id = 1"
        )).fetchone()
        self.assertTrue(row[0].startswith(_ENCRYPT_PREFIX))
        self.assertTrue(row[1].startswith(_ENCRYPT_PREFIX))

    def test_migracao_nao_cifra_none(self):
        """Linhas com patient_name NULL permanecem NULL após migração."""
        from sqlalchemy import text
        db.session.execute(text(
            "INSERT INTO evaluations "
            "(user_id, features, label, probability, confidence, confidence_band, "
             "model_name, threshold_used, mode, patient_name, patient_record, "
             "elapsed_ms, created_at) "
            "VALUES (NULL, '{}', 'benigno', 0.1, 0.9, 'alta', "
             "'modelo_teste', 0.5, 'anonimo', NULL, NULL, "
             "100, CURRENT_TIMESTAMP)"
        ))
        db.session.commit()
        migrar_dados_existentes()
        row = db.session.execute(text(
            "SELECT patient_name, patient_record FROM evaluations WHERE id = 2"
        )).fetchone()
        self.assertIsNone(row[0])
        self.assertIsNone(row[1])


if __name__ == "__main__":
    unittest.main()
