"""
Criptografia AES-256-GCM de dados sensíveis do paciente (Tarefa B2).

Este módulo contém toda a lógica de criptografia: constantes, funções
`encrypt`/`decrypt`, o `TypeDecorator` `EncryptedText` e o script de migração.
`db/models.py` importa `EncryptedText` daqui, evitando duplicação.

Chave: variável de ambiente `PATIENT_DATA_KEY` = 32 bytes em base64.
Gere com:
    python -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"

Uso da migração (dados existentes em texto puro):
    python -m services.crypto --migrar
"""
import base64
import os
import sys
from typing import Any

from sqlalchemy import TypeDecorator, text
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ENCRYPT_PREFIX = "enc:v1:"
_AD_EVAL_NAME = b"evaluations.patient_name"
_AD_EVAL_RECORD = b"evaluations.patient_record"


def _get_key() -> bytes | None:
    """Lê e valida a chave de criptografia do ambiente."""
    raw = os.environ.get("PATIENT_DATA_KEY")
    if raw is None:
        return None
    try:
        key = base64.b64decode(raw)
    except Exception:
        return None
    return key if len(key) == 32 else None


def encrypt(plaintext: str, associated_data: bytes) -> str:
    """Cifra `plaintext` com AES-256-GCM e devolve a string armazenada."""
    key = _get_key()
    if key is None:
        raise RuntimeError("PATIENT_DATA_KEY não configurada ou inválida")
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), associated_data)
    return f"{_ENCRYPT_PREFIX}{base64.b64encode(nonce + ct).decode('ascii')}"


def decrypt(ciphertext: str, associated_data: bytes) -> str:
    """Decifra uma string armazenada e devolve o texto em claro."""
    if not ciphertext or not ciphertext.startswith(_ENCRYPT_PREFIX):
        return ciphertext
    key = _get_key()
    if key is None:
        raise RuntimeError("PATIENT_DATA_KEY não configurada ou inválida")
    raw = base64.b64decode(ciphertext[len(_ENCRYPT_PREFIX):])
    nonce, ct = raw[:12], raw[12:]
    return AESGCM(key).decrypt(nonce, ct, associated_data).decode("utf-8")


class EncryptedText(TypeDecorator):
    """TypeDecorator: cifra/descifra transparentemente com AES-256-GCM."""

    impl = None  # set via load_dialect_impl
    cache_ok = True

    def __init__(self, associated_data: bytes, **kw: Any):
        super().__init__(**kw)
        self.ad = associated_data

    def load_dialect_impl(self, dialect):
        from sqlalchemy import Text
        return Text()

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return encrypt(value, self.ad)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return decrypt(value, self.ad)


def migrar_dados_existentes():
    """
    Cifra in-place as linhas de `evaluations` cujo `patient_name` ou
    `patient_record` ainda estão em texto puro.

    Usa SQLAlchemy Core (`UPDATE`) para contornar os listeners de imutabilidade.
    Deve ser chamado dentro do app_context do Flask.
    """
    # Import tardio para evitar ciclo: db.models → services.crypto → db.models
    from db.models import db as _db

    key = _get_key()
    if key is None:
        print("ERRO: PATIENT_DATA_KEY não configurada. Migração abortada.")
        sys.exit(1)

    with _db.engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, patient_name, patient_record FROM evaluations"
        )).fetchall()

    count = 0
    for row in rows:
        eid, name, record = row
        needs_update = False
        new_name = name
        new_record = record

        if name is not None and not name.startswith(_ENCRYPT_PREFIX):
            new_name = encrypt(name, _AD_EVAL_NAME)
            needs_update = True

        if record is not None and not record.startswith(_ENCRYPT_PREFIX):
            new_record = encrypt(record, _AD_EVAL_RECORD)
            needs_update = True

        if needs_update:
            _db.session.execute(
                text(
                    "UPDATE evaluations SET patient_name = :name, patient_record = :rec WHERE id = :id"
                ),
                {"name": new_name, "rec": new_record, "id": eid},
            )
            _db.session.commit()
            count += 1

    print(f"Migração concluída: {count} linha(s) cifrada(s).")


if __name__ == "__main__":
    if "--migrar" in sys.argv:
        from app import create_app
        app = create_app()
        with app.app_context():
            migrar_dados_existentes()
    else:
        print("Uso: python -m services.crypto --migrar")
        print('Gera chave: python -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"')
