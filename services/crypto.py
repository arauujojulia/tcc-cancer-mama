"""
Criptografia AES-256-GCM de dados sensíveis do paciente (Tarefa B2).

Este módulo contém toda a lógica de criptografia: constantes, funções
`encrypt`/`decrypt` e o script de migração. O dado associado amarra cada valor à
coluna e à linha (`ad_linha`): cifrado copiado de outro lugar não decifra.

Chave: variável de ambiente `PATIENT_DATA_KEY` = 32 bytes em base64.
Gere com:
    python -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"

Uso da migração (dados existentes em texto puro):
    python -m services.crypto --migrar
"""
import base64
import os
import sys

from sqlalchemy import text
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from db.models import db as _db

_ENCRYPT_PREFIX = "enc:v2:"
_PREFIXO_V1 = "enc:v1:"  # dado associado só por coluna; a migração converte para v2
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


def chave_disponivel() -> bool:
    """Indica se `PATIENT_DATA_KEY` está configurada e é válida."""
    return _get_key() is not None


def ad_linha(campo: bytes, evaluation_id: int) -> bytes:
    """Dado associado de um valor: identifica a coluna e a linha."""
    return b"%s:%d" % (campo, evaluation_id)


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
    if not ciphertext.startswith(_ENCRYPT_PREFIX):
        raise ValueError("Valor não está cifrado.")
    key = _get_key()
    if key is None:
        raise RuntimeError("PATIENT_DATA_KEY não configurada ou inválida")
    raw = base64.b64decode(ciphertext[len(_ENCRYPT_PREFIX):])
    nonce, ct = raw[:12], raw[12:]
    return AESGCM(key).decrypt(nonce, ct, associated_data).decode("utf-8")


def _recifrar(valor: str | None, campo: bytes, evaluation_id: int) -> str | None:
    """Valor em texto puro ou em `enc:v1:` no formato atual; o resto fica igual."""
    if valor is None or valor.startswith(_ENCRYPT_PREFIX):
        return valor
    if valor.startswith(_PREFIXO_V1):
        valor = decrypt(_ENCRYPT_PREFIX + valor[len(_PREFIXO_V1):], campo)
    return encrypt(valor, ad_linha(campo, evaluation_id))


def migrar_dados_existentes():
    """
    Cifra in-place as linhas de `evaluations` cujo `patient_name` ou
    `patient_record` ainda estão em texto puro ou no formato `enc:v1:`.

    Usa SQLAlchemy Core (`UPDATE`) para contornar os listeners de imutabilidade.
    Deve ser chamado dentro do app_context do Flask.
    """
    key = _get_key()
    if key is None:
        print("ERRO: PATIENT_DATA_KEY não configurada. Migração abortada.")
        sys.exit(1)

    with _db.engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, patient_name, patient_record FROM evaluations"
        )).fetchall()

    count = 0
    for eid, name, record in rows:
        new_name = _recifrar(name, _AD_EVAL_NAME, eid)
        new_record = _recifrar(record, _AD_EVAL_RECORD, eid)

        if (new_name, new_record) != (name, record):
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
