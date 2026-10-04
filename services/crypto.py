"""
Cifragem simétrica AES-256-GCM dos dados identificados do paciente
(RNF05 / LGPD, UC07 FA01).

A chave de 32 bytes vem da variável de ambiente PACIENTE_ENCRYPTION_KEY
(base64). Sem chave, o registro identificado é recusado — nunca se grava
dado de paciente em texto puro nem com chave embutida no código.

Gerar uma chave:  python -m services.crypto
"""
import base64
import os
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ENV_KEY = "PACIENTE_ENCRYPTION_KEY"
NONCE_BYTES = 12


class CryptoConfigError(RuntimeError):
    """Chave de cifragem ausente ou inválida."""


class CryptoDecryptError(ValueError):
    """Dado cifrado corrompido ou chave incorreta."""


def gerar_chave() -> str:
    return base64.b64encode(secrets.token_bytes(32)).decode("ascii")


def _carregar_chave() -> bytes:
    bruto = os.environ.get(ENV_KEY)
    if not bruto:
        raise CryptoConfigError(
            f"Variável {ENV_KEY} não configurada; registros identificados exigem cifragem."
        )
    try:
        chave = base64.b64decode(bruto, validate=True)
    except Exception as e:  # noqa: BLE001
        raise CryptoConfigError(f"{ENV_KEY} não é base64 válido.") from e
    if len(chave) != 32:
        raise CryptoConfigError(f"{ENV_KEY} deve ter 32 bytes (AES-256); recebidos {len(chave)}.")
    return chave


def cifrar(texto: str) -> str:
    """Retorna base64(nonce || ciphertext+tag). Nonce aleatório por chamada."""
    nonce = secrets.token_bytes(NONCE_BYTES)
    ct = AESGCM(_carregar_chave()).encrypt(nonce, texto.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("ascii")


def decifrar(token: str) -> str:
    try:
        bruto = base64.b64decode(token)
        nonce, ct = bruto[:NONCE_BYTES], bruto[NONCE_BYTES:]
        return AESGCM(_carregar_chave()).decrypt(nonce, ct, None).decode("utf-8")
    except InvalidTag as e:
        raise CryptoDecryptError("Falha na autenticação do dado cifrado (chave incorreta ou dado alterado).") from e
    except (ValueError, TypeError) as e:
        raise CryptoDecryptError("Dado cifrado malformado.") from e


if __name__ == "__main__":
    print(f"{ENV_KEY}={gerar_chave()}")
