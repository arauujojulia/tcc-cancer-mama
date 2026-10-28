"""
Cadastro e login (D4).

Sessão de servidor em cookie assinado (`flask.session`), sem JWT, sem controle
de acesso por perfil e sem registro de auditoria. A senha nunca é gravada em
texto: este é o único módulo que escreve `users.password_hash`, sempre com
`werkzeug.security` e com o algoritmo **fixado** em `pbkdf2:sha256`
(`METODO_HASH`), para o código corresponder ao que D4 documenta em vez de seguir
o padrão da versão instalada do Werkzeug (hoje scrypt).
"""
from flask import session
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from db.models import User, db
from services.validation import ErroValidacao

CHAVE_SESSAO = "user_id"

# D4: algoritmo explícito, não o padrão da biblioteca. `check_password_hash`
# identifica o algoritmo pelo prefixo do hash armazenado, então contas antigas
# gravadas com o padrão anterior (scrypt) continuam autenticando.
METODO_HASH = "pbkdf2:sha256"


def register(name: str, email: str, password: str, profile: str) -> User:
    """
    Cria o usuário e abre a sessão. E-mail duplicado vira erro de validação
    (checado antes do insert e, em corrida perdida, pelo IntegrityError).
    """
    if User.query.filter_by(email=email).first() is not None:
        raise ErroValidacao("E-mail já cadastrado.")

    user = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password, method=METODO_HASH),
        profile=profile,
    )
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        raise ErroValidacao("E-mail já cadastrado.") from None

    abrir_sessao(user)
    return user


def authenticate(email: str, password: str) -> User | None:
    """Usuário correspondente ou None. Não revela qual campo falhou."""
    user = User.query.filter_by(email=email).first()
    if user is None or not check_password_hash(user.password_hash, password):
        return None
    abrir_sessao(user)
    return user


def abrir_sessao(user: User) -> None:
    session.clear()
    session[CHAVE_SESSAO] = user.id


def encerrar_sessao() -> None:
    session.clear()


def current_user() -> User | None:
    """Usuário da sessão, ou None se não houver sessão válida."""
    user_id = session.get(CHAVE_SESSAO)
    if user_id is None:
        return None
    user = db.session.get(User, user_id)
    if user is None:
        # Sessão apontando para usuário removido: descarta.
        session.clear()
    return user


def current_user_id() -> int | None:
    """Id do usuário da sessão, para associar autoria à avaliação (pode ser None)."""
    user = current_user()
    return user.id if user else None
