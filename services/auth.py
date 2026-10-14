"""
Autenticação por cookie de sessão (assinado pelo Flask, HttpOnly).

O cookie guarda só o id do usuário. A cada requisição o usuário é relido do banco,
então desativar uma conta (ativo=false) vale na hora. O papel (RBAC) também vem
do banco, nunca do cliente.
"""
from functools import wraps

from flask import g, jsonify, session

from db.models import Papel, Usuario, db

CHAVE_SESSAO = "usuario_id"

# Hash de uma senha qualquer: usado para gastar o mesmo tempo quando o e-mail não
# existe, para o login não revelar se a conta existe.
_HASH_FALSO = Usuario(nome="", email="", papel=Papel.MEDICO)
_HASH_FALSO.definir_senha("senha-falsa-apenas-para-igualar-o-tempo")


def autenticar(email: str, senha: str) -> Usuario | None:
    """Devolve o usuário se e-mail e senha conferem e a conta está ativa."""
    usuario = Usuario.query.filter_by(email=(email or "").strip().lower()).first()
    if usuario is None:
        _HASH_FALSO.senha_confere(senha or "")
        return None
    if not usuario.senha_confere(senha or "") or not usuario.ativo:
        return None
    return usuario


def iniciar_sessao(usuario: Usuario):
    session.clear()
    session[CHAVE_SESSAO] = usuario.id
    session.permanent = True  # dura PERMANENT_SESSION_LIFETIME (8h)


def encerrar_sessao():
    session.clear()


def usuario_atual() -> Usuario | None:
    usuario_id = session.get(CHAVE_SESSAO)
    if usuario_id is None:
        return None
    usuario = db.session.get(Usuario, usuario_id)
    if usuario is None or not usuario.ativo:
        session.clear()
        return None
    return usuario


def _erro(mensagem, status, codigo):
    return jsonify({"erro": mensagem, "codigo": codigo}), status


def exigir_login(*papeis: Papel):
    """Protege a rota. Sem papéis: qualquer usuário autenticado; com papéis: só eles."""

    def decorador(f):
        @wraps(f)
        def interno(*args, **kwargs):
            usuario = usuario_atual()
            if usuario is None:
                return _erro("Autenticação necessária.", 401, "NAO_AUTENTICADO")
            if papeis and usuario.papel not in papeis:
                return _erro("Seu perfil não tem acesso a este recurso.", 403, "SEM_PERMISSAO")
            g.usuario = usuario
            return f(*args, **kwargs)

        return interno

    return decorador
