"""
Criação de contas de acesso.

`criar_usuario` é usada pela linha de comando para criar usuários fora da API.
O cadastro da API usa `services.auth.register` diretamente.

Criar um usuário pela linha de comando (a senha é pedida no prompt, sem eco):
    python -m services.usuarios --nome "Fulana" --email fulana@exemplo.com --perfil medico
"""
import argparse
import getpass
import sys

from werkzeug.security import generate_password_hash

from db.models import User, db

SENHA_MIN = 8
PERFIS_PERMITIDOS = ("medico", "pesquisador")


class ErroUsuario(ValueError):
    """Dados inválidos ou e-mail já cadastrado."""


def criar_usuario(*, nome: str, email: str, senha: str, perfil: str) -> User:
    nome, email = (nome or "").strip(), (email or "").strip().lower()
    if not nome:
        raise ErroUsuario("Informe o nome.")
    if "@" not in email:
        raise ErroUsuario("E-mail inválido.")
    if len(senha or "") < SENHA_MIN:
        raise ErroUsuario(f"A senha deve ter ao menos {SENHA_MIN} caracteres.")
    if perfil not in PERFIS_PERMITIDOS:
        raise ErroUsuario(f"Perfil inválido: use {', '.join(PERFIS_PERMITIDOS)}.")
    if User.query.filter_by(email=email).first():
        raise ErroUsuario("Já existe um usuário com esse e-mail.")

    usuario = User(
        name=nome,
        email=email,
        password_hash=generate_password_hash(senha, method="pbkdf2:sha256"),
        profile=perfil,
    )
    db.session.add(usuario)
    db.session.commit()
    return usuario


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cria um usuário (médico ou pesquisador).")
    ap.add_argument("--nome", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--perfil", required=True, choices=PERFIS_PERMITIDOS)
    args = ap.parse_args(argv)

    senha = getpass.getpass("Senha: ")
    if senha != getpass.getpass("Repita a senha: "):
        print("As senhas não conferem.", file=sys.stderr)
        return 1

    from app import create_app  # import tardio: só o CLI precisa da aplicação

    with create_app().app_context():
        try:
            u = criar_usuario(nome=args.nome, email=args.email, senha=senha, perfil=args.perfil)
        except ErroUsuario as e:
            print(f"Erro: {e}", file=sys.stderr)
            return 1
        print(f"Usuário criado: #{u.id} {u.email} ({u.profile})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())