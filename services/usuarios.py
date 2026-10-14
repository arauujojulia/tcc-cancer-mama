"""
Criação de contas de acesso.

`criar_usuario` é usada pelo cadastro da API (POST /api/auth/registrar, aberto aos
dois papéis) e pela linha de comando abaixo.

Criar um usuário pela linha de comando (a senha é pedida no prompt, sem eco):
    python -m services.usuarios --nome "Fulana" --email fulana@exemplo.com --papel medico
"""
import argparse
import getpass
import sys

from db.models import Papel, Usuario, db

SENHA_MIN = 8


class ErroUsuario(ValueError):
    """Dados inválidos ou e-mail já cadastrado."""


def criar_usuario(*, nome: str, email: str, senha: str, papel: Papel | str) -> Usuario:
    nome, email = (nome or "").strip(), (email or "").strip().lower()
    if not nome:
        raise ErroUsuario("Informe o nome.")
    if "@" not in email:
        raise ErroUsuario("E-mail inválido.")
    if len(senha or "") < SENHA_MIN:
        raise ErroUsuario(f"A senha deve ter ao menos {SENHA_MIN} caracteres.")
    try:
        papel = Papel(papel)
    except ValueError:
        raise ErroUsuario(f"Papel inválido: use {', '.join(p.value for p in Papel)}.") from None
    if Usuario.query.filter_by(email=email).first():
        raise ErroUsuario("Já existe um usuário com esse e-mail.")

    usuario = Usuario(nome=nome, email=email, papel=papel)
    usuario.definir_senha(senha)
    db.session.add(usuario)
    db.session.commit()
    return usuario


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cria um usuário (médico ou pesquisador).")
    ap.add_argument("--nome", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--papel", required=True, choices=[p.value for p in Papel])
    args = ap.parse_args(argv)

    senha = getpass.getpass("Senha: ")
    if senha != getpass.getpass("Repita a senha: "):
        print("As senhas não conferem.", file=sys.stderr)
        return 1

    from app import create_app  # import tardio: só o CLI precisa da aplicação

    with create_app().app_context():
        try:
            u = criar_usuario(nome=args.nome, email=args.email, senha=senha, papel=args.papel)
        except ErroUsuario as e:
            print(f"Erro: {e}", file=sys.stderr)
            return 1
        print(f"Usuário criado: #{u.id} {u.email} ({u.papel.value})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
