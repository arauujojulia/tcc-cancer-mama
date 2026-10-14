import json
import os
import time

from flask import Blueprint, Response, current_app, g, jsonify, request

from ml.data import FEATURE_COLUMNS, feature_schema
from ml.predict import ModeloIndisponivel, load_bundle, predict
from ml.report import metricas_csv
from ml.train import ARTIFACTS_DIR, SENSIBILIDADE_MINIMA
from db.models import Papel
from services import auth as svc_auth
from services import avaliacoes as svc_avaliacoes
from services.auth import exigir_login
from services.usuarios import ErroUsuario, criar_usuario
from services.validation import (
    ErroValidacao,
    validar_caracteristicas,
    validar_filtros_historico,
    validar_paciente,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")

LIMITE_TEMPO_MS = 2000  # RNF02
AVISO = (
    "Esta classificação é uma ferramenta de apoio e não substitui "
    "o laudo histopatológico e a avaliação médica."
)


def _erro(mensagem, status, codigo, **extra):
    return jsonify({"erro": mensagem, "codigo": codigo, **extra}), status


def _artefato(nome):
    with open(os.path.join(ARTIFACTS_DIR, nome), encoding="utf-8") as f:
        return json.load(f)


@api_bp.before_request
def _exigir_json_nas_escritas():
    # Defesa de CSRF, somada ao cookie SameSite=Lax: um formulário de outro site não
    # consegue enviar Content-Type application/json sem passar por CORS.
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and not request.is_json:
        return _erro("Use Content-Type: application/json.", 415, "CONTENT_TYPE_INVALIDO")


@api_bp.errorhandler(ErroValidacao)
def _tratar_validacao(e):
    return _erro(e.mensagem, 400, e.codigo, **e.detalhes)


@api_bp.errorhandler(ModeloIndisponivel)
def _tratar_modelo_indisponivel(e):
    # FA01 do UC02: modelo indisponível → orienta contato com o suporte.
    current_app.logger.error("MODEL_UNAVAILABLE: %s", e)
    return _erro(
        "Modelo classificador indisponível. Contate o suporte técnico.", 503, "MODEL_UNAVAILABLE"
    )


# ----------------------------------------------------------------------
# Autenticação (cookie de sessão)
# ----------------------------------------------------------------------
def _campo_texto(payload, nome):
    valor = payload.get(nome)
    return valor if isinstance(valor, str) else ""


@api_bp.route("/auth/registrar", methods=["POST"])
def registrar_usuario():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")
    try:
        usuario = criar_usuario(
            nome=_campo_texto(payload, "nome"),
            email=_campo_texto(payload, "email"),
            senha=_campo_texto(payload, "senha"),
            papel=_campo_texto(payload, "papel"),
        )
    except ErroUsuario as e:
        status, codigo = (409, "EMAIL_JA_CADASTRADO") if "Já existe" in str(e) else (400, "DADOS_INVALIDOS")
        return _erro(str(e), status, codigo)
    return jsonify(usuario.to_dict()), 201


@api_bp.route("/auth/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")
    usuario = svc_auth.autenticar(_campo_texto(payload, "email"), _campo_texto(payload, "senha"))
    if usuario is None:
        # mensagem única: não revela se o e-mail existe ou se a conta está inativa
        return _erro("E-mail ou senha inválidos.", 401, "CREDENCIAIS_INVALIDAS")
    svc_auth.iniciar_sessao(usuario)
    return jsonify(usuario.to_dict())


@api_bp.route("/auth/logout", methods=["POST"])
def logout():
    svc_auth.encerrar_sessao()
    return jsonify({"ok": True})


@api_bp.route("/auth/me", methods=["GET"])
@exigir_login()
def eu():
    return jsonify(g.usuario.to_dict())


# ----------------------------------------------------------------------
# US01 — formulário
# ----------------------------------------------------------------------
@api_bp.route("/features/schema", methods=["GET"])
@exigir_login()
def schema():
    return jsonify(feature_schema())


# ----------------------------------------------------------------------
# US02 / US03 / US07 — classificação com registro automático
# ----------------------------------------------------------------------
@api_bp.route("/biopsias/classificar", methods=["POST"])
@exigir_login(Papel.MEDICO)
def classificar():
    inicio = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")

    # Características no objeto 'caracteristicas' (preferido) ou na raiz (Sprint 1).
    entrada = payload["caracteristicas"] if "caracteristicas" in payload else payload
    features = validar_caracteristicas(entrada)
    identificado, nome, prontuario = validar_paciente(
        payload.get("modo", "anonimo"), payload.get("paciente")
    )

    resultado = predict(features)
    tempo_ms = (time.perf_counter() - inicio) * 1000
    excedeu = tempo_ms > LIMITE_TEMPO_MS
    if excedeu:
        current_app.logger.warning("TIMEOUT: classificação levou %.0f ms (limite %d ms)", tempo_ms, LIMITE_TEMPO_MS)

    av = svc_avaliacoes.registrar(
        caracteristicas=dict(zip(FEATURE_COLUMNS, features)),
        resultado=resultado,
        tempo_ms=tempo_ms,
        identificado=identificado,
        nome=nome,
        prontuario=prontuario,
        medico_id=g.usuario.id,  # sempre o médico da sessão, nunca o do corpo
    )

    return jsonify({
        **resultado,
        "avaliacao_id": av.id,
        "identificado": identificado,
        "tempo_ms": round(tempo_ms, 2),
        "excedeu_limite_tempo": excedeu,
        "aviso": AVISO,
    }), 200


# ----------------------------------------------------------------------
# US07 — histórico
# ----------------------------------------------------------------------
@api_bp.route("/avaliacoes", methods=["GET"])
@exigir_login()
def listar_avaliacoes():
    return jsonify(svc_avaliacoes.listar(validar_filtros_historico(request.args)))


@api_bp.route("/avaliacoes/<int:avaliacao_id>", methods=["GET"])
@exigir_login(Papel.MEDICO)
def obter_avaliacao(avaliacao_id):
    av = svc_avaliacoes.obter(avaliacao_id)
    if av is None:
        return _erro("Avaliação não encontrada.", 404, "AVALIACAO_NAO_ENCONTRADA")
    return jsonify(av)


# ----------------------------------------------------------------------
# US05 / US06 — métricas, importância, reprodutibilidade
# ----------------------------------------------------------------------
@api_bp.route("/metricas", methods=["GET"])
@exigir_login(Papel.PESQUISADOR)
def metricas():
    bundle = load_bundle()  # levanta ModeloIndisponivel se não houver treino
    metrics, config = bundle["metrics"], _artefato("train_config.json")
    return jsonify({
        "classe_positiva": metrics["classe_positiva"],
        "meta_sensibilidade": metrics.get("meta_sensibilidade", SENSIBILIDADE_MINIMA),
        "melhor_modelo": metrics["melhor_modelo"],
        "criterio_melhor_modelo": metrics["criterio_melhor_modelo"],
        "gerado_em": metrics["gerado_em"],
        "modelos": metrics["modelos"],
        "reprodutibilidade": {
            k: config[k] for k in (
                "test_size", "random_state", "cv_folds", "n_treino", "n_teste",
                "distribuicao_treino", "distribuicao_teste", "hiperparametros",
                "hashes_sha256", "versoes", "commit_git",
            )
        },
    })


@api_bp.route("/metricas/features", methods=["GET"])
@exigir_login(Papel.PESQUISADOR)
def importancia_features():
    load_bundle()
    dados = _artefato("feature_importance.json")
    modelo = request.args.get("modelo") or dados["melhor_modelo"]
    if modelo not in dados["modelos"]:
        return _erro("Modelo desconhecido.", 404, "MODELO_DESCONHECIDO")
    try:
        top = int(request.args.get("top", 30))
    except ValueError:
        raise ErroValidacao("Parâmetro 'top' deve ser inteiro.") from None
    top = max(1, min(top, 30))
    return jsonify({
        "metodo": dados["metodo"],
        "repeticoes": dados["repeticoes"],
        "modelo": modelo,
        "melhor_modelo": dados["melhor_modelo"],
        "ranking": dados["modelos"][modelo][:top],
    })


@api_bp.route("/metricas/export", methods=["GET"])
@exigir_login(Papel.PESQUISADOR)
def exportar_metricas():
    if request.args.get("format", "csv") != "csv":
        raise ErroValidacao("Formato não suportado; use format=csv.")
    load_bundle()
    return Response(
        metricas_csv(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=metricas_classificadores.csv"},
    )


@api_bp.route("/modelos", methods=["GET"])
@exigir_login(Papel.PESQUISADOR)
def listar_modelos():
    from db.models import ModeloTreinado

    modelos = ModeloTreinado.query.order_by(ModeloTreinado.registrado_em.desc(), ModeloTreinado.id.desc()).all()
    return jsonify({"itens": [m.to_dict() for m in modelos]})
