"""
API REST da ferramenta, sob o prefixo /api/v1.

Os identificadores (rotas, campos e códigos de erro) são em inglês; as
mensagens devolvidas ao usuário, em português. O envelope de erro é sempre
{"error", "code"}.

Toda rota de dados exige sessão válida (401 `UNAUTHENTICATED` sem ela) e
autorização por perfil (403 `FORBIDDEN` se o perfil não tem permissão).
`/auth/*` e `/health` são públicos.

Artefatos: `ml/artifacts/` é a referência versionada do artigo (RNF03),
regerada apenas por `python -m ml.train`; o treino sob demanda desta API grava
em `ml/artifacts/runs/{training_run_id}/` e nunca sobrescreve a referência (D8).
"""
import functools
import json
import os
import time

from flask import Blueprint, Response, current_app, jsonify, request

from db.models import TrainedModel, TrainingRun, db
from ml.data import FEATURE_COLUMNS, feature_schema
from ml.predict import ModeloIndisponivel, clear_cache, load_bundle, predict
from ml.report import gerar_figuras, metricas_csv
from ml.train import (
    ARTIFACTS_DIR,
    SENSIBILIDADE_MINIMA,
    diretorio_execucao,
    train_and_evaluate,
)
from services import auth as svc_auth
from services import avaliacoes as svc_avaliacoes
from services import modelos as svc_modelos
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

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

LIMITE_TEMPO_MS = 2000  # RNF02
DISCLAIMER = (
    "Esta classificação é uma ferramenta de apoio e não substitui "
    "o laudo histopatológico e a avaliação médica."
)

# Métricas vindas dos artefatos (chaves em português) → campos da API.
_METRICAS_PT_EN = {
    "acuracia": "accuracy",
    "sensibilidade": "sensitivity",
    "especificidade": "specificity",
    "auc": "auc",
    "taxa_falsos_negativos": "false_negative_rate",
    "matriz_confusao": "confusion_matrix",
}


def _erro(mensagem, status, code, **extra):
    return jsonify({"error": mensagem, "code": code, **extra}), status


def exigir_sessao(view):
    """
    Exige sessão válida na rota: sem ela, 401 `UNAUTHENTICATED`, no mesmo
    envelope de erro de `GET /auth/me`.

    O preflight CORS não é afetado: o `OPTIONS` é respondido pelo
    `provide_automatic_options` do Flask, sem entrar na view decorada.
    """
    @functools.wraps(view)
    def _view(*args, **kwargs):
        if svc_auth.current_user() is None:
            return _erro("Autenticação necessária.", 401, "UNAUTHENTICATED")
        return view(*args, **kwargs)

    return _view


def exigir_perfil(*perfis):
    """
    Decorador de autorização por perfil. Aplica-se **depois** de `exigir_sessao`.

    `perfis` é uma tupla de strings permitidas (ex.: ("medico", "pesquisador")).
    Perfil não autorizado → 403 `FORBIDDEN`. Sem sessão continua 401.

    O `profile` é lido do usuário da sessão no servidor, nunca do corpo da
    requisição. Matriz de permissões documentada no docstring do módulo.
    """
    def decorator(view):
        @functools.wraps(view)
        def _view(*args, **kwargs):
            user = svc_auth.current_user()
            if user is None:
                return _erro("Autenticação necessária.", 401, "UNAUTHENTICATED")
            if user.profile not in perfis:
                return _erro("Acesso negado para o seu perfil.", 403, "FORBIDDEN")
            return view(*args, **kwargs)
        _view.__perfis_permitidos__ = perfis
        return _view
    return decorator


def _artefato(nome):
    """Lê um artefato de ml/artifacts/. Artefato ausente conta como modelo indisponível."""
    caminho = os.path.join(ARTIFACTS_DIR, nome)
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise ModeloIndisponivel(f"Artefato '{nome}' indisponível em {ARTIFACTS_DIR}.") from e


def _metricas_en(bloco: dict) -> dict:
    return {novo: bloco[antigo] for antigo, novo in _METRICAS_PT_EN.items() if antigo in bloco}


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
# Autenticação simples (cadastro, login, sessão)
# ----------------------------------------------------------------------
@api_bp.route("/auth/register", methods=["POST"])
def register():
    name, email, password, profile = validar_cadastro(request.get_json(silent=True))
    user = svc_auth.register(name, email, password, profile)
    return jsonify({"user": user.to_dict()}), 201


@api_bp.route("/auth/login", methods=["POST"])
def login():
    email, password = validar_login(request.get_json(silent=True))
    user = svc_auth.authenticate(email, password)
    if user is None:
        # Mensagem genérica: não revela se o e-mail existe.
        return _erro("Credenciais inválidas.", 401, "UNAUTHENTICATED")
    return jsonify({"user": user.to_dict()}), 200


@api_bp.route("/auth/logout", methods=["POST"])
def logout():
    svc_auth.encerrar_sessao()
    return "", 204


@api_bp.route("/auth/me", methods=["GET"])
def me():
    user = svc_auth.current_user()
    if user is None:
        return _erro("Sessão não encontrada.", 401, "UNAUTHENTICATED")
    return jsonify({"user": user.to_dict()}), 200


# ----------------------------------------------------------------------
# US01 — formulário das 30 características
# ----------------------------------------------------------------------
@api_bp.route("/features/schema", methods=["GET"])
@exigir_perfil("medico", "pesquisador")
def schema():
    return jsonify(feature_schema())


# ----------------------------------------------------------------------
# US02 / US03 / US07 — classificação com registro automático
# ----------------------------------------------------------------------
@api_bp.route("/predictions", methods=["POST"])
@exigir_perfil("medico")
def criar_predicao():
    inicio = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ErroValidacao("O corpo da requisição deve ser um objeto JSON.")

    features = validar_caracteristicas(payload.get("features"))
    mode = payload.get("mode", "anonimo")
    identificado, patient_name, patient_record = validar_paciente(
        mode, payload.get("patient_name"), payload.get("patient_record")
    )

    resultado = predict(features)  # levanta ModeloIndisponivel → 503
    tempo_ms = (time.perf_counter() - inicio) * 1000
    if tempo_ms > LIMITE_TEMPO_MS:
        # RNF02: estouro do limite é erro e nada é gravado.
        current_app.logger.warning(
            "TIMEOUT: classificação levou %.0f ms (limite %d ms); avaliação não registrada.",
            tempo_ms, LIMITE_TEMPO_MS,
        )
        return _erro(
            "A classificação excedeu o tempo limite de 2 segundos. Tente novamente.",
            504, "TIMEOUT",
        )

    av = svc_avaliacoes.registrar(
        features=dict(zip(FEATURE_COLUMNS, features)),
        resultado=resultado,
        tempo_ms=tempo_ms,
        mode="identificado" if identificado else "anonimo",
        patient_name=patient_name,
        patient_record=patient_record,
        user_id=svc_auth.current_user_id(),
    )

    return jsonify({
        "evaluation_id": av.id,
        "label": av.label,
        "probability": av.probability,
        "confidence": av.confidence,
        "confidence_band": av.confidence_band,
        "model_name": av.model_name,
        "threshold_used": av.threshold_used,
        "elapsed_ms": av.elapsed_ms,
        # extra não contratual: maligno decidido abaixo de 50% por efeito do
        # limiar de segurança (nota exibida na interface).
        "safety_threshold_triggered": resultado["acionou_limiar_seguranca"],
        "disclaimer": DISCLAIMER,
    }), 200


# ----------------------------------------------------------------------
# US07 — histórico
# ----------------------------------------------------------------------
@api_bp.route("/predictions", methods=["GET"])
@exigir_perfil("medico", "pesquisador")
def listar_predicoes():
    user = svc_auth.current_user()
    return jsonify(svc_avaliacoes.listar(
        validar_filtros_historico(request.args), profile=user.profile if user else None
    ))


@api_bp.route("/predictions/<int:evaluation_id>", methods=["GET"])
@exigir_perfil("medico", "pesquisador")
def obter_predicao(evaluation_id):
    user = svc_auth.current_user()
    av = svc_avaliacoes.obter(evaluation_id, profile=user.profile if user else None)
    if av is None:
        return _erro("Avaliação não encontrada.", 404, "NOT_FOUND")
    return jsonify(av)


# ----------------------------------------------------------------------
# US05 / US06 — modelos, métricas, importância, reprodutibilidade
# ----------------------------------------------------------------------
@api_bp.route("/models", methods=["GET"])
@exigir_perfil("pesquisador")
def listar_modelos():
    modelos = (TrainedModel.query
               .order_by(TrainedModel.trained_at.desc(), TrainedModel.id.desc()).all())
    return jsonify({"items": [m.to_dict() for m in modelos]})


def _composicao_do_teste(matriz):
    """
    Tamanho e número de malignos do conjunto de teste daquele modelo, derivados
    da própria matriz de confusão (defeito C): `total = vn+fp+fn+vp` e
    `malignos = vp+fn`. Matriz ausente ou malformada → (None, None), para a
    tela mostrar "—" em vez de um número de outra partição.
    """
    if not isinstance(matriz, dict):
        return None, None
    try:
        vn, fp, fn, vp = (int(matriz[k]) for k in ("vn", "fp", "fn", "vp"))
    except (KeyError, TypeError, ValueError):
        return None, None
    return vn + fp + fn + vp, vp + fn


def _metrics_da_execucao(registro, cache):
    """
    `metrics.json` da execução do registro, ou None se a execução já foi
    descartada. Nunca cai na referência: os blocos que dependem da partição
    (limiar padrão, validação cruzada) não podem vir de outro split.
    """
    diretorio = svc_modelos.diretorio_artefatos_da_execucao(registro)
    if diretorio is None:
        return None
    if diretorio not in cache:
        caminho = os.path.join(diretorio, "metrics.json")
        try:
            with open(caminho, encoding="utf-8") as f:
                cache[diretorio] = json.load(f)
        except (OSError, ValueError):
            cache[diretorio] = None
    return cache[diretorio]


def _linhas_mais_recentes_por_modelo():
    """
    Uma linha de `trained_models` por nome de modelo: a mais recente. Com
    `D12` (uma linha por execução) a tabela tem várias linhas do mesmo nome, e
    "a última do dict" era uma escolha arbitrária.
    """
    recentes = {}
    consulta = (TrainedModel.query
                .order_by(TrainedModel.trained_at.desc(), TrainedModel.id.desc()).all())
    for m in consulta:
        recentes.setdefault(m.name, m)
    return list(recentes.values())


@api_bp.route("/models/comparison", methods=["GET"])
@exigir_perfil("pesquisador")
def comparar_modelos():
    bundle = load_bundle()  # levanta ModeloIndisponivel se não houver treino
    metrics, config = bundle["metrics"], _artefato("train_config.json")
    meta = metrics.get("meta_sensibilidade", SENSIBILIDADE_MINIMA)

    # Fonte autoritativa das métricas escalares: a tabela trained_models, uma
    # linha por (execução, modelo). O laço é dirigido pelo banco, não pelo
    # metrics.json da referência: os modelos exibidos podem vir de execuções
    # com partições diferentes da 80/20 versionada.
    cache_metrics = {}
    modelos, particoes = [], []
    for registro in _linhas_mais_recentes_por_modelo():
        metrics_exec = _metrics_da_execucao(registro, cache_metrics)
        r = (metrics_exec or {}).get("modelos", {}).get(registro.name) or {}
        total_teste, malignos_teste = _composicao_do_teste(registro.confusion_matrix)
        run = registro.training_run

        modelos.append({
            "id": registro.id,
            "name": registro.name,
            "training_run_id": registro.training_run_id,
            # D11/D12: procedência do modelo (ml/artifacts/ para a referência,
            # ml/artifacts/runs/{id}/ para um treino sob demanda).
            "artifact_path": registro.artifact_path,
            "is_production": registro.is_production,
            "threshold": registro.threshold,
            "safety_threshold": registro.safety_threshold,
            "hyperparameters": registro.hyperparameters,
            "random_seed": registro.random_seed,
            "trained_at": registro.to_dict()["trained_at"],
            # Composição do conjunto de teste DESTE modelo, vinda dos dados
            # (defeito C): nada de texto fixo derivado da referência 80/20.
            "test_size": run.test_size if run is not None else None,
            "test_set_size": total_teste,
            "test_set_malignant": malignos_teste,
            # Métricas reportadas: conjunto de teste no limiar vigente (D11).
            "operating_point": {
                "threshold": registro.threshold,
                "accuracy": registro.accuracy,
                "sensitivity": registro.sensitivity,
                "specificity": registro.specificity,
                "auc": registro.auc,
                "false_negative_rate": registro.false_negative_rate,
                "confusion_matrix": registro.confusion_matrix,
            },
            "meets_sensitivity_target": registro.sensitivity >= meta,
            # Portão de promoção do RNF04: out-of-fold do TREINO, nunca
            # confundido com a sensibilidade de teste acima.
            "train_oof_sensitivity": registro.train_oof_sensitivity,
            "gate_meets_sensitivity_target": (
                registro.train_oof_sensitivity is not None
                and registro.train_oof_sensitivity >= meta
            ),
            # Blocos que dependem da partição: sempre da execução deste modelo,
            # None quando a execução foi descartada.
            "default_point": (
                {"threshold": r["limiar_padrao"], **_metricas_en(r["teste_padrao"])}
                if "teste_padrao" in r else None
            ),
            "cross_validation": r.get("validacao_cruzada"),
            "safety_threshold_reached_target": r.get("limiar_atingiu_meta_no_treino"),
            "roc_points": registro.roc_points or r.get("roc"),
        })
        particoes.append({
            "training_run_id": registro.training_run_id,
            "test_size": run.test_size if run is not None else None,
            "test_set_size": total_teste,
            "test_set_malignant": malignos_teste,
        })

    return jsonify({
        "positive_class": metrics["classe_positiva"],
        "sensitivity_target": meta,
        "best_model": metrics["melhor_modelo"],
        "best_model_criterion": metrics["criterio_melhor_modelo"],
        "generated_at": metrics["gerado_em"],
        "models": modelos,
        # Defeito C: comparar métricas entre partições distintas não é válido,
        # então quando a tabela mistura execuções isso tem de ficar visível.
        "test_partitions_mixed": (
            len({p["training_run_id"] for p in particoes}) > 1
            or len({p["test_set_size"] for p in particoes}) > 1
        ),
        "test_partitions": particoes,
        # Reprodutibilidade (RNF03): chaves do artefato preservadas. Este bloco
        # descreve os artefatos de REFERÊNCIA de ml/artifacts/ (split 80/20), e
        # não necessariamente a execução dos modelos listados acima; a
        # procedência de cada um é `training_run_id` + `artifact_path` no item.
        "reproducibility": {
            k: config[k] for k in (
                "test_size", "random_state", "cv_folds", "n_treino", "n_teste",
                "distribuicao_treino", "distribuicao_teste", "hiperparametros",
                "hashes_sha256", "versoes", "commit_git",
            )
        },
    })


@api_bp.route("/models/features", methods=["GET"])
@exigir_perfil("pesquisador")
def importancia_features():
    load_bundle()
    dados = _artefato("feature_importance.json")
    modelo = request.args.get("model") or dados["melhor_modelo"]
    if modelo not in dados["modelos"]:
        return _erro("Modelo desconhecido.", 404, "NOT_FOUND")
    try:
        top = int(request.args.get("top", 30))
    except ValueError:
        raise ErroValidacao("Parâmetro 'top' deve ser inteiro.") from None
    top = max(1, min(top, 30))
    return jsonify({
        "metodo": dados["metodo"],
        "repeticoes": dados["repeticoes"],
        "model": modelo,
        "best_model": dados["melhor_modelo"],
        "ranking": dados["modelos"][modelo][:top],
    })


@api_bp.route("/models/export", methods=["GET"])
@exigir_perfil("pesquisador")
def exportar_metricas():
    if request.args.get("format", "csv") != "csv":
        raise ErroValidacao("Formato não suportado; use format=csv.")
    load_bundle()
    return Response(
        metricas_csv(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=metricas_classificadores.csv"},
    )


# ----------------------------------------------------------------------
# US04 / US05 / US06 — treino sob demanda e promoção de modelo (D6, RNF04)
# ----------------------------------------------------------------------
@api_bp.route("/training-runs", methods=["POST"])
@exigir_perfil("pesquisador")
def criar_training_run():
    models, test_size, grid_search, hyperparameters = validar_training_run(request.get_json(silent=True))

    run = svc_modelos.criar_training_run(models, test_size, grid_search)
    # D8 / RNF03: os artefatos desta execução vão para
    # ml/artifacts/runs/{run.id}/ — a referência versionada em ml/artifacts/
    # (split 80/20 citado no artigo) nunca é sobrescrita por esta rota.
    diretorio = diretorio_execucao(run.id)
    try:
        resultado = train_and_evaluate(
            output_dir=diretorio,
            models=models, test_size=test_size, grid_search=grid_search,
            hyperparameters=hyperparameters,
        )
        svc_modelos.concluir_training_run_sucesso(run, resultado)
        # Sem clear_cache(): o treino sob demanda não toca nos artefatos de
        # referência, então o bundle já carregado em memória continua válido
        # (D9 — a inferência classifica sempre pela referência).
        try:
            # Figuras/CSV 300 DPI da execução, em runs/{id}/report/, para não
            # sobrescrever o report da referência.
            gerar_figuras(artifacts_dir=diretorio)
        except Exception as e:  # exportação não derruba um treino bem-sucedido
            current_app.logger.warning("Falha ao gerar figuras/CSV do treino: %s", e)
    except Exception as e:
        # Falha durante o treino é um resultado esperado e registrado, não um
        # erro não tratado do servidor: o run já existe e é consultável.
        svc_modelos.falhar_training_run(run, str(e))

    return jsonify({
        "training_run_id": run.id,
        "status": run.status,
        # extra não contratual: onde ficaram os artefatos desta execução.
        "artifacts_dir": svc_modelos.caminho_relativo(diretorio),
    }), 201


@api_bp.route("/training-runs/<int:training_run_id>", methods=["GET"])
@exigir_perfil("pesquisador")
def obter_training_run(training_run_id):
    run = db.session.get(TrainingRun, training_run_id)
    if run is None:
        return _erro("Execução de treino não encontrada.", 404, "NOT_FOUND")
    corpo = run.to_dict()
    corpo["models"] = [m.to_dict() for m in run.models]
    return jsonify(corpo)


@api_bp.route("/models/<int:model_id>/threshold", methods=["PUT"])
@exigir_perfil("pesquisador")
def ajustar_threshold(model_id):
    threshold = validar_threshold(request.get_json(silent=True))

    registro = db.session.get(TrainedModel, model_id)
    if registro is None:
        return _erro("Modelo não encontrado.", 404, "NOT_FOUND")

    diretorio = svc_modelos.diretorio_artefatos_do_registro(registro)

    try:
        # Portão do RNF04: sempre out-of-fold do TREINO, lido do diretório do
        # próprio registro — ml/artifacts/ para os modelos de referência,
        # ml/artifacts/runs/{id}/ para execuções (D8). Validar limiar pelo
        # conjunto de teste vazaria informação, então isso não muda (D11).
        sensibilidade = svc_modelos.sensibilidade_oof(
            registro.name, threshold, artifacts_dir=diretorio,
        )
    except svc_modelos.OofIndisponivel as e:
        current_app.logger.error("MODEL_UNAVAILABLE: %s", e)
        return _erro(
            "Vetor out-of-fold indisponível para recalcular a sensibilidade.",
            503, "MODEL_UNAVAILABLE",
        )

    if sensibilidade < SENSIBILIDADE_MINIMA:
        return _erro(
            "Sensibilidade obtida abaixo da meta de segurança clínica.",
            400, "SENSITIVITY_BELOW_TARGET",
            obtained=round(sensibilidade, 4), required=SENSIBILIDADE_MINIMA,
        )

    try:
        # Métricas reportadas: todas do CONJUNTO DE TESTE da execução deste
        # modelo, recalculadas em conjunto no limiar novo (D11). Obtidas ANTES
        # de qualquer escrita — uma linha meio atualizada, misturando o limiar
        # novo com métricas do limiar antigo, é exatamente o defeito corrigido.
        metricas = svc_modelos.metricas_teste_no_limiar(
            registro.name, threshold, artifacts_dir=diretorio,
        )
    except svc_modelos.TesteProbaIndisponivel as e:
        current_app.logger.error("MODEL_UNAVAILABLE: %s", e)
        return _erro(
            "Probabilidades do conjunto de teste indisponíveis para recalcular "
            "as métricas no limiar pedido.",
            503, "MODEL_UNAVAILABLE",
        )

    svc_modelos.promover_modelo(registro, threshold, sensibilidade, metricas)
    clear_cache()
    return jsonify({
        "model": registro.name,
        "threshold": registro.threshold,
        # Portão out-of-fold do TREINO: é o que decidiu a promoção.
        "train_oof_sensitivity": round(sensibilidade, 4),
        # Métricas reportadas, todas do conjunto de teste no limiar novo.
        "test_metrics": _metricas_en(metricas),
    }), 200