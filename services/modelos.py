"""
Registro dos modelos treinados no banco (US06 — reprodutibilidade).

Lê os artefatos de ml/artifacts/ e grava um `trained_models` por classificador
e por execução: a identidade do registro é o par `(training_run_id, name)`
(D12), não o hash do artefato. Dois treinos deterministas com a mesma semente e
os mesmos dados produzem legitimamente os mesmos bytes, então o hash continua
gravado como procedência (RNF03), mas não identifica o registro — e
`artifact_path` aponta **sempre** para o artefato da própria execução. Os
modelos pré-treinados versionados no repositório são representados por um
`training_runs` sintético de status 'success' (origem "treino inicial
versionado"), para que a classificação funcione logo após clonar o projeto. O
melhor modelo do treino mais recente fica marcado como `is_production`.

Semântica das métricas (D11): `accuracy`, `sensitivity`, `specificity`,
`false_negative_rate` e `confusion_matrix` são sempre do CONJUNTO DE TESTE da
execução, no limiar vigente da linha, calculadas por uma única função
(`metricas_teste_no_limiar`, que reusa `ml.train.compute_metrics`). O portão de
promoção do RNF04 é `train_oof_sensitivity`, das predições out-of-fold do
TREINO, e nunca é gravado em `sensitivity`.

Artefatos de treino sob demanda vêm de `ml/artifacts/runs/{training_run_id}/`
(D8): `ml/artifacts/` guarda só a referência versionada do artigo, regerada
por `python -m ml.train`. É por isso que `concluir_training_run_sucesso`
resolve o diretório a partir do próprio resultado do treino e que
`diretorio_artefatos_do_registro` reconstrói esse diretório na promoção.

Executar manualmente:  python -m services.modelos
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db.models import TrainedModel, TrainingRun, db
from ml.data import load_data
from ml.train import ARTIFACTS_DIR, CV_FOLDS, RANDOM_STATE, compute_metrics

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ler(artifacts_dir, nome):
    with open(os.path.join(artifacts_dir, nome), encoding="utf-8") as f:
        return json.load(f)


def _ler_opcional(artifacts_dir, nome):
    caminho = os.path.join(artifacts_dir, nome)
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def _data(valor):
    return datetime.fromisoformat(valor) if valor else None


def caminho_relativo(caminho) -> str:
    """
    Caminho relativo à raiz do projeto, com barras normais
    (ex.: ml/artifacts/runs/7). Caminho em outro volume é devolvido absoluto.
    """
    try:
        return os.path.relpath(caminho, _RAIZ).replace(os.sep, "/")
    except ValueError:  # volume diferente no Windows
        return caminho


def _caminho_artefato(artifacts_dir, nome):
    """Caminho do .joblib relativo à raiz do projeto (ex.: ml/artifacts/svm.joblib)."""
    return caminho_relativo(os.path.join(artifacts_dir, f"{nome}.joblib"))


def diretorio_artefatos_da_execucao(registro: TrainedModel) -> str | None:
    """
    Diretório real dos artefatos deste registro, **sem** degradar para
    ARTIFACTS_DIR: devolve None quando a execução já foi descartada.

    Usado onde cair na referência seria errado — `GET /models/comparison` não
    pode mostrar o limiar padrão ou a validação cruzada da referência 80/20 ao
    lado de métricas de uma execução 70/30, porque são partições diferentes.
    """
    caminho = registro.artifact_path or ""
    absoluto = caminho if os.path.isabs(caminho) else os.path.join(_RAIZ, caminho)
    diretorio = os.path.dirname(absoluto)
    return diretorio if os.path.isdir(diretorio) else None


def diretorio_artefatos_do_registro(registro: TrainedModel) -> str:
    """
    Diretório onde estão os artefatos de um `trained_models`: resolve
    `artifact_path` contra a raiz do projeto quando relativo e devolve o
    dirname (ml/artifacts/ para a referência, ml/artifacts/runs/{id}/ para
    execuções sob demanda — D8).

    Diretório inexistente (execução já apagada, por exemplo) degrada para
    ARTIFACTS_DIR; quem depende do OOF trata a ausência devolvendo 503.
    """
    return diretorio_artefatos_da_execucao(registro) or ARTIFACTS_DIR


def _run_sintetico(metrics, config, split):
    """TrainingRun que representa o treino inicial já versionado em ml/artifacts/."""
    gerado_em = _data(metrics.get("gerado_em"))
    run = TrainingRun(
        models_requested=list(metrics["modelos"].keys()),
        test_size=config["test_size"],
        random_seed=config["random_state"],
        grid_search=False,  # os artefatos versionados não usaram GridSearchCV
        dataset_sha256=config.get("hashes_sha256", {}).get("dataset"),
        split=split,
        library_versions=config.get("versoes"),
        git_commit=config.get("commit_git"),
        status="success",
        error_message=None,
        started_at=gerado_em,
        finished_at=gerado_em,
    )
    db.session.add(run)
    db.session.flush()
    return run


def _caminho_oof(artifacts_dir, nome):
    return os.path.join(artifacts_dir, f"{nome}_oof.json")


def regenerar_oof_ausentes(metrics: dict, artifacts_dir=ARTIFACTS_DIR) -> None:
    """
    Regenera deterministicamente `{model}_oof.json` para cada modelo
    pré-semeado em `artifacts_dir` cujo arquivo ainda não exista (design.md
    §2.4, §6). Necessário porque os modelos versionados no repositório
    nasceram antes de o treino sob demanda passar a gravar o vetor
    out-of-fold, e `PUT /models/{id}/threshold` depende dele para recalcular
    a sensibilidade sem re-treinar.

    Usa split.json (chave 'treino', em português — não renomear), clona o
    estimador carregado do .joblib (traz os mesmos get_params do treino) e
    reproduz exatamente a mesma validação cruzada do treino
    (StratifiedKFold(5, shuffle=True, random_state=42)). Uma vez gravado, o
    arquivo é reaproveitado nas subidas seguintes — não recomputa de novo.
    """
    split = _ler_opcional(artifacts_dir, "split.json")
    if split is None:
        return  # sem split.json não há como reproduzir o treino; nada a fazer

    pendentes = [nome for nome in metrics["modelos"] if not os.path.exists(_caminho_oof(artifacts_dir, nome))]
    if not pendentes:
        return

    X, y, _ = load_data()
    idx_treino = np.array(split["treino"])
    X_train, y_train = X[idx_treino], y[idx_treino]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for nome in pendentes:
        caminho_joblib = os.path.join(artifacts_dir, f"{nome}.joblib")
        if not os.path.exists(caminho_joblib):
            continue  # sem o modelo treinado não há o que clonar
        modelo_base = clone(joblib.load(caminho_joblib))
        oof = cross_val_predict(
            make_pipeline(StandardScaler(), modelo_base), X_train, y_train,
            cv=cv, method="predict_proba",
        )[:, 1]
        with open(_caminho_oof(artifacts_dir, nome), "w", encoding="utf-8") as f:
            json.dump({"y_true": y_train.tolist(), "proba": oof.tolist()}, f, ensure_ascii=False)


def _caminho_test_proba(artifacts_dir, nome):
    return os.path.join(artifacts_dir, f"{nome}_test_proba.json")


def regenerar_test_proba_ausentes(metrics: dict, artifacts_dir=ARTIFACTS_DIR) -> None:
    """
    Regenera deterministicamente `{model}_test_proba.json` para cada modelo
    pré-semeado em `artifacts_dir` cujo arquivo ainda não exista. Espelha
    `regenerar_oof_ausentes` e existe pela mesma razão: os artefatos de
    referência versionados no repositório nasceram antes de o treino passar a
    gravar as probabilidades do conjunto de teste, e
    `PUT /models/{id}/threshold` depende delas para recalcular as métricas
    reportadas no limiar novo sem re-treinar (D11).

    Usa split.json (chave 'teste', em português — não renomear), o
    `scaler.joblib` do mesmo diretório (padronização ajustada só no treino) e o
    `{modelo}.joblib` já ajustado — reproduz exatamente o
    `modelo.predict_proba(scaler.transform(X_teste))` do treino.

    Degrada igual ao OOF: sem split.json, sem scaler ou sem o .joblib, retorna
    sem gravar; quem depende do arquivo responde 503.
    """
    split = _ler_opcional(artifacts_dir, "split.json")
    if split is None or "teste" not in split:
        return  # sem a partição de teste não há como reproduzir as probabilidades

    pendentes = [
        nome for nome in metrics["modelos"]
        if not os.path.exists(_caminho_test_proba(artifacts_dir, nome))
    ]
    if not pendentes:
        return

    caminho_scaler = os.path.join(artifacts_dir, "scaler.joblib")
    if not os.path.exists(caminho_scaler):
        return  # sem o scaler do treino a padronização não é reproduzível
    scaler = joblib.load(caminho_scaler)

    X, y, _ = load_data()
    idx_teste = np.array(split["teste"])
    X_teste_scaled = scaler.transform(X[idx_teste])
    y_teste = y[idx_teste]

    for nome in pendentes:
        caminho_joblib = os.path.join(artifacts_dir, f"{nome}.joblib")
        if not os.path.exists(caminho_joblib):
            continue  # sem o modelo treinado não há o que avaliar
        proba = joblib.load(caminho_joblib).predict_proba(X_teste_scaled)[:, 1]
        with open(_caminho_test_proba(artifacts_dir, nome), "w", encoding="utf-8") as f:
            json.dump({"y_true": y_teste.tolist(), "proba": proba.tolist()}, f, ensure_ascii=False)


def sincronizar_modelos(artifacts_dir=ARTIFACTS_DIR) -> list[TrainedModel]:
    """Cria os registros que faltam e atualiza `is_production`. Retorna os registros do treino atual."""
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    config_path = os.path.join(artifacts_dir, "train_config.json")
    if not (os.path.exists(metrics_path) and os.path.exists(config_path)):
        return []

    metrics = _ler(artifacts_dir, "metrics.json")
    config = _ler(artifacts_dir, "train_config.json")
    split = _ler_opcional(artifacts_dir, "split.json")
    treinado_em = _data(metrics.get("gerado_em"))

    regenerar_oof_ausentes(metrics, artifacts_dir)
    regenerar_test_proba_ausentes(metrics, artifacts_dir)

    # A identidade do registro é (execução, modelo) (D12), então o critério de
    # "já sincronizado" é existir uma linha **desta** referência: mesmo
    # artifact_path e mesmo hash dos bytes em disco. Reexecutar a sincronização
    # não duplica linhas; artefato regerado (hash novo) entra como pendente.
    def _registro_desta_referencia(nome):
        return TrainedModel.query.filter_by(
            artifact_path=_caminho_artefato(artifacts_dir, nome),
            artifact_sha256=config["hashes_sha256"][nome],
        ).first()

    pendentes = {
        nome: config["hashes_sha256"][nome]
        for nome in metrics["modelos"]
        if _registro_desta_referencia(nome) is None
    }
    run = _run_sintetico(metrics, config, split) if pendentes else None

    atuais = []
    for nome, r in metrics["modelos"].items():
        sha = config["hashes_sha256"][nome]
        if nome in pendentes:
            # teste_operacao é justamente o conjunto de teste avaliado no
            # limiar de segurança, que é o threshold inicial da linha: a
            # invariante sensitivity == vp/(vp+fn) vale por construção (D11).
            op = r["teste_operacao"]
            registro = TrainedModel(
                training_run_id=run.id,
                name=nome,
                accuracy=op["acuracia"],
                sensitivity=op["sensibilidade"],
                specificity=op["especificidade"],
                auc=op["auc"],
                false_negative_rate=op["taxa_falsos_negativos"],
                train_oof_sensitivity=_sensibilidade_oof_ou_none(
                    nome, r["limiar_seguranca"], artifacts_dir
                ),
                # ponto de operação vigente começa no limiar de segurança do treino
                threshold=r["limiar_seguranca"],
                safety_threshold=r["limiar_seguranca"],
                hyperparameters=config["hiperparametros"][nome],
                random_seed=config["random_state"],
                artifact_path=_caminho_artefato(artifacts_dir, nome),
                artifact_sha256=sha,
                roc_points=r.get("roc"),  # mesmo objeto {"fpr": [...], "tpr": [...]}
                confusion_matrix=op.get("matriz_confusao"),
                trained_at=treinado_em,
            )
            db.session.add(registro)
        else:
            registro = _registro_desta_referencia(nome)
        atuais.append(registro)

    db.session.flush()
    ids_atuais = {m.id for m in atuais}
    for m in TrainedModel.query.all():
        m.is_production = m.id in ids_atuais and m.name == metrics["melhor_modelo"]
    db.session.commit()
    return atuais


def modelo_registrado_mais_recente(nome: str) -> TrainedModel | None:
    """Registro mais recente do modelo `nome` em trained_models."""
    return (TrainedModel.query.filter_by(name=nome)
            .order_by(TrainedModel.trained_at.desc(), TrainedModel.id.desc()).first())


def modelo_em_producao() -> TrainedModel | None:
    """Modelo marcado como ponto de operação vigente da ferramenta."""
    return TrainedModel.query.filter_by(is_production=True).order_by(TrainedModel.id.desc()).first()


# ----------------------------------------------------------------------
# Treino sob demanda (POST /api/v1/training-runs) — design.md §5.1, §5.4
# ----------------------------------------------------------------------
def criar_training_run(models_requested: list[str], test_size: float, grid_search: bool) -> TrainingRun:
    """
    Cria o registro `training_runs` com status='running' e commit imediato,
    para o GET /training-runs/{id} refletir o andamento se consultado durante
    a execução síncrona do treino.
    """
    run = TrainingRun(
        models_requested=models_requested,
        test_size=test_size,
        random_seed=RANDOM_STATE,
        grid_search=grid_search,
        status="running",
    )
    db.session.add(run)
    db.session.commit()
    return run


def concluir_training_run_sucesso(run: TrainingRun, resultado: dict, artifacts_dir=None) -> list[TrainedModel]:
    """
    Preenche o `training_run` com os dados de reprodutibilidade do treino
    (resultado de ml.train.train_and_evaluate) e grava **um registro novo** de
    `trained_models` por modelo treinado, sempre com `artifact_path` apontando
    para o artefato desta execução (D12). Marca o melhor modelo do run como
    is_production e desmarca os demais. Retorna os registros criados.

    O diretório dos artefatos vem do próprio resultado do treino
    (`output_dir`), que para a API é `ml/artifacts/runs/{run.id}/` (D8);
    `artifacts_dir` explícito tem precedência e ARTIFACTS_DIR é só o último
    recurso, para resultados antigos sem `output_dir`.
    """
    metrics, config, split = resultado["metrics"], resultado["config"], resultado["split"]
    artifacts_dir = artifacts_dir or resultado.get("output_dir") or ARTIFACTS_DIR

    run.dataset_sha256 = config["hashes_sha256"].get("dataset")
    run.split = split
    run.library_versions = config.get("versoes")
    run.git_commit = config.get("commit_git")
    run.status = "success"
    run.finished_at = _data(metrics.get("gerado_em"))

    treinado_em = _data(metrics.get("gerado_em"))
    criados = []
    for nome, r in metrics["modelos"].items():
        op = r["teste_operacao"]
        # D12: uma linha por (execução, modelo). O mesmo conjunto de bytes pode
        # aparecer legitimamente em execuções distintas (treino determinista com
        # a mesma semente), então o hash não identifica o registro — ele é só
        # procedência. Reaproveitar a linha por hash era o que fazia modelos
        # nascidos de uma execução 70/30 apontarem para a referência 80/20.
        registro = TrainedModel(
            training_run_id=run.id,
            name=nome,
            accuracy=op["acuracia"],
            sensitivity=op["sensibilidade"],
            specificity=op["especificidade"],
            auc=op["auc"],
            false_negative_rate=op["taxa_falsos_negativos"],
            train_oof_sensitivity=_sensibilidade_oof_ou_none(
                nome, r["limiar_seguranca"], artifacts_dir
            ),
            threshold=r["limiar_seguranca"],
            safety_threshold=r["limiar_seguranca"],
            hyperparameters=config["hiperparametros"][nome],
            random_seed=config["random_state"],
            artifact_path=_caminho_artefato(artifacts_dir, nome),
            artifact_sha256=config["hashes_sha256"][nome],
            roc_points=r.get("roc"),
            confusion_matrix=op.get("matriz_confusao"),
            trained_at=treinado_em,
        )
        db.session.add(registro)
        criados.append(registro)

    db.session.flush()
    melhor = metrics["melhor_modelo"]
    ids_deste_run = {m.id for m in criados}
    for m in TrainedModel.query.all():
        m.is_production = m.id in ids_deste_run and m.name == melhor
    db.session.commit()
    return criados


def falhar_training_run(run: TrainingRun, mensagem: str) -> None:
    """Marca o run como falho, preservando o registro (não é apagado)."""
    run.status = "failed"
    run.error_message = mensagem
    run.finished_at = datetime.now(run.started_at.tzinfo) if run.started_at else datetime.now()
    db.session.commit()


# ----------------------------------------------------------------------
# Promoção de modelo (PUT /api/v1/models/{id}/threshold) — design.md §5.3, RNF04
# ----------------------------------------------------------------------
class OofIndisponivel(Exception):
    """{model}_oof.json ausente e não foi possível regenerá-lo."""


class TesteProbaIndisponivel(Exception):
    """{model}_test_proba.json ausente e não foi possível regenerá-lo."""


def sensibilidade_oof(nome_modelo: str, threshold: float, artifacts_dir=ARTIFACTS_DIR) -> float:
    """
    Sensibilidade de (oof_proba >= threshold) contra y_true, lida de
    {model}_oof.json. É o **portão de promoção** do RNF04 e nunca uma métrica
    reportada: escolher ou validar limiar olhando o conjunto de teste vazaria
    informação (design.md §6). O valor vai para
    `trained_models.train_oof_sensitivity`.
    """
    oof = _ler_opcional(artifacts_dir, f"{nome_modelo}_oof.json")
    if oof is None:
        raise OofIndisponivel(f"Vetor out-of-fold de '{nome_modelo}' indisponível.")
    y_true = np.array(oof["y_true"])
    proba = np.array(oof["proba"])
    positivos = y_true == 1
    if not positivos.any():
        return 0.0
    previsto_positivo = proba >= threshold
    return float((previsto_positivo & positivos).sum() / positivos.sum())


def _sensibilidade_oof_ou_none(nome_modelo, threshold, artifacts_dir):
    """
    Portão out-of-fold para preencher `train_oof_sensitivity` na criação da
    linha. Devolve None quando o OOF não está legível: um treino bem-sucedido
    não pode falhar por causa de um campo informativo. Onde o portão decide
    (promoção), a ausência vira 503 antes de qualquer escrita.
    """
    try:
        return sensibilidade_oof(nome_modelo, threshold, artifacts_dir)
    except OofIndisponivel:
        return None


def metricas_teste_no_limiar(nome_modelo: str, threshold: float,
                             artifacts_dir=ARTIFACTS_DIR) -> dict:
    """
    Métricas do CONJUNTO DE TESTE no `threshold` pedido, a partir das
    probabilidades persistidas em `{modelo}_test_proba.json` (D11).

    Devolve o dict de `ml.train.compute_metrics` (chaves em português:
    acuracia, sensibilidade, especificidade, auc, taxa_falsos_negativos,
    matriz_confusao). Reusar `compute_metrics` é deliberado: existe uma única
    implementação da matriz de confusão e das razões em todo o projeto, e é o
    que faz a invariante `sensibilidade == vp/(vp+fn)` valer por construção.

    O conjunto de teste nunca escolhe o limiar — só reporta as métricas no
    limiar já decidido pelo out-of-fold do treino.
    """
    dados = _ler_opcional(artifacts_dir, f"{nome_modelo}_test_proba.json")
    if dados is None:
        raise TesteProbaIndisponivel(
            f"Probabilidades do conjunto de teste de '{nome_modelo}' indisponíveis."
        )
    y_true = np.array(dados["y_true"])
    proba = np.array(dados["proba"])
    return compute_metrics(y_true, (proba >= threshold).astype(int), proba)


def promover_modelo(registro: TrainedModel, threshold: float,
                    sensibilidade_oof_valor: float, metricas_teste: dict) -> None:
    """
    Atualiza o ponto de operação e marca este modelo como produção (RNF04).

    Grava a linha **inteira e coerente**, separando as duas grandezas que o
    defeito A confundia:

    - `train_oof_sensitivity` recebe `sensibilidade_oof_valor`, o portão
      calculado nas predições out-of-fold do TREINO no limiar pedido;
    - `accuracy`, `sensitivity`, `specificity`, `auc`, `false_negative_rate` e
      `confusion_matrix` recebem `metricas_teste`, todas do CONJUNTO DE TESTE
      recalculadas em conjunto no mesmo limiar. `auc` é reescrita com o mesmo
      valor de antes: ela depende só das probabilidades, não do limiar.

    O valor out-of-fold **nunca** é gravado em `sensitivity`.
    """
    registro.threshold = threshold
    registro.train_oof_sensitivity = sensibilidade_oof_valor
    registro.accuracy = metricas_teste["acuracia"]
    registro.sensitivity = metricas_teste["sensibilidade"]
    registro.specificity = metricas_teste["especificidade"]
    registro.auc = metricas_teste["auc"]
    registro.false_negative_rate = metricas_teste["taxa_falsos_negativos"]
    registro.confusion_matrix = metricas_teste["matriz_confusao"]
    for m in TrainedModel.query.all():
        m.is_production = (m.id == registro.id)
    db.session.commit()


if __name__ == "__main__":
    from app import create_app

    app = create_app()
    with app.app_context():
        db.create_all()
        for m in sincronizar_modelos():
            print(f"{m.name}: {m.artifact_sha256[:12]}… is_production={m.is_production}")
