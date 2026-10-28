# Rode com: python -m ml.train
"""
Sprint 1 (US04) + Sprint 2 (US05, US06):

- treina Regressão Logística e SVM sobre a mesma partição treino/teste;
- avalia com acurácia, sensibilidade, especificidade, AUC e taxa de
  falsos negativos (US05), no conjunto de teste e em validação cruzada
  estratificada (5 folds) sobre o treino;
- escolhe um limiar de decisão de segurança usando apenas predições
  out-of-fold do treino (sem vazamento do teste), priorizando
  sensibilidade >= 0,97 (RNF04 / meta do módulo Laboratório de Modelos);
- calcula a importância das características por permutação (US06);
- registra semente, partições, hiperparâmetros, versões das bibliotecas
  e hashes SHA-256 dos artefatos para reprodutibilidade (US06 / RNF03).

Onde os artefatos são gravados (D8 — correções pós-revisão):

- `ARTIFACTS_DIR` (ml/artifacts/) é a **referência versionada do artigo**
  (split 80/20, semente 42). Só `python -m ml.train` grava ali: é o caminho
  deliberado de regerar a referência, e o commit resultante é o que o artigo
  cita. Nenhuma rota da API escreve nesse diretório.
- O treino sob demanda (`POST /api/v1/training-runs`) sempre passa
  `output_dir=diretorio_execucao(training_run_id)`, isto é,
  `ml/artifacts/runs/{training_run_id}/` — descartável, fora do versionamento
  (`.gitignore`). Assim um treino disparado pela tela do Laboratório não
  altera silenciosamente as métricas publicadas (RNF03).
"""
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.data import DATA_PATH, load_data

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
LIMIAR_PADRAO = 0.5
SENSIBILIDADE_MINIMA = 0.97  # meta de segurança clínica (docs, módulo 4.2.4)
PERMUTATION_REPEATS = 30
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
# Raiz das execuções de treino sob demanda: cada run tem a sua subpasta, para
# nunca sobrescrever a referência versionada em ARTIFACTS_DIR (D8 / RNF03).
RUNS_DIR = os.path.join(ARTIFACTS_DIR, "runs")

MODELS = {
    "regressao_logistica": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

# Grades padrão do GridSearchCV (design.md §5.2), sobrescrevíveis por modelo
# via o parâmetro `hyperparameters` de train_and_evaluate(). O prefixo
# "clone__" é o passo do Pipeline (make_pipeline(StandardScaler(), clone(modelo))).
PARAM_GRIDS = {
    "regressao_logistica": {
        "clone__C": [0.01, 0.1, 1, 10],
        "clone__penalty": ["l2"],
        "clone__max_iter": [1000],
    },
    "svm": {
        "clone__C": [0.1, 1, 10],
        "clone__gamma": ["scale", 0.01, 0.1],
        "clone__kernel": ["rbf"],
        "clone__probability": [True],
    },
}

CRITERIO_MELHOR_MODELO = (
    "Maior sensibilidade no ponto de operação (limiar de segurança); "
    "desempate por AUC e depois por especificidade."
)


def diretorio_execucao(training_run_id) -> str:
    """
    Diretório de artefatos de uma execução de treino sob demanda:
    `ml/artifacts/runs/{training_run_id}/` (D8). Nunca devolve ARTIFACTS_DIR,
    que é a referência versionada regerada só por `python -m ml.train`.
    """
    return os.path.join(RUNS_DIR, str(training_run_id))


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    """
    Sensibilidade = recall da classe maligno (evita falso negativo).
    Especificidade = recall da classe benigno.
    Taxa de falsos negativos = malignos classificados como benignos —
    métrica prioritária do projeto (RNF04).
    """
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())

    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "sensibilidade": tp / (tp + fn) if (tp + fn) else 0.0,
        "especificidade": tn / (tn + fp) if (tn + fp) else 0.0,
        "auc": float(roc_auc_score(y_true, y_proba)),
        "taxa_falsos_negativos": fn / (fn + tp) if (fn + tp) else 0.0,
        "matriz_confusao": {"vn": tn, "fp": fp, "fn": fn, "vp": tp},
    }


def _resumo(valores):
    arr = np.asarray(valores, dtype=float)
    return {"media": float(arr.mean()), "desvio": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def escolher_limiar_seguranca(y_true, y_proba, sens_minima=SENSIBILIDADE_MINIMA):
    """
    Maior limiar (<= 0,5) em que a sensibilidade das predições
    out-of-fold ainda atinge `sens_minima`. Maior limiar = menos falsos
    positivos, dentro da restrição de segurança. Se nenhum atinge, usa o
    menor limiar da grade (máxima cautela) e sinaliza em `atingiu_meta`.
    """
    grade = np.round(np.arange(0.01, LIMIAR_PADRAO + 1e-9, 0.01), 2)
    positivos = y_true == 1
    escolhido, atingiu = float(grade[0]), False
    for t in grade[::-1]:
        sens = float(((y_proba >= t) & positivos).sum() / positivos.sum())
        if sens >= sens_minima:
            escolhido, atingiu = float(t), True
            break
    return escolhido, atingiu


def _roc_pontos(y_true, y_proba, max_pontos=200):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    if len(fpr) > max_pontos:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_pontos).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": [round(float(v), 5) for v in fpr], "tpr": [round(float(v), 5) for v in tpr]}


def _cv_metrics(modelo, X_train, y_train, cv):
    """Métricas por fold (padronização reajustada dentro de cada fold)."""
    por_fold = []
    for tr, va in cv.split(X_train, y_train):
        pipe = make_pipeline(StandardScaler(), clone(modelo))
        pipe.fit(X_train[tr], y_train[tr])
        proba = pipe.predict_proba(X_train[va])[:, 1]
        m = compute_metrics(y_train[va], (proba >= LIMIAR_PADRAO).astype(int), proba)
        por_fold.append(m)
    chaves = ["acuracia", "sensibilidade", "especificidade", "auc", "taxa_falsos_negativos"]
    return {k: _resumo([f[k] for f in por_fold]) for k in chaves} | {"folds": CV_FOLDS}


# ----------------------------------------------------------------------
# Reprodutibilidade
# ----------------------------------------------------------------------
def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _commit_git():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=os.path.dirname(__file__), timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _selecionar_melhor(resultados):
    return max(
        resultados,
        key=lambda n: (
            round(resultados[n]["teste_operacao"]["sensibilidade"], 12),
            round(resultados[n]["teste_operacao"]["auc"], 12),
            round(resultados[n]["teste_operacao"]["especificidade"], 12),
        ),
    )


def _oof_proba(modelo, X_train, y_train, cv):
    """Predições out-of-fold (probabilidade de maligno) do treino, sem tocar no teste."""
    return cross_val_predict(
        make_pipeline(StandardScaler(), clone(modelo)), X_train, y_train,
        cv=cv, method="predict_proba",
    )[:, 1]


def _ajustar_modelo(nome, modelo, X_train, X_train_scaled, y_train, grid_search, hyperparameters):
    """
    Ajusta `modelo` no treino completo e devolve
    (modelo_ajustado, hiperparametros_persistidos).

    Quando `grid_search` é verdadeiro, usa GridSearchCV com a grade padrão de
    `nome` (sobrescrita pelas chaves correspondentes em `hyperparameters`, se
    fornecidas) sobre um pipeline StandardScaler+modelo, cv=StratifiedKFold(5,
    shuffle, random_state=42), scoring="recall" (D6). O estimador retornado já
    é só o classificador (sem o StandardScaler): é reajustado em `X_train_scaled`
    (já padronizado pelo `scaler` global) para o restante do pipeline de
    avaliação continuar usando o mesmo scaler salvo em disco.

    Quando `grid_search` é falso, ajusta `modelo` direto em `X_train_scaled`
    com os hiperparâmetros padrão (comportamento anterior).
    """
    if not grid_search:
        ajustado = clone(modelo)
        ajustado.fit(X_train_scaled, y_train)
        return ajustado, ajustado.get_params()

    grade = dict(PARAM_GRIDS.get(nome, {}))
    if hyperparameters and nome in hyperparameters and isinstance(hyperparameters[nome], dict):
        for chave, valores in hyperparameters[nome].items():
            chave_pipeline = chave if chave.startswith("clone__") else f"clone__{chave}"
            if chave_pipeline in grade:
                grade[chave_pipeline] = valores if isinstance(valores, list) else [valores]

    pipeline = Pipeline([("scaler", StandardScaler()), ("clone", clone(modelo))])
    busca = GridSearchCV(
        pipeline, grade,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="recall", refit=True,
    )
    busca.fit(X_train, y_train)
    melhores_params = {
        k.removeprefix("clone__"): v for k, v in busca.best_params_.items()
    }
    ajustado = clone(modelo).set_params(**melhores_params)
    ajustado.fit(X_train_scaled, y_train)
    return ajustado, ajustado.get_params()


# ----------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------
def train_and_evaluate(output_dir=ARTIFACTS_DIR, models=None, test_size=TEST_SIZE,
                        grid_search=True, hyperparameters=None):
    """
    Treina e avalia os modelos pedidos (ou todos, por padrão).

    models: subconjunto de MODELS (default: todos).
    test_size: proporção do conjunto de teste (default 0.2).
    grid_search: se True (default), ajusta cada modelo com GridSearchCV
        (5 folds estratificados, scoring="recall") sobre as grades padrão da
        seção 5.2 do design, sobrescritas por `hyperparameters` quando dado.
    hyperparameters: dict opcional {nome_modelo: {hiperparametro: [valores]}}
        para sobrescrever a grade padrão.

    Grava os artefatos em `output_dir` (scaler, {modelo}.joblib,
    {modelo}_oof.json, {modelo}_test_proba.json, metrics.json,
    feature_importance.json, train_config.json, split.json) e RETORNA um dict
    com `metrics`, `config`, `split`, `artifact_paths` e `output_dir`.

    `{modelo}_test_proba.json` ({"y_true", "proba"}) é o insumo do recálculo das
    métricas reportadas quando o limiar vigente muda
    (`PUT /api/v1/models/{id}/threshold`): o conjunto de teste nunca escolhe o
    limiar, só reporta as métricas no limiar já decidido pelo out-of-fold.

    `output_dir` default é ARTIFACTS_DIR, a **referência versionada do artigo**
    (split 80/20, semente 42), regerada somente por `python -m ml.train`. A API
    sempre passa `output_dir=diretorio_execucao(training_run_id)` para gravar em
    `ml/artifacts/runs/{id}/` e não sobrescrever essa referência (D8 / RNF03);
    o `output_dir` devolvido é o canal por onde `services/modelos.py` descobre
    onde os artefatos foram gravados.
    """
    os.makedirs(output_dir, exist_ok=True)
    modelos_selecionados = {k: v for k, v in MODELS.items() if models is None or k in models}

    X, y, feature_names = load_data()
    idx = np.arange(len(y))

    # stratify=y garante a mesma proporção de malignos/benignos no treino
    # e no teste (357 benignos, 212 malignos, na proporção pedida).
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Padronização ajustada somente no treino (importante sobretudo para o SVM).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    resultados, importancias, hashes = {}, {}, {"scaler": sha256_arquivo(scaler_path)}
    hiperparametros_persistidos = {}
    for nome, modelo_base in modelos_selecionados.items():
        # 1) validação cruzada estratificada no treino (com os hiperparâmetros padrão,
        #    para o relatório de estabilidade do modelo-base continuar comparável)
        cv_res = _cv_metrics(modelo_base, X_train, y_train, cv)

        # 2) ajuste final (com ou sem GridSearchCV) sobre o treino completo
        modelo, params = _ajustar_modelo(
            nome, modelo_base, X_train, X_train_scaled, y_train, grid_search, hyperparameters
        )
        hiperparametros_persistidos[nome] = params

        # 3) limiar de segurança a partir de predições out-of-fold do AJUSTADO
        #    (sem usar o teste) — o mesmo vetor é persistido em {modelo}_oof.json
        #    para a promoção (PUT /models/{id}/threshold) recalcular sensibilidade
        #    sem re-treinar.
        oof = _oof_proba(modelo, X_train, y_train, cv)
        limiar, atingiu_oof = escolher_limiar_seguranca(y_train, oof)
        _salvar_json(os.path.join(output_dir, f"{nome}_oof.json"), {
            "y_true": y_train.tolist(),
            "proba": oof.tolist(),
        })

        # 4) avaliação no teste (nunca usado para escolher o limiar) — `modelo`
        #    já foi ajustado em X_train_scaled dentro de _ajustar_modelo. O vetor
        #    de probabilidades é persistido em {modelo}_test_proba.json para que
        #    PUT /models/{id}/threshold recalcule as métricas reportadas no limiar
        #    novo sem re-treinar e sem carregar o dataset.
        proba = modelo.predict_proba(X_test_scaled)[:, 1]
        _salvar_json(os.path.join(output_dir, f"{nome}_test_proba.json"), {
            "y_true": y_test.tolist(),
            "proba": proba.tolist(),
        })
        teste_padrao = compute_metrics(y_test, (proba >= LIMIAR_PADRAO).astype(int), proba)
        teste_op = compute_metrics(y_test, (proba >= limiar).astype(int), proba)

        resultados[nome] = {
            "limiar_padrao": LIMIAR_PADRAO,
            "limiar_seguranca": limiar,
            "limiar_atingiu_meta_no_treino": atingiu_oof,
            "teste_padrao": teste_padrao,
            "teste_operacao": teste_op,
            "atende_meta_sensibilidade_teste": teste_op["sensibilidade"] >= SENSIBILIDADE_MINIMA,
            "validacao_cruzada": cv_res,
            "roc": _roc_pontos(y_test, proba),
        }

        # 5) importância por permutação (US06) — aumento da log-loss no teste (mais estável que AUC com base quase separável)
        perm = permutation_importance(
            modelo, X_test_scaled, y_test, scoring="neg_log_loss",
            n_repeats=PERMUTATION_REPEATS, random_state=RANDOM_STATE,
        )
        ranking = sorted(
            (
                {"caracteristica": feature_names[i],
                 "importancia": float(perm.importances_mean[i]),
                 "desvio": float(perm.importances_std[i])}
                for i in range(len(feature_names))
            ),
            key=lambda r: r["importancia"], reverse=True,
        )
        importancias[nome] = ranking

        caminho = os.path.join(output_dir, f"{nome}.joblib")
        joblib.dump(modelo, caminho)
        hashes[nome] = sha256_arquivo(caminho)

    melhor = _selecionar_melhor(resultados)

    metrics = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "classe_positiva": "maligno",
        "meta_sensibilidade": SENSIBILIDADE_MINIMA,
        "melhor_modelo": melhor,
        "criterio_melhor_modelo": CRITERIO_MELHOR_MODELO,
        "modelos": resultados,
    }
    _salvar_json(os.path.join(output_dir, "metrics.json"), metrics)
    _salvar_json(os.path.join(output_dir, "feature_importance.json"), {
        "metodo": "permutação (aumento médio da log-loss no conjunto de teste)",
        "repeticoes": PERMUTATION_REPEATS,
        "melhor_modelo": melhor,
        "modelos": importancias,
    })

    # RNF03 / US06: partições, hiperparâmetros, semente, versões e hashes.
    # metrics.json NÃO contém hiperparâmetros — eles ficam aqui e no .joblib.
    config = {
        "test_size": test_size,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "grid_search": grid_search,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "n_treino": int(len(idx_train)),
        "n_teste": int(len(idx_test)),
        "distribuicao_treino": {"benigno": int((y_train == 0).sum()), "maligno": int((y_train == 1).sum())},
        "distribuicao_teste": {"benigno": int((y_test == 0).sum()), "maligno": int((y_test == 1).sum())},
        "hiperparametros": hiperparametros_persistidos,
        "hashes_sha256": {**hashes, "dataset": sha256_arquivo(DATA_PATH)},
        "versoes": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "commit_git": _commit_git(),
    }
    _salvar_json(os.path.join(output_dir, "train_config.json"), config)
    # Índices das linhas de cada partição (reconstrução exata do experimento).
    split = {
        "random_state": RANDOM_STATE,
        "treino": idx_train.tolist(),
        "teste": idx_test.tolist(),
    }
    _salvar_json(os.path.join(output_dir, "split.json"), split)

    return {
        "metrics": metrics,
        "config": config,
        "split": split,
        "artifact_paths": {
            nome: os.path.join(output_dir, f"{nome}.joblib") for nome in modelos_selecionados
        },
        "output_dir": output_dir,
    }


def _salvar_json(caminho, obj):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    # Único caminho que regenera a referência versionada (D8): o treino
    # disparado pela API grava em ml/artifacts/runs/{training_run_id}/.
    print("Regenerando os artefatos de referência em ml/artifacts/ "
          "(RNF03 — split 80/20, semente 42)")
    resultado = train_and_evaluate()["metrics"]
    print(f"Melhor modelo: {resultado['melhor_modelo']}")
    for nome, r in resultado["modelos"].items():
        for rotulo in ("teste_padrao", "teste_operacao"):
            m = r[rotulo]
            lim = r["limiar_padrao"] if rotulo == "teste_padrao" else r["limiar_seguranca"]
            print(f"{nome:20s} {rotulo:15s} limiar={lim:.2f} "
                  f"acc={m['acuracia']:.4f} sens={m['sensibilidade']:.4f} "
                  f"esp={m['especificidade']:.4f} auc={m['auc']:.4f} "
                  f"FN={m['taxa_falsos_negativos']:.4f}")
