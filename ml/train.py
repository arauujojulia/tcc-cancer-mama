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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
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

MODELS = {
    "regressao_logistica": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

CRITERIO_MELHOR_MODELO = (
    "Maior sensibilidade no ponto de operação (limiar de segurança); "
    "desempate por AUC e depois por especificidade."
)


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


# ----------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------
def train_and_evaluate(output_dir=ARTIFACTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    X, y, feature_names = load_data()
    idx = np.arange(len(y))

    # stratify=y garante a mesma proporção de malignos/benignos no treino
    # e no teste (357 benignos, 212 malignos).
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Padronização ajustada somente no treino (importante sobretudo para o SVM).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    resultados, importancias, hashes = {}, {}, {"scaler": sha256_arquivo(scaler_path)}
    for nome, modelo in MODELS.items():
        # 1) validação cruzada estratificada no treino
        cv_res = _cv_metrics(modelo, X_train, y_train, cv)

        # 2) limiar de segurança a partir de predições out-of-fold (sem usar o teste)
        oof = cross_val_predict(
            make_pipeline(StandardScaler(), clone(modelo)), X_train, y_train,
            cv=cv, method="predict_proba",
        )[:, 1]
        limiar, atingiu_oof = escolher_limiar_seguranca(y_train, oof)

        # 3) ajuste final no treino completo e avaliação no teste
        modelo.fit(X_train_scaled, y_train)
        proba = modelo.predict_proba(X_test_scaled)[:, 1]
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

        # 4) importância por permutação (US06) — aumento da log-loss no teste (mais estável que AUC com base quase separável)
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
    config = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "n_treino": int(len(idx_train)),
        "n_teste": int(len(idx_test)),
        "distribuicao_treino": {"benigno": int((y_train == 0).sum()), "maligno": int((y_train == 1).sum())},
        "distribuicao_teste": {"benigno": int((y_test == 0).sum()), "maligno": int((y_test == 1).sum())},
        "hiperparametros": {nome: modelo.get_params() for nome, modelo in MODELS.items()},
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
    _salvar_json(os.path.join(output_dir, "split.json"), {
        "random_state": RANDOM_STATE,
        "treino": idx_train.tolist(),
        "teste": idx_test.tolist(),
    })

    return metrics


def _salvar_json(caminho, obj):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    resultado = train_and_evaluate()
    print(f"Melhor modelo: {resultado['melhor_modelo']}")
    for nome, r in resultado["modelos"].items():
        for rotulo in ("teste_padrao", "teste_operacao"):
            m = r[rotulo]
            lim = r["limiar_padrao"] if rotulo == "teste_padrao" else r["limiar_seguranca"]
            print(f"{nome:20s} {rotulo:15s} limiar={lim:.2f} "
                  f"acc={m['acuracia']:.4f} sens={m['sensibilidade']:.4f} "
                  f"esp={m['especificidade']:.4f} auc={m['auc']:.4f} "
                  f"FN={m['taxa_falsos_negativos']:.4f}")
