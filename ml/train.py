"""
Treinamento e avaliação dos classificadores, com cada modelo logado
como um run separado no Weights & Biases (mesmo projeto/grupo), pra
comparar os dois lado a lado no painel.

Antes de rodar, faça login uma vez no terminal (não precisa repetir
depois, fica salvo na máquina):
    wandb login
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import wandb
from ml.data import load_data

RANDOM_STATE = 42
TEST_SIZE = 0.2
WANDB_PROJECT = "tcc-cancer-mama"

MODELS = {
    "regressao_logistica": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}


def compute_metrics(y_true, y_pred, y_proba):
    """
    Sensibilidade = recall da classe maligno (evita falso negativo).
    Especificidade = recall da classe benigno.
    Taxa de falsos negativos = malignos classificados como benignos —
    métrica prioritária do projeto.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "sensibilidade": tp / (tp + fn) if (tp + fn) else 0.0,
        "especificidade": tn / (tn + fp) if (tn + fp) else 0.0,
        "auc": roc_auc_score(y_true, y_proba),
        "taxa_falsos_negativos": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def train_and_evaluate():
    X, y, feature_names = load_data()

    # stratify=y garante que a proporção de malignos/benignos seja a
    # mesma no treino e no teste — importante porque as classes não são
    # 50/50 (357 benignos, 212 malignos).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Padronização: importante sobretudo para o SVM, que é sensível à
    # escala das features (regressão logística também se beneficia).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    resultados = {}
    for nome, modelo in MODELS.items():
        run = wandb.init(
            project=WANDB_PROJECT,
            name=nome,
            group="comparacao-classificadores",
            config={
                "modelo": nome,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                **modelo.get_params(),
            },
            reinit=True,
        )

        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        wandb.log(metrics)
        resultados[nome] = metrics

        run.finish()

    return resultados


if __name__ == "__main__":
    resultados = train_and_evaluate()
    for nome, metrics in resultados.items():
        print(nome, metrics)