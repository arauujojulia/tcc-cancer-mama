# Rode com: python -m ml.report
"""
Exportação científica (US06): figuras em 300 DPI e tabelas CSV a partir
dos artefatos gerados por `python -m ml.train`.

Saída em ml/artifacts/report/:
  roc_comparacao.png, matrizes_confusao.png, importancia_<modelo>.png,
  metricas.csv
"""
import csv
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.train import ARTIFACTS_DIR  # noqa: E402

DPI = 300
NOMES = {"regressao_logistica": "Regressão Logística", "svm": "SVM (RBF)"}


def _ler(artifacts_dir, nome):
    with open(os.path.join(artifacts_dir, nome), encoding="utf-8") as f:
        return json.load(f)


def metricas_csv(artifacts_dir=ARTIFACTS_DIR) -> str:
    """Tabela comparativa (modelos x métricas x ponto de operação) + parâmetros."""
    metrics = _ler(artifacts_dir, "metrics.json")
    config = _ler(artifacts_dir, "train_config.json")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["modelo", "ponto_operacao", "limiar", "acuracia", "sensibilidade",
                "especificidade", "auc", "taxa_falsos_negativos", "vn", "fp", "fn", "vp",
                "cv_sens_media", "cv_sens_desvio", "hiperparametros", "semente",
                "sha256_modelo", "melhor_modelo"])
    for nome, r in metrics["modelos"].items():
        for rotulo, chave, limiar in (("padrao", "teste_padrao", r["limiar_padrao"]),
                                      ("seguranca", "teste_operacao", r["limiar_seguranca"])):
            m = r[chave]
            c = m["matriz_confusao"]
            w.writerow([
                nome, rotulo, limiar, f"{m['acuracia']:.4f}", f"{m['sensibilidade']:.4f}",
                f"{m['especificidade']:.4f}", f"{m['auc']:.4f}", f"{m['taxa_falsos_negativos']:.4f}",
                c["vn"], c["fp"], c["fn"], c["vp"],
                f"{r['validacao_cruzada']['sensibilidade']['media']:.4f}",
                f"{r['validacao_cruzada']['sensibilidade']['desvio']:.4f}",
                json.dumps(config["hiperparametros"][nome], sort_keys=True),
                config["random_state"], config["hashes_sha256"].get(nome, ""),
                "sim" if nome == metrics["melhor_modelo"] else "nao",
            ])
    return buf.getvalue()


def gerar_figuras(artifacts_dir=ARTIFACTS_DIR, out_dir=None):
    out_dir = out_dir or os.path.join(artifacts_dir, "report")
    os.makedirs(out_dir, exist_ok=True)
    metrics = _ler(artifacts_dir, "metrics.json")
    importancia = _ler(artifacts_dir, "feature_importance.json")
    arquivos = []

    # Curvas ROC sobrepostas
    fig, ax = plt.subplots(figsize=(5, 5))
    for nome, r in metrics["modelos"].items():
        auc = r["teste_padrao"]["auc"]
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{NOMES.get(nome, nome)} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="Taxa de falsos positivos (1 − especificidade)",
           ylabel="Sensibilidade", title="Curvas ROC — conjunto de teste")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = os.path.join(out_dir, "roc_comparacao.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    arquivos.append(p)

    # Matrizes de confusão no ponto de operação (limiar de segurança)
    modelos = list(metrics["modelos"].items())
    fig, axes = plt.subplots(1, len(modelos), figsize=(4.2 * len(modelos), 4))
    axes = [axes] if len(modelos) == 1 else list(axes)
    for ax, (nome, r) in zip(axes, modelos):
        c = r["teste_operacao"]["matriz_confusao"]
        mat = [[c["vn"], c["fp"]], [c["fn"], c["vp"]]]
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mat[i][j], ha="center", va="center",
                        color="white" if mat[i][j] > max(map(max, mat)) / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1], ["benigno", "maligno"])
        ax.set_yticks([0, 1], ["benigno", "maligno"])
        ax.set(xlabel="Previsto", ylabel="Real",
               title=f"{NOMES.get(nome, nome)}\nlimiar {r['limiar_seguranca']:.2f}")
    fig.tight_layout()
    p = os.path.join(out_dir, "matrizes_confusao.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    arquivos.append(p)

    # Ranking de características (top 15) por modelo
    for nome, ranking in importancia["modelos"].items():
        top = ranking[:15][::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([r["caracteristica"] for r in top], [r["importancia"] for r in top],
                xerr=[r["desvio"] for r in top], color="#2b6cb0")
        ax.set(xlabel="Importância por permutação (Δ log-loss)",
               title=f"Características mais relevantes — {NOMES.get(nome, nome)}")
        fig.tight_layout()
        p = os.path.join(out_dir, f"importancia_{nome}.png")
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        arquivos.append(p)

    csv_path = os.path.join(out_dir, "metricas.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(metricas_csv(artifacts_dir))
    arquivos.append(csv_path)
    return arquivos


if __name__ == "__main__":
    for caminho in gerar_figuras():
        print(caminho)
