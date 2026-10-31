#!/usr/bin/env python3
"""
Reavalia a seleção do melhor modelo usando apenas predições out-of-fold do treino.

Lê os arquivos *_oof.json e metrics.json de ml/artifacts/, aplica o critério
correto (sensibilidade OOF > AUC OOF > especificidade OOF) e regrava somente
`melhor_modelo` e `criterio_melhor_modelo` em metrics.json, além de
`melhor_modelo` em feature_importance.json.

Uso:
    python -m ml.reavaliar_selecao
"""
import json
import os
import sys

import numpy as np

from ml.train import ARTIFACTS_DIR, CRITERIO_MELHOR_MODELO, compute_metrics


def main():
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.json")

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    modelos = metrics.get("modelos", {})
    if not modelos:
        print("Nenhum modelo encontrado em metrics.json.")
        sys.exit(1)

    oof_por_modelo = {}
    for nome in modelos:
        oof_path = os.path.join(ARTIFACTS_DIR, f"{nome}_oof.json")
        if not os.path.exists(oof_path):
            print(f"Aviso: {oof_path} não encontrado, pulando {nome}.")
            continue
        with open(oof_path, encoding="utf-8") as f:
            oof_data = json.load(f)
        limiar = modelos[nome].get("limiar_seguranca", 0.5)
        oof_por_modelo[nome] = {
            "y_true": oof_data["y_true"],
            "proba": oof_data["proba"],
            "limiar": limiar,
        }

    if not oof_por_modelo:
        print("Nenhum arquivo _oof.json encontrado.")
        sys.exit(1)

    # Seleção pelo critério OOF
    def _chave_oof(nome):
        oof = oof_por_modelo[nome]
        yt = np.array(oof["y_true"])
        pr = np.array(oof["proba"])
        limiar = oof["limiar"]
        yp = (pr >= limiar).astype(int)
        m = compute_metrics(yt, yp, pr)
        return (
            round(m["sensibilidade"], 12),
            round(m["auc"], 12),
            round(m["especificidade"], 12),
        )

    melhor = max(oof_por_modelo, key=_chave_oof)

    # Exibe métricas OOF de cada modelo para verificação
    print("Métricas out-of-fold do treino (no limiar de segurança):")
    for nome in sorted(oof_por_modelo):
        oof = oof_por_modelo[nome]
        yt = np.array(oof["y_true"])
        pr = np.array(oof["proba"])
        limiar = oof["limiar"]
        yp = (pr >= limiar).astype(int)
        m = compute_metrics(yt, yp, pr)
        print(f"  {nome:20s} limiar={limiar:.2f} "
              f"sens={m['sensibilidade']:.4f} "
              f"auc={m['auc']:.4f} "
              f"esp={m['especificidade']:.4f}")

    print(f"\nMelhor modelo selecionado: {melhor}")
    print(f"Critério: {CRITERIO_MELHOR_MODELO}")

    # Atualiza metrics.json
    antigo = metrics.get("melhor_modelo")
    metrics["melhor_modelo"] = melhor
    metrics["criterio_melhor_modelo"] = CRITERIO_MELHOR_MODELO

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # Atualiza feature_importance.json
    if os.path.exists(fi_path):
        with open(fi_path, encoding="utf-8") as f:
            fi = json.load(f)
        fi["melhor_modelo"] = melhor
        with open(fi_path, "w", encoding="utf-8") as f:
            json.dump(fi, f, indent=2, ensure_ascii=False, default=str)

    if antigo != melhor:
        print(f"\nAlteração: melhor_modelo mudou de '{antigo}' para '{melhor}'.")
    else:
        print(f"\nSem alteração: melhor_modelo continua '{melhor}'.")


if __name__ == "__main__":
    main()
