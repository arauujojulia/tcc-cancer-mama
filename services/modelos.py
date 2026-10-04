"""
Registro dos modelos treinados no banco (US06 — reprodutibilidade).

Lê os artefatos de ml/artifacts/ e grava um ModeloTreinado por classificador,
identificado pelo hash SHA-256 do arquivo .joblib: reexecutar a sincronização
não duplica registros, e um novo treino (novo hash) gera novo registro. O
melhor modelo do treino mais recente fica marcado como `em_producao`.

Executar manualmente:  python -m services.modelos
"""
import json
import os
from datetime import datetime

from db.models import ModeloTreinado, db
from ml.train import ARTIFACTS_DIR


def _ler(artifacts_dir, nome):
    with open(os.path.join(artifacts_dir, nome), encoding="utf-8") as f:
        return json.load(f)


def sincronizar_modelos(artifacts_dir=ARTIFACTS_DIR) -> list[ModeloTreinado]:
    """Cria os registros que faltam e atualiza `em_producao`. Retorna os registros do treino atual."""
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    config_path = os.path.join(artifacts_dir, "train_config.json")
    if not (os.path.exists(metrics_path) and os.path.exists(config_path)):
        return []

    metrics, config = _ler(artifacts_dir, "metrics.json"), _ler(artifacts_dir, "train_config.json")
    treinado_em = datetime.fromisoformat(metrics["gerado_em"]) if metrics.get("gerado_em") else None

    atuais = []
    for nome, r in metrics["modelos"].items():
        sha = config["hashes_sha256"][nome]
        registro = ModeloTreinado.query.filter_by(hash_sha256=sha).first()
        if registro is None:
            registro = ModeloTreinado(
                nome=nome,
                hash_sha256=sha,
                hiperparametros=config["hiperparametros"][nome],
                semente=config["random_state"],
                test_size=config["test_size"],
                n_treino=config["n_treino"],
                n_teste=config["n_teste"],
                limiar_seguranca=r["limiar_seguranca"],
                metricas={k: r[k] for k in ("teste_padrao", "teste_operacao", "validacao_cruzada")},
                versoes=config.get("versoes"),
                commit_git=config.get("commit_git"),
                treinado_em=treinado_em,
            )
            db.session.add(registro)
        atuais.append(registro)

    db.session.flush()
    ids_atuais = {m.id for m in atuais}
    for m in ModeloTreinado.query.all():
        m.em_producao = m.id in ids_atuais and m.nome == metrics["melhor_modelo"]
    db.session.commit()
    return atuais


def modelo_registrado_mais_recente(nome: str) -> ModeloTreinado | None:
    """Registro mais recente do modelo `nome` (liga a Avaliacao ao modelo usado)."""
    return (ModeloTreinado.query.filter_by(nome=nome)
            .order_by(ModeloTreinado.registrado_em.desc(), ModeloTreinado.id.desc()).first())


if __name__ == "__main__":
    from app import create_app

    app = create_app()
    with app.app_context():
        db.create_all()
        for m in sincronizar_modelos():
            print(f"{m.nome}: {m.hash_sha256[:12]}… em_producao={m.em_producao}")
