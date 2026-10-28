import React, { useEffect, useRef, useState } from "react";
import { ApiError, createTrainingRun, getComparison, getFeatures, getTrainingRun, putThreshold, URL_EXPORT_CSV } from "../api.js";
import { BarChart, MatrizConfusao, RocChart } from "../components/charts.jsx";
import { Alert, StateBlock } from "../components/ui.jsx";
import { nomeModelo, pct, rotuloCaracteristica } from "../components/format.js";

// Todas as métricas escalares exibidas são do CONJUNTO DE TESTE da execução de
// cada modelo, no limiar da tabela. O portão de promoção (out-of-fold do
// treino) tem coluna própria, para ninguém reportar um número pelo outro.
const COLUNAS = [
  ["accuracy", "Acurácia (teste)"],
  ["sensitivity", "Sensibilidade (teste)"],
  ["specificity", "Especificidade (teste)"],
  ["auc", "AUC (teste)"],
  ["false_negative_rate", "Falsos negativos (teste)"],
];
const num = (v) => (v == null ? "—" : v);
const MODELOS_DISPONIVEIS = [
  ["regressao_logistica", "Regressão logística"],
  ["svm", "SVM (RBF)"],
];
const POLL_MS = 1500;

export default function ModelosPage() {
  const [comparacao, setComparacao] = useState(null);
  const [erro, setErro] = useState(null);
  const [carregando, setCarregando] = useState(true);

  const [modeloRanking, setModeloRanking] = useState(null);
  const [topRanking, setTopRanking] = useState("");
  const [ranking, setRanking] = useState(null);
  const [erroRanking, setErroRanking] = useState(null);

  const [treino, setTreino] = useState({ models: ["regressao_logistica", "svm"], test_size: 0.2, grid_search: true });
  const [run, setRun] = useState(null);
  const [erroTreino, setErroTreino] = useState(null);
  const pollRef = useRef(null);

  const [limiares, setLimiares] = useState({});
  const [promocao, setPromocao] = useState({});

  const carregarComparacao = () => {
    setCarregando(true);
    getComparison()
      .then((d) => {
        setComparacao(d);
        setErro(null);
        setModeloRanking((atual) => atual || d.best_model);
        const iniciais = {};
        d.models.forEach((m) => { if (m.id != null) iniciais[m.id] = m.threshold; });
        setLimiares((atual) => ({ ...iniciais, ...atual }));
      })
      .catch((e) => setErro(e.message))
      .finally(() => setCarregando(false));
  };

  useEffect(() => {
    carregarComparacao();
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!modeloRanking) return;
    setErroRanking(null);
    getFeatures(modeloRanking, topRanking || undefined)
      .then((d) => setRanking(d.ranking))
      .catch((e) => setErroRanking(e.message));
  }, [modeloRanking, topRanking]);

  const dispararTreino = async (e) => {
    e.preventDefault();
    setErroTreino(null);
    setRun(null);
    try {
      const resposta = await createTrainingRun({
        models: treino.models,
        test_size: Number(treino.test_size),
        grid_search: treino.grid_search,
      });
      setRun({ id: resposta.training_run_id, status: resposta.status });
      if (resposta.status === "running") {
        pollRef.current = setInterval(async () => {
          try {
            const atual = await getTrainingRun(resposta.training_run_id);
            setRun(atual);
            if (atual.status !== "running") {
              clearInterval(pollRef.current);
              carregarComparacao();
            }
          } catch {
            clearInterval(pollRef.current);
          }
        }, POLL_MS);
      } else {
        carregarComparacao();
      }
    } catch (err) {
      setErroTreino(err instanceof ApiError ? err : new ApiError(err.message));
    }
  };

  const alternarModeloTreino = (id) => {
    setTreino((t) => ({
      ...t,
      models: t.models.includes(id) ? t.models.filter((m) => m !== id) : [...t.models, id],
    }));
  };

  const aplicarLimiar = async (modelId, nome) => {
    const threshold = limiares[modelId];
    setPromocao((p) => ({ ...p, [modelId]: { status: "enviando" } }));
    try {
      const resposta = await putThreshold(modelId, threshold);
      setPromocao((p) => ({
        ...p,
        [modelId]: {
          status: "ok",
          trainOofSensitivity: resposta.train_oof_sensitivity,
          testSensitivity: resposta.test_metrics?.sensitivity,
        },
      }));
      carregarComparacao();
    } catch (err) {
      if (err instanceof ApiError && err.code === "SENSITIVITY_BELOW_TARGET") {
        setPromocao((p) => ({
          ...p,
          [modelId]: { status: "below_target", obtained: err.details.obtained, required: err.details.required },
        }));
      } else {
        setPromocao((p) => ({ ...p, [modelId]: { status: "error", message: err.message } }));
      }
    }
  };

  if (erro) return <StateBlock kind="error" title="Não foi possível carregar os modelos">{erro}</StateBlock>;
  if (carregando && !comparacao) return <StateBlock kind="loading" title="Carregando laboratório de modelos…" />;
  if (!comparacao) return <StateBlock kind="empty" title="Nenhum modelo treinado ainda" />;

  const nomes = comparacao.models.map((m) => m.name);
  const r = comparacao.reproducibility;

  const celulasMetricas = (bloco) =>
    COLUNAS.map(([k]) => (
      <td key={k} className={k === "false_negative_rate" ? "cell-critical" : ""}>
        {bloco[k] == null ? "—" : k === "auc" ? bloco[k].toFixed(4) : pct(bloco[k], 2)}
      </td>
    ));

  // Ponto de operação: composição do conjunto de teste por execução (cada
  // modelo pode ter sido treinado sobre outra partição) e o portão
  // out-of-fold do treino em coluna separada das métricas de teste.
  const tabelaOperacao = () => (
    <div className="table-scroll">
      <table className="data-table">
        <caption>
          Ponto de operação (limiar vigente) — meta de sensibilidade ≥ {pct(comparacao.sensitivity_target, 0)}.
          Métricas no CONJUNTO DE TESTE da execução de cada modelo, no limiar vigente; a última
          coluna é o portão de promoção, calculado em predições out-of-fold do TREINO — não
          reportar uma pela outra.
        </caption>
        <thead>
          <tr>
            <th>Modelo</th><th>Limiar</th><th>Teste (casos)</th><th>Malignos no teste</th>
            {COLUNAS.map(([, rot]) => <th key={rot}>{rot}</th>)}
            <th>Sensibilidade OOF treino (portão)</th>
          </tr>
        </thead>
        <tbody>
          {comparacao.models.map((m) => (
            <tr key={m.name} className={m.name === comparacao.best_model ? "row-highlight" : ""}>
              <th scope="row">{nomeModelo(m.name)}{m.name === comparacao.best_model && " ★"}</th>
              <td>{m.operating_point.threshold.toFixed(2)}</td>
              <td>{num(m.test_set_size)}</td>
              <td>{num(m.test_set_malignant)}</td>
              {celulasMetricas(m.operating_point)}
              <td>{m.train_oof_sensitivity == null ? "—" : pct(m.train_oof_sensitivity, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  // O limiar padrão só existe para modelos cuja execução ainda está em disco:
  // exibir o limiar padrão da referência ao lado de uma execução com outra
  // partição seria comparar números de conjuntos de teste diferentes.
  const comLimiarPadrao = comparacao.models.filter((m) => m.default_point != null);
  const tabelaLimiarPadrao = () => (
    <div className="table-scroll">
      <table className="data-table">
        <caption>Limiar padrão (0,50) na mesma partição de cada modelo</caption>
        <thead>
          <tr>
            <th>Modelo</th><th>Limiar</th><th>Teste (casos)</th><th>Malignos no teste</th>
            {COLUNAS.map(([, rot]) => <th key={rot}>{rot}</th>)}
          </tr>
        </thead>
        <tbody>
          {comLimiarPadrao.map((m) => (
            <tr key={m.name} className={m.name === comparacao.best_model ? "row-highlight" : ""}>
              <th scope="row">{nomeModelo(m.name)}{m.name === comparacao.best_model && " ★"}</th>
              <td>{m.default_point.threshold.toFixed(2)}</td>
              <td>{num(m.test_set_size)}</td>
              <td>{num(m.test_set_malignant)}</td>
              {celulasMetricas(m.default_point)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <>
      <section className="card">
        <h2>Disparar treino</h2>
        <p className="card-help">
          Executa GridSearchCV (5 folds, scoring=recall) sobre os modelos selecionados e grava as métricas.
        </p>
        <form onSubmit={dispararTreino} className="stack">
          <fieldset>
            <legend className="field-label">Modelos</legend>
            <div className="radiogroup" role="group" aria-label="Modelos a treinar">
              {MODELOS_DISPONIVEIS.map(([id, rotulo]) => (
                <label key={id}>
                  <input type="checkbox" checked={treino.models.includes(id)} onChange={() => alternarModeloTreino(id)} />
                  {rotulo}
                </label>
              ))}
            </div>
          </fieldset>
          <div className="field-row">
            <label className="field" style={{ maxWidth: 180 }}>
              <span className="field-label">Fração de teste</span>
              <input
                type="number" min="0.1" max="0.5" step="0.05" value={treino.test_size}
                onChange={(e) => setTreino({ ...treino, test_size: e.target.value })}
              />
            </label>
            <label className="field field-inline">
              <input
                type="checkbox" checked={treino.grid_search}
                onChange={(e) => setTreino({ ...treino, grid_search: e.target.checked })}
              />
              <span className="field-label">Usar GridSearchCV</span>
            </label>
            <button type="submit" className="btn" disabled={!treino.models.length || run?.status === "running"}>
              {run?.status === "running" ? "Treinando…" : "Disparar treino"}
            </button>
          </div>
        </form>

        {erroTreino && <Alert tone="error" title="Não foi possível disparar o treino">{erroTreino.message}</Alert>}
        {run && (
          <Alert tone={run.status === "success" ? "success" : run.status === "failed" ? "error" : "info"} title={`Execução #${run.id} — ${run.status}`}>
            {run.status === "running" && "Acompanhando status…"}
            {run.status === "failed" && (run.error_message || "Falha não detalhada.")}
            {run.status === "success" && "Treino concluído; métricas atualizadas abaixo."}
          </Alert>
        )}
      </section>

      <section className="card">
        <h2>Comparação dos classificadores</h2>
        <p className="card-help">
          Classe positiva: {comparacao.positive_class}.
          Melhor modelo (★): {nomeModelo(comparacao.best_model)}. {comparacao.best_model_criterion}
        </p>
        {comparacao.test_partitions_mixed && (
          <Alert tone="warning" title="A tabela mistura partições de teste diferentes">
            Os modelos abaixo vêm de execuções com conjuntos de teste distintos, então comparar as
            métricas entre eles não é válido.{" "}
            {comparacao.test_partitions.map((p, i) => (
              <span key={`${p.training_run_id}-${i}`}>
                Execução #{p.training_run_id}: fração de teste {p.test_size == null ? "—" : pct(p.test_size, 0)},{" "}
                {num(p.test_set_size)} casos ({num(p.test_set_malignant)} malignos).{" "}
              </span>
            ))}
          </Alert>
        )}
        {tabelaOperacao()}
        {comLimiarPadrao.length > 0 && tabelaLimiarPadrao()}
        <p className="card-help">
          {comparacao.models.map((m) => (
            <span key={m.name}>
              {nomeModelo(m.name)}: {m.meets_sensitivity_target ? "atende" : "NÃO atende"} a meta no teste.{" "}
            </span>
          ))}
        </p>
      </section>

      <section className="card card-grid">
        <div>
          <h2>Curvas ROC sobrepostas</h2>
          {/* AUC do ponto de operação: independe do limiar, então é a mesma do limiar padrão. */}
          <RocChart series={comparacao.models.map((m) => ({ nome: nomeModelo(m.name), auc: m.operating_point.auc, roc: m.roc_points }))} />
        </div>
        <div>
          <h2>Matrizes de confusão</h2>
          {comparacao.models.map((m) => (
            <div key={m.name}>
              <h3>
                {nomeModelo(m.name)} (limiar {m.operating_point.threshold.toFixed(2)}) — teste da
                execução #{m.training_run_id}: {num(m.test_set_size)} casos, {num(m.test_set_malignant)} malignos
              </h3>
              <MatrizConfusao c={m.operating_point.confusion_matrix} />
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Limiar de operação (regra de segurança: sensibilidade ≥ 0,97)</h2>
        <p className="card-help">
          Mova o controle e aplique: o modelo só é promovido à produção se a sensibilidade recalculada
          no limiar escolhido atingir a meta. O portão é sempre a sensibilidade das predições
          out-of-fold do TREINO — o conjunto de teste nunca escolhe limiar, só reporta as métricas
          depois que ele já foi decidido.
        </p>
        {comparacao.models.filter((m) => m.id != null).map((m) => {
          const estado = promocao[m.id];
          const valor = limiares[m.id] ?? m.threshold;
          return (
            <div key={m.id} className="threshold-control">
              <label htmlFor={`limiar-${m.id}`} className="field-label">
                {nomeModelo(m.name)} {m.is_production && <span className="badge badge-neutral">em produção</span>}
              </label>
              <input
                id={`limiar-${m.id}`}
                type="range"
                min="0" max="1" step="0.01"
                value={valor}
                aria-describedby={`limiar-leitura-${m.id}`}
                onChange={(e) => setLimiares((l) => ({ ...l, [m.id]: Number(e.target.value) }))}
              />
              <div className="threshold-readout" id={`limiar-leitura-${m.id}`}>
                <span>Limiar: <strong>{Number(valor).toFixed(2)}</strong></span>
                <span>Limiar de segurança calculado no treino: <strong>{m.safety_threshold.toFixed(2)}</strong></span>
                <span>
                  Sensibilidade no limiar vigente (portão out-of-fold do treino):{" "}
                  <strong>{m.train_oof_sensitivity == null ? "—" : pct(m.train_oof_sensitivity, 2)}</strong>
                </span>
              </div>
              <div className="btn-group" style={{ marginTop: "var(--space-2)" }}>
                <button type="button" className="btn btn-sm" onClick={() => aplicarLimiar(m.id, m.name)} disabled={estado?.status === "enviando"}>
                  {estado?.status === "enviando" ? "Aplicando…" : "Aplicar e promover"}
                </button>
              </div>
              {estado?.status === "below_target" && (
                <Alert tone="warning" title="Sensibilidade abaixo da meta — modelo NÃO promovido">
                  Sensibilidade obtida: {pct(estado.obtained, 2)}. Exigida: {pct(estado.required, 2)}.
                </Alert>
              )}
              {estado?.status === "ok" && (
                <Alert tone="success" title="Modelo promovido">
                  Portão out-of-fold do treino: {pct(estado.trainOofSensitivity, 2)} (foi o que
                  autorizou a promoção). Sensibilidade recalculada no CONJUNTO DE TESTE no limiar
                  novo: {estado.testSensitivity == null ? "—" : pct(estado.testSensitivity, 2)}.
                </Alert>
              )}
              {estado?.status === "error" && <Alert tone="error" title="Não foi possível aplicar o limiar">{estado.message}</Alert>}
            </div>
          );
        })}
      </section>

      <section className="card">
        <h2>Características mais relevantes</h2>
        <div className="field-row">
          <label className="field" style={{ maxWidth: 240 }}>
            <span className="field-label">Modelo</span>
            <select value={modeloRanking || ""} onChange={(e) => setModeloRanking(e.target.value)}>
              {nomes.map((n) => <option key={n} value={n}>{nomeModelo(n)}{n === comparacao.best_model ? " (melhor)" : ""}</option>)}
            </select>
          </label>
          <label className="field" style={{ maxWidth: 160 }}>
            <span className="field-label">Quantas exibir (top)</span>
            <input type="number" min="1" max="30" placeholder="padrão" value={topRanking} onChange={(e) => setTopRanking(e.target.value)} />
          </label>
        </div>
        <p className="card-help">Importância por permutação: aumento médio da log-loss no teste ao embaralhar a característica.</p>
        {erroRanking && <Alert tone="error" title="Não foi possível carregar o ranking">{erroRanking}</Alert>}
        {ranking && (
          <BarChart
            titulo="Importância das características"
            itens={ranking.map((x) => ({ rotulo: rotuloCaracteristica(x.caracteristica), valor: x.importancia, desvio: x.desvio }))}
          />
        )}
      </section>

      <section className="card">
        <h2>Reprodutibilidade</h2>
        <p className="card-help">
          Este bloco descreve os artefatos de referência (partição 80/20, semente 42) citados no
          artigo — não necessariamente a execução dos modelos listados acima, cuja procedência está
          em “Teste (casos)” e no caminho do artefato de cada um.
        </p>
        <dl className="definition-list">
          <div><dt>Semente</dt><dd>{r.random_state}</dd></div>
          <div><dt>Partição</dt><dd>{r.n_treino} treino / {r.n_teste} teste ({pct(r.test_size, 0)} teste, estratificada)</dd></div>
          <div><dt>Versões</dt><dd>Python {r.versoes.python} · scikit-learn {r.versoes.scikit_learn}</dd></div>
          <div><dt>Commit</dt><dd>{r.commit_git ? r.commit_git.slice(0, 10) : "—"}</dd></div>
        </dl>
        <details>
          <summary>Hiperparâmetros e hashes SHA-256</summary>
          <pre>{JSON.stringify({ hiperparametros: r.hiperparametros, hashes_sha256: r.hashes_sha256 }, null, 2)}</pre>
        </details>
        <a className="btn btn-secondary" href={URL_EXPORT_CSV} download>Exportar tabela (CSV)</a>
      </section>
    </>
  );
}
