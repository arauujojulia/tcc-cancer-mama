import React, { useEffect, useState } from "react";
import { getImportancia, getMetricas, URL_EXPORT_CSV } from "../api.js";
import { BarChart, MatrizConfusao, RocChart } from "../components/charts.jsx";
import { nomeModelo, pct, rotuloCaracteristica } from "../components/format.js";

const COLUNAS = [
  ["acuracia", "Acurácia"],
  ["sensibilidade", "Sensibilidade"],
  ["especificidade", "Especificidade"],
  ["auc", "AUC"],
  ["taxa_falsos_negativos", "Falsos negativos"],
];

export default function ModelosPage() {
  const [m, setM] = useState(null);
  const [erro, setErro] = useState(null);
  const [modeloRanking, setModeloRanking] = useState(null);
  const [ranking, setRanking] = useState(null);

  useEffect(() => {
    getMetricas().then((d) => { setM(d); setModeloRanking(d.melhor_modelo); }).catch((e) => setErro(e.message));
  }, []);
  useEffect(() => {
    if (modeloRanking) getImportancia(modeloRanking).then((d) => setRanking(d.ranking.slice(0, 15))).catch((e) => setErro(e.message));
  }, [modeloRanking]);

  if (erro) return <p className="erro" role="alert">{erro}</p>;
  if (!m) return <p>Carregando métricas…</p>;

  const nomes = Object.keys(m.modelos);
  const tabela = (chave, titulo, limiarKey) => (
    <div className="rolagem">
      <table>
        <caption>{titulo}</caption>
        <thead><tr><th>Modelo</th><th>Limiar</th>{COLUNAS.map(([, r]) => <th key={r}>{r}</th>)}</tr></thead>
        <tbody>
          {nomes.map((n) => (
            <tr key={n} className={n === m.melhor_modelo ? "destaque" : ""}>
              <th scope="row">{nomeModelo(n)}{n === m.melhor_modelo && " ★"}</th>
              <td>{m.modelos[n][limiarKey].toFixed(2)}</td>
              {COLUNAS.map(([k]) => (
                <td key={k} className={k === "taxa_falsos_negativos" ? "critico" : ""}>
                  {k === "auc" ? m.modelos[n][chave][k].toFixed(4) : pct(m.modelos[n][chave][k], 2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
  const r = m.reprodutibilidade;

  return (
    <>
      <section className="cartao">
        <h2>Comparação dos classificadores</h2>
        <p className="ajuda">
          Conjunto de teste ({r.n_teste} casos, {r.distribuicao_teste.maligno} malignos). Classe positiva: {m.classe_positiva}.
          Melhor modelo (★): {nomeModelo(m.melhor_modelo)}. {m.criterio_melhor_modelo}
        </p>
        {tabela("teste_operacao", `Ponto de operação (limiar de segurança) — meta de sensibilidade ≥ ${pct(m.meta_sensibilidade, 0)}`, "limiar_seguranca")}
        {tabela("teste_padrao", "Limiar padrão (0,50), para comparação", "limiar_padrao")}
        <p className="ajuda">
          {nomes.map((n) => (
            <span key={n}>
              {nomeModelo(n)}: {m.modelos[n].atende_meta_sensibilidade_teste ? "atende" : "NÃO atende"} a meta no teste.{" "}
            </span>
          ))}
        </p>
        <div className="rolagem">
          <table>
            <caption>Validação cruzada estratificada ({r.cv_folds} folds) no treino — média ± desvio</caption>
            <thead><tr><th>Modelo</th>{COLUNAS.map(([, rot]) => <th key={rot}>{rot}</th>)}</tr></thead>
            <tbody>
              {nomes.map((n) => (
                <tr key={n}>
                  <th scope="row">{nomeModelo(n)}</th>
                  {COLUNAS.map(([k]) => {
                    const c = m.modelos[n].validacao_cruzada[k];
                    return <td key={k}>{pct(c.media, 2)} ± {pct(c.desvio, 2)}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="cartao dois-cols">
        <div>
          <h2>Curvas ROC</h2>
          <RocChart series={nomes.map((n) => ({ nome: nomeModelo(n), auc: m.modelos[n].teste_padrao.auc, roc: m.modelos[n].roc }))} />
        </div>
        <div>
          <h2>Matrizes de confusão</h2>
          {nomes.map((n) => (
            <div key={n}>
              <h3>{nomeModelo(n)} (limiar {m.modelos[n].limiar_seguranca.toFixed(2)})</h3>
              <MatrizConfusao c={m.modelos[n].teste_operacao.matriz_confusao} />
            </div>
          ))}
        </div>
      </section>

      <section className="cartao">
        <h2>Características mais relevantes</h2>
        <label className="campo inline"><span>Modelo</span>
          <select value={modeloRanking} onChange={(e) => setModeloRanking(e.target.value)}>
            {nomes.map((n) => <option key={n} value={n}>{nomeModelo(n)}{n === m.melhor_modelo ? " (melhor)" : ""}</option>)}
          </select>
        </label>
        <p className="ajuda">Importância por permutação: aumento médio da log-loss no teste ao embaralhar a característica.</p>
        {ranking && <BarChart titulo="Importância das características" itens={ranking.map((x) => ({ rotulo: rotuloCaracteristica(x.caracteristica), valor: x.importancia, desvio: x.desvio }))} />}
      </section>

      <section className="cartao">
        <h2>Reprodutibilidade</h2>
        <dl className="duas">
          <div><dt>Semente</dt><dd>{r.random_state}</dd></div>
          <div><dt>Partição</dt><dd>{r.n_treino} treino / {r.n_teste} teste ({pct(r.test_size, 0)} teste, estratificada)</dd></div>
          <div><dt>Versões</dt><dd>Python {r.versoes.python} · scikit-learn {r.versoes.scikit_learn}</dd></div>
          <div><dt>Commit</dt><dd>{r.commit_git ? r.commit_git.slice(0, 10) : "—"}</dd></div>
        </dl>
        <details>
          <summary>Hiperparâmetros e hashes SHA-256</summary>
          <pre>{JSON.stringify({ hiperparametros: r.hiperparametros, hashes_sha256: r.hashes_sha256 }, null, 2)}</pre>
        </details>
        <a className="botao" href={URL_EXPORT_CSV} download>Exportar tabela (CSV)</a>
      </section>
    </>
  );
}
