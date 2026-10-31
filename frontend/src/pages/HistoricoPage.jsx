import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { getPredictions } from "../api.js";
import { ResultBadge, StateBlock } from "../components/ui.jsx";
import { dataHora, nomeModelo, pct } from "../components/format.js";

// Nomes em inglês do contrato (design.md §7.4): refletidos na URL via useSearchParams.
const FILTROS = ["label", "mode", "date_from", "date_to", "confidence_min", "confidence_max"];
const LIMITE = 15;

export default function HistoricoPage() {
  const [params, setParams] = useSearchParams();
  const [dados, setDados] = useState(null);
  const [erro, setErro] = useState(null);
  const [carregando, setCarregando] = useState(true);

  const filtros = Object.fromEntries(FILTROS.map((k) => [k, params.get(k) || ""]));
  const page = Number(params.get("page")) || 1;

  useEffect(() => {
    const query = Object.fromEntries(Object.entries(filtros).filter(([, v]) => v !== ""));
    query.page = page;
    query.limit = LIMITE;
    let ativo = true;
    setCarregando(true);
    getPredictions(query)
      .then((d) => ativo && (setDados(d), setErro(null)))
      .catch((e) => ativo && setErro(e.message))
      .finally(() => ativo && setCarregando(false));
    return () => { ativo = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.toString()]);

  const atualizarUrl = (novosFiltros, novaPagina) => {
    const query = Object.fromEntries(Object.entries(novosFiltros).filter(([, v]) => v !== ""));
    if (novaPagina > 1) query.page = novaPagina;
    setParams(query);
  };

  const mudar = (campo) => (e) => atualizarUrl({ ...filtros, [campo]: e.target.value }, 1);
  const limpar = () => setParams({});
  const irPara = (novaPagina) => atualizarUrl(filtros, novaPagina);

  const totalPaginas = dados ? Math.max(1, Math.ceil(dados.total / dados.limit)) : 1;

  return (
    <section className="card">
      <h2>Histórico de avaliações</h2>

      <div className="field-row" role="group" aria-label="Filtros do histórico">
        <label className="field">
          <span className="field-label">Classificação</span>
          <select value={filtros.label} onChange={mudar("label")}>
            <option value="">Todas</option>
            <option value="maligno">Maligno</option>
            <option value="benigno">Benigno</option>
          </select>
        </label>
        <label className="field">
          <span className="field-label">Registro</span>
          <select value={filtros.mode} onChange={mudar("mode")}>
            <option value="">Todos</option>
            <option value="anonimo">Anônimo</option>
            <option value="identificado">Identificado</option>
          </select>
        </label>
        <label className="field">
          <span className="field-label">De</span>
          <input type="date" value={filtros.date_from} onChange={mudar("date_from")} />
        </label>
        <label className="field">
          <span className="field-label">Até</span>
          <input type="date" value={filtros.date_to} onChange={mudar("date_to")} />
        </label>
        <label className="field">
          <span className="field-label">Confiança mín. (0–1)</span>
          <input type="number" min="0" max="1" step="0.05" value={filtros.confidence_min} onChange={mudar("confidence_min")} />
        </label>
        <label className="field">
          <span className="field-label">Confiança máx. (0–1)</span>
          <input type="number" min="0" max="1" step="0.05" value={filtros.confidence_max} onChange={mudar("confidence_max")} />
        </label>
        <button type="button" className="btn btn-secondary" onClick={limpar}>Limpar filtros</button>
      </div>

      {erro && <StateBlock kind="error" title="Não foi possível carregar o histórico">{erro}</StateBlock>}
      {carregando && !dados && !erro && <StateBlock kind="loading" title="Carregando histórico…" />}

      {dados && !erro && (
        <>
          <p className="card-help" aria-live="polite">
            {dados.total} avaliação(ões){carregando ? " — atualizando…" : ""}
          </p>
          {dados.total === 0 ? (
            <StateBlock kind="empty" title="Nenhuma avaliação encontrada">
              Ajuste os filtros ou registre uma nova classificação.
            </StateBlock>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th><th>Data/hora</th><th>Classificação</th><th>Confiança</th>
                    <th>P(maligno)</th><th>Modelo</th><th>Registro</th><th>Tempo</th>
                  </tr>
                </thead>
                <tbody>
                  {dados.items.map((a) => (
                    <tr key={a.id}>
                      <td>{a.id}</td>
                      <td>{dataHora(a.created_at)}</td>
                      <td>
                        <ResultBadge
                          kind={a.label === "maligno" ? "malignant" : "benign"}
                          label={a.label === "maligno" ? "Maligno" : "Benigno"}
                        />
                      </td>
                      <td>{pct(a.confidence)}</td>
                      <td>{pct(a.probability)}</td>
                      <td>{a.model_name ? nomeModelo(a.model_name) : "—"}</td>
                      <td>
                        {a.mode === "anonimo"
                          ? "Anônimo"
                          : a.patient_name == null && a.patient_record == null
                            ? "Identificado (oculto)"
                            : `${a.patient_name} · ${a.patient_record}`}
                      </td>
                      <td>{Math.round(a.elapsed_ms)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="pagination">
            <button className="btn btn-secondary btn-sm" disabled={page <= 1} onClick={() => irPara(page - 1)}>Anterior</button>
            <span>Página {page} de {totalPaginas}</span>
            <button className="btn btn-secondary btn-sm" disabled={page >= totalPaginas} onClick={() => irPara(page + 1)}>Próxima</button>
          </div>
        </>
      )}
    </section>
  );
}
