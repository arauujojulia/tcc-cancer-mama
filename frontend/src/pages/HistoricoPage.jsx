import React, { useEffect, useState } from "react";
import { getAvaliacoes } from "../api.js";
import { dataHora, nomeModelo, pct } from "../components/format.js";

const FILTROS = ["classificacao", "identificado", "data_inicio", "data_fim", "confianca_min", "confianca_max"];

// Os filtros e a página vivem na URL, para retomar e compartilhar a consulta (UC07).
const lerUrl = () => {
  const p = new URLSearchParams(window.location.search);
  const f = Object.fromEntries(FILTROS.map((k) => [k, p.get(k) || ""]));
  return { filtros: f, pagina: Number(p.get("pagina")) || 1 };
};

export default function HistoricoPage() {
  const inicial = lerUrl();
  const [filtros, setFiltros] = useState(inicial.filtros);
  const [pagina, setPagina] = useState(inicial.pagina);
  const [dados, setDados] = useState(null);
  const [erro, setErro] = useState(null);
  const [carregando, setCarregando] = useState(true);

  useEffect(() => {
    const params = Object.fromEntries(Object.entries(filtros).filter(([, v]) => v !== ""));
    params.pagina = pagina;
    params.limite = 15;
    window.history.replaceState(null, "", `?${new URLSearchParams(params)}`);
    let ativo = true;
    setCarregando(true);
    getAvaliacoes(params)
      .then((d) => ativo && (setDados(d), setErro(null)))
      .catch((e) => ativo && setErro(e.message))
      .finally(() => ativo && setCarregando(false));
    return () => { ativo = false; };
  }, [filtros, pagina]);

  const mudar = (campo) => (e) => { setFiltros({ ...filtros, [campo]: e.target.value }); setPagina(1); };
  const limpar = () => { setFiltros(Object.fromEntries(FILTROS.map((k) => [k, ""]))); setPagina(1); };
  const p = dados?.paginacao;

  return (
    <section className="cartao">
      <h2>Histórico de avaliações</h2>
      <div className="filtros">
        <label className="campo"><span>Classificação</span>
          <select value={filtros.classificacao} onChange={mudar("classificacao")}>
            <option value="">Todas</option><option value="maligno">Maligno</option><option value="benigno">Benigno</option>
          </select></label>
        <label className="campo"><span>Registro</span>
          <select value={filtros.identificado} onChange={mudar("identificado")}>
            <option value="">Todos</option><option value="false">Anônimo</option><option value="true">Identificado</option>
          </select></label>
        <label className="campo"><span>De</span><input type="date" value={filtros.data_inicio} onChange={mudar("data_inicio")} /></label>
        <label className="campo"><span>Até</span><input type="date" value={filtros.data_fim} onChange={mudar("data_fim")} /></label>
        <label className="campo"><span>Confiança mín. (0–1)</span><input type="number" min="0" max="1" step="0.05" value={filtros.confianca_min} onChange={mudar("confianca_min")} /></label>
        <label className="campo"><span>Confiança máx. (0–1)</span><input type="number" min="0" max="1" step="0.05" value={filtros.confianca_max} onChange={mudar("confianca_max")} /></label>
        <button type="button" className="secundario" onClick={limpar}>Limpar filtros</button>
      </div>

      {erro && <p className="erro" role="alert">{erro}</p>}
      {carregando && !dados && <p>Carregando…</p>}
      {dados && (
        <>
          <p className="ajuda" aria-live="polite">{p.total} avaliação(ões){carregando ? " — atualizando…" : ""}</p>
          {p.total === 0 ? <p>Nenhuma avaliação encontrada para os filtros atuais.</p> : (
            <div className="rolagem">
              <table>
                <thead><tr><th>#</th><th>Data/hora</th><th>Classificação</th><th>Confiança</th><th>P(maligno)</th><th>Modelo</th><th>Registro</th><th>Tempo</th></tr></thead>
                <tbody>
                  {dados.itens.map((a) => (
                    <tr key={a.id}>
                      <td>{a.id}</td>
                      <td>{dataHora(a.criado_em)}</td>
                      <td><span className={`selo ${a.classificacao}`}>{a.classificacao}</span></td>
                      <td>{pct(a.confianca)}</td>
                      <td>{pct(a.probabilidade_maligno)}</td>
                      <td>{a.modelo_utilizado ? nomeModelo(a.modelo_utilizado) : "—"}</td>
                      <td>{!a.identificado ? "Anônimo" : a.paciente?.indisponivel ? "Identificado (ilegível)" : `${a.paciente?.nome} · ${a.paciente?.prontuario}`}</td>
                      <td>{Math.round(a.tempo_resposta_ms)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="paginacao">
            <button className="secundario" disabled={pagina <= 1} onClick={() => setPagina(pagina - 1)}>Anterior</button>
            <span>Página {p.pagina} de {p.paginas}</span>
            <button className="secundario" disabled={pagina >= p.paginas} onClick={() => setPagina(pagina + 1)}>Próxima</button>
          </div>
        </>
      )}
    </section>
  );
}
