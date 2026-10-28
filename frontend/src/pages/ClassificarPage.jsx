import React, { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getFeatureSchema, predict } from "../api.js";
import { Alert, ResultBadge, StateBlock } from "../components/ui.jsx";
import { GRUPOS, VARIACOES, agruparPorMedidaBase, nomeModelo, pct, rotuloFaixaConfianca } from "../components/format.js";

/** Lê um CSV (UC01 / BSC-13): cabeçalho com os nomes + 1 linha, ou 30 valores em sequência. */
export function parseCsv(texto, nomes) {
  const linhas = texto.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!linhas.length) throw new Error("Arquivo vazio.");
  const celulas = (l) => l.split(/[;,\t]/).map((c) => c.trim());
  const primeira = celulas(linhas[0]);
  if (primeira.some((c) => nomes.includes(c))) {
    if (linhas.length < 2) throw new Error("O CSV tem cabeçalho mas nenhuma linha de dados.");
    const dados = celulas(linhas[1]);
    const out = {};
    nomes.forEach((n) => {
      const i = primeira.indexOf(n);
      if (i < 0) throw new Error(`Coluna ausente no CSV: ${n}`);
      out[n] = dados[i];
    });
    return out;
  }
  let valores = linhas.length === 1 ? primeira : linhas.map((l) => celulas(l)[0]);
  if (valores.length === nomes.length + 2) valores = valores.slice(2); // formato original do WDBC: id,diagnóstico,...
  if (valores.length !== nomes.length) throw new Error(`Esperados ${nomes.length} valores; encontrados ${valores.length}.`);
  return Object.fromEntries(nomes.map((n, i) => [n, valores[i]]));
}

function validarCampo(valor, campo) {
  if (valor === undefined || String(valor).trim() === "") return "Obrigatório";
  const v = Number(String(valor).replace(",", "."));
  if (!Number.isFinite(v)) return "Valor numérico inválido";
  // Margem de 20% já embutida em min_aceito/max_aceito do schema.
  if (v < campo.min_aceito || v > campo.max_aceito)
    return `Fora da faixa esperada (${campo.min_aceito.toPrecision(3)} a ${campo.max_aceito.toPrecision(3)})`;
  return null;
}

export default function ClassificarPage() {
  const [schema, setSchema] = useState(null);
  const [erroSchema, setErroSchema] = useState(null);
  const [valores, setValores] = useState({});
  const [tocados, setTocados] = useState({});
  const [modo, setModo] = useState("anonimo");
  const [paciente, setPaciente] = useState({ nome: "", prontuario: "" });
  const [enviando, setEnviando] = useState(false);
  const [resultado, setResultado] = useState(null);
  const [erro, setErro] = useState(null);
  const [avisoCsv, setAvisoCsv] = useState(null);
  const resultadoRef = useRef(null);

  useEffect(() => {
    getFeatureSchema().then(setSchema).catch((e) => setErroSchema(e.message));
  }, []);

  const grupos = useMemo(() => (schema ? agruparPorMedidaBase(schema.campos) : []), [schema]);
  const erros = useMemo(() => {
    if (!schema) return {};
    return Object.fromEntries(schema.campos.map((c) => [c.nome, validarCampo(valores[c.nome], c)]));
  }, [schema, valores]);
  const invalidos = Object.values(erros).filter(Boolean).length;
  const pacienteOk = modo === "anonimo" || (paciente.nome.trim() && paciente.prontuario.trim());

  const preencher = (dados) => {
    setValores(Object.fromEntries(Object.entries(dados).map(([k, v]) => [k, String(v)])));
    setTocados({});
    setResultado(null);
    setErro(null);
  };

  const importarCsv = async (e) => {
    const arquivo = e.target.files?.[0];
    e.target.value = "";
    if (!arquivo) return;
    try {
      preencher(parseCsv(await arquivo.text(), schema.campos.map((c) => c.nome)));
      setAvisoCsv(`Valores importados de ${arquivo.name}. Confira antes de classificar.`);
    } catch (err) {
      setAvisoCsv(`Não foi possível importar: ${err.message}`);
    }
  };

  const enviar = async (e) => {
    e.preventDefault();
    setTocados(Object.fromEntries(schema.campos.map((c) => [c.nome, true])));
    if (invalidos || !pacienteOk) return;
    setEnviando(true);
    setErro(null);
    setResultado(null);
    try {
      const features = Object.fromEntries(
        schema.campos.map((c) => [c.nome, Number(String(valores[c.nome]).replace(",", "."))])
      );
      const corpo = { features, mode: modo };
      if (modo === "identificado") {
        corpo.patient_name = paciente.nome.trim();
        corpo.patient_record = paciente.prontuario.trim();
      }
      const resposta = await predict(corpo);
      setResultado(resposta);
      setTimeout(() => resultadoRef.current?.focus(), 0);
    } catch (err) {
      // BSC-16 / RNF — erro de classificação (503/504) NÃO limpa o formulário.
      setErro(err instanceof ApiError ? err : new ApiError(err.message));
    } finally {
      setEnviando(false);
    }
  };

  if (erroSchema) {
    return (
      <section className="card">
        <StateBlock kind="error" title="Não foi possível carregar o formulário">{erroSchema}</StateBlock>
      </section>
    );
  }
  if (!schema) {
    return (
      <section className="card">
        <StateBlock kind="loading" title="Carregando formulário…" />
      </section>
    );
  }

  return (
    <form onSubmit={enviar} noValidate>
      <section className="card">
        <h2>Características da biópsia</h2>
        <p className="card-help">
          Informe as 30 medidas quantitativas extraídas da punção aspirativa por agulha fina (padrão WDBC),
          agrupadas pelas 10 medidas base. Cada medida tem três variações: média, erro padrão e pior valor.
        </p>
        <div className="btn-group" style={{ marginBottom: "var(--space-4)" }}>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => preencher(schema.exemplos.maligno)}>
            Exemplo da base (maligno)
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => preencher(schema.exemplos.benigno)}>
            Exemplo da base (benigno)
          </button>
          <label className="btn btn-secondary btn-sm">
            Importar CSV
            <input type="file" accept=".csv,.txt" onChange={importarCsv} hidden />
          </label>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => preencher({})}>Limpar</button>
        </div>
        {avisoCsv && <p className="card-help" role="status">{avisoCsv}</p>}

        {grupos.map((g) => (
          <fieldset key={g.medida} className="feature-group">
            <legend>{g.label}</legend>
            <div className="feature-group-grid">
              {g.campos.map((c) => {
                const msg = tocados[c.nome] ? erros[c.nome] : null;
                return (
                  <label key={c.nome} className={`field ${msg ? "has-error" : ""}`}>
                    <span className="field-label">{GRUPOS[c.grupo]}</span>
                    <input
                      inputMode="decimal"
                      value={valores[c.nome] ?? ""}
                      onChange={(e) => setValores({ ...valores, [c.nome]: e.target.value })}
                      onBlur={() => setTocados({ ...tocados, [c.nome]: true })}
                      aria-invalid={msg ? "true" : undefined}
                      aria-describedby={msg ? `e-${c.nome}` : undefined}
                    />
                    {msg && <span className="field-error" id={`e-${c.nome}`}>{msg}</span>}
                  </label>
                );
              })}
            </div>
          </fieldset>
        ))}
      </section>

      <section className="card">
        <h2>Registro da avaliação</h2>
        <div className="radiogroup" role="radiogroup" aria-label="Modo de registro">
          <label><input type="radio" name="modo" checked={modo === "anonimo"} onChange={() => setModo("anonimo")} /> Anônimo</label>
          <label><input type="radio" name="modo" checked={modo === "identificado"} onChange={() => setModo("identificado")} /> Identificado</label>
        </div>
        {modo === "identificado" ? (
          <div className="features-grid">
            <label className="field">
              <span className="field-label">Nome da paciente</span>
              <input value={paciente.nome} onChange={(e) => setPaciente({ ...paciente, nome: e.target.value })} maxLength={200} />
            </label>
            <label className="field">
              <span className="field-label">Prontuário</span>
              <input value={paciente.prontuario} onChange={(e) => setPaciente({ ...paciente, prontuario: e.target.value })} maxLength={100} />
            </label>
            <div style={{ gridColumn: "1 / -1" }}>
              <Alert tone="warning" title="Sem criptografia">
                No nome e no prontuário informados, os dados são gravados em texto no banco de dados,
                sem criptografia. Não use esta ferramenta fora de um ambiente controlado.
              </Alert>
            </div>
          </div>
        ) : (
          <p className="card-help">Nenhum dado da paciente é armazenado; apenas as características e o resultado.</p>
        )}
        <button type="submit" className="btn" disabled={enviando || !pacienteOk}>
          {enviando ? "Classificando…" : "Classificar tumor"}
        </button>
        {Object.keys(tocados).length > 0 && invalidos > 0 && (
          <p className="field-error" role="alert" style={{ marginTop: "var(--space-3)" }}>
            {invalidos} campo(s) precisam de correção.
          </p>
        )}
      </section>

      {erro && (
        <section className="card">
          <Alert tone="error" title="Não foi possível classificar">
            <p className="mb-0">{erro.message}</p>
            {erro.code === "MODEL_UNAVAILABLE" && <p className="mb-0">Seus dados foram mantidos; tente novamente em instantes.</p>}
            {erro.code === "TIMEOUT" && <p className="mb-0">Seus dados foram mantidos; tente novamente.</p>}
            {erro.details?.campos_fora_da_faixa && (
              <ul>{erro.details.campos_fora_da_faixa.map((c) => <li key={c.campo}>{c.campo}: {c.valor}</li>)}</ul>
            )}
          </Alert>
        </section>
      )}

      {resultado && (
        <section
          className={`card result-panel ${resultado.label === "maligno" ? "result-malignant" : "result-benign"}`}
          tabIndex={-1}
          ref={resultadoRef}
          aria-live="polite"
        >
          <h2>Resultado</h2>
          <p className="result-verdict">
            Classificação:
            <ResultBadge
              kind={resultado.label === "maligno" ? "malignant" : "benign"}
              label={resultado.label === "maligno" ? "Maligno" : "Benigno"}
            />
          </p>
          <dl className="definition-list">
            <div>
              <dt>Grau de confiança</dt>
              <dd>
                {pct(resultado.confidence)}
                <span className="confidence-tag">confiança {rotuloFaixaConfianca(resultado.confidence_band)}</span>
              </dd>
            </div>
            <div><dt>Probabilidade de malignidade</dt><dd>{pct(resultado.probability)}</dd></div>
            <div><dt>Modelo</dt><dd>{nomeModelo(resultado.model_name)} (limiar {resultado.threshold_used.toFixed(2)})</dd></div>
            <div><dt>Tempo de resposta</dt><dd>{Math.round(resultado.elapsed_ms)} ms</dd></div>
            <div><dt>Registro</dt><dd>#{resultado.evaluation_id} ({modo === "identificado" ? "identificado" : "anônimo"})</dd></div>
          </dl>
          {resultado.safety_threshold_triggered && (
            <p className="card-help">
              Classificado como maligno pelo limiar de segurança (abaixo de 50%), adotado para reduzir
              falsos negativos. Recomenda-se atenção especial a este caso.
            </p>
          )}
          <p className="card-help mb-0">{resultado.disclaimer}</p>
        </section>
      )}
    </form>
  );
}
