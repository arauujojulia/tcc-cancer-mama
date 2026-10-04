import React, { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, classificar, getSchema } from "../api.js";
import { GRUPOS, MEDIDAS, nomeModelo, pct } from "../components/format.js";

const ORDEM_GRUPOS = ["mean", "se", "worst"];

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
    getSchema().then(setSchema).catch((e) => setErroSchema(e.message));
  }, []);

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
      const caracteristicas = Object.fromEntries(
        schema.campos.map((c) => [c.nome, Number(String(valores[c.nome]).replace(",", "."))])
      );
      const corpo = { caracteristicas, modo };
      if (modo === "identificado") corpo.paciente = { nome: paciente.nome.trim(), prontuario: paciente.prontuario.trim() };
      setResultado(await classificar(corpo));
      setTimeout(() => resultadoRef.current?.focus(), 0);
    } catch (err) {
      // Os dados do formulário são mantidos para uma nova tentativa (BSC-16).
      setErro(err instanceof ApiError ? err : new ApiError(err.message));
    } finally {
      setEnviando(false);
    }
  };

  if (erroSchema) return <p className="erro" role="alert">Não foi possível carregar o formulário: {erroSchema}</p>;
  if (!schema) return <p>Carregando formulário…</p>;

  return (
    <form onSubmit={enviar} noValidate>
      <section className="cartao">
        <h2>Características da biópsia</h2>
        <p className="ajuda">
          Informe as 30 medidas quantitativas extraídas da punção aspirativa (padrão WDBC).
        </p>
        <div className="acoes">
          <button type="button" className="secundario" onClick={() => preencher(schema.exemplos.maligno)}>
            Exemplo da base (maligno)
          </button>
          <button type="button" className="secundario" onClick={() => preencher(schema.exemplos.benigno)}>
            Exemplo da base (benigno)
          </button>
          <label className="botao secundario">
            Importar CSV
            <input type="file" accept=".csv,.txt" onChange={importarCsv} hidden />
          </label>
          <button type="button" className="secundario" onClick={() => preencher({})}>Limpar</button>
        </div>
        {avisoCsv && <p className="ajuda" role="status">{avisoCsv}</p>}

        {ORDEM_GRUPOS.map((g) => (
          <fieldset key={g}>
            <legend>{GRUPOS[g]}</legend>
            <div className="grade-campos">
              {schema.campos.filter((c) => c.grupo === g).map((c) => {
                const msg = tocados[c.nome] ? erros[c.nome] : null;
                return (
                  <label key={c.nome} className={msg ? "campo com-erro" : "campo"}>
                    <span>{MEDIDAS[c.medida] || c.medida}</span>
                    <input
                      inputMode="decimal"
                      value={valores[c.nome] ?? ""}
                      onChange={(e) => setValores({ ...valores, [c.nome]: e.target.value })}
                      onBlur={() => setTocados({ ...tocados, [c.nome]: true })}
                      aria-invalid={msg ? "true" : undefined}
                      aria-describedby={msg ? `e-${c.nome}` : undefined}
                    />
                    {msg && <small id={`e-${c.nome}`}>{msg}</small>}
                  </label>
                );
              })}
            </div>
          </fieldset>
        ))}
      </section>

      <section className="cartao">
        <h2>Registro da avaliação</h2>
        <div className="radios" role="radiogroup" aria-label="Modo de registro">
          <label><input type="radio" name="modo" checked={modo === "anonimo"} onChange={() => setModo("anonimo")} /> Anônimo</label>
          <label><input type="radio" name="modo" checked={modo === "identificado"} onChange={() => setModo("identificado")} /> Identificado</label>
        </div>
        {modo === "identificado" ? (
          <div className="grade-campos duas">
            <label className="campo"><span>Nome da paciente</span>
              <input value={paciente.nome} onChange={(e) => setPaciente({ ...paciente, nome: e.target.value })} maxLength={200} />
            </label>
            <label className="campo"><span>Prontuário</span>
              <input value={paciente.prontuario} onChange={(e) => setPaciente({ ...paciente, prontuario: e.target.value })} maxLength={100} />
            </label>
            <p className="ajuda">Nome e prontuário são armazenados cifrados (AES-256-GCM), conforme a LGPD.</p>
          </div>
        ) : (
          <p className="ajuda">Nenhum dado da paciente é armazenado; apenas as características e o resultado.</p>
        )}
        <button type="submit" disabled={enviando || !pacienteOk}>{enviando ? "Classificando…" : "Classificar tumor"}</button>
        {tocados && Object.keys(tocados).length > 0 && invalidos > 0 && (
          <p className="erro" role="alert">{invalidos} campo(s) precisam de correção.</p>
        )}
      </section>

      {erro && (
        <section className="cartao erro" role="alert">
          <h2>Não foi possível classificar</h2>
          <p>{erro.message}</p>
          {erro.codigo === "MODEL_UNAVAILABLE" && <p>Seus dados foram mantidos; tente novamente em instantes.</p>}
          {erro.detalhes.campos_fora_da_faixa && (
            <ul>{erro.detalhes.campos_fora_da_faixa.map((c) => <li key={c.campo}>{c.campo}: {c.valor}</li>)}</ul>
          )}
        </section>
      )}

      {resultado && (
        <section className={`cartao resultado ${resultado.classificacao}`} tabIndex={-1} ref={resultadoRef} aria-live="polite">
          <h2>Resultado</h2>
          <p className="veredito">Classificação: <strong>{resultado.classificacao === "maligno" ? "Maligno" : "Benigno"}</strong></p>
          <dl>
            <div><dt>Grau de confiança</dt><dd>{pct(resultado.confianca)} <span className={`faixa ${resultado.faixa_confianca}`}>confiança {resultado.faixa_confianca === "media" ? "média" : resultado.faixa_confianca}</span></dd></div>
            <div><dt>Probabilidade de malignidade</dt><dd>{pct(resultado.probabilidade_maligno)}</dd></div>
            <div><dt>Modelo</dt><dd>{nomeModelo(resultado.modelo_utilizado)} (limiar {resultado.limiar_utilizado.toFixed(2)})</dd></div>
            <div><dt>Tempo de resposta</dt><dd>{resultado.tempo_ms.toFixed(0)} ms</dd></div>
            <div><dt>Registro</dt><dd>#{resultado.avaliacao_id} ({resultado.identificado ? "identificado" : "anônimo"})</dd></div>
          </dl>
          {resultado.acionou_limiar_seguranca && (
            <p className="nota">
              Classificado como maligno pelo limiar de segurança (abaixo de 50%), adotado para reduzir
              falsos negativos. Recomenda-se atenção especial a este caso.
            </p>
          )}
          <p className="nota">{resultado.aviso}</p>
        </section>
      )}
    </form>
  );
}
