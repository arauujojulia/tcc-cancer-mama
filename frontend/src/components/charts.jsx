import React from "react";

const CORES = ["#1d4ed8", "#c2410c"];
const TRACOS = [undefined, "6 4"]; // além da cor, o tracejado distingue as séries

/** Curvas ROC sobrepostas. series: [{nome, auc, roc: {fpr, tpr}}] */
export function RocChart({ series }) {
  const L = 320, M = 36;
  const x = (v) => M + v * (L - M - 8);
  const y = (v) => L - M - v * (L - M - 8);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <figure className="grafico">
      <svg viewBox={`0 0 ${L} ${L + 28 + series.length * 16}`} role="img" aria-label="Curvas ROC dos classificadores">
        {ticks.map((t) => (
          <g key={t} className="grade">
            <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} />
            <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} />
            <text x={M - 6} y={y(t) + 4} textAnchor="end">{t}</text>
            <text x={x(t)} y={y(0) + 14} textAnchor="middle">{t}</text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="diagonal" />
        {series.map((s, i) => (
          <polyline
            key={s.nome}
            fill="none"
            stroke={CORES[i % CORES.length]}
            strokeWidth="2"
            strokeDasharray={TRACOS[i % TRACOS.length]}
            points={s.roc.fpr.map((f, k) => `${x(f)},${y(s.roc.tpr[k])}`).join(" ")}
          />
        ))}
        <text x={x(0.5)} y={L - 6} textAnchor="middle" className="eixo">1 − especificidade</text>
        <text transform={`translate(10 ${y(0.5)}) rotate(-90)`} textAnchor="middle" className="eixo">sensibilidade</text>
        {series.map((s, i) => (
          <g key={s.nome} transform={`translate(${M} ${L + 8 + i * 16})`}>
            <line x1="0" x2="22" y1="-4" y2="-4" stroke={CORES[i % CORES.length]} strokeWidth="2" strokeDasharray={TRACOS[i % TRACOS.length]} />
            <text x="28" y="0" className="legenda">{s.nome} — AUC {s.auc.toFixed(3)}</text>
          </g>
        ))}
      </svg>
    </figure>
  );
}

/** Barras horizontais. itens: [{rotulo, valor, desvio}] */
export function BarChart({ itens, titulo }) {
  const max = Math.max(...itens.map((i) => i.valor + (i.desvio || 0)), 1e-9);
  return (
    <ul className="barras" aria-label={titulo}>
      {itens.map((i) => (
        <li key={i.rotulo}>
          <span className="barra-rotulo">{i.rotulo}</span>
          <span className="barra-trilho">
            <span className="barra" style={{ width: `${Math.max(0, (i.valor / max) * 100)}%` }} />
          </span>
          <span className="barra-valor">{i.valor.toFixed(3)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Matriz de confusão 2×2 (real × previsto). */
export function MatrizConfusao({ c }) {
  return (
    <table className="matriz">
      <thead>
        <tr><th></th><th>Prev. benigno</th><th>Prev. maligno</th></tr>
      </thead>
      <tbody>
        <tr><th>Real benigno</th><td>{c.vn}</td><td>{c.fp}</td></tr>
        <tr><th>Real maligno</th><td className="critico">{c.fn}</td><td>{c.vp}</td></tr>
      </tbody>
    </table>
  );
}
