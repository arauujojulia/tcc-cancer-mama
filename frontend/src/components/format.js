export const pct = (v, casas = 1) => `${(v * 100).toFixed(casas)}%`;

export const NOMES_MODELO = { regressao_logistica: "Regressão logística", svm: "SVM (RBF)" };
export const nomeModelo = (id) => NOMES_MODELO[id] || id;

// As 10 medidas base do WDBC, na ordem em que aparecem no formulário.
export const MEDIDAS_BASE = [
  "radius", "texture", "perimeter", "area", "smoothness",
  "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
];
export const MEDIDAS = {
  radius: "Raio", texture: "Textura", perimeter: "Perímetro", area: "Área",
  smoothness: "Suavidade", compactness: "Compacidade", concavity: "Concavidade",
  concave_points: "Pontos côncavos", symmetry: "Simetria", fractal_dimension: "Dimensão fractal",
};
// As 3 variações estatísticas, na ordem exibida lado a lado dentro de cada grupo.
export const VARIACOES = ["mean", "se", "worst"];
export const GRUPOS = { mean: "Média", se: "Erro padrão", worst: "Pior valor" };

export const rotuloCaracteristica = (nome) => {
  const [grupo, ...resto] = nome.split("_");
  return `${MEDIDAS[resto.join("_")] || resto.join("_")} (${(GRUPOS[grupo] || grupo).toLowerCase()})`;
};

/**
 * Agrupa os 30 campos do schema pelas 10 medidas base, cada uma com os 3
 * campos (mean/se/worst) na mesma ordem de VARIACOES — para o formulário
 * usar um fieldset por medida, com as 3 variações lado a lado, em vez de
 * uma lista corrida de 30 inputs ou 3 fieldsets por variação.
 */
export const agruparPorMedidaBase = (campos) => {
  const porNome = Object.fromEntries(campos.map((c) => [c.nome, c]));
  return MEDIDAS_BASE.map((medida) => ({
    medida,
    label: MEDIDAS[medida],
    campos: VARIACOES.map((variacao) => porNome[`${variacao}_${medida}`]).filter(Boolean),
  })).filter((g) => g.campos.length > 0);
};

// A API usa a chave sem acento ("media"); a exibição usa "média" acentuado.
export const FAIXAS_CONFIANCA = { alta: "alta", media: "média", baixa: "baixa" };
export const rotuloFaixaConfianca = (faixa) => FAIXAS_CONFIANCA[faixa] || faixa;

export const dataHora = (iso) =>
  new Date(iso).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "medium" });
