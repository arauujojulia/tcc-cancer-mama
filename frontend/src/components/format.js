export const pct = (v, casas = 1) => `${(v * 100).toFixed(casas)}%`;

export const NOMES_MODELO = { regressao_logistica: "Regressão logística", svm: "SVM (RBF)" };
export const nomeModelo = (id) => NOMES_MODELO[id] || id;

export const MEDIDAS = {
  radius: "Raio", texture: "Textura", perimeter: "Perímetro", area: "Área",
  smoothness: "Suavidade", compactness: "Compacidade", concavity: "Concavidade",
  concave_points: "Pontos côncavos", symmetry: "Simetria", fractal_dimension: "Dimensão fractal",
};
export const GRUPOS = { mean: "Média", se: "Erro padrão", worst: "Pior valor" };

export const rotuloCaracteristica = (nome) => {
  const [grupo, ...resto] = nome.split("_");
  return `${MEDIDAS[resto.join("_")] || resto.join("_")} (${(GRUPOS[grupo] || grupo).toLowerCase()})`;
};

export const dataHora = (iso) =>
  new Date(iso).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "medium" });
