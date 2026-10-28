// Cliente da API REST (Flask), sob o prefixo /api/v1. Todas as telas
// consomem só esta camada. O cookie de sessão viaja em toda chamada
// (credentials: "include"), inclusive em dev, onde o Vite encaminha /api
// para o Flask na porta 5000 (vite.config.js).

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(mensagem, { status, code, details } = {}) {
    super(mensagem);
    this.status = status;
    this.code = code;
    this.details = details || {};
  }
}

async function request(caminho, opcoes = {}) {
  let resposta;
  try {
    resposta = await fetch(`${BASE}${caminho}`, {
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      ...opcoes,
    });
  } catch {
    throw new ApiError("Não foi possível contatar o servidor. Verifique a conexão e tente novamente.", {
      code: "REDE",
    });
  }
  let corpo = null;
  try {
    corpo = await resposta.json();
  } catch {
    /* resposta sem corpo (ex.: 204) */
  }
  if (!resposta.ok) {
    const { error, code, ...details } = corpo || {};
    throw new ApiError(error || `Erro ${resposta.status}`, { status: resposta.status, code, details });
  }
  return corpo;
}

const qs = (params) => {
  const limpo = Object.fromEntries(
    Object.entries(params || {}).filter(([, v]) => v !== undefined && v !== null && v !== "")
  );
  const s = new URLSearchParams(limpo).toString();
  return s ? `?${s}` : "";
};

// ----------------------------------------------------------------------
// Autenticação
// ----------------------------------------------------------------------
export const register = (dados) => request("/auth/register", { method: "POST", body: JSON.stringify(dados) });
export const login = (dados) => request("/auth/login", { method: "POST", body: JSON.stringify(dados) });
export const logout = () => request("/auth/logout", { method: "POST" });
export const me = () => request("/auth/me");

// ----------------------------------------------------------------------
// Características e classificação
// ----------------------------------------------------------------------
export const getFeatureSchema = () => request("/features/schema");
export const predict = (corpo) => request("/predictions", { method: "POST", body: JSON.stringify(corpo) });

// ----------------------------------------------------------------------
// Histórico
// ----------------------------------------------------------------------
export const getPredictions = (params) => request(`/predictions${qs(params)}`);
export const getPrediction = (id) => request(`/predictions/${id}`);

// ----------------------------------------------------------------------
// Modelos e laboratório
// ----------------------------------------------------------------------
export const getModels = () => request("/models");
export const getComparison = () => request("/models/comparison");
// `top` só é enviado quando o chamador escolhe um valor; omitido, o backend usa o default.
export const getFeatures = (model, top) => request(`/models/features${qs({ model, top })}`);
export const createTrainingRun = (corpo) => request("/training-runs", { method: "POST", body: JSON.stringify(corpo) });
export const getTrainingRun = (id) => request(`/training-runs/${id}`);
export const putThreshold = (modelId, threshold) =>
  request(`/models/${modelId}/threshold`, { method: "PUT", body: JSON.stringify({ threshold }) });
export const URL_EXPORT_CSV = `${BASE}/models/export?format=csv`;
