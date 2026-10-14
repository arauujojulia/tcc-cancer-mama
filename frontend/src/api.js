// Cliente da API REST (Flask). Todas as telas consomem só esta camada.

export class ApiError extends Error {
  constructor(mensagem, { status, codigo, detalhes } = {}) {
    super(mensagem);
    this.status = status;
    this.codigo = codigo;
    this.detalhes = detalhes || {};
  }
}

async function request(caminho, opcoes = {}) {
  let resposta;
  try {
    resposta = await fetch(`/api${caminho}`, {
      credentials: "same-origin", // envia o cookie de sessão (HttpOnly)
      headers: { "Content-Type": "application/json" },
      ...opcoes,
    });
  } catch {
    throw new ApiError("Não foi possível contatar o servidor. Verifique a conexão e tente novamente.", {
      codigo: "REDE",
    });
  }
  let corpo = null;
  try {
    corpo = await resposta.json();
  } catch {
    /* resposta sem JSON */
  }
  if (!resposta.ok) {
    const { erro, codigo, ...detalhes } = corpo || {};
    // Sessão expirada ou encerrada no servidor: o App volta para a tela de login.
    if (resposta.status === 401 && codigo === "NAO_AUTENTICADO") {
      window.dispatchEvent(new Event("sessao-expirada"));
    }
    throw new ApiError(erro || `Erro ${resposta.status}`, { status: resposta.status, codigo, detalhes });
  }
  return corpo;
}

export const getUsuarioAtual = () => request("/auth/me");
export const entrar = (email, senha) => request("/auth/login", { method: "POST", body: JSON.stringify({ email, senha }) });
export const sair = () => request("/auth/logout", { method: "POST", body: "{}" });
export const registrar = (dados) => request("/auth/registrar", { method: "POST", body: JSON.stringify(dados) });
export const getSchema = () => request("/features/schema");
export const classificar = (corpo) => request("/biopsias/classificar", { method: "POST", body: JSON.stringify(corpo) });
export const getAvaliacoes = (params) => request(`/avaliacoes?${new URLSearchParams(params)}`);
export const getMetricas = () => request("/metricas");
export const getImportancia = (modelo) =>
  request(`/metricas/features${modelo ? `?modelo=${encodeURIComponent(modelo)}` : ""}`);
export const URL_EXPORT_CSV = "/api/metricas/export?format=csv";
