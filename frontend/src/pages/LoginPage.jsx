import React, { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { ApiError, login, register } from "../api.js";
import { useSession } from "../SessionContext.jsx";
import { Alert } from "../components/ui.jsx";

const SENHA_MINIMA = 8;
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

function validar(modo, campos) {
  const erros = {};
  if (!campos.email.trim()) erros.email = "Informe o e-mail.";
  else if (!EMAIL_RE.test(campos.email.trim())) erros.email = "E-mail inválido.";

  if (!campos.password) erros.password = "Informe a senha.";
  else if (modo === "cadastro" && campos.password.length < SENHA_MINIMA)
    erros.password = `A senha deve ter no mínimo ${SENHA_MINIMA} caracteres.`;

  if (modo === "cadastro") {
    if (!campos.name.trim()) erros.name = "Informe o nome.";
    if (!campos.profile) erros.profile = "Escolha um perfil.";
  }
  return erros;
}

export default function LoginPage() {
  const { user, loading, setUser } = useSession();
  const navigate = useNavigate();
  const [modo, setModo] = useState("entrar");
  const [campos, setCampos] = useState({ name: "", email: "", password: "", profile: "" });
  const [tocados, setTocados] = useState({});
  const [enviando, setEnviando] = useState(false);
  const [erroServidor, setErroServidor] = useState(null);

  if (!loading && user) {
    const primeira = user.profile === "pesquisador" ? "/laboratorio" : "/classificar";
    return <Navigate to={primeira} replace />;
  }

  const erros = validar(modo, campos);
  const invalido = Object.keys(erros).length > 0;
  const campo = (nome) => ({
    value: campos[nome],
    onChange: (e) => setCampos({ ...campos, [nome]: e.target.value }),
    onBlur: () => setTocados({ ...tocados, [nome]: true }),
    "aria-invalid": tocados[nome] && erros[nome] ? "true" : undefined,
    "aria-describedby": tocados[nome] && erros[nome] ? `erro-${nome}` : undefined,
  });

  const trocarModo = (novo) => {
    setModo(novo);
    setTocados({});
    setErroServidor(null);
  };

  const enviar = async (e) => {
    e.preventDefault();
    setTocados({ name: true, email: true, password: true, profile: true });
    if (invalido) return;
    setEnviando(true);
    setErroServidor(null);
    try {
      const email = campos.email.trim().toLowerCase();
      const resposta =
        modo === "entrar"
          ? await login({ email, password: campos.password })
          : await register({ name: campos.name.trim(), email, password: campos.password, profile: campos.profile });
      setUser(resposta.user);
      navigate(resposta.user.profile === "pesquisador" ? "/laboratorio" : "/classificar");
    } catch (err) {
      setErroServidor(err instanceof ApiError ? err : new ApiError(err.message));
    } finally {
      setEnviando(false);
    }
  };

  return (
    <div className="auth-shell">
      <section className="card">
        <div className="auth-switch" role="tablist" aria-label="Entrar ou cadastrar">
          <button type="button" role="tab" aria-selected={modo === "entrar"} className={modo === "entrar" ? "active" : ""} onClick={() => trocarModo("entrar")}>
            Entrar
          </button>
          <button type="button" role="tab" aria-selected={modo === "cadastro"} className={modo === "cadastro" ? "active" : ""} onClick={() => trocarModo("cadastro")}>
            Cadastrar
          </button>
        </div>

        <h2>{modo === "entrar" ? "Entrar" : "Criar conta"}</h2>
        <p className="card-help">
          {modo === "entrar"
            ? "Use o e-mail e a senha cadastrados."
            : "Informe seus dados e escolha o perfil que melhor descreve seu uso da ferramenta."}
        </p>

        <form onSubmit={enviar} noValidate className="stack">
          {modo === "cadastro" && (
            <label className={`field ${tocados.name && erros.name ? "has-error" : ""}`}>
              <span className="field-label">Nome</span>
              <input type="text" autoComplete="name" {...campo("name")} />
              {tocados.name && erros.name && <span className="field-error" id="erro-name">{erros.name}</span>}
            </label>
          )}

          <label className={`field ${tocados.email && erros.email ? "has-error" : ""}`}>
            <span className="field-label">E-mail</span>
            <input type="email" autoComplete="email" {...campo("email")} />
            {tocados.email && erros.email && <span className="field-error" id="erro-email">{erros.email}</span>}
          </label>

          <label className={`field ${tocados.password && erros.password ? "has-error" : ""}`}>
            <span className="field-label">Senha{modo === "cadastro" && <span className="field-hint"> (mínimo {SENHA_MINIMA} caracteres)</span>}</span>
            <input type="password" autoComplete={modo === "entrar" ? "current-password" : "new-password"} {...campo("password")} />
            {tocados.password && erros.password && <span className="field-error" id="erro-password">{erros.password}</span>}
          </label>

          {modo === "cadastro" && (
            <fieldset>
              <legend className="field-label">Perfil</legend>
              <div
                className="radiogroup"
                role="radiogroup"
                aria-describedby={tocados.profile && erros.profile ? "erro-profile" : undefined}
              >
                <label>
                  <input
                    type="radio"
                    name="profile"
                    value="medico"
                    checked={campos.profile === "medico"}
                    onChange={() => setCampos({ ...campos, profile: "medico" })}
                  />
                  Médico
                </label>
                <label>
                  <input
                    type="radio"
                    name="profile"
                    value="pesquisador"
                    checked={campos.profile === "pesquisador"}
                    onChange={() => setCampos({ ...campos, profile: "pesquisador" })}
                  />
                  Pesquisador
                </label>
              </div>
              {tocados.profile && erros.profile && <span className="field-error" id="erro-profile">{erros.profile}</span>}
            </fieldset>
          )}

          {erroServidor && (
            <Alert tone="error" title="Não foi possível continuar">
              {erroServidor.message}
            </Alert>
          )}

          <button type="submit" className="btn btn-block" disabled={enviando}>
            {enviando ? "Enviando…" : modo === "entrar" ? "Entrar" : "Criar conta"}
          </button>
        </form>
      </section>
    </div>
  );
}
