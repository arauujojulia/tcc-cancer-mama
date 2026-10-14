import React, { useState } from "react";
import { ApiError, entrar, registrar } from "../api.js";
import { PERFIS } from "../perfis.js";

const SENHA_MIN = 8;
const EMAIL_RE = /^\S+@\S+\.\S+$/;

function validar(tela, v) {
  const e = {};
  if (tela === "cadastrar" && !v.nome.trim()) e.nome = "Obrigatório";
  if (!EMAIL_RE.test(v.email.trim())) e.email = "E-mail inválido";
  if (tela === "entrar" && !v.senha) e.senha = "Obrigatório";
  if (tela === "cadastrar") {
    if (v.senha.length < SENHA_MIN) e.senha = `Mínimo de ${SENHA_MIN} caracteres`;
    if (v.confirmacao !== v.senha) e.confirmacao = "As senhas não conferem";
  }
  return e;
}

// Login e cadastro. O perfil (médico ou pesquisador) é escolhido só no cadastro e
// fica na conta; o login não pergunta, lê o papel da conta e abre a tela dele.
// O servidor valida a senha e abre a sessão por cookie HttpOnly.
export default function AuthPage({ onEntrar }) {
  const [tela, setTela] = useState("entrar");
  const [papel, setPapel] = useState("medico");
  const [valores, setValores] = useState({ nome: "", email: "", senha: "", confirmacao: "" });
  const [tentou, setTentou] = useState(false);
  const [aviso, setAviso] = useState(null);
  const [erroGeral, setErroGeral] = useState(null);

  const erros = validar(tela, valores);
  const mudar = (campo) => (e) => setValores({ ...valores, [campo]: e.target.value });

  const trocarTela = (nova) => {
    setTela(nova);
    setTentou(false);
    setAviso(null);
    setErroGeral(null);
    setValores({ ...valores, senha: "", confirmacao: "" });
  };

  const [enviando, setEnviando] = useState(false);

  const enviar = async (e) => {
    e.preventDefault();
    setTentou(true);
    setErroGeral(null);
    if (Object.keys(erros).length) return;

    setEnviando(true);
    try {
      if (tela === "entrar") {
        onEntrar(await entrar(valores.email.trim(), valores.senha));
        return;
      }
      await registrar({ nome: valores.nome.trim(), email: valores.email.trim(), senha: valores.senha, papel });
      trocarTela("entrar");
      setAviso(`Conta criada como ${PERFIS[papel].rotulo}. Entre com o seu e-mail e senha.`);
    } catch (err) {
      setErroGeral(err instanceof ApiError ? err.message : "Erro inesperado. Tente novamente.");
    } finally {
      setEnviando(false);
    }
  };

  const campo = (nome, rotulo, extra = {}) => (
    <label className={tentou && erros[nome] ? "campo com-erro" : "campo"}>
      <span>{rotulo}</span>
      <input
        value={valores[nome]}
        onChange={mudar(nome)}
        aria-invalid={tentou && erros[nome] ? "true" : undefined}
        aria-describedby={tentou && erros[nome] ? `e-${nome}` : undefined}
        {...extra}
      />
      {tentou && erros[nome] && <small id={`e-${nome}`}>{erros[nome]}</small>}
    </label>
  );

  return (
    <section className="cartao auth">
      <h2>{tela === "entrar" ? "Entrar" : "Criar conta"}</h2>
      {aviso && <p className="ajuda" role="status">{aviso}</p>}

      <form onSubmit={enviar} noValidate>
        {tela === "cadastrar" && (
          <div className="radios" role="radiogroup" aria-label="Perfil">
            {Object.entries(PERFIS).map(([id, p]) => (
              <label key={id}>
                <input type="radio" name="papel" checked={papel === id} onChange={() => setPapel(id)} /> {p.rotulo}
              </label>
            ))}
          </div>
        )}

        {tela === "cadastrar" && campo("nome", "Nome", { autoComplete: "name", maxLength: 120 })}
        {campo("email", "E-mail", { type: "email", autoComplete: "email", maxLength: 120 })}
        {campo("senha", "Senha", { type: "password", autoComplete: tela === "entrar" ? "current-password" : "new-password" })}
        {tela === "cadastrar" && campo("confirmacao", "Repita a senha", { type: "password", autoComplete: "new-password" })}

        {erroGeral && <p className="erro" role="alert">{erroGeral}</p>}
        <button type="submit" disabled={enviando}>{tela === "entrar" ? "Entrar" : "Criar conta"}</button>
      </form>

      <p className="nota">
        {tela === "entrar" ? "Ainda não tem conta? " : "Já tem conta? "}
        <button type="button" className="secundario" onClick={() => trocarTela(tela === "entrar" ? "cadastrar" : "entrar")}>
          {tela === "entrar" ? "Criar conta" : "Entrar"}
        </button>
      </p>
    </section>
  );
}
