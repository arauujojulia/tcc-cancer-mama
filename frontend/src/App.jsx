import React, { useEffect, useState } from "react";
import AuthPage from "./pages/AuthPage.jsx";
import ClassificarPage from "./pages/ClassificarPage.jsx";
import HistoricoPage from "./pages/HistoricoPage.jsx";
import ModelosPage from "./pages/ModelosPage.jsx";
import { getUsuarioAtual, sair as sairApi } from "./api.js";
import { PERFIS } from "./perfis.js";

const abaInicial = (s) => (s ? PERFIS[s.papel].abas[0][0] : null);

export default function App() {
  // A sessão vive no cookie HttpOnly do servidor; aqui só guardamos o usuário que ele devolve.
  const [sessao, setSessao] = useState(null);
  const [aba, setAba] = useState(null);
  const [verificando, setVerificando] = useState(true);

  useEffect(() => {
    getUsuarioAtual()
      .then((u) => {
        setSessao(u);
        setAba(abaInicial(u));
      })
      .catch(() => {})
      .finally(() => setVerificando(false));
    const expirou = () => {
      setSessao(null);
      setAba(null);
    };
    window.addEventListener("sessao-expirada", expirou);
    return () => window.removeEventListener("sessao-expirada", expirou);
  }, []);

  const entrar = (usuario) => {
    setSessao(usuario);
    setAba(abaInicial(usuario));
  };

  const sair = async () => {
    try {
      await sairApi();
    } catch {
      /* mesmo sem resposta, volta para o login */
    }
    setSessao(null);
    setAba(null);
  };

  return (
    <>
      {/* RNF01: aviso permanente, sempre visível, em qualquer tela */}
      <div className="aviso-permanente" role="note">
        <strong>Ferramenta de apoio à decisão.</strong> A classificação não substitui o laudo
        histopatológico nem a avaliação médica.
      </div>

      <header className="topo">
        <h1>Triagem de risco de câncer de mama</h1>
        {sessao && (
          <div className="perfil">
            <span>{PERFIS[sessao.papel].rotulo} · {sessao.nome}</span>
            <button type="button" className="secundario" onClick={sair}>Sair</button>
          </div>
        )}
      </header>

      {verificando ? (
        <main><p className="ajuda">Carregando…</p></main>
      ) : !sessao ? (
        <main>
          <AuthPage onEntrar={entrar} />
        </main>
      ) : (
        <>
          <nav className="abas" aria-label="Seções">
            {PERFIS[sessao.papel].abas.map(([id, rotulo]) => (
              <button key={id} className={aba === id ? "ativa" : ""} aria-current={aba === id ? "page" : undefined} onClick={() => setAba(id)}>
                {rotulo}
              </button>
            ))}
          </nav>

          <main>
            {aba === "classificar" && <ClassificarPage />}
            {aba === "historico" && <HistoricoPage />}
            {aba === "modelos" && <ModelosPage />}
          </main>
        </>
      )}
    </>
  );
}
