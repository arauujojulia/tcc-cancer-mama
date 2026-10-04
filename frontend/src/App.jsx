import React, { useState } from "react";
import ClassificarPage from "./pages/ClassificarPage.jsx";
import HistoricoPage from "./pages/HistoricoPage.jsx";
import ModelosPage from "./pages/ModelosPage.jsx";

// Perfis (HT03). Ainda sem login: o perfil só decide quais telas o menu mostra.
const PERFIS = {
  medico: {
    rotulo: "Médico",
    abas: [
      ["classificar", "Nova classificação"],
      ["historico", "Histórico"],
    ],
  },
  pesquisador: {
    rotulo: "Pesquisador",
    abas: [
      ["modelos", "Laboratório de modelos"],
      ["historico", "Histórico"],
    ],
  },
};

const lerPerfil = () => {
  try {
    const salvo = localStorage.getItem("perfil");
    return salvo in PERFIS ? salvo : "medico";
  } catch {
    return "medico";
  }
};

export default function App() {
  const [perfil, setPerfil] = useState(lerPerfil);
  const [aba, setAba] = useState(PERFIS[lerPerfil()].abas[0][0]);

  const trocarPerfil = (novo) => {
    setPerfil(novo);
    setAba(PERFIS[novo].abas[0][0]);
    try {
      localStorage.setItem("perfil", novo);
    } catch {
      /* armazenamento indisponível */
    }
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
        <label className="perfil">
          Perfil
          <select value={perfil} onChange={(e) => trocarPerfil(e.target.value)}>
            {Object.entries(PERFIS).map(([id, p]) => (
              <option key={id} value={id}>
                {p.rotulo}
              </option>
            ))}
          </select>
        </label>
      </header>

      <nav className="abas" aria-label="Seções">
        {PERFIS[perfil].abas.map(([id, rotulo]) => (
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
  );
}
