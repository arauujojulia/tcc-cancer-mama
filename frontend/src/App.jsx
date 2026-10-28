import React from "react";
import { Navigate, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { AlertTriangleIcon } from "./components/icons.jsx";
import { logout as apiLogout } from "./api.js";
import { useSession } from "./SessionContext.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import ClassificarPage from "./pages/ClassificarPage.jsx";
import HistoricoPage from "./pages/HistoricoPage.jsx";
import ModelosPage from "./pages/ModelosPage.jsx";

// Menu por perfil (D3): guarda de UI, não de autorização — digitar a URL de
// outra tela carrega normalmente, sem trava nenhuma no servidor nem aqui.
const ITENS_MENU = {
  medico: [
    ["/classificar", "Classificar"],
    ["/historico", "Histórico"],
  ],
  pesquisador: [
    ["/laboratorio", "Laboratório de modelos"],
    ["/historico", "Histórico"],
  ],
};

function RotaProtegida({ children }) {
  const { user, loading } = useSession();
  if (loading) return null;
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <div className="app-shell">
      {/* RNF01: aviso permanente, visível em todas as telas, não só no resultado. */}
      <div className="disclaimer-bar" role="note">
        <AlertTriangleIcon className="icon" style={{ width: 16, height: 16, verticalAlign: "-3px" }} />
        <strong>Ferramenta de apoio à decisão.</strong> Não substitui o laudo histopatológico nem a avaliação médica.
      </div>

      <AppHeader />

      <main className="app-main">
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/classificar" element={<RotaProtegida><ClassificarPage /></RotaProtegida>} />
          <Route path="/historico" element={<RotaProtegida><HistoricoPage /></RotaProtegida>} />
          <Route path="/laboratorio" element={<RotaProtegida><ModelosPage /></RotaProtegida>} />
          <Route path="/" element={<RedirecionarInicial />} />
          <Route path="*" element={<RedirecionarInicial />} />
        </Routes>
      </main>
    </div>
  );
}

function RedirecionarInicial() {
  const { user, loading } = useSession();
  if (loading) return null;
  if (!user) return <Navigate to="/login" replace />;
  const primeira = ITENS_MENU[user.profile]?.[0]?.[0] || "/historico";
  return <Navigate to={primeira} replace />;
}

function AppHeader() {
  const { user, setUser, loading } = useSession();
  const navigate = useNavigate();
  const itens = user ? ITENS_MENU[user.profile] || [] : [];

  const sair = async () => {
    try {
      await apiLogout();
    } finally {
      setUser(null);
      navigate("/login");
    }
  };

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="app-brand">
          <h1>Triagem de câncer de mama</h1>
          <span className="tag">WDBC · apoio à decisão</span>
        </div>

        {!loading && user && (
          <nav className="app-nav" aria-label="Seções do sistema">
            {itens.map(([caminho, rotulo]) => (
              <NavLink key={caminho} to={caminho} className={({ isActive }) => (isActive ? "active" : "")}>
                {rotulo}
              </NavLink>
            ))}
          </nav>
        )}

        {!loading && (
          <div className="app-session">
            {user ? (
              <>
                <span className="who">
                  Olá, <strong>{user.name}</strong> ({user.profile === "medico" ? "médico" : "pesquisador"})
                </span>
                <button type="button" className="btn btn-secondary btn-sm" onClick={sair}>Sair</button>
              </>
            ) : (
              <span className="who">Não autenticado</span>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
