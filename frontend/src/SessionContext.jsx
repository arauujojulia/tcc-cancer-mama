import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { me as apiMe } from "./api.js";

/**
 * Estado de sessão compartilhado, alimentado por GET /auth/me no boot.
 * Guarda de UI (D3): sem trava de autorização nenhuma — serve só para o
 * menu e para redirecionar ao /login quando a sessão expira.
 */
const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const dados = await apiMe();
      setUser(dados.user);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const value = { user, loading, setUser, refresh };
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession deve ser usado dentro de SessionProvider.");
  return ctx;
}
