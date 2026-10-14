# Login por perfil (Médico / Pesquisador)

Registro das decisões e do estado da autenticação.

## 1. Decisões

| # | Decisão | Escolha |
|---|---|---|
| 1 | Cadastro de usuários | **Aberto aos dois papéis**, pela tela de cadastro (`POST /api/auth/registrar`) ou pela linha de comando (`python -m services.usuarios`). Decisão final do projeto. |
| 2 | Mecanismo | Cookie de sessão assinado pelo Flask (`HttpOnly`, `SameSite=Lax`, `Secure` em produção), 8 h. Escolhido no lugar de JWT: o token não fica acessível ao JavaScript (XSS) e o logout é imediato no navegador. |
| 3 | Modelo de dados | Tabela única `usuarios` com enum `Papel` (`medico`, `pesquisador`); senha com hash (werkzeug). |
| 4 | Onde se escolhe o papel | Só no cadastro. O login lê o papel da conta. |
| 5 | Segurança | O papel é aplicado no servidor (RBAC). O menu do frontend é só experiência de uso. |

## 2. Permissões

| Rota | Quem acessa |
|---|---|
| `POST /api/auth/registrar`, `POST /api/auth/login`, `POST /api/auth/logout` | Público |
| `GET /api/auth/me`, `GET /api/features/schema`, `GET /api/avaliacoes` | Qualquer usuário logado |
| `POST /api/biopsias/classificar`, `GET /api/avaliacoes/<id>` | Médico |
| `GET /api/metricas`, `/metricas/features`, `/metricas/export`, `/modelos` | Pesquisador |

Sem login: `401 NAO_AUTENTICADO`. Papel errado: `403 SEM_PERMISSAO`. Login inválido:
`401 CREDENCIAIS_INVALIDAS`, com mensagem única (não revela se o e-mail existe).

## 3. Implementado

- `services/auth.py`: autenticação, sessão e decorator `exigir_login(*papeis)`. O usuário é relido do banco a cada requisição, então `ativo = false` vale na hora.
- `api/routes.py`: endpoints de autenticação, rotas protegidas, `medico_id` vindo da sessão (o valor enviado no corpo é ignorado) e exigência de `Content-Type: application/json` nas escritas (defesa de CSRF, somada ao `SameSite=Lax`).
- `app.py`: `SECRET_KEY` do ambiente, flags do cookie, sessão de 8 h e CORS com origens explícitas e `supports_credentials`.
- Frontend: `api.js` com `credentials`, login e cadastro ligados à API (`AuthPage.jsx`), sessão restaurada por `GET /api/auth/me` ao abrir o app, 401 devolve à tela de login. `contas.js` foi removido.
- Testes: `tests/test_auth.py` (registro, login, cookie, 401/403 em cada rota, conta inativa, logout, `medico_id` da sessão, CSRF) e `tests/test_usuarios.py`.

## 4. Pendências e riscos conhecidos

| Item | Observação |
|---|---|
| Cadastro aberto com papel livre | Qualquer pessoa se cadastra como pesquisador e acessa métricas, modelos e o histórico (que mostra nome e prontuário). Decisão assumida pelo projeto. Mitigação futura: aprovação por administrador ou verificação (CRM, ORCID, CAFe, gov.br). |
| Sem limite de tentativas de login | Fora do MVP. |
| Pesquisador vê o histórico identificado | `HistoricoPage` e `/avaliacoes` mostram nome e prontuário a qualquer usuário logado. Decidir se o pesquisador vê só dados anônimos. |
| Histórico: médico vê o próprio ou o de todos | Hoje, o de todos. |
| Produção | Definir `SECRET_KEY` fixa, `SESSION_COOKIE_SECURE=true`, `CORS_ORIGINS` real e servir por HTTPS. |
| Sem trilha de auditoria de acesso | Pendente. |
| Migração de esquema (Alembic) | Hoje só há `create_all`; avaliar antes de mudar o esquema de novo. |

## 5. Documentação do TCC (PDF e diagramas)

| Item | Onde |
|---|---|
| §4.2.1 "Seleção de Perfil" passa a descrever login e controle de acesso por papel | PDF |
| Figura 3 refeita como fluxo de login; sequência "Seleção de Perfil (BSC-12)" | PDF e `docs/diagrams.md` |
| Diagrama de classes (Fig. 2): `Medico` e `Pesquisador` como especializações de `Usuario` com o enum | PDF |
| §5.2.4 "credenciais criptografadas": o correto é senha com hash | PDF |
| Sumário, legendas das Figuras 3, 4 e 6 divergem dos títulos | PDF |
| Cadastro aberto aos dois papéis e verificação por CRM/ORCID como trabalho futuro | PDF (Considerações Finais) |
| Laboratório de modelos é somente leitura; o PDF (Fig. 6, §4.2.4) descreve treino com GridSearch e promoção por limiar | Decidir se entra antes de 02/11 ou se o PDF muda |
