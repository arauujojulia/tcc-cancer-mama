# Classificação de Risco de Câncer de Mama — TCC

MVP de apoio à decisão para classificação de risco de câncer de mama
(benigno/maligno) a partir de características quantitativas extraídas de
biópsias, com comparação de desempenho entre classificadores simples.

> **Este projeto é uma ferramenta de apoio e não substitui o laudo
> histopatológico nem a avaliação médica especializada.**

## Nota para quem for desenvolver neste projeto

Antes de implementar qualquer mudança, consulte a documentação do projeto
(diagramas, requisitos funcionais e não funcionais, histórias de usuário).
Se uma mudança contradisser algo documentado, a documentação deve ser
atualizada junto com o código.

## Sobre o projeto

Compara regressão logística e SVM treinados sobre o Wisconsin Diagnostic
Breast Cancer Dataset (WDBC/UCI) com acurácia, sensibilidade,
especificidade, AUC e taxa de falsos negativos. O sistema web recebe as 30
características de uma biópsia e devolve a classificação com grau de
confiança, mantém o histórico das avaliações e tem um painel para o
pesquisador comparar os modelos.

## Arquitetura

```
Frontend (React/Vite) ──HTTP/JSON──► API Flask ──► ML (scikit-learn, em memória)
                                        │
                                        ▼
                                  SQL Server Express (SQLAlchemy)
```

```
app.py                  # aplicação Flask (config por variáveis de ambiente, serve frontend/dist)
api/routes.py           # endpoints REST
services/               # regras de negócio: validation, avaliacoes, modelos
db/models.py            # Usuario (papel médico/pesquisador), ModeloTreinado, Avaliacao
ml/                     # data, train, predict, report
ml/artifacts/           # modelos .joblib, metrics.json, feature_importance.json,
                        #   train_config.json, split.json (versionados — RNF03)
frontend/               # React + Vite
tests/                  # unittest
data/wdbc.data          # base pública WDBC
```

## Como rodar

1. Dependências Python (use Python 3.14 ou anterior; o `pyodbc` ainda não tem pacote para o 3.15):
   `pip install -r requirements.txt`. Requer o "ODBC Driver 18 for SQL Server".
2. Configuração: copie `.env.example` para `.env` e preencha `DATABASE_URL` e `SECRET_KEY`
   (SQL Server Express; crie antes o banco vazio `tcc_cancer_mama`, as tabelas são criadas pela aplicação).
3. Treinar e avaliar (gera `ml/artifacts/`): `python -m ml.train`
4. Figuras 300 DPI e CSV para o artigo: `python -m ml.report`
   (saída em `ml/artifacts/report/`)
5. API: `python app.py` (porta 5000; cria as tabelas e registra os modelos treinados)
6. Frontend em desenvolvimento: `cd frontend && npm install && npm run dev`
   (http://localhost:5173, com proxy para a API). Para servir tudo pelo
   Flask: `npm run build` e abra http://localhost:5000.
7. Criar usuários: pela tela de cadastro, ou `python -m services.usuarios --nome ... --email ... --papel medico|pesquisador`.
8. Testes: `python -m unittest discover -s tests -t . -v`

## API

| Método | Rota | História |
|---|---|---|
| GET | `/api/features/schema` | US01 — 30 campos, faixas aceitas e exemplos |
| POST | `/api/biopsias/classificar` | US02, US03, US07 — classifica **e registra** a avaliação |
| GET | `/api/avaliacoes` | US07 — histórico paginado; filtros `classificacao`, `identificado`, `data_inicio`, `data_fim`, `confianca_min`, `confianca_max`, `pagina`, `limite` |
| GET | `/api/avaliacoes/<id>` | US07 |
| GET | `/api/metricas` | US05 — métricas, matriz de confusão, ROC, validação cruzada, reprodutibilidade |
| GET | `/api/metricas/features?modelo=&top=` | US06 — importância por permutação |
| GET | `/api/metricas/export?format=csv` | US06 — tabela para o artigo |
| GET | `/api/modelos` | US06 — modelos registrados (hiperparâmetros, semente, SHA-256) |

Exemplo de classificação:

```json
POST /api/biopsias/classificar
{ "caracteristicas": { "mean_radius": 17.99, "...": "(30 campos)" },
  "modo": "anonimo" }
```

Para registro identificado: `"modo": "identificado", "paciente": {"nome": "...", "prontuario": "..."}`
(texto livre). Erros: `{"erro": "...", "codigo": "..."}` com 400 (dados inválidos),
404, 503 (`MODEL_UNAVAILABLE`).

## Decisões de projeto importantes

- **Limiar de segurança (RNF04).** Cada modelo tem um limiar de decisão
  escolhido com predições *out-of-fold* do treino (sem usar o teste): o
  maior limiar (≤ 0,5) que mantém sensibilidade ≥ 0,97. O caso é maligno
  quando P(maligno) ≥ limiar. As métricas são reportadas nos dois pontos
  (limiar 0,5 e limiar de segurança). Isso muda o resultado do SVM em
  relação à Sprint 1, que usava `predict()` (fronteira de decisão do SVM);
  agora toda decisão usa a probabilidade, de forma consistente com a ROC.
- **Melhor modelo:** maior sensibilidade no ponto de operação; desempate
  por AUC e depois especificidade. É o modelo usado na classificação.
- **Reprodutibilidade (RNF03):** semente 42, índices das partições
  (`split.json`), hiperparâmetros, versões, commit e SHA-256 dos artefatos
  e do dataset em `ml/artifacts/`.
- **Histórico imutável:** `Avaliacao` não pode ser alterada nem removida
  pela aplicação.
- **Autenticação por cookie de sessão** (`services/auth.py`): cookie `HttpOnly`,
  `SameSite=Lax` e, em produção, `Secure`; sessão de 8 h. Cadastro aberto aos dois
  papéis (`POST /api/auth/registrar`), `POST /api/auth/login`, `POST /api/auth/logout`
  e `GET /api/auth/me`. Papéis aplicados no servidor (RBAC): o médico classifica
  (`/biopsias/classificar`, `/avaliacoes/<id>`); o pesquisador acessa `/metricas*` e
  `/modelos`; `/avaliacoes` e `/features/schema` valem para qualquer usuário logado.
  Escritas exigem `Content-Type: application/json` (defesa de CSRF). O `medico_id`
  da avaliação vem da sessão, nunca do corpo. A senha é guardada com hash (werkzeug).
- **Dados do paciente em texto simples** no banco (nome e prontuário), por decisão do
  projeto. Use apenas dados fictícios ou anonimizados fora de ambiente controlado.

## Progresso por sprint

- ✅ Sprint 0 — HT01, HT02, HT03
- ✅ Sprint 1 — US01, US02, US03, US04
- ✅ Sprint 2 — US05 (métricas, FN, validação cruzada), US06 (importância,
  partições e hiperparâmetros registrados, figuras 300 DPI), US07 (modelos
  `Avaliacao` e `ModeloTreinado`)
- ✅ Sprint 3 — API REST completa, registro automático e histórico
  filtrável (anônimo/identificado), frontend React integrado, validação de
  tempo (< 2 s), testes

## Pendências conhecidas

- Trilha de auditoria de acesso (`access_audit`) e limite de tentativas de login.
- Busca em grade (GridSearchCV) e promoção de modelo por limiar via API
  (fluxo 4 dos diagramas) — fora das histórias das sprints 2 e 3.
