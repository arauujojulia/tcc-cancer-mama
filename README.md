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
                                  PostgreSQL (SQLAlchemy)
```

```
app.py                  # aplicação Flask (config por variáveis de ambiente, serve frontend/dist)
api/routes.py           # endpoints REST
services/               # regras de negócio: validation, crypto (AES-256-GCM), avaliacoes, modelos
db/models.py            # Medico, Pesquisador, ModeloTreinado, Avaliacao
ml/                     # data, train, predict, report
ml/artifacts/           # modelos .joblib, metrics.json, feature_importance.json,
                        #   train_config.json, split.json (versionados — RNF03)
frontend/               # React + Vite
tests/                  # unittest
data/wdbc.data          # base pública WDBC
```

## Como rodar

1. Dependências Python: `pip install -r requirements.txt`
2. Configuração: copie `.env.example` para `.env` e preencha `DATABASE_URL`
   (PostgreSQL; sem ela usa SQLite local) e `PACIENTE_ENCRYPTION_KEY`
   (gere com `python -m services.crypto`; sem chave o modo *identificado*
   é recusado).
3. Treinar e avaliar (gera `ml/artifacts/`): `python -m ml.train`
4. Figuras 300 DPI e CSV para o artigo: `python -m ml.report`
   (saída em `ml/artifacts/report/`)
5. API: `python app.py` (porta 5000; cria as tabelas e registra os modelos treinados)
6. Frontend em desenvolvimento: `cd frontend && npm install && npm run dev`
   (http://localhost:5173, com proxy para a API). Para servir tudo pelo
   Flask: `npm run build` e abra http://localhost:5000.
7. Testes: `python -m unittest discover -s tests -t . -v`

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
(cifrados com AES-256-GCM). Erros: `{"erro": "...", "codigo": "..."}` com 400 (dados inválidos),
404, 503 (`MODEL_UNAVAILABLE`, `ENCRYPTION_UNAVAILABLE`).

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
- **Ainda sem autenticação:** os perfis Médico/Pesquisador só definem o
  menu do frontend. Login/JWT e RBAC (diagramas, fluxo 1) não fazem parte
  das sprints 2 e 3 e continuam pendentes; até lá, a API não deve ser
  exposta fora de ambiente controlado, pois o histórico identificado é
  legível por quem acessar a rota.

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

- Autenticação JWT/RBAC e trilha de auditoria de acesso (`access_audit`).
- Busca em grade (GridSearchCV) e promoção de modelo por limiar via API
  (fluxo 4 dos diagramas) — fora das histórias das sprints 2 e 3.
