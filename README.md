# Classificação de Risco de Câncer de Mama — TCC (Grupo 05)

Sistema web de **apoio à decisão** que classifica uma amostra entre **benigna**
e **maligna** a partir das 30 medidas quantitativas do *Wisconsin Diagnostic
Breast Cancer Dataset* (WDBC/UCI), com grau de confiança, histórico das
avaliações e um laboratório para o pesquisador treinar, comparar e promover
modelos (Regressão Logística e SVM).

A base é **tabular**: 30 medidas numéricas extraídas por processamento de
imagem de **punção aspirativa por agulha fina (FNA)** — raio, textura,
perímetro, área etc. em três agregações (média, erro padrão e pior valor). A
ferramenta **não** processa imagens de biópsia.

> ## O que este sistema NÃO é
>
> - **Não tem caráter diagnóstico.** É uma ferramenta de apoio à decisão.
> - **Não substitui o laudo histopatológico** nem a avaliação médica
>   especializada.
> - **Não é um dispositivo médico certificado** e não deve ser usado para
>   decisão clínica sobre pacientes reais.

## ⚠️ Aviso de segurança (leia antes de expor a aplicação)

**Todas as rotas de dados exigem sessão válida.** Sem o cookie de sessão, elas
respondem 401 `UNAUTHENTICATED`; só `/auth/*` e `/health` são públicas. A trava
é de **autenticação, não de autorização**.

Esta aplicação **não implementa controle de acesso por perfil**. O campo
`profile` (`medico` / `pesquisador`) decide apenas o menu exibido no frontend;
**nenhuma rota é bloqueada por perfil** (D3). Logo, **qualquer usuário
autenticado** — médico ou pesquisador, indistintamente — consegue ler o
histórico identificado inteiro, disparar treinos e promover modelos.

No modo de registro **identificado**, o nome e o prontuário da paciente são
gravados **em texto puro** no banco, sem nenhuma criptografia.

**Não exponha esta aplicação fora de um ambiente controlado.** Ela foi
construída para fins acadêmicos (TCC) e para demonstração; não há segregação de
dados, cifragem em repouso nem trilha de auditoria.

## Documentação

| Documento | Para quê |
|---|---|
| **[`docs/COMO-FUNCIONA.md`](docs/COMO-FUNCIONA.md)** | **documentação funcional**: o que o sistema faz, por que faz assim e onde cada decisão está no código — dados, caminho completo de uma classificação, limiar de segurança, reprodutibilidade, perfis, as quatro telas, API por fluxo, rastreabilidade e limitações |
| [`.agents/ESPECIFICACAO-CONSOLIDADA.md`](.agents/ESPECIFICACAO-CONSOLIDADA.md) | fonte da verdade da implementação: decisões D1–D12, modelo de dados, contrato da API, requisitos verificáveis |
| [`.agents/RELATORIO-IMPLEMENTACAO.md`](.agents/RELATORIO-IMPLEMENTACAO.md) | o que foi construído, com os números reais, os defeitos corrigidos e a lista de correções do `.docx` do TCC |
| [`.agents/tasks/mvp-alinhado/design.md`](.agents/tasks/mvp-alinhado/design.md) | design técnico |
| [`docs/diagrams.md`](docs/diagrams.md) | os 6 diagramas UML |

Comece por `docs/COMO-FUNCIONA.md` se a pergunta é "como isso funciona?"; vá à
especificação consolidada se a pergunta é "o que vale como contrato?".

## Nota para quem for desenvolver neste projeto

A fonte da verdade é a documentação em `.agents/`
(`ESPECIFICACAO-CONSOLIDADA.md`, `tasks/mvp-alinhado/design.md`,
`docs/diagrams.md`). **Documentação e código nunca devem divergir**: se uma
mudança de código contrariar o que está documentado, atualize a documentação no
mesmo commit — e, se a mudança altera o comportamento visível, atualize também
`docs/COMO-FUNCIONA.md`.

---

## Arquitetura (4 camadas)

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)                                     │
│  SPA — consome exclusivamente a API REST, nunca o banco/ML   │
└───────────────────────────┬─────────────────────────────────┘
                            │  HTTP/JSON (/api/v1)
┌───────────────────────────▼─────────────────────────────────┐
│  Backend (Flask)                                             │
│  rotas, validação, orquestração, erros padronizados          │
└──────────────┬───────────────────────────┬──────────────────┘
               │                           │
┌──────────────▼─────────────┐  ┌───────────▼──────────────────┐
│  ML Engine (scikit-learn)  │  │  Persistência (SQLAlchemy)    │
│  modelos em memória,       │  │  PostgreSQL (ou SQLite em dev)│
│  StandardScaler, limiares  │  │  users, evaluations,          │
│                            │  │  trained_models, training_runs│
└────────────────────────────┘  └───────────────────────────────┘
```

Mapa de pastas:

```
app.py                   # app Flask (config por variáveis de ambiente, serve frontend/dist)
api/routes.py            # endpoints REST sob /api/v1
services/                # regras de negócio: validation, auth, avaliacoes, modelos
db/models.py             # User, Evaluation, TrainedModel, TrainingRun
ml/data.py               # carga do WDBC e schema das 30 características
ml/train.py              # treino, limiar de segurança, métricas, reprodutibilidade
ml/predict.py            # inferência (modelos em memória)
ml/report.py             # figuras 300 DPI + CSV
ml/artifacts/            # REFERÊNCIA VERSIONADA (RNF03): scaler, {modelo}.joblib,
                         #   {modelo}_oof.json, {modelo}_test_proba.json,
                         #   metrics.json, feature_importance.json,
                         #   train_config.json, split.json
                         #   regerada apenas por `python -m ml.train`
ml/artifacts/runs/{training_run_id}/  # artefatos de cada treino sob demanda (não versionado)
frontend/                # React + Vite (login, classificar, histórico, laboratório)
tests/                   # unittest (unitários + integração HTTP)
data/wdbc.data           # base pública WDBC
```

---

## Como rodar

### 1. Dependências Python

```bash
pip install -r requirements.txt
```

(Flask, Flask-CORS, Flask-SQLAlchemy, psycopg v3, scikit-learn, pandas, numpy,
joblib, python-dotenv, matplotlib. Python 3.14.)

### 2. Variáveis de ambiente

Copie `.env.example` para `.env`:

| Variável | Função |
|---|---|
| `DATABASE_URL` | PostgreSQL via driver `psycopg` v3, ex.: `postgresql+psycopg://usuario:senha@localhost:5432/tcc_cancer_mama`. **Se omitida**, a aplicação usa SQLite local (`sqlite:///tcc_cancer_mama.db`) — só para desenvolvimento. |
| `CORS_ORIGINS` | Lista explícita das origens do frontend (separadas por vírgula). O cookie de sessão viaja com credenciais, então `*` é recusado pelo navegador. Ex.: `http://localhost:5173`. |
| `SECRET_KEY` | Assina o cookie de sessão. Sem ela, uma chave temporária é gerada em memória (as sessões caem a cada reinício) e um aviso é registrado no log. |
| `SESSION_COOKIE_SECURE` | `true` atrás de HTTPS; `false` em desenvolvimento (HTTP). |

> O modelo de dados não tem migração de linhas entre versões de schema. Se já
> existir um banco de um estado anterior, **apague-o** antes de subir
> (`rm tcc_cancer_mama.db` ou recrie o schema no PostgreSQL).

> **Recriação obrigatória para D11/D12.** A coluna `train_oof_sensitivity` e a
> unicidade `(training_run_id, name)` em `trained_models` mudam o schema, e as
> linhas de um banco anterior estão justamente inconsistentes (métricas de dois
> conjuntos de avaliação na mesma linha). Recrie:
> - **SQLite:** apague `instance/tcc_cancer_mama.db` e suba a aplicação.
> - **PostgreSQL:** `DROP SCHEMA public CASCADE; CREATE SCHEMA public;` e suba a
>   aplicação.
>
> Em qualquer dos dois, `create_all()` recria o schema e `sincronizar_modelos()`
> registra os dois modelos de referência 80/20 na subida.

### 3. Treinar e avaliar os modelos

```bash
python -m ml.train
```

Gera `ml/artifacts/`: `scaler.joblib`, `regressao_logistica.joblib`,
`svm.joblib`, os vetores out-of-fold (`{modelo}_oof.json`), as probabilidades do
conjunto de teste (`{modelo}_test_proba.json`), `metrics.json`,
`feature_importance.json`, `train_config.json` e `split.json`.

`{modelo}_test_proba.json` (`{"y_true", "proba"}`) é o insumo do recálculo das
métricas reportadas quando o limiar vigente muda: o conjunto de teste nunca
escolhe o limiar, só reporta as métricas no limiar já decidido pelo out-of-fold
do treino (D11).

> **Este comando regenera a referência versionada do artigo** (split 80/20,
> semente 42 — RNF03): é o único caminho que escreve em `ml/artifacts/`.
> Treinos disparados pela tela do Laboratório (`POST /training-runs`) **não**
> alteram essa referência: eles gravam em
> `ml/artifacts/runs/{training_run_id}/`, que fica fora do versionamento. Para
> tornar uma execução a nova referência, rode `python -m ml.train` com os
> parâmetros desejados e recomite os artefatos.

### 4. Gerar as figuras científicas (300 DPI) e o CSV

```bash
python -m ml.report
```

Saída em `ml/artifacts/report/`: `roc_comparacao.png`,
`matrizes_confusao.png`, `importancia_regressao_logistica.png`,
`importancia_svm.png`, `metricas.csv`.

As figuras e o CSV de um treino sob demanda ficam em
`ml/artifacts/runs/{training_run_id}/report/`, separados do report da
referência.

### 5. API

```bash
python app.py
```

Sobe na porta 5000, cria as tabelas e sincroniza os modelos pré-treinados em
`trained_models`.

### 6. Frontend

```bash
cd frontend && npm install && npm run dev     # http://localhost:5173 (proxy para a API)
```

Para servir tudo pelo próprio Flask: `npm run build` e abra
`http://localhost:5000`.

### 7. Testes

```bash
python -m unittest discover -s tests -t . -v
```

---

## Contrato da API (`/api/v1`)

Identificadores (rotas, campos, códigos de erro) em inglês; mensagens ao usuário
em português. Envelope de erro sempre `{"error": "...", "code": "..."}`.
Códigos: `VALIDATION_ERROR` (400), `UNAUTHENTICATED` (401), `NOT_FOUND` (404),
`MODEL_UNAVAILABLE` (503), `TIMEOUT` (504), `SENSITIVITY_BELOW_TARGET` (400),
`INTERNAL_ERROR` (500).

A coluna **Sessão** indica se a rota exige cookie de sessão válido. Sem sessão,
as rotas marcadas respondem 401 `UNAUTHENTICATED`. Não há verificação de
perfil em nenhuma delas (D3).

| Método | Rota | Sessão | O que faz | História de usuário |
|---|---|---|---|---|
| POST | `/auth/register` | pública | cadastra `{name, email, password, profile}` → 201 `{user}` e abre sessão | HT (cadastro com perfil) |
| POST | `/auth/login` | pública | `{email, password}` → 200 `{user}` + cookie de sessão | Seleção de Perfil |
| POST | `/auth/logout` | pública | encerra a sessão → 204 | Seleção de Perfil |
| GET | `/auth/me` | pública | usuário da sessão, ou 401 | Seleção de Perfil |
| GET | `/features/schema` | **exige sessão** | 30 campos: nome, rótulo PT, faixa observada, faixa aceita (±20%), exemplo | US01 |
| POST | `/predictions` | **exige sessão** | classifica e grava a avaliação (autoria em `user_id`) | US02, US03, US07 |
| GET | `/predictions` | **exige sessão** | histórico paginado — filtros `label`, `mode`, `date_from`, `date_to`, `confidence_min`, `confidence_max`, `page`, `limit` | US07 |
| GET | `/predictions/{id}` | **exige sessão** | detalhe de uma avaliação | US07 |
| POST | `/training-runs` | **exige sessão** | treino sob demanda `{models, test_size, grid_search, hyperparameters}` → 201 `{training_run_id, status, artifacts_dir}` | US04 |
| GET | `/training-runs/{id}` | **exige sessão** | status e modelos resultantes do treino | US04 |
| GET | `/models` | **exige sessão** | modelos registrados | US05 |
| GET | `/models/comparison` | **exige sessão** | métricas do conjunto de teste **de cada execução** + composição desse teste (`test_size`, `test_set_size`, `test_set_malignant`) + portão `train_oof_sensitivity` + `test_partitions_mixed` / `test_partitions` + pontos da ROC + reprodutibilidade da referência | US05 |
| GET | `/models/features?model=&top=` | **exige sessão** | importância por permutação do melhor modelo | US06 |
| PUT | `/models/{id}/threshold` | **exige sessão** | valida o limiar pelo **portão out-of-fold do treino** (≥ 0,97 — RNF04), **promove o modelo** e recalcula as métricas do conjunto de teste no limiar novo → 200 `{model, threshold, train_oof_sensitivity, test_metrics}` | US05, US06 |
| GET | `/models/export?format=csv` | **exige sessão** | CSV com métricas e hiperparâmetros | US06 |

Exemplo de classificação:

```json
POST /api/v1/predictions
{ "features": { "mean_radius": 17.99, "...": "(30 campos)" },
  "mode": "anonimo", "patient_name": null, "patient_record": null }
```

No modo identificado, `mode` é `"identificado"` e `patient_name` /
`patient_record` são obrigatórios (gravados **em texto puro** — ver aviso de
segurança). O modo anônimo descarta qualquer dado de paciente enviado.

> Nota de contrato: `confidence_band` assume os valores `alta`, `media`
> (**sem acento**) e `baixa`. O acento em "média" aparece apenas no texto
> exibido na interface, nunca no valor da API nem nos filtros.

---

## Decisões de projeto (D1 a D7)

Resolvem conflitos entre os artefatos (documento, diagramas e os ajustes
combinados por WhatsApp em 09/10/2026). Detalhe completo em
`.agents/ESPECIFICACAO-CONSOLIDADA.md`.

| # | Decisão | Justificativa |
|---|---|---|
| **D1** | API em inglês sob `/api/v1`; tabelas `evaluations` / `training_runs`. | Os diagramas de sequência são o artefato mais recente e é o que o professor avalia. |
| **D2** | **Remover a criptografia AES-256-GCM.** Modo identificado continua existindo, gravando `patient_name` / `patient_record` em texto puro. | Pedido explícito da equipe (prazo curto). O RNF05 precisa ser reescrito no documento. |
| **D3** | **Sem JWT, sem RBAC, sem auditoria.** Só cadastro e login; o perfil é declarado no cadastro e serve apenas para escolher as telas, sem trava de autorização. | Pedido explícito da equipe e coerente com o texto "sem a necessidade de mecanismos adicionais de controle de acesso nesta etapa". |
| **D4** | Senha com hash `werkzeug.security` (pbkdf2), nunca em texto. | Não é RBAC nem segregação de perfil; é só não deixar a senha legível no banco. |
| **D5** | **PostgreSQL** como alvo via `DATABASE_URL`; sem a variável, cai em SQLite local. | O documento fixa PostgreSQL 16; a máquina de desenvolvimento não tem Postgres. |
| **D6** | **Implementar GridSearchCV** (5 folds estratificados, `scoring="recall"`). | Está no diagrama do Laboratório de Modelos, e o diagrama é avaliado. |
| **D7** | **Não usar `wandb`.** | Apontado pelo professor como fora dos requisitos; a equipe deve removê-lo da Figura 2. |

Outras decisões técnicas:

- **Limiar de segurança (RNF04).** O maior limiar ≤ 0,5 que mantém sensibilidade
  ≥ 0,97, calculado com predições *out-of-fold* do treino (o conjunto de teste
  nunca entra na escolha do limiar). A classe é decidida por
  `P(maligno) ≥ limiar`, **nunca** por `sklearn.predict()`. As métricas são
  reportadas nos dois pontos de operação (limiar 0,5 e limiar de segurança).
- **Promoção de modelo (`PUT /models/{id}/threshold`).** O **portão** é a
  sensibilidade recalculada no limiar pedido a partir do vetor out-of-fold
  persistido (`{modelo}_oof.json`); abaixo de 0,97 devolve 400
  `SENSITIVITY_BELOW_TARGET` com a sensibilidade obtida e a exigida, sem alterar
  nada. Passando o portão, o valor vai para `train_oof_sensitivity` e **todas**
  as métricas reportadas (`accuracy`, `sensitivity`, `specificity`,
  `false_negative_rate`, `confusion_matrix`) são recalculadas em conjunto no
  conjunto de teste, no limiar novo, a partir de `{modelo}_test_proba.json`. Se
  esse arquivo faltar, a rota devolve 503 `MODEL_UNAVAILABLE` **sem gravar
  nada** — uma linha meio atualizada é o defeito que D11 corrige. A resposta é
  `{model, threshold, train_oof_sensitivity, test_metrics{...}}`.
- **Histórico imutável.** `Evaluation` não pode ser alterada nem removida pela
  aplicação (barrado por listeners `before_update` / `before_delete`).
- **Reprodutibilidade (RNF03).** Semente 42, índices das partições
  (`split.json`), hiperparâmetros (`train_config.json` e no `.joblib`), versões
  das bibliotecas, commit do git e SHA-256 do dataset e dos artefatos.
- **D8 — artefatos de referência isolados das execuções.** `ml/artifacts/` é a
  referência versionada do artigo e só `python -m ml.train` escreve nela. O
  treino sob demanda (`POST /training-runs`) grava em
  `ml/artifacts/runs/{training_run_id}/` (incluindo `report/`), o
  `artifact_path` do `trained_models` aponta para esse diretório e a pasta está
  no `.gitignore`. Assim um treino disparado durante a defesa não altera
  silenciosamente as métricas publicadas.
- **D9 — a classificação usa sempre a referência.** `POST /predictions` carrega
  os artefatos de `ml/artifacts/`. Promover um modelo treinado sob demanda
  registra o novo ponto de operação no banco e no Laboratório, mas **não** troca
  o artefato que classifica; para isso, rode `python -m ml.train` e reinicie a
  API.
- **D10 — sessão obrigatória, sem autorização por perfil.** Todas as rotas fora
  de `/auth/*` exigem sessão válida (401 `UNAUTHENTICATED` sem ela), aplicada
  rota a rota pelo decorador `exigir_sessao` em `api/routes.py`. Nenhuma rota
  verifica `profile`: um médico autenticado chama as rotas do pesquisador e
  vice-versa (D3). É autenticação, não autorização.
- **D11 — as métricas reportadas são do teste; o portão é do treino
  out-of-fold.** Em `trained_models`, `accuracy`, `sensitivity`, `specificity`,
  `false_negative_rate` e `confusion_matrix` são **sempre** do conjunto de teste
  da execução daquele modelo, no limiar vigente da linha, e são gravadas sempre
  em conjunto; `auc` é do teste e independe do limiar. O portão de promoção do
  RNF04 mora na coluna própria `train_oof_sensitivity` (predições out-of-fold do
  **treino**, no limiar da linha) e nunca é gravado em `sensitivity`. Vale a
  invariante, verificada automaticamente em `tests/test_metricas_coerentes.py`:

  ```
  sensitivity         == vp / (vp + fn)   da própria confusion_matrix
  false_negative_rate == fn / (vp + fn)
  specificity         == vn / (vn + fp)
  accuracy            == (vn + vp) / (vn + fp + fn + vp)
  ```

  Escolher ou validar limiar olhando o conjunto de teste vazaria informação, e é
  por isso que o portão continua no out-of-fold do treino; o teste só reporta.
  Existe uma única implementação do cálculo (`ml.train.compute_metrics`,
  chamada por `services.modelos.metricas_teste_no_limiar`), o que faz a
  invariante valer por construção.
- **D12 — uma linha de `trained_models` por execução.** A unicidade é
  `(training_run_id, name)` (`uq_trained_models_run_name`), não
  `artifact_sha256`. Dois treinos deterministas com a mesma semente e os mesmos
  dados produzem legitimamente os mesmos bytes, então o hash fica como
  procedência (RNF03) e `artifact_path` aponta **sempre** para o artefato da
  própria execução — `ml/artifacts/{nome}.joblib` para a referência,
  `ml/artifacts/runs/{id}/{nome}.joblib` para um treino sob demanda. Como não há
  Alembic no projeto, a mudança de schema é aplicada por recriação: apagar
  `instance/tcc_cancer_mama.db` (SQLite) ou
  `DROP SCHEMA public CASCADE; CREATE SCHEMA public;` (PostgreSQL) e subir a
  aplicação, que executa `create_all()` + `sincronizar_modelos()`.
  Consequência em `GET /models/comparison`: como a tabela pode ter linhas de
  execuções com partições diferentes, cada item traz a composição do seu próprio
  conjunto de teste e a resposta sinaliza `test_partitions_mixed` — comparar
  métricas entre partições distintas não é válido.

---

## Estado por requisito

| ID | Requisito | Estado |
|---|---|---|
| RF01 | Formulário aceita e valida as 30 características | ✅ Implementado |
| RF02 | Classifica entre benigno e maligno | ✅ Implementado |
| RF03 | Exibe grau de confiança | ✅ Implementado |
| RF04 | Treina e compara Regressão Logística e SVM na mesma base | ✅ Implementado |
| RF05 | Exibe acurácia, sensibilidade, especificidade e AUC | ✅ Implementado |
| RF06 | Exibe as características mais relevantes do melhor modelo | ✅ Implementado |
| RF07 | Mantém histórico, sem exigir identificação do paciente | ✅ Implementado |
| RNF01 | Aviso de apoio permanente em todas as telas | ✅ Implementado |
| RNF02 | Classificação em < 2 s; `elapsed_ms` gravado e testado | ✅ Implementado |
| RNF03 | Partições, hiperparâmetros, semente e versões versionados | ✅ Implementado |
| RNF04 | Taxa de falsos negativos reportada; limiar de segurança 0,97 | ✅ Implementado |
| RNF05 | **Alterado por D2** — modo identificado sem criptografia | ⚠️ Documento a corrigir |

> O estado detalhado, com onde cada requisito foi implementado e como foi
> verificado, está em `.agents/RELATORIO-IMPLEMENTACAO.md` — inclusive os três
> defeitos de métricas corrigidos nesta entrega (seção 2.6: `PUT /threshold`
> deixando a linha com métricas de dois conjuntos, `artifact_path` apontando
> para a referência e o cabeçalho fixo do Laboratório). A explicação funcional
> do comportamento corrigido está em `docs/COMO-FUNCIONA.md`.
