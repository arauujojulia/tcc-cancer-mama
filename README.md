# Classificação de Risco de Câncer de Mama — TCC

MVP de apoio à decisão para classificação de risco de câncer de mama
(benigno/maligno) a partir de características quantitativas extraídas de
biópsias, com comparação de desempenho entre classificadores simples.

> **Este projeto é uma ferramenta de apoio e não substitui o laudo
> histopatológico nem a avaliação médica especializada.**

---

## ⚠️ Nota para quem for desenvolver neste projeto

Antes de implementar qualquer mudança, correção ou nova funcionalidade,
**consulte a documentação do projeto** — especialmente:

- os **diagramas** de arquitetura e de dados;
- os **requisitos funcionais e não funcionais**;
- as **histórias de usuário**.

Essa documentação define o escopo, as prioridades (em especial a
minimização de falsos negativos) e as decisões de arquitetura já
validadas com o orientador. Qualquer alteração deve ser coerente com
ela — e, se a mudança contradisser algo documentado, a documentação
deve ser atualizada junto com o código, não ignorada.

---

## Sobre o projeto

O objetivo científico é comparar o desempenho de pelo menos dois
classificadores simples (regressão logística e SVM) treinados sobre uma
base pública de referência (Wisconsin Diagnostic Breast Cancer Dataset —
WDBC/UCI), usando acurácia, sensibilidade, especificidade e AUC como
métricas de comparação.

O MVP expõe isso por trás de um formulário: o usuário informa as
características quantitativas de uma biópsia e recebe uma classificação
(benigno/maligno) com grau de confiança, além de um painel comparando os
classificadores treinados.

### Requisitos funcionais

- Formulário com as características quantitativas da biópsia usadas na
  classificação.
- Classificação do caso entre tumor benigno e maligno a partir das
  características informadas.
- Exibição do grau de confiança da classificação retornada.
- Treino e comparação de pelo menos dois classificadores simples sobre a
  mesma base de referência.
- Exibição da comparação de desempenho (acurácia, sensibilidade,
  especificidade, AUC) entre os classificadores treinados.
- Exibição das características mais relevantes (feature importance) do
  modelo de melhor desempenho.
- Histórico das avaliações realizadas, sem exigir identificação de
  paciente quando não necessário.

### Requisitos não funcionais

- Aviso permanente e visível de que a classificação é uma ferramenta de
  apoio e não substitui o laudo histopatológico e a avaliação médica.
- Classificação de um caso concluída em até 2 segundos.
- Divisão treino/teste e hiperparâmetros dos classificadores documentados
  e versionados, para reprodutibilidade do artigo.
- Priorização da minimização de falsos negativos (casos malignos
  classificados como benignos), com essa taxa reportada explicitamente.
- Quando usado de forma identificada, dados armazenados criptografados,
  em conformidade com a LGPD.

---

## Arquitetura

Padrão C — Web/API + Serviço de ML:

```
Formulário (React)
      │
      ▼
POST /api/biopsias/classificar ──► Serviço de ML (scikit-learn)
      │                                  │
      ▼                                  ▼
  PostgreSQL  ◄──────────────  Avaliacao / ModeloTreinado
      ▲
      │
GET /api/metricas
```

- **Serviço de ML** (Python/scikit-learn): treina e avalia regressão
  logística e SVM sobre o dataset WDBC.
- **API** (Flask): expõe a classificação e as métricas dos modelos.
- **Persistência** (PostgreSQL): histórico de avaliações realizadas e
  métricas de cada modelo treinado.
- **Frontend** (React): formulário de entrada e painel comparativo dos
  classificadores.

## Estrutura do código

```
app.py                  # aplicação Flask (API + conexão com o banco)
requirements.txt
.gitignore
data/
  wdbc.data              # base pública WDBC (UCI), sem cabeçalho
  wdbc.names             # descrição das colunas do dataset
ml/
  __init__.py
  data.py                # carregamento e preparação do WDBC
  train.py               # treino e avaliação dos modelos (US04)
  predict.py              # classificação de um novo caso (US02, US03)
  artifacts/               # gerado pelo treino: modelos .joblib + metrics.json
api/
  __init__.py
  routes.py               # POST /api/biopsias/classificar (US01, US02, US03)
```

Não vão para o repositório (ver `.gitignore`): `.venv/`, `.idea/`,
`__pycache__/` e `.env`.

### `ml/data.py`

Lê o `wdbc.data` (CSV sem cabeçalho: ID, diagnóstico M/B, 30
características) e retorna `X`, `y` e os nomes das features
(`FEATURE_COLUMNS`). O rótulo é invertido em relação ao arquivo original
para que `y = 1` signifique **maligno** — isso simplifica o cálculo de
sensibilidade e da taxa de falsos negativos, tratando maligno como
classe positiva.

### `ml/train.py` (Sprint 1 — US04)

Faz o split treino/teste (80/20, estratificado, `random_state` fixo para
reprodutibilidade — RNF03), padroniza as features com `StandardScaler`,
treina regressão logística e SVM, e calcula acurácia, sensibilidade,
especificidade, AUC e taxa de falsos negativos para cada um. Salva os
modelos (`.joblib`) e as métricas (`metrics.json`) em `ml/artifacts/`.

### `ml/predict.py` (Sprint 1 — US02, US03)

Carrega o scaler e o modelo treinados e classifica uma lista de
características, retornando a classificação (benigno/maligno) e o grau
de confiança.

### `api/routes.py` (Sprint 1 — US01, US02, US03)

`POST /api/biopsias/classificar` recebe as 30 características via JSON,
valida que todas foram enviadas e são numéricas, classifica e retorna o
resultado com o grau de confiança e o aviso de que é uma ferramenta de
apoio. **Ainda não grava no banco** — isso é a US07 (histórico das
avaliações), prevista para a Sprint 2.

### `app.py`

Cria a aplicação Flask, conecta ao PostgreSQL via
`SQLALCHEMY_DATABASE_URI` e registra o blueprint de `api/routes.py`.

---

## Como rodar

1. Instalar as dependências:
   ```
   pip install -r requirements.txt
   ```

2. Ter um PostgreSQL rodando localmente, com um banco criado (ex.:
   `tcc_cancer_mama`), e configurar a string de conexão em `app.py`
   (`SQLALCHEMY_DATABASE_URI`).

3. Treinar e comparar os classificadores:
   ```
   python -m ml.train
   ```
   Isso gera `ml/artifacts/` com os modelos e as métricas.

4. Subir a API:
   ```
   python app.py
   ```

5. Testar a classificação (exemplo com curl, usando as 30 características
   nos nomes de `ml/data.py FEATURE_COLUMNS`):
   ```
   curl -X POST http://localhost:5000/api/biopsias/classificar \
        -H "Content-Type: application/json" \
        -d '{"mean_radius": 17.99, "mean_texture": 10.38, ...}'
   ```

---

## Progresso por sprint

- ✅ **Sprint 0** — HT01 (pipeline WDBC), HT02 (estrutura da app + ambiente
  Python). HT03 (perfis de médico/pesquisador) fica com outro integrante
  do grupo.
- ✅ **Sprint 1** — US01, US02, US03, US04: formulário recebido via API,
  classificação com grau de confiança, comparação entre regressão
  logística e SVM.
- ⬜ **Sprint 2** — US05, US06, US07: métricas completas (incluindo
  feature importance), reprodutibilidade documentada, histórico de
  avaliações persistido no PostgreSQL (`Avaliacao`, `ModeloTreinado`).
- ⬜ **Sprint 3** — endpoint `GET /api/metricas`, integração com o
  frontend React, validação do tempo de resposta (< 2s), testes finais.

## O que ainda falta implementar

- [ ] Persistência de `Avaliacao` e `ModeloTreinado` no PostgreSQL (US07)
- [ ] Endpoint `GET /api/metricas`
- [ ] Feature importance do modelo de melhor desempenho (US06)
- [ ] Criptografia de dados identificados (LGPD), caso o formulário passe
      a aceitar identificação opcional
- [ ] Medição e garantia do tempo de resposta (< 2s) do endpoint de
      classificação
- [ ] Testes automatizados
- [ ] Frontend React (formulário + painel comparativo)
