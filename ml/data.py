import os
from functools import lru_cache

import pandas as pd

# Caminho do arquivo de dados. Ajuste se você mover o wdbc.data para
# outra pasta do projeto (ex.: uma pasta "data/").
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wdbc.data")

_MEDIDAS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension",
]
COLUMN_NAMES = (
    ["id", "diagnosis"]
    + [f"mean_{m}" for m in _MEDIDAS]
    + [f"se_{m}" for m in _MEDIDAS]
    + [f"worst_{m}" for m in _MEDIDAS]
)
FEATURE_COLUMNS = COLUMN_NAMES[2:]  # tudo, exceto id e diagnosis

# Margem aplicada às faixas observadas na base para validar entradas
# (documentação, fluxo 2: "faixas do schema + margem de 20%").
MARGEM_FAIXA = 0.20

# Rótulos em português das 10 medidas base e dos 3 agregados do WDBC, usados
# para montar o `label` de cada característica no schema.
MEDIDAS_PT = {
    "radius": "Raio", "texture": "Textura", "perimeter": "Perímetro", "area": "Área",
    "smoothness": "Suavidade", "compactness": "Compacidade", "concavity": "Concavidade",
    "concave_points": "Pontos côncavos", "symmetry": "Simetria",
    "fractal_dimension": "Dimensão fractal",
}
GRUPOS_PT = {"mean": "média", "se": "erro padrão", "worst": "pior valor"}


def rotulo_caracteristica(nome: str) -> str:
    """Rótulo legível em português, ex.: 'mean_radius' -> 'Raio (média)'."""
    grupo, medida = nome.split("_", 1)
    return f"{MEDIDAS_PT.get(medida, medida)} ({GRUPOS_PT.get(grupo, grupo)})"


def load_data(path: str = DATA_PATH):
    """
    Retorna (X, y, feature_names):
      - X: matriz de características (569 x 30)
      - y: rótulo (1 = maligno, 0 = benigno)
      - feature_names: nomes das 30 colunas de X, na mesma ordem
    """
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    X = df[FEATURE_COLUMNS].values
    y = (df["diagnosis"] == "M").astype(int).values  # 1 = maligno, 0 = benigno

    return X, y, FEATURE_COLUMNS


@lru_cache(maxsize=1)
def feature_schema():
    """
    Schema das 30 características (US01): para cada uma, o intervalo
    observado na base WDBC (min/max) e o intervalo aceito na validação
    (observado ± MARGEM_FAIXA da amplitude, nunca abaixo de zero, pois
    todas as medidas do WDBC são não negativas).

    Cada campo expõe `label` (rótulo em português), `unit`, `min`/`max` (a faixa
    observada) e `exemplo` (o valor do caso maligno de demonstração). Também
    devolve um exemplo real de cada classe, para preencher o formulário inteiro.
    """
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)

    def _exemplo(diag):
        linha = df[df["diagnosis"] == diag].iloc[0]
        return {n: float(linha[n]) for n in FEATURE_COLUMNS}

    exemplos = {"maligno": _exemplo("M"), "benigno": _exemplo("B")}

    campos = []
    for nome in FEATURE_COLUMNS:
        lo, hi = float(df[nome].min()), float(df[nome].max())
        folga = (hi - lo) * MARGEM_FAIXA
        campos.append({
            "nome": nome,
            "label": rotulo_caracteristica(nome),
            # As medidas do WDBC são adimensionais ou derivadas (forma e textura
            # em pixels e razões); a base não publica unidades físicas, então
            # `unit` é sempre nulo e o frontend simplesmente não exibe unidade.
            "unit": None,
            "grupo": nome.split("_", 1)[0],          # mean | se | worst
            "medida": nome.split("_", 1)[1],
            # min/max do contrato = faixa observada na base
            "min": lo,
            "max": hi,
            "min_observado": lo,
            "max_observado": hi,
            # faixa efetivamente aceita na validação (observada ± margem)
            "min_aceito": max(0.0, lo - folga),
            "max_aceito": hi + folga,
            "exemplo": exemplos["maligno"][nome],
        })

    return {
        "n_caracteristicas": len(campos),
        "margem_faixa": MARGEM_FAIXA,
        "campos": campos,
        "exemplos": exemplos,
    }
