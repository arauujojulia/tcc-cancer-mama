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

    Também devolve um exemplo real de cada classe, para demonstração.
    """
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    campos = []
    for nome in FEATURE_COLUMNS:
        lo, hi = float(df[nome].min()), float(df[nome].max())
        folga = (hi - lo) * MARGEM_FAIXA
        campos.append({
            "nome": nome,
            "grupo": nome.split("_", 1)[0],          # mean | se | worst
            "medida": nome.split("_", 1)[1],
            "min_observado": lo,
            "max_observado": hi,
            "min_aceito": max(0.0, lo - folga),
            "max_aceito": hi + folga,
        })

    def _exemplo(diag):
        linha = df[df["diagnosis"] == diag].iloc[0]
        return {n: float(linha[n]) for n in FEATURE_COLUMNS}

    return {
        "n_caracteristicas": len(campos),
        "margem_faixa": MARGEM_FAIXA,
        "campos": campos,
        "exemplos": {"maligno": _exemplo("M"), "benigno": _exemplo("B")},
    }
