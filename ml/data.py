import os

import wandb
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