"""
Testes unitários de ml/train.py: escolher_limiar_seguranca().

Casos sintéticos: quando existe um limiar <= 0,5 que mantém sensibilidade
>= meta, deve escolher o MAIOR desses limiares; quando nenhum atinge a meta,
cai no menor limiar da grade (máxima cautela) e sinaliza atingiu_meta=False.
O conjunto usado aqui simula sempre predições OUT-OF-FOLD do treino (nunca
o conjunto de teste).
"""
import unittest

import numpy as np

from ml.train import escolher_limiar_seguranca


class TestEscolherLimiarSeguranca(unittest.TestCase):
    def test_escolhe_o_maior_limiar_que_atinge_a_meta(self):
        # 10 malignos (y=1) com proba variando de 0.05 a 0.95 em passos de 0.10;
        # 10 benignos (y=0) com proba baixa. Em t=0.10, todos os 10 malignos
        # com proba >= 0.10 (9 deles) dão sensibilidade 0.9; a meta de 0.97
        # só é atingida em limiares bem baixos.
        y_true = np.array([1] * 10 + [0] * 10)
        proba_malignos = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        proba_benignos = np.array([0.01] * 10)
        y_proba = np.concatenate([proba_malignos, proba_benignos])

        limiar, atingiu = escolher_limiar_seguranca(y_true, y_proba, sens_minima=0.9)
        # sensibilidade >= 0.9 exige pelo menos 9 dos 10 malignos capturados;
        # o maior limiar que ainda inclui 9 malignos é 0.15 (exclui só o de 0.05).
        self.assertTrue(atingiu)
        self.assertAlmostEqual(limiar, 0.15, places=2)

    def test_quando_nenhum_limiar_atinge_a_meta_usa_o_menor_e_sinaliza(self):
        # 1 em cada 10 malignos tem probabilidade exatamente 0 (abaixo de
        # qualquer limiar da grade, inclusive o menor, 0.01): a sensibilidade
        # máxima possível é 9/10 = 0.9, que nunca atinge a meta de 0,97.
        y_true = np.array([1] * 10 + [0] * 10)
        proba_malignos = np.array([0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        proba_benignos = np.array([0.01] * 10)
        y_proba = np.concatenate([proba_malignos, proba_benignos])

        limiar, atingiu = escolher_limiar_seguranca(y_true, y_proba, sens_minima=0.97)
        self.assertFalse(atingiu)
        self.assertAlmostEqual(limiar, 0.01, places=2)

    def test_limiar_nunca_excede_0_5(self):
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.99, 0.98, 0.01, 0.02])
        limiar, atingiu = escolher_limiar_seguranca(y_true, y_proba, sens_minima=0.97)
        self.assertTrue(atingiu)
        self.assertLessEqual(limiar, 0.5)


if __name__ == "__main__":
    unittest.main()
