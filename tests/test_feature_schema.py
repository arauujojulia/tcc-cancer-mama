"""
Testes unitários de ml/data.py: feature_schema() (US01, design.md §1.3).
"""
import unittest

from ml.data import FEATURE_COLUMNS, MARGEM_FAIXA, feature_schema


class TestFeatureSchema(unittest.TestCase):
    def setUp(self):
        self.schema = feature_schema()

    def test_devolve_30_campos(self):
        self.assertEqual(self.schema["n_caracteristicas"], 30)
        self.assertEqual(len(self.schema["campos"]), 30)

    def test_margem_e_20_por_cento(self):
        self.assertEqual(self.schema["margem_faixa"], MARGEM_FAIXA)
        self.assertEqual(MARGEM_FAIXA, 0.20)

    def test_cada_campo_tem_label_unit_exemplo(self):
        for campo in self.schema["campos"]:
            self.assertIn("label", campo)
            self.assertIsInstance(campo["label"], str)
            self.assertTrue(campo["label"])
            self.assertIn("unit", campo)
            self.assertIsNone(campo["unit"])  # WDBC não publica unidades físicas
            self.assertIn("exemplo", campo)
            self.assertIsInstance(campo["exemplo"], float)

    def test_aliases_min_max_batem_com_observado(self):
        for campo in self.schema["campos"]:
            self.assertEqual(campo["min"], campo["min_observado"])
            self.assertEqual(campo["max"], campo["max_observado"])

    def test_faixa_aceita_tem_margem_de_20_por_cento_da_amplitude(self):
        for campo in self.schema["campos"]:
            amplitude = campo["max_observado"] - campo["min_observado"]
            folga_esperada = amplitude * MARGEM_FAIXA
            self.assertAlmostEqual(campo["max_aceito"] - campo["max_observado"], folga_esperada, places=6)
            folga_min = campo["min_observado"] - campo["min_aceito"]
            self.assertLessEqual(folga_min, folga_esperada + 1e-9)
            self.assertGreaterEqual(campo["min_aceito"], 0.0)  # nunca negativo

    def test_nomes_dos_campos_cobrem_feature_columns(self):
        nomes = {c["nome"] for c in self.schema["campos"]}
        self.assertEqual(nomes, set(FEATURE_COLUMNS))

    def test_exemplos_maligno_e_benigno_presentes(self):
        self.assertIn("maligno", self.schema["exemplos"])
        self.assertIn("benigno", self.schema["exemplos"])
        for nome in FEATURE_COLUMNS:
            self.assertIn(nome, self.schema["exemplos"]["maligno"])
            self.assertIn(nome, self.schema["exemplos"]["benigno"])

    def test_label_exemplo_de_mean_radius(self):
        campo = next(c for c in self.schema["campos"] if c["nome"] == "mean_radius")
        self.assertEqual(campo["label"], "Raio (média)")


if __name__ == "__main__":
    unittest.main()
