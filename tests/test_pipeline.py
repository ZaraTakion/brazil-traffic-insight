import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline import clean_data


class CleanDataTests(unittest.TestCase):
    def test_pipeline_downloads_and_creates_clean_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upstream = root / "upstream.csv"
            pd.DataFrame(
                [
                    {
                        "UF": " sp ",
                        "municipio": "sao paulo",
                        "data_inversa": "2023-01-02",
                        "tipo_pista": "Simples",
                        "fase_dia": "Pleno dia",
                        "condicao_metereologica": "Céu claro",
                        "tipo_acidente": "Colisão",
                        "classificacao_acidente": "Com vítimas",
                    },
                    {
                        "UF": "RJ",
                        "municipio": "Rio de Janeiro",
                        "data_inversa": "invalid",
                        "tipo_pista": "Dupla",
                        "fase_dia": "Noite",
                        "condicao_metereologica": "Chuva",
                        "tipo_acidente": "Saída de leito carroçável",
                        "classificacao_acidente": "Sem vítimas",
                    },
                ]
            ).to_csv(upstream, index=False)
            raw = root / "data/raw/accidents_brazil.csv"
            clean = root / "data/processed/accidents_clean.csv"

            def fake_download(dataset, path):
                return str(upstream)

            result = clean_data(raw_path=raw, clean_path=clean, downloader=fake_download)

            self.assertTrue(raw.is_file())
            self.assertTrue(clean.is_file())
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["uf"], "SP")
            self.assertEqual(result.iloc[0]["municipio"], "Sao Paulo")
            self.assertIn("classificacao_acidente", pd.read_csv(clean).columns)

    def test_schema_mismatch_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw.csv"
            pd.DataFrame({"city": ["X"], "state": ["Y"]}).to_csv(raw, index=False)
            with self.assertRaisesRegex(ValueError, "does not match"):
                clean_data(raw_path=raw, clean_path=root / "clean.csv")


if __name__ == "__main__":
    unittest.main()
