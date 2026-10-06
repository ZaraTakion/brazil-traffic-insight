import tempfile
import unittest
from pathlib import Path

from src.data_source import SOURCE_FILENAME, ensure_raw_data


class EnsureRawDataTests(unittest.TestCase):
    def test_existing_raw_file_skips_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "data/raw/accidents_brazil.csv"
            destination.parent.mkdir(parents=True)
            destination.write_text("existing,data\n1,2\n", encoding="utf-8")

            def unexpected_download(*args, **kwargs):
                raise AssertionError("download should not be called")

            self.assertEqual(ensure_raw_data(destination, unexpected_download), destination)

    def test_downloaded_csv_is_copied_to_expected_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upstream = root / SOURCE_FILENAME
            upstream.write_text("uf,municipio\nSP,Sao Paulo\n", encoding="utf-8")
            destination = root / "data/raw/accidents_brazil.csv"

            def fake_download(dataset, path):
                self.assertEqual(path, SOURCE_FILENAME)
                return str(upstream)

            self.assertEqual(ensure_raw_data(destination, fake_download), destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), upstream.read_text(encoding="utf-8"))

    def test_download_failure_has_manual_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "data/raw/accidents_brazil.csv"

            def failed_download(*args, **kwargs):
                raise OSError("offline")

            with self.assertRaisesRegex(RuntimeError, "download the CSV manually"):
                ensure_raw_data(destination, failed_download)


if __name__ == "__main__":
    unittest.main()
