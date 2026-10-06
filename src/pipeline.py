"""Prepare the upstream accident CSV for the notebook, model, and dashboard."""

from pathlib import Path

import pandas as pd

from src.data_source import ensure_raw_data

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/accidents_brazil.csv"
CLEAN = ROOT / "data/processed/accidents_clean.csv"

REQUIRED_COLUMNS = {
    "uf",
    "municipio",
    "data_inversa",
    "tipo_pista",
    "fase_dia",
    "condicao_metereologica",
    "tipo_acidente",
    "classificacao_acidente",
}


def clean_data(raw_path: Path = RAW, clean_path: Path = CLEAN, downloader=None):
    """Fetch (if needed), validate, normalize, and save the clean dataset."""
    raw_path = ensure_raw_data(raw_path, downloader=downloader)
    df = pd.read_csv(raw_path)
    print("Antes:", df.shape)

    df.columns = df.columns.str.lower().str.strip()
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "The source CSV does not match the columns used by this project. "
            f"Missing: {', '.join(missing)}. Available: {', '.join(df.columns)}"
        )

    df.dropna(subset=["uf", "municipio"], inplace=True)
    df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    df = df[df["data_inversa"].notna()].copy()
    df["uf"] = df["uf"].astype(str).str.upper().str.strip()
    df["municipio"] = df["municipio"].astype(str).str.title().str.strip()
    df.drop_duplicates(inplace=True)

    clean_path = Path(clean_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    print("Depois:", df.shape)
    print("Arquivo limpo salvo em:", clean_path)
    return df


if __name__ == "__main__":
    clean_data()
