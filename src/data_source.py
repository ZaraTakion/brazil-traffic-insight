"""Download the upstream accident dataset on demand."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

DATASET_HANDLE = "mlippo/car-accidents-in-brazil-2017-2023"
SOURCE_FILENAME = "accidents_2017_to_2023_portugues.csv"


def ensure_raw_data(
    destination: Path,
    downloader: Callable[..., str] | None = None,
) -> Path:
    """Return a local raw CSV, downloading it from KaggleHub if necessary.

    ``downloader`` is injectable so the behavior can be tested without network
    access or Kaggle credentials.
    """
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "KaggleHub is required to fetch the dataset. Install project "
                "dependencies with `python -m pip install -r requirements.txt`, "
                "or place the source CSV at "
                f"{destination.as_posix()!r}."
            ) from exc
        downloader = kagglehub.dataset_download

    try:
        downloaded_path = Path(downloader(DATASET_HANDLE, path=SOURCE_FILENAME))
    except Exception as exc:
        raise RuntimeError(
            "Could not download the Brazil traffic dataset from Kaggle. Check "
            "your internet connection and Kaggle dataset access/terms. If Kaggle "
            "requires consent or authentication, configure KaggleHub locally; "
            "alternatively, download the CSV manually and save it as "
            f"{destination.as_posix()!r}. No credentials are stored by this project."
        ) from exc

    if not downloaded_path.is_file() or downloaded_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"KaggleHub did not return a non-empty {SOURCE_FILENAME!r}. "
            "Check the upstream dataset version or place the expected CSV at "
            f"{destination.as_posix()!r}."
        )

    shutil.copy2(downloaded_path, destination)
    return destination
