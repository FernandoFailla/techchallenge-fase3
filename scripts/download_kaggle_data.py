"""Download the Kaggle source without replacing versioned DVC pointers."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import kagglehub

DATASET_HANDLE = "alanjafari/kurmed-triage/versions/1"
SOURCE_FILENAME = "synthetic_v1.csv"
TARGET_PATH = Path("data/raw") / SOURCE_FILENAME


def _source_path(download_directory: Path) -> Path:
    matches = tuple(download_directory.rglob(SOURCE_FILENAME))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {SOURCE_FILENAME} in the Kaggle dataset, found "
            f"{len(matches)}."
        )
    return matches[0]


def main() -> None:
    """Copy the public source CSV into the ignored raw-data path."""
    with TemporaryDirectory(prefix="techchallenge-kaggle-") as temporary_directory:
        download_directory = Path(temporary_directory)
        kagglehub.dataset_download(DATASET_HANDLE, output_dir=download_directory)
        source_path = _source_path(download_directory)
        TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source_path, TARGET_PATH)
    print(f"Downloaded {SOURCE_FILENAME} to {TARGET_PATH}.")


if __name__ == "__main__":
    main()
