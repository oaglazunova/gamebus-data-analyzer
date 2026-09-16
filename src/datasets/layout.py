from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"

COHORT_MANIFEST_FILENAME = "cohort_manifest.json"
EXTRACTION_MANIFEST_FILENAME = "extraction_manifest.json"
CAMPAIGN_USERS_FILENAME = "campaign_users.json"
RAW_DATA_DIRNAME = "data_raw"
ANALYSIS_DIRNAME = "data_analysis"


def _safe_name_part(value: str) -> str:
    """
    Convert a campaign abbreviation into a filesystem-safe name
    while keeping it human-readable.
    """
    value = value.strip()

    if not value:
        raise ValueError("Campaign abbreviation cannot be empty")

    value = re.sub(
        r'[<>:"/\\|?*]+',
        "-",
        value,
    )

    value = re.sub(
        r"\s+",
        "_",
        value,
    )

    return value.strip("._-")


def build_dataset_name(
    campaign_abbreviation: str,
    campaign_id: int | str,
    extracted_at: datetime | None = None,
) -> str:
    """
    Build the standard dataset folder name:

        ABBREVIATION_CAMPAIGNID_YYYY-MM-DD_HHMM

    Example:

        UNISG_379_2026-09-11_1343
    """
    abbreviation = _safe_name_part(
        campaign_abbreviation
    )

    campaign_id_text = str(
        campaign_id
    ).strip()

    if not campaign_id_text:
        raise ValueError(
            "Campaign ID cannot be empty"
        )

    timestamp = (
        extracted_at
        if extracted_at is not None
        else datetime.now()
    )

    return (
        f"{abbreviation}_"
        f"{campaign_id_text}_"
        f"{timestamp:%Y-%m-%d_%H%M}"
    )


def create_dataset_directory(
    campaign_abbreviation: str,
    campaign_id: int | str,
    extracted_at: datetime | None = None,
    datasets_dir: Path = DATASETS_DIR,
) -> Path:
    """
    Create one self-contained dataset directory.

    Original GameBus filenames placed in this directory
    are not renamed.
    """
    dataset_name = build_dataset_name(
        campaign_abbreviation=campaign_abbreviation,
        campaign_id=campaign_id,
        extracted_at=extracted_at,
    )

    dataset_dir = (
        Path(datasets_dir)
        / dataset_name
    )

    if dataset_dir.exists():
        raise FileExistsError(
            "Dataset directory already exists: "
            f"{dataset_dir}"
        )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        dataset_dir
        / RAW_DATA_DIRNAME
    ).mkdir()

    return dataset_dir


def get_raw_data_dir(
    dataset_dir: Path,
) -> Path:
    return (
        Path(dataset_dir)
        / RAW_DATA_DIRNAME
    )


def get_analysis_dir(
    dataset_dir: Path,
) -> Path:
    return (
        Path(dataset_dir)
        / ANALYSIS_DIRNAME
    )


def get_cohort_manifest_path(
    dataset_dir: Path,
) -> Path:
    return (
        Path(dataset_dir)
        / COHORT_MANIFEST_FILENAME
    )


def get_extraction_manifest_path(
    dataset_dir: Path,
) -> Path:
    return (
        Path(dataset_dir)
        / EXTRACTION_MANIFEST_FILENAME
    )


def get_campaign_users_path(
    dataset_dir: Path,
) -> Path:
    return (
        Path(dataset_dir)
        / CAMPAIGN_USERS_FILENAME
    )