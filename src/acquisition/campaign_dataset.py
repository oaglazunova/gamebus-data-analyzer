from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from src.acquisition.gamebus_campaigns import (
    DEFAULT_CAMPAIGNS_BASE_URL,
    download_campaign_data_export,
    download_campaign_description,
    extract_campaign_id_from_filename,
)
from src.datasets.layout import (
    DATASETS_DIR,
    create_dataset_directory,
    get_extraction_manifest_path,
)
from src.datasets.manifests import (
    build_extraction_manifest,
    write_manifest,
)


def bootstrap_campaign_dataset(
    *,
    campaign_abbreviation: str,
    organizer_email: str | None = None,
    organizer_password: str | None = None,
    cookie_file: Path | None = None,
    datasets_dir: Path = DATASETS_DIR,
    base_url: str = DEFAULT_CAMPAIGNS_BASE_URL,
    extracted_at: datetime | None = None,
) -> Path:
    """
    Create a new organizer-level GameBus dataset.

    The bootstrap downloads:

      1. campaign-description XLSX;
      2. campaign analytics/data ZIP.

    Both original GameBus filenames are preserved.

    Participant-level extraction is deliberately not performed
    here. That happens only after cohort verification and,
    optionally, provision of the participant credentials XLSX.
    """
    abbreviation = campaign_abbreviation.strip()

    if not abbreviation:
        raise ValueError(
            "Campaign abbreviation cannot be empty."
        )

    timestamp = (
        extracted_at
        if extracted_at is not None
        else datetime.now().astimezone()
    )

    with tempfile.TemporaryDirectory(
        prefix="gamebus_campaign_bootstrap_"
    ) as temp_dir:
        staging_dir = Path(temp_dir)

        description_path = (
            download_campaign_description(
                campaign_abbreviation=abbreviation,
                destination_dir=staging_dir,
                organizer_email=organizer_email,
                organizer_password=organizer_password,
                cookie_file=cookie_file,
                base_url=base_url,
            )
        )

        data_export_path = (
            download_campaign_data_export(
                campaign_abbreviation=abbreviation,
                destination_dir=staging_dir,
                organizer_email=organizer_email,
                organizer_password=organizer_password,
                cookie_file=cookie_file,
                base_url=base_url,
            )
        )

        description_campaign_id = (
            extract_campaign_id_from_filename(
                description_path.name
            )
        )

        export_campaign_id = (
            extract_campaign_id_from_filename(
                data_export_path.name
            )
        )

        if (
            description_campaign_id
            != export_campaign_id
        ):
            raise RuntimeError(
                "Campaign ID mismatch between "
                "description and data export: "
                f"{description_campaign_id} != "
                f"{export_campaign_id}"
            )

        campaign_id = description_campaign_id

        dataset_dir = create_dataset_directory(
            campaign_abbreviation=abbreviation,
            campaign_id=campaign_id,
            extracted_at=timestamp,
            datasets_dir=datasets_dir,
        )

        try:
            final_description_path = (
                dataset_dir
                / description_path.name
            )

            final_data_export_path = (
                dataset_dir
                / data_export_path.name
            )

            shutil.move(
                str(description_path),
                str(final_description_path),
            )

            shutil.move(
                str(data_export_path),
                str(final_data_export_path),
            )

            manifest = build_extraction_manifest(
                campaign_abbreviation=abbreviation,
                campaign_id=campaign_id,
                extracted_at=timestamp,
                campaign_data_filename=(
                    final_data_export_path.name
                ),
                campaign_description_filename=(
                    final_description_path.name
                ),
                credentials_supplied=False,
                credentials_source_filename=None,
            )

            write_manifest(
                get_extraction_manifest_path(
                    dataset_dir
                ),
                manifest,
            )

        except Exception:
            shutil.rmtree(
                dataset_dir,
                ignore_errors=True,
            )
            raise

    return dataset_dir