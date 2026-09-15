from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

from src.datasets.layout import (
    get_analysis_dir,
    get_cohort_manifest_path,
    get_extraction_manifest_path,
    get_raw_data_dir,
)
from src.datasets.manifests import (
    read_manifest,
)


EXPECTED_ANALYTICS_FILES = {
    "1-aggregated-data.csv",
    "2-activities.csv",
    "3-navigation-events.csv",
    "4-notification-events.csv",
    "5-sensor-events.csv",
}


class DatasetValidationError(
    ValueError
):
    pass


@dataclass(
    frozen=True
)
class DatasetInspection:
    dataset_dir: Path

    campaign_abbreviation: str | None
    campaign_id: str | None

    campaign_data_path: Path
    campaign_description_path: Path

    extraction_manifest: dict | None
    cohort_manifest: dict | None

    raw_data_dir: Path
    analysis_dir: Path

    @property
    def has_extraction_manifest(
        self,
    ) -> bool:
        return (
            self.extraction_manifest
            is not None
        )

    @property
    def has_cohort_manifest(
        self,
    ) -> bool:
        return (
            self.cohort_manifest
            is not None
        )

    @property
    def participant_count(
        self,
    ) -> int:
        if (
            self.cohort_manifest
            is None
        ):
            return 0

        return len(
            self.cohort_manifest.get(
                "participants",
                [],
            )
        )

    @property
    def selected_for_analysis_count(
        self,
    ) -> int:
        if (
            self.cohort_manifest
            is None
        ):
            return 0

        return sum(
            1
            for participant in (
                self.cohort_manifest.get(
                    "participants",
                    [],
                )
            )
            if participant.get(
                "selected_for_analysis",
                False,
            )
        )

    @property
    def raw_json_count(
        self,
    ) -> int:
        if not self.raw_data_dir.exists():
            return 0

        return len(
            list(
                self.raw_data_dir.glob(
                    "*.json"
                )
            )
        )


def _validate_dataset_directory(
    dataset_dir: Path,
) -> Path:
    dataset_dir = Path(
        dataset_dir
    ).expanduser()

    if not dataset_dir.exists():
        raise DatasetValidationError(
            "Dataset folder does not exist: "
            f"{dataset_dir}"
        )

    if not dataset_dir.is_dir():
        raise DatasetValidationError(
            "Selected path is not a folder: "
            f"{dataset_dir}"
        )

    return dataset_dir.resolve()


def _zip_contains_campaign_data(
    path: Path,
) -> bool:
    try:
        with zipfile.ZipFile(
            path,
            "r",
        ) as archive:
            members = {
                Path(name).name.casefold()
                for name in archive.namelist()
                if not name.endswith("/")
            }

    except (
        zipfile.BadZipFile,
        OSError,
    ):
        return False

    # Activities are the minimum input needed
    # to recognise this as a GameBus campaign
    # analytics export.
    return (
        "2-activities.csv"
        in members
    )


def _campaign_data_from_manifest(
    dataset_dir: Path,
    extraction_manifest: dict,
) -> Path:
    campaign_files = (
        extraction_manifest.get(
            "campaign_files",
            {},
        )
    )

    filename = campaign_files.get(
        "data_export"
    )

    if not filename:
        raise DatasetValidationError(
            "extraction_manifest.json does not "
            "specify the campaign data export."
        )

    path = (
        dataset_dir
        / str(
            filename
        )
    )

    if not path.exists():
        raise DatasetValidationError(
            "Campaign data export listed in "
            "extraction_manifest.json was not "
            "found: "
            f"{path.name}"
        )

    if not _zip_contains_campaign_data(
        path
    ):
        raise DatasetValidationError(
            "Campaign data export is not a "
            "valid GameBus analytics ZIP: "
            f"{path.name}"
        )

    return path


def _campaign_description_from_manifest(
    dataset_dir: Path,
    extraction_manifest: dict,
) -> Path:
    campaign_files = (
        extraction_manifest.get(
            "campaign_files",
            {},
        )
    )

    filename = campaign_files.get(
        "description"
    )

    if not filename:
        raise DatasetValidationError(
            "extraction_manifest.json does not "
            "specify the campaign description."
        )

    path = (
        dataset_dir
        / str(
            filename
        )
    )

    if not path.exists():
        raise DatasetValidationError(
            "Campaign description listed in "
            "extraction_manifest.json was not "
            "found: "
            f"{path.name}"
        )

    if path.suffix.casefold() != ".xlsx":
        raise DatasetValidationError(
            "Campaign description is not an "
            "XLSX file: "
            f"{path.name}"
        )

    return path


def _discover_legacy_campaign_data(
    dataset_dir: Path,
) -> Path:
    candidates = [
        path
        for path in dataset_dir.glob(
            "*.zip"
        )
        if _zip_contains_campaign_data(
            path
        )
    ]

    if not candidates:
        raise DatasetValidationError(
            "No GameBus campaign analytics ZIP "
            "was found in the selected folder."
        )

    if len(candidates) > 1:
        names = ", ".join(
            sorted(
                path.name
                for path in candidates
            )
        )

        raise DatasetValidationError(
            "More than one possible campaign "
            "analytics ZIP was found. "
            f"Candidates: {names}"
        )

    return candidates[0]


def _discover_legacy_campaign_description(
    dataset_dir: Path,
) -> Path:
    candidates = []

    for path in dataset_dir.glob(
        "*.xlsx"
    ):
        name = path.name.casefold()

        # Participant credential files are not
        # campaign descriptions.
        if (
            "user" in name
            or "credential" in name
        ):
            continue

        candidates.append(
            path
        )

    if not candidates:
        raise DatasetValidationError(
            "No campaign description XLSX "
            "was found in the selected folder."
        )

    if len(candidates) > 1:
        names = ", ".join(
            sorted(
                path.name
                for path in candidates
            )
        )

        raise DatasetValidationError(
            "More than one possible campaign "
            "description XLSX was found. "
            f"Candidates: {names}"
        )

    return candidates[0]


def _read_optional_manifest(
    path: Path,
) -> dict | None:
    if not path.exists():
        return None

    try:
        return read_manifest(
            path
        )

    except Exception as exc:
        raise DatasetValidationError(
            f"Could not read {path.name}: "
            f"{exc}"
        ) from exc


def _campaign_metadata(
    extraction_manifest: dict | None,
    cohort_manifest: dict | None,
) -> tuple[
    str | None,
    str | None,
]:
    abbreviation = None
    campaign_id = None

    if extraction_manifest is not None:
        campaign = (
            extraction_manifest.get(
                "campaign",
                {},
            )
        )

        abbreviation = campaign.get(
            "abbreviation"
        )

        campaign_id = campaign.get(
            "id"
        )

    if cohort_manifest is not None:
        cohort_campaign = (
            cohort_manifest.get(
                "campaign",
                {},
            )
        )

        cohort_abbreviation = (
            cohort_campaign.get(
                "abbreviation"
            )
        )

        cohort_campaign_id = (
            cohort_campaign.get(
                "id"
            )
        )

        if (
            campaign_id is not None
            and cohort_campaign_id is not None
            and str(
                campaign_id
            )
            != str(
                cohort_campaign_id
            )
        ):
            raise DatasetValidationError(
                "Campaign ID in "
                "cohort_manifest.json does not "
                "match extraction_manifest.json."
            )

        if (
            abbreviation is not None
            and cohort_abbreviation is not None
            and str(
                abbreviation
            )
            != str(
                cohort_abbreviation
            )
        ):
            raise DatasetValidationError(
                "Campaign abbreviation in "
                "cohort_manifest.json does not "
                "match extraction_manifest.json."
            )

        if abbreviation is None:
            abbreviation = (
                cohort_abbreviation
            )

        if campaign_id is None:
            campaign_id = (
                cohort_campaign_id
            )

    return (
        (
            str(
                abbreviation
            )
            if abbreviation is not None
            else None
        ),
        (
            str(
                campaign_id
            )
            if campaign_id is not None
            else None
        ),
    )


def inspect_dataset(
    dataset_dir: Path,
) -> DatasetInspection:
    dataset_dir = (
        _validate_dataset_directory(
            dataset_dir
        )
    )

    extraction_manifest_path = (
        get_extraction_manifest_path(
            dataset_dir
        )
    )

    cohort_manifest_path = (
        get_cohort_manifest_path(
            dataset_dir
        )
    )

    extraction_manifest = (
        _read_optional_manifest(
            extraction_manifest_path
        )
    )

    cohort_manifest = (
        _read_optional_manifest(
            cohort_manifest_path
        )
    )

    if extraction_manifest is not None:
        campaign_data_path = (
            _campaign_data_from_manifest(
                dataset_dir,
                extraction_manifest,
            )
        )

        campaign_description_path = (
            _campaign_description_from_manifest(
                dataset_dir,
                extraction_manifest,
            )
        )

    else:
        campaign_data_path = (
            _discover_legacy_campaign_data(
                dataset_dir
            )
        )

        campaign_description_path = (
            _discover_legacy_campaign_description(
                dataset_dir
            )
        )

    (
        campaign_abbreviation,
        campaign_id,
    ) = _campaign_metadata(
        extraction_manifest,
        cohort_manifest,
    )

    return DatasetInspection(
        dataset_dir=(
            dataset_dir
        ),
        campaign_abbreviation=(
            campaign_abbreviation
        ),
        campaign_id=(
            campaign_id
        ),
        campaign_data_path=(
            campaign_data_path
        ),
        campaign_description_path=(
            campaign_description_path
        ),
        extraction_manifest=(
            extraction_manifest
        ),
        cohort_manifest=(
            cohort_manifest
        ),
        raw_data_dir=(
            get_raw_data_dir(
                dataset_dir
            )
        ),
        analysis_dir=(
            get_analysis_dir(
                dataset_dir
            )
        ),
    )