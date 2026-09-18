from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

from src.analysis.cohort import (
    resolve_analysis_user_ids_from_manifest,
)
from src.analysis.common import (
    ensure_output_dirs,
    logger,
    use_analysis_output_directory,
)
from src.analysis.data_analysis import (
    analyze_activities,
    analyze_day_aggregate_steps,
    analyze_visualizations_challenges_tasks,
)
from src.analysis.geofence_analysis import (
    analyze_geofence_data,
)
from src.analysis.loaders import (
    load_excel_files,
    load_json_files,
)
from src.analysis.activity_metrics import (
    filter_json_data_to_user_ids,
    filter_tabular_data_to_user_ids,
)
from src.analysis.reporting import (
    create_complete_report,
)
from src.datasets.inspection import (
    DatasetInspection,
    inspect_dataset,
)


ProgressCallback = Callable[
    [
        str,
        str,
    ],
    None,
]


class DatasetAnalysisError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True
)
class DatasetAnalysisResult:
    dataset_dir: Path
    output_dir: Path

    selected_participants: int
    campaign_tables_loaded: int
    raw_json_files_loaded: int

    unique_active_users: int | None

    report_path: Path | None


def _notify(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
) -> None:
    if callback is not None:
        callback(
            stage,
            message,
        )


def _prepare_output_directory(
    inspection: DatasetInspection,
) -> Path:
    """
    Regenerate data_analysis/ from scratch.

    Previous analysis output is deliberately removed.
    Downloaded source data and manifests are untouched.
    """
    output_dir = (
        inspection.analysis_dir
        .resolve()
    )

    if output_dir.exists():
        if not output_dir.is_dir():
            raise DatasetAnalysisError(
                "Analysis output path exists but "
                "is not a directory: "
                f"{output_dir}"
            )

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_dir


def run_dataset_analysis(
    dataset_dir: str | Path,
    *,
    progress_callback: (
        ProgressCallback
        | None
    ) = None,
) -> DatasetAnalysisResult:
    """
    Run the existing GameBus analysis pipeline against
    one self-contained dataset folder.

    The authoritative participant cohort comes from
    cohort_manifest.json.

    Analysis never reads participant credentials and
    never changes acquisition history.
    """
    _notify(
        progress_callback,
        "validating",
        "Validating dataset",
    )

    inspection = inspect_dataset(
        Path(
            dataset_dir
        )
    )

    cohort_manifest = (
        inspection.cohort_manifest
    )

    if cohort_manifest is None:
        raise DatasetAnalysisError(
            "This dataset does not have "
            "cohort_manifest.json. "
            "Review and save the participant cohort "
            "before running analysis."
        )

    try:
        participant_ids = (
            resolve_analysis_user_ids_from_manifest(
                cohort_manifest
            )
        )

    except ValueError as exc:
        raise DatasetAnalysisError(
            str(
                exc
            )
        ) from exc

    _notify(
        progress_callback,
        "preparing",
        "Preparing analysis output",
    )

    output_dir = (
        _prepare_output_directory(
            inspection
        )
    )

    report_path: Path | None = None
    unique_active_users: int | None = None

    with use_analysis_output_directory(
        output_dir
    ):
        ensure_output_dirs()

        _notify(
            progress_callback,
            "loading_campaign",
            "Loading campaign data",
        )

        try:
            csv_data = load_excel_files(
                campaign_data_path=(
                    inspection
                    .campaign_data_path
                ),
                campaign_description_path=(
                    inspection
                    .campaign_description_path
                ),
            )

        except Exception as exc:
            raise DatasetAnalysisError(
                "Could not load campaign data: "
                f"{exc}"
            ) from exc

        if not csv_data:
            raise DatasetAnalysisError(
                "No campaign data could be loaded "
                "from the selected dataset."
            )

        _notify(
            progress_callback,
            "loading_participant_data",
            "Loading participant data",
        )

        try:
            json_data = load_json_files(
                raw_data_dir=(
                    inspection
                    .raw_data_dir
                )
            )

        except Exception as exc:
            raise DatasetAnalysisError(
                "Could not load participant "
                f"data: {exc}"
            ) from exc

        campaign_tables_loaded = len(
            csv_data
        )

        raw_json_files_loaded = len(
            json_data
        )

        _notify(
            progress_callback,
            "filtering",
            (
                "Applying analysis cohort "
                f"({len(participant_ids)} participant(s))"
            ),
        )

        try:
            csv_data = (
                filter_tabular_data_to_user_ids(
                    csv_data,
                    participant_ids,
                )
            )

            json_data = (
                filter_json_data_to_user_ids(
                    json_data,
                    participant_ids,
                )
            )

        except Exception as exc:
            raise DatasetAnalysisError(
                "Could not apply the selected "
                f"participant cohort: {exc}"
            ) from exc

        logger.info(
            "Dataset analysis cohort: "
            f"{len(participant_ids)} participant(s)"
        )

        _notify(
            progress_callback,
            "activities",
            "Analyzing campaign activities",
        )

        try:
            activities_result = (
                analyze_activities(
                    csv_data
                )
            )

        except Exception as exc:
            raise DatasetAnalysisError(
                "Activity analysis failed: "
                f"{exc}"
            ) from exc

        campaign_metrics = None

        if activities_result:
            try:
                (
                    _activities,
                    unique_active_users,
                    _dropout_metrics,
                    _joining_metrics,
                    campaign_metrics,
                ) = activities_result

            except Exception as exc:
                raise DatasetAnalysisError(
                    "Activity analysis returned an "
                    "unexpected result."
                ) from exc

        _notify(
            progress_callback,
            "geofence",
            "Analyzing geofence data",
        )

        try:
            analyze_geofence_data(
                json_data
            )

        except Exception as exc:
            logger.exception(
                "Geofence analysis failed"
            )

            raise DatasetAnalysisError(
                "Geofence analysis failed: "
                f"{exc}"
            ) from exc

        _notify(
            progress_callback,
            "steps",
            "Analyzing daily steps",
        )

        try:
            analyze_day_aggregate_steps(
                json_data,
                campaign_metrics,
            )

        except Exception as exc:
            logger.exception(
                "Daily-steps analysis failed"
            )

            raise DatasetAnalysisError(
                "Daily-steps analysis failed: "
                f"{exc}"
            ) from exc

        _notify(
            progress_callback,
            "campaign_configuration",
            (
                "Analyzing campaign tasks "
                "and challenges"
            ),
        )

        try:
            (
                analyze_visualizations_challenges_tasks(
                    csv_data
                )
            )

        except Exception as exc:
            logger.exception(
                "Campaign configuration analysis "
                "failed"
            )

            raise DatasetAnalysisError(
                "Campaign configuration analysis "
                f"failed: {exc}"
            ) from exc

        _notify(
            progress_callback,
            "report",
            "Generating analysis report",
        )

        try:
            generated_report = (
                create_complete_report(
                    csv_data,
                    json_data,
                    activities_result,
                )
            )

            if generated_report:
                report_path = Path(
                    generated_report
                )

        except Exception as exc:
            logger.exception(
                "Report generation failed"
            )

            raise DatasetAnalysisError(
                "Report generation failed: "
                f"{exc}"
            ) from exc

    _notify(
        progress_callback,
        "complete",
        "Analysis complete",
    )

    return DatasetAnalysisResult(
        dataset_dir=(
            inspection.dataset_dir
        ),
        output_dir=(
            output_dir
        ),
        selected_participants=len(
            participant_ids
        ),
        campaign_tables_loaded=(
            campaign_tables_loaded
        ),
        raw_json_files_loaded=(
            raw_json_files_loaded
        ),
        unique_active_users=(
            unique_active_users
        ),
        report_path=(
            report_path
        ),
    )