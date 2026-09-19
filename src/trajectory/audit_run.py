from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Callable

import pandas as pd

from src.trajectory.audit_runner import (
    run_trajectory_audit,
)
from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_desc,
    load_campaign_export,
    write_json,
)


DEFAULT_TRAJECTORY_AUDITS_DIR = (
    Path(__file__).resolve().parents[2]
    / "trajectory_audits"
)

SUPPORTED_SOURCE_TYPES = {
    "uploaded_files",
    "gamebus",
}


class TrajectoryAuditRunError(
    RuntimeError
):
    pass


@dataclass(
    frozen=True
)
class TrajectoryAuditRunResult:
    run_dir: Path
    inputs_dir: Path
    results_dir: Path
    cohort_path: Path
    run_manifest_path: Path

    campaign_data_path: Path
    campaign_description_path: Path

    source_type: str

    results: dict[str, Any]


def _normalize_pid(
    value: Any,
) -> int | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        number = float(
            str(value).strip()
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not number.is_integer():
        return None

    return int(
        number
    )


def _normalize_pid_set(
    values: Iterable[Any],
    *,
    label: str,
) -> set[int]:
    result: set[int] = set()

    for value in values:
        pid = _normalize_pid(
            value
        )

        if pid is None:
            raise TrajectoryAuditRunError(
                f"{label} contains an invalid "
                f"GameBus player ID: {value!r}"
            )

        result.add(
            pid
        )

    return result


def read_campaign_export_participant_ids(
    campaign_data_path: str | Path,
) -> list[int]:
    """
    Read the candidate trajectory cohort from
    1-aggregated-data.csv.

    Used for historical/uploaded campaign exports.
    """
    path = Path(
        campaign_data_path
    )

    export_data, _ = load_campaign_export(
        str(
            path
        )
    )

    aggregation = export_data.get(
        "aggregation",
        pd.DataFrame(),
    )

    if (
        aggregation.empty
        or "pid" not in aggregation.columns
    ):
        raise TrajectoryAuditRunError(
            "Could not determine the campaign cohort "
            "from 1-aggregated-data.csv."
        )

    participant_ids: set[int] = set()

    for value in aggregation[
        "pid"
    ]:
        pid = _normalize_pid(
            value
        )

        if pid is not None:
            participant_ids.add(
                pid
            )

    if not participant_ids:
        raise TrajectoryAuditRunError(
            "The campaign export does not contain "
            "any valid participant IDs."
        )

    return sorted(
        participant_ids
    )


def read_campaign_identity(
    campaign_description_path: str | Path,
) -> tuple[
    str | None,
    str | None,
]:
    """
    Read campaign abbreviation and ID from the
    campaign-description workbook.
    """
    sheets, _ = load_campaign_desc(
        str(
            campaign_description_path
        )
    )

    campaigns = sheets.get(
        "campaigns",
        pd.DataFrame(),
    )

    if campaigns.empty:
        return (
            None,
            None,
        )

    row = campaigns.iloc[
        0
    ]

    abbreviation = None
    campaign_id = None

    if "abbreviation" in campaigns.columns:
        raw = row.get(
            "abbreviation"
        )

        if pd.notna(raw):
            text = str(
                raw
            ).strip()

            if text:
                abbreviation = text

    if "id" in campaigns.columns:
        raw = row.get(
            "id"
        )

        normalized_id = _normalize_pid(
            raw
        )

        if normalized_id is not None:
            campaign_id = str(
                normalized_id
            )

        elif pd.notna(raw):
            text = str(
                raw
            ).strip()

            if text:
                campaign_id = text

    return (
        abbreviation,
        campaign_id,
    )


def _safe_component(
    value: str | None,
    fallback: str,
) -> str:
    text = (
        value
        or fallback
    ).strip()

    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text,
    ).strip(
        "._-"
    )

    return (
        cleaned
        or fallback
    )


def _timestamp_text(
    created_at: datetime,
) -> str:
    return created_at.strftime(
        "%Y-%m-%d_%H%M%S"
    )


def _unique_run_directory(
    output_root: Path,
    base_name: str,
) -> Path:
    candidate = (
        output_root
        / base_name
    )

    if not candidate.exists():
        return candidate

    counter = 2

    while True:
        candidate = (
            output_root
            / f"{base_name}_{counter}"
        )

        if not candidate.exists():
            return candidate

        counter += 1


def run_trajectory_audit_snapshot(
    *,
    campaign_data_path: str | Path,
    campaign_description_path: str | Path,
    candidate_participant_ids: Iterable[Any],
    selected_participant_ids: Iterable[Any],
    source_type: str,
    analysis_cutoff: str | None = None,
    analysis_cutoff_source: str | None = None,
    output_root: str | Path = (
        DEFAULT_TRAJECTORY_AUDITS_DIR
    ),
    created_at: datetime | None = None,
    progress_callback: (
        Callable[
            [
                str,
                str,
            ],
            None,
        ]
        | None
    ) = None,
) -> TrajectoryAuditRunResult:
    """
    Create one immutable, timestamped trajectory-audit run.

    The source can be either manually uploaded campaign
    files or a current snapshot retrieved from GameBus.

    In both cases the exact campaign files are preserved
    inside the audit run.

    This workflow is deliberately independent of the HW8
    dataset/extraction workflow.
    """
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise TrajectoryAuditRunError(
            "Unsupported trajectory-audit source: "
            f"{source_type!r}"
        )

    campaign_data_source = Path(
        campaign_data_path
    ).expanduser().resolve()

    campaign_description_source = Path(
        campaign_description_path
    ).expanduser().resolve()

    if not campaign_data_source.is_file():
        raise TrajectoryAuditRunError(
            "Campaign analytics ZIP not found: "
            f"{campaign_data_source}"
        )

    if not campaign_description_source.is_file():
        raise TrajectoryAuditRunError(
            "Campaign description XLSX not found: "
            f"{campaign_description_source}"
        )

    candidate_ids = _normalize_pid_set(
        candidate_participant_ids,
        label="Candidate cohort",
    )

    selected_ids = _normalize_pid_set(
        selected_participant_ids,
        label="Selected cohort",
    )

    if not candidate_ids:
        raise TrajectoryAuditRunError(
            "The trajectory audit requires at least "
            "one candidate participant."
        )

    if not selected_ids:
        raise TrajectoryAuditRunError(
            "Select at least one participant before "
            "running the trajectory audit."
        )

    unknown_ids = (
        selected_ids
        - candidate_ids
    )

    if unknown_ids:
        raise TrajectoryAuditRunError(
            "The selected cohort contains participant "
            "IDs that are not present in the candidate "
            "cohort: "
            + ", ".join(
                str(pid)
                for pid in sorted(
                    unknown_ids
                )
            )
        )

    abbreviation, campaign_id = (
        read_campaign_identity(
            campaign_description_source
        )
    )

    if not campaign_id:
        raise TrajectoryAuditRunError(
            "The trajectory audit requires a valid "
            "GameBus campaign ID in the campaign "
            "description."
        )

    when = (
            created_at
            or datetime.now().astimezone()
    )

    output_root_path = Path(
        output_root
    ).expanduser().resolve()

    output_root_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    campaign_dir = (
            output_root_path
            / (
                "campaign_"
                f"{_safe_component(campaign_id, 'unknown')}"
            )
    )

    audits_dir = (
            campaign_dir
            / "audits"
    )

    audits_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_name = _timestamp_text(
        when
    )

    run_dir = _unique_run_directory(
        audits_dir,
        run_name,
    )

    inputs_dir = (
        run_dir
        / "inputs"
    )

    results_dir = (
        run_dir
        / "results"
    )

    inputs_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    campaign_data_copy = (
        inputs_dir
        / campaign_data_source.name
    )

    campaign_description_copy = (
        inputs_dir
        / campaign_description_source.name
    )

    shutil.copy2(
        campaign_data_source,
        campaign_data_copy,
    )

    shutil.copy2(
        campaign_description_source,
        campaign_description_copy,
    )

    cohort_path = (
        run_dir
        / "cohort.json"
    )

    cohort_payload = {
        "schema_version": 1,
        "created_at": when.isoformat(),
        "source": (
            "gamebus_studio_users"
            if source_type == "gamebus"
            else "campaign_export"
        ),
        "participants": [
            {
                "pid": str(
                    pid
                ),
                "included": (
                    pid in selected_ids
                ),
            }
            for pid in sorted(
                candidate_ids
            )
        ],
    }

    write_json(
        str(
            cohort_path
        ),
        cohort_payload,
    )

    run_manifest_path = (
        run_dir
        / "run_manifest.json"
    )

    run_manifest = {
        "schema_version": 1,
        "created_at": when.isoformat(),
        "source_type": source_type,
        "campaign": {
            "abbreviation": abbreviation,
            "id": campaign_id,
        },
        "observation": {
            "analysis_cutoff": analysis_cutoff,
            "cutoff_source": (
                analysis_cutoff_source
            ),
        },
        "inputs": {
            "campaign_data": str(
                Path("inputs")
                / campaign_data_copy.name
            ),
            "campaign_description": str(
                Path("inputs")
                / campaign_description_copy.name
            ),
        },
        "cohort": "cohort.json",
        "results": "results",
    }

    write_json(
        str(
            run_manifest_path
        ),
        run_manifest,
    )

    config = TrajectoryAuditConfig(
        campaign_data_path=str(
            campaign_data_copy
        ),
        campaign_desc_path=str(
            campaign_description_copy
        ),
        raw_data_dir=str(
            inputs_dir
            / "participant_data"
        ),
        output_dir=str(
            results_dir
        ),
        analysis_cutoff=analysis_cutoff,
        analysis_cutoff_source=(
            analysis_cutoff_source
        ),
        analysis_participant_ids=(
            set(
                selected_ids
            )
        ),
    )

    if progress_callback is not None:
        progress_callback(
            "preparing",
            "Creating timestamped audit snapshot",
        )

    try:
        results = run_trajectory_audit(
            config,
            progress_callback=(
                progress_callback
            ),
        )

    except Exception as exc:
        raise TrajectoryAuditRunError(
            "Trajectory audit failed: "
            f"{exc}"
        ) from exc

    return TrajectoryAuditRunResult(
        run_dir=run_dir,
        inputs_dir=inputs_dir,
        results_dir=results_dir,
        cohort_path=cohort_path,
        run_manifest_path=(
            run_manifest_path
        ),
        campaign_data_path=(
            campaign_data_copy
        ),
        campaign_description_path=(
            campaign_description_copy
        ),
        source_type=source_type,
        results=results,
    )

