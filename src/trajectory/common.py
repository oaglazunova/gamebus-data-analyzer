from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Tuple

import pandas as pd

from config.paths import PROJECT_ROOT, RAW_DATA_DIR


CORE_EXPORT_MEMBERS = {
    "aggregation": "1-aggregated-data.csv",
    "activities": "2-activities.csv",
    "navigation": "3-navigation-events.csv",
    "notification_events": "4-notification-events.csv",
    "sensor_events": "5-sensor-events.csv",
}


@dataclass
class TrajectoryAuditConfig:
    campaign_data_path: str = os.path.join(
        PROJECT_ROOT,
        "config",
        "campaign_data.zip",
    )

    campaign_desc_path: str = os.path.join(
        PROJECT_ROOT,
        "config",
        "campaign_desc.xlsx",
    )

    raw_data_dir: str = RAW_DATA_DIR

    output_dir: str = os.path.join(
        PROJECT_ROOT,
        "data_analysis",
        "trajectory_audit",
    )

    episode_gap_days: int = 14

    observation_start_grace_days: int = 14

    # Optional manual override.
    # Example:
    # analysis_cutoff = "2026-09-11"
    #
    # None means:
    # use the latest trustworthy observed event date.
    analysis_cutoff: str | None = None

    # Candidate-pattern windows.
    #
    # "Recent" means the final 28 observed days.
    # "Reference" means the 28 days immediately
    # preceding the recent window.
    pattern_recent_window_days: int = 28
    pattern_reference_window_days: int = 28

    # Avoid declaring a decline/disappearance based on
    # one isolated historical event.
    pattern_min_reference_events: int = 2

    # A recent explicit-engagement count <= 50% of the
    # preceding reference period is a candidate decline.
    # This is exploratory, not a validated risk threshold.
    pattern_decline_ratio: float = 0.5

    # Repeated long-gap candidate.
    pattern_repeated_gap_count: int = 2

    # Maximum number of compact participant cases
    # exported by the Trajectory Audit.
    case_export_max_cases: int = 10

    analysis_participant_ids: set[int] | None = None


def jsonable(value: Any) -> Any:
    """
    Convert pandas/numpy/date values into values that json.dump can serialize.
    """

    if isinstance(
            value,
            (
                    pd.Timestamp,
                    datetime,
                    date,
            ),
    ):
        if pd.isna(value):
            return None
        return value.isoformat()

    if value is pd.NA or value is pd.NaT:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # numpy scalar values such as np.int64
    item = getattr(value, "item", None)

    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Write a dictionary as readable JSON.
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=jsonable,
        )


def parse_datetime_series(df: pd.DataFrame) -> pd.Series:
    """
    Find the most appropriate timestamp column for an export stream.

    Most GameBus export streams use createdAt.
    Some streams additionally contain timestamp.
    """

    if df is None or df.empty:
        return pd.Series(
            dtype="datetime64[ns, UTC]"
        )

    if "createdAt" in df.columns:
        result = pd.to_datetime(
            df["createdAt"],
            utc=True,
            errors="coerce",
        )

        if result.notna().any():
            return result

    if "timestamp" in df.columns:
        return pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

    return pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns, UTC]",
    )


def load_campaign_export(
    path: str,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict[str, Any]],
]:
    """
    Load the five standard files from campaign_data.zip.

    This loader is deliberately different from the existing analyzer loader.

    The existing loader skips empty CSV files.

    For trajectory analysis we must preserve that information because:

        file missing       !=
        file exists, 0 rows

    Therefore every stream receives one of:

        available
        available_empty
        unavailable
    """

    data: Dict[str, pd.DataFrame] = {}
    streams: Dict[str, Dict[str, Any]] = {}

    if not path or not os.path.exists(path):

        for key, member in CORE_EXPORT_MEMBERS.items():
            streams[key] = {
                "status": "unavailable",
                "member": member,
                "rows": 0,
                "reason": "campaign export ZIP not found",
            }

        return data, streams

    try:
        with zipfile.ZipFile(path, "r") as zf:

            members = {
                os.path.basename(name).lower(): name
                for name in zf.namelist()
                if not name.endswith("/")
            }

            for key, expected_name in CORE_EXPORT_MEMBERS.items():

                actual_name = members.get(
                    expected_name.lower()
                )

                if actual_name is None:

                    streams[key] = {
                        "status": "unavailable",
                        "member": expected_name,
                        "rows": 0,
                        "reason": (
                            "member missing from "
                            "campaign export ZIP"
                        ),
                    }

                    continue

                try:
                    with zf.open(actual_name) as f:
                        df = pd.read_csv(f)

                    # IMPORTANT:
                    # keep the DataFrame even when it is empty.
                    data[key] = df

                    streams[key] = {
                        "status": (
                            "available"
                            if not df.empty
                            else "available_empty"
                        ),
                        "member": actual_name,
                        "rows": int(len(df)),
                        "columns": list(df.columns),
                    }

                except Exception as exc:

                    streams[key] = {
                        "status": "unavailable",
                        "member": actual_name,
                        "rows": 0,
                        "reason": (
                            f"could not parse CSV: {exc}"
                        ),
                    }

    except zipfile.BadZipFile as exc:

        for key, member in CORE_EXPORT_MEMBERS.items():

            streams[key] = {
                "status": "unavailable",
                "member": member,
                "rows": 0,
                "reason": (
                    f"invalid campaign export ZIP: {exc}"
                ),
            }

    return data, streams


def load_campaign_desc(
    path: str,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Any],
]:
    """
    Load every sheet from campaign_desc.xlsx.

    Empty sheets are retained because their emptiness is itself information.
    """

    sheets: Dict[str, pd.DataFrame] = {}

    if not path or not os.path.exists(path):

        return sheets, {
            "status": "unavailable",
            "path": path,
            "reason": (
                "campaign description workbook not found"
            ),
        }

    try:
        with pd.ExcelFile(path) as xl:

            for sheet_name in xl.sheet_names:

                try:
                    sheets[sheet_name] = xl.parse(
                        sheet_name
                    )

                except Exception as exc:

                    sheets[sheet_name] = pd.DataFrame()

                    print(
                        "Warning: could not parse "
                        f"sheet '{sheet_name}': {exc}"
                    )

        return sheets, {
            "status": "available",
            "path": path,
            "sheets": {
                name: int(len(df))
                for name, df in sheets.items()
            },
        }

    except Exception as exc:

        return {}, {
            "status": "unavailable",
            "path": path,
            "reason": (
                f"could not open workbook: {exc}"
            ),
        }