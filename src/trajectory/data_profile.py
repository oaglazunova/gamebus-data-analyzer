from __future__ import annotations

import glob
import os
from typing import Any, Dict

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    jsonable,
    load_campaign_desc,
    load_campaign_export,
    parse_datetime_series,
    write_json,
)


def _campaign_metadata(
    desc: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Extract basic campaign and wave metadata.
    """

    result: Dict[str, Any] = {}

    campaigns = desc.get(
        "campaigns",
        pd.DataFrame(),
    )

    if not campaigns.empty:

        row = campaigns.iloc[0]

        for column in (
            "id",
            "abbreviation",
            "name",
        ):
            if column in campaigns.columns:
                result[column] = jsonable(
                    row.get(column)
                )

    waves = desc.get(
        "waves",
        pd.DataFrame(),
    ).copy()

    if not waves.empty:

        if "start" in waves.columns:

            starts = pd.to_datetime(
                waves["start"],
                utc=True,
                errors="coerce",
            ).dropna()

            result["first_wave_start"] = (
                starts.min()
                if not starts.empty
                else None
            )

        if "end" in waves.columns:

            ends = pd.to_datetime(
                waves["end"],
                utc=True,
                errors="coerce",
            ).dropna()

            result["last_wave_end"] = (
                ends.max()
                if not ends.empty
                else None
            )

        result["number_of_waves"] = int(
            len(waves)
        )

    return result


def _cohort_profile(
    export_data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Use 1-aggregated-data.csv as the primary export cohort.

    This is important because it can contain participants
    who never generated an activity.
    """

    aggregation = export_data.get(
        "aggregation",
        pd.DataFrame(),
    )

    if (
        aggregation.empty
        or "pid" not in aggregation.columns
    ):
        return {
            "source": (
                "campaign_export/"
                "1-aggregated-data.csv"
            ),
            "status": "unavailable",
            "participants": None,
        }

    participant_ids = (
        aggregation["pid"]
        .dropna()
        .drop_duplicates()
    )

    result: Dict[str, Any] = {
        "source": (
            "campaign_export/"
            "1-aggregated-data.csv"
        ),
        "status": "available",
        "participants": int(
            len(participant_ids)
        ),
    }

    if "numberOfActivities" in aggregation.columns:

        activity_counts = pd.to_numeric(
            aggregation["numberOfActivities"],
            errors="coerce",
        ).fillna(0)

        by_pid = (
            pd.DataFrame(
                {
                    "pid": aggregation["pid"],
                    "n": activity_counts,
                }
            )
            .dropna(subset=["pid"])
            .groupby("pid")["n"]
            .max()
        )

        result[
            "participants_with_zero_aggregated_activities"
        ] = int(
            (by_pid == 0).sum()
        )

    return result


def _stream_date_range(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Determine the actual observed date range
    of one data stream.
    """

    timestamps = (
        parse_datetime_series(df)
        .dropna()
    )

    if timestamps.empty:

        return {
            "first_observed_at": None,
            "last_observed_at": None,
        }

    return {
        "first_observed_at": timestamps.min(),
        "last_observed_at": timestamps.max(),
    }


def _activity_profile(
    export_data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Summarize activity providers and activity types.
    """

    activities = export_data.get(
        "activities",
        pd.DataFrame(),
    )

    if activities.empty:

        return {
            "rows": int(len(activities)),
            "participants": 0,
            "providers": [],
            "activity_types": [],
        }

    providers = []

    if "provider" in activities.columns:

        providers = sorted(
            activities["provider"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

    activity_types = []

    if "type" in activities.columns:

        activity_types = sorted(
            activities["type"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

    participants = 0

    if "pid" in activities.columns:

        participants = int(
            activities["pid"]
            .dropna()
            .nunique()
        )

    return {
        "rows": int(len(activities)),
        "participants": participants,
        "providers": providers,
        "activity_types": activity_types,
    }


def _credential_extraction_profile(
    raw_data_dir: str,
) -> Dict[str, Any]:
    """
    Inspect already-extracted participant-specific JSON.

    IMPORTANT:
    this does NOT use users.xlsx and does NOT authenticate.

    It only answers:
    "Do participant-specific extracted files already exist?"
    """

    json_files = glob.glob(
        os.path.join(
            raw_data_dir,
            "*.json",
        )
    )

    if not json_files:

        return {
            "status": "unavailable",
            "files": 0,
            "reason": (
                "no participant-specific "
                "JSON files found"
            ),
        }

    return {
        "status": "available",
        "files": int(len(json_files)),
        "filenames": sorted(
            os.path.basename(path)
            for path in json_files
        ),
    }


def build_data_profile(
    config: TrajectoryAuditConfig,
) -> Dict[str, Any]:
    """
    Build the complete input-data profile.
    """

    export_data, stream_status = (
        load_campaign_export(
            config.campaign_data_path
        )
    )

    desc, desc_status = load_campaign_desc(
        config.campaign_desc_path
    )

    # Add observed dates and participant counts
    # to every export stream.
    for (
        stream_name,
        stream_info,
    ) in stream_status.items():

        df = export_data.get(
            stream_name,
            pd.DataFrame(),
        )

        stream_info.update(
            _stream_date_range(df)
        )

        if (
            not df.empty
            and "pid" in df.columns
        ):
            stream_info["participants"] = int(
                df["pid"]
                .dropna()
                .nunique()
            )

        else:
            stream_info["participants"] = 0

    profile: Dict[str, Any] = {
        "campaign": _campaign_metadata(desc),

        "cohort": _cohort_profile(
            export_data
        ),

        "campaign_export": {
            "path": config.campaign_data_path,
            "streams": stream_status,
        },

        "campaign_description": (
            desc_status
        ),

        "credential_extraction": (
            _credential_extraction_profile(
                config.raw_data_dir
            )
        ),

        "activities": _activity_profile(
            export_data
        ),
    }

    return profile


def run_data_profile(
    config: TrajectoryAuditConfig | None = None,
) -> Dict[str, Any]:
    """
    Run the profiling step and save data_profile.json.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    profile = build_data_profile(
        config
    )

    output_path = os.path.join(
        config.output_dir,
        "data_profile.json",
    )

    write_json(
        output_path,
        profile,
    )

    return profile


if __name__ == "__main__":

    profile = run_data_profile()

    print(
        "Trajectory data profile "
        "written successfully."
    )

    print(
        "Campaign:",
        profile["campaign"].get("id"),
    )

    print(
        "Participants in export cohort:",
        profile["cohort"].get(
            "participants"
        ),
    )

    print("Streams:")

    for (
        name,
        info,
    ) in (
        profile["campaign_export"]
        ["streams"]
        .items()
    ):

        print(
            f"  {name}: "
            f"{info['status']} "
            f"({info['rows']} rows)"
        )