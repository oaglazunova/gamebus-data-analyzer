from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_export,
)

from src.trajectory.engagement import (
    explicit_engagement_mask,
)

from src.trajectory.event_normalization import (
    build_normalized_events,
)

from src.trajectory.observation_window import (
    build_observation_window,
)


GAP_THRESHOLDS = (
    7,
    14,
    21,
)


GAP_COLUMNS = [
    "participant_id",
    "gap_id",

    "gap_type",
    "gap_status",

    "previous_engagement_date",
    "gap_start",
    "gap_end",
    "next_engagement_date",

    "inactive_days",

    "reengaged",
    "right_censored",

    "reached_7d",
    "reached_14d",
    "reached_21d",

    "exceeds_episode_boundary",

    "analysis_cutoff",
]


def _prepare_engagement_days(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one row per participant per calendar day
    on which explicit engagement occurred.

    Multiple task completions on the same day still
    represent one engagement day for gap calculation.
    """

    if events.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "engagement_date",
                "events_on_day",
                "points_on_day",
            ]
        )

    result = events.copy()

    result["participant_id"] = pd.to_numeric(
        result["participant_id"],
        errors="coerce",
    ).astype("Int64")

    result["occurred_at"] = pd.to_datetime(
        result["occurred_at"],
        utc=True,
        errors="coerce",
    )

    result["points"] = pd.to_numeric(
        result["points"],
        errors="coerce",
    ).fillna(0)

    result = result.dropna(
        subset=[
            "participant_id",
            "occurred_at",
        ]
    )

    explicit = explicit_engagement_mask(
        result
    )

    result = result.loc[
        explicit
    ].copy()

    if result.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "engagement_date",
                "events_on_day",
                "points_on_day",
            ]
        )

    result["engagement_date"] = (
        result["occurred_at"]
        .dt.floor("D")
    )

    return (
        result
        .groupby(
            [
                "participant_id",
                "engagement_date",
            ],
            as_index=False,
        )
        .agg(
            events_on_day=(
                "event_id",
                "count",
            ),
            points_on_day=(
                "points",
                "sum",
            ),
        )
        .sort_values(
            [
                "participant_id",
                "engagement_date",
            ]
        )
        .reset_index(drop=True)
    )


def _threshold_flags(
    inactive_days: int,
) -> Dict[str, bool]:
    """
    Mark the sensitivity thresholds reached by a gap.
    """

    return {
        f"reached_{threshold}d": (
            inactive_days >= threshold
        )
        for threshold in GAP_THRESHOLDS
    }


def build_inactivity_gaps(
    config: TrajectoryAuditConfig,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build retrospective inactivity-gap summaries.

    Two kinds of gaps are produced:

    1. between_engagements
       We observe engagement before and after the gap.

    2. right_censored
       We observe the start of the gap, but no later
       engagement before the current analysis cutoff.

    Never-engaged participants are intentionally NOT
    represented here. Their condition is not a
    post-engagement inactivity gap and will be handled
    separately in candidate-pattern analysis.
    """

    engagement_days = (
        _prepare_engagement_days(
            events
        )
    )

    if engagement_days.empty:
        return pd.DataFrame(
            columns=GAP_COLUMNS
        )

    window = build_observation_window(
        config,
        events,
    )

    cutoff = window[
        "analysis_cutoff"
    ]

    if cutoff is None:
        return pd.DataFrame(
            columns=GAP_COLUMNS
        )

    cutoff = pd.Timestamp(
        cutoff
    ).floor("D")

    rows: List[Dict[str, Any]] = []

    for (
        participant_id,
        participant_days,
    ) in engagement_days.groupby(
        "participant_id",
        sort=True,
    ):

        dates = (
            participant_days[
                "engagement_date"
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        gap_number = 0

        # ---------------------------------------------
        # Closed gaps between two engagement days
        # ---------------------------------------------

        for (
            previous_date,
            next_date,
        ) in zip(
            dates[:-1],
            dates[1:],
        ):

            previous_date = pd.Timestamp(
                previous_date
            )

            next_date = pd.Timestamp(
                next_date
            )

            inactive_days = int(
                (
                    next_date
                    - previous_date
                ).days
                - 1
            )

            # Consecutive engagement days have no
            # inactivity gap to report.
            if inactive_days <= 0:
                continue

            gap_number += 1

            gap_start = (
                previous_date
                + pd.Timedelta(days=1)
            )

            gap_end = (
                next_date
                - pd.Timedelta(days=1)
            )

            row = {
                "participant_id": (
                    participant_id
                ),

                "gap_id": (
                    f"{int(participant_id)}:"
                    f"G{gap_number:03d}"
                ),

                "gap_type": (
                    "between_engagements"
                ),

                "gap_status": (
                    "reengaged"
                ),

                "previous_engagement_date": (
                    previous_date
                ),

                "gap_start": gap_start,

                "gap_end": gap_end,

                "next_engagement_date": (
                    next_date
                ),

                "inactive_days": (
                    inactive_days
                ),

                "reengaged": True,

                "right_censored": False,

                "exceeds_episode_boundary": (
                    inactive_days
                    >= config.episode_gap_days
                ),

                "analysis_cutoff": cutoff,
            }

            row.update(
                _threshold_flags(
                    inactive_days
                )
            )

            rows.append(row)

        # ---------------------------------------------
        # Trailing gap up to current evidence cutoff
        # ---------------------------------------------

        last_engagement = pd.Timestamp(
            dates[-1]
        )

        trailing_inactive_days = int(
            (
                cutoff
                - last_engagement
            ).days
        )

        if trailing_inactive_days > 0:

            gap_number += 1

            gap_start = (
                last_engagement
                + pd.Timedelta(days=1)
            )

            row = {
                "participant_id": (
                    participant_id
                ),

                "gap_id": (
                    f"{int(participant_id)}:"
                    f"G{gap_number:03d}"
                ),

                "gap_type": (
                    "right_censored"
                ),

                "gap_status": (
                    "ongoing_at_cutoff"
                ),

                "previous_engagement_date": (
                    last_engagement
                ),

                "gap_start": gap_start,

                "gap_end": cutoff,

                "next_engagement_date": (
                    pd.NaT
                ),

                "inactive_days": (
                    trailing_inactive_days
                ),

                "reengaged": False,

                "right_censored": True,

                "exceeds_episode_boundary": (
                    trailing_inactive_days
                    >= config.episode_gap_days
                ),

                "analysis_cutoff": cutoff,
            }

            row.update(
                _threshold_flags(
                    trailing_inactive_days
                )
            )

            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=GAP_COLUMNS
        )

    result = pd.DataFrame(
        rows
    )

    # ---------------------------------------------
    # Human-readable dates
    # ---------------------------------------------

    date_columns = [
        "previous_engagement_date",
        "gap_start",
        "gap_end",
        "next_engagement_date",
        "analysis_cutoff",
    ]

    for column in date_columns:

        result[column] = (
            pd.to_datetime(
                result[column],
                utc=True,
                errors="coerce",
            )
            .dt.date
        )

    # ---------------------------------------------
    # Stable output order
    # ---------------------------------------------

    for column in GAP_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    return (
        result[
            GAP_COLUMNS
        ]
        .sort_values(
            [
                "participant_id",
                "gap_start",
            ]
        )
        .reset_index(drop=True)
    )


def run_inactivity_gap_analysis(
    config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
    """
    Build and save inactivity_gaps.csv.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    # Check source availability explicitly.
    _, stream_info = load_campaign_export(
        config.campaign_data_path
    )

    activity_status = (
        stream_info[
            "activities"
        ]["status"]
    )

    if activity_status == "unavailable":

        gaps = pd.DataFrame(
            columns=GAP_COLUMNS
        )

    else:

        events = build_normalized_events(
            config
        )

        gaps = build_inactivity_gaps(
            config,
            events,
        )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "inactivity_gaps.csv",
    )

    gaps.to_csv(
        output_path,
        index=False,
    )

    return gaps


if __name__ == "__main__":

    gaps = run_inactivity_gap_analysis()

    print(
        "Inactivity gap analysis "
        "written successfully."
    )

    print()

    print(
        "Total inactivity gaps:",
        len(gaps),
    )

    if not gaps.empty:

        closed = (
            gaps["gap_type"]
            == "between_engagements"
        )

        censored = (
            gaps["gap_type"]
            == "right_censored"
        )

        print(
            "Closed gaps with re-engagement:",
            int(closed.sum()),
        )

        print(
            "Ongoing gaps at cutoff:",
            int(censored.sum()),
        )

        print()

        print(
            "Threshold sensitivity:"
        )

        for threshold in GAP_THRESHOLDS:

            column = (
                f"reached_{threshold}d"
            )

            total = int(
                gaps[column].sum()
            )

            closed_total = int(
                gaps.loc[
                    closed,
                    column,
                ].sum()
            )

            print(
                f"  >= {threshold} inactive days: "
                f"{total} gaps "
                f"({closed_total} later re-engaged)"
            )

        print()

        print(
            "Inactive-day distribution:"
        )

        print(
            gaps["inactive_days"]
            .describe()
            .to_string()
        )
