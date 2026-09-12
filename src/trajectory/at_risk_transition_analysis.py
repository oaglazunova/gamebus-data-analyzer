from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_export,
)

from src.trajectory.event_normalization import (
    build_normalized_events,
)

from src.trajectory.inactivity_gap_analysis import (
    GAP_THRESHOLDS,
    build_inactivity_gaps,
)


TRANSITION_COLUMNS = [
    "participant_id",
    "signal_id",
    "gap_id",

    "threshold_days",
    "signal_name",
    "signal_date",

    "previous_engagement_date",

    "inactive_days_observed",

    "reached_14d",
    "reached_21d",

    "reengaged",
    "reengagement_date",

    "days_from_signal_to_reengagement",

    "outcome",
    "right_censored",

    "analysis_cutoff",
]


def _signal_name(
    threshold: int,
) -> str:
    """
    Produce a neutral operational signal name.

    We deliberately avoid names such as:
        dropout
        high_risk
        failed participant

    because these thresholds have not yet been
    validated as predictive risk indicators.
    """

    return (
        f"prolonged_inactivity_{threshold}d"
    )


def _transition_outcome(
    reengaged: bool,
    right_censored: bool,
) -> str:
    """
    Describe what we actually observed after
    the inactivity threshold was crossed.
    """

    if reengaged:
        return "reengaged"

    if right_censored:
        return "ongoing_at_cutoff"

    return "unknown"


def build_at_risk_transitions(
    config: TrajectoryAuditConfig,
    gaps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert inactivity gaps into threshold-crossing
    signal events.

    One long inactivity gap can generate several
    signal rows.

    Example:

        25 inactive days

    generates:

        7-day signal
        14-day signal
        21-day signal

    This is intentional.

    These rows represent different possible
    intervention decision points along the same
    trajectory.
    """

    if gaps.empty:

        return pd.DataFrame(
            columns=TRANSITION_COLUMNS
        )

    result_rows: List[
        Dict[str, Any]
    ] = []

    # Convert once so date arithmetic is safe.
    prepared = gaps.copy()

    date_columns = [
        "previous_engagement_date",
        "gap_start",
        "gap_end",
        "next_engagement_date",
        "analysis_cutoff",
    ]

    for column in date_columns:

        prepared[column] = pd.to_datetime(
            prepared[column],
            utc=True,
            errors="coerce",
        )

    for _, gap in prepared.iterrows():

        participant_id = gap[
            "participant_id"
        ]

        gap_id = gap[
            "gap_id"
        ]

        inactive_days = int(
            gap["inactive_days"]
        )

        gap_start = gap[
            "gap_start"
        ]

        if pd.isna(gap_start):
            continue

        reengaged = bool(
            gap["reengaged"]
        )

        right_censored = bool(
            gap["right_censored"]
        )

        next_engagement = gap[
            "next_engagement_date"
        ]

        analysis_cutoff = gap[
            "analysis_cutoff"
        ]

        # -----------------------------------------
        # Create one signal per threshold reached.
        # -----------------------------------------

        for threshold in GAP_THRESHOLDS:

            if inactive_days < threshold:
                continue

            # gap_start is inactivity day 1.
            #
            # Therefore:
            #
            # day 7 signal =
            #     gap_start + 6 days
            #
            signal_date = (
                gap_start
                + pd.Timedelta(
                    days=threshold - 1
                )
            )

            # -------------------------------------
            # Time from signal to re-engagement
            # -------------------------------------

            if (
                reengaged
                and pd.notna(
                    next_engagement
                )
            ):

                days_to_reengagement = int(
                    (
                        next_engagement
                        - signal_date
                    ).days
                )

                reengagement_date = (
                    next_engagement
                )

            else:

                days_to_reengagement = (
                    pd.NA
                )

                reengagement_date = (
                    pd.NaT
                )

            participant_text = str(
                int(participant_id)
            )

            signal_id = (
                f"{participant_text}:"
                f"{gap_id.split(':')[-1]}:"
                f"I{threshold:02d}"
            )

            row = {
                "participant_id": (
                    participant_id
                ),

                "signal_id": (
                    signal_id
                ),

                "gap_id": (
                    gap_id
                ),

                "threshold_days": (
                    threshold
                ),

                "signal_name": (
                    _signal_name(
                        threshold
                    )
                ),

                "signal_date": (
                    signal_date
                ),

                "previous_engagement_date": (
                    gap[
                        "previous_engagement_date"
                    ]
                ),

                "inactive_days_observed": (
                    inactive_days
                ),

                "reached_14d": (
                    inactive_days >= 14
                ),

                "reached_21d": (
                    inactive_days >= 21
                ),

                "reengaged": (
                    reengaged
                ),

                "reengagement_date": (
                    reengagement_date
                ),

                "days_from_signal_to_reengagement": (
                    days_to_reengagement
                ),

                "outcome": (
                    _transition_outcome(
                        reengaged,
                        right_censored,
                    )
                ),

                "right_censored": (
                    right_censored
                ),

                "analysis_cutoff": (
                    analysis_cutoff
                ),
            }

            result_rows.append(
                row
            )

    if not result_rows:

        return pd.DataFrame(
            columns=TRANSITION_COLUMNS
        )

    result = pd.DataFrame(
        result_rows
    )

    # ---------------------------------------------
    # Convert timestamps back to readable dates.
    # ---------------------------------------------

    date_columns = [
        "signal_date",
        "previous_engagement_date",
        "reengagement_date",
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
    # Stable ordering
    # ---------------------------------------------

    result = result.sort_values(
        [
            "participant_id",
            "signal_date",
            "threshold_days",
        ]
    ).reset_index(
        drop=True
    )

    for column in TRANSITION_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    return result[
        TRANSITION_COLUMNS
    ]


def run_at_risk_transition_analysis(
    config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
    """
    Build and save at_risk_transitions.csv.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    # ---------------------------------------------
    # If activity data itself is unavailable,
    # we cannot derive inactivity transitions.
    # ---------------------------------------------

    _, stream_info = load_campaign_export(
        config.campaign_data_path
    )

    activity_status = (
        stream_info[
            "activities"
        ]["status"]
    )

    if activity_status == "unavailable":

        transitions = pd.DataFrame(
            columns=TRANSITION_COLUMNS
        )

    else:

        events = build_normalized_events(
            config
        )

        gaps = build_inactivity_gaps(
            config,
            events,
        )

        transitions = (
            build_at_risk_transitions(
                config,
                gaps,
            )
        )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "at_risk_transitions.csv",
    )

    transitions.to_csv(
        output_path,
        index=False,
    )

    return transitions


if __name__ == "__main__":

    transitions = (
        run_at_risk_transition_analysis()
    )

    print(
        "At-risk transition analysis "
        "written successfully."
    )

    print()

    print(
        "Total threshold-crossing signals:",
        len(transitions),
    )

    if not transitions.empty:

        print()

        print(
            "Signals by threshold:"
        )

        for threshold in GAP_THRESHOLDS:

            selected = transitions[
                transitions[
                    "threshold_days"
                ]
                == threshold
            ]

            total = len(
                selected
            )

            reengaged = int(
                selected[
                    "reengaged"
                ].sum()
            )

            censored = int(
                selected[
                    "right_censored"
                ].sum()
            )

            print(
                f"  {threshold} days: "
                f"{total} signals, "
                f"{reengaged} observed re-engagements, "
                f"{censored} ongoing at cutoff"
            )

        print()

        print(
            "Observed time from signal "
            "to re-engagement:"
        )

        for threshold in GAP_THRESHOLDS:

            selected = transitions[
                (
                    transitions[
                        "threshold_days"
                    ]
                    == threshold
                )
                &
                transitions[
                    "reengaged"
                ]
            ]

            values = pd.to_numeric(
                selected[
                    "days_from_signal_to_reengagement"
                ],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            print()

            print(
                f"  {threshold}-day signal:"
            )

            print(
                values
                .describe()
                .to_string()
            )