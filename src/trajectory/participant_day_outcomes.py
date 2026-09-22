from __future__ import annotations

import os

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)




OUTCOME_HORIZONS_DAYS = (
    7,
    14,
    21,
    30,
)


OUTCOME_COLUMNS = [
    "participant_id",
    "date",

    "observation_end_date",
    "days_observed_after_landmark",

    "next_explicit_engagement_date",
    "next_explicit_engagement_observed",

    "event_or_censor_date",
    "time_to_next_explicit_engagement_days",
    "time_to_event_or_censor_days",

    "right_censored",

    "next_day_outcome_available",
    "explicit_engagement_next_day",

    "outcome_available_7d",
    "explicit_engagement_within_7d",
    "continued_inactivity_7d",

    "outcome_available_14d",
    "explicit_engagement_within_14d",
    "continued_inactivity_14d",

    "outcome_available_21d",
    "explicit_engagement_within_21d",
    "continued_inactivity_21d",

    "outcome_available_30d",
    "explicit_engagement_within_30d",
    "continued_inactivity_30d",
]


def _add_horizon_outcomes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add fixed-horizon outcomes with explicit handling
    of incomplete follow-up.

    For horizon h:

        explicit_engagement_within_hd = True
            when the next explicit engagement occurs
            within h days after the landmark day.

        explicit_engagement_within_hd = False
            only when the full h-day horizon is
            observable and no engagement occurs.

        continued_inactivity_hd = logical inverse
            of engagement-within-h when the outcome
            is observable.

        outcome_available_hd = False
            when follow-up ends before h days and no
            engagement has already been observed.

    Thus incomplete follow-up is never silently
    labelled as continued inactivity.
    """

    result = frame.copy()

    time_to_event = pd.to_numeric(
        result[
            "time_to_next_explicit_engagement_days"
        ],
        errors="coerce",
    )

    followup_days = pd.to_numeric(
        result[
            "days_observed_after_landmark"
        ],
        errors="coerce",
    )

    event_observed = (
        result[
            "next_explicit_engagement_observed"
        ]
        .astype("boolean")
        .fillna(False)
    )

    for horizon in OUTCOME_HORIZONS_DAYS:

        available_column = (
            f"outcome_available_{horizon}d"
        )

        engagement_column = (
            f"explicit_engagement_within_{horizon}d"
        )

        inactivity_column = (
            f"continued_inactivity_{horizon}d"
        )

        event_within_horizon = (
            event_observed
            & time_to_event.le(
                horizon
            )
        )

        full_horizon_observed = (
            followup_days.ge(
                horizon
            )
        )

        # An event inside the horizon makes the
        # outcome known even if the nominal horizon
        # has not otherwise been fully observed.
        outcome_available = (
            event_within_horizon
            | full_horizon_observed
        )

        result[
            available_column
        ] = (
            outcome_available
            .astype("boolean")
        )

        result[
            engagement_column
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

        result.loc[
            outcome_available,
            engagement_column,
        ] = (
            event_within_horizon.loc[
                outcome_available
            ]
            .astype("boolean")
        )

        result[
            inactivity_column
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

        result.loc[
            outcome_available,
            inactivity_column,
        ] = (
            ~result.loc[
                outcome_available,
                engagement_column,
            ]
        )

    return result



def build_participant_day_outcomes(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct maintenance/re-engagement outcomes for
    the participant-day feature table.

    The feature table itself remains causal.

    This table deliberately uses future information
    and must therefore only be used as an outcome
    table for retrospective model development and
    evaluation.

    For each participant-day t:

        next_explicit_engagement_date
            first explicit-engagement day strictly
            after t

        time_to_next_explicit_engagement_days
            observed waiting time when a later
            engagement exists

        right_censored
            no later explicit engagement was observed
            before the participant's observation end

        explicit_engagement_next_day
            discrete-time hazard outcome for t -> t+1

    Rows before a participant's first observed
    explicit engagement are excluded. They belong to
    the initiation problem, not the maintenance /
    re-engagement problem.
    """

    if features.empty:
        return pd.DataFrame(
            columns=OUTCOME_COLUMNS
        )

    required = {
        "participant_id",
        "date",
        "maintenance_reengagement_eligible",
        "explicit_events_today",
    }

    missing = (
        required
        - set(
            features.columns
        )
    )

    if missing:
        raise ValueError(
            "Participant-day features are missing "
            "required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result = features.copy()

    result[
        "participant_id"
    ] = pd.to_numeric(
        result[
            "participant_id"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    result["date"] = pd.to_datetime(
        result["date"],
        utc=True,
        errors="coerce",
    )

    result[
        "explicit_events_today"
    ] = pd.to_numeric(
        result[
            "explicit_events_today"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    result = (
        result
        .dropna(
            subset=[
                "participant_id",
                "date",
            ]
        )
        .sort_values(
            [
                "participant_id",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------
    # Observation end
    # ---------------------------------------------

    result[
        "observation_end_date"
    ] = (
        result
        .groupby(
            "participant_id",
            sort=False,
        )[
            "date"
        ]
        .transform(
            "max"
        )
    )

    result[
        "days_observed_after_landmark"
    ] = (
        result[
            "observation_end_date"
        ]
        - result[
            "date"
        ]
    ).dt.days.astype(
        "Int64"
    )

    # ---------------------------------------------
    # Next explicit-engagement date
    #
    # Important:
    # this must be STRICTLY AFTER the current row.
    # An engagement occurring on t is not the
    # outcome for the landmark at t.
    # ---------------------------------------------

    next_explicit = pd.Series(
        pd.NaT,
        index=result.index,
        dtype="datetime64[ns, UTC]",
    )

    for (
        _participant_id,
        group,
    ) in result.groupby(
        "participant_id",
        sort=False,
    ):

        event_dates = (
            group["date"]
            .where(
                group[
                    "explicit_events_today"
                ].gt(0)
            )
        )

        next_dates = (
            event_dates
            .shift(-1)
            .bfill()
        )

        next_explicit.loc[
            group.index
        ] = next_dates

    result[
        "next_explicit_engagement_date"
    ] = next_explicit

    result[
        "next_explicit_engagement_observed"
    ] = (
        result[
            "next_explicit_engagement_date"
        ].notna()
    )

    result[
        "right_censored"
    ] = ~result[
        "next_explicit_engagement_observed"
    ]

    # ---------------------------------------------
    # Time-to-event / censoring
    # ---------------------------------------------

    result[
        "time_to_next_explicit_engagement_days"
    ] = (
        result[
            "next_explicit_engagement_date"
        ]
        - result[
            "date"
        ]
    ).dt.days.astype(
        "Int64"
    )

    result[
        "event_or_censor_date"
    ] = result[
        "next_explicit_engagement_date"
    ].where(
        result[
            "next_explicit_engagement_observed"
        ],
        result[
            "observation_end_date"
        ],
    )

    result[
        "time_to_event_or_censor_days"
    ] = (
        result[
            "event_or_censor_date"
        ]
        - result[
            "date"
        ]
    ).dt.days.astype(
        "Int64"
    )

    # ---------------------------------------------
    # Discrete-time next-day hazard outcome
    #
    # On the final observed day we cannot know what
    # happens on t+1, so the target is missing rather
    # than False.
    # ---------------------------------------------

    result[
        "next_day_outcome_available"
    ] = (
        result[
            "days_observed_after_landmark"
        ].gt(0)
    )

    result[
        "explicit_engagement_next_day"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    next_day_event = (
        result[
            "next_explicit_engagement_date"
        ]
        == (
            result["date"]
            + pd.Timedelta(
                days=1
            )
        )
    )

    available = result[
        "next_day_outcome_available"
    ]

    result.loc[
        available,
        "explicit_engagement_next_day",
    ] = (
        next_day_event.loc[
            available
        ]
        .astype(
            "boolean"
        )
    )

    # ---------------------------------------------
    # Fixed-horizon outcomes
    # ---------------------------------------------

    result = _add_horizon_outcomes(
        result
    )


    # ---------------------------------------------
    # Maintenance/re-engagement risk set only
    # ---------------------------------------------

    maintenance_eligible = (
        result[
            "maintenance_reengagement_eligible"
        ]
        .astype(
            "boolean"
        )
        .fillna(
            False
        )
    )

    result = result.loc[
        maintenance_eligible
    ].copy()

    # ---------------------------------------------
    # Stable output dates
    # ---------------------------------------------

    for column in (
        "date",
        "observation_end_date",
        "next_explicit_engagement_date",
        "event_or_censor_date",
    ):

        result[column] = (
            pd.to_datetime(
                result[column],
                utc=True,
                errors="coerce",
            )
            .dt.date
        )

    for column in OUTCOME_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    return (
        result[
            OUTCOME_COLUMNS
        ]
        .reset_index(
            drop=True
        )
    )


def run_participant_day_outcomes(
    config: TrajectoryAuditConfig,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build and save participant_day_outcomes.csv.
    """

    outcomes = (
        build_participant_day_outcomes(
            features
        )
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "participant_day_outcomes.csv",
    )

    outcomes.to_csv(
        output_path,
        index=False,
    )

    return outcomes