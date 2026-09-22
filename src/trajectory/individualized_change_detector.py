from __future__ import annotations

from dataclasses import dataclass
import os

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)


@dataclass(
    frozen=True
)
class IndividualizedChangeConfig:
    """
    Configuration for the transparent participant-
    specific engagement change detector.

    The default detector compares:

        recent 7 days

    with:

        preceding 28 days

    and requires the decline criterion to persist
    before a human-review trigger is generated.
    """

    recent_window_days: int = 7
    reference_window_days: int = 28

    minimum_reference_active_days: int = 3

    decline_ratio_threshold: float = 0.50

    persistence_days: int = 3


CHANGE_COLUMNS = [
    "participant_id",
    "date",

    "maintenance_day_index",

    "recent_active_days_7d",
    "reference_active_days_28d",

    "recent_active_day_rate_7d",
    "reference_active_day_rate_28d",

    "recent_to_reference_rate_ratio",
    "decline_score",

    "individual_change_assessable",
    "individual_change_signal",

    "signal_streak_days",
    "change_episode_number",

    "current_change_onset_date",

    "individual_change_confirmed",
    "individual_change_trigger",

    "current_change_trigger_date",
]


def _rolling_sum(
    frame: pd.DataFrame,
    column: str,
    window: int,
) -> pd.Series:

    return (
        frame
        .groupby(
            "participant_id",
            sort=False,
        )[column]
        .transform(
            lambda values: (
                values
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )
    )


def _preceding_window_sum(
    frame: pd.DataFrame,
    column: str,
    *,
    recent_window: int,
    reference_window: int,
) -> pd.Series:
    """
    Sum the reference window immediately preceding
    the recent window.

    At landmark day t:

        recent:
            t-(recent_window-1) ... t

        reference:
            t-(recent_window+reference_window-1)
            ...
            t-recent_window

    The windows therefore do not overlap.
    """

    return (
        frame
        .groupby(
            "participant_id",
            sort=False,
        )[column]
        .transform(
            lambda values: (
                values
                .shift(
                    recent_window
                )
                .rolling(
                    window=reference_window,
                    min_periods=1,
                )
                .sum()
            )
        )
    )


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = pd.Series(
        pd.NA,
        index=numerator.index,
        dtype="Float64",
    )

    valid = (
        denominator.notna()
        & denominator.gt(0)
    )

    result.loc[
        valid
    ] = (
        numerator.loc[
            valid
        ]
        / denominator.loc[
            valid
        ]
    )

    return result


def build_individualized_change_detection(
    features: pd.DataFrame,
    detector_config: (
        IndividualizedChangeConfig | None
    ) = None,
) -> pd.DataFrame:
    """
    Detect participant-specific engagement decline.

    Only maintenance/re-engagement rows are used.

    The detector is causal:
    every assessment at day t uses information
    available on or before t.

    A personalized signal is generated when:

        recent active-day rate
        ----------------------  <= threshold
        reference active-day rate

    provided that enough reference engagement exists.

    A trigger is generated only after the signal has
    persisted for the configured number of
    consecutive assessment days.
    """

    detector_config = (
        detector_config
        or IndividualizedChangeConfig()
    )

    if features.empty:
        return pd.DataFrame(
            columns=CHANGE_COLUMNS
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

    result = features[
        [
            "participant_id",
            "date",
            "maintenance_reengagement_eligible",
            "explicit_events_today",
        ]
    ].copy()

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

    if result.empty:
        return pd.DataFrame(
            columns=CHANGE_COLUMNS
        )

    result[
        "maintenance_day_index"
    ] = (
        result
        .groupby(
            "participant_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    result[
        "_active_today"
    ] = (
        result[
            "explicit_events_today"
        ]
        .gt(0)
        .astype(int)
    )

    result[
        "_observed_day"
    ] = 1

    recent_window = (
        detector_config.recent_window_days
    )

    reference_window = (
        detector_config.reference_window_days
    )

    # ---------------------------------------------
    # Recent 7-day history
    # ---------------------------------------------

    result[
        "recent_active_days_7d"
    ] = _rolling_sum(
        result,
        "_active_today",
        recent_window,
    )

    recent_days_observed = _rolling_sum(
        result,
        "_observed_day",
        recent_window,
    )

    # ---------------------------------------------
    # Preceding 28-day personal reference period
    # ---------------------------------------------

    result[
        "reference_active_days_28d"
    ] = _preceding_window_sum(
        result,
        "_active_today",
        recent_window=recent_window,
        reference_window=reference_window,
    )

    reference_days_observed = (
        _preceding_window_sum(
            result,
            "_observed_day",
            recent_window=recent_window,
            reference_window=reference_window,
        )
    )

    result[
        "recent_active_day_rate_7d"
    ] = _safe_ratio(
        result[
            "recent_active_days_7d"
        ],
        recent_days_observed,
    )

    result[
        "reference_active_day_rate_28d"
    ] = _safe_ratio(
        result[
            "reference_active_days_28d"
        ],
        reference_days_observed,
    )

    result[
        "recent_to_reference_rate_ratio"
    ] = _safe_ratio(
        result[
            "recent_active_day_rate_7d"
        ],
        result[
            "reference_active_day_rate_28d"
        ],
    )

    result[
        "decline_score"
    ] = (
        1.0
        - result[
            "recent_to_reference_rate_ratio"
        ]
    )

    # ---------------------------------------------
    # Assessability
    #
    # Require complete recent and reference windows
    # plus evidence that the participant actually
    # had a meaningful individual baseline.
    # ---------------------------------------------

    result[
        "individual_change_assessable"
    ] = (
        recent_days_observed.eq(
            recent_window
        )
        & reference_days_observed.eq(
            reference_window
        )
        & result[
            "reference_active_days_28d"
        ].ge(
            detector_config
            .minimum_reference_active_days
        )
    ).astype(
        "boolean"
    )

    result[
        "individual_change_signal"
    ] = (
        result[
            "individual_change_assessable"
        ]
        & result[
            "recent_to_reference_rate_ratio"
        ].le(
            detector_config
            .decline_ratio_threshold
        )
    ).astype(
        "boolean"
    )

    # ---------------------------------------------
    # Temporal persistence
    #
    # t_change:
    #     first day of the current personalized
    #     decline signal.
    #
    # t_trigger:
    #     first day on which persistence reaches the
    #     configured evidence threshold.
    # ---------------------------------------------

    result[
        "signal_streak_days"
    ] = 0

    result[
        "change_episode_number"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result[
        "current_change_onset_date"
    ] = pd.NaT

    result[
        "individual_change_confirmed"
    ] = False

    result[
        "individual_change_trigger"
    ] = False

    result[
        "current_change_trigger_date"
    ] = pd.NaT

    for (
        _participant_id,
        group,
    ) in result.groupby(
        "participant_id",
        sort=False,
    ):

        streak = 0
        episode_number = 0

        onset_date = None
        trigger_date = None

        for index in group.index:

            signal = bool(
                result.at[
                    index,
                    "individual_change_signal",
                ]
            )

            if signal:

                if streak == 0:
                    episode_number += 1

                    onset_date = result.at[
                        index,
                        "date",
                    ]

                    trigger_date = None

                streak += 1

                result.at[
                    index,
                    "signal_streak_days",
                ] = streak

                result.at[
                    index,
                    "change_episode_number",
                ] = episode_number

                result.at[
                    index,
                    "current_change_onset_date",
                ] = onset_date

                confirmed = (
                    streak
                    >= detector_config.persistence_days
                )

                result.at[
                    index,
                    "individual_change_confirmed",
                ] = confirmed

                if (
                    streak
                    == detector_config.persistence_days
                ):
                    trigger_date = result.at[
                        index,
                        "date",
                    ]

                    result.at[
                        index,
                        "individual_change_trigger",
                    ] = True

                if trigger_date is not None:

                    result.at[
                        index,
                        "current_change_trigger_date",
                    ] = trigger_date

            else:

                streak = 0
                onset_date = None
                trigger_date = None

    result[
        "signal_streak_days"
    ] = pd.to_numeric(
        result[
            "signal_streak_days"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    result[
        "individual_change_confirmed"
    ] = result[
        "individual_change_confirmed"
    ].astype(
        "boolean"
    )

    result[
        "individual_change_trigger"
    ] = result[
        "individual_change_trigger"
    ].astype(
        "boolean"
    )

    result = result.drop(
        columns=[
            "_active_today",
            "_observed_day",
        ]
    )

    for column in (
        "date",
        "current_change_onset_date",
        "current_change_trigger_date",
    ):

        result[column] = (
            pd.to_datetime(
                result[column],
                utc=True,
                errors="coerce",
            )
            .dt.date
        )

    for column in CHANGE_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    return (
        result[
            CHANGE_COLUMNS
        ]
        .reset_index(
            drop=True
        )
    )


def run_individualized_change_detection(
    config: TrajectoryAuditConfig,
    features: pd.DataFrame,
    detector_config: (
        IndividualizedChangeConfig | None
    ) = None,
) -> pd.DataFrame:

    result = (
        build_individualized_change_detection(
            features,
            detector_config,
        )
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    result.to_csv(
        os.path.join(
            config.output_dir,
            "individualized_change_detection.csv",
        ),
        index=False,
    )

    return result