from __future__ import annotations

import os

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)


RULE_THRESHOLDS_DAYS = (
    7,
    14,
    21,
)

OUTCOME_HORIZONS_DAYS = (
    7,
    14,
    21,
    30,
)


def build_participant_day_modeling_table(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join causal landmark features with retrospective
    outcomes.

    The resulting table is intended for model
    development/evaluation.

    Predictor columns describe information available
    by day t.

    Outcome columns describe what happened after t.

    Inactivity-rule baselines are calculated only
    from information available at t.
    """

    if (
        features.empty
        or outcomes.empty
    ):
        return pd.DataFrame()

    keys = [
        "participant_id",
        "date",
    ]

    for name, frame in (
        (
            "features",
            features,
        ),
        (
            "outcomes",
            outcomes,
        ),
    ):
        missing = (
            set(keys)
            - set(
                frame.columns
            )
        )

        if missing:
            raise ValueError(
                f"{name} table is missing keys: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        if frame.duplicated(
            subset=keys
        ).any():
            raise ValueError(
                f"{name} table contains duplicate "
                "participant-day rows."
            )

    result = features.merge(
        outcomes,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_outcome",
        ),
    )

    result[
        "days_since_last_explicit_engagement"
    ] = pd.to_numeric(
        result[
            "days_since_last_explicit_engagement"
        ],
        errors="coerce",
    )

    # ---------------------------------------------
    # Transparent rule baselines
    #
    # active:
    # rule is currently satisfied at landmark t.
    #
    # trigger:
    # this is the first calendar day at which the
    # threshold is crossed in the current inactivity
    # spell.
    # ---------------------------------------------

    for threshold in RULE_THRESHOLDS_DAYS:

        days_since = result[
            "days_since_last_explicit_engagement"
        ]

        result[
            f"inactivity_rule_{threshold}d_active"
        ] = (
            days_since.ge(
                threshold
            )
            .fillna(False)
            .astype("boolean")
        )

        result[
            f"inactivity_rule_{threshold}d_trigger"
        ] = (
            days_since.eq(
                threshold
            )
            .fillna(False)
            .astype("boolean")
        )

    result = (
        result
        .sort_values(
            keys
        )
        .reset_index(
            drop=True
        )
    )

    return result


def run_participant_day_modeling_table(
    config: TrajectoryAuditConfig,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build and save participant_day_modeling.csv.
    """

    result = (
        build_participant_day_modeling_table(
            features,
            outcomes,
        )
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    result.to_csv(
        os.path.join(
            config.output_dir,
            "participant_day_modeling.csv",
        ),
        index=False,
    )

    return result
