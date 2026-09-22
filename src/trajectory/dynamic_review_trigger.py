from __future__ import annotations

from dataclasses import dataclass
import os

import pandas as pd


PROBABILITY_COLUMN = (
    "predicted_next_day_reengagement_probability"
)


@dataclass(
    frozen=True
)
class DynamicReviewTriggerConfig:
    """
    Frozen review-trigger configuration derived
    exclusively from development data.

    inactivity_risk_score =

        1 - predicted next-day re-engagement
            probability

    This score is used as a ranking/calibration score.
    It is not interpreted as a probability of
    14-day continued inactivity.
    """

    inactivity_risk_threshold: float

    persistence_days: int = 3

    calibration_outcome_horizon_days: int = 14

    selection_metric: str = (
        "balanced_accuracy"
    )


def _safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float | None:

    if denominator == 0:
        return None

    return float(
        numerator
        / denominator
    )


def _metrics(
    predicted: pd.Series,
    observed: pd.Series,
) -> dict[str, float | int | None]:

    predicted = (
        predicted
        .astype("boolean")
        .fillna(False)
    )

    observed = observed.astype(
        "boolean"
    )

    if observed.isna().any():
        raise ValueError(
            "Observed calibration outcomes "
            "contain missing values."
        )

    tp = int(
        (
            predicted
            & observed
        ).sum()
    )

    fp = int(
        (
            predicted
            & ~observed
        ).sum()
    )

    tn = int(
        (
            ~predicted
            & ~observed
        ).sum()
    )

    fn = int(
        (
            ~predicted
            & observed
        ).sum()
    )

    precision = _safe_divide(
        tp,
        tp + fp,
    )

    recall = _safe_divide(
        tp,
        tp + fn,
    )

    specificity = _safe_divide(
        tn,
        tn + fp,
    )

    balanced_accuracy = (
        (
            recall
            + specificity
        )
        / 2
        if (
            recall is not None
            and specificity is not None
        )
        else None
    )

    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,

        "precision": precision,
        "recall": recall,
        "specificity": specificity,

        "balanced_accuracy": (
            balanced_accuracy
        ),

        "prediction_positive_rate": (
            _safe_divide(
                int(
                    predicted.sum()
                ),
                len(
                    predicted
                ),
            )
        ),
    }


def calibrate_dynamic_review_trigger(
    development_predictions: pd.DataFrame,
    *,
    outcome_horizon_days: int = 14,
    persistence_days: int = 3,
) -> tuple[
    DynamicReviewTriggerConfig,
    pd.DataFrame,
]:
    """
    Select a low-reengagement threshold using
    DEVELOPMENT DATA ONLY.

    The calibration target is future continued
    inactivity over the configured horizon.

    Right-censored rows without complete outcome
    information are excluded.

    Threshold selection criterion:

        maximum balanced accuracy

    Tie-breaking:

        1. higher precision
        2. higher threshold

    The final tie-break is intentionally conservative:
    when empirical performance is identical, prefer
    the threshold that generates fewer review
    signals.
    """

    if development_predictions.empty:
        raise ValueError(
            "Development predictions are empty."
        )

    available_column = (
        f"outcome_available_"
        f"{outcome_horizon_days}d"
    )

    outcome_column = (
        f"continued_inactivity_"
        f"{outcome_horizon_days}d"
    )

    required = {
        PROBABILITY_COLUMN,
        available_column,
        outcome_column,
    }

    missing = (
        required
        - set(
            development_predictions.columns
        )
    )

    if missing:
        raise ValueError(
            "Development predictions are missing "
            "trigger-calibration columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    frame = (
        development_predictions
        .copy()
    )

    frame[
        PROBABILITY_COLUMN
    ] = pd.to_numeric(
        frame[
            PROBABILITY_COLUMN
        ],
        errors="coerce",
    )

    available = (
        frame[
            available_column
        ]
        .astype("boolean")
        .fillna(False)
    )

    frame = frame.loc[
        available
        & frame[
            PROBABILITY_COLUMN
        ].notna()
        & frame[
            outcome_column
        ].notna()
    ].copy()

    if frame.empty:
        raise ValueError(
            "No development rows have an "
            "observable calibration outcome."
        )

    observed = (
        frame[
            outcome_column
        ]
        .astype("boolean")
    )

    if observed.nunique(
        dropna=True
    ) < 2:
        raise ValueError(
            "Trigger calibration requires both "
            "continued-inactivity and re-engagement "
            "outcomes."
        )

    frame[
        "inactivity_risk_score"
    ] = (
        1.0
        - frame[
            PROBABILITY_COLUMN
        ]
    )

    candidates = sorted(
        frame[
            "inactivity_risk_score"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    rows: list[
        dict[str, object]
    ] = []

    for threshold in candidates:

        predicted = (
            frame[
                "inactivity_risk_score"
            ]
            .ge(
                threshold
            )
        )

        metrics = _metrics(
            predicted,
            observed,
        )

        rows.append(
            {
                "inactivity_risk_threshold": (
                    float(
                        threshold
                    )
                ),

                "rows": len(
                    frame
                ),

                "participants": int(
                    frame[
                        "participant_id"
                    ].nunique()
                )
                if (
                    "participant_id"
                    in frame.columns
                )
                else pd.NA,

                "outcome_horizon_days": (
                    outcome_horizon_days
                ),

                **metrics,
            }
        )

    calibration = pd.DataFrame(
        rows
    )

    calibration[
        "_precision_rank"
    ] = (
        pd.to_numeric(
            calibration[
                "precision"
            ],
            errors="coerce",
        )
        .fillna(
            -1.0
        )
    )

    calibration[
        "_balanced_rank"
    ] = (
        pd.to_numeric(
            calibration[
                "balanced_accuracy"
            ],
            errors="coerce",
        )
        .fillna(
            -1.0
        )
    )

    selected = (
        calibration
        .sort_values(
            [
                "_balanced_rank",
                "_precision_rank",
                "inactivity_risk_threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[
            0
        ]
    )

    calibration[
        "selected"
    ] = False

    calibration.loc[
        calibration[
            "inactivity_risk_threshold"
        ].eq(
            selected[
                "inactivity_risk_threshold"
            ]
        ),
        "selected",
    ] = True

    calibration = calibration.drop(
        columns=[
            "_precision_rank",
            "_balanced_rank",
        ]
    )

    config = DynamicReviewTriggerConfig(
        inactivity_risk_threshold=float(
            selected[
                "inactivity_risk_threshold"
            ]
        ),
        persistence_days=(
            persistence_days
        ),
        calibration_outcome_horizon_days=(
            outcome_horizon_days
        ),
    )

    return (
        config,
        calibration,
    )


def apply_dynamic_review_trigger(
    predictions: pd.DataFrame,
    config: DynamicReviewTriggerConfig,
) -> pd.DataFrame:
    """
    Apply a FROZEN trigger configuration.

    t_change:
        first day of the current uninterrupted
        low-reengagement signal.

    t_trigger:
        first day on which that signal has persisted
        for the required number of consecutive
        calendar days.

    A missing risk-set day breaks the streak. This is
    important because an omitted day may correspond
    to explicit re-engagement.
    """

    if predictions.empty:
        result = predictions.copy()

        for column in (
            "inactivity_risk_score",
            "dynamic_low_reengagement_signal",
            "dynamic_signal_streak_days",
            "dynamic_change_onset_date",
            "dynamic_review_confirmed",
            "dynamic_review_trigger",
            "current_dynamic_trigger_date",
        ):
            result[
                column
            ] = pd.Series(
                dtype="object"
            )

        return result

    required = {
        "participant_id",
        "date",
        PROBABILITY_COLUMN,
    }

    missing = (
        required
        - set(
            predictions.columns
        )
    )

    if missing:
        raise ValueError(
            "Dynamic predictions are missing "
            "review-trigger columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result = predictions.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        utc=True,
        errors="coerce",
    )

    result[
        PROBABILITY_COLUMN
    ] = pd.to_numeric(
        result[
            PROBABILITY_COLUMN
        ],
        errors="coerce",
    )

    result[
        "inactivity_risk_score"
    ] = (
        1.0
        - result[
            PROBABILITY_COLUMN
        ]
    )

    result[
        "dynamic_low_reengagement_signal"
    ] = (
        result[
            "inactivity_risk_score"
        ]
        .ge(
            config
            .inactivity_risk_threshold
        )
        .fillna(False)
        .astype("boolean")
    )

    result[
        "dynamic_signal_streak_days"
    ] = 0

    result[
        "dynamic_change_onset_date"
    ] = pd.NaT

    result[
        "dynamic_review_confirmed"
    ] = False

    result[
        "dynamic_review_trigger"
    ] = False

    result[
        "current_dynamic_trigger_date"
    ] = pd.NaT

    group_columns = [
        "participant_id",
    ]

    if "campaign_id" in result.columns:
        group_columns.insert(
            0,
            "campaign_id",
        )

    result = (
        result
        .sort_values(
            group_columns
            + [
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    for (
        _group_key,
        group,
    ) in result.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):

        streak = 0

        onset_date = None
        trigger_date = None

        previous_date = None

        for index in group.index:

            current_date = result.at[
                index,
                "date",
            ]

            contiguous = (
                previous_date is not None
                and pd.notna(
                    previous_date
                )
                and pd.notna(
                    current_date
                )
                and (
                    current_date
                    - previous_date
                ).days
                == 1
            )

            if (
                previous_date is not None
                and not contiguous
            ):
                streak = 0
                onset_date = None
                trigger_date = None

            signal = bool(
                result.at[
                    index,
                    (
                        "dynamic_low_"
                        "reengagement_signal"
                    ),
                ]
            )

            if signal:

                if streak == 0:
                    onset_date = (
                        current_date
                    )

                    trigger_date = None

                streak += 1

                result.at[
                    index,
                    "dynamic_signal_streak_days",
                ] = streak

                result.at[
                    index,
                    "dynamic_change_onset_date",
                ] = onset_date

                confirmed = (
                    streak
                    >= config.persistence_days
                )

                result.at[
                    index,
                    "dynamic_review_confirmed",
                ] = confirmed

                if (
                    streak
                    == config.persistence_days
                ):
                    trigger_date = (
                        current_date
                    )

                    result.at[
                        index,
                        "dynamic_review_trigger",
                    ] = True

                if trigger_date is not None:
                    result.at[
                        index,
                        (
                            "current_dynamic_"
                            "trigger_date"
                        ),
                    ] = trigger_date

            else:
                streak = 0
                onset_date = None
                trigger_date = None

            previous_date = (
                current_date
            )

    result[
        "dynamic_signal_streak_days"
    ] = pd.to_numeric(
        result[
            "dynamic_signal_streak_days"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    for column in (
        "dynamic_low_reengagement_signal",
        "dynamic_review_confirmed",
        "dynamic_review_trigger",
    ):
        result[
            column
        ] = result[
            column
        ].astype(
            "boolean"
        )

    for column in (
        "date",
        "dynamic_change_onset_date",
        "current_dynamic_trigger_date",
    ):
        result[
            column
        ] = (
            pd.to_datetime(
                result[
                    column
                ],
                utc=True,
                errors="coerce",
            )
            .dt.date
        )

    return result


def save_dynamic_review_trigger(
    *,
    output_dir: str,
    config: DynamicReviewTriggerConfig,
    calibration: pd.DataFrame,
) -> None:

    from dataclasses import asdict
    import json

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    with open(
        os.path.join(
            output_dir,
            "dynamic_review_trigger.json",
        ),
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            asdict(
                config
            ),
            handle,
            indent=2,
        )

    calibration.to_csv(
        os.path.join(
            output_dir,
            "dynamic_review_trigger_calibration.csv",
        ),
        index=False,
    )