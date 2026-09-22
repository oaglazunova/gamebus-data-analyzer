from __future__ import annotations

import os

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)

from src.trajectory.participant_day_modeling import (
    OUTCOME_HORIZONS_DAYS,
    RULE_THRESHOLDS_DAYS,
)


EVALUATION_COLUMNS = [
    "rule_threshold_days",
    "outcome_horizon_days",

    "rows",
    "participants",

    "outcome_positive_rows",
    "outcome_positive_rate",

    "rule_positive_rows",
    "rule_positive_rate",

    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",

    "precision",
    "recall",
    "specificity",
    "negative_predictive_value",
    "accuracy",
    "balanced_accuracy",
    "f1",
]


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


def evaluate_inactivity_rule_baselines(
    modeling: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate 7/14/21-day inactivity rules against
    future continued-inactivity outcomes.

    Evaluation is performed at participant-day
    landmarks.

    Only rows whose future horizon is actually
    observable are included.

    Therefore right-censored rows with insufficient
    follow-up are excluded rather than incorrectly
    treated as negative outcomes.
    """

    rows: list[
        dict[str, object]
    ] = []

    if modeling.empty:
        return pd.DataFrame(
            columns=EVALUATION_COLUMNS
        )

    for threshold in RULE_THRESHOLDS_DAYS:

        prediction_column = (
            f"inactivity_rule_{threshold}d_active"
        )

        if (
            prediction_column
            not in modeling.columns
        ):
            raise ValueError(
                "Modeling table is missing "
                f"{prediction_column}."
            )

        for horizon in OUTCOME_HORIZONS_DAYS:

            available_column = (
                f"outcome_available_{horizon}d"
            )

            outcome_column = (
                f"continued_inactivity_{horizon}d"
            )

            missing = [
                column
                for column in (
                    available_column,
                    outcome_column,
                )
                if (
                    column
                    not in modeling.columns
                )
            ]

            if missing:
                raise ValueError(
                    "Modeling table is missing "
                    "outcome columns: "
                    + ", ".join(
                        missing
                    )
                )

            available = (
                modeling[
                    available_column
                ]
                .astype("boolean")
                .fillna(False)
            )

            evaluated = modeling.loc[
                available
            ].copy()

            if evaluated.empty:

                rows.append(
                    {
                        "rule_threshold_days": (
                            threshold
                        ),
                        "outcome_horizon_days": (
                            horizon
                        ),
                        "rows": 0,
                        "participants": 0,
                    }
                )

                continue

            predicted = (
                evaluated[
                    prediction_column
                ]
                .astype("boolean")
                .fillna(False)
            )

            observed = (
                evaluated[
                    outcome_column
                ]
                .astype("boolean")
            )

            # outcome_available=True should imply a
            # known outcome. Keep this invariant
            # explicit rather than silently filling.
            if observed.isna().any():
                raise ValueError(
                    "An outcome marked available "
                    f"for {horizon} days contains "
                    "missing values."
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

            n = len(
                evaluated
            )

            positives = int(
                observed.sum()
            )

            predicted_positives = int(
                predicted.sum()
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

            npv = _safe_divide(
                tn,
                tn + fn,
            )

            accuracy = _safe_divide(
                tp + tn,
                n,
            )

            balanced_accuracy = (
                (
                    recall
                    + specificity
                )
                / 2
                if (
                    recall is not None
                    and specificity
                    is not None
                )
                else None
            )

            f1 = (
                (
                    2
                    * precision
                    * recall
                    / (
                        precision
                        + recall
                    )
                )
                if (
                    precision is not None
                    and recall is not None
                    and (
                        precision
                        + recall
                    ) > 0
                )
                else None
            )

            rows.append(
                {
                    "rule_threshold_days": (
                        threshold
                    ),
                    "outcome_horizon_days": (
                        horizon
                    ),

                    "rows": n,

                    "participants": int(
                        evaluated[
                            "participant_id"
                        ].nunique()
                    ),

                    "outcome_positive_rows": (
                        positives
                    ),

                    "outcome_positive_rate": (
                        _safe_divide(
                            positives,
                            n,
                        )
                    ),

                    "rule_positive_rows": (
                        predicted_positives
                    ),

                    "rule_positive_rate": (
                        _safe_divide(
                            predicted_positives,
                            n,
                        )
                    ),

                    "true_positive": tp,
                    "false_positive": fp,
                    "true_negative": tn,
                    "false_negative": fn,

                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,

                    "negative_predictive_value": (
                        npv
                    ),

                    "accuracy": accuracy,

                    "balanced_accuracy": (
                        balanced_accuracy
                    ),

                    "f1": f1,
                }
            )

    result = pd.DataFrame(
        rows
    )

    for column in EVALUATION_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    return result[
        EVALUATION_COLUMNS
    ]


def run_rule_baseline_evaluation(
    config: TrajectoryAuditConfig,
    modeling: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        evaluate_inactivity_rule_baselines(
            modeling
        )
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    result.to_csv(
        os.path.join(
            config.output_dir,
            "rule_baseline_evaluation.csv",
        ),
        index=False,
    )

    return result