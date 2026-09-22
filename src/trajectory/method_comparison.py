from __future__ import annotations

import os

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)

from src.trajectory.participant_day_modeling import (
    OUTCOME_HORIZONS_DAYS,
)


COMPARISON_COLUMNS = [
    "method_id",
    "method_family",
    "method_parameter",
    "evaluation_scope",

    "outcome_horizon_days",

    "rows",
    "participants",

    "outcome_positive_rows",
    "outcome_positive_rate",

    "method_positive_rows",
    "method_positive_rate",

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


def _metric_row(
    *,
    frame: pd.DataFrame,
    predicted: pd.Series,
    observed: pd.Series,
    method_id: str,
    method_family: str,
    method_parameter: str,
    horizon: int,
) -> dict[str, object]:

    n = len(
        frame
    )

    if n == 0:

        return {
            "method_id": method_id,
            "method_family": method_family,
            "method_parameter": (
                method_parameter
            ),
            "evaluation_scope": (
                "common_individualized_assessable"
            ),
            "outcome_horizon_days": (
                horizon
            ),
            "rows": 0,
            "participants": 0,
        }

    predicted = (
        predicted
        .astype("boolean")
        .fillna(False)
    )

    observed = (
        observed
        .astype("boolean")
    )

    if observed.isna().any():

        raise ValueError(
            "An outcome marked available "
            "contains missing values."
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
            and specificity is not None
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

    return {
        "method_id": method_id,
        "method_family": method_family,
        "method_parameter": (
            method_parameter
        ),

        "evaluation_scope": (
            "common_individualized_assessable"
        ),

        "outcome_horizon_days": (
            horizon
        ),

        "rows": n,

        "participants": int(
            frame[
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

        "method_positive_rows": (
            predicted_positives
        ),

        "method_positive_rate": (
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


def build_method_comparison(
    modeling: pd.DataFrame,
    individualized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare fixed inactivity rules with the
    individualized change detector.

    For a fair head-to-head comparison, all methods
    are evaluated on the same participant-day rows:

        future outcome is observable

    AND

        individualized change detection is
        assessable.

    The existing rule_baseline_evaluation.csv still
    provides the fixed-rule results over their full
    evaluable population.
    """

    if (
        modeling.empty
        or individualized.empty
    ):
        return pd.DataFrame(
            columns=COMPARISON_COLUMNS
        )

    keys = [
        "participant_id",
        "date",
    ]

    left = modeling.copy()
    right = individualized.copy()

    for frame in (
        left,
        right,
    ):

        frame[
            "participant_id"
        ] = pd.to_numeric(
            frame[
                "participant_id"
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

        frame["date"] = (
            pd.to_datetime(
                frame["date"],
                errors="coerce",
            )
            .dt.date
        )

        if frame.duplicated(
            subset=keys
        ).any():

            raise ValueError(
                "Method comparison requires unique "
                "participant-day rows."
            )

    change_columns = [
        "participant_id",
        "date",
        "individual_change_assessable",
        "individual_change_signal",
        "individual_change_confirmed",
        "individual_change_trigger",
        "current_change_onset_date",
        "current_change_trigger_date",
    ]

    missing = [
        column
        for column in change_columns
        if (
            column
            not in right.columns
        )
    ]

    if missing:

        raise ValueError(
            "Individualized change table is missing "
            "columns: "
            + ", ".join(
                missing
            )
        )

    merged = left.merge(
        right[
            change_columns
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    assessable = (
        merged[
            "individual_change_assessable"
        ]
        .astype("boolean")
        .fillna(False)
    )

    methods = [
        (
            "fixed_inactivity_7d",
            "fixed_inactivity_rule",
            "threshold=7d",
            "inactivity_rule_7d_active",
        ),
        (
            "fixed_inactivity_14d",
            "fixed_inactivity_rule",
            "threshold=14d",
            "inactivity_rule_14d_active",
        ),
        (
            "fixed_inactivity_21d",
            "fixed_inactivity_rule",
            "threshold=21d",
            "inactivity_rule_21d_active",
        ),
        (
            "individualized_change",
            "individualized_change",
            (
                "recent=7d;"
                "reference=28d;"
                "ratio<=0.50;"
                "persistence=3d"
            ),
            "individual_change_confirmed",
        ),
    ]

    rows: list[
        dict[str, object]
    ] = []

    for horizon in OUTCOME_HORIZONS_DAYS:

        available_column = (
            f"outcome_available_{horizon}d"
        )

        outcome_column = (
            f"continued_inactivity_{horizon}d"
        )

        for column in (
            available_column,
            outcome_column,
        ):

            if column not in merged.columns:

                raise ValueError(
                    "Modeling table is missing "
                    f"{column}."
                )

        outcome_available = (
            merged[
                available_column
            ]
            .astype("boolean")
            .fillna(False)
        )

        common = (
            assessable
            & outcome_available
        )

        evaluated = merged.loc[
            common
        ].copy()

        for (
            method_id,
            method_family,
            method_parameter,
            prediction_column,
        ) in methods:

            if (
                prediction_column
                not in evaluated.columns
            ):

                raise ValueError(
                    "Method comparison is missing "
                    f"{prediction_column}."
                )

            rows.append(
                _metric_row(
                    frame=evaluated,

                    predicted=(
                        evaluated[
                            prediction_column
                        ]
                    ),

                    observed=(
                        evaluated[
                            outcome_column
                        ]
                    ),

                    method_id=method_id,

                    method_family=(
                        method_family
                    ),

                    method_parameter=(
                        method_parameter
                    ),

                    horizon=horizon,
                )
            )

    result = pd.DataFrame(
        rows
    )

    for column in COMPARISON_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    return result[
        COMPARISON_COLUMNS
    ]


def run_method_comparison(
    config: TrajectoryAuditConfig,
    modeling: pd.DataFrame,
    individualized: pd.DataFrame,
) -> pd.DataFrame:

    result = build_method_comparison(
        modeling,
        individualized,
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    result.to_csv(
        os.path.join(
            config.output_dir,
            "method_comparison.csv",
        ),
        index=False,
    )

    return result