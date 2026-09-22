from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit


DEFAULT_DYNAMIC_FEATURES = (
    "log1p_days_since_last_explicit_engagement",
    "active_days_28d",
    "active_day_ratio_28d",
    "reengagements_to_date",
    "explicit_domain_diversity_28d",
    "explicit_tool_diversity_28d",
)

TARGET_COLUMN = (
    "explicit_engagement_next_day"
)


@dataclass(
    frozen=True
)
class DiscreteTimeHazardModel:
    """
    Transparent pooled-logistic discrete-time
    re-engagement hazard model.

    Coefficients operate on standardized predictors.
    No inferential p-values or confidence intervals
    are implied by this object.
    """

    feature_names: tuple[str, ...]

    intercept: float
    coefficients: tuple[float, ...]

    imputation_values: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    l2_penalty: float

    def predict_hazard(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """
        Estimate:

            P(
                explicit engagement on t+1
                |
                history available by t
            )
        """

        matrix = _transform_matrix(
            frame,
            feature_names=(
                self.feature_names
            ),
            imputation_values=(
                self.imputation_values
            ),
            means=self.means,
            scales=self.scales,
        )

        linear_predictor = (
            self.intercept
            + matrix
            @ np.asarray(
                self.coefficients,
                dtype=float,
            )
        )

        return pd.Series(
            expit(
                linear_predictor
            ),
            index=frame.index,
            dtype="Float64",
        )

    def coefficient_table(
        self,
    ) -> pd.DataFrame:
        """
        Return standardized model coefficients.

        exp(coefficient) is the multiplicative change
        in next-day re-engagement odds associated with
        a one-standard-deviation increase in the
        predictor, conditional on the other included
        predictors.

        These are descriptive model parameters, not
        causal effects.
        """

        coefficients = np.asarray(
            self.coefficients,
            dtype=float,
        )

        return pd.DataFrame(
            {
                "feature": (
                    self.feature_names
                ),
                (
                    "standardized_coefficient"
                ): coefficients,
                (
                    "standardized_odds_ratio"
                ): np.exp(
                    coefficients
                ),
            }
        )


def build_reengagement_hazard_dataset(
    modeling: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the discrete-time re-engagement
    risk set.

    Included landmark days must satisfy:

        maintenance/re-engagement phase already
        started

        participant has been inactive for at least
        one day

        the next calendar day is observable

        next-day engagement outcome is known

    Therefore active days themselves are not used as
    re-engagement risk-set rows.
    """

    if modeling.empty:
        return pd.DataFrame()

    required = {
        "participant_id",
        "date",
        "days_since_last_explicit_engagement",
        "next_day_outcome_available",
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(
            modeling.columns
        )
    )

    if missing:
        raise ValueError(
            "Modeling table is missing required "
            "dynamic event-history columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result = modeling.copy()

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
        errors="coerce",
    ).dt.date

    result[
        "days_since_last_explicit_engagement"
    ] = pd.to_numeric(
        result[
            "days_since_last_explicit_engagement"
        ],
        errors="coerce",
    )

    outcome_available = (
        result[
            "next_day_outcome_available"
        ]
        .astype("boolean")
        .fillna(False)
    )

    target_known = result[
        TARGET_COLUMN
    ].notna()

    inactive = result[
        "days_since_last_explicit_engagement"
    ].ge(
        1
    )

    result = result.loc[
        outcome_available
        & target_known
        & inactive
    ].copy()

    result[
        TARGET_COLUMN
    ] = (
        result[
            TARGET_COLUMN
        ]
        .astype("boolean")
        .astype(int)
    )

    result[
        "log1p_days_since_last_explicit_engagement"
    ] = np.log1p(
        result[
            "days_since_last_explicit_engagement"
        ]
    )

    return (
        result
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


def _fit_preprocessing(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> tuple[
    np.ndarray,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """
    Learn numeric imputation and standardization from
    TRAINING data only.
    """

    columns: list[
        np.ndarray
    ] = []

    imputation_values: list[
        float
    ] = []

    means: list[
        float
    ] = []

    scales: list[
        float
    ] = []

    for feature in feature_names:

        if feature not in frame.columns:
            raise ValueError(
                "Dynamic model feature is missing: "
                f"{feature}"
            )

        values = pd.to_numeric(
            frame[
                feature
            ],
            errors="coerce",
        ).astype(
            float
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        if finite.empty:
            imputation = 0.0
        else:
            imputation = float(
                finite.median()
            )

        filled = (
            values
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                imputation
            )
            .to_numpy(
                dtype=float
            )
        )

        mean = float(
            np.mean(
                filled
            )
        )

        scale = float(
            np.std(
                filled,
                ddof=0,
            )
        )

        if (
            not np.isfinite(
                scale
            )
            or scale < 1e-12
        ):
            scale = 1.0

        standardized = (
            filled
            - mean
        ) / scale

        columns.append(
            standardized
        )

        imputation_values.append(
            imputation
        )

        means.append(
            mean
        )

        scales.append(
            scale
        )

    matrix = np.column_stack(
        columns
    )

    return (
        matrix,
        tuple(
            imputation_values
        ),
        tuple(
            means
        ),
        tuple(
            scales
        ),
    )


def _transform_matrix(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    imputation_values: tuple[
        float,
        ...
    ],
    means: tuple[
        float,
        ...
    ],
    scales: tuple[
        float,
        ...
    ],
) -> np.ndarray:

    if not (
        len(
            feature_names
        )
        == len(
            imputation_values
        )
        == len(
            means
        )
        == len(
            scales
        )
    ):
        raise ValueError(
            "Dynamic model preprocessing metadata "
            "has inconsistent lengths."
        )

    columns: list[
        np.ndarray
    ] = []

    for (
        feature,
        imputation,
        mean,
        scale,
    ) in zip(
        feature_names,
        imputation_values,
        means,
        scales,
        strict=True,
    ):

        if feature not in frame.columns:
            raise ValueError(
                "Dynamic model feature is missing: "
                f"{feature}"
            )

        values = (
            pd.to_numeric(
                frame[
                    feature
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                imputation
            )
            .to_numpy(
                dtype=float
            )
        )

        columns.append(
            (
                values
                - mean
            )
            / scale
        )

    return np.column_stack(
        columns
    )


def fit_discrete_time_hazard_model(
    training: pd.DataFrame,
    *,
    feature_names: tuple[
        str,
        ...
    ] = DEFAULT_DYNAMIC_FEATURES,
    l2_penalty: float = 0.10,
) -> DiscreteTimeHazardModel:
    """
    Fit a regularized pooled-logistic discrete-time
    re-engagement hazard model.

    The L2 penalty stabilizes estimates in relatively
    small campaign datasets and under partial
    separation.

    The intercept is not penalized.
    """

    if training.empty:
        raise ValueError(
            "Cannot fit the dynamic event-history "
            "model on an empty training table."
        )

    if TARGET_COLUMN not in training.columns:
        raise ValueError(
            "Training table is missing "
            f"{TARGET_COLUMN}."
        )

    if l2_penalty < 0:
        raise ValueError(
            "l2_penalty must be non-negative."
        )

    target = pd.to_numeric(
        training[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    valid_target = target.isin(
        [
            0,
            1,
        ]
    )

    frame = training.loc[
        valid_target
    ].copy()

    target = (
        target.loc[
            valid_target
        ]
        .to_numpy(
            dtype=float
        )
    )

    if len(
        np.unique(
            target
        )
    ) < 2:
        raise ValueError(
            "Dynamic event-history training requires "
            "both re-engagement and non-re-engagement "
            "next-day outcomes."
        )

    (
        matrix,
        imputation_values,
        means,
        scales,
    ) = _fit_preprocessing(
        frame,
        feature_names,
    )

    sample_count = len(
        target
    )

    def objective(
        parameters: np.ndarray,
    ) -> tuple[
        float,
        np.ndarray,
    ]:

        intercept = parameters[
            0
        ]

        coefficients = parameters[
            1:
        ]

        linear_predictor = (
            intercept
            + matrix
            @ coefficients
        )

        # Numerically stable logistic negative
        # log-likelihood.
        data_loss = np.mean(
            np.logaddexp(
                0.0,
                linear_predictor,
            )
            - (
                target
                * linear_predictor
            )
        )

        penalty = (
            0.5
            * l2_penalty
            * float(
                coefficients
                @ coefficients
            )
        )

        probability = expit(
            linear_predictor
        )

        residual = (
            probability
            - target
        )

        gradient_intercept = float(
            np.mean(
                residual
            )
        )

        gradient_coefficients = (
            (
                matrix.T
                @ residual
            )
            / sample_count
            + (
                l2_penalty
                * coefficients
            )
        )

        gradient = np.concatenate(
            [
                np.asarray(
                    [
                        gradient_intercept,
                    ]
                ),
                gradient_coefficients,
            ]
        )

        return (
            float(
                data_loss
                + penalty
            ),
            gradient,
        )

    initial = np.zeros(
        len(
            feature_names
        )
        + 1,
        dtype=float,
    )

    fitted = minimize(
        fun=lambda parameters: (
            objective(
                parameters
            )[0]
        ),
        x0=initial,
        jac=lambda parameters: (
            objective(
                parameters
            )[1]
        ),
        method="L-BFGS-B",
    )

    if not fitted.success:
        raise RuntimeError(
            "Dynamic event-history model fitting "
            "failed: "
            f"{fitted.message}"
        )

    parameters = fitted.x

    return DiscreteTimeHazardModel(
        feature_names=tuple(
            feature_names
        ),
        intercept=float(
            parameters[
                0
            ]
        ),
        coefficients=tuple(
            float(
                value
            )
            for value in parameters[
                1:
            ]
        ),
        imputation_values=(
            imputation_values
        ),
        means=means,
        scales=scales,
        l2_penalty=float(
            l2_penalty
        ),
    )


def predict_reengagement_hazard(
    model: DiscreteTimeHazardModel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach estimated next-day re-engagement hazard.
    """

    result = frame.copy()

    result[
        "predicted_next_day_reengagement_probability"
    ] = model.predict_hazard(
        result
    )

    return result