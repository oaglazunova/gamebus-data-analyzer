from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.trajectory.dynamic_event_history import (
    build_reengagement_hazard_dataset,
    fit_discrete_time_hazard_model,
)


class TestDynamicEventHistory(
    unittest.TestCase
):

    def test_risk_set_excludes_active_and_censored_rows(
        self,
    ) -> None:

        modeling = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    1,
                    1,
                    1,
                ],

                "date": pd.date_range(
                    "2026-01-01",
                    periods=4,
                    freq="D",
                ).date,

                (
                    "days_since_last_explicit_"
                    "engagement"
                ): [
                    0,
                    1,
                    2,
                    3,
                ],

                "next_day_outcome_available": [
                    True,
                    True,
                    True,
                    False,
                ],

                "explicit_engagement_next_day": (
                    pd.Series(
                        [
                            False,
                            False,
                            True,
                            pd.NA,
                        ],
                        dtype="boolean",
                    )
                ),

                "active_days_28d": [
                    5,
                    5,
                    5,
                    5,
                ],

                "active_day_ratio_28d": [
                    1.0,
                    0.8,
                    0.6,
                    0.4,
                ],

                "reengagements_to_date": [
                    0,
                    0,
                    0,
                    0,
                ],

                "explicit_domain_diversity_28d": [
                    2,
                    2,
                    2,
                    2,
                ],

                "explicit_tool_diversity_28d": [
                    1,
                    1,
                    1,
                    1,
                ],
            }
        )

        result = (
            build_reengagement_hazard_dataset(
                modeling
            )
        )

        self.assertEqual(
            len(
                result
            ),
            2,
        )

        self.assertEqual(
            result[
                (
                    "days_since_last_explicit_"
                    "engagement"
                )
            ].tolist(),
            [
                1,
                2,
            ],
        )

        self.assertEqual(
            result[
                (
                    "explicit_engagement_"
                    "next_day"
                )
            ].tolist(),
            [
                0,
                1,
            ],
        )

    def test_model_learns_lower_hazard_for_longer_gap(
        self,
    ) -> None:

        rows = []

        for participant in range(
            1,
            21,
        ):

            for gap in range(
                1,
                11,
            ):

                rows.append(
                    {
                        "participant_id": (
                            participant
                        ),

                        "date": (
                            pd.Timestamp(
                                "2026-01-01"
                            )
                            + pd.Timedelta(
                                days=gap
                            )
                        ).date(),

                        (
                            "log1p_days_since_last_"
                            "explicit_engagement"
                        ): np.log1p(
                            gap
                        ),

                        (
                            "explicit_engagement_"
                            "next_day"
                        ): (
                            1
                            if gap <= 2
                            else 0
                        ),
                    }
                )

        training = pd.DataFrame(
            rows
        )

        model = (
            fit_discrete_time_hazard_model(
                training,
                feature_names=(
                    (
                        "log1p_days_since_last_"
                        "explicit_engagement"
                    ),
                ),
            )
        )

        test = pd.DataFrame(
            {
                (
                    "log1p_days_since_last_"
                    "explicit_engagement"
                ): [
                    np.log1p(
                        1
                    ),
                    np.log1p(
                        10
                    ),
                ]
            }
        )

        hazard = model.predict_hazard(
            test
        )

        self.assertGreater(
            hazard.iloc[
                0
            ],
            hazard.iloc[
                1
            ],
        )

        self.assertTrue(
            hazard.between(
                0,
                1,
            ).all()
        )

    def test_training_requires_both_outcomes(
        self,
    ) -> None:

        training = pd.DataFrame(
            {
                (
                    "log1p_days_since_last_"
                    "explicit_engagement"
                ): [
                    1.0,
                    2.0,
                    3.0,
                ],

                (
                    "explicit_engagement_"
                    "next_day"
                ): [
                    0,
                    0,
                    0,
                ],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "both re-engagement and non-re-engagement",
        ):
            fit_discrete_time_hazard_model(
                training,
                feature_names=(
                    (
                        "log1p_days_since_last_"
                        "explicit_engagement"
                    ),
                ),
            )

    def test_training_preprocessing_is_reused_for_prediction(
        self,
    ) -> None:

        training = pd.DataFrame(
            {
                (
                    "log1p_days_since_last_"
                    "explicit_engagement"
                ): [
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                ],

                (
                    "explicit_engagement_"
                    "next_day"
                ): [
                    1,
                    1,
                    0,
                    0,
                ],
            }
        )

        model = (
            fit_discrete_time_hazard_model(
                training,
                feature_names=(
                    (
                        "log1p_days_since_last_"
                        "explicit_engagement"
                    ),
                ),
            )
        )

        original_mean = model.means[
            0
        ]

        # Extreme evaluation values must not alter
        # parameters learned from training.
        evaluation = pd.DataFrame(
            {
                (
                    "log1p_days_since_last_"
                    "explicit_engagement"
                ): [
                    1000.0,
                    2000.0,
                ]
            }
        )

        model.predict_hazard(
            evaluation
        )

        self.assertEqual(
            model.means[
                0
            ],
            original_mean,
        )


if __name__ == "__main__":
    unittest.main()