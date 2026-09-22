from __future__ import annotations

import unittest

import pandas as pd

from src.trajectory.participant_day_modeling import (
    build_participant_day_modeling_table,
)

from src.trajectory.rule_baseline_evaluation import (
    evaluate_inactivity_rule_baselines,
)


class TestRuleBaselineEvaluation(
    unittest.TestCase
):

    def test_rule_columns_follow_landmark_state(
        self,
    ) -> None:

        features = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    1,
                    1,
                ],
                "date": pd.to_datetime(
                    [
                        "2026-01-07",
                        "2026-01-08",
                        "2026-01-15",
                    ]
                ).date,
                (
                    "days_since_last_explicit_"
                    "engagement"
                ): [
                    6,
                    7,
                    14,
                ],
            }
        )

        outcomes = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    1,
                    1,
                ],
                "date": features[
                    "date"
                ],
                "outcome_available_7d": [
                    True,
                    True,
                    True,
                ],
                "continued_inactivity_7d": [
                    True,
                    True,
                    False,
                ],
            }
        )

        result = (
            build_participant_day_modeling_table(
                features,
                outcomes,
            )
        )

        self.assertEqual(
            result[
                "inactivity_rule_7d_active"
            ].tolist(),
            [
                False,
                True,
                True,
            ],
        )

        self.assertEqual(
            result[
                "inactivity_rule_7d_trigger"
            ].tolist(),
            [
                False,
                True,
                False,
            ],
        )

        self.assertEqual(
            result[
                "inactivity_rule_14d_trigger"
            ].tolist(),
            [
                False,
                False,
                True,
            ],
        )

    def test_censored_rows_are_not_evaluated(
        self,
    ) -> None:

        modeling = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    2,
                    3,
                ],
                (
                    "inactivity_rule_7d_active"
                ): [
                    True,
                    True,
                    False,
                ],
                (
                    "inactivity_rule_14d_active"
                ): [
                    False,
                    False,
                    False,
                ],
                (
                    "inactivity_rule_21d_active"
                ): [
                    False,
                    False,
                    False,
                ],
            }
        )

        for horizon in (
            7,
            14,
            21,
            30,
        ):

            modeling[
                f"outcome_available_{horizon}d"
            ] = [
                True,
                False,
                True,
            ]

            modeling[
                f"continued_inactivity_{horizon}d"
            ] = pd.Series(
                [
                    True,
                    pd.NA,
                    False,
                ],
                dtype="boolean",
            )

        result = (
            evaluate_inactivity_rule_baselines(
                modeling
            )
        )

        row = result.loc[
            (
                result[
                    "rule_threshold_days"
                ]
                == 7
            )
            & (
                result[
                    "outcome_horizon_days"
                ]
                == 7
            )
        ].iloc[
            0
        ]

        # Participant 2 is censored before the
        # horizon and therefore excluded.
        self.assertEqual(
            row["rows"],
            2,
        )

        self.assertEqual(
            row["true_positive"],
            1,
        )

        self.assertEqual(
            row["true_negative"],
            1,
        )

        self.assertEqual(
            row["false_positive"],
            0,
        )

        self.assertEqual(
            row["false_negative"],
            0,
        )


if __name__ == "__main__":
    unittest.main()