from __future__ import annotations

import unittest

import pandas as pd

from src.trajectory.method_comparison import (
    build_method_comparison,
)


class TestMethodComparison(
    unittest.TestCase
):

    def _modeling(
        self,
    ) -> pd.DataFrame:

        dates = pd.date_range(
            "2026-01-01",
            periods=4,
            freq="D",
        ).date

        frame = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    1,
                    1,
                    1,
                ],
                "date": dates,

                "inactivity_rule_7d_active": [
                    True,
                    True,
                    False,
                    False,
                ],

                "inactivity_rule_14d_active": [
                    False,
                    True,
                    True,
                    False,
                ],

                "inactivity_rule_21d_active": [
                    False,
                    False,
                    True,
                    True,
                ],
            }
        )

        for horizon in (
            7,
            14,
            21,
            30,
        ):

            frame[
                f"outcome_available_{horizon}d"
            ] = [
                True,
                True,
                True,
                True,
            ]

            frame[
                f"continued_inactivity_{horizon}d"
            ] = pd.Series(
                [
                    True,
                    False,
                    True,
                    False,
                ],
                dtype="boolean",
            )

        return frame

    def _individualized(
        self,
    ) -> pd.DataFrame:

        dates = pd.date_range(
            "2026-01-01",
            periods=4,
            freq="D",
        ).date

        return pd.DataFrame(
            {
                "participant_id": [
                    1,
                    1,
                    1,
                    1,
                ],

                "date": dates,

                "individual_change_assessable": [
                    True,
                    False,
                    True,
                    True,
                ],

                "individual_change_signal": [
                    True,
                    False,
                    False,
                    True,
                ],

                "individual_change_confirmed": [
                    True,
                    False,
                    False,
                    True,
                ],

                "individual_change_trigger": [
                    True,
                    False,
                    False,
                    True,
                ],

                "current_change_onset_date": [
                    dates[0],
                    pd.NaT,
                    pd.NaT,
                    dates[3],
                ],

                "current_change_trigger_date": [
                    dates[0],
                    pd.NaT,
                    pd.NaT,
                    dates[3],
                ],
            }
        )

    def test_all_methods_use_same_common_rows(
        self,
    ) -> None:

        result = build_method_comparison(
            self._modeling(),
            self._individualized(),
        )

        horizon_7 = result.loc[
            result[
                "outcome_horizon_days"
            ].eq(
                7
            )
        ]

        self.assertEqual(
            len(
                horizon_7
            ),
            4,
        )

        self.assertTrue(
            horizon_7[
                "rows"
            ].eq(
                3
            ).all()
        )

        self.assertTrue(
            horizon_7[
                "participants"
            ].eq(
                1
            ).all()
        )

    def test_fixed_rule_metrics_on_common_rows(
        self,
    ) -> None:

        result = build_method_comparison(
            self._modeling(),
            self._individualized(),
        )

        row = result.loc[
            (
                result[
                    "method_id"
                ]
                == "fixed_inactivity_7d"
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

        self.assertEqual(
            row[
                "true_positive"
            ],
            1,
        )

        self.assertEqual(
            row[
                "false_positive"
            ],
            0,
        )

        self.assertEqual(
            row[
                "true_negative"
            ],
            1,
        )

        self.assertEqual(
            row[
                "false_negative"
            ],
            1,
        )

    def test_individualized_metrics_on_common_rows(
        self,
    ) -> None:

        result = build_method_comparison(
            self._modeling(),
            self._individualized(),
        )

        row = result.loc[
            (
                result[
                    "method_id"
                ]
                == "individualized_change"
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

        self.assertEqual(
            row[
                "true_positive"
            ],
            1,
        )

        self.assertEqual(
            row[
                "false_positive"
            ],
            1,
        )

        self.assertEqual(
            row[
                "true_negative"
            ],
            0,
        )

        self.assertEqual(
            row[
                "false_negative"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()