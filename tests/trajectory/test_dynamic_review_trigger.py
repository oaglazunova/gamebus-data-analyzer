from __future__ import annotations

import unittest

import pandas as pd

from src.trajectory.dynamic_review_trigger import (
    DynamicReviewTriggerConfig,
    apply_dynamic_review_trigger,
    calibrate_dynamic_review_trigger,
)


class TestDynamicReviewTrigger(
    unittest.TestCase
):

    def test_calibration_selects_development_threshold(
        self,
    ) -> None:

        frame = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    2,
                    3,
                    4,
                ],

                (
                    "predicted_next_day_"
                    "reengagement_probability"
                ): [
                    0.90,
                    0.80,
                    0.30,
                    0.20,
                ],

                "outcome_available_14d": [
                    True,
                    True,
                    True,
                    True,
                ],

                "continued_inactivity_14d": (
                    pd.Series(
                        [
                            False,
                            False,
                            True,
                            True,
                        ],
                        dtype="boolean",
                    )
                ),
            }
        )

        config, calibration = (
            calibrate_dynamic_review_trigger(
                frame
            )
        )

        self.assertAlmostEqual(
            config.inactivity_risk_threshold,
            0.70,
        )

        selected = calibration.loc[
            calibration[
                "selected"
            ]
        ]

        self.assertEqual(
            len(
                selected
            ),
            1,
        )

        self.assertEqual(
            selected.iloc[
                0
            ][
                "balanced_accuracy"
            ],
            1.0,
        )

    def test_censored_rows_are_excluded_from_calibration(
        self,
    ) -> None:

        frame = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],

                (
                    "predicted_next_day_"
                    "reengagement_probability"
                ): [
                    0.90,
                    0.80,
                    0.30,
                    0.20,
                    0.01,
                ],

                "outcome_available_14d": [
                    True,
                    True,
                    True,
                    True,
                    False,
                ],

                "continued_inactivity_14d": (
                    pd.Series(
                        [
                            False,
                            False,
                            True,
                            True,
                            pd.NA,
                        ],
                        dtype="boolean",
                    )
                ),
            }
        )

        _, calibration = (
            calibrate_dynamic_review_trigger(
                frame
            )
        )

        self.assertTrue(
            calibration[
                "rows"
            ].eq(
                4
            ).all()
        )

    def test_persistence_creates_separate_change_and_trigger_dates(
        self,
    ) -> None:

        dates = pd.date_range(
            "2026-01-01",
            periods=7,
            freq="D",
        )

        frame = pd.DataFrame(
            {
                "campaign_id": [
                    "A"
                    for _ in dates
                ],

                "participant_id": [
                    1
                    for _ in dates
                ],

                "date": dates,

                (
                    "predicted_next_day_"
                    "reengagement_probability"
                ): [
                    0.20,
                    0.20,
                    0.20,
                    0.90,
                    0.20,
                    0.20,
                    0.20,
                ],
            }
        )

        config = (
            DynamicReviewTriggerConfig(
                inactivity_risk_threshold=(
                    0.70
                ),
                persistence_days=3,
            )
        )

        result = (
            apply_dynamic_review_trigger(
                frame,
                config,
            )
        )

        self.assertEqual(
            result.loc[
                0,
                "dynamic_change_onset_date",
            ],
            pd.Timestamp(
                "2026-01-01"
            ).date(),
        )

        self.assertFalse(
            result.loc[
                0,
                "dynamic_review_trigger",
            ]
        )

        self.assertTrue(
            result.loc[
                2,
                "dynamic_review_trigger",
            ]
        )

        self.assertEqual(
            result.loc[
                2,
                "current_dynamic_trigger_date",
            ],
            pd.Timestamp(
                "2026-01-03"
            ).date(),
        )

        # High predicted re-engagement probability
        # resets the signal.
        self.assertEqual(
            result.loc[
                3,
                "dynamic_signal_streak_days",
            ],
            0,
        )

        # A second episode can subsequently trigger.
        self.assertTrue(
            result.loc[
                6,
                "dynamic_review_trigger",
            ]
        )

        self.assertEqual(
            result.loc[
                6,
                "dynamic_change_onset_date",
            ],
            pd.Timestamp(
                "2026-01-05"
            ).date(),
        )

    def test_missing_calendar_day_breaks_persistence(
        self,
    ) -> None:

        frame = pd.DataFrame(
            {
                "campaign_id": [
                    "A",
                    "A",
                    "A",
                ],

                "participant_id": [
                    1,
                    1,
                    1,
                ],

                "date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-04",
                    ]
                ),

                (
                    "predicted_next_day_"
                    "reengagement_probability"
                ): [
                    0.10,
                    0.10,
                    0.10,
                ],
            }
        )

        config = (
            DynamicReviewTriggerConfig(
                inactivity_risk_threshold=(
                    0.70
                ),
                persistence_days=3,
            )
        )

        result = (
            apply_dynamic_review_trigger(
                frame,
                config,
            )
        )

        self.assertEqual(
            result[
                "dynamic_signal_streak_days"
            ].tolist(),
            [
                1,
                2,
                1,
            ],
        )

        self.assertFalse(
            result[
                "dynamic_review_trigger"
            ].any()
        )


if __name__ == "__main__":
    unittest.main()