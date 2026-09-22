from __future__ import annotations

import unittest

import pandas as pd

from src.trajectory.individualized_change_detector import (
    build_individualized_change_detection,
)


class TestIndividualizedChangeDetector(
    unittest.TestCase
):

    def _features(
        self,
    ) -> pd.DataFrame:

        dates = pd.date_range(
            "2026-01-01",
            periods=45,
            freq="D",
            tz="UTC",
        )

        explicit = [
            0
            for _ in dates
        ]

        # Personal reference period:
        # active on every second day during
        # the first 28 days.
        for index in range(
            0,
            28,
            2,
        ):
            explicit[
                index
            ] = 1

        # From day 29 onwards there is no explicit
        # engagement, creating a clear decline from
        # the participant's own earlier pattern.

        return pd.DataFrame(
            {
                "participant_id": [
                    1
                    for _ in dates
                ],
                "date": dates,
                (
                    "maintenance_reengagement_"
                    "eligible"
                ): [
                    True
                    for _ in dates
                ],
                "explicit_events_today": (
                    explicit
                ),
            }
        )

    def test_change_uses_non_overlapping_personal_baseline(
        self,
    ) -> None:

        result = (
            build_individualized_change_detection(
                self._features()
            )
        )

        # Day 35 is the first day with:
        #
        #   28 complete reference days
        #   followed by
        #   7 complete recent days.
        day_35 = result.iloc[
            34
        ]

        self.assertTrue(
            day_35[
                "individual_change_assessable"
            ]
        )

        self.assertEqual(
            day_35[
                "reference_active_days_28d"
            ],
            14,
        )

        self.assertEqual(
            day_35[
                "recent_active_days_7d"
            ],
            0,
        )

        self.assertEqual(
            day_35[
                "recent_active_day_rate_7d"
            ],
            0.0,
        )

        self.assertEqual(
            day_35[
                "reference_active_day_rate_28d"
            ],
            0.5,
        )

        self.assertEqual(
            day_35[
                "recent_to_reference_rate_ratio"
            ],
            0.0,
        )

        self.assertTrue(
            day_35[
                "individual_change_signal"
            ]
        )

    def test_change_onset_and_trigger_are_separate(
        self,
    ) -> None:

        result = (
            build_individualized_change_detection(
                self._features()
            )
        )

        # First assessable decline:
        # day 35 = 04-02-2026.
        day_35 = result.iloc[
            34
        ]

        # Persistence reaches three days:
        # day 37 = 06-02-2026.
        day_37 = result.iloc[
            36
        ]

        self.assertEqual(
            day_35[
                "current_change_onset_date"
            ],
            pd.Timestamp(
                "2026-02-04"
            ).date(),
        )

        self.assertFalse(
            day_35[
                "individual_change_confirmed"
            ]
        )

        self.assertFalse(
            day_35[
                "individual_change_trigger"
            ]
        )

        self.assertEqual(
            day_37[
                "signal_streak_days"
            ],
            3,
        )

        self.assertTrue(
            day_37[
                "individual_change_confirmed"
            ]
        )

        self.assertTrue(
            day_37[
                "individual_change_trigger"
            ]
        )

        self.assertEqual(
            day_37[
                "current_change_onset_date"
            ],
            pd.Timestamp(
                "2026-02-04"
            ).date(),
        )

        self.assertEqual(
            day_37[
                "current_change_trigger_date"
            ],
            pd.Timestamp(
                "2026-02-06"
            ).date(),
        )

    def test_future_events_do_not_change_earlier_detection(
        self,
    ) -> None:

        features = self._features()

        first = (
            build_individualized_change_detection(
                features
            )
        )

        earlier = first.iloc[
            36
        ].copy()

        # Modify an event that occurs after the
        # landmark being compared.
        features.loc[
            features["date"]
            == pd.Timestamp(
                "2026-02-12",
                tz="UTC",
            ),
            "explicit_events_today",
        ] = 1000

        second = (
            build_individualized_change_detection(
                features
            )
        )

        earlier_after = second.iloc[
            36
        ]

        for column in (
            "recent_active_days_7d",
            "reference_active_days_28d",
            "recent_to_reference_rate_ratio",
            "individual_change_signal",
            "signal_streak_days",
            "individual_change_trigger",
        ):
            self.assertEqual(
                earlier[
                    column
                ],
                earlier_after[
                    column
                ],
            )

    def test_sparse_reference_is_not_interpreted_as_change(
        self,
    ) -> None:

        features = self._features()

        # Leave only one active day in the
        # participant's personal reference period.
        features.loc[
            features.index[
                :28
            ],
            "explicit_events_today",
        ] = 0

        features.loc[
            0,
            "explicit_events_today",
        ] = 1

        result = (
            build_individualized_change_detection(
                features
            )
        )

        day_35 = result.iloc[
            34
        ]

        self.assertFalse(
            day_35[
                "individual_change_assessable"
            ]
        )

        self.assertFalse(
            day_35[
                "individual_change_signal"
            ]
        )


if __name__ == "__main__":
    unittest.main()