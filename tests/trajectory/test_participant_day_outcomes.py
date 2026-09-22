from __future__ import annotations

import unittest

import pandas as pd

from src.trajectory.participant_day_outcomes import (
    build_participant_day_outcomes,
)


class TestParticipantDayOutcomes(
    unittest.TestCase
):

    def _features(
        self,
    ) -> pd.DataFrame:

        dates = pd.date_range(
            "2026-01-01",
            periods=10,
            freq="D",
            tz="UTC",
        )

        # Explicit engagement on:
        # day 1, day 4, day 10.
        explicit = [
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
        ]

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

    def test_next_engagement_is_strictly_future(
        self,
    ) -> None:

        result = (
            build_participant_day_outcomes(
                self._features()
            )
        )

        day_1 = result.iloc[
            0
        ]

        self.assertEqual(
            day_1[
                "next_explicit_engagement_date"
            ],
            pd.Timestamp(
                "2026-01-04"
            ).date(),
        )

        self.assertEqual(
            day_1[
                (
                    "time_to_next_explicit_"
                    "engagement_days"
                )
            ],
            3,
        )

        self.assertTrue(
            day_1[
                (
                    "next_explicit_engagement_"
                    "observed"
                )
            ]
        )

        self.assertFalse(
            day_1[
                "right_censored"
            ]
        )

    def test_next_day_hazard_target(
        self,
    ) -> None:

        result = (
            build_participant_day_outcomes(
                self._features()
            )
        )

        # Day 3 is immediately before the next
        # explicit engagement on day 4.
        day_3 = result.iloc[
            2
        ]

        self.assertTrue(
            day_3[
                "explicit_engagement_next_day"
            ]
        )

        # Day 2 still has observed follow-up,
        # but no engagement on day 3.
        day_2 = result.iloc[
            1
        ]

        self.assertFalse(
            day_2[
                "explicit_engagement_next_day"
            ]
        )

    def test_tail_is_right_censored(
        self,
    ) -> None:

        features = self._features()

        # Remove the final engagement so the tail
        # contains no subsequent event.
        features.loc[
            features["date"]
            == pd.Timestamp(
                "2026-01-10",
                tz="UTC",
            ),
            "explicit_events_today",
        ] = 0

        result = (
            build_participant_day_outcomes(
                features
            )
        )

        day_8 = result.iloc[
            7
        ]

        self.assertTrue(
            day_8[
                "right_censored"
            ]
        )

        self.assertFalse(
            day_8[
                (
                    "next_explicit_engagement_"
                    "observed"
                )
            ]
        )

        self.assertTrue(
            pd.isna(
                day_8[
                    (
                        "time_to_next_explicit_"
                        "engagement_days"
                    )
                ]
            )
        )

        self.assertEqual(
            day_8[
                "time_to_event_or_censor_days"
            ],
            2,
        )

        self.assertEqual(
            day_8[
                "event_or_censor_date"
            ],
            pd.Timestamp(
                "2026-01-10"
            ).date(),
        )

    def test_final_day_has_no_next_day_target(
        self,
    ) -> None:

        result = (
            build_participant_day_outcomes(
                self._features()
            )
        )

        final_day = result.iloc[
            -1
        ]

        self.assertFalse(
            final_day[
                "next_day_outcome_available"
            ]
        )

        self.assertTrue(
            pd.isna(
                final_day[
                    "explicit_engagement_next_day"
                ]
            )
        )

        self.assertTrue(
            final_day[
                "right_censored"
            ]
        )

        self.assertEqual(
            final_day[
                "time_to_event_or_censor_days"
            ],
            0,
        )

    def test_pre_engagement_rows_are_excluded(
        self,
    ) -> None:

        features = self._features()

        features.loc[
            features.index[
                :2
            ],
            (
                "maintenance_reengagement_"
                "eligible"
            ),
        ] = False

        result = (
            build_participant_day_outcomes(
                features
            )
        )

        self.assertEqual(
            result.iloc[
                0
            ]["date"],
            pd.Timestamp(
                "2026-01-03"
            ).date(),
        )

    def test_fixed_horizon_engagement_outcomes(
        self,
    ) -> None:

        result = (
            build_participant_day_outcomes(
                self._features()
            )
        )

        # Day 1 -> next engagement on day 4:
        # waiting time = 3 days.
        day_1 = result.iloc[
            0
        ]

        for horizon in (
            7,
            14,
            21,
            30,
        ):
            self.assertTrue(
                day_1[
                    f"outcome_available_{horizon}d"
                ]
            )

            self.assertTrue(
                day_1[
                    (
                        "explicit_engagement_"
                        f"within_{horizon}d"
                    )
                ]
            )

            self.assertFalse(
                day_1[
                    (
                        "continued_inactivity_"
                        f"{horizon}d"
                    )
                ]
            )
    def test_fully_observed_horizon_can_be_negative(
        self,
    ) -> None:

        features = self._features()

        # Remove all engagements after day 1.
        features.loc[
            features.index[
                1:
            ],
            "explicit_events_today",
        ] = 0

        result = (
            build_participant_day_outcomes(
                features
            )
        )

        day_1 = result.iloc[
            0
        ]

        # Nine days of follow-up remain.
        self.assertTrue(
            day_1[
                "outcome_available_7d"
            ]
        )

        self.assertFalse(
            day_1[
                "explicit_engagement_within_7d"
            ]
        )

        self.assertTrue(
            day_1[
                "continued_inactivity_7d"
            ]
        )

    def test_incomplete_horizon_is_not_labelled_negative(
        self,
    ) -> None:

        features = self._features()

        features.loc[
            features.index[
                1:
            ],
            "explicit_events_today",
        ] = 0

        result = (
            build_participant_day_outcomes(
                features
            )
        )

        # Day 8 has only two days of follow-up.
        day_8 = result.iloc[
            7
        ]

        self.assertFalse(
            day_8[
                "outcome_available_7d"
            ]
        )

        self.assertTrue(
            pd.isna(
                day_8[
                    "explicit_engagement_within_7d"
                ]
            )
        )

        self.assertTrue(
            pd.isna(
                day_8[
                    "continued_inactivity_7d"
                ]
            )
        )

    def test_event_exactly_on_horizon_counts_within_horizon(
        self,
    ) -> None:

        dates = pd.date_range(
            "2026-01-01",
            periods=8,
            freq="D",
            tz="UTC",
        )

        features = pd.DataFrame(
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
                "explicit_events_today": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
            }
        )

        result = (
            build_participant_day_outcomes(
                features
            )
        )

        day_1 = result.iloc[
            0
        ]

        self.assertEqual(
            day_1[
                (
                    "time_to_next_explicit_"
                    "engagement_days"
                )
            ],
            7,
        )

        self.assertTrue(
            day_1[
                "explicit_engagement_within_7d"
            ]
        )

        self.assertFalse(
            day_1[
                "continued_inactivity_7d"
            ]
        )

        
if __name__ == "__main__":
    unittest.main()