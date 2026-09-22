from __future__ import annotations

import unittest

import pandas as pd

from src.trajectory.participant_state_daily import (
    _add_engagement_state,
)


class TestParticipantStateDaily(
    unittest.TestCase
):

    def test_maintenance_starts_at_first_explicit_engagement(
        self,
    ) -> None:
        state = pd.DataFrame(
            {
                "participant_id": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                "date": pd.to_datetime(
                    [
                        "2026-09-01",
                        "2026-09-02",
                        "2026-09-03",
                        "2026-09-04",
                        "2026-09-05",
                        "2026-09-01",
                        "2026-09-02",
                        "2026-09-03",
                        "2026-09-04",
                        "2026-09-05",
                    ],
                    utc=True,
                ),
                "explicit_events_today": [
                    0,
                    0,
                    2,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        )

        result = _add_engagement_state(
            state,
            "available",
        )

        participant_1 = result.loc[
            result["participant_id"] == 1
        ].reset_index(
            drop=True
        )

        self.assertEqual(
            participant_1[
                "analysis_phase"
            ].tolist(),
            [
                (
                    "before_first_observed_"
                    "explicit_engagement"
                ),
                (
                    "before_first_observed_"
                    "explicit_engagement"
                ),
                "maintenance_reengagement",
                "maintenance_reengagement",
                "maintenance_reengagement",
            ],
        )

        self.assertEqual(
            participant_1[
                "maintenance_reengagement_eligible"
            ].tolist(),
            [
                False,
                False,
                True,
                True,
                True,
            ],
        )

        self.assertTrue(
            pd.isna(
                participant_1.loc[
                    0,
                    (
                        "first_observed_"
                        "explicit_engagement_date"
                    ),
                ]
            )
        )

        self.assertEqual(
            participant_1.loc[
                2,
                (
                    "first_observed_"
                    "explicit_engagement_date"
                ),
            ],
            pd.Timestamp(
                "2026-09-03",
                tz="UTC",
            ),
        )

        self.assertEqual(
            participant_1.loc[
                4,
                (
                    "first_observed_"
                    "explicit_engagement_date"
                ),
            ],
            pd.Timestamp(
                "2026-09-03",
                tz="UTC",
            ),
        )

        participant_2 = result.loc[
            result["participant_id"] == 2
        ]

        self.assertFalse(
            participant_2[
                "maintenance_reengagement_eligible"
            ].any()
        )

        self.assertTrue(
            participant_2[
                (
                    "first_observed_"
                    "explicit_engagement_date"
                )
            ].isna().all()
        )


if __name__ == "__main__":
    unittest.main()