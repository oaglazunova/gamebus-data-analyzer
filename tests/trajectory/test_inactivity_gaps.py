import unittest
from unittest.mock import patch

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)

from src.trajectory.inactivity_gap_analysis import (
    build_inactivity_gaps,
)


def _explicit_event(
    event_id: str,
    occurred_at: str,
) -> dict:
    return {
        "event_id": event_id,
        "participant_id": 1,
        "occurred_at": occurred_at,
        "event_kind": "activity",
        "event_type": "GENERAL_ACTIVITY",
        "provider": "GameBus Studio",
        "is_deleted": False,
        "points": 10,
    }


def _garmin_event(
    event_id: str,
    occurred_at: str,
) -> dict:
    return {
        "event_id": event_id,
        "participant_id": 1,
        "occurred_at": occurred_at,
        "event_kind": "activity",
        "event_type": "DAY_AGGREGATE",
        "provider": "Garmin",
        "is_deleted": False,
        "points": 0,
    }


class TestInactivityGaps(
    unittest.TestCase
):

    def setUp(self):
        self.config = TrajectoryAuditConfig()
        self.config.episode_gap_days = 14

    def _build(
        self,
        events: pd.DataFrame,
        cutoff: str,
    ) -> pd.DataFrame:

        window = {
            "analysis_cutoff": pd.Timestamp(
                cutoff,
                tz="UTC",
            )
        }

        with patch(
            "src.trajectory.inactivity_gap_analysis."
            "build_observation_window",
            return_value=window,
        ):
            return build_inactivity_gaps(
                self.config,
                events,
            )

    def test_closed_14_day_gap_is_reengaged(
        self,
    ):
        events = pd.DataFrame(
            [
                _explicit_event(
                    "E1",
                    "2026-01-01T10:00:00Z",
                ),
                _explicit_event(
                    "E2",
                    "2026-01-16T10:00:00Z",
                ),
            ]
        )

        gaps = self._build(
            events,
            "2026-01-16",
        )

        self.assertEqual(
            len(gaps),
            1,
        )

        gap = gaps.iloc[0]

        self.assertEqual(
            gap["gap_type"],
            "between_engagements",
        )

        self.assertEqual(
            int(gap["inactive_days"]),
            14,
        )

        self.assertTrue(
            bool(gap["reengaged"])
        )

        self.assertFalse(
            bool(gap["right_censored"])
        )

        self.assertTrue(
            bool(gap["reached_7d"])
        )

        self.assertTrue(
            bool(gap["reached_14d"])
        )

        self.assertFalse(
            bool(gap["reached_21d"])
        )

        self.assertTrue(
            bool(
                gap[
                    "exceeds_episode_boundary"
                ]
            )
        )

    def test_trailing_gap_is_right_censored(
        self,
    ):
        events = pd.DataFrame(
            [
                _explicit_event(
                    "E1",
                    "2026-01-01T10:00:00Z",
                ),
            ]
        )

        gaps = self._build(
            events,
            "2026-01-21",
        )

        self.assertEqual(
            len(gaps),
            1,
        )

        gap = gaps.iloc[0]

        self.assertEqual(
            gap["gap_type"],
            "right_censored",
        )

        self.assertEqual(
            gap["gap_status"],
            "ongoing_at_cutoff",
        )

        self.assertEqual(
            int(gap["inactive_days"]),
            20,
        )

        self.assertFalse(
            bool(gap["reengaged"])
        )

        self.assertTrue(
            bool(gap["right_censored"])
        )

        self.assertTrue(
            pd.isna(
                gap[
                    "next_engagement_date"
                ]
            )
        )

        self.assertTrue(
            bool(gap["reached_7d"])
        )

        self.assertTrue(
            bool(gap["reached_14d"])
        )

        self.assertFalse(
            bool(gap["reached_21d"])
        )

    def test_no_trailing_gap_when_engagement_occurs_at_cutoff(
        self,
    ):
        events = pd.DataFrame(
            [
                _explicit_event(
                    "E1",
                    "2026-01-21T10:00:00Z",
                ),
            ]
        )

        gaps = self._build(
            events,
            "2026-01-21",
        )

        self.assertTrue(
            gaps.empty
        )

    def test_passive_garmin_event_does_not_end_inactivity_gap(
        self,
    ):
        events = pd.DataFrame(
            [
                _explicit_event(
                    "E1",
                    "2026-01-01T10:00:00Z",
                ),
                _garmin_event(
                    "G1",
                    "2026-01-10T10:00:00Z",
                ),
            ]
        )

        gaps = self._build(
            events,
            "2026-01-20",
        )

        self.assertEqual(
            len(gaps),
            1,
        )

        gap = gaps.iloc[0]

        # Jan 2 through Jan 20:
        # 19 days without explicit engagement.
        self.assertEqual(
            int(gap["inactive_days"]),
            19,
        )

        self.assertEqual(
            gap["gap_type"],
            "right_censored",
        )

        self.assertFalse(
            bool(gap["reengaged"])
        )

        self.assertTrue(
            bool(gap["right_censored"])
        )

    def test_passive_only_participant_has_no_post_engagement_gap(
        self,
    ):
        events = pd.DataFrame(
            [
                _garmin_event(
                    "G1",
                    "2026-01-10T10:00:00Z",
                ),
            ]
        )

        gaps = self._build(
            events,
            "2026-01-20",
        )

        # A participant with no explicit engagement
        # has not entered a post-engagement gap.
        self.assertTrue(
            gaps.empty
        )


if __name__ == "__main__":
    unittest.main()