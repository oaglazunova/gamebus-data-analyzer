import unittest

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)

from src.trajectory.participation_episode_builder import (
    build_participation_episodes,
)


def _event(
    event_id: str,
    participant_id: int,
    occurred_at: str,
) -> dict:
    """
    Build one minimal explicit-engagement event
    for episode tests.
    """

    return {
        "event_id": event_id,
        "participant_id": participant_id,
        "occurred_at": occurred_at,
        "event_kind": "activity",
        "event_type": "GENERAL_ACTIVITY",
        "provider": "GameBus Studio",
        "is_deleted": False,
        "points": 10,
    }


class TestParticipationEpisodes(
    unittest.TestCase
):

    def setUp(
        self,
    ):
        self.config = (
            TrajectoryAuditConfig()
        )

        self.config.episode_gap_days = 14

    def test_exactly_14_day_date_difference_stays_same_episode(
        self,
    ):
        """
        Jan 1 -> Jan 15 has a 14-day date difference.

        There are only 13 complete inactive days
        between the engagement dates, so both events
        remain in the same episode.
        """

        events = pd.DataFrame(
            [
                _event(
                    "E1",
                    1,
                    "2026-01-01T10:00:00Z",
                ),
                _event(
                    "E2",
                    1,
                    "2026-01-15T10:00:00Z",
                ),
            ]
        )

        episodes = (
            build_participation_episodes(
                events,
                self.config,
            )
        )

        self.assertEqual(
            len(episodes),
            1,
        )

        self.assertEqual(
            int(
                episodes.iloc[0][
                    "event_count"
                ]
            ),
            2,
        )

    def test_more_than_14_day_date_difference_starts_new_episode(
        self,
    ):
        """
        Jan 1 -> Jan 16 has a 15-day date difference.

        Jan 2 through Jan 15 are 14 complete inactive
        days, so Jan 16 begins a new episode.
        """

        events = pd.DataFrame(
            [
                _event(
                    "E1",
                    1,
                    "2026-01-01T10:00:00Z",
                ),
                _event(
                    "E2",
                    1,
                    "2026-01-16T10:00:00Z",
                ),
            ]
        )

        episodes = (
            build_participation_episodes(
                events,
                self.config,
            )
        )

        self.assertEqual(
            len(episodes),
            2,
        )

        self.assertEqual(
            list(
                episodes[
                    "episode_number"
                ]
            ),
            [
                1,
                2,
            ],
        )

        self.assertEqual(
            int(
                episodes.iloc[1][
                    "engagement_gap_days_before"
                ]
            ),
            15,
        )

        self.assertEqual(
            int(
                episodes.iloc[1][
                    "inactive_days_before"
                ]
            ),
            14,
        )

    def test_multiple_events_same_day_do_not_create_new_episode(
        self,
    ):
        events = pd.DataFrame(
            [
                _event(
                    "E1",
                    1,
                    "2026-01-01T09:00:00Z",
                ),
                _event(
                    "E2",
                    1,
                    "2026-01-01T18:00:00Z",
                ),
            ]
        )

        episodes = (
            build_participation_episodes(
                events,
                self.config,
            )
        )

        self.assertEqual(
            len(episodes),
            1,
        )

        self.assertEqual(
            int(
                episodes.iloc[0][
                    "active_days"
                ]
            ),
            1,
        )

        self.assertEqual(
            int(
                episodes.iloc[0][
                    "event_count"
                ]
            ),
            2,
        )

    def test_participants_are_segmented_independently(
        self,
    ):
        events = pd.DataFrame(
            [
                _event(
                    "A1",
                    1,
                    "2026-01-01T10:00:00Z",
                ),
                _event(
                    "A2",
                    1,
                    "2026-01-16T10:00:00Z",
                ),
                _event(
                    "B1",
                    2,
                    "2026-01-05T10:00:00Z",
                ),
                _event(
                    "B2",
                    2,
                    "2026-01-06T10:00:00Z",
                ),
            ]
        )

        episodes = (
            build_participation_episodes(
                events,
                self.config,
            )
        )

        counts = (
            episodes.groupby(
                "participant_id"
            )
            .size()
            .to_dict()
        )

        self.assertEqual(
            counts[1],
            2,
        )

        self.assertEqual(
            counts[2],
            1,
        )

    def test_passive_event_does_not_split_or_extend_episode(
        self,
    ):
        events = pd.DataFrame(
            [
                _event(
                    "E1",
                    1,
                    "2026-01-01T10:00:00Z",
                ),

                {
                    "event_id": "G1",
                    "participant_id": 1,
                    "occurred_at": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "event_kind": "activity",
                    "event_type": "DAY_AGGREGATE",
                    "provider": "Garmin",
                    "is_deleted": False,
                    "points": 0,
                },

                _event(
                    "E2",
                    1,
                    "2026-01-16T10:00:00Z",
                ),
            ]
        )

        episodes = (
            build_participation_episodes(
                events,
                self.config,
            )
        )

        # Garmin on Jan 10 must not reset the
        # explicit-engagement clock.
        self.assertEqual(
            len(episodes),
            2,
        )

        self.assertEqual(
            int(
                episodes.iloc[1][
                    "inactive_days_before"
                ]
            ),
            14,
        )


if __name__ == "__main__":
    unittest.main()