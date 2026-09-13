import json
import unittest

import pandas as pd

from src.trajectory.event_normalization import (
    _deduplicate_events,
    _normalize_activities,
)


class TestEventNormalization(
    unittest.TestCase
):

    def test_same_activity_in_two_waves_becomes_one_event(
        self,
    ):
        raw = pd.DataFrame(
            [
                {
                    "aid": 1001,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q1End",
                    "wave": 1,
                },
                {
                    "aid": 1001,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q2End",
                    "wave": 2,
                },
            ]
        )

        normalized_raw = (
            _normalize_activities(
                raw
            )
        )

        events = _deduplicate_events(
            normalized_raw
        )

        self.assertEqual(
            len(events),
            1,
        )

        event = events.iloc[0]

        self.assertEqual(
            event["event_id"],
            "activity:1001",
        )

        self.assertEqual(
            int(
                event[
                    "source_row_count"
                ]
            ),
            2,
        )

        self.assertFalse(
            bool(
                event[
                    "dedup_conflict"
                ]
            )
        )

        self.assertEqual(
            json.loads(
                event[
                    "source_wave_references"
                ]
            ),
            [
                "q1End",
                "q2End",
            ],
        )

        self.assertEqual(
            json.loads(
                event[
                    "source_waves"
                ]
            ),
            [
                1,
                2,
            ],
        )

    def test_different_activity_ids_are_not_deduplicated(
        self,
    ):
        raw = pd.DataFrame(
            [
                {
                    "aid": 1001,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q1End",
                    "wave": 1,
                },
                {
                    "aid": 1002,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q2End",
                    "wave": 2,
                },
            ]
        )

        events = _deduplicate_events(
            _normalize_activities(
                raw
            )
        )

        self.assertEqual(
            len(events),
            2,
        )

    def test_duplicate_with_conflicting_points_is_flagged(
        self,
    ):
        raw = pd.DataFrame(
            [
                {
                    "aid": 1001,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q1End",
                    "wave": 1,
                },
                {
                    "aid": 1001,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 20,
                    "isDeleted": False,
                    "waveReference": "q2End",
                    "wave": 2,
                },
            ]
        )

        events = _deduplicate_events(
            _normalize_activities(
                raw
            )
        )

        self.assertEqual(
            len(events),
            1,
        )

        self.assertTrue(
            bool(
                events.iloc[0][
                    "dedup_conflict"
                ]
            )
        )

    def test_duplicate_with_conflicting_participant_is_flagged(
        self,
    ):
        raw = pd.DataFrame(
            [
                {
                    "aid": 1001,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q1End",
                    "wave": 1,
                },
                {
                    "aid": 1001,
                    "pid": 99,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q2End",
                    "wave": 2,
                },
            ]
        )

        events = _deduplicate_events(
            _normalize_activities(
                raw
            )
        )

        self.assertEqual(
            len(events),
            1,
        )

        self.assertTrue(
            bool(
                events.iloc[0][
                    "dedup_conflict"
                ]
            )
        )

    def test_row_without_activity_id_is_not_cross_wave_deduplicated(
        self,
    ):
        raw = pd.DataFrame(
            [
                {
                    "aid": pd.NA,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q1End",
                    "wave": 1,
                },
                {
                    "aid": pd.NA,
                    "pid": 42,
                    "createdAt": (
                        "2026-01-10T10:00:00Z"
                    ),
                    "type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "averageNumberOfPoints": 10,
                    "isDeleted": False,
                    "waveReference": "q2End",
                    "wave": 2,
                },
            ]
        )

        events = _deduplicate_events(
            _normalize_activities(
                raw
            )
        )

        # Without a stable GameBus activity ID,
        # the two rows cannot safely be assumed
        # to represent the same underlying event.
        self.assertEqual(
            len(events),
            2,
        )

        self.assertTrue(
            all(
                str(event_id).startswith(
                    "activity:row:"
                )
                for event_id
                in events[
                    "event_id"
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()