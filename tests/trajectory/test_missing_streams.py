import unittest
from unittest.mock import patch

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)

from src.trajectory.participant_state_daily import (
    build_participant_state_daily,
)


def _window():
    return {
        "effective_start": pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        ),
        "analysis_cutoff": pd.Timestamp(
            "2026-01-02",
            tz="UTC",
        ),
    }


def _cohort():
    return {
        "source": "campaign_export",
        "participant_ids": [1],
    }


def _navigation_event():
    return pd.DataFrame(
        [
            {
                "event_id": "N1",
                "participant_id": 1,
                "occurred_at": (
                    "2026-01-01T10:00:00Z"
                ),
                "event_kind": "navigation",
                "event_type": "PAGE_VIEW",
                "provider": "GameBus",
                "is_deleted": False,
                "points": 0,
            }
        ]
    )


def _gamebus_activity():
    return pd.DataFrame(
        [
            {
                "event_id": "E1",
                "participant_id": 1,
                "occurred_at": (
                    "2026-01-01T10:00:00Z"
                ),
                "event_kind": "activity",
                "event_type": "GENERAL_ACTIVITY",
                "provider": "GameBus Studio",
                "is_deleted": False,
                "points": 10,
            }
        ]
    )


class TestMissingStreamSemantics(
    unittest.TestCase
):

    def setUp(self):
        self.config = (
            TrajectoryAuditConfig()
        )

    def _build(
        self,
        events,
        stream_info,
    ):
        with (
            patch(
                "src.trajectory."
                "participant_state_daily."
                "build_observation_window",
                return_value=_window(),
            ),
            patch(
                "src.trajectory."
                "participant_state_daily."
                "_export_cohort",
                return_value=_cohort(),
            ),
            patch(
                "src.trajectory."
                "participant_state_daily."
                "load_campaign_export",
                return_value=(
                    {},
                    stream_info,
                ),
            ),
        ):
            return (
                build_participant_state_daily(
                    self.config,
                    events,
                )
            )

    def test_unavailable_activity_is_na_not_zero(
        self,
    ):
        stream_info = {
            "activities": {
                "status": "unavailable",
            },
            "navigation": {
                "status": "available",
            },
            "notification_events": {
                "status": "unavailable",
            },
            "sensor_events": {
                "status": "unavailable",
            },
        }

        state = self._build(
            _navigation_event(),
            stream_info,
        )

        self.assertTrue(
            state[
                "explicit_events_today"
            ].isna().all()
        )

        self.assertTrue(
            state[
                "points_today"
            ].isna().all()
        )

        self.assertTrue(
            (
                state[
                    "engagement_state"
                ]
                == "unavailable"
            ).all()
        )

        self.assertTrue(
            (
                state[
                    "activity_stream_state"
                ]
                == "unavailable"
            ).all()
        )

    def test_unavailable_navigation_is_na_not_zero(
        self,
    ):
        stream_info = {
            "activities": {
                "status": "available",
            },
            "navigation": {
                "status": "unavailable",
            },
            "notification_events": {
                "status": "available_empty",
            },
            "sensor_events": {
                "status": "available_empty",
            },
        }

        state = self._build(
            _gamebus_activity(),
            stream_info,
        )

        self.assertTrue(
            state[
                "navigation_events_today"
            ].isna().all()
        )

    def test_available_empty_stream_is_zero(
        self,
    ):
        stream_info = {
            "activities": {
                "status": "available",
            },
            "navigation": {
                "status": "available_empty",
            },
            "notification_events": {
                "status": "available_empty",
            },
            "sensor_events": {
                "status": "available_empty",
            },
        }

        state = self._build(
            _gamebus_activity(),
            stream_info,
        )

        self.assertTrue(
            (
                state[
                    "navigation_events_today"
                ]
                == 0
            ).all()
        )

        self.assertTrue(
            (
                state[
                    "notification_events_today"
                ]
                == 0
            ).all()
        )

        self.assertTrue(
            (
                state[
                    "sensor_events_today"
                ]
                == 0
            ).all()
        )

    def test_absent_optional_provider_is_na_not_zero(
        self,
    ):
        """
        Activity data exist, but there is no Garmin
        provider in the observed activity stream.

        We therefore do not infer zero Garmin behavior;
        Garmin is treated as unavailable.
        """

        stream_info = {
            "activities": {
                "status": "available",
            },
            "navigation": {
                "status": "available_empty",
            },
            "notification_events": {
                "status": "available_empty",
            },
            "sensor_events": {
                "status": "available_empty",
            },
        }

        state = self._build(
            _gamebus_activity(),
            stream_info,
        )

        self.assertTrue(
            state[
                "garmin_events_today"
            ].isna().all()
        )

        self.assertTrue(
            (
                state[
                    "garmin_stream_state"
                ]
                == "unavailable"
            ).all()
        )

    def test_available_activity_zero_on_inactive_day(
        self,
    ):
        """
        When activity data are available, zero explicit
        activity on a day is a meaningful observed zero.
        """

        stream_info = {
            "activities": {
                "status": "available",
            },
            "navigation": {
                "status": "available_empty",
            },
            "notification_events": {
                "status": "available_empty",
            },
            "sensor_events": {
                "status": "available_empty",
            },
        }

        state = self._build(
            _gamebus_activity(),
            stream_info,
        )

        day_two = state.loc[
            state["date"]
            == pd.Timestamp(
                "2026-01-02"
            ).date()
        ].iloc[0]

        self.assertEqual(
            int(
                day_two[
                    "explicit_events_today"
                ]
            ),
            0,
        )

        self.assertEqual(
            day_two[
                "activity_stream_state"
            ],
            "available",
        )


if __name__ == "__main__":
    unittest.main()