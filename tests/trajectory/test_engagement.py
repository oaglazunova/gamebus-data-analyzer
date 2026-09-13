import unittest

import pandas as pd

from src.trajectory.engagement import (
    behavioral_sensor_mask,
    deleted_activity_mask,
    explicit_engagement_mask,
)


class TestEngagementClassification(
    unittest.TestCase
):

    def test_gamebus_activity_is_explicit_engagement(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "activity",
                    "event_type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "is_deleted": False,
                }
            ]
        )

        result = explicit_engagement_mask(
            events
        )

        self.assertTrue(
            bool(
                result.iloc[0]
            )
        )

    def test_nutrida_activity_is_explicit_engagement(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "activity",
                    "event_type": "VISIT_APP_PAGE",
                    "provider": "Nutrida",
                    "is_deleted": False,
                }
            ]
        )

        result = explicit_engagement_mask(
            events
        )

        self.assertTrue(
            bool(
                result.iloc[0]
            )
        )

    def test_garmin_day_aggregate_is_not_explicit_engagement(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "activity",
                    "event_type": "DAY_AGGREGATE",
                    "provider": "Garmin",
                    "is_deleted": False,
                }
            ]
        )

        explicit = explicit_engagement_mask(
            events
        )

        passive = behavioral_sensor_mask(
            events
        )

        self.assertFalse(
            bool(
                explicit.iloc[0]
            )
        )

        self.assertTrue(
            bool(
                passive.iloc[0]
            )
        )

    def test_navigation_is_not_explicit_engagement(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "navigation",
                    "event_type": "PAGE_VIEW",
                    "provider": "GameBus",
                    "is_deleted": False,
                }
            ]
        )

        result = explicit_engagement_mask(
            events
        )

        self.assertFalse(
            bool(
                result.iloc[0]
            )
        )

    def test_sensor_event_is_not_explicit_engagement(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "sensor",
                    "event_type": "SENSOR_READING",
                    "provider": "Some Sensor",
                    "is_deleted": False,
                }
            ]
        )

        explicit = explicit_engagement_mask(
            events
        )

        passive = behavioral_sensor_mask(
            events
        )

        self.assertFalse(
            bool(
                explicit.iloc[0]
            )
        )

        self.assertTrue(
            bool(
                passive.iloc[0]
            )
        )

    def test_deleted_activity_is_not_explicit_engagement(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "activity",
                    "event_type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "is_deleted": True,
                }
            ]
        )

        explicit = explicit_engagement_mask(
            events
        )

        deleted = deleted_activity_mask(
            events
        )

        self.assertFalse(
            bool(
                explicit.iloc[0]
            )
        )

        self.assertTrue(
            bool(
                deleted.iloc[0]
            )
        )

    def test_deleted_text_true_is_recognized(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "activity",
                    "event_type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "is_deleted": "true",
                }
            ]
        )

        result = explicit_engagement_mask(
            events
        )

        self.assertFalse(
            bool(
                result.iloc[0]
            )
        )

    def test_non_deleted_text_false_remains_explicit(
        self,
    ):
        events = pd.DataFrame(
            [
                {
                    "event_kind": "activity",
                    "event_type": "GENERAL_ACTIVITY",
                    "provider": "GameBus Studio",
                    "is_deleted": "false",
                }
            ]
        )

        result = explicit_engagement_mask(
            events
        )

        self.assertTrue(
            bool(
                result.iloc[0]
            )
        )


if __name__ == "__main__":
    unittest.main()