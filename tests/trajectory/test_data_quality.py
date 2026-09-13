import unittest

from src.trajectory.data_quality_state import (
    _core_quality_state,
)


class TestCoreQualityState(
    unittest.TestCase
):

    def test_no_observed_events_has_no_observed_trajectory(
        self,
    ):
        result = _core_quality_state(
            activity_stream_state="available",
            cohort_available=True,
            has_any_event=False,
            quality_flags=[],
        )

        self.assertEqual(
            result,
            "no_observed_trajectory",
        )

    def test_observed_events_with_good_data_are_sufficient(
        self,
    ):
        result = _core_quality_state(
            activity_stream_state="available",
            cohort_available=True,
            has_any_event=True,
            quality_flags=[],
        )

        self.assertEqual(
            result,
            "sufficient_for_core_trajectory",
        )

    def test_observed_events_with_quality_flags_have_cautions(
        self,
    ):
        result = _core_quality_state(
            activity_stream_state="available",
            cohort_available=True,
            has_any_event=True,
            quality_flags=[
                "example_quality_problem"
            ],
        )

        self.assertEqual(
            result,
            "core_trajectory_with_cautions",
        )

    def test_unavailable_activity_stream_is_insufficient(
        self,
    ):
        result = _core_quality_state(
            activity_stream_state="unavailable",
            cohort_available=True,
            has_any_event=False,
            quality_flags=[],
        )

        self.assertEqual(
            result,
            "insufficient_for_core_trajectory",
        )

    def test_missing_cohort_is_insufficient(
        self,
    ):
        result = _core_quality_state(
            activity_stream_state="available",
            cohort_available=False,
            has_any_event=True,
            quality_flags=[],
        )

        self.assertEqual(
            result,
            "insufficient_for_core_trajectory",
        )

    def test_no_observed_trajectory_is_distinct_from_insufficient_data(
        self,
    ):
        no_events = _core_quality_state(
            activity_stream_state="available",
            cohort_available=True,
            has_any_event=False,
            quality_flags=[],
        )

        unavailable_data = _core_quality_state(
            activity_stream_state="unavailable",
            cohort_available=True,
            has_any_event=False,
            quality_flags=[],
        )

        self.assertNotEqual(
            no_events,
            unavailable_data,
        )

        self.assertEqual(
            no_events,
            "no_observed_trajectory",
        )

        self.assertEqual(
            unavailable_data,
            "insufficient_for_core_trajectory",
        )


if __name__ == "__main__":
    unittest.main()