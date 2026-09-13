import unittest

import pandas as pd

from src.trajectory.candidate_patterns import (
    _add_cutoff_patterns,
    _quality_lookup,
)


def _state_row(
    participant_id: int,
    engagement_state: str,
    *,
    days_since_last=None,
    last_explicit_date=None,
) -> dict:
    return {
        "participant_id": participant_id,
        "date": pd.Timestamp(
            "2026-09-11",
            tz="UTC",
        ),
        "engagement_state": engagement_state,
        "days_since_last_explicit_engagement": (
            days_since_last
        ),
        "last_explicit_engagement_date": (
            last_explicit_date
        ),
    }


def _quality_row(
    participant_id: int,
    *,
    has_any_event: bool,
    has_explicit_engagement: bool,
    navigation_events: int = 0,
    behavioral_sensor_events: int = 0,
    core_quality_state: str = (
        "sufficient_for_core_trajectory"
    ),
) -> dict:
    return {
        "participant_id": participant_id,
        "has_any_event": has_any_event,
        "has_explicit_engagement": (
            has_explicit_engagement
        ),
        "total_events": (
            navigation_events
            + behavioral_sensor_events
            + (
                1
                if has_explicit_engagement
                else 0
            )
        ),
        "navigation_events": (
            navigation_events
        ),
        "behavioral_sensor_events": (
            behavioral_sensor_events
        ),
        "core_quality_state": (
            core_quality_state
        ),
        "quality_flags": "[]",
        "observation_flags": "[]",
    }


class TestCandidatePatternSemantics(
    unittest.TestCase
):

    def setUp(self):
        self.cutoff = pd.Timestamp(
            "2026-09-11",
            tz="UTC",
        )

    def _patterns(
        self,
        state_rows,
        quality_rows,
    ):
        state = pd.DataFrame(
            state_rows
        )

        quality = pd.DataFrame(
            quality_rows
        )

        rows = []

        _add_cutoff_patterns(
            rows=rows,
            quality_lookup=(
                _quality_lookup(
                    quality
                )
            ),
            state=state,
            quality=quality,
            cutoff=self.cutoff,
        )

        return pd.DataFrame(
            rows
        )

    def test_no_events_uses_observational_pattern_name(
        self,
    ):
        patterns = self._patterns(
            [
                _state_row(
                    1,
                    (
                        "no_explicit_engagement_"
                        "observed_yet"
                    ),
                )
            ],
            [
                _quality_row(
                    1,
                    has_any_event=False,
                    has_explicit_engagement=False,
                    core_quality_state=(
                        "no_observed_trajectory"
                    ),
                )
            ],
        )

        self.assertEqual(
            len(patterns),
            1,
        )

        pattern = patterns.iloc[0]

        self.assertEqual(
            pattern["pattern_type"],
            (
                "no_explicit_engagement_"
                "observed_by_cutoff"
            ),
        )

        self.assertTrue(
            bool(
                pattern["right_censored"]
            )
        )

        self.assertEqual(
            pattern[
                "core_quality_state"
            ],
            "no_observed_trajectory",
        )

        # Deliberately ensure we do not introduce
        # stronger interpretations.
        pattern_names = " ".join(
            patterns[
                "pattern_type"
            ].astype(str)
        ).lower()

        self.assertNotIn(
            "dropout",
            pattern_names,
        )

        self.assertNotIn(
            "never_engaged",
            pattern_names,
        )

    def test_navigation_without_explicit_engagement_is_separate_pattern(
        self,
    ):
        patterns = self._patterns(
            [
                _state_row(
                    2,
                    (
                        "no_explicit_engagement_"
                        "observed_yet"
                    ),
                )
            ],
            [
                _quality_row(
                    2,
                    has_any_event=True,
                    has_explicit_engagement=False,
                    navigation_events=5,
                )
            ],
        )

        pattern_types = set(
            patterns[
                "pattern_type"
            ]
        )

        self.assertEqual(
            pattern_types,
            {
                (
                    "no_explicit_engagement_"
                    "observed_by_cutoff"
                ),
                (
                    "navigation_without_"
                    "explicit_engagement"
                ),
            },
        )

    def test_current_inactivity_requires_previous_explicit_engagement(
        self,
    ):
        patterns = self._patterns(
            [
                _state_row(
                    3,
                    "prolonged_inactivity_14d",
                    days_since_last=18,
                    last_explicit_date=(
                        pd.Timestamp(
                            "2026-08-24"
                        ).date()
                    ),
                )
            ],
            [
                _quality_row(
                    3,
                    has_any_event=True,
                    has_explicit_engagement=True,
                )
            ],
        )

        pattern_types = set(
            patterns[
                "pattern_type"
            ]
        )

        self.assertEqual(
            pattern_types,
            {
                (
                    "current_prolonged_"
                    "inactivity_14d"
                )
            },
        )

        self.assertNotIn(
            (
                "no_explicit_engagement_"
                "observed_by_cutoff"
            ),
            pattern_types,
        )

    def test_insufficient_core_data_suppresses_behavioral_pattern(
        self,
    ):
        patterns = self._patterns(
            [
                _state_row(
                    4,
                    (
                        "no_explicit_engagement_"
                        "observed_yet"
                    ),
                )
            ],
            [
                _quality_row(
                    4,
                    has_any_event=False,
                    has_explicit_engagement=False,
                    core_quality_state=(
                        "insufficient_for_"
                        "core_trajectory"
                    ),
                )
            ],
        )

        self.assertTrue(
            patterns.empty
        )


if __name__ == "__main__":
    unittest.main()