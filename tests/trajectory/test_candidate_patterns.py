import unittest

import pandas as pd

from src.trajectory.candidate_patterns import (
    _add_cutoff_patterns,
    _add_domain_disappearance_patterns,
    _add_tool_disappearance_patterns,
    _quality_lookup,
)
from src.trajectory.common import (
    TrajectoryAuditConfig,
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

        pattern = patterns.iloc[
            0
        ]

        self.assertEqual(
            pattern["signal_date"],
            pd.Timestamp(
                "2026-09-07",
                tz="UTC",
            ),
        )

        self.assertEqual(
            pattern["assessment_date"],
            self.cutoff,
        )

        self.assertEqual(
            pattern["detected_at"],
            pd.Timestamp(
                "2026-09-07",
                tz="UTC",
            ),
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

    def test_domain_disappearance_is_suppressed_when_overall_engagement_collapses(
        self,
    ):
        rows = []

        data = []

        # Reference window:
        # 8 nutrition + 8 physical-activity events.
        for index in range(8):
            data.append(
                {
                    "event_id": f"n{index}",
                    "participant_id": 10,
                    "date": pd.Timestamp(
                        "2026-08-01",
                        tz="UTC",
                    ),
                    "event_channel": (
                        "explicit_engagement"
                    ),
                    "domain": "nutrition",
                    "tool": "GameBus",
                    "event_weight": 1.0,
                }
            )

            data.append(
                {
                    "event_id": f"p{index}",
                    "participant_id": 10,
                    "date": pd.Timestamp(
                        "2026-08-01",
                        tz="UTC",
                    ),
                    "event_channel": (
                        "explicit_engagement"
                    ),
                    "domain": (
                        "physical_activity"
                    ),
                    "tool": "GameBus",
                    "event_weight": 1.0,
                }
            )

        # Recent window: only one event elsewhere.
        data.append(
            {
                "event_id": "recent",
                "participant_id": 10,
                "date": pd.Timestamp(
                    "2026-09-01",
                    tz="UTC",
                ),
                "event_channel": (
                    "explicit_engagement"
                ),
                "domain": (
                    "physical_activity"
                ),
                "tool": "GameBus",
                "event_weight": 1.0,
            }
        )

        _add_domain_disappearance_patterns(
            rows=rows,
            quality_lookup={},
            domain_tool=pd.DataFrame(
                data
            ),
            config=TrajectoryAuditConfig(),
            cutoff=self.cutoff,
        )

        self.assertFalse(
            any(
                row["pattern_type"]
                == "selective_domain_disappearance"
                for row in rows
            )
        )

    def test_domain_disappearance_requires_meaningful_continuation_elsewhere(
        self,
    ):
        rows = []

        data = []

        # Reference:
        # nutrition = 4, physical activity = 6.
        for index in range(4):
            data.append(
                {
                    "event_id": f"n{index}",
                    "participant_id": 11,
                    "date": pd.Timestamp(
                        "2026-08-01",
                        tz="UTC",
                    ),
                    "event_channel": (
                        "explicit_engagement"
                    ),
                    "domain": "nutrition",
                    "tool": "GameBus",
                    "event_weight": 1.0,
                }
            )

        for index in range(6):
            data.append(
                {
                    "event_id": f"p{index}",
                    "participant_id": 11,
                    "date": pd.Timestamp(
                        "2026-08-01",
                        tz="UTC",
                    ),
                    "event_channel": (
                        "explicit_engagement"
                    ),
                    "domain": (
                        "physical_activity"
                    ),
                    "tool": "GameBus",
                    "event_weight": 1.0,
                }
            )

        # Recent:
        # nutrition disappears, but physical activity
        # remains substantial: 6 / 10 = 0.60.
        for index in range(6):
            data.append(
                {
                    "event_id": f"r{index}",
                    "participant_id": 11,
                    "date": pd.Timestamp(
                        "2026-09-01",
                        tz="UTC",
                    ),
                    "event_channel": (
                        "explicit_engagement"
                    ),
                    "domain": (
                        "physical_activity"
                    ),
                    "tool": "GameBus",
                    "event_weight": 1.0,
                }
            )

        _add_domain_disappearance_patterns(
            rows=rows,
            quality_lookup={},
            domain_tool=pd.DataFrame(
                data
            ),
            config=TrajectoryAuditConfig(),
            cutoff=self.cutoff,
        )

        patterns = [
            row
            for row in rows
            if (
                row["pattern_type"]
                == "selective_domain_disappearance"
            )
        ]

        self.assertEqual(
            len(patterns),
            1,
        )

        self.assertEqual(
            patterns[0]["subject"],
            "nutrition",
        )


if __name__ == "__main__":
    unittest.main()