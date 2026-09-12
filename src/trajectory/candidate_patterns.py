from __future__ import annotations

import json
import os

from typing import Any, Dict, List
from datetime import date, datetime

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)

from src.trajectory.data_quality_state import (
    build_data_quality_state,
)

from src.trajectory.domain_tool_engagement import (
    build_domain_tool_engagement,
)

from src.trajectory.event_normalization import (
    build_normalized_events,
)

from src.trajectory.inactivity_gap_analysis import (
    build_inactivity_gaps,
)

from src.trajectory.participant_state_daily import (
    build_participant_state_daily,
)


PATTERN_COLUMNS = [
    "participant_id",
    "pattern_id",

    "pattern_type",
    "pattern_family",
    "subject",

    "detected_at",

    "status",
    "right_censored",

    "core_quality_state",
    "quality_flags",
    "observation_flags",

    "evidence",
]


def _json_evidence(
    payload: Dict[str, Any],
) -> str:
    """
    Serialize pattern evidence in a readable,
    machine-processable way.

    Evidence may contain:
        Python date/datetime values
        pandas timestamps
        pandas NA/NaT
        NumPy scalar values

    Convert all of them to normal JSON-compatible
    Python values.
    """

    def make_jsonable(
        value: Any,
    ) -> Any:

        # -----------------------------------------
        # Dates and timestamps
        # -----------------------------------------

        if isinstance(
            value,
            (
                pd.Timestamp,
                datetime,
                date,
            ),
        ):
            if pd.isna(value):
                return None

            return value.isoformat()

        # -----------------------------------------
        # Explicit pandas missing values
        # -----------------------------------------

        if (
            value is pd.NA
            or value is pd.NaT
        ):
            return None

        # -----------------------------------------
        # Dictionaries
        # -----------------------------------------

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): make_jsonable(
                    item
                )
                for key, item
                in value.items()
            }

        # -----------------------------------------
        # Lists / tuples / sets
        # -----------------------------------------

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                make_jsonable(
                    item
                )
                for item in value
            ]

        # -----------------------------------------
        # NumPy scalar values:
        # np.int64, np.float64, np.bool_, etc.
        # -----------------------------------------

        item_method = getattr(
            value,
            "item",
            None,
        )

        if callable(
            item_method
        ):
            try:
                return make_jsonable(
                    item_method()
                )
            except Exception:
                pass

        # -----------------------------------------
        # Generic missing-value check
        # -----------------------------------------

        try:
            if pd.isna(
                value
            ):
                return None
        except Exception:
            pass

        return value

    cleaned = make_jsonable(
        payload
    )

    return json.dumps(
        cleaned,
        ensure_ascii=False,
    )


def _quality_lookup(
    quality: pd.DataFrame,
) -> Dict[int, Dict[str, Any]]:
    """
    Make participant quality metadata easy to attach
    to every candidate pattern.
    """

    result: Dict[
        int,
        Dict[str, Any],
    ] = {}

    if quality.empty:
        return result

    for _, row in quality.iterrows():

        participant_id = int(
            row[
                "participant_id"
            ]
        )

        result[
            participant_id
        ] = {
            "core_quality_state": (
                row[
                    "core_quality_state"
                ]
            ),

            "quality_flags": (
                row[
                    "quality_flags"
                ]
            ),

            "observation_flags": (
                row[
                    "observation_flags"
                ]
            ),
        }

    return result


def _append_pattern(
    rows: List[Dict[str, Any]],
    quality_lookup: Dict[
        int,
        Dict[str, Any],
    ],
    participant_id: int,
    pattern_type: str,
    pattern_family: str,
    subject: str | None,
    detected_at: Any,
    evidence: Dict[str, Any],
    right_censored: bool = False,
) -> None:
    """
    Add one exploratory candidate pattern.
    """

    quality = (
        quality_lookup.get(
            participant_id,
            {},
        )
    )

    # If the evidence is insufficient for core
    # trajectory analysis, do not silently create
    # behavioral interpretations.
    if (
        quality.get(
            "core_quality_state"
        )
        == "insufficient_for_core_trajectory"
    ):
        return

    rows.append(
        {
            "participant_id": (
                participant_id
            ),

            "pattern_type": (
                pattern_type
            ),

            "pattern_family": (
                pattern_family
            ),

            "subject": (
                subject
            ),

            "detected_at": (
                detected_at
            ),

            "status": (
                "exploratory_candidate"
            ),

            "right_censored": (
                right_censored
            ),

            "core_quality_state": (
                quality.get(
                    "core_quality_state",
                    "unknown",
                )
            ),

            "quality_flags": (
                quality.get(
                    "quality_flags",
                    "[]",
                )
            ),

            "observation_flags": (
                quality.get(
                    "observation_flags",
                    "[]",
                )
            ),

            "evidence": (
                _json_evidence(
                    evidence
                )
            ),
        }
    )


def _window_dates(
    cutoff: pd.Timestamp,
    config: TrajectoryAuditConfig,
) -> Dict[str, pd.Timestamp]:
    """
    Build recent and immediately preceding
    reference windows.

    Example with 28 + 28 days:

        reference:
        cutoff-55 ... cutoff-28

        recent:
        cutoff-27 ... cutoff
    """

    recent_start = (
        cutoff
        - pd.Timedelta(
            days=(
                config
                .pattern_recent_window_days
                - 1
            )
        )
    )

    reference_end = (
        recent_start
        - pd.Timedelta(days=1)
    )

    reference_start = (
        reference_end
        - pd.Timedelta(
            days=(
                config
                .pattern_reference_window_days
                - 1
            )
        )
    )

    return {
        "reference_start": (
            reference_start
        ),

        "reference_end": (
            reference_end
        ),

        "recent_start": (
            recent_start
        ),

        "recent_end": (
            cutoff
        ),
    }


def _date_mask(
    dates: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Inclusive date-window selection.
    """

    return (
        dates.ge(start)
        & dates.le(end)
    )


def _add_cutoff_patterns(
    rows: List[Dict[str, Any]],
    quality_lookup: Dict[
        int,
        Dict[str, Any],
    ],
    state: pd.DataFrame,
    quality: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> None:
    """
    Patterns observable directly at the current
    analysis cutoff.
    """

    if state.empty:
        return

    prepared = state.copy()

    prepared["date"] = pd.to_datetime(
        prepared["date"],
        utc=True,
        errors="coerce",
    )

    cutoff_state = prepared.loc[
        prepared[
            "date"
        ]
        == cutoff
    ]

    quality_by_pid = (
        quality.set_index(
            "participant_id"
        )
        if not quality.empty
        else pd.DataFrame()
    )

    for _, current in (
        cutoff_state.iterrows()
    ):

        participant_id = int(
            current[
                "participant_id"
            ]
        )

        quality_row = None

        if (
            not quality.empty
            and participant_id
            in quality_by_pid.index
        ):

            quality_row = (
                quality_by_pid.loc[
                    participant_id
                ]
            )

        # =================================================
        # No explicit engagement observed by cutoff
        # =================================================

        has_explicit = (
            bool(
                quality_row[
                    "has_explicit_engagement"
                ]
            )
            if quality_row is not None
            else False
        )

        if not has_explicit:

            _append_pattern(
                rows,
                quality_lookup,
                participant_id,
                (
                    "no_explicit_engagement_"
                    "observed_by_cutoff"
                ),
                "engagement",
                None,
                cutoff,
                {
                    "analysis_cutoff": (
                        cutoff
                    ),

                    "has_any_event": (
                        quality_row[
                            "has_any_event"
                        ]
                        if quality_row
                        is not None
                        else None
                    ),

                    "total_events": (
                        quality_row[
                            "total_events"
                        ]
                        if quality_row
                        is not None
                        else None
                    ),

                    "navigation_events": (
                        quality_row[
                            "navigation_events"
                        ]
                        if quality_row
                        is not None
                        else None
                    ),

                    "behavioral_sensor_events": (
                        quality_row[
                            "behavioral_sensor_events"
                        ]
                        if quality_row
                        is not None
                        else None
                    ),
                },
                right_censored=True,
            )

        # =================================================
        # Navigation without explicit engagement
        # =================================================

        if (
            quality_row is not None
            and int(
                quality_row[
                    "navigation_events"
                ]
            )
            > 0
            and not has_explicit
        ):

            _append_pattern(
                rows,
                quality_lookup,
                participant_id,
                (
                    "navigation_without_"
                    "explicit_engagement"
                ),
                "channel_mismatch",
                "navigation",
                cutoff,
                {
                    "navigation_events": int(
                        quality_row[
                            "navigation_events"
                        ]
                    ),

                    "explicit_engagement_events": 0,
                },
                right_censored=True,
            )

        # =================================================
        # Behavioral sensor without explicit engagement
        # =================================================

        if (
            quality_row is not None
            and int(
                quality_row[
                    "behavioral_sensor_events"
                ]
            )
            > 0
            and not has_explicit
        ):

            _append_pattern(
                rows,
                quality_lookup,
                participant_id,
                (
                    "behavioral_sensor_without_"
                    "explicit_engagement"
                ),
                "channel_mismatch",
                "behavioral_sensor",
                cutoff,
                {
                    "behavioral_sensor_events": int(
                        quality_row[
                            "behavioral_sensor_events"
                        ]
                    ),

                    "explicit_engagement_events": 0,
                },
                right_censored=True,
            )

        # =================================================
        # Current inactivity
        # =================================================

        engagement_state = str(
            current[
                "engagement_state"
            ]
        )

        if (
            engagement_state
            == "prolonged_inactivity_7d"
        ):

            _append_pattern(
                rows,
                quality_lookup,
                participant_id,
                (
                    "current_prolonged_"
                    "inactivity_7d"
                ),
                "inactivity",
                None,
                cutoff,
                {
                    "days_since_last_explicit_engagement": (
                        current[
                            "days_since_last_explicit_engagement"
                        ]
                    ),

                    "last_explicit_engagement_date": (
                        current[
                            "last_explicit_engagement_date"
                        ]
                    ),
                },
                right_censored=True,
            )

        elif (
            engagement_state
            == "prolonged_inactivity_14d"
        ):

            _append_pattern(
                rows,
                quality_lookup,
                participant_id,
                (
                    "current_prolonged_"
                    "inactivity_14d"
                ),
                "inactivity",
                None,
                cutoff,
                {
                    "days_since_last_explicit_engagement": (
                        current[
                            "days_since_last_explicit_engagement"
                        ]
                    ),

                    "last_explicit_engagement_date": (
                        current[
                            "last_explicit_engagement_date"
                        ]
                    ),
                },
                right_censored=True,
            )


def _add_gap_patterns(
    rows: List[Dict[str, Any]],
    quality_lookup: Dict[
        int,
        Dict[str, Any],
    ],
    gaps: pd.DataFrame,
    config: TrajectoryAuditConfig,
    cutoff: pd.Timestamp,
) -> None:
    """
    Retrospective re-engagement and repeated-gap
    candidate patterns.
    """

    if gaps.empty:
        return

    prepared = gaps.copy()

    prepared[
        "next_engagement_date"
    ] = pd.to_datetime(
        prepared[
            "next_engagement_date"
        ],
        utc=True,
        errors="coerce",
    )

    prepared[
        "inactive_days"
    ] = pd.to_numeric(
        prepared[
            "inactive_days"
        ],
        errors="coerce",
    )

    # =====================================================
    # Re-engagement after a long gap
    # =====================================================

    long_reengaged = prepared.loc[
        (
            prepared[
                "inactive_days"
            ]
            >= config.episode_gap_days
        )
        &
        prepared[
            "reengaged"
        ]
    ]

    for _, gap in (
        long_reengaged.iterrows()
    ):

        participant_id = int(
            gap[
                "participant_id"
            ]
        )

        _append_pattern(
            rows,
            quality_lookup,
            participant_id,
            "reengaged_after_long_gap",
            "reengagement",
            None,
            gap[
                "next_engagement_date"
            ],
            {
                "gap_id": (
                    gap[
                        "gap_id"
                    ]
                ),

                "inactive_days": int(
                    gap[
                        "inactive_days"
                    ]
                ),

                "previous_engagement_date": (
                    gap[
                        "previous_engagement_date"
                    ]
                ),

                "reengagement_date": (
                    gap[
                        "next_engagement_date"
                    ]
                ),
            },
            right_censored=False,
        )

    # =====================================================
    # Repeated long gaps
    # =====================================================

    long_gaps = prepared.loc[
        prepared[
            "inactive_days"
        ]
        >= config.episode_gap_days
    ]

    for (
        participant_id,
        participant_gaps,
    ) in long_gaps.groupby(
        "participant_id"
    ):

        if (
            len(participant_gaps)
            < config.pattern_repeated_gap_count
        ):
            continue

        participant_id = int(
            participant_id
        )

        _append_pattern(
            rows,
            quality_lookup,
            participant_id,
            "repeated_long_gaps",
            "inactivity",
            None,
            cutoff,
            {
                "long_gap_count": int(
                    len(
                        participant_gaps
                    )
                ),

                "required_count": (
                    config
                    .pattern_repeated_gap_count
                ),

                "max_inactive_days": int(
                    participant_gaps[
                        "inactive_days"
                    ].max()
                ),

                "right_censored_long_gaps": int(
                    participant_gaps[
                        "right_censored"
                    ].sum()
                ),
            },
            right_censored=bool(
                participant_gaps[
                    "right_censored"
                ].any()
            ),
        )


def _add_recent_decline_patterns(
    rows: List[Dict[str, Any]],
    quality_lookup: Dict[
        int,
        Dict[str, Any],
    ],
    domain_tool: pd.DataFrame,
    config: TrajectoryAuditConfig,
    cutoff: pd.Timestamp,
) -> None:
    """
    Compare recent explicit engagement with the
    immediately preceding reference window.

    We count UNIQUE activity events here because
    domain_tool is long-format and multi-domain
    activities appear more than once.
    """

    if domain_tool.empty:
        return

    data = domain_tool.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        utc=True,
        errors="coerce",
    )

    data = data.loc[
        data[
            "event_channel"
        ]
        == "explicit_engagement"
    ].copy()

    if data.empty:
        return

    windows = _window_dates(
        cutoff,
        config,
    )

    # One row per actual activity.
    unique_events = (
        data[
            [
                "event_id",
                "participant_id",
                "date",
            ]
        ]
        .drop_duplicates(
            "event_id"
        )
    )

    for (
        participant_id,
        participant_events,
    ) in unique_events.groupby(
        "participant_id"
    ):

        reference_count = int(
            _date_mask(
                participant_events[
                    "date"
                ],
                windows[
                    "reference_start"
                ],
                windows[
                    "reference_end"
                ],
            ).sum()
        )

        recent_count = int(
            _date_mask(
                participant_events[
                    "date"
                ],
                windows[
                    "recent_start"
                ],
                windows[
                    "recent_end"
                ],
            ).sum()
        )

        if (
            reference_count
            < config.pattern_min_reference_events
        ):
            continue

        # Complete disappearance is handled by the
        # current inactivity patterns. Here we want
        # a decline while engagement still exists.
        if recent_count == 0:
            continue

        ratio = (
            recent_count
            / reference_count
        )

        if (
            ratio
            > config.pattern_decline_ratio
        ):
            continue

        _append_pattern(
            rows,
            quality_lookup,
            int(participant_id),
            "recent_engagement_decline",
            "engagement",
            None,
            cutoff,
            {
                "reference_start": (
                    windows[
                        "reference_start"
                    ]
                ),

                "reference_end": (
                    windows[
                        "reference_end"
                    ]
                ),

                "recent_start": (
                    windows[
                        "recent_start"
                    ]
                ),

                "recent_end": (
                    windows[
                        "recent_end"
                    ]
                ),

                "reference_events": (
                    reference_count
                ),

                "recent_events": (
                    recent_count
                ),

                "recent_to_reference_ratio": (
                    round(
                        ratio,
                        4,
                    )
                ),

                "decline_ratio_threshold": (
                    config
                    .pattern_decline_ratio
                ),
            },
            right_censored=True,
        )


def _add_domain_disappearance_patterns(
    rows: List[Dict[str, Any]],
    quality_lookup: Dict[
        int,
        Dict[str, Any],
    ],
    domain_tool: pd.DataFrame,
    config: TrajectoryAuditConfig,
    cutoff: pd.Timestamp,
) -> None:
    """
    Candidate selective domain disappearance.

    Requirements:

        domain was meaningfully present
        in the reference window

        domain has zero weighted events
        in the recent window

        participant still has explicit engagement
        in some OTHER domain recently

    Therefore global inactivity does not generate
    three redundant domain-disappearance patterns.
    """

    if domain_tool.empty:
        return

    data = domain_tool.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        utc=True,
        errors="coerce",
    )

    data = data.loc[
        (
            data[
                "event_channel"
            ]
            == "explicit_engagement"
        )
        &
        (
            data[
                "domain"
            ]
            != "unmapped"
        )
    ].copy()

    if data.empty:
        return

    data[
        "event_weight"
    ] = pd.to_numeric(
        data[
            "event_weight"
        ],
        errors="coerce",
    ).fillna(0)

    windows = _window_dates(
        cutoff,
        config,
    )

    recent_mask = _date_mask(
        data[
            "date"
        ],
        windows[
            "recent_start"
        ],
        windows[
            "recent_end"
        ],
    )

    reference_mask = _date_mask(
        data[
            "date"
        ],
        windows[
            "reference_start"
        ],
        windows[
            "reference_end"
        ],
    )

    recent = data.loc[
        recent_mask
    ]

    reference = data.loc[
        reference_mask
    ]

    if reference.empty:
        return

    recent_total_by_participant = (
        recent.groupby(
            "participant_id"
        )[
            "event_weight"
        ]
        .sum()
    )

    reference_domain = (
        reference.groupby(
            [
                "participant_id",
                "domain",
            ]
        )[
            "event_weight"
        ]
        .sum()
    )

    recent_domain = (
        recent.groupby(
            [
                "participant_id",
                "domain",
            ]
        )[
            "event_weight"
        ]
        .sum()
    )

    for (
        participant_id,
        domain,
    ), reference_weight in (
        reference_domain.items()
    ):

        if (
            reference_weight
            < config.pattern_min_reference_events
        ):
            continue

        recent_weight = float(
            recent_domain.get(
                (
                    participant_id,
                    domain,
                ),
                0.0,
            )
        )

        if recent_weight > 0:
            continue

        recent_total = float(
            recent_total_by_participant.get(
                participant_id,
                0.0,
            )
        )

        # Participant must still be explicitly
        # engaging elsewhere.
        if recent_total <= 0:
            continue

        _append_pattern(
            rows,
            quality_lookup,
            int(participant_id),
            "selective_domain_disappearance",
            "domain",
            str(domain),
            cutoff,
            {
                "domain": (
                    domain
                ),

                "reference_domain_weight": (
                    round(
                        float(
                            reference_weight
                        ),
                        4,
                    )
                ),

                "recent_domain_weight": (
                    0.0
                ),

                "recent_total_explicit_weight": (
                    round(
                        recent_total,
                        4,
                    )
                ),

                "reference_start": (
                    windows[
                        "reference_start"
                    ]
                ),

                "reference_end": (
                    windows[
                        "reference_end"
                    ]
                ),

                "recent_start": (
                    windows[
                        "recent_start"
                    ]
                ),

                "recent_end": (
                    windows[
                        "recent_end"
                    ]
                ),
            },
            right_censored=True,
        )


def _add_tool_disappearance_patterns(
    rows: List[Dict[str, Any]],
    quality_lookup: Dict[
        int,
        Dict[str, Any],
    ],
    domain_tool: pd.DataFrame,
    config: TrajectoryAuditConfig,
    cutoff: pd.Timestamp,
) -> None:
    """
    Candidate selective tool disappearance.

    To reduce false interpretation of a system-wide
    data outage, the tool must still be observed for
    SOME participant in the recent window.
    """

    if domain_tool.empty:
        return

    data = domain_tool.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        utc=True,
        errors="coerce",
    )

    # Because multi-domain events appear more than once,
    # reduce back to one event/tool record.
    data = (
        data[
            [
                "event_id",
                "participant_id",
                "date",
                "tool",
            ]
        ]
        .drop_duplicates(
            [
                "event_id",
                "tool",
            ]
        )
    )

    windows = _window_dates(
        cutoff,
        config,
    )

    recent_mask = _date_mask(
        data[
            "date"
        ],
        windows[
            "recent_start"
        ],
        windows[
            "recent_end"
        ],
    )

    reference_mask = _date_mask(
        data[
            "date"
        ],
        windows[
            "reference_start"
        ],
        windows[
            "reference_end"
        ],
    )

    recent = data.loc[
        recent_mask
    ]

    reference = data.loc[
        reference_mask
    ]

    if reference.empty:
        return

    # Critical safeguard:
    # only evaluate disappearance for tools for which
    # SOME recent data still exists in the cohort.
    cohort_recent_tools = set(
        recent[
            "tool"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    recent_total_by_participant = (
        recent.groupby(
            "participant_id"
        )
        .size()
    )

    reference_tool = (
        reference.groupby(
            [
                "participant_id",
                "tool",
            ]
        )
        .size()
    )

    recent_tool = (
        recent.groupby(
            [
                "participant_id",
                "tool",
            ]
        )
        .size()
    )

    for (
        participant_id,
        tool,
    ), reference_count in (
        reference_tool.items()
    ):

        if (
            reference_count
            < config.pattern_min_reference_events
        ):
            continue

        if (
            str(tool)
            not in cohort_recent_tools
        ):
            # Could be a stream-level/data-pipeline
            # disappearance rather than participant
            # behavior.
            continue

        recent_count = int(
            recent_tool.get(
                (
                    participant_id,
                    tool,
                ),
                0,
            )
        )

        if recent_count > 0:
            continue

        recent_total = int(
            recent_total_by_participant.get(
                participant_id,
                0,
            )
        )

        # Require continued observation through
        # another tool.
        if recent_total <= 0:
            continue

        _append_pattern(
            rows,
            quality_lookup,
            int(participant_id),
            "selective_tool_disappearance",
            "tool",
            str(tool),
            cutoff,
            {
                "tool": (
                    tool
                ),

                "reference_tool_events": int(
                    reference_count
                ),

                "recent_tool_events": 0,

                "recent_events_other_tools": (
                    recent_total
                ),

                "reference_start": (
                    windows[
                        "reference_start"
                    ]
                ),

                "reference_end": (
                    windows[
                        "reference_end"
                    ]
                ),

                "recent_start": (
                    windows[
                        "recent_start"
                    ]
                ),

                "recent_end": (
                    windows[
                        "recent_end"
                    ]
                ),

                "cohort_tool_still_observed_recently": (
                    True
                ),
            },
            right_censored=True,
        )


def build_candidate_patterns(
    config: TrajectoryAuditConfig,
    state: pd.DataFrame,
    gaps: pd.DataFrame,
    domain_tool: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine trajectory layers into exploratory,
    auditable candidate patterns.

    These are NOT validated predictors.
    """

    if state.empty:

        return pd.DataFrame(
            columns=PATTERN_COLUMNS
        )

    state_dates = pd.to_datetime(
        state[
            "date"
        ],
        utc=True,
        errors="coerce",
    ).dropna()

    if state_dates.empty:

        return pd.DataFrame(
            columns=PATTERN_COLUMNS
        )

    cutoff = (
        state_dates
        .max()
        .floor("D")
    )

    quality_lookup = (
        _quality_lookup(
            quality
        )
    )

    rows: List[
        Dict[str, Any]
    ] = []

    # -------------------------------------------------
    # Current participant state
    # -------------------------------------------------

    _add_cutoff_patterns(
        rows,
        quality_lookup,
        state,
        quality,
        cutoff,
    )

    # -------------------------------------------------
    # Historical gap / re-engagement structure
    # -------------------------------------------------

    _add_gap_patterns(
        rows,
        quality_lookup,
        gaps,
        config,
        cutoff,
    )

    # -------------------------------------------------
    # Recent decline
    # -------------------------------------------------

    _add_recent_decline_patterns(
        rows,
        quality_lookup,
        domain_tool,
        config,
        cutoff,
    )

    # -------------------------------------------------
    # Selective domain change
    # -------------------------------------------------

    _add_domain_disappearance_patterns(
        rows,
        quality_lookup,
        domain_tool,
        config,
        cutoff,
    )

    # -------------------------------------------------
    # Selective tool change
    # -------------------------------------------------

    _add_tool_disappearance_patterns(
        rows,
        quality_lookup,
        domain_tool,
        config,
        cutoff,
    )

    if not rows:

        return pd.DataFrame(
            columns=PATTERN_COLUMNS
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "detected_at"
    ] = pd.to_datetime(
        result[
            "detected_at"
        ],
        utc=True,
        errors="coerce",
    )

    result = result.sort_values(
        [
            "participant_id",
            "detected_at",
            "pattern_family",
            "pattern_type",
            "subject",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------
    # Stable pattern IDs
    # -------------------------------------------------

    result[
        "_pattern_number"
    ] = (
        result
        .groupby(
            "participant_id"
        )
        .cumcount()
        + 1
    )

    result[
        "pattern_id"
    ] = (
        result[
            "participant_id"
        ]
        .astype(int)
        .astype(str)
        + ":P"
        + result[
            "_pattern_number"
        ]
        .astype(str)
        .str.zfill(3)
    )

    result = result.drop(
        columns=[
            "_pattern_number"
        ]
    )

    result[
        "detected_at"
    ] = (
        result[
            "detected_at"
        ]
        .dt.date
    )

    for column in PATTERN_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    return result[
        PATTERN_COLUMNS
    ]


def run_candidate_patterns(
    config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
    """
    Build and save candidate_patterns.csv.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    events = build_normalized_events(
        config
    )

    state = (
        build_participant_state_daily(
            config,
            events,
        )
    )

    gaps = (
        build_inactivity_gaps(
            config,
            events,
        )
    )

    domain_tool = (
        build_domain_tool_engagement(
            config,
            events,
        )
    )

    quality, _ = (
        build_data_quality_state(
            config,
            events,
            domain_tool,
        )
    )

    patterns = (
        build_candidate_patterns(
            config,
            state,
            gaps,
            domain_tool,
            quality,
        )
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "candidate_patterns.csv",
    )

    patterns.to_csv(
        output_path,
        index=False,
    )

    return patterns


if __name__ == "__main__":

    patterns = (
        run_candidate_patterns()
    )

    print(
        "Candidate patterns "
        "written successfully."
    )

    print()

    print(
        "Candidate pattern rows:",
        len(patterns),
    )

    if not patterns.empty:

        print(
            "Participants with >=1 pattern:",
            patterns[
                "participant_id"
            ].nunique(),
        )

        print()

        print(
            "Patterns by type:"
        )

        counts = (
            patterns[
                "pattern_type"
            ]
            .value_counts()
        )

        for (
            pattern_type,
            count,
        ) in counts.items():

            print(
                f"  {pattern_type}: "
                f"{count}"
            )

        print()

        print(
            "Patterns by family:"
        )

        family_counts = (
            patterns[
                "pattern_family"
            ]
            .value_counts()
        )

        for (
            family,
            count,
        ) in (
            family_counts.items()
        ):

            print(
                f"  {family}: "
                f"{count}"
            )