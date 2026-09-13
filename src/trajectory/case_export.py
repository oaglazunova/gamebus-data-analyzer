from __future__ import annotations

import json
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from typing import Any, Dict, List
from src.trajectory.candidate_patterns import (
    build_candidate_patterns,
)
from src.trajectory.common import (
    TrajectoryAuditConfig,
    write_json,
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
from src.trajectory.participation_episode_builder import (
    build_participation_episodes,
)



CASE_INDEX_COLUMNS = [
    "selection_rank",
    "participant_id",

    "selection_reason",
    "pattern_count",
    "pattern_types",

    "current_engagement_state",
    "days_since_last_explicit_engagement",

    "explicit_engagement_events",
    "navigation_events",
    "behavioral_sensor_events",

    "episode_count",
    "gap_count",
    "max_inactive_days",

    "case_directory",
    "trajectory_plot",
    "domain_tool_plot",
]


# ---------------------------------------------------------
# Case diversity priorities
# ---------------------------------------------------------
#
# Lower number = try to include this type earlier.
#
# This is NOT a clinical or scientific severity ranking.
# It is only a case-selection strategy designed to give
# us qualitatively different trajectories to inspect.
#
PATTERN_SELECTION_PRIORITY = [
    "selective_tool_disappearance",
    "selective_domain_disappearance",
    "recent_engagement_decline",
    "reengaged_after_long_gap",
    "repeated_long_gaps",
    "current_prolonged_inactivity_14d",
    "current_prolonged_inactivity_7d",
    "navigation_without_explicit_engagement",
    "no_explicit_engagement_observed_by_cutoff",
]


def _safe_json_load(
    value: Any,
) -> Any:
    """
    Parse a JSON string when possible.
    Otherwise return the original value.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(
        value,
        (
            dict,
            list,
        ),
    ):
        return value

    try:
        return json.loads(
            str(value)
        )
    except Exception:
        return value


def _participant_pattern_summary(
    patterns: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per participant with pattern count
    and distinct pattern types.
    """

    if patterns.empty:

        return pd.DataFrame(
            columns=[
                "participant_id",
                "pattern_count",
                "pattern_types",
            ]
        )

    rows = []

    for (
        participant_id,
        group,
    ) in patterns.groupby(
        "participant_id"
    ):

        pattern_types = (
            group[
                "pattern_type"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        rows.append(
            {
                "participant_id": int(
                    participant_id
                ),

                "pattern_count": int(
                    len(group)
                ),

                "pattern_types": (
                    pattern_types
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def select_case_participants(
    patterns: pd.DataFrame,
    max_cases: int,
) -> pd.DataFrame:
    """
    Select a diversified set of participant cases.

    The goal is qualitative coverage, not risk ranking.

    Strategy:

    1. Walk through candidate pattern types in a
       deterministic priority order.

    2. Skip a pattern type if it is already represented
       by a previously selected participant.

    3. When choosing a participant for a new type,
       prefer the participant whose other patterns
       overlap least with patterns already represented.

    4. Fill remaining capacity with rich cases having
       combinations not already well represented.
    """

    if (
        patterns.empty
        or max_cases <= 0
    ):
        return pd.DataFrame(
            columns=[
                "participant_id",
                "selection_reason",
            ]
        )

    summary = (
        _participant_pattern_summary(
            patterns
        )
    )

    pattern_types_lookup = {
        int(row["participant_id"]): set(
            row["pattern_types"]
        )
        for _, row
        in summary.iterrows()
    }

    pattern_count_lookup = {
        int(row["participant_id"]): int(
            row["pattern_count"]
        )
        for _, row
        in summary.iterrows()
    }

    selected: List[int] = []

    reason: Dict[
        int,
        str,
    ] = {}

    represented_types = set()

    # -------------------------------------------------
    # First pass:
    # cover as many distinct pattern types as possible.
    # -------------------------------------------------

    for pattern_type in (
        PATTERN_SELECTION_PRIORITY
    ):

        # This type is already visible in one of
        # the selected cases.
        if (
            pattern_type
            in represented_types
        ):
            continue

        candidates = (
            patterns.loc[
                patterns[
                    "pattern_type"
                ]
                == pattern_type,
                "participant_id",
            ]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )

        candidates = [
            participant_id
            for participant_id
            in candidates
            if participant_id
            not in selected
        ]

        if not candidates:
            continue

        def candidate_score(
            participant_id: int,
        ):
            participant_types = (
                pattern_types_lookup.get(
                    participant_id,
                    set(),
                )
            )

            # How much does this participant merely
            # repeat things we already selected?
            overlap = len(
                participant_types
                & represented_types
            )

            # How many genuinely new pattern types
            # would this participant add?
            new_types = len(
                participant_types
                - represented_types
            )

            return (
                overlap,
                -new_types,
                -pattern_count_lookup.get(
                    participant_id,
                    0,
                ),
                participant_id,
            )

        candidates = sorted(
            candidates,
            key=candidate_score,
        )

        chosen = (
            candidates[0]
        )

        selected.append(
            chosen
        )

        reason[
            chosen
        ] = (
            f"representative_of:"
            f"{pattern_type}"
        )

        represented_types.update(
            pattern_types_lookup.get(
                chosen,
                set(),
            )
        )

        if (
            len(selected)
            >= max_cases
        ):
            break

    # -------------------------------------------------
    # Second pass:
    # fill remaining slots with useful combinations.
    # -------------------------------------------------

    while (
        len(selected)
        < max_cases
    ):

        candidates = [
            int(
                participant_id
            )
            for participant_id
            in summary[
                "participant_id"
            ]
            if int(
                participant_id
            )
            not in selected
        ]

        if not candidates:
            break

        def fill_score(
            participant_id: int,
        ):
            participant_types = (
                pattern_types_lookup.get(
                    participant_id,
                    set(),
                )
            )

            new_types = len(
                participant_types
                - represented_types
            )

            # Prefer rich trajectories after
            # maximizing remaining novelty.
            pattern_count = (
                pattern_count_lookup.get(
                    participant_id,
                    0,
                )
            )

            return (
                -new_types,
                -pattern_count,
                participant_id,
            )

        chosen = sorted(
            candidates,
            key=fill_score,
        )[0]

        selected.append(
            chosen
        )

        reason[
            chosen
        ] = (
            "additional_diverse_case"
        )

        represented_types.update(
            pattern_types_lookup.get(
                chosen,
                set(),
            )
        )

    return pd.DataFrame(
        {
            "participant_id": (
                selected
            ),

            "selection_reason": [
                reason[
                    participant_id
                ]
                for participant_id
                in selected
            ],
        }
    )


def _filter_participant(
    df: pd.DataFrame,
    participant_id: int,
) -> pd.DataFrame:
    """
    Filter any participant-level table safely.
    """

    if (
        df.empty
        or "participant_id"
        not in df.columns
    ):

        return df.iloc[
            0:0
        ].copy()

    participant_values = (
        pd.to_numeric(
            df[
                "participant_id"
            ],
            errors="coerce",
        )
    )

    return (
        df.loc[
            participant_values
            == participant_id
        ]
        .copy()
    )


def _current_state(
    participant_state: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Return participant state at the latest
    observation date.
    """

    if participant_state.empty:

        return {
            "date": None,
            "engagement_state": None,
            "days_since_last_explicit_engagement": None,
            "last_explicit_engagement_date": None,
            "episode_id": None,
        }

    prepared = (
        participant_state
        .copy()
    )

    prepared["date"] = pd.to_datetime(
        prepared[
            "date"
        ],
        errors="coerce",
    )

    row = (
        prepared
        .sort_values(
            "date"
        )
        .iloc[-1]
    )

    return {
        "date": (
            str(
                row[
                    "date"
                ].date()
            )
            if pd.notna(
                row[
                    "date"
                ]
            )
            else None
        ),

        "engagement_state": (
            row[
                "engagement_state"
            ]
        ),

        "days_since_last_explicit_engagement": (
            row[
                "days_since_last_explicit_engagement"
            ]
        ),

        "last_explicit_engagement_date": (
            row[
                "last_explicit_engagement_date"
            ]
        ),

        "episode_id": (
            row[
                "episode_id"
            ]
        ),
    }


def _case_summary(
    participant_id: int,
    participant_events: pd.DataFrame,
    participant_state: pd.DataFrame,
    participant_episodes: pd.DataFrame,
    participant_gaps: pd.DataFrame,
    participant_patterns: pd.DataFrame,
    participant_domain_tool: pd.DataFrame,
    participant_quality: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Create compact human-readable case metadata.
    """

    current = _current_state(
        participant_state
    )

    event_kind = (
        participant_events[
            "event_kind"
        ]
        .astype("string")
        .fillna("")
        .str.lower()
        if not participant_events.empty
        else pd.Series(
            dtype="string"
        )
    )

    pattern_entries = []

    for _, row in (
        participant_patterns.iterrows()
    ):

        pattern_entries.append(
            {
                "pattern_id": (
                    row[
                        "pattern_id"
                    ]
                ),

                "pattern_type": (
                    row[
                        "pattern_type"
                    ]
                ),

                "pattern_family": (
                    row[
                        "pattern_family"
                    ]
                ),

                "subject": (
                    row[
                        "subject"
                    ]
                ),

                "detected_at": (
                    row[
                        "detected_at"
                    ]
                ),

                "right_censored": (
                    row[
                        "right_censored"
                    ]
                ),

                "evidence": (
                    _safe_json_load(
                        row[
                            "evidence"
                        ]
                    )
                ),
            }
        )

    domain_summary = {}

    if not participant_domain_tool.empty:

        weighted = (
            participant_domain_tool
            .groupby(
                "domain"
            )[
                "event_weight"
            ]
            .sum()
        )

        domain_summary = {
            str(domain): float(
                value
            )
            for domain, value
            in weighted.items()
        }

    tool_summary = {}

    if not participant_domain_tool.empty:

        tool_counts = (
            participant_domain_tool[
                [
                    "event_id",
                    "tool",
                ]
            ]
            .drop_duplicates()
            .groupby(
                "tool"
            )
            .size()
        )

        tool_summary = {
            str(tool): int(
                value
            )
            for tool, value
            in tool_counts.items()
        }

    quality_entry = {}

    if not participant_quality.empty:

        quality_row = (
            participant_quality
            .iloc[0]
        )

        quality_entry = {
            "core_quality_state": (
                quality_row[
                    "core_quality_state"
                ]
            ),

            "quality_flags": (
                _safe_json_load(
                    quality_row[
                        "quality_flags"
                    ]
                )
            ),

            "observation_flags": (
                _safe_json_load(
                    quality_row[
                        "observation_flags"
                    ]
                )
            ),

            "activity_stream_state": (
                quality_row[
                    "activity_stream_state"
                ]
            ),

            "navigation_stream_state": (
                quality_row[
                    "navigation_stream_state"
                ]
            ),

            "notification_stream_state": (
                quality_row[
                    "notification_stream_state"
                ]
            ),

            "sensor_stream_state": (
                quality_row[
                    "sensor_stream_state"
                ]
            ),

            "garmin_stream_state": (
                quality_row[
                    "garmin_stream_state"
                ]
            ),

            "nutrida_stream_state": (
                quality_row[
                    "nutrida_stream_state"
                ]
            ),

            "credential_extraction_state": (
                quality_row[
                    "credential_extraction_state"
                ]
            ),
        }

    max_gap = None

    if not participant_gaps.empty:

        gap_values = pd.to_numeric(
            participant_gaps[
                "inactive_days"
            ],
            errors="coerce",
        ).dropna()

        if not gap_values.empty:

            max_gap = int(
                gap_values.max()
            )

    return {
        "participant_id": (
            participant_id
        ),

        "current_state": (
            current
        ),

        "event_summary": {
            "normalized_events": int(
                len(
                    participant_events
                )
            ),

            "activity_events": int(
                event_kind.eq(
                    "activity"
                ).sum()
            ),

            "navigation_events": int(
                event_kind.eq(
                    "navigation"
                ).sum()
            ),

            "notification_events": int(
                event_kind.eq(
                    "notification"
                ).sum()
            ),
        },

        "trajectory_summary": {
            "episodes": int(
                len(
                    participant_episodes
                )
            ),

            "inactivity_gaps": int(
                len(
                    participant_gaps
                )
            ),

            "max_inactive_days": (
                max_gap
            ),

            "candidate_patterns": int(
                len(
                    participant_patterns
                )
            ),
        },

        "domain_weighted_events": (
            domain_summary
        ),

        "tool_events": (
            tool_summary
        ),

        "patterns": (
            pattern_entries
        ),

        "data_quality": (
            quality_entry
        ),

        "interpretation_note": (
            "Candidate patterns are exploratory "
            "descriptive signals, not validated "
            "dropout predictions or causal effects."
        ),
    }


def _plot_participant_trajectory(
    participant_id: int,
    participant_state: pd.DataFrame,
    participant_patterns: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot daily explicit engagement and inactivity.

    Top:
        explicit events today
        trailing 7-day explicit events

    Bottom:
        days since last explicit engagement

    Candidate-pattern dates are marked with
    vertical dashed lines.
    """

    if participant_state.empty:
        return

    data = (
        participant_state
        .copy()
    )

    data["date"] = pd.to_datetime(
        data[
            "date"
        ],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=[
                "date"
            ]
        )
        .sort_values(
            "date"
        )
    )

    if data.empty:
        return

    explicit_today = pd.to_numeric(
        data[
            "explicit_events_today"
        ],
        errors="coerce",
    ).fillna(0)

    explicit_7d = pd.to_numeric(
        data[
            "explicit_events_7d"
        ],
        errors="coerce",
    )

    days_since = pd.to_numeric(
        data[
            "days_since_last_explicit_engagement"
        ],
        errors="coerce",
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            14,
            8,
        ),
        sharex=True,
    )

    # -------------------------------------------------
    # Explicit engagement
    # -------------------------------------------------

    axes[0].bar(
        data[
            "date"
        ],
        explicit_today,
        width=1.0,
        label=(
            "Explicit engagement events today"
        ),
    )

    axes[0].plot(
        data[
            "date"
        ],
        explicit_7d,
        label=(
            "Explicit engagement events, "
            "trailing 7 days"
        ),
    )

    axes[0].set_ylabel(
        "Events"
    )

    axes[0].set_title(
        f"Participant {participant_id} "
        "— engagement trajectory"
    )

    axes[0].legend()

    # -------------------------------------------------
    # Inactivity clock
    # -------------------------------------------------

    axes[1].plot(
        data[
            "date"
        ],
        days_since,
        label=(
            "Days since last explicit engagement"
        ),
    )

    axes[1].axhline(
        7,
        linestyle="--",
        label="7-day signal",
    )

    axes[1].axhline(
        14,
        linestyle="--",
        label="14-day signal",
    )

    axes[1].axhline(
        21,
        linestyle=":",
        label="21-day sensitivity threshold",
    )

    axes[1].set_ylabel(
        "Days"
    )

    axes[1].set_xlabel(
        "Date"
    )

    # -------------------------------------------------
    # Candidate pattern dates
    # -------------------------------------------------

    if not participant_patterns.empty:

        pattern_data = (
            participant_patterns
            .copy()
        )

        pattern_data[
            "detected_at"
        ] = pd.to_datetime(
            pattern_data[
                "detected_at"
            ],
            errors="coerce",
        )

        pattern_data = (
            pattern_data
            .dropna(
                subset=[
                    "detected_at"
                ]
            )
        )

        # Avoid excessive annotation if a future case
        # accumulates many patterns.
        pattern_data = (
            pattern_data.head(
                10
            )
        )

        for _, pattern in (
            pattern_data.iterrows()
        ):

            detected_at = (
                pattern[
                    "detected_at"
                ]
            )

            axes[0].axvline(
                detected_at,
                linestyle=":",
                alpha=0.5,
            )

            axes[1].axvline(
                detected_at,
                linestyle=":",
                alpha=0.5,
            )

    axes[1].legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def _plot_domain_tool_trajectory(
    participant_id: int,
    participant_domain_tool: pd.DataFrame,
    participant_state: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot trailing 7-day evidence by behavioral
    domain and tool across the full observation
    period.

    Domain panel:
        explicit intervention engagement only.

    Tool panel:
        all observed activity-provider evidence,
        including Garmin behavioral observations.

    This distinction prevents passive Garmin records
    from being interpreted as active intervention
    engagement.
    """

    if (
        participant_domain_tool.empty
        and participant_state.empty
    ):
        return

    data = (
        participant_domain_tool
        .copy()
    )

    if not data.empty:

        data["date"] = (
            pd.to_datetime(
                data[
                    "date"
                ],
                errors="coerce",
            )
        )

        data[
            "event_weight"
        ] = pd.to_numeric(
            data[
                "event_weight"
            ],
            errors="coerce",
        ).fillna(0)

        data = data.dropna(
            subset=[
                "date"
            ]
        )

    # -------------------------------------------------
    # Use the ParticipantState window, not merely
    # first/last domain activity.
    #
    # This ensures that disappearance remains visible
    # all the way to the analysis cutoff.
    # -------------------------------------------------

    state = (
        participant_state
        .copy()
    )

    if not state.empty:

        state["date"] = (
            pd.to_datetime(
                state[
                    "date"
                ],
                errors="coerce",
            )
        )

        state_dates = (
            state[
                "date"
            ]
            .dropna()
        )

    else:

        state_dates = pd.Series(
            dtype="datetime64[ns]"
        )

    if not state_dates.empty:

        start = (
            state_dates.min()
        )

        end = (
            state_dates.max()
        )

    elif not data.empty:

        start = (
            data[
                "date"
            ].min()
        )

        end = (
            data[
                "date"
            ].max()
        )

    else:

        return

    full_dates = (
        pd.date_range(
            start=start,
            end=end,
            freq="D",
        )
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            14,
            8,
        ),
        sharex=True,
    )

    # =================================================
    # DOMAIN PANEL
    # Explicit engagement only
    # =================================================

    if not data.empty:

        domain_data = (
            data.loc[
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
            ]
            .copy()
        )

    else:

        domain_data = pd.DataFrame()

    if not domain_data.empty:

        domain_daily = (
            domain_data.groupby(
                [
                    "date",
                    "domain",
                ]
            )[
                "event_weight"
            ]
            .sum()
            .unstack(
                fill_value=0
            )
            .reindex(
                full_dates,
                fill_value=0,
            )
        )

        domain_rolling = (
            domain_daily
            .rolling(
                window=7,
                min_periods=1,
            )
            .sum()
        )

        for domain in (
            domain_rolling.columns
        ):

            axes[0].plot(
                domain_rolling.index,
                domain_rolling[
                    domain
                ],
                label=str(
                    domain
                ),
            )

        axes[0].legend()

    axes[0].set_ylabel(
        "Explicit weighted events\n"
        "(trailing 7 days)"
    )

    axes[0].set_title(
        f"Participant {participant_id} "
        "— domain/tool trajectory"
    )

    # =================================================
    # TOOL PANEL
    # All observed activity providers
    # =================================================

    if not data.empty:

        tool_events = (
            data[
                [
                    "event_id",
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

    else:

        tool_events = pd.DataFrame()

    if not tool_events.empty:

        tool_daily = (
            tool_events.groupby(
                [
                    "date",
                    "tool",
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
            .reindex(
                full_dates,
                fill_value=0,
            )
        )

        tool_rolling = (
            tool_daily
            .rolling(
                window=7,
                min_periods=1,
            )
            .sum()
        )

        for tool in (
            tool_rolling.columns
        ):

            axes[1].plot(
                tool_rolling.index,
                tool_rolling[
                    tool
                ],
                label=str(
                    tool
                ),
            )

        axes[1].legend()

    axes[1].set_ylabel(
        "Activity-provider events\n"
        "(trailing 7 days)"
    )

    axes[1].set_xlabel(
        "Date"
    )

    # Explicitly span the complete observation window.
    axes[0].set_xlim(
        start,
        end,
    )

    axes[1].set_xlim(
        start,
        end,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def _plot_cohort_engagement(
    state: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot the number of participants in each
    engagement state over time.
    """

    if state.empty:
        return

    data = state.copy()

    data["date"] = pd.to_datetime(
        data[
            "date"
        ],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "date"
        ]
    )

    if data.empty:
        return

    cohort = (
        data.groupby(
            [
                "date",
                "engagement_state",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    state_order = [
        "no_explicit_engagement_observed_yet",
        "active",
        "quiet",
        "prolonged_inactivity_7d",
        "prolonged_inactivity_14d",
        "unavailable",
    ]

    fig, ax = plt.subplots(
        figsize=(
            14,
            6,
        )
    )

    for state_name in (
        state_order
    ):

        if (
            state_name
            not in cohort.columns
        ):
            continue

        ax.plot(
            cohort.index,
            cohort[
                state_name
            ],
            label=state_name,
        )

    ax.set_title(
        "Cohort engagement state over time"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Participants"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def run_case_export(
    config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
    """
    Build compact diversified case exports
    and trajectory plots.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    # -------------------------------------------------
    # Build all upstream layers.
    # -------------------------------------------------

    events = build_normalized_events(
        config
    )

    episodes = (
        build_participation_episodes(
            events,
            config,
        )
    )

    state = (
        build_participant_state_daily(
            config,
            events,
        )
    )

    gaps = build_inactivity_gaps(
        config,
        events,
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

    # -------------------------------------------------
    # Output folders.
    # -------------------------------------------------

    cases_dir = os.path.join(
        config.output_dir,
        "cases",
    )

    plots_dir = os.path.join(
        config.output_dir,
        "plots",
    )

    # -------------------------------------------------
    # Case selection can change between audit runs.
    #
    # Remove the previous case bundle first so stale
    # participant folders are never mistaken for cases
    # selected by the current run.
    # -------------------------------------------------

    if os.path.isdir(
            cases_dir
    ):
        shutil.rmtree(
            cases_dir
        )

    os.makedirs(
        cases_dir,
        exist_ok=True,
    )

    os.makedirs(
        plots_dir,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Cohort-level plot.
    # -------------------------------------------------

    cohort_plot_path = (
        os.path.join(
            plots_dir,
            "cohort_engagement_over_time.png",
        )
    )

    _plot_cohort_engagement(
        state,
        cohort_plot_path,
    )

    # -------------------------------------------------
    # Select diverse participant cases.
    # -------------------------------------------------

    selected = (
        select_case_participants(
            patterns,
            config.case_export_max_cases,
        )
    )

    index_rows = []

    for selection_index, (
        selection
    ) in selected.iterrows():

        participant_id = int(
            selection[
                "participant_id"
            ]
        )

        selection_rank = (
            selection_index
            + 1
        )

        selection_reason = (
            selection[
                "selection_reason"
            ]
        )

        participant_events = (
            _filter_participant(
                events,
                participant_id,
            )
        )

        participant_state = (
            _filter_participant(
                state,
                participant_id,
            )
        )

        participant_episodes = (
            _filter_participant(
                episodes,
                participant_id,
            )
        )

        participant_gaps = (
            _filter_participant(
                gaps,
                participant_id,
            )
        )

        participant_patterns = (
            _filter_participant(
                patterns,
                participant_id,
            )
        )

        participant_domain_tool = (
            _filter_participant(
                domain_tool,
                participant_id,
            )
        )

        participant_quality = (
            _filter_participant(
                quality,
                participant_id,
            )
        )

        # ---------------------------------------------
        # Participant directory.
        # ---------------------------------------------

        case_dir = os.path.join(
            cases_dir,
            f"participant_{participant_id}",
        )

        os.makedirs(
            case_dir,
            exist_ok=True,
        )

        # ---------------------------------------------
        # CSV slices.
        # ---------------------------------------------

        exports = {
            "normalized_events.csv": (
                participant_events
            ),

            "participant_state_daily.csv": (
                participant_state
            ),

            "participation_episodes.csv": (
                participant_episodes
            ),

            "inactivity_gaps.csv": (
                participant_gaps
            ),

            "candidate_patterns.csv": (
                participant_patterns
            ),

            "domain_tool_engagement.csv": (
                participant_domain_tool
            ),
        }

        for (
            filename,
            dataframe,
        ) in exports.items():

            dataframe.to_csv(
                os.path.join(
                    case_dir,
                    filename,
                ),
                index=False,
            )

        # ---------------------------------------------
        # Summary JSON.
        # ---------------------------------------------

        summary = _case_summary(
            participant_id,
            participant_events,
            participant_state,
            participant_episodes,
            participant_gaps,
            participant_patterns,
            participant_domain_tool,
            participant_quality,
        )

        summary[
            "selection_reason"
        ] = selection_reason

        summary[
            "selection_rank"
        ] = selection_rank

        summary_path = (
            os.path.join(
                case_dir,
                "summary.json",
            )
        )

        write_json(
            summary_path,
            summary,
        )

        # ---------------------------------------------
        # Plots.
        # ---------------------------------------------

        trajectory_plot = (
            os.path.join(
                case_dir,
                "trajectory.png",
            )
        )

        domain_tool_plot = (
            os.path.join(
                case_dir,
                "domain_tool_trajectory.png",
            )
        )

        _plot_participant_trajectory(
            participant_id,
            participant_state,
            participant_patterns,
            trajectory_plot,
        )

        _plot_domain_tool_trajectory(
            participant_id,
            participant_domain_tool,
            participant_state,
            domain_tool_plot,
        )

        # ---------------------------------------------
        # Case index information.
        # ---------------------------------------------

        current = _current_state(
            participant_state
        )

        pattern_types = (
            participant_patterns[
                "pattern_type"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        max_gap = None

        if not participant_gaps.empty:

            values = pd.to_numeric(
                participant_gaps[
                    "inactive_days"
                ],
                errors="coerce",
            ).dropna()

            if not values.empty:

                max_gap = int(
                    values.max()
                )

        quality_row = (
            participant_quality.iloc[0]
            if not participant_quality.empty
            else None
        )

        index_rows.append(
            {
                "selection_rank": (
                    selection_rank
                ),

                "participant_id": (
                    participant_id
                ),

                "selection_reason": (
                    selection_reason
                ),

                "pattern_count": int(
                    len(
                        participant_patterns
                    )
                ),

                "pattern_types": (
                    json.dumps(
                        pattern_types,
                        ensure_ascii=False,
                    )
                ),

                "current_engagement_state": (
                    current[
                        "engagement_state"
                    ]
                ),

                "days_since_last_explicit_engagement": (
                    current[
                        "days_since_last_explicit_engagement"
                    ]
                ),

                "explicit_engagement_events": (
                    int(
                        quality_row[
                            "explicit_engagement_events"
                        ]
                    )
                    if quality_row is not None
                    else pd.NA
                ),

                "navigation_events": (
                    int(
                        quality_row[
                            "navigation_events"
                        ]
                    )
                    if quality_row is not None
                    else pd.NA
                ),

                "behavioral_sensor_events": (
                    int(
                        quality_row[
                            "behavioral_sensor_events"
                        ]
                    )
                    if quality_row is not None
                    else pd.NA
                ),

                "episode_count": int(
                    len(
                        participant_episodes
                    )
                ),

                "gap_count": int(
                    len(
                        participant_gaps
                    )
                ),

                "max_inactive_days": (
                    max_gap
                ),

                "case_directory": (
                    os.path.relpath(
                        case_dir,
                        config.output_dir,
                    )
                ),

                "trajectory_plot": (
                    os.path.relpath(
                        trajectory_plot,
                        config.output_dir,
                    )
                ),

                "domain_tool_plot": (
                    os.path.relpath(
                        domain_tool_plot,
                        config.output_dir,
                    )
                    if os.path.exists(
                        domain_tool_plot
                    )
                    else None
                ),
            }
        )

    case_index = pd.DataFrame(
        index_rows
    )

    for column in (
        CASE_INDEX_COLUMNS
    ):

        if (
            column
            not in case_index.columns
        ):

            case_index[
                column
            ] = pd.NA

    case_index = case_index[
        CASE_INDEX_COLUMNS
    ]

    case_index_path = (
        os.path.join(
            config.output_dir,
            "case_index.csv",
        )
    )

    case_index.to_csv(
        case_index_path,
        index=False,
    )

    return case_index


if __name__ == "__main__":

    case_index = (
        run_case_export()
    )

    print(
        "Case export written successfully."
    )

    print()

    print(
        "Cases exported:",
        len(
            case_index
        ),
    )

    if not case_index.empty:

        print()

        print(
            "Selected participants:"
        )

        for _, row in (
            case_index.iterrows()
        ):

            print(
                f"  #{int(row['selection_rank'])} "
                f"participant "
                f"{int(row['participant_id'])}: "
                f"{row['selection_reason']}"
            )

        print()

        print(
            "Case index:"
        )

        print(
            os.path.join(
                TrajectoryAuditConfig()
                .output_dir,
                "case_index.csv",
            )
        )