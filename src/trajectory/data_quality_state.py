from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_export,
    write_json,
)

from src.trajectory.domain_tool_engagement import (
    build_domain_tool_engagement,
)

from src.trajectory.engagement import (
    behavioral_sensor_mask,
    explicit_engagement_mask,
)

from src.trajectory.event_normalization import (
    build_normalized_events,
)

from src.trajectory.observation_window import (
    _export_cohort,
    build_observation_window,
)


QUALITY_COLUMNS = [
    "participant_id",

    "core_quality_state",
    "quality_flags",
    "observation_flags",

    "has_any_event",
    "has_explicit_engagement",

    "total_events",
    "activity_events",
    "explicit_engagement_events",
    "navigation_events",
    "notification_events",
    "behavioral_sensor_events",
    "garmin_events",
    "nutrida_events",

    "first_observed_at",
    "last_observed_at",

    "dedup_conflict_events",
    "unmapped_domain_events",
    "multi_domain_events",

    "activity_stream_state",
    "navigation_stream_state",
    "notification_stream_state",
    "sensor_stream_state",
    "garmin_stream_state",
    "nutrida_stream_state",
    "credential_extraction_state",
]


def _json_list(
    values: List[str],
) -> str:
    """
    Store flags as stable JSON arrays.
    """

    return json.dumps(
        sorted(set(values)),
        ensure_ascii=False,
    )


def _bool_series(
    series: pd.Series,
) -> pd.Series:
    """
    Robust conversion of CSV/pandas booleans.
    """

    return (
        series
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def _credential_extraction_state(
    config: TrajectoryAuditConfig,
) -> str:
    """
    Check whether participant-specific extracted JSON
    already exists.

    This does NOT authenticate or use users.xlsx.
    """

    files = glob.glob(
        os.path.join(
            config.raw_data_dir,
            "*.json",
        )
    )

    return (
        "available"
        if files
        else "unavailable"
    )


def _provider_state(
    domain_tool: pd.DataFrame,
    activity_stream_state: str,
    tool: str,
) -> str:
    """
    Determine whether a provider-specific stream
    is represented in the current campaign export.

    Missing provider != zero participant behavior.
    """

    if activity_stream_state == "unavailable":
        return "unavailable"

    if domain_tool.empty:
        return "unavailable"

    if (
        domain_tool["tool"]
        .astype("string")
        .eq(tool)
        .any()
    ):
        return "available"

    return "unavailable"


def _prepare_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize event fields used in quality analysis.
    """

    if events.empty:
        return events.copy()

    result = events.copy()

    result["participant_id"] = pd.to_numeric(
        result["participant_id"],
        errors="coerce",
    ).astype("Int64")

    result["occurred_at"] = pd.to_datetime(
        result["occurred_at"],
        utc=True,
        errors="coerce",
    )

    return result


def _participant_counts(
    participant_events: pd.DataFrame,
) -> Dict[str, int]:
    """
    Count major event channels for one participant.
    """

    if participant_events.empty:

        return {
            "total_events": 0,
            "activity_events": 0,
            "explicit_engagement_events": 0,
            "navigation_events": 0,
            "notification_events": 0,
            "behavioral_sensor_events": 0,
            "garmin_events": 0,
            "nutrida_events": 0,
        }

    event_kind = (
        participant_events[
            "event_kind"
        ]
        .astype("string")
        .fillna("")
        .str.lower()
    )

    provider = (
        participant_events[
            "provider"
        ]
        .astype("string")
        .fillna("")
        .str.lower()
    )

    explicit = (
        explicit_engagement_mask(
            participant_events
        )
    )

    behavioral_sensor = (
        behavioral_sensor_mask(
            participant_events
        )
    )

    return {
        "total_events": int(
            len(participant_events)
        ),

        "activity_events": int(
            event_kind.eq(
                "activity"
            ).sum()
        ),

        "explicit_engagement_events": int(
            explicit.sum()
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

        "behavioral_sensor_events": int(
            behavioral_sensor.sum()
        ),

        "garmin_events": int(
            provider.str.contains(
                "garmin",
                na=False,
            ).sum()
        ),

        "nutrida_events": int(
            provider.str.contains(
                "nutrida",
                na=False,
            ).sum()
        ),
    }


def _participant_observation_flags(
    counts: Dict[str, int],
) -> List[str]:
    """
    Describe participant evidence without treating
    non-engagement as a data error.
    """

    flags: List[str] = []

    if counts[
        "total_events"
    ] == 0:

        flags.append(
            "no_observed_events"
        )

        return flags

    if counts[
        "explicit_engagement_events"
    ] == 0:

        flags.append(
            "no_explicit_engagement"
        )

    if (
        counts[
            "navigation_events"
        ] > 0
        and counts[
            "explicit_engagement_events"
        ] == 0
    ):

        flags.append(
            "navigation_without_"
            "explicit_engagement"
        )

    if (
        counts[
            "behavioral_sensor_events"
        ] > 0
        and counts[
            "explicit_engagement_events"
        ] == 0
    ):

        flags.append(
            "behavioral_sensor_without_"
            "explicit_engagement"
        )

    return flags


def _core_quality_state(
    activity_stream_state: str,
    cohort_available: bool,
    quality_flags: List[str],
) -> str:
    """
    Determine whether the participant evidence is
    sufficient for core engagement trajectory analysis.

    Optional enrichment streams are deliberately
    NOT used to downgrade this status.
    """

    if (
        activity_stream_state
        == "unavailable"
        or not cohort_available
    ):

        return (
            "insufficient_for_core_trajectory"
        )

    if quality_flags:

        return (
            "core_trajectory_with_cautions"
        )

    return (
        "sufficient_for_core_trajectory"
    )


def build_data_quality_state(
    config: TrajectoryAuditConfig,
    events: pd.DataFrame,
    domain_tool: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Build participant-level and campaign-level
    data-quality state.
    """

    prepared = _prepare_events(
        events
    )

    export_data, stream_info = (
        load_campaign_export(
            config.campaign_data_path
        )
    )

    cohort = _export_cohort(
        config,
        prepared,
    )

    window = build_observation_window(
        config,
        prepared,
    )

    participant_ids = cohort[
        "participant_ids"
    ]

    cohort_available = (
        cohort["source"]
        != "unavailable"
    )

    activity_state = (
        stream_info[
            "activities"
        ]["status"]
    )

    navigation_state = (
        stream_info[
            "navigation"
        ]["status"]
    )

    notification_state = (
        stream_info[
            "notification_events"
        ]["status"]
    )

    sensor_state = (
        stream_info[
            "sensor_events"
        ]["status"]
    )

    garmin_state = _provider_state(
        domain_tool,
        activity_state,
        "garmin",
    )

    nutrida_state = _provider_state(
        domain_tool,
        activity_state,
        "nutrida",
    )

    credential_state = (
        _credential_extraction_state(
            config
        )
    )

    # -------------------------------------------------
    # Event-level quality diagnostics
    # -------------------------------------------------

    if (
        "dedup_conflict"
        in prepared.columns
    ):

        dedup_conflicts = _bool_series(
            prepared[
                "dedup_conflict"
            ]
        )

    else:

        dedup_conflicts = pd.Series(
            False,
            index=prepared.index,
        )

    missing_participant = int(
        prepared[
            "participant_id"
        ]
        .isna()
        .sum()
    )

    missing_timestamp = int(
        prepared[
            "occurred_at"
        ]
        .isna()
        .sum()
    )

    total_dedup_conflicts = int(
        dedup_conflicts.sum()
    )

    # -------------------------------------------------
    # Domain diagnostics
    # -------------------------------------------------

    if domain_tool.empty:

        unmapped_event_ids = set()
        multi_domain_event_ids = set()

    else:

        unmapped_event_ids = set(
            domain_tool.loc[
                domain_tool[
                    "domain"
                ]
                == "unmapped",
                "event_id",
            ]
            .dropna()
            .astype(str)
        )

        multi_domain_event_ids = set(
            domain_tool.loc[
                domain_tool[
                    "domain_membership_count"
                ]
                > 1,
                "event_id",
            ]
            .dropna()
            .astype(str)
        )

    # -------------------------------------------------
    # Participant rows
    # -------------------------------------------------

    rows = []

    for participant_id in (
        participant_ids
    ):

        participant_events = (
            prepared.loc[
                prepared[
                    "participant_id"
                ]
                == participant_id
            ]
            .copy()
        )

        counts = _participant_counts(
            participant_events
        )

        # ---------------------------------------------
        # Participant quality flags
        # ---------------------------------------------

        quality_flags: List[str] = []

        participant_conflicts = 0

        if not participant_events.empty:

            participant_conflicts = int(
                _bool_series(
                    participant_events[
                        "dedup_conflict"
                    ]
                ).sum()
            )

        if participant_conflicts > 0:

            quality_flags.append(
                "deduplication_conflict"
            )

        participant_event_ids = set(
            participant_events[
                "event_id"
            ]
            .dropna()
            .astype(str)
        )

        participant_unmapped = len(
            participant_event_ids
            & unmapped_event_ids
        )

        participant_multi_domain = len(
            participant_event_ids
            & multi_domain_event_ids
        )

        if participant_unmapped > 0:

            quality_flags.append(
                "unmapped_domain_events"
            )

        # NOTE:
        # multi-domain events are NOT a quality error.
        # They are retained as a descriptive field only.

        observation_flags = (
            _participant_observation_flags(
                counts
            )
        )

        # ---------------------------------------------
        # Observed range
        # ---------------------------------------------

        timestamps = (
            participant_events[
                "occurred_at"
            ]
            .dropna()
        )

        if timestamps.empty:

            first_observed = None
            last_observed = None

        else:

            first_observed = (
                timestamps.min()
            )

            last_observed = (
                timestamps.max()
            )

        rows.append(
            {
                "participant_id": (
                    participant_id
                ),

                "core_quality_state": (
                    _core_quality_state(
                        activity_state,
                        cohort_available,
                        quality_flags,
                    )
                ),

                "quality_flags": (
                    _json_list(
                        quality_flags
                    )
                ),

                "observation_flags": (
                    _json_list(
                        observation_flags
                    )
                ),

                "has_any_event": (
                    counts[
                        "total_events"
                    ] > 0
                ),

                "has_explicit_engagement": (
                    counts[
                        "explicit_engagement_events"
                    ] > 0
                ),

                **counts,

                "first_observed_at": (
                    first_observed
                ),

                "last_observed_at": (
                    last_observed
                ),

                "dedup_conflict_events": (
                    participant_conflicts
                ),

                "unmapped_domain_events": (
                    participant_unmapped
                ),

                "multi_domain_events": (
                    participant_multi_domain
                ),

                "activity_stream_state": (
                    activity_state
                ),

                "navigation_stream_state": (
                    navigation_state
                ),

                "notification_stream_state": (
                    notification_state
                ),

                "sensor_stream_state": (
                    sensor_state
                ),

                "garmin_stream_state": (
                    garmin_state
                ),

                "nutrida_stream_state": (
                    nutrida_state
                ),

                "credential_extraction_state": (
                    credential_state
                ),
            }
        )

    quality = pd.DataFrame(
        rows
    )

    if not quality.empty:

        quality[
            "first_observed_at"
        ] = pd.to_datetime(
            quality[
                "first_observed_at"
            ],
            utc=True,
            errors="coerce",
        )

        quality[
            "last_observed_at"
        ] = pd.to_datetime(
            quality[
                "last_observed_at"
            ],
            utc=True,
            errors="coerce",
        )

    # -------------------------------------------------
    # Campaign-level warnings
    # -------------------------------------------------

    campaign_flags: List[str] = []

    if activity_state == "unavailable":

        campaign_flags.append(
            "activity_stream_unavailable"
        )

    if not cohort_available:

        campaign_flags.append(
            "cohort_unavailable"
        )

    if total_dedup_conflicts > 0:

        campaign_flags.append(
            "deduplication_conflicts_present"
        )

    if missing_participant > 0:

        campaign_flags.append(
            "events_missing_participant_id"
        )

    if missing_timestamp > 0:

        campaign_flags.append(
            "events_missing_timestamp"
        )

    configured_start = (
        window["configured_start"]
    )

    effective_start = (
        window["effective_start"]
    )

    if (
        configured_start is not None
        and effective_start is not None
        and effective_start
        > configured_start
    ):

        campaign_flags.append(
            "configured_start_adjusted"
        )

    optional_unavailable = []

    if garmin_state == "unavailable":
        optional_unavailable.append(
            "garmin"
        )

    if nutrida_state == "unavailable":
        optional_unavailable.append(
            "nutrida"
        )

    if credential_state == "unavailable":
        optional_unavailable.append(
            "credential_extraction"
        )

    summary: Dict[str, Any] = {
        "cohort": {
            "source": cohort["source"],
            "participants": int(
                len(participant_ids)
            ),
        },

        "observation_window": {
            "configured_start": (
                configured_start
            ),

            "effective_start": (
                effective_start
            ),

            "analysis_cutoff": (
                window[
                    "analysis_cutoff"
                ]
            ),

            "start_source": (
                window[
                    "start_source"
                ]
            ),

            "cutoff_source": (
                window[
                    "cutoff_source"
                ]
            ),
        },

        "streams": {
            "activity": activity_state,
            "navigation": navigation_state,
            "notification": (
                notification_state
            ),
            "sensor": sensor_state,
            "garmin": garmin_state,
            "nutrida": nutrida_state,
            "credential_extraction": (
                credential_state
            ),
        },

        "event_quality": {
            "normalized_events": int(
                len(prepared)
            ),

            "events_missing_participant_id": (
                missing_participant
            ),

            "events_missing_timestamp": (
                missing_timestamp
            ),

            "dedup_conflict_events": (
                total_dedup_conflicts
            ),

            "unmapped_domain_events": int(
                len(
                    unmapped_event_ids
                )
            ),

            "multi_domain_events": int(
                len(
                    multi_domain_event_ids
                )
            ),
        },

        "campaign_quality_flags": (
            campaign_flags
        ),

        "optional_enrichment_unavailable": (
            optional_unavailable
        ),

        "interpretation": {
            "core_trajectory": (
                "Activity export and campaign cohort "
                "are sufficient for core engagement "
                "trajectory analysis."
                if (
                    activity_state
                    != "unavailable"
                    and cohort_available
                )
                else (
                    "Core engagement trajectory "
                    "analysis is not fully supported "
                    "by the available data."
                )
            ),

            "optional_stream_rule": (
                "Unavailable optional streams are "
                "treated as unavailable, not as "
                "zero participant behavior."
            ),

            "non_engagement_rule": (
                "No explicit engagement is an "
                "observation, not automatically a "
                "data-quality problem."
            ),
        },
    }

    # -------------------------------------------------
    # Stable participant schema
    # -------------------------------------------------

    for column in QUALITY_COLUMNS:

        if column not in (
            quality.columns
        ):

            quality[column] = pd.NA

    quality = quality[
        QUALITY_COLUMNS
    ]

    return (
        quality,
        summary,
    )


def run_data_quality_state(
    config: TrajectoryAuditConfig | None = None,
) -> tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Build and save participant and campaign
    data-quality outputs.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    events = build_normalized_events(
        config
    )

    domain_tool = (
        build_domain_tool_engagement(
            config,
            events,
        )
    )

    quality, summary = (
        build_data_quality_state(
            config,
            events,
            domain_tool,
        )
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    quality_path = os.path.join(
        config.output_dir,
        "data_quality.csv",
    )

    summary_path = os.path.join(
        config.output_dir,
        "data_quality_summary.json",
    )

    quality.to_csv(
        quality_path,
        index=False,
    )

    write_json(
        summary_path,
        summary,
    )

    return (
        quality,
        summary,
    )


if __name__ == "__main__":

    quality, summary = (
        run_data_quality_state()
    )

    print(
        "Data quality state "
        "written successfully."
    )

    print()

    print(
        "Participants:",
        len(quality),
    )

    if not quality.empty:

        print()

        print(
            "Core quality states:"
        )

        counts = (
            quality[
                "core_quality_state"
            ]
            .value_counts(
                dropna=False
            )
        )

        for state, count in (
            counts.items()
        ):

            print(
                f"  {state}: {count}"
            )

        print()

        print(
            "Participant observation flags:"
        )

        flag_counts: Dict[
            str,
            int,
        ] = {}

        for raw_flags in (
            quality[
                "observation_flags"
            ]
        ):

            try:
                flags = json.loads(
                    raw_flags
                )
            except Exception:
                flags = []

            for flag in flags:

                flag_counts[flag] = (
                    flag_counts.get(
                        flag,
                        0,
                    )
                    + 1
                )

        if flag_counts:

            for (
                flag,
                count,
            ) in sorted(
                flag_counts.items()
            ):

                print(
                    f"  {flag}: {count}"
                )

        else:

            print("  none")

        print()

    print(
        "Campaign quality flags:",
        summary[
            "campaign_quality_flags"
        ],
    )

    print(
        "Optional enrichment unavailable:",
        summary[
            "optional_enrichment_unavailable"
        ],
    )