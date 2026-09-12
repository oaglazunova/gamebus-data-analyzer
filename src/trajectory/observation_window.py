from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_desc,
    load_campaign_export,
)

from src.trajectory.event_normalization import (
    build_normalized_events,
)


def _configured_campaign_window(
    config: TrajectoryAuditConfig,
) -> Dict[str, Any]:
    """
    Read configured wave start/end dates.

    We use:
        earliest wave start
        latest wave end

    These are campaign-design dates, not necessarily
    the dates on which usable participant data begins.
    """

    desc, _ = load_campaign_desc(
        config.campaign_desc_path
    )

    waves = desc.get(
        "waves",
        pd.DataFrame(),
    )

    result = {
        "configured_start": None,
        "configured_end": None,
    }

    if waves.empty:
        return result

    if "start" in waves.columns:

        starts = pd.to_datetime(
            waves["start"],
            utc=True,
            errors="coerce",
        ).dropna()

        if not starts.empty:
            result["configured_start"] = (
                starts.min().floor("D")
            )

    if "end" in waves.columns:

        ends = pd.to_datetime(
            waves["end"],
            utc=True,
            errors="coerce",
        ).dropna()

        if not ends.empty:
            result["configured_end"] = (
                ends.max().floor("D")
            )

    return result


def _observed_data_window(
    events: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Determine first and last timestamps for any
    normalized participant event.

    This includes:
        activities
        navigation
        notifications
        sensors

    because the question here is:

        "When do we actually observe participant data?"

    It is NOT yet asking:

        "When did explicit engagement occur?"
    """

    if (
        events.empty
        or "occurred_at" not in events.columns
    ):
        return {
            "first_observed": None,
            "last_observed": None,
        }

    timestamps = pd.to_datetime(
        events["occurred_at"],
        utc=True,
        errors="coerce",
    ).dropna()

    if timestamps.empty:
        return {
            "first_observed": None,
            "last_observed": None,
        }

    return {
        "first_observed": (
            timestamps.min().floor("D")
        ),
        "last_observed": (
            timestamps.max().floor("D")
        ),
    }


def _export_cohort(
    config: TrajectoryAuditConfig,
    events: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build the participant cohort.

    Preferred source:
        1-aggregated-data.csv

    This is important because it can contain
    participants with zero activities.

    Fallback:
        participant IDs observed in normalized events.
    """

    export_data, _ = load_campaign_export(
        config.campaign_data_path
    )

    aggregation = export_data.get(
        "aggregation",
        pd.DataFrame(),
    )

    if (
        not aggregation.empty
        and "pid" in aggregation.columns
    ):

        ids = (
            pd.to_numeric(
                aggregation["pid"],
                errors="coerce",
            )
            .dropna()
            .astype("int64")
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        return {
            "source": "campaign_export",
            "participant_ids": ids,
        }

    if (
        not events.empty
        and "participant_id" in events.columns
    ):

        ids = (
            pd.to_numeric(
                events["participant_id"],
                errors="coerce",
            )
            .dropna()
            .astype("int64")
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        return {
            "source": "observed_events",
            "participant_ids": ids,
        }

    return {
        "source": "unavailable",
        "participant_ids": [],
    }


def build_observation_window(
    config: TrajectoryAuditConfig,
    events: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Determine the effective longitudinal observation window.

    START
    -----
    Normally use the first configured wave start.

    However, some historical campaigns contain stale
    configuration dates. If participant data only begins
    much later than the configured start, using the old
    date would manufacture a long period of apparent
    non-engagement.

    Therefore:

        configured start
            |
            | data starts <= grace period later
            v
        keep configured start

    otherwise:

        use first observed data date


    END / CUTOFF
    ------------
    By default use the latest observed event date.

    This is intentionally conservative.

    It prevents a running campaign whose configured end
    is in the future from generating artificial future
    inactivity.

    A manual analysis_cutoff can override this later.
    """

    configured = _configured_campaign_window(
        config
    )

    observed = _observed_data_window(
        events
    )

    configured_start = configured[
        "configured_start"
    ]

    configured_end = configured[
        "configured_end"
    ]

    first_observed = observed[
        "first_observed"
    ]

    last_observed = observed[
        "last_observed"
    ]

    # -------------------------------------------------
    # Effective observation start
    # -------------------------------------------------

    effective_start = None
    start_source = None
    start_shift_days = None

    if configured_start is not None:

        if first_observed is None:

            effective_start = configured_start
            start_source = "configured_first_wave_start"

        else:

            start_shift_days = int(
                (
                    first_observed
                    - configured_start
                ).days
            )

            if (
                start_shift_days
                <= config.observation_start_grace_days
            ):

                effective_start = configured_start

                start_source = (
                    "configured_first_wave_start"
                )

            else:

                effective_start = first_observed

                start_source = (
                    "first_observed_event_"
                    "because_configured_start_"
                    "appears_stale"
                )

    elif first_observed is not None:

        effective_start = first_observed
        start_source = "first_observed_event"

    # -------------------------------------------------
    # Analysis cutoff
    # -------------------------------------------------

    if config.analysis_cutoff is not None:

        analysis_cutoff = pd.to_datetime(
            config.analysis_cutoff,
            utc=True,
            errors="raise",
        ).floor("D")

        cutoff_source = "manual_override"

    else:

        analysis_cutoff = last_observed
        cutoff_source = (
            "latest_observed_event"
            if last_observed is not None
            else None
        )

    # -------------------------------------------------
    # Guard against an impossible window
    # -------------------------------------------------

    if (
        effective_start is not None
        and analysis_cutoff is not None
        and analysis_cutoff < effective_start
    ):
        raise ValueError(
            "Analysis cutoff occurs before "
            "effective observation start."
        )

    return {
        "configured_start": configured_start,
        "configured_end": configured_end,

        "first_observed": first_observed,
        "last_observed": last_observed,

        "effective_start": effective_start,
        "start_source": start_source,

        "configured_to_observed_start_gap_days": (
            start_shift_days
        ),

        "analysis_cutoff": analysis_cutoff,
        "cutoff_source": cutoff_source,

        "observation_days": (
            int(
                (
                    analysis_cutoff
                    - effective_start
                ).days
                + 1
            )
            if (
                effective_start is not None
                and analysis_cutoff is not None
            )
            else None
        ),
    }


def run_observation_window(
    config: TrajectoryAuditConfig | None = None,
) -> Dict[str, Any]:
    """
    Convenience runner for inspecting the window
    before building ParticipantState(t).
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    events = build_normalized_events(
        config
    )

    window = build_observation_window(
        config,
        events,
    )

    cohort = _export_cohort(
        config,
        events,
    )

    return {
        "window": window,
        "cohort": cohort,
    }


def _date_text(
    value: Any,
) -> str:
    """
    Pretty-print pandas timestamps.
    """

    if value is None:
        return "unavailable"

    return str(
        pd.Timestamp(value).date()
    )


if __name__ == "__main__":

    result = run_observation_window()

    window = result["window"]
    cohort = result["cohort"]

    print("Trajectory observation window")
    print()

    print(
        "Configured start:",
        _date_text(
            window["configured_start"]
        ),
    )

    print(
        "First observed event:",
        _date_text(
            window["first_observed"]
        ),
    )

    print(
        "Effective start:",
        _date_text(
            window["effective_start"]
        ),
    )

    print(
        "Start source:",
        window["start_source"],
    )

    print()

    print(
        "Configured end:",
        _date_text(
            window["configured_end"]
        ),
    )

    print(
        "Last observed event:",
        _date_text(
            window["last_observed"]
        ),
    )

    print(
        "Analysis cutoff:",
        _date_text(
            window["analysis_cutoff"]
        ),
    )

    print(
        "Cutoff source:",
        window["cutoff_source"],
    )

    print()

    print(
        "Observation days:",
        window["observation_days"],
    )

    print(
        "Cohort source:",
        cohort["source"],
    )

    print(
        "Participants in cohort:",
        len(
            cohort["participant_ids"]
        ),
    )