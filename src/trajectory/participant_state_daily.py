from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_export,
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


STATE_COLUMNS = [
    "participant_id",
    "date",

    "engagement_state",
    "has_ever_engaged",

    "last_explicit_engagement_date",
    "days_since_last_explicit_engagement",

    "episode_number",
    "episode_id",

    "explicit_events_today",
    "explicit_events_7d",
    "explicit_events_14d",

    "active_days_7d",
    "active_days_14d",

    "points_today",
    "points_7d",
    "points_14d",

    "navigation_events_today",
    "notification_events_today",

    "garmin_events_today",
    "nutrida_events_today",
    "sensor_events_today",
    "behavioral_sensor_events_today",

    "activity_stream_state",
    "navigation_stream_state",
    "notification_stream_state",
    "sensor_stream_state",

    "garmin_stream_state",
    "nutrida_stream_state",
]


def _prepare_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize participant IDs and timestamps
    before daily aggregation.
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

    result = result.dropna(
        subset=[
            "participant_id",
            "occurred_at",
        ]
    )

    result["date"] = (
        result["occurred_at"]
        .dt.floor("D")
    )

    result["points"] = pd.to_numeric(
        result["points"],
        errors="coerce",
    ).fillna(0)

    return result


def _build_daily_grid(
    participant_ids: list[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build one row per participant per observed day.
    """

    dates = pd.date_range(
        start=start,
        end=end,
        freq="D",
        tz="UTC",
    )

    index = pd.MultiIndex.from_product(
        [
            participant_ids,
            dates,
        ],
        names=[
            "participant_id",
            "date",
        ],
    )

    return (
        index
        .to_frame(index=False)
        .sort_values(
            [
                "participant_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


def _daily_count(
    events: pd.DataFrame,
    mask: pd.Series,
    name: str,
) -> pd.DataFrame:
    """
    Count matching events per participant/day.
    """

    selected = events.loc[
        mask,
        [
            "participant_id",
            "date",
        ],
    ]

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "date",
                name,
            ]
        )

    return (
        selected
        .groupby(
            [
                "participant_id",
                "date",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": name,
            }
        )
    )


def _daily_points(
    events: pd.DataFrame,
    explicit_mask: pd.Series,
) -> pd.DataFrame:
    """
    Sum points earned through explicit engagement
    on each participant/day.
    """

    selected = events.loc[
        explicit_mask,
        [
            "participant_id",
            "date",
            "points",
        ],
    ]

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "date",
                "points_today",
            ]
        )

    return (
        selected
        .groupby(
            [
                "participant_id",
                "date",
            ],
            as_index=False,
        )["points"]
        .sum()
        .rename(
            columns={
                "points": "points_today",
            }
        )
    )


def _provider_mask(
    events: pd.DataFrame,
    text: str,
) -> pd.Series:
    """
    Match a provider name conservatively.
    """

    if events.empty:
        return pd.Series(
            False,
            index=events.index,
            dtype=bool,
        )

    provider = (
        events["provider"]
        .astype("string")
        .fillna("")
        .str.lower()
    )

    return provider.str.contains(
        text.lower(),
        na=False,
    )


def _provider_stream_state(
    events: pd.DataFrame,
    activity_stream_state: str,
    provider_name: str,
) -> str:
    """
    Determine whether a provider-specific stream
    is actually observed in the export.

    If the whole activity export is unavailable,
    the provider stream is unavailable too.

    If activity data exists but that provider never
    occurs, we conservatively call it unavailable.

    We do NOT interpret this as zero participant behavior.
    """

    if activity_stream_state == "unavailable":
        return "unavailable"

    provider_present = bool(
        _provider_mask(
            events,
            provider_name,
        ).any()
    )

    if provider_present:
        return "available"

    return "unavailable"


def _merge_metric(
    state: pd.DataFrame,
    metric: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge one daily metric onto the participant-day grid.
    """

    return state.merge(
        metric,
        on=[
            "participant_id",
            "date",
        ],
        how="left",
    )


def _add_rolling_metrics(
    state: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add 7-day and 14-day trailing metrics.

    Because state contains every calendar day,
    a rolling window of 7 rows really means
    7 calendar days.
    """

    result = state.copy()

    result["_active_today"] = (
        result["explicit_events_today"] > 0
    ).astype(int)

    grouped = result.groupby(
        "participant_id",
        sort=False,
    )

    for window in (
        7,
        14,
    ):

        result[
            f"explicit_events_{window}d"
        ] = grouped[
            "explicit_events_today"
        ].transform(
            lambda s: (
                s.rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )

        result[
            f"active_days_{window}d"
        ] = grouped[
            "_active_today"
        ].transform(
            lambda s: (
                s.rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )

        result[
            f"points_{window}d"
        ] = grouped[
            "points_today"
        ].transform(
            lambda s: (
                s.rolling(
                    window=window,
                    min_periods=1,
                )
                .sum()
            )
        )

    return result.drop(
        columns=[
            "_active_today"
        ]
    )


def _add_engagement_state(
    state: pd.DataFrame,
    activity_stream_state: str,
) -> pd.DataFrame:
    """
    Add causal engagement state and
    days-since-last-engagement.

    No future events are consulted.
    """

    result = state.copy()

    # -------------------------------------------------
    # If explicit activity data is unavailable,
    # do NOT pretend that zero means non-engagement.
    # -------------------------------------------------

    if activity_stream_state == "unavailable":

        result["has_ever_engaged"] = pd.NA

        result[
            "last_explicit_engagement_date"
        ] = pd.NaT

        result[
            "days_since_last_explicit_engagement"
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

        result["engagement_state"] = (
            "unavailable"
        )

        return result

    # -------------------------------------------------
    # Has participant engaged by this day?
    # -------------------------------------------------

    result["has_ever_engaged"] = (
        result
        .groupby("participant_id")[
            "explicit_events_today"
        ]
        .cumsum()
        .gt(0)
    )

    # -------------------------------------------------
    # Last explicit engagement date
    # -------------------------------------------------

    result[
        "last_explicit_engagement_date"
    ] = result["date"].where(
        result["explicit_events_today"] > 0
    )

    result[
        "last_explicit_engagement_date"
    ] = (
        result
        .groupby("participant_id")[
            "last_explicit_engagement_date"
        ]
        .ffill()
    )

    # -------------------------------------------------
    # Days since explicit engagement
    # -------------------------------------------------

    days_since = (
        result["date"]
        - result[
            "last_explicit_engagement_date"
        ]
    ).dt.days

    result[
        "days_since_last_explicit_engagement"
    ] = days_since.astype("Int64")

    # -------------------------------------------------
    # Daily state
    # -------------------------------------------------

    result["engagement_state"] = (
        "not_yet_engaged"
    )

    active = (
        result["explicit_events_today"] > 0
    )

    result.loc[
        active,
        "engagement_state",
    ] = "active"

    quiet = (
        result["has_ever_engaged"]
        & ~active
        & (
            result[
                "days_since_last_explicit_engagement"
            ] <= 6
        )
    )

    result.loc[
        quiet,
        "engagement_state",
    ] = "quiet"

    inactivity_7d = (
        result["has_ever_engaged"]
        & ~active
        & (
            result[
                "days_since_last_explicit_engagement"
            ].between(
                7,
                13,
            )
        )
    )

    result.loc[
        inactivity_7d,
        "engagement_state",
    ] = "prolonged_inactivity_7d"

    inactivity_14d = (
        result["has_ever_engaged"]
        & ~active
        & (
            result[
                "days_since_last_explicit_engagement"
            ] >= 14
        )
    )

    result.loc[
        inactivity_14d,
        "engagement_state",
    ] = "prolonged_inactivity_14d"

    return result


def _add_episode_state(
    state: pd.DataFrame,
    config: TrajectoryAuditConfig,
) -> pd.DataFrame:
    """
    Add a causal current episode number.

    A new episode begins on an explicit-engagement day
    when the previous explicit-engagement day was more
    than episode_gap_days earlier.

    The current episode remains assigned through the
    allowed inactivity window.

    Once days_since_last_explicit_engagement exceeds
    episode_gap_days, there is no currently open episode.
    """

    result = state.copy()

    engagement_days = result.loc[
        result["explicit_events_today"] > 0,
        [
            "participant_id",
            "date",
        ],
    ].copy()

    if engagement_days.empty:

        result["episode_number"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

        result["episode_id"] = pd.NA

        return result

    engagement_days[
        "previous_engagement_date"
    ] = (
        engagement_days
        .groupby("participant_id")["date"]
        .shift(1)
    )

    engagement_days["gap_days"] = (
        engagement_days["date"]
        - engagement_days[
            "previous_engagement_date"
        ]
    ).dt.days

    engagement_days["new_episode"] = (
        engagement_days[
            "previous_engagement_date"
        ].isna()
        |
        (
            engagement_days["gap_days"]
            > config.episode_gap_days
        )
    )

    engagement_days["episode_number"] = (
        engagement_days
        .groupby("participant_id")[
            "new_episode"
        ]
        .cumsum()
        .astype("Int64")
    )

    result = result.merge(
        engagement_days[
            [
                "participant_id",
                "date",
                "episode_number",
            ]
        ],
        on=[
            "participant_id",
            "date",
        ],
        how="left",
    )

    result["episode_number"] = (
        result
        .groupby("participant_id")[
            "episode_number"
        ]
        .ffill()
        .astype("Int64")
    )

    # Once the permitted gap is exceeded,
    # the previous episode is no longer current.
    closed_episode = (
        result[
            "days_since_last_explicit_engagement"
        ]
        > config.episode_gap_days
    )

    result.loc[
        closed_episode.fillna(False),
        "episode_number",
    ] = pd.NA

    result["episode_id"] = pd.NA

    has_episode = (
        result["episode_number"].notna()
    )

    result.loc[
        has_episode,
        "episode_id",
    ] = (
        result.loc[
            has_episode,
            "participant_id",
        ]
        .astype(str)
        + ":E"
        + result.loc[
            has_episode,
            "episode_number",
        ]
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )

    return result


def build_participant_state_daily(
    config: TrajectoryAuditConfig,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one causal participant state per participant/day.
    """

    prepared = _prepare_events(
        events
    )

    window = build_observation_window(
        config,
        prepared,
    )

    cohort = _export_cohort(
        config,
        prepared,
    )

    start = window["effective_start"]
    end = window["analysis_cutoff"]

    participant_ids = cohort[
        "participant_ids"
    ]

    if (
        start is None
        or end is None
        or not participant_ids
    ):
        return pd.DataFrame(
            columns=STATE_COLUMNS
        )

    state = _build_daily_grid(
        participant_ids,
        start,
        end,
    )

    # -------------------------------------------------
    # Stream availability
    # -------------------------------------------------

    _, stream_info = load_campaign_export(
        config.campaign_data_path
    )

    activity_stream_state = (
        stream_info["activities"]["status"]
    )

    navigation_stream_state = (
        stream_info["navigation"]["status"]
    )

    notification_stream_state = (
        stream_info[
            "notification_events"
        ]["status"]
    )

    sensor_stream_state = (
        stream_info[
            "sensor_events"
        ]["status"]
    )

    garmin_stream_state = (
        _provider_stream_state(
            prepared,
            activity_stream_state,
            "garmin",
        )
    )

    nutrida_stream_state = (
        _provider_stream_state(
            prepared,
            activity_stream_state,
            "nutrida",
        )
    )

    # -------------------------------------------------
    # Event masks
    # -------------------------------------------------

    explicit = explicit_engagement_mask(
        prepared
    )

    behavioral_sensor = behavioral_sensor_mask(
        prepared
    )

    event_kind = (
        prepared["event_kind"]
        .astype("string")
        .fillna("")
        .str.lower()
    )

    navigation = event_kind.eq(
        "navigation"
    )

    notifications = event_kind.eq(
        "notification"
    )

    sensor = event_kind.eq(
        "sensor"
    )

    garmin = _provider_mask(
        prepared,
        "garmin",
    )

    nutrida = _provider_mask(
        prepared,
        "nutrida",
    )

    # -------------------------------------------------
    # Daily metrics
    # -------------------------------------------------

    metrics = [
        _daily_count(
            prepared,
            explicit,
            "explicit_events_today",
        ),

        _daily_count(
            prepared,
            navigation,
            "navigation_events_today",
        ),

        _daily_count(
            prepared,
            notifications,
            "notification_events_today",
        ),

        _daily_count(
            prepared,
            garmin,
            "garmin_events_today",
        ),

        _daily_count(
            prepared,
            nutrida,
            "nutrida_events_today",
        ),

        _daily_count(
            prepared,
            sensor,
            "sensor_events_today",
        ),

        _daily_count(
            prepared,
            behavioral_sensor,
            "behavioral_sensor_events_today",
        ),

        _daily_points(
            prepared,
            explicit,
        ),
    ]

    for metric in metrics:
        state = _merge_metric(
            state,
            metric,
        )

    # -------------------------------------------------
    # Missing daily observations initially become 0,
    # but only for streams we actually possess.
    # -------------------------------------------------

    count_columns = [
        "explicit_events_today",
        "navigation_events_today",
        "notification_events_today",
        "garmin_events_today",
        "nutrida_events_today",
        "sensor_events_today",
        "behavioral_sensor_events_today",
    ]

    for column in count_columns:
        state[column] = (
            pd.to_numeric(
                state[column],
                errors="coerce",
            )
            .fillna(0)
            .astype("Int64")
        )

    state["points_today"] = (
        pd.to_numeric(
            state["points_today"],
            errors="coerce",
        )
        .fillna(0)
    )

    # -------------------------------------------------
    # Preserve unavailable != zero
    # -------------------------------------------------

    if activity_stream_state == "unavailable":

        state["explicit_events_today"] = pd.NA
        state["points_today"] = pd.NA

    if navigation_stream_state == "unavailable":
        state["navigation_events_today"] = pd.NA

    if notification_stream_state == "unavailable":
        state["notification_events_today"] = pd.NA

    if sensor_stream_state == "unavailable":
        state["sensor_events_today"] = pd.NA

    if garmin_stream_state == "unavailable":
        state["garmin_events_today"] = pd.NA

    if nutrida_stream_state == "unavailable":
        state["nutrida_events_today"] = pd.NA

    if (
        garmin_stream_state == "unavailable"
        and sensor_stream_state == "unavailable"
    ):
        state[
            "behavioral_sensor_events_today"
        ] = pd.NA

    # -------------------------------------------------
    # Engagement-derived state
    # -------------------------------------------------

    state = _add_engagement_state(
        state,
        activity_stream_state,
    )

    # Rolling metrics require activity data.
    if activity_stream_state != "unavailable":

        state = _add_rolling_metrics(
            state
        )

        state = _add_episode_state(
            state,
            config,
        )

    else:

        for column in [
            "explicit_events_7d",
            "explicit_events_14d",
            "active_days_7d",
            "active_days_14d",
            "points_7d",
            "points_14d",
        ]:
            state[column] = pd.NA

        state["episode_number"] = pd.NA
        state["episode_id"] = pd.NA

    # -------------------------------------------------
    # Stream-state columns
    # -------------------------------------------------

    state["activity_stream_state"] = (
        activity_stream_state
    )

    state["navigation_stream_state"] = (
        navigation_stream_state
    )

    state["notification_stream_state"] = (
        notification_stream_state
    )

    state["sensor_stream_state"] = (
        sensor_stream_state
    )

    state["garmin_stream_state"] = (
        garmin_stream_state
    )

    state["nutrida_stream_state"] = (
        nutrida_stream_state
    )

    # -------------------------------------------------
    # Human-readable dates
    # -------------------------------------------------

    state["date"] = (
        pd.to_datetime(
            state["date"],
            utc=True,
        )
        .dt.date
    )

    if (
        "last_explicit_engagement_date"
        in state.columns
    ):

        state[
            "last_explicit_engagement_date"
        ] = (
            pd.to_datetime(
                state[
                    "last_explicit_engagement_date"
                ],
                utc=True,
                errors="coerce",
            )
            .dt.date
        )

    # -------------------------------------------------
    # Stable output schema
    # -------------------------------------------------

    for column in STATE_COLUMNS:

        if column not in state.columns:
            state[column] = pd.NA

    return state[
        STATE_COLUMNS
    ]


def run_participant_state_daily(
    config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
    """
    Build and save participant_state_daily.csv.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    events = build_normalized_events(
        config
    )

    state = build_participant_state_daily(
        config,
        events,
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "participant_state_daily.csv",
    )

    state.to_csv(
        output_path,
        index=False,
    )

    return state


if __name__ == "__main__":

    state = run_participant_state_daily()

    print(
        "ParticipantState(t) written successfully."
    )

    print()

    print(
        "Daily state rows:",
        len(state),
    )

    if not state.empty:

        print(
            "Participants:",
            state["participant_id"].nunique(),
        )

        print(
            "First date:",
            state["date"].min(),
        )

        print(
            "Last date:",
            state["date"].max(),
        )

        print()

        last_date = state["date"].max()

        cutoff_state = state[
            state["date"] == last_date
        ]

        print(
            "States at analysis cutoff:"
        )

        counts = (
            cutoff_state[
                "engagement_state"
            ]
            .value_counts(
                dropna=False
            )
        )

        for name, count in counts.items():

            print(
                f"  {name}: {count}"
            )

        print()

        print("Stream states:")

        for column in [
            "activity_stream_state",
            "navigation_stream_state",
            "notification_stream_state",
            "sensor_stream_state",
            "garmin_stream_state",
            "nutrida_stream_state",
        ]:

            values = (
                state[column]
                .drop_duplicates()
                .tolist()
            )

            print(
                f"  {column}: {values}"
            )