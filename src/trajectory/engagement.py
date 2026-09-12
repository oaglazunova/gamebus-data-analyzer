from __future__ import annotations

import pandas as pd


PASSIVE_ACTIVITY_TYPES = {
    "DAY_AGGREGATE",
}


def behavioral_sensor_mask(
    events: pd.DataFrame,
) -> pd.Series:
    """
    Identify passive behavioral/sensor observations.

    These observations tell us something about participant
    behavior, but they do not imply active engagement with
    the digital intervention.

    Examples:
        Garmin DAY_AGGREGATE
        sensor events
    """

    if events.empty:
        return pd.Series(
            False,
            index=events.index,
            dtype=bool,
        )

    event_kind = (
        events.get(
            "event_kind",
            pd.Series(
                "",
                index=events.index,
            ),
        )
        .fillna("")
        .astype(str)
        .str.lower()
    )

    event_type = (
        events.get(
            "event_type",
            pd.Series(
                "",
                index=events.index,
            ),
        )
        .fillna("")
        .astype(str)
        .str.upper()
    )

    provider = (
        events.get(
            "provider",
            pd.Series(
                "",
                index=events.index,
            ),
        )
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return (
        event_kind.eq("sensor")
        |
        event_type.isin(
            PASSIVE_ACTIVITY_TYPES
        )
        |
        provider.str.contains(
            "garmin",
            na=False,
        )
    )


def deleted_activity_mask(
    events: pd.DataFrame,
) -> pd.Series:
    """
    Robustly interpret the normalized is_deleted field.

    This works whether the value came directly from pandas
    as a boolean or was read back from CSV as text.
    """

    if (
        events.empty
        or "is_deleted" not in events.columns
    ):
        return pd.Series(
            False,
            index=events.index,
            dtype=bool,
        )

    return (
        events["is_deleted"]
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


def explicit_engagement_mask(
    events: pd.DataFrame,
) -> pd.Series:
    """
    Identify participant-generated intervention engagement.

    For now:

        activity
        AND not deleted
        AND not passive sensor data

    This includes, for example:
        GameBus Studio GENERAL_ACTIVITY
        GameBus Studio WATCH_STORY
        PHYSICAL_ACTIVITY task completion
        Nutrida activity

    It excludes:
        Garmin DAY_AGGREGATE
        navigation
        notifications
        sensor records
    """

    if events.empty:
        return pd.Series(
            False,
            index=events.index,
            dtype=bool,
        )

    event_kind = (
        events.get(
            "event_kind",
            pd.Series(
                "",
                index=events.index,
            ),
        )
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return (
        event_kind.eq("activity")
        & ~deleted_activity_mask(events)
        & ~behavioral_sensor_mask(events)
    )