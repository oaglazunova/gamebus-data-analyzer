from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
)

from src.trajectory.engagement import (
    behavioral_sensor_mask,
    explicit_engagement_mask,
)

from src.trajectory.event_normalization import (
    build_normalized_events,
)


EPISODE_COLUMNS = [
    "participant_id",
    "episode_id",
    "episode_number",

    "episode_start",
    "episode_end",
    "duration_days",

    "active_days",
    "event_count",
    "points",

    "first_event_id",
    "last_event_id",

    "providers",
    "event_types",

    "engagement_gap_days_before",
    "inactive_days_before",
]


def _json_unique(
    values: pd.Series,
) -> str:
    """
    Store unique non-null values as a JSON array.
    """

    result = []

    for value in values.dropna():

        value = str(value)

        if value not in result:
            result.append(value)

    return json.dumps(
        result,
        ensure_ascii=False,
    )


def _participant_id_text(
    value: Any,
) -> str:
    """
    Produce a clean participant identifier
    for episode_id.

    For example:

        308.0 -> "308"
    """

    try:
        numeric = float(value)

        if numeric.is_integer():
            return str(int(numeric))

    except (TypeError, ValueError):
        pass

    return str(value)


def select_explicit_engagement(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select only events that define active
    participation episodes.
    """

    if events.empty:
        return events.copy()

    mask = explicit_engagement_mask(
        events
    )

    result = events.loc[
        mask
    ].copy()

    # A trajectory event without participant or time
    # cannot be placed into an episode.
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

    # Episode segmentation is based on calendar dates,
    # not exact timestamps.
    result["engagement_day"] = (
        result["occurred_at"]
        .dt.floor("D")
    )

    result["points"] = pd.to_numeric(
        result["points"],
        errors="coerce",
    ).fillna(0)

    result = result.sort_values(
        [
            "participant_id",
            "occurred_at",
            "event_id",
        ]
    ).reset_index(
        drop=True
    )

    return result


def build_participation_episodes(
    events: pd.DataFrame,
    config: TrajectoryAuditConfig,
) -> pd.DataFrame:
    """
    Build participation episodes from explicit engagement.

    A new episode begins when the current explicit
    engagement occurs more than episode_gap_days
    calendar days after the previous explicit engagement.

    With episode_gap_days = 14:

        Jan 1 -> Jan 15
            same episode

        Jan 1 -> Jan 16
            new episode

    because Jan 2 through Jan 15 are 14 complete
    inactive calendar days.
    """

    engagement = select_explicit_engagement(
        events
    )

    if engagement.empty:
        return pd.DataFrame(
            columns=EPISODE_COLUMNS
        )

    # -------------------------------------------------
    # Difference from previous engagement day
    # -------------------------------------------------

    engagement["previous_engagement_day"] = (
        engagement
        .groupby("participant_id")[
            "engagement_day"
        ]
        .shift(1)
    )

    engagement["day_gap"] = (
        engagement["engagement_day"]
        - engagement[
            "previous_engagement_day"
        ]
    ).dt.days

    # -------------------------------------------------
    # Episode boundaries
    # -------------------------------------------------

    first_event = (
        engagement[
            "previous_engagement_day"
        ]
        .isna()
    )

    after_long_gap = (
        engagement["day_gap"]
        > config.episode_gap_days
    )

    engagement["starts_new_episode"] = (
        first_event
        | after_long_gap
    )

    engagement["episode_number"] = (
        engagement
        .groupby("participant_id")[
            "starts_new_episode"
        ]
        .cumsum()
        .astype(int)
    )

    # -------------------------------------------------
    # Aggregate one row per episode
    # -------------------------------------------------

    rows = []

    for (
        participant_id,
        episode_number,
    ), group in engagement.groupby(
        [
            "participant_id",
            "episode_number",
        ],
        sort=True,
    ):

        group = group.sort_values(
            "occurred_at"
        )

        start = group[
            "engagement_day"
        ].min()

        end = group[
            "engagement_day"
        ].max()

        participant_text = (
            _participant_id_text(
                participant_id
            )
        )

        rows.append(
            {
                "participant_id": (
                    participant_id
                ),

                "episode_id": (
                    f"{participant_text}:"
                    f"E{episode_number:03d}"
                ),

                "episode_number": int(
                    episode_number
                ),

                "episode_start": start,

                "episode_end": end,

                "duration_days": int(
                    (end - start).days + 1
                ),

                "active_days": int(
                    group[
                        "engagement_day"
                    ].nunique()
                ),

                "event_count": int(
                    len(group)
                ),

                "points": float(
                    group["points"].sum()
                ),

                "first_event_id": (
                    group.iloc[0][
                        "event_id"
                    ]
                ),

                "last_event_id": (
                    group.iloc[-1][
                        "event_id"
                    ]
                ),

                "providers": _json_unique(
                    group["provider"]
                ),

                "event_types": _json_unique(
                    group["event_type"]
                ),
            }
        )

    episodes = pd.DataFrame(
        rows
    )

    # -------------------------------------------------
    # Gap between episodes
    # -------------------------------------------------

    episodes = episodes.sort_values(
        [
            "participant_id",
            "episode_number",
        ]
    ).reset_index(
        drop=True
    )

    episodes[
        "previous_episode_end"
    ] = (
        episodes
        .groupby("participant_id")[
            "episode_end"
        ]
        .shift(1)
    )

    episodes[
        "engagement_gap_days_before"
    ] = (
        episodes["episode_start"]
        - episodes["previous_episode_end"]
    ).dt.days

    # Example:
    #
    # previous engagement = Jan 1
    # next engagement     = Jan 16
    #
    # difference          = 15 days
    # completely inactive = Jan 2 ... Jan 15 = 14 days
    #
    episodes[
        "inactive_days_before"
    ] = (
        episodes[
            "engagement_gap_days_before"
        ]
        - 1
    )

    episodes.loc[
        episodes["episode_number"] == 1,
        [
            "engagement_gap_days_before",
            "inactive_days_before",
        ],
    ] = pd.NA

    episodes = episodes.drop(
        columns=[
            "previous_episode_end"
        ]
    )

    # Convert timestamps to simple dates in output.
    episodes["episode_start"] = (
        episodes["episode_start"]
        .dt.date
    )

    episodes["episode_end"] = (
        episodes["episode_end"]
        .dt.date
    )

    # Keep a predictable column order.
    for column in EPISODE_COLUMNS:

        if column not in episodes.columns:
            episodes[column] = pd.NA

    return episodes[
        EPISODE_COLUMNS
    ]


def run_participation_episode_builder(
    config: TrajectoryAuditConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize current campaign data,
    build episodes, and save the result.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    events = build_normalized_events(
        config
    )

    episodes = build_participation_episodes(
        events,
        config,
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "participation_episodes.csv",
    )

    episodes.to_csv(
        output_path,
        index=False,
    )

    return events, episodes


if __name__ == "__main__":

    config = TrajectoryAuditConfig()

    events, episodes = (
        run_participation_episode_builder(
            config
        )
    )

    explicit = explicit_engagement_mask(
        events
    )

    behavioral_sensor = (
        behavioral_sensor_mask(
            events
        )
    )

    explicit_events = events.loc[
        explicit
    ]

    print(
        "Participation episodes "
        "written successfully."
    )

    print()

    print(
        "Episode gap threshold:",
        config.episode_gap_days,
        "days",
    )

    print(
        "Explicit engagement events:",
        len(explicit_events),
    )

    print(
        "Behavioral sensor events:",
        int(
            behavioral_sensor.sum()
        ),
    )

    print(
        "Participants with explicit engagement:",
        explicit_events[
            "participant_id"
        ].nunique(),
    )

    print(
        "Participation episodes:",
        len(episodes),
    )

    if not episodes.empty:

        print()

        print(
            "Participants by number of episodes:"
        )

        counts = (
            episodes
            .groupby("participant_id")[
                "episode_number"
            ]
            .max()
            .value_counts()
            .sort_index()
        )

        for (
            episode_count,
            participant_count,
        ) in counts.items():

            print(
                f"  {episode_count} episode(s): "
                f"{participant_count} participants"
            )