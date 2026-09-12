from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.trajectory.common import (
    TrajectoryAuditConfig,
    load_campaign_export,
    parse_datetime_series,
)


NORMALIZED_EVENT_COLUMNS = [
    "event_id",
    "participant_id",
    "occurred_at",
    "event_date",

    "event_kind",
    "event_type",
    "provider",

    "points",

    "reward_count",
    "reward_rule_names",
    "challenge_ids",
    "challenge_names",

    "navigation_uri",
    "session_id",

    "notification_action",

    "device_type",

    "is_deleted",

    "source_wave_references",
    "source_waves",
    "source_row_count",

    "dedup_conflict",

    "raw_properties",
    "raw_rewarded_participations",
]


def _json_list(values: Iterable[Any]) -> str:
    """
    Convert a collection into a JSON list string.

    Example:

        ["q1End", "q2End"]

    rather than Python's:

        ['q1End', 'q2End']

    JSON is easier to read reliably later.
    """

    result = []

    for value in values:

        if pd.isna(value):
            continue

        # Convert numpy values such as np.int64
        # into normal Python values.
        item_method = getattr(value, "item", None)

        if callable(item_method):
            try:
                value = item_method()
            except Exception:
                pass

        if value not in result:
            result.append(value)

    return json.dumps(
        result,
        ensure_ascii=False,
    )


def _parse_rewarded_participations(
    value: Any,
) -> Dict[str, Any]:
    """
    Parse the GameBus rewardedParticipations JSON.

    One activity can reward one or more rules/challenges.

    We deliberately keep both:
        - useful extracted fields
        - the original JSON

    because later domain/tool analysis may need
    information we do not yet use.
    """

    result = {
        "reward_count": 0,
        "reward_rule_names": [],
        "challenge_ids": [],
        "challenge_names": [],
    }

    if value is None or pd.isna(value):
        return result

    try:
        rewards = json.loads(value)

    except (TypeError, json.JSONDecodeError):
        return result

    if not isinstance(rewards, list):
        return result

    result["reward_count"] = len(rewards)

    for reward in rewards:

        if not isinstance(reward, dict):
            continue

        rule = reward.get("rule")

        if (
            rule is not None
            and rule not in result["reward_rule_names"]
        ):
            result["reward_rule_names"].append(rule)

        challenge = reward.get("challenge")

        if isinstance(challenge, dict):

            challenge_id = challenge.get("xid")
            challenge_name = challenge.get("name")

            if (
                challenge_id is not None
                and challenge_id
                not in result["challenge_ids"]
            ):
                result["challenge_ids"].append(
                    challenge_id
                )

            if (
                challenge_name is not None
                and challenge_name
                not in result["challenge_names"]
            ):
                result["challenge_names"].append(
                    challenge_name
                )

    return result


def _base_event_frame(
    df: pd.DataFrame,
    event_kind: str,
) -> pd.DataFrame:
    """
    Create fields that are common to every event stream.
    """

    if df.empty:
        return pd.DataFrame(
            columns=NORMALIZED_EVENT_COLUMNS
        )

    result = pd.DataFrame(
        index=df.index
    )

    # -------------------------------------------------
    # Participant
    # -------------------------------------------------

    if "pid" in df.columns:

        result["participant_id"] = pd.to_numeric(
            df["pid"],
            errors="coerce",
        ).astype("Int64")

    else:

        result["participant_id"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Int64",
        )

    # -------------------------------------------------
    # Timestamp
    # -------------------------------------------------

    result["occurred_at"] = (
        parse_datetime_series(df)
    )

    result["event_date"] = (
        result["occurred_at"]
        .dt.date
    )

    # -------------------------------------------------
    # Source type
    # -------------------------------------------------

    result["event_kind"] = event_kind

    # -------------------------------------------------
    # Original GameBus event ID
    # -------------------------------------------------

    if "aid" in df.columns:

        aid = df["aid"]

    else:

        aid = pd.Series(
            pd.NA,
            index=df.index,
        )

    event_ids = []

    for row_index, raw_aid in aid.items():

        if pd.notna(raw_aid):

            try:
                clean_aid = int(raw_aid)

            except (TypeError, ValueError):
                clean_aid = str(raw_aid)

            event_ids.append(
                f"{event_kind}:{clean_aid}"
            )

        else:

            # Rare fallback when an exported row
            # has no GameBus activity/event ID.
            #
            # Such rows cannot safely be deduplicated
            # across waves, so we retain them individually.
            event_ids.append(
                f"{event_kind}:row:{row_index}"
            )

    result["event_id"] = event_ids

    # -------------------------------------------------
    # Wave provenance
    # -------------------------------------------------

    if "waveReference" in df.columns:

        result["_wave_reference"] = (
            df["waveReference"]
        )

    else:

        result["_wave_reference"] = pd.NA

    if "wave" in df.columns:

        result["_wave"] = df["wave"]

    else:

        result["_wave"] = pd.NA

    return result


def _normalize_activities(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize 2-activities.csv.
    """

    if df.empty:
        return pd.DataFrame(
            columns=NORMALIZED_EVENT_COLUMNS
        )

    result = _base_event_frame(
        df,
        event_kind="activity",
    )

    # -------------------------------------------------
    # Activity-specific fields
    # -------------------------------------------------

    result["event_type"] = (
        df["type"]
        if "type" in df.columns
        else pd.NA
    )

    result["provider"] = (
        df["provider"]
        if "provider" in df.columns
        else pd.NA
    )

    if "averageNumberOfPoints" in df.columns:

        result["points"] = pd.to_numeric(
            df["averageNumberOfPoints"],
            errors="coerce",
        )

    else:

        result["points"] = pd.NA

    # -------------------------------------------------
    # Reward / task information
    # -------------------------------------------------

    reward_count = []
    reward_rule_names = []
    challenge_ids = []
    challenge_names = []

    rewarded_values = (
        df["rewardedParticipations"]
        if "rewardedParticipations" in df.columns
        else pd.Series(
            pd.NA,
            index=df.index,
        )
    )

    for value in rewarded_values:

        parsed = _parse_rewarded_participations(
            value
        )

        reward_count.append(
            parsed["reward_count"]
        )

        reward_rule_names.append(
            json.dumps(
                parsed["reward_rule_names"],
                ensure_ascii=False,
            )
        )

        challenge_ids.append(
            json.dumps(
                parsed["challenge_ids"],
                ensure_ascii=False,
            )
        )

        challenge_names.append(
            json.dumps(
                parsed["challenge_names"],
                ensure_ascii=False,
            )
        )

    result["reward_count"] = reward_count

    result["reward_rule_names"] = (
        reward_rule_names
    )

    result["challenge_ids"] = (
        challenge_ids
    )

    result["challenge_names"] = (
        challenge_names
    )

    # -------------------------------------------------
    # Deleted activity flag
    # -------------------------------------------------

    if "isDeleted" in df.columns:

        deleted = (
            df["isDeleted"]
            .astype("string")
            .fillna("false")
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

        result["is_deleted"] = deleted

    else:

        result["is_deleted"] = False

    # -------------------------------------------------
    # Preserve raw JSON fields
    # -------------------------------------------------

    result["raw_properties"] = (
        df["properties"]
        if "properties" in df.columns
        else pd.NA
    )

    result[
        "raw_rewarded_participations"
    ] = (
        df["rewardedParticipations"]
        if "rewardedParticipations" in df.columns
        else pd.NA
    )

    # Fields belonging to other event types.
    result["navigation_uri"] = pd.NA
    result["session_id"] = pd.NA
    result["notification_action"] = pd.NA
    result["device_type"] = pd.NA

    return result


def _normalize_navigation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize 3-navigation-events.csv.
    """

    if df.empty:
        return pd.DataFrame(
            columns=NORMALIZED_EVENT_COLUMNS
        )

    result = _base_event_frame(
        df,
        event_kind="navigation",
    )

    result["event_type"] = "NAVIGATION"

    result["provider"] = "GameBus"

    result["points"] = 0

    result["navigation_uri"] = (
        df["uri"]
        if "uri" in df.columns
        else pd.NA
    )

    result["session_id"] = (
        df["session"]
        if "session" in df.columns
        else pd.NA
    )

    result["notification_action"] = pd.NA
    result["device_type"] = pd.NA

    result["reward_count"] = 0
    result["reward_rule_names"] = "[]"
    result["challenge_ids"] = "[]"
    result["challenge_names"] = "[]"

    result["is_deleted"] = False

    result["raw_properties"] = pd.NA

    result[
        "raw_rewarded_participations"
    ] = pd.NA

    return result


def _normalize_notifications(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize 4-notification-events.csv.
    """

    if df.empty:
        return pd.DataFrame(
            columns=NORMALIZED_EVENT_COLUMNS
        )

    result = _base_event_frame(
        df,
        event_kind="notification",
    )

    action = (
        df["action"]
        if "action" in df.columns
        else pd.Series(
            pd.NA,
            index=df.index,
        )
    )

    # For notifications the action is also
    # the most informative event subtype:
    #
    # RECEIVED
    # CLICKED
    # etc.
    result["event_type"] = action

    result["provider"] = "GameBus"

    result["points"] = 0

    result["notification_action"] = action

    result["navigation_uri"] = pd.NA
    result["session_id"] = pd.NA
    result["device_type"] = pd.NA

    result["reward_count"] = 0
    result["reward_rule_names"] = "[]"
    result["challenge_ids"] = "[]"
    result["challenge_names"] = "[]"

    result["is_deleted"] = False

    result["raw_properties"] = pd.NA

    result[
        "raw_rewarded_participations"
    ] = pd.NA

    return result


def _normalize_sensors(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize 5-sensor-events.csv.

    We intentionally do not parse the large sensor logs yet.

    At this stage we preserve the event and device type.
    Detailed sensor extraction belongs to a later
    stream-specific enrichment step.
    """

    if df.empty:
        return pd.DataFrame(
            columns=NORMALIZED_EVENT_COLUMNS
        )

    result = _base_event_frame(
        df,
        event_kind="sensor",
    )

    result["event_type"] = "SENSOR_EVENT"

    result["provider"] = "GameBus"

    result["points"] = 0

    result["device_type"] = (
        df["deviceType"]
        if "deviceType" in df.columns
        else pd.NA
    )

    result["navigation_uri"] = pd.NA
    result["session_id"] = pd.NA
    result["notification_action"] = pd.NA

    result["reward_count"] = 0
    result["reward_rule_names"] = "[]"
    result["challenge_ids"] = "[]"
    result["challenge_names"] = "[]"

    result["is_deleted"] = False

    result["raw_properties"] = pd.NA

    result[
        "raw_rewarded_participations"
    ] = pd.NA

    return result


def _first_non_null(
    values: pd.Series,
) -> Any:
    """
    Return the first non-null value in a Series.
    """

    non_null = values.dropna()

    if non_null.empty:
        return pd.NA

    return non_null.iloc[0]


def _has_conflict(
    group: pd.DataFrame,
) -> bool:
    """
    Detect whether duplicate representations of
    the same event disagree on important fields.

    Duplicates caused only by overlapping campaign
    waves should normally be identical except for
    waveReference/wave.
    """

    fields = [
        "participant_id",
        "occurred_at",
        "event_type",
        "provider",
        "points",
    ]

    for field in fields:

        if field not in group.columns:
            continue

        values = (
            group[field]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(values) > 1:
            return True

    return False


def _deduplicate_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deduplicate the same event appearing in
    multiple overlapping campaign waves.

    Example:

        activity:1499578
        wave = q1End

        activity:1499578
        wave = q2End

    becomes one normalized event with:

        source_wave_references =
            ["q1End", "q2End"]

        source_row_count = 2
    """

    if events.empty:
        return pd.DataFrame(
            columns=NORMALIZED_EVENT_COLUMNS
        )

    rows: List[Dict[str, Any]] = []

    for event_id, group in events.groupby(
        "event_id",
        sort=False,
        dropna=False,
    ):

        first = group.iloc[0]

        row: Dict[str, Any] = {
            "event_id": event_id,
        }

        # ---------------------------------------------
        # Scalar fields
        # ---------------------------------------------

        scalar_columns = [
            "participant_id",
            "occurred_at",
            "event_date",
            "event_kind",
            "event_type",
            "provider",
            "points",

            "reward_count",
            "reward_rule_names",
            "challenge_ids",
            "challenge_names",

            "navigation_uri",
            "session_id",

            "notification_action",
            "device_type",

            "is_deleted",

            "raw_properties",
            "raw_rewarded_participations",
        ]

        for column in scalar_columns:

            if column in group.columns:

                row[column] = _first_non_null(
                    group[column]
                )

            else:

                row[column] = pd.NA

        # ---------------------------------------------
        # Preserve all wave memberships
        # ---------------------------------------------

        row["source_wave_references"] = (
            _json_list(
                group["_wave_reference"]
            )
        )

        row["source_waves"] = (
            _json_list(
                group["_wave"]
            )
        )

        row["source_row_count"] = int(
            len(group)
        )

        # ---------------------------------------------
        # Quality check
        # ---------------------------------------------

        row["dedup_conflict"] = (
            _has_conflict(group)
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    # Ensure every expected column exists.
    for column in NORMALIZED_EVENT_COLUMNS:

        if column not in result.columns:
            result[column] = pd.NA

    result = result[
        NORMALIZED_EVENT_COLUMNS
    ]

    return result


def build_normalized_events(
    config: TrajectoryAuditConfig,
) -> pd.DataFrame:
    """
    Build one normalized event table
    from all available campaign-export streams.
    """

    export_data, stream_status = (
        load_campaign_export(
            config.campaign_data_path
        )
    )

    frames = []

    # -------------------------------------------------
    # Activities
    # -------------------------------------------------

    activities = export_data.get(
        "activities",
        pd.DataFrame(),
    )

    if not activities.empty:

        frames.append(
            _normalize_activities(
                activities
            )
        )

    # -------------------------------------------------
    # Navigation
    # -------------------------------------------------

    navigation = export_data.get(
        "navigation",
        pd.DataFrame(),
    )

    if not navigation.empty:

        frames.append(
            _normalize_navigation(
                navigation
            )
        )

    # -------------------------------------------------
    # Notifications
    # -------------------------------------------------

    notifications = export_data.get(
        "notification_events",
        pd.DataFrame(),
    )

    if not notifications.empty:

        frames.append(
            _normalize_notifications(
                notifications
            )
        )

    # -------------------------------------------------
    # Sensor events
    # -------------------------------------------------

    sensors = export_data.get(
        "sensor_events",
        pd.DataFrame(),
    )

    if not sensors.empty:

        frames.append(
            _normalize_sensors(
                sensors
            )
        )

    if not frames:

        return pd.DataFrame(
            columns=NORMALIZED_EVENT_COLUMNS
        )

    raw_events = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    normalized = _deduplicate_events(
        raw_events
    )

    # -------------------------------------------------
    # Stable chronological ordering
    # -------------------------------------------------

    normalized = normalized.sort_values(
        by=[
            "participant_id",
            "occurred_at",
            "event_id",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    return normalized


def run_event_normalization(
    config: TrajectoryAuditConfig | None = None,
) -> pd.DataFrame:
    """
    Run normalization and save normalized_events.csv.
    """

    config = (
        config
        or TrajectoryAuditConfig()
    )

    events = build_normalized_events(
        config
    )

    os.makedirs(
        config.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        config.output_dir,
        "normalized_events.csv",
    )

    events.to_csv(
        output_path,
        index=False,
    )

    return events


if __name__ == "__main__":

    events = run_event_normalization()

    print(
        "Normalized events written successfully."
    )

    print(
        "Total normalized events:",
        len(events),
    )

    if not events.empty:

        print()

        print("Events by kind:")

        counts = (
            events["event_kind"]
            .value_counts()
        )

        for event_kind, count in counts.items():

            print(
                f"  {event_kind}: {count}"
            )

        print()

        duplicate_source_rows = int(
            (
                events["source_row_count"] > 1
            ).sum()
        )

        print(
            "Events represented in multiple waves:",
            duplicate_source_rows,
        )

        conflicts = int(
            events["dedup_conflict"]
            .astype("boolean")
            .fillna(False)
            .sum()
        )

        print(
            "Deduplication conflicts:",
            conflicts,
        )