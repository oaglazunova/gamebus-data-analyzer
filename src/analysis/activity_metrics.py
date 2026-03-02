from __future__ import annotations

from typing import Dict, Optional, Any
import pandas as pd

from src.analysis.loaders import extract_points, extract_detailed_rewards
from src.analysis.common import WEEKDAY_ORDER, HOUR_BUCKET_LABELS, HOUR_BUCKET_BINS


GAMEBUS_ACTIVE_TYPES = [
    "DRINKING_DIARY",
    "GENERAL_ACTIVITY",
    "NUTRITION_DIARY",
    "PHYSICAL_ACTIVITY",
    "WALK",
    "BIKE",
]

GAMEBUS_METRIC_TYPES = GAMEBUS_ACTIVE_TYPES + ["DAY_AGGREGATE"]

GAMEBUS_USAGE_TIME_TYPES = GAMEBUS_ACTIVE_TYPES  # DAY_AGGREGATE excluded

NUTRIDA_TYPES = [
    "PLAN_MEAL",
    "NUTRITION_DIARY_VID",
]

COMBINED_ACTIVE_TYPES = GAMEBUS_ACTIVE_TYPES + NUTRIDA_TYPES
COMBINED_METRIC_TYPES = GAMEBUS_METRIC_TYPES + NUTRIDA_TYPES
COMBINED_USAGE_TIME_TYPES = GAMEBUS_USAGE_TIME_TYPES + NUTRIDA_TYPES


APP_SCOPES = {
    "gamebus": {
        "label": "GameBus",
        "active_types": GAMEBUS_ACTIVE_TYPES,
        "metric_types": GAMEBUS_METRIC_TYPES,
        "usage_time_types": GAMEBUS_USAGE_TIME_TYPES,
    },
    "nutrida": {
        "label": "Nutrida",
        "active_types": NUTRIDA_TYPES,
        "metric_types": NUTRIDA_TYPES,
        "usage_time_types": NUTRIDA_TYPES,
    },
    "combined": {
        "label": "Combined",
        "active_types": COMBINED_ACTIVE_TYPES,
        "metric_types": COMBINED_METRIC_TYPES,
        "usage_time_types": COMBINED_USAGE_TIME_TYPES,
    },
}


def build_activities_frame(csv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Validate and return a copy of the activities sheet.
    Raises ValueError if the required input is missing or invalid.
    """
    if "activities" not in csv_data:
        raise ValueError("Activities data not found in the provided data (missing key 'activities')")

    activities = csv_data["activities"].copy()
    if activities.empty:
        raise ValueError("Activities DataFrame is empty")

    required_columns = ["createdAt", "pid", "type", "rewardedParticipations"]
    missing_columns = [c for c in required_columns if c not in activities.columns]
    if missing_columns:
        raise ValueError(f"Activities DataFrame missing required columns: {missing_columns}")

    return activities


def normalize_activity_columns(activities: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize timestamps and derive commonly used activity columns.
    """
    activities = activities.copy()

    activities["createdAt"] = pd.to_datetime(activities["createdAt"], errors="coerce", utc=True)
    activities["date"] = activities["createdAt"].dt.floor("D")
    activities["hour"] = activities["createdAt"].dt.hour

    activities["points"] = activities["rewardedParticipations"].apply(extract_points)
    activities["detailed_rewards"] = activities["rewardedParticipations"].apply(extract_detailed_rewards)
    activities["num_rewards"] = activities["detailed_rewards"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    return activities


def get_campaign_waves(csv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return normalized campaign waves from desc_waves, if available.
    """
    waves = csv_data.get("desc_waves")
    if not isinstance(waves, pd.DataFrame) or waves.empty:
        return pd.DataFrame()

    required = {"start", "end"}
    if not required.issubset(waves.columns):
        return pd.DataFrame()

    out = waves.copy()
    out["start"] = pd.to_datetime(out["start"], errors="coerce", utc=True)
    out["end"] = pd.to_datetime(out["end"], errors="coerce", utc=True)
    out = out.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)
    return out


def _get_campaign_bounds(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Return (campaign_start, campaign_end), preferring desc_waves when available.
    """
    waves = get_campaign_waves(csv_data)
    if not waves.empty:
        return waves["start"].min(), waves["end"].max()

    return activities["createdAt"].min(), activities["createdAt"].max()


def assign_campaign_wave(activities: pd.DataFrame, csv_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Assign each activity to a campaign wave.
    Uses desc_waves when available; otherwise falls back to 7-day buckets
    from the earliest activity date.
    """
    waves = get_campaign_waves(csv_data)

    if waves.empty:
        start = activities["date"].min()
        if pd.isna(start):
            return pd.Series([pd.NA] * len(activities), index=activities.index)
        return ((activities["date"] - start).dt.days // 7) + 1

    result = pd.Series([pd.NA] * len(activities), index=activities.index, dtype="object")

    for i, row in waves.iterrows():
        mask = (
            (activities["createdAt"] >= row["start"])
            & (activities["createdAt"] <= row["end"])
        )
        result.loc[mask] = i + 1

    return result


def _extract_campaign_identity(csv_data: Dict[str, pd.DataFrame]) -> tuple[str, str]:
    """
    Extract campaign name and abbreviation from desc_* sheets if available.
    """
    campaign_name = "GameBus Campaign"
    campaign_abbr = ""

    for key, df in csv_data.items():
        if (
            isinstance(key, str)
            and key.startswith("desc_")
            and isinstance(df, pd.DataFrame)
            and not df.empty
            and "name" in df.columns
        ):
            try:
                names = df["name"].dropna()
                if names.empty:
                    continue

                campaign_name = names.iloc[0]

                if "abbreviation" in df.columns:
                    row_idx = names.index[0]
                    if row_idx in df.index and not pd.isna(df.loc[row_idx, "abbreviation"]):
                        campaign_abbr = df.loc[row_idx, "abbreviation"]

                break
            except Exception:
                continue

    return campaign_name, campaign_abbr


def compute_campaign_metrics(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
) -> dict:
    """
    Compute campaign-level metrics that do not depend on plotting.
    """
    unique_users_count = int(activities["pid"].nunique())

    campaign_start, campaign_end = _get_campaign_bounds(activities, csv_data)

    campaign_length_days = 0
    if pd.notna(campaign_start) and pd.notna(campaign_end):
        campaign_length_days = max(0, int((campaign_end - campaign_start).days))

    campaign_name, campaign_abbr = _extract_campaign_identity(csv_data)

    return {
        "unique_users": unique_users_count,
        "length_days": campaign_length_days,
        "start_date": campaign_start.date() if pd.notna(campaign_start) else None,
        "end_date": campaign_end.date() if pd.notna(campaign_end) else None,
        "name": campaign_name,
        "abbreviation": campaign_abbr,
    }


def compute_dropout_metrics(activities: pd.DataFrame) -> dict:
    """
    Compute dropout metrics and return both the metrics and the per-user table.
    """
    user_dropout = (
        activities.groupby("pid")
        .agg(first_activity=("createdAt", "min"), last_activity=("createdAt", "max"))
        .dropna()
    )

    if user_dropout.empty:
        return {
            "metrics": None,
            "user_dropout": user_dropout,
        }

    user_dropout["dropout_days"] = (
        user_dropout["last_activity"] - user_dropout["first_activity"]
    ).dt.days
    user_dropout.loc[user_dropout["dropout_days"] < 0, "dropout_days"] = 0

    dropout_stats = user_dropout["dropout_days"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    ).round(2)

    metrics = {
        "avg_dropout_days": float(dropout_stats.get("mean", 0.0)),
        "median_dropout_days": float(dropout_stats.get("50%", 0.0)),
        "min_dropout_days": float(dropout_stats.get("min", 0.0)),
        "max_dropout_days": float(dropout_stats.get("max", 0.0)),
    }

    return {
        "metrics": metrics,
        "user_dropout": user_dropout,
    }


def compute_joining_metrics(
    activities: pd.DataFrame,
    csv_data: Dict[str, pd.DataFrame],
) -> dict:
    """
    Compute joining metrics and return both the metrics and the per-user table.
    """
    campaign_start, _ = _get_campaign_bounds(activities, csv_data)

    user_dropout = (
        activities.groupby("pid")
        .agg(first_activity=("createdAt", "min"), last_activity=("createdAt", "max"))
        .dropna()
    )

    if user_dropout.empty or pd.isna(campaign_start):
        return {
            "metrics": None,
            "user_dropout": user_dropout,
        }

    user_dropout["dropout_days"] = (
        user_dropout["last_activity"] - user_dropout["first_activity"]
    ).dt.days
    user_dropout.loc[user_dropout["dropout_days"] < 0, "dropout_days"] = 0

    user_dropout["joining_days"] = (
        user_dropout["first_activity"] - campaign_start
    ).dt.days

    joining_stats = user_dropout["joining_days"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]
    ).round(2)

    metrics = {
        "avg_joining_days": float(joining_stats.get("mean", 0.0)),
        "median_joining_days": float(joining_stats.get("50%", 0.0)),
        "min_joining_days": float(joining_stats.get("min", 0.0)),
        "max_joining_days": float(joining_stats.get("max", 0.0)),
    }

    return {
        "metrics": metrics,
        "user_dropout": user_dropout,
    }


def _series_to_id_set(series: pd.Series) -> set:
    """
    Convert a Series of identifiers into a de-duplicated set,
    dropping NaN and blank-string values.
    """
    values = series.dropna()
    if values.empty:
        return set()

    if values.dtype == object:
        as_text = values.astype(str).str.strip()
        values = values[as_text != ""]

    return set(values.tolist())


def get_enrolled_user_ids(
    csv_data: Dict[str, pd.DataFrame],
    activities: pd.DataFrame,
) -> set:
    """
    Determine the enrolled user base.

    Preferred behavior:
    - scan all non-desc, non-activities sheets for user-id-like columns
    - if nothing usable is found, fall back to all users seen in activities
    """
    candidate_columns = ("pid", "playerId", "participantId", "userId", "id")
    enrolled_user_ids: set = set()

    for sheet_name, df in csv_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        if sheet_name == "activities":
            continue

        if isinstance(sheet_name, str) and sheet_name.startswith("desc_"):
            continue

        for col in candidate_columns:
            if col in df.columns:
                enrolled_user_ids.update(_series_to_id_set(df[col]))

    if enrolled_user_ids:
        return enrolled_user_ids

    return _series_to_id_set(activities["pid"])


def filter_activities_by_types(
    activities: pd.DataFrame,
    allowed_types: list[str],
) -> pd.DataFrame:
    """
    Return a copy of activities filtered to the provided descriptor types.
    """
    if activities.empty or "type" not in activities.columns:
        return activities.iloc[0:0].copy()

    return activities[activities["type"].isin(allowed_types)].copy()


def compute_reward_based_active_user_ids(activities: pd.DataFrame) -> set:
    """
    Legacy comparison metric:
    active user = any user who earned points or has one or more rewards.

    This preserves the current implementation logic so you can compare
    descriptor-based active users vs reward-based active users.
    """
    frame = activities.copy()

    if "points" not in frame.columns:
        frame["points"] = frame["rewardedParticipations"].apply(extract_points)

    if "num_rewards" not in frame.columns:
        if "detailed_rewards" in frame.columns:
            frame["num_rewards"] = frame["detailed_rewards"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:
            frame["detailed_rewards"] = frame["rewardedParticipations"].apply(extract_detailed_rewards)
            frame["num_rewards"] = frame["detailed_rewards"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

    rewarded = frame[(frame["points"] > 0) | (frame["num_rewards"] > 0)]

    return _series_to_id_set(rewarded["pid"])


def compute_active_passive_by_types(
    activities: pd.DataFrame,
    enrolled_user_ids: set,
    active_types: list[str],
) -> dict:
    """
    Descriptor-based active/passive classification.

    active = enrolled user with at least one activity record whose type is in active_types
    passive = enrolled user with zero activity records whose type is in active_types
    """
    scoped_activities = filter_activities_by_types(activities, active_types)
    active_user_ids = _series_to_id_set(scoped_activities["pid"])

    if enrolled_user_ids:
        passive_user_ids = set(enrolled_user_ids) - set(active_user_ids)
    else:
        passive_user_ids = set()

    return {
        "active_user_ids": active_user_ids,
        "passive_user_ids": passive_user_ids,
        "active_users_count": len(active_user_ids),
        "passive_users_count": len(passive_user_ids),
        "active_records_count": int(len(scoped_activities)),
    }


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _safe_median(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.median())


def _safe_min(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.min())


def _safe_max(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.max())


def compute_scope_usage_metrics(
    activities: pd.DataFrame,
    enrolled_user_ids: set,
    metric_types: list[str],
    usage_time_types: list[str],
) -> dict:
    """
    Compute per-scope usage metrics.

    Rules:
    - Average active days / participant, active players per day, and average time to first inactivity
      use metric_types (GameBus includes DAY_AGGREGATE here).
    - Usage time / day of week, usage time / hour of day buckets, and peak activity hour
      use usage_time_types (GameBus excludes DAY_AGGREGATE here).
    """
    metric_activities = filter_activities_by_types(activities, metric_types)
    usage_activities = filter_activities_by_types(activities, usage_time_types)

    # ------------------------------------------------------------------
    # 1) Average active days / participant
    # Includes passive enrolled users as 0 if enrolled_user_ids is available
    # ------------------------------------------------------------------
    user_active_days = (
        metric_activities.groupby("pid")["date"]
        .nunique()
        if not metric_activities.empty
        else pd.Series(dtype="int64")
    )

    if enrolled_user_ids:
        active_days_per_participant = user_active_days.reindex(
            pd.Index(list(enrolled_user_ids)),
            fill_value=0,
        )
    else:
        active_days_per_participant = user_active_days

    # ------------------------------------------------------------------
    # 2) Active players per day
    # ------------------------------------------------------------------
    active_players_per_day = (
        metric_activities.groupby("date")["pid"]
        .nunique()
        .sort_index()
        if not metric_activities.empty
        else pd.Series(dtype="int64")
    )

    # ------------------------------------------------------------------
    # 3) Average time to first inactivity (keep current proxy)
    # Current proxy = last_activity - first_activity
    # Only for users with at least one scoped activity
    # ------------------------------------------------------------------
    user_inactivity = (
        metric_activities.groupby("pid")
        .agg(first_activity=("createdAt", "min"), last_activity=("createdAt", "max"))
        .dropna()
    )

    if not user_inactivity.empty:
        user_inactivity["time_to_first_inactivity_days"] = (
            user_inactivity["last_activity"] - user_inactivity["first_activity"]
        ).dt.days
        user_inactivity.loc[
            user_inactivity["time_to_first_inactivity_days"] < 0,
            "time_to_first_inactivity_days",
        ] = 0

        time_to_first_inactivity = user_inactivity["time_to_first_inactivity_days"]
    else:
        time_to_first_inactivity = pd.Series(dtype="float64")

    # ------------------------------------------------------------------
    # 4) Usage time / day of week
    # Uses usage_time_types (e.g., excludes DAY_AGGREGATE for GameBus)
    # ------------------------------------------------------------------
    if not usage_activities.empty:
        day_names = usage_activities["createdAt"].dt.day_name()
        usage_by_day_of_week = day_names.value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
    else:
        usage_by_day_of_week = pd.Series(0, index=WEEKDAY_ORDER, dtype="int64")

    # ------------------------------------------------------------------
    # 5) Usage time / hour of day
    # ------------------------------------------------------------------
    if not usage_activities.empty:
        usage_by_hour = (
            usage_activities["hour"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
    else:
        usage_by_hour = pd.Series(dtype="int64")

    # ------------------------------------------------------------------
    # 6) Usage time / hour buckets
    # ------------------------------------------------------------------
    if not usage_by_hour.empty:
        bucket_series = pd.cut(
            usage_activities["hour"].dropna().astype(int),
            bins=HOUR_BUCKET_BINS,
            labels=HOUR_BUCKET_LABELS,
        )
        usage_by_hour_buckets = bucket_series.value_counts().reindex(HOUR_BUCKET_LABELS, fill_value=0)
    else:
        usage_by_hour_buckets = pd.Series(0, index=HOUR_BUCKET_LABELS, dtype="int64")

    # ------------------------------------------------------------------
    # 7) Peak activity hour
    # ------------------------------------------------------------------
    peak_activity_hour = None
    if not usage_by_hour.empty:
        peak_activity_hour = int(usage_by_hour.idxmax())

    return {
        "metric_activities_count": int(len(metric_activities)),
        "usage_time_activities_count": int(len(usage_activities)),

        "active_days_per_participant": active_days_per_participant,
        "avg_active_days_per_participant": _safe_mean(active_days_per_participant),
        "median_active_days_per_participant": _safe_median(active_days_per_participant),
        "min_active_days_per_participant": _safe_min(active_days_per_participant),
        "max_active_days_per_participant": _safe_max(active_days_per_participant),

        "active_players_per_day": active_players_per_day,
        "avg_active_players_per_day": _safe_mean(active_players_per_day),
        "median_active_players_per_day": _safe_median(active_players_per_day),
        "min_active_players_per_day": _safe_min(active_players_per_day),
        "max_active_players_per_day": _safe_max(active_players_per_day),

        "time_to_first_inactivity_by_user": time_to_first_inactivity,
        "avg_time_to_first_inactivity_days": _safe_mean(time_to_first_inactivity),
        "median_time_to_first_inactivity_days": _safe_median(time_to_first_inactivity),
        "min_time_to_first_inactivity_days": _safe_min(time_to_first_inactivity),
        "max_time_to_first_inactivity_days": _safe_max(time_to_first_inactivity),

        "usage_by_day_of_week": usage_by_day_of_week,
        "usage_by_hour": usage_by_hour,
        "usage_by_hour_buckets": usage_by_hour_buckets,
        "peak_activity_hour": peak_activity_hour,
    }