from __future__ import annotations

from typing import Dict, Optional, Any
import pandas as pd

from src.analysis.loaders import extract_points, extract_detailed_rewards


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